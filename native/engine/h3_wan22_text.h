#ifndef H3_WAN22_TEXT_H
#define H3_WAN22_TEXT_H

#include "h3_wan22_dit.h"

#include <stddef.h>
#include <stdint.h>

/* Wan 2.2 T2V 5B Phase 1: umt5-xxl text encoder (CPU reference path).
 *
 * Architecture (Wan2GP models/wan/modules/t5.py, umt5_xxl config):
 * 24 encoder layers, d_model = dim_attn = 4096, FFN 10240, 64 heads x 64,
 * vocab 256384, num_buckets 32, shared_pos = False (every layer owns its
 * relative-attention-bias table), text_len 512.  T5LayerNorm is an RMSNorm
 * (eps 1e-6, BF16 weight) and reuses h3_wan22_rms; the Linear stack reuses
 * h3_wan22_linear with the quanto qint8 loader from Phase 0.
 *
 * Per block (T5SelfAttention):
 *   x += attn(norm1(x), pos_bias)     bidirectional, NO 1/sqrt(d) scaling
 *   x += fc2( gelu_tanh(gate(norm2(x))) * fc1(norm2(x)) )
 * The attention bias is pos_bias[head][bucket(q, k)] plus -inf on padded
 * keys; softmax runs in F32.  The relative bucket is the bidirectional T5
 * layout: bucket = (rel > 0) * 16 + log-bucketed |rel| with max_exact 8 and
 * max_dist 128, rel = key_pos - query_pos (t5.py T5RelativeEmbedding).
 * After the last block a final RMSNorm produces the context [seq, 4096].
 *
 * text_embedding (DiT side, plain BF16 in the DiT checkpoint):
 *   Linear(4096 -> 3072) -> GELU-tanh -> Linear(3072 -> 3072)
 * maps the encoder context onto the DiT cross-attention width.
 *
 * Weights stay in host memory like Phase 0: quanto I8 dequantized to BF16 at
 * load, all math in F32 (BF16 widened per element), OpenMP over rows/heads.
 */

typedef struct {
    uint32_t vocab;        /* 256384 */
    uint32_t dim;          /* 4096 */
    uint32_t dim_attn;     /* 4096 */
    uint32_t ffn_dim;      /* 10240 */
    uint32_t heads;        /* 64 */
    uint32_t head_dim;     /* 64 (= dim_attn / heads) */
    uint32_t num_layers;   /* 24 */
    uint32_t num_buckets;  /* 32 relative position buckets */
    uint32_t text_len;     /* 512 */
    float norm_eps;        /* 1e-6 */
    float max_dist;        /* 128, relative bucket log span */
} h3_wan22_text_config;

h3_wan22_text_config h3_wan22_text_default_config(void);

typedef struct h3_wan22_encoder h3_wan22_encoder;

/* Load token_embedding + 24 blocks + final norm from a quanto qint8
 * umt5-xxl encoder checkpoint (a single-file directory). */
h3_wan22_encoder *h3_wan22_encoder_load(const h3_weight_store *store,
                                        const h3_wan22_text_config *config,
                                        char *error, size_t error_size);
void h3_wan22_encoder_free(h3_wan22_encoder *encoder);

/* ids  int32  [seq]                      token ids (padded with 0)
 * mask uint8  [seq], may be NULL         1 = real token, 0 = padding
 * out  float  [seq, dim]                 final normed encoder context
 * probe_after / probe_out: when probe_after > 0 the hidden state right
 * after that many blocks is copied to probe_out [seq, dim] (numerics
 * tests); 0 disables the probe.
 */
int h3_wan22_encoder_forward(const h3_wan22_encoder *encoder,
                             const int32_t *ids, const uint8_t *mask,
                             uint32_t seq, float *out,
                             uint32_t probe_after, float *probe_out,
                             char *error, size_t error_size);

/* --- serve context cache ----------------------------------------------------
 * A resident server replays an encoder output from disk when the exact token
 * window has been encoded before: the forward is ~62 s per branch at the
 * 512-token serve window, the entry is 8 MiB.  The key covers the encoder
 * identity (tokenizer content + umt5 shard name/size/mtime), the window
 * length, and the token ids, so a different checkpoint or a truncated window
 * can never hit an entry it did not produce.  Payload bytes are stored
 * verbatim and read back verbatim, i.e. a hit is bit-identical to a miss.
 */
#define H3_WAN22_TEXT_CACHE_MAGIC "H3W2C001"

typedef struct {
    char dir[1024];            /* empty disables the cache */
} h3_wan22_text_cache;

/* mkdir -p the directory and arm the cache; an empty dir disables it.
 * Returns 1 when the cache state is usable, 0 when the directory cannot be
 * created (the cache stays disabled). */
int h3_wan22_text_cache_init(h3_wan22_text_cache *cache, const char *dir);

/* key = sha256("wan22-text-cache-1|identity|seq|dim|ids[0..seq)") as hex. */
void h3_wan22_text_cache_key(const char *identity, const int32_t *ids,
                             uint32_t seq, uint32_t dim, char hex[65]);

/* 1 = hit, out [seq, dim] holds the stored plane; 0 = miss or bad entry. */
int h3_wan22_text_cache_load(const h3_wan22_text_cache *cache,
                             const char *identity, const int32_t *ids,
                             uint32_t seq, uint32_t dim, float *out);

/* Publish values [seq, dim] under the same key (temp file + rename).
 * Returns 1 on success; a failure is not fatal for the request. */
int h3_wan22_text_cache_store(const h3_wan22_text_cache *cache,
                              const char *identity, const int32_t *ids,
                              uint32_t seq, uint32_t dim,
                              const float *values);

/* --- optional GPU forward path (WAN22_TEXT_GPU=1) ----------------------
 * The 24 encoder blocks run on the device with the host's quanto-dequantized
 * BF16 weights uploaded verbatim and F32 activations throughout (mixed
 * BF16-weight/F32 GEMMs with exact FP32 FMAs, plus a T5 attention kernel
 * with the per-layer relative position bias).  That is the same arithmetic
 * the CPU reference path performs -- BF16 weights widened per element, F32
 * accumulation -- so the GPU forward matches it to summation order only.
 * The token embedding lookup stays on the host (one [seq, dim] plane
 * uploaded per forward); the CPU path is untouched and remains the default
 * when WAN22_TEXT_GPU is unset/0 or CUDA is unavailable.
 *
 * Device-state residency: the weights (~8.6 GiB BF16 for the 24 blocks) are
 * uploaded on the first forward and dropped by h3_wan22_encoder_release_gpu
 * at the end of the text phase, so the DiT+VAE phases see the same device
 * budget as before.  The next forward re-uploads from the host copy (the CPU
 * path needs those bytes anyway), like the DiT's release/resume swap.
 * ----------------------------------------------------------------------- */
typedef struct h3_wan22_text_gpu h3_wan22_text_gpu;

/* Create (or return the existing) device context for this encoder.  Returns
 * NULL when CUDA is unavailable; the caller then stays on the CPU path. */
h3_wan22_text_gpu *h3_wan22_text_gpu_get(h3_wan22_encoder *encoder,
                                         char *error, size_t error_size);
void h3_wan22_text_gpu_drop(h3_wan22_text_gpu *tgpu);
/* 1 when the encoder holds a live device context (weights may or may not be
 * uploaded), 0 otherwise. */
int h3_wan22_text_gpu_active(const h3_wan22_encoder *encoder);
/* "gpu" when the last forward ran on the device, else "cpu". */
const char *h3_wan22_text_device(const h3_wan22_encoder *encoder);
/* Drop the device weights (and scratch).  Idempotent; a later forward
 * re-uploads.  Returns 1 on success. */
int h3_wan22_encoder_release_gpu(h3_wan22_encoder *encoder, char *error,
                                 size_t error_size);

/* text_embedding MLP (DiT checkpoint, plain BF16 with bias):
 * Linear(dim -> hidden) -> GELU-tanh -> Linear(hidden -> out). */
typedef struct {
    uint32_t out;
    uint32_t hidden;
    uint32_t in;
    h3_wan22_linear fc0;
    h3_wan22_linear fc2;
} h3_wan22_text_mlp;

int h3_wan22_text_mlp_load(const h3_weight_store *store,
                           uint32_t in_dim, uint32_t hidden_dim,
                           uint32_t out_dim, h3_wan22_text_mlp *mlp,
                           char *error, size_t error_size);
void h3_wan22_text_mlp_free(h3_wan22_text_mlp *mlp);
/* input [rows, in] -> output [rows, out]. */
int h3_wan22_text_mlp_forward(const h3_wan22_text_mlp *mlp,
                              const float *input, uint32_t rows,
                              float *output);

#endif
