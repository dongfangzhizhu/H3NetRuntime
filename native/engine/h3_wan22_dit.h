#ifndef H3_WAN22_DIT_H
#define H3_WAN22_DIT_H

#include "h3_quanto.h"

#include <stddef.h>
#include <stdint.h>

/* Wan 2.2 TI2V-5B DiT block forward, CPU reference path (Phase 0).
 *
 * Architecture (Wan2GP models/wan/modules/model.py, Wan2.2 TI2V-5B config):
 * 30 cross-attention blocks, dim 3072, 24 heads x 128, FFN 14336 with
 * GELU-tanh (not SwiGLU); per block: self-attn -> cross-attn(text) -> FFN.
 * adaLN: the global timestep chain (sinusoid -> time_embedding -> SiLU ->
 * time_projection) produces e0 [6, dim] externally; every block then adds its
 * own modulation parameter [1, 6, dim] and chunks into
 * shift_msa / scale_msa / gate_msa / shift_mlp / scale_mlp / gate_mlp.
 * norm1/norm2 and the Head's norm are affine-free LayerNorms (eps 1e-6);
 * norm3 (before cross-attn) is a LayerNorm WITH affine, stored as F32.
 * QK RMSNorm spans the whole 3072-wide row (not per head), eps 1e-6, and is
 * applied before the 3-axis RoPE (44/42/42, theta 10000, interleaved pairs).
 * The Head applies LN -> modulation chunk 2 -> Linear(3072 -> 192).
 *
 * Phase 0 keeps every weight in host memory (quanto I8 dequantized to BF16 at
 * load; the time/head/norm3 tensors are stored F32 and kept as F32) and
 * computes in F32 with OpenMP-parallel row loops, mirroring the runtime plan
 * (I8 resident, dequantize, BF16 GEMM widened per element) and giving the
 * numerics reference for the later CUDA port.
 */

typedef struct {
    uint32_t dim;             /* 3072 */
    uint32_t heads;           /* 24 */
    uint32_t head_dim;        /* 128 */
    uint32_t ffn_dim;         /* 14336 */
    uint32_t freq_dim;        /* 256 (sinusoidal embedding + time MLP input) */
    uint32_t text_len;        /* 512 context tokens (fixtures may use fewer) */
    float norm_eps;           /* 1e-6, all LayerNorms / RMSNorms */
    uint32_t rope_axes[3];    /* 44, 42, 42 (T/H/W, sums to head_dim) */
    float rope_theta;         /* 10000 */
} h3_wan22_config;

h3_wan22_config h3_wan22_default_config(void);

/* GPU-resident copies of the weights (h3_gpu.h's tensor type; only
 * touched when the WAN22_GPU=1 upload path runs). */
struct h3_gpu_tensor;
typedef struct h3_gpu_tensor h3_gpu_tensor;

/* A Linear: BF16 weight [out, in] widened per element, or an F32 weight for
 * the tensors the checkpoint stores unquantized F32 (time_embedding,
 * time_projection, head.head).  Bias is always F32. */
typedef struct {
    uint32_t out;
    uint32_t in;
    uint16_t *weight;    /* BF16 [out, in]; NULL when weight_f32 is set */
    float *weight_f32;   /* F32 [out, in]; NULL when weight is set */
    float *bias;
    struct h3_gpu_tensor *gpu_weight; /* WAN22_GPU upload; else NULL */
    struct h3_gpu_tensor *gpu_bias;
} h3_wan22_linear;

/* Quanto qint8 pair -> BF16 weight. */
int h3_wan22_linear_load_quanto(const h3_weight_store *store, const char *name,
                                uint32_t out, uint32_t in, int has_bias,
                                h3_wan22_linear *linear,
                                char *error, size_t error_size);
/* Plain BF16 tensor -> BF16 weight. */
int h3_wan22_linear_load_bf16(const h3_weight_store *store, const char *name,
                              uint32_t out, uint32_t in, int has_bias,
                              h3_wan22_linear *linear,
                              char *error, size_t error_size);
/* F32 tensor -> F32 weight (read verbatim, no dtype hop). */
int h3_wan22_linear_load_f32(const h3_weight_store *store, const char *name,
                             uint32_t out, uint32_t in, int has_bias,
                             h3_wan22_linear *linear,
                             char *error, size_t error_size);
void h3_wan22_linear_free(h3_wan22_linear *linear);
/* out[rows, out] = input[rows, in] @ weight^T + bias. */
int h3_wan22_linear_forward(const h3_wan22_linear *linear,
                            const float *input, uint32_t rows, float *output);

/* RMSNorm over the full row (weight BF16 [width], eps 1e-6); used for the
 * whole-row QK norms. */
typedef struct {
    uint32_t width;
    float eps;
    uint16_t *weight;
    struct h3_gpu_tensor *gpu_weight;
} h3_wan22_rms;

int h3_wan22_rms_load(const h3_weight_store *store, const char *name,
                      uint32_t width, float eps, h3_wan22_rms *rms,
                      char *error, size_t error_size);
void h3_wan22_rms_free(h3_wan22_rms *rms);
/* out = in * rsqrt(mean(in^2) + eps) * weight, row-major [rows, width]. */
int h3_wan22_rms_forward(const h3_wan22_rms *rms, const float *input,
                         uint32_t rows, float *output);

/* LayerNorm, affine-free (norm1/norm2/head.norm) or affine (norm3, whose
 * weight/bias are stored F32). */
typedef struct {
    uint32_t width;
    float eps;
    int affine;
    float *weight;
    float *bias;
    struct h3_gpu_tensor *gpu_weight; /* WAN22_GPU upload (BF16) */
    struct h3_gpu_tensor *gpu_bias;
} h3_wan22_ln;

int h3_wan22_ln_load(const h3_weight_store *store, const char *name,
                     uint32_t width, float eps, int affine, h3_wan22_ln *ln,
                     char *error, size_t error_size);
void h3_wan22_ln_free(h3_wan22_ln *ln);
int h3_wan22_ln_forward(const h3_wan22_ln *ln, const float *input,
                        uint32_t rows, float *output);

/* One transformer block ("blocks.<i>" in the checkpoint). */
typedef struct h3_wan22_block h3_wan22_block;

h3_wan22_block *h3_wan22_block_load(const h3_weight_store *store,
                                    uint32_t index,
                                    const h3_wan22_config *config,
                                    char *error, size_t error_size);
void h3_wan22_block_free(h3_wan22_block *block);

/* Forward over x [seq, dim] in place (row-major F32).
 *   e0       [6 * dim] global modulation from the timestep chain
 *   context  [ctx_seq, dim] embedded text (post text_embedding)
 *   rope_cos / rope_sin  [seq, head_dim / 2] pair tables from
 *                        h3_wan22_rope_tables (self-attention only)
 */
int h3_wan22_block_forward(const h3_wan22_block *block, float *x,
                           uint32_t seq, const float *e0,
                           const float *context, uint32_t ctx_seq,
                           const float *rope_cos, const float *rope_sin,
                           char *error, size_t error_size);

/* RoPE pair tables for a patchified latent grid: seq = F * H * W tokens in
 * f-major order (the meshgrid "ij" flatten of the patch grid), pair tables
 * [seq, head_dim / 2] with axis layout 44/42/42 (T pairs first). */
int h3_wan22_rope_tables(const h3_wan22_config *config, uint32_t frames,
                         uint32_t grid_h, uint32_t grid_w,
                         float *cos, float *sin);

/* TimestepEmbedder sinusoidal embedding: [cos(args) | sin(args)], dim 256
 * (freq_dim), max_period 10000, float32 math. */
void h3_wan22_timestep_embedding(float t, uint32_t dim, float *out);

/* time_embedding: Linear(freq_dim -> dim) -> SiLU -> Linear(dim -> dim),
 * stored F32. */
typedef struct {
    h3_wan22_linear fc0;
    h3_wan22_linear fc1;
} h3_wan22_time_embedder;

int h3_wan22_time_embedder_load(const h3_weight_store *store,
                                const h3_wan22_config *config,
                                h3_wan22_time_embedder *embedder,
                                char *error, size_t error_size);
void h3_wan22_time_embedder_free(h3_wan22_time_embedder *embedder);
/* input: sinusoidal embedding [rows, freq_dim]; output [rows, dim]. */
int h3_wan22_time_embedder_forward(const h3_wan22_time_embedder *embedder,
                                   const float *input, uint32_t rows,
                                   float *output);

/* FinalLayer: affine-free LayerNorm -> modulation chunk 2 (shift, scale)
 * -> x * (1 + scale) + shift -> Linear(dim -> patch_volume * out_dim) = 192.
 * modulation and the output Linear are stored F32. */
typedef struct {
    uint32_t dim;
    uint32_t out;
    float norm_eps;
    float *shift_scale;      /* F32 [2 * dim]: shift, scale (modulation param) */
    h3_wan22_linear out_proj;
} h3_wan22_head;

int h3_wan22_head_load(const h3_weight_store *store,
                       const h3_wan22_config *config, uint32_t out,
                       h3_wan22_head *head, char *error, size_t error_size);
void h3_wan22_head_free(h3_wan22_head *head);
/* x [rows, dim]; e [dim] (the time_embedding output, NOT the 6*dim e0);
 * output [rows, out]. */
int h3_wan22_head_forward(const h3_wan22_head *head, const float *x,
                          uint32_t rows, const float *e, float *output);


/* --- optional GPU forward path (WAN22_GPU=1) ---------------------------
 * The 30 blocks run on the GPU with BF16 weights resident on the device
 * (~9.8 GB); the patch embedding, timestep chain and Head stay on the host
 * in F32 exactly like the CPU path.  Requires the CUDA build.
 *                                                                        */
typedef struct h3_wan22_gpu h3_wan22_gpu;

h3_wan22_gpu *h3_wan22_gpu_create(const h3_wan22_config *config,
                                  char *error, size_t error_size);
void h3_wan22_gpu_free(h3_wan22_gpu *gpu);
/* Allocate the [seq, ...] / [ctx_seq, ...] scratch tensors (idempotent). */
int h3_wan22_gpu_reserve(h3_wan22_gpu *gpu, uint32_t seq, uint32_t ctx_seq,
                         char *error, size_t error_size);
/* Upload one block's weights.  self_attn q/k rows are permuted per head
 * (even dimensions first) so h3_gpu_rope_text_bf16's half-split pairing
 * reproduces Wan's interleaved (2i, 2i+1) RoPE.
 */
h3_gpu_tensor *h3_wan22_block_gpu_norm_q(h3_wan22_block *block); /* probe access */
h3_gpu_tensor *h3_wan22_block_gpu_norm_k(h3_wan22_block *block); /* probe access */
int h3_wan22_block_upload_gpu(h3_wan22_block *block, h3_wan22_gpu *gpu,
                              char *error, size_t error_size);
/* One block on the GPU: x is [seq, dim] BF16, modified in place;
 * modulation is the per-block BF16 [6, dim] table with rows
 * shift_msa / scale_msa / gate_msa / shift_mlp / scale_mlp / gate_mlp
 * (scales raw - the kernel adds 1; gates raw - Wan has no tanh).
 */
/* Per-step e0 upload into the shared [6, dim] carrier; per-block adaLN
 * tables are uploaded once and summed with e0 in F32 on the device. */
int h3_wan22_gpu_set_e0_gpu(h3_wan22_gpu *gpu, const float *e0,
                            char *error, size_t error_size);
int h3_wan22_block_forward_gpu(
    h3_wan22_block *block, h3_wan22_gpu *gpu, h3_gpu_tensor *x,
    uint32_t seq, const h3_gpu_tensor *context, uint32_t ctx_seq,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    char *error, size_t error_size);
h3_gpu_tensor *h3_wan22_gpu_modulation(h3_wan22_gpu *gpu);
h3_gpu_tensor *h3_wan22_gpu_rope_cos(h3_wan22_gpu *gpu);
h3_gpu_tensor *h3_wan22_gpu_rope_sin(h3_wan22_gpu *gpu);
h3_gpu_tensor *h3_wan22_gpu_x(h3_wan22_gpu *gpu); /* F32 residual carrier (GPU path) */
h3_gpu_tensor *h3_wan22_gpu_row_map(h3_wan22_gpu *gpu);
h3_gpu_tensor *h3_wan22_gpu_normed(h3_wan22_gpu *gpu);
h3_gpu_tensor *h3_wan22_gpu_q(h3_wan22_gpu *gpu); /* post-RMS/RoPE scratch */
h3_gpu_tensor *h3_wan22_gpu_k(h3_wan22_gpu *gpu);
h3_gpu_tensor *h3_wan22_gpu_v(h3_wan22_gpu *gpu);
h3_gpu_tensor *h3_wan22_gpu_attn(h3_wan22_gpu *gpu);
h3_gpu *h3_wan22_gpu_handle(h3_wan22_gpu *gpu); /* for h3_gpu_error */
h3_gpu_tensor *h3_wan22_gpu_upload_f32_bf16(h3_wan22_gpu *gpu,
                                            const float *values,
                                            size_t count);
int h3_wan22_gpu_read_f32(h3_wan22_gpu *gpu, h3_gpu_tensor *tensor,
                          float *out, size_t count);

/* --- device-state residency: swap the DiT out for the VAE decode ---------
 * The 30 blocks' BF16 weights (~9.8 GiB) plus the activation scratch cannot
 * coexist with the streaming VAE decode on a 24 GB card.  Release drops both
 * while keeping the host structures, the host BF16 weights (the CPU path
 * needs them anyway) and the CUDA handle; a released context returns an
 * error from h3_wan22_block_forward_gpu instead of touching freed tensors.
 * Resume re-uploads the weights (one PCIe pass, no checkpoint re-read) and
 * re-reserves scratch for a given geometry; a release/resume round trip is
 * bit-exact because the upload is the same BF16 source.
 *
 * blocks is the pipeline's [block_count] array (h3_wan22_block_upload_gpu
 * order).  Both calls are idempotent; resume with seq == 0 only re-uploads.
 */
int h3_wan22_gpu_state_resident(const h3_wan22_gpu *gpu);
void h3_wan22_block_release_gpu(h3_wan22_block *block);
int h3_wan22_gpu_state_release(h3_wan22_block **blocks, uint32_t block_count,
                               h3_wan22_gpu *gpu, char *error,
                               size_t error_size);
int h3_wan22_gpu_state_resume(h3_wan22_block **blocks, uint32_t block_count,
                              h3_wan22_gpu *gpu, uint32_t seq, uint32_t ctx_seq,
                              char *error, size_t error_size);
#endif
