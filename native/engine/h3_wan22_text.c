#include "h3_wan22_text.h"

#include "h3_gpu.h"
#include "h3_hash.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* Bias value for padded keys (t5.py fills with finfo(dtype).min; any value
 * that underflows softmax to exactly 0 in F32 matches). */
#define TEXT_MASK_BIAS (-3.389531389e38f)

static float text_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void text_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* WAN22_TEXT_GPU=1 arms the optional GPU forward path (see the GPU section
 * below); unset/0/empty keeps the CPU reference. */
static int wan22_text_gpu_enabled(void) {
    const char *value = getenv("WAN22_TEXT_GPU");
    return value && *value && strcmp(value, "0") != 0 && atoi(value) != 0;
}

h3_wan22_text_config h3_wan22_text_default_config(void) {
    h3_wan22_text_config config;
    config.vocab = 256384;
    config.dim = 4096;
    config.dim_attn = 4096;
    config.ffn_dim = 10240;
    config.heads = 64;
    config.head_dim = 64;
    config.num_layers = 24;
    config.num_buckets = 32;
    config.text_len = 512;
    config.norm_eps = 1e-6f;
    config.max_dist = 128.0f;
    return config;
}

/* --- block ------------------------------------------------------------------- */

typedef struct {
    h3_wan22_linear q, k, v, o;        /* T5Attention, no bias */
    h3_wan22_linear gate, fc1, fc2;    /* T5FeedForward, no bias */
    h3_wan22_rms norm1, norm2;         /* T5LayerNorm == RMSNorm */
    float *pos_bias;                   /* F32 [num_buckets, heads] dequantized */
} h3_wan22_text_block;

struct h3_wan22_encoder {
    h3_wan22_text_config config;
    uint16_t *token_embedding;         /* BF16 [vocab, dim] */
    h3_wan22_text_block *blocks;       /* [num_layers] */
    h3_wan22_rms final_norm;
    /* Optional GPU forward path (WAN22_TEXT_GPU=1); NULL stays on the CPU. */
    struct h3_wan22_text_gpu *gpu;
    int gpu_failed;                    /* a device error retired the path */
};

static void text_block_free(h3_wan22_text_block *block) {
    h3_wan22_linear_free(&block->q);
    h3_wan22_linear_free(&block->k);
    h3_wan22_linear_free(&block->v);
    h3_wan22_linear_free(&block->o);
    h3_wan22_linear_free(&block->gate);
    h3_wan22_linear_free(&block->fc1);
    h3_wan22_linear_free(&block->fc2);
    h3_wan22_rms_free(&block->norm1);
    h3_wan22_rms_free(&block->norm2);
    free(block->pos_bias);
    block->pos_bias = NULL;
}

/* Load the per-layer relative-attention-bias table: quanto qint8
 * [num_buckets, heads] -> F32. */
static int text_load_pos_bias(const h3_weight_store *store, const char *name,
                              uint32_t buckets, uint32_t heads, float **out,
                              char *error, size_t error_size) {
    uint16_t *staged = h3_quanto_load_bf16(store, name, buckets, heads,
                                           error, error_size);
    if (!staged) return 0;
    float *bias = malloc((size_t)buckets * heads * sizeof(float));
    if (!bias) {
        free(staged);
        text_fail(error, error_size, "cannot allocate %s", name);
        return 0;
    }
    for (uint32_t index = 0; index < buckets * heads; index++) {
        bias[index] = text_bf16_to_f32(staged[index]);
    }
    free(staged);
    *out = bias;
    return 1;
}

h3_wan22_encoder *h3_wan22_encoder_load(const h3_weight_store *store,
                                        const h3_wan22_text_config *config,
                                        char *error, size_t error_size) {
    h3_wan22_encoder *encoder = calloc(1, sizeof(h3_wan22_encoder));
    if (!encoder) {
        text_fail(error, error_size, "cannot allocate umt5 encoder");
        return NULL;
    }
    encoder->config = *config;

    /* token_embedding.weight BF16 [vocab, dim]. */
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, "token_embedding.weight",
                                                &header);
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(tensor) !=
            (uint64_t)config->vocab * config->dim) {
        text_fail(error, error_size,
                  "token_embedding.weight missing or wrong shape/dtype");
        h3_wan22_encoder_free(encoder);
        return NULL;
    }
    encoder->token_embedding =
        malloc((size_t)config->vocab * config->dim * 2);
    if (!encoder->token_embedding ||
        !h3_st_read_data(header, tensor, encoder->token_embedding,
                         (size_t)config->vocab * config->dim * 2,
                         error, error_size)) {
        h3_wan22_encoder_free(encoder);
        return NULL;
    }

    encoder->blocks = calloc(config->num_layers, sizeof(h3_wan22_text_block));
    if (!encoder->blocks) {
        h3_wan22_encoder_free(encoder);
        text_fail(error, error_size, "cannot allocate umt5 blocks");
        return NULL;
    }
    char name[224];
    for (uint32_t layer = 0; layer < config->num_layers; layer++) {
        h3_wan22_text_block *block = &encoder->blocks[layer];
        snprintf(name, sizeof(name), "blocks.%u.attn.q.weight", layer);
        if (!h3_wan22_linear_load_quanto(store, name, config->dim_attn,
                                         config->dim, 0, &block->q,
                                         error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.attn.k.weight", layer);
        if (!h3_wan22_linear_load_quanto(store, name, config->dim_attn,
                                         config->dim, 0, &block->k,
                                         error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.attn.v.weight", layer);
        if (!h3_wan22_linear_load_quanto(store, name, config->dim_attn,
                                         config->dim, 0, &block->v,
                                         error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.attn.o.weight", layer);
        if (!h3_wan22_linear_load_quanto(store, name, config->dim,
                                         config->dim_attn, 0, &block->o,
                                         error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.ffn.gate.0.weight", layer);
        if (!h3_wan22_linear_load_quanto(store, name, config->ffn_dim,
                                         config->dim, 0, &block->gate,
                                         error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.ffn.fc1.weight", layer);
        if (!h3_wan22_linear_load_quanto(store, name, config->ffn_dim,
                                         config->dim, 0, &block->fc1,
                                         error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.ffn.fc2.weight", layer);
        if (!h3_wan22_linear_load_quanto(store, name, config->dim,
                                         config->ffn_dim, 0, &block->fc2,
                                         error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.norm1.weight", layer);
        if (!h3_wan22_rms_load(store, name, config->dim, config->norm_eps,
                               &block->norm1, error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name), "blocks.%u.norm2.weight", layer);
        if (!h3_wan22_rms_load(store, name, config->dim, config->norm_eps,
                               &block->norm2, error, error_size)) {
            goto failure;
        }
        snprintf(name, sizeof(name),
                 "blocks.%u.pos_embedding.embedding.weight", layer);
        if (!text_load_pos_bias(store, name, config->num_buckets,
                                config->heads, &block->pos_bias,
                                error, error_size)) {
            goto failure;
        }
    }

    if (!h3_wan22_rms_load(store, "norm.weight", config->dim,
                           config->norm_eps,
                           &encoder->final_norm, error, error_size)) {
        goto failure;
    }
    /* Optional GPU forward path (WAN22_TEXT_GPU=1): the device context is
     * created here but the weights upload lazily on the first forward, so a
     * request served from the disk cache never touches the GPU.  A CUDA
     * failure leaves the encoder on the CPU path. */
    if (wan22_text_gpu_enabled()) {
        char gpu_error[256] = {0};
        if (!h3_wan22_text_gpu_get(encoder, gpu_error, sizeof(gpu_error))) {
            fprintf(stderr,
                    "wan22 text: GPU unavailable (%s), CPU path\n", gpu_error);
            fflush(stderr);
        }
    }
    return encoder;

failure:
    h3_wan22_encoder_free(encoder);
    return NULL;
}

void h3_wan22_encoder_free(h3_wan22_encoder *encoder) {
    if (!encoder) return;
    free(encoder->token_embedding);
    if (encoder->blocks) {
        for (uint32_t layer = 0; layer < encoder->config.num_layers; layer++) {
            text_block_free(&encoder->blocks[layer]);
        }
        free(encoder->blocks);
    }
    h3_wan22_rms_free(&encoder->final_norm);
    h3_wan22_text_gpu_drop(encoder->gpu);
    free(encoder);
}

/* Bidirectional T5 relative bucket, t5.py T5RelativeEmbedding:
 * rel = key - query; sign splits the bucket range in half; small |rel| map
 * linearly, larger log-bucketed between max_exact and num_buckets / 2 - 1. */
static void text_bucket_table(const h3_wan22_text_config *config,
                              uint32_t seq, int32_t *bucket) {
    const int32_t half_buckets = (int32_t)(config->num_buckets / 2);
    const int32_t max_exact = half_buckets / 2;             /* 8 */
    const float log_span = logf(config->max_dist / (float)max_exact);
    for (uint32_t i = 0; i < seq; i++) {
        for (uint32_t j = 0; j < seq; j++) {
            int32_t rel = (int32_t)j - (int32_t)i;
            int32_t sign_bucket = rel > 0 ? half_buckets : 0;
            int32_t absolute = rel > 0 ? rel : -rel;
            int32_t index;
            if (absolute < max_exact) {
                index = absolute;
            } else {
                float scaled = logf((float)absolute / (float)max_exact) /
                               log_span * (float)(half_buckets - max_exact);
                index = max_exact + (int32_t)scaled;        /* trunc toward 0 */
                if (index > half_buckets - 1) index = half_buckets - 1;
            }
            bucket[(size_t)i * seq + j] = sign_bucket + index;
        }
    }
}

/* T5 attention: no scaling, additive [heads, seq, seq] bias, softmax in F32.
 * q/k/v [seq, heads, head_dim] views over the projections. */
static void text_attention(const h3_wan22_text_block *block,
                           const h3_wan22_text_config *config,
                           const int32_t *bucket, const uint8_t *mask,
                           uint32_t seq, float *q, float *k, float *v,
                           float *out, float *scores) {
    const uint32_t heads = config->heads;
    const uint32_t head_dim = config->head_dim;
    {
        int64_t head64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (head64 = 0; head64 < (int64_t)heads; head64++) {
            const uint32_t head = (uint32_t)head64;
            float *head_scores = scores + (size_t)head * seq;
            for (uint32_t i = 0; i < seq; i++) {
                const float *q_row = q + ((size_t)i * heads + head) * head_dim;
                const int32_t *bucket_row = bucket + (size_t)i * seq;
                float maximum = -INFINITY;
                for (uint32_t j = 0; j < seq; j++) {
                    const float *k_row = k + ((size_t)j * heads + head) * head_dim;
                    float dot = 0.0f;
                    for (uint32_t c = 0; c < head_dim; c++) {
                        dot += q_row[c] * k_row[c];
                    }
                    /* pos_bias is [num_buckets, heads] (bucket-major) */
                    dot += block->pos_bias[(size_t)bucket_row[j] * heads + head];
                    if (mask && !mask[j]) dot += TEXT_MASK_BIAS;
                    head_scores[j] = dot;
                    if (dot > maximum) maximum = dot;
                }
                float sum = 0.0f;
                for (uint32_t j = 0; j < seq; j++) {
                    head_scores[j] = expf(head_scores[j] - maximum);
                    sum += head_scores[j];
                }
                float inverse = 1.0f / sum;
                float *out_row = out + ((size_t)i * heads + head) * head_dim;
                memset(out_row, 0, head_dim * sizeof(float));
                for (uint32_t j = 0; j < seq; j++) {
                    const float *v_row = v + ((size_t)j * heads + head) * head_dim;
                    float weight = head_scores[j] * inverse;
                    for (uint32_t c = 0; c < head_dim; c++) {
                        out_row[c] += weight * v_row[c];
                    }
                }
            }
        }
    }
}

/* Defined in the GPU section below (WAN22_TEXT_GPU=1). */
static int wan22_encoder_forward_gpu(h3_wan22_encoder *encoder,
                                     h3_wan22_text_gpu *tgpu,
                                     const int32_t *ids, const uint8_t *mask,
                                     uint32_t seq, float *out,
                                     uint32_t probe_after, float *probe_out,
                                     char *error, size_t error_size);

int h3_wan22_encoder_forward(const h3_wan22_encoder *encoder,
                             const int32_t *ids, const uint8_t *mask,
                             uint32_t seq, float *out,
                             uint32_t probe_after, float *probe_out,
                             char *error, size_t error_size) {
    /* GPU path (WAN22_TEXT_GPU=1 and a live device context): a device error
     * retires the path for this encoder and the CPU reference takes over, so
     * the request still completes -- just at CPU speed. */
    if (encoder->gpu && !encoder->gpu_failed) {
        if (wan22_encoder_forward_gpu((h3_wan22_encoder *)encoder,
                                      encoder->gpu, ids, mask, seq, out,
                                      probe_after, probe_out, error,
                                      error_size))
            return 1;
        fprintf(stderr,
                "wan22 text: GPU forward failed (%s), CPU path\n", error);
        fflush(stderr);
        ((h3_wan22_encoder *)encoder)->gpu_failed = 1;
    }
    const h3_wan22_text_config *config = &encoder->config;
    const uint32_t dim = config->dim;
    const uint32_t ffn_dim = config->ffn_dim;
    const uint32_t heads = config->heads;

    if (seq > config->text_len) {
        text_fail(error, error_size, "seq %u exceeds text_len %u", seq,
                  config->text_len);
        return 0;
    }

    size_t plane = (size_t)seq * dim;
    int32_t *bucket = malloc((size_t)seq * seq * sizeof(int32_t));
    float *x = malloc(plane * sizeof(float));
    float *normed = malloc(plane * sizeof(float));
    float *q = malloc(plane * sizeof(float));
    float *k = malloc(plane * sizeof(float));
    float *v = malloc(plane * sizeof(float));
    float *attn = malloc(plane * sizeof(float));
    float *branch = malloc(plane * sizeof(float));
    float *ffn_h = malloc((size_t)seq * ffn_dim * sizeof(float));
    float *ffn_g = malloc((size_t)seq * ffn_dim * sizeof(float));
    float *scores = malloc((size_t)heads * seq * sizeof(float));
    if (!bucket || !x || !normed || !q || !k || !v || !attn || !branch ||
        !ffn_h || !ffn_g || !scores) {
        text_fail(error, error_size, "cannot allocate umt5 forward buffers");
        free(bucket); free(x); free(normed); free(q); free(k); free(v);
        free(attn); free(branch); free(ffn_h); free(ffn_g); free(scores);
        return 0;
    }

    /* Token embedding lookup (BF16 widened). */
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)seq; row64++) {
            uint32_t id = (uint32_t)ids[row64];
            if (id >= config->vocab) id = 0;    /* out of range -> <pad> */
            const uint16_t *row = encoder->token_embedding +
                                  (size_t)id * dim;
            float *dst = x + (size_t)row64 * dim;
            for (uint32_t c = 0; c < dim; c++) dst[c] = text_bf16_to_f32(row[c]);
        }
    }

    text_bucket_table(config, seq, bucket);

    for (uint32_t layer = 0; layer < config->num_layers; layer++) {
        const h3_wan22_text_block *block = &encoder->blocks[layer];

        /* attention: x += o(softmax(qk + bias) v) */
        h3_wan22_rms_forward(&block->norm1, x, seq, normed);
        h3_wan22_linear_forward(&block->q, normed, seq, q);
        h3_wan22_linear_forward(&block->k, normed, seq, k);
        h3_wan22_linear_forward(&block->v, normed, seq, v);
        text_attention(block, config, bucket, mask, seq, q, k, v, attn,
                       scores);
        h3_wan22_linear_forward(&block->o, attn, seq, branch);
        for (size_t index = 0; index < plane; index++) x[index] += branch[index];

        /* FFN: x += fc2(gelu_tanh(gate(n)) * fc1(n)) */
        h3_wan22_rms_forward(&block->norm2, x, seq, normed);
        h3_wan22_linear_forward(&block->gate, normed, seq, ffn_g);
        h3_wan22_linear_forward(&block->fc1, normed, seq, ffn_h);
        for (size_t index = 0; index < (size_t)seq * ffn_dim; index++) {
            float value = ffn_g[index];
            float inner = 0.7978845608028654f *
                          (value + 0.044715f * value * value * value);
            ffn_h[index] *= 0.5f * value * (1.0f + tanhf(inner));
        }
        h3_wan22_linear_forward(&block->fc2, ffn_h, seq, branch);
        for (size_t index = 0; index < plane; index++) x[index] += branch[index];

        if (probe_after && layer + 1 == probe_after && probe_out) {
            memcpy(probe_out, x, plane * sizeof(float));
        }
    }

    h3_wan22_rms_forward(&encoder->final_norm, x, seq, out);

    free(bucket);
    free(x);
    free(normed);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(branch);
    free(ffn_h);
    free(ffn_g);
    free(scores);
    return 1;
}

/* --- GPU forward path (WAN22_TEXT_GPU=1) ------------------------------------
 * Same arithmetic as the CPU reference above: the host quanto-dequantized
 * BF16 weights are uploaded verbatim, activations stay F32 (the GEMMs are
 * BF16-weight x F32 with exact FP32 FMAs, no TF32), and the attention runs
 * through h3_gpu_t5_attn_f32, which reproduces text_attention()'s math
 * (no 1/sqrt(d) scale, additive per-head relative position bias, padded keys
 * masked to -inf, F32 softmax).  Differences vs the CPU path are limited to
 * summation order inside the GEMMs and the softmax reductions.
 *
 * The token embedding lookup stays on the host (one [seq, dim] plane per
 * forward, 8 MiB at the serve window): uploading the 2.1 GiB table for a
 * 512-row gather is not worth the device budget.
 *
 * Residency mirrors the DiT (h3_wan22_dit.c): weights are uploaded on the
 * first forward, kept for that request, and dropped by
 * h3_wan22_encoder_release_gpu at the end of the text phase so the DiT+VAE
 * phases keep their device budget.  A later forward re-uploads from the host
 * copy through pinned staging (pageable cudaMemcpy measures ~1.8 GB/s here).
 * --------------------------------------------------------------------------- */

static double wan22_text_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    h3_gpu_tensor *q, *k, *v, *o;      /* BF16 [dim, dim] */
    h3_gpu_tensor *gate, *fc1, *fc2;   /* BF16 [ffn, dim] / [dim, ffn] */
    h3_gpu_tensor *norm1, *norm2;      /* F32 [dim], widened from BF16 */
    h3_gpu_tensor *pos_bias;           /* F32 [num_buckets * heads] */
} wan22_text_gpu_block;

struct h3_wan22_text_gpu {
    h3_gpu *gpu;
    h3_wan22_text_config config;
    wan22_text_gpu_block *blocks;      /* [num_layers] */
    h3_gpu_tensor *final_norm;         /* F32 [dim] */
    int uploaded;                      /* weights are on the device */
    /* Scratch, reserved for the largest window seen so far. */
    uint32_t seq;
    h3_gpu_tensor *x;          /* F32 [seq, dim] residual carrier */
    h3_gpu_tensor *normed;     /* F32 [seq, dim] */
    h3_gpu_tensor *q, *k, *v;  /* F32 [seq, dim] */
    h3_gpu_tensor *attn;       /* F32 [seq, dim] */
    h3_gpu_tensor *branch;     /* F32 [seq, dim] */
    h3_gpu_tensor *ffn_g;      /* F32 [seq, ffn_dim] */
    h3_gpu_tensor *ffn_h;      /* F32 [seq, ffn_dim] */
    h3_gpu_tensor *buckets;    /* U32 [seq * seq] */
    uint32_t bucket_seq;       /* window the bucket table was built for */
    h3_gpu_tensor *mask;       /* U32 [seq], 0 = padded key */
    int last_forward_gpu;      /* the previous forward ran on the device */
};

static void wan22_text_gpu_block_free(wan22_text_gpu_block *block) {
    if (!block) return;
    h3_gpu_tensor_free(block->q);
    h3_gpu_tensor_free(block->k);
    h3_gpu_tensor_free(block->v);
    h3_gpu_tensor_free(block->o);
    h3_gpu_tensor_free(block->gate);
    h3_gpu_tensor_free(block->fc1);
    h3_gpu_tensor_free(block->fc2);
    h3_gpu_tensor_free(block->norm1);
    h3_gpu_tensor_free(block->norm2);
    h3_gpu_tensor_free(block->pos_bias);
    memset(block, 0, sizeof(*block));
}

static void wan22_text_gpu_release_scratch(h3_wan22_text_gpu *tgpu) {
    h3_gpu_tensor_free(tgpu->x);
    h3_gpu_tensor_free(tgpu->normed);
    h3_gpu_tensor_free(tgpu->q);
    h3_gpu_tensor_free(tgpu->k);
    h3_gpu_tensor_free(tgpu->v);
    h3_gpu_tensor_free(tgpu->attn);
    h3_gpu_tensor_free(tgpu->branch);
    h3_gpu_tensor_free(tgpu->ffn_g);
    h3_gpu_tensor_free(tgpu->ffn_h);
    h3_gpu_tensor_free(tgpu->buckets);
    h3_gpu_tensor_free(tgpu->mask);
    tgpu->x = tgpu->normed = tgpu->q = tgpu->k = tgpu->v = NULL;
    tgpu->attn = tgpu->branch = tgpu->ffn_g = tgpu->ffn_h = NULL;
    tgpu->buckets = tgpu->mask = NULL;
    tgpu->seq = 0;
    tgpu->bucket_seq = 0;
}

void h3_wan22_text_gpu_drop(h3_wan22_text_gpu *tgpu) {
    if (!tgpu) return;
    if (tgpu->gpu) {
        /* Frees are stream-ordered pool returns; drain before teardown. */
        (void)h3_gpu_submit(tgpu->gpu);
    }
    if (tgpu->blocks) {
        for (uint32_t layer = 0; layer < tgpu->config.num_layers; layer++)
            wan22_text_gpu_block_free(&tgpu->blocks[layer]);
        free(tgpu->blocks);
    }
    h3_gpu_tensor_free(tgpu->final_norm);
    wan22_text_gpu_release_scratch(tgpu);
    if (tgpu->gpu) h3_gpu_free(tgpu->gpu);
    free(tgpu);
}

h3_wan22_text_gpu *h3_wan22_text_gpu_get(h3_wan22_encoder *encoder,
                                         char *error, size_t error_size) {
    if (!encoder) return NULL;
    if (encoder->gpu) return encoder->gpu;
    if (!wan22_text_gpu_enabled()) return NULL;
    h3_wan22_text_gpu *tgpu = calloc(1, sizeof(*tgpu));
    if (!tgpu) {
        text_fail(error, error_size, "cannot allocate umt5 GPU context");
        return NULL;
    }
    tgpu->config = encoder->config;
    tgpu->gpu = h3_gpu_create(NULL, error, error_size);
    if (!tgpu->gpu) {
        free(tgpu);
        return NULL;
    }
    tgpu->blocks = calloc(encoder->config.num_layers,
                          sizeof(*tgpu->blocks));
    if (!tgpu->blocks) {
        text_fail(error, error_size, "cannot allocate umt5 GPU blocks");
        h3_wan22_text_gpu_drop(tgpu);
        return NULL;
    }
    encoder->gpu = tgpu;
    return tgpu;
}

int h3_wan22_text_gpu_active(const h3_wan22_encoder *encoder) {
    return encoder && encoder->gpu && !encoder->gpu_failed;
}

const char *h3_wan22_text_device(const h3_wan22_encoder *encoder) {
    if (encoder && encoder->gpu && !encoder->gpu_failed &&
        encoder->gpu->last_forward_gpu)
        return "gpu";
    return "cpu";
}

/* Upload one block's host BF16 weights (verbatim) plus the widened F32 norm
 * and position-bias tables.  Returns 1 on success. */
static int wan22_text_block_upload_gpu(const h3_wan22_text_block *block,
                                       wan22_text_gpu_block *dev,
                                       h3_wan22_text_gpu *tgpu, char *error,
                                       size_t error_size) {
    h3_gpu *gpu = tgpu->gpu;
    const h3_wan22_text_config *config = &tgpu->config;
    uint32_t dim = config->dim;
    uint32_t ffn = config->ffn_dim;

    struct {
        const h3_wan22_linear *src;
        h3_gpu_tensor **slot;
        uint32_t out, in;
        const char *name;
    } linears[] = {
        {&block->q, &dev->q, dim, dim, "q"},
        {&block->k, &dev->k, dim, dim, "k"},
        {&block->v, &dev->v, dim, dim, "v"},
        {&block->o, &dev->o, dim, dim, "o"},
        {&block->gate, &dev->gate, ffn, dim, "gate"},
        {&block->fc1, &dev->fc1, ffn, dim, "fc1"},
        {&block->fc2, &dev->fc2, dim, ffn, "fc2"},
    };
    for (size_t index = 0; index < sizeof(linears) / sizeof(linears[0]);
         index++) {
        size_t count = (size_t)linears[index].out * linears[index].in;
        if (*linears[index].slot)
            h3_gpu_tensor_free(*linears[index].slot);
        *linears[index].slot = h3_gpu_tensor_new_bf16(gpu, count);
        if (!*linears[index].slot ||
            !h3_gpu_upload_bf16_pinned(gpu, *linears[index].slot,
                                       linears[index].src->weight, count)) {
            text_fail(error, error_size, "cannot upload umt5 %s weight: %s",
                      linears[index].name, h3_gpu_error(gpu));
            return 0;
        }
    }

    const h3_wan22_rms *norms[2] = {&block->norm1, &block->norm2};
    h3_gpu_tensor **norm_slots[2] = {&dev->norm1, &dev->norm2};
    for (int index = 0; index < 2; index++) {
        float *staged = malloc((size_t)dim * sizeof(float));
        if (!staged) {
            text_fail(error, error_size, "cannot stage umt5 norm weight");
            return 0;
        }
        for (uint32_t c = 0; c < dim; c++)
            staged[c] = text_bf16_to_f32(norms[index]->weight[c]);
        if (*norm_slots[index]) h3_gpu_tensor_free(*norm_slots[index]);
        *norm_slots[index] =
            h3_gpu_tensor_from_f32(gpu, staged, dim);
        free(staged);
        if (!*norm_slots[index]) {
            text_fail(error, error_size, "cannot upload umt5 norm: %s",
                      h3_gpu_error(gpu));
            return 0;
        }
    }

    if (dev->pos_bias) h3_gpu_tensor_free(dev->pos_bias);
    dev->pos_bias = h3_gpu_tensor_from_f32(
        gpu, block->pos_bias,
        (size_t)config->num_buckets * config->heads);
    if (!dev->pos_bias) {
        text_fail(error, error_size, "cannot upload umt5 pos_bias: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

static int wan22_text_gpu_upload(h3_wan22_encoder *encoder,
                                 h3_wan22_text_gpu *tgpu, char *error,
                                 size_t error_size) {
    if (tgpu->uploaded) return 1;
    double started = wan22_text_now();
    for (uint32_t layer = 0; layer < encoder->config.num_layers; layer++) {
        if (!wan22_text_block_upload_gpu(&encoder->blocks[layer],
                                         &tgpu->blocks[layer], tgpu, error,
                                         error_size))
            return 0;
    }
    float *staged = malloc((size_t)encoder->config.dim * sizeof(float));
    if (!staged) {
        text_fail(error, error_size, "cannot stage umt5 final norm");
        return 0;
    }
    for (uint32_t c = 0; c < encoder->config.dim; c++)
        staged[c] = text_bf16_to_f32(encoder->final_norm.weight[c]);
    if (tgpu->final_norm) h3_gpu_tensor_free(tgpu->final_norm);
    tgpu->final_norm = h3_gpu_tensor_from_f32(tgpu->gpu, staged,
                                              encoder->config.dim);
    free(staged);
    if (!tgpu->final_norm) {
        text_fail(error, error_size, "cannot upload umt5 final norm: %s",
                  h3_gpu_error(tgpu->gpu));
        return 0;
    }
    tgpu->uploaded = 1;
    h3_gpu_stats stats = {0};
    h3_gpu_get_stats(tgpu->gpu, &stats);
    fprintf(stderr,
            "wan22 text: device weights uploaded in %6.2fs (%.2f GiB live)\n",
            wan22_text_now() - started,
            (double)stats.live_bytes / (double)(1u << 30));
    fflush(stderr);
    return 1;
}

/* Reserve the [seq, ...] scratch (idempotent for a smaller or equal window). */
static int wan22_text_gpu_reserve(h3_wan22_text_gpu *tgpu, uint32_t seq,
                                  char *error, size_t error_size) {
    if (seq <= tgpu->seq) return 1;
    wan22_text_gpu_release_scratch(tgpu);
    const h3_wan22_text_config *config = &tgpu->config;
    size_t plane = (size_t)seq * config->dim;
    size_t wide = (size_t)seq * config->ffn_dim;
    h3_gpu *gpu = tgpu->gpu;
    tgpu->x = h3_gpu_tensor_new_f32(gpu, plane);
    tgpu->normed = h3_gpu_tensor_new_f32(gpu, plane);
    tgpu->q = h3_gpu_tensor_new_f32(gpu, plane);
    tgpu->k = h3_gpu_tensor_new_f32(gpu, plane);
    tgpu->v = h3_gpu_tensor_new_f32(gpu, plane);
    tgpu->attn = h3_gpu_tensor_new_f32(gpu, plane);
    tgpu->branch = h3_gpu_tensor_new_f32(gpu, plane);
    tgpu->ffn_g = h3_gpu_tensor_new_f32(gpu, wide);
    tgpu->ffn_h = h3_gpu_tensor_new_f32(gpu, wide);
    if (!tgpu->x || !tgpu->normed || !tgpu->q || !tgpu->k || !tgpu->v ||
        !tgpu->attn || !tgpu->branch || !tgpu->ffn_g || !tgpu->ffn_h) {
        text_fail(error, error_size, "cannot allocate umt5 GPU scratch: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    tgpu->seq = seq;
    return 1;
}

int h3_wan22_encoder_release_gpu(h3_wan22_encoder *encoder, char *error,
                                 size_t error_size) {
    (void)error;
    (void)error_size;
    if (!encoder || !encoder->gpu || !encoder->gpu->uploaded) return 1;
    h3_wan22_text_gpu *tgpu = encoder->gpu;
    /* Drain the stream first: the frees below hand pool blocks back, and the
     * VAE decode's own context takes them from the same pool. */
    (void)h3_gpu_submit(tgpu->gpu);
    for (uint32_t layer = 0; layer < tgpu->config.num_layers; layer++)
        wan22_text_gpu_block_free(&tgpu->blocks[layer]);
    h3_gpu_tensor_free(tgpu->final_norm);
    tgpu->final_norm = NULL;
    wan22_text_gpu_release_scratch(tgpu);
    tgpu->uploaded = 0;
    return 1;
}

static int wan22_encoder_forward_gpu(h3_wan22_encoder *encoder,
                                     h3_wan22_text_gpu *tgpu,
                                     const int32_t *ids, const uint8_t *mask,
                                     uint32_t seq, float *out,
                                     uint32_t probe_after, float *probe_out,
                                     char *error, size_t error_size) {
    const h3_wan22_text_config *config = &tgpu->config;
    uint32_t dim = config->dim;
    uint32_t ffn_dim = config->ffn_dim;
    uint32_t heads = config->heads;
    uint32_t head_dim = config->head_dim;
    h3_gpu *gpu = tgpu->gpu;

    if (!wan22_text_gpu_upload(encoder, tgpu, error, error_size)) return 0;
    if (!wan22_text_gpu_reserve(tgpu, seq, error, error_size)) return 0;

    size_t plane = (size_t)seq * dim;
    size_t wide = (size_t)seq * ffn_dim;

    /* Token embedding lookup on the host, one F32 plane uploaded. */
    float *embedded = malloc(plane * sizeof(float));
    if (!embedded) {
        text_fail(error, error_size, "cannot stage umt5 embedding plane");
        return 0;
    }
    for (uint32_t row = 0; row < seq; row++) {
        uint32_t id = (uint32_t)ids[row];
        if (id >= config->vocab) id = 0;    /* out of range -> <pad> */
        const uint16_t *source =
            encoder->token_embedding + (size_t)id * dim;
        float *dst = embedded + (size_t)row * dim;
        for (uint32_t c = 0; c < dim; c++)
            dst[c] = text_bf16_to_f32(source[c]);
    }
    if (!h3_gpu_tensor_write_f32(tgpu->x, embedded, plane)) {
        free(embedded);
        text_fail(error, error_size, "cannot upload umt5 embedding: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    free(embedded);

    /* Key mask (U32 view of the uint8_t flags); NULL mask = all real. */
    if (mask) {
        uint32_t *staged = malloc((size_t)seq * sizeof(uint32_t));
        if (!staged) {
            text_fail(error, error_size, "cannot stage umt5 mask");
            return 0;
        }
        for (uint32_t row = 0; row < seq; row++) staged[row] = mask[row] ? 1u : 0u;
        if (tgpu->mask) h3_gpu_tensor_free(tgpu->mask);
        tgpu->mask = h3_gpu_tensor_from_u32(gpu, staged, seq);
        free(staged);
        if (!tgpu->mask) {
            text_fail(error, error_size, "cannot upload umt5 mask: %s",
                      h3_gpu_error(gpu));
            return 0;
        }
    } else if (tgpu->mask) {
        h3_gpu_tensor_free(tgpu->mask);
        tgpu->mask = NULL;
    }

    /* Relative-position bucket table: identical indices to the CPU path's
     * text_bucket_table(), cached per window length. */
    if (tgpu->bucket_seq != seq) {
        int32_t *table = malloc((size_t)seq * seq * sizeof(int32_t));
        if (!table) {
            text_fail(error, error_size, "cannot stage umt5 bucket table");
            return 0;
        }
        text_bucket_table(config, seq, table);
        if (tgpu->buckets) h3_gpu_tensor_free(tgpu->buckets);
        tgpu->buckets = h3_gpu_tensor_from_u32(gpu, (const uint32_t *)table,
                                               (size_t)seq * seq);
        free(table);
        if (!tgpu->buckets) {
            text_fail(error, error_size, "cannot upload umt5 buckets: %s",
                      h3_gpu_error(gpu));
            return 0;
        }
        tgpu->bucket_seq = seq;
    }

    if (!h3_gpu_begin(gpu)) {
        text_fail(error, error_size, "cannot begin umt5 GPU forward");
        return 0;
    }
    for (uint32_t layer = 0; layer < config->num_layers; layer++) {
        const wan22_text_gpu_block *dev = &tgpu->blocks[layer];

        /* attention: x += o(softmax(qk + bias) v) */
        if (!h3_gpu_rms_norm_f32(gpu, tgpu->normed, tgpu->x, dev->norm1, seq,
                                 dim, config->norm_eps) ||
            !h3_gpu_linear_bf16w_f32(gpu, tgpu->q, tgpu->normed, dev->q, seq,
                                     dim, dim) ||
            !h3_gpu_linear_bf16w_f32(gpu, tgpu->k, tgpu->normed, dev->k, seq,
                                     dim, dim) ||
            !h3_gpu_linear_bf16w_f32(gpu, tgpu->v, tgpu->normed, dev->v, seq,
                                     dim, dim) ||
            !h3_gpu_t5_attn_f32(gpu, tgpu->attn, tgpu->q, tgpu->k, tgpu->v,
                                tgpu->buckets, dev->pos_bias, tgpu->mask, seq,
                                heads, head_dim, config->num_buckets) ||
            !h3_gpu_linear_bf16w_f32(gpu, tgpu->branch, tgpu->attn, dev->o, seq,
                                     dim, dim) ||
            !h3_gpu_add_f32(gpu, tgpu->x, tgpu->x, tgpu->branch,
                            (uint32_t)plane)) {
            text_fail(error, error_size, "umt5 GPU attention failed: %s",
                      h3_gpu_error(gpu));
            return 0;
        }

        /* FFN: x += fc2(gelu_tanh(gate(n)) * fc1(n)) */
        if (!h3_gpu_rms_norm_f32(gpu, tgpu->normed, tgpu->x, dev->norm2, seq,
                                 dim, config->norm_eps) ||
            !h3_gpu_linear_bf16w_f32(gpu, tgpu->ffn_g, tgpu->normed, dev->gate,
                                     seq, dim, ffn_dim) ||
            !h3_gpu_linear_bf16w_f32(gpu, tgpu->ffn_h, tgpu->normed, dev->fc1,
                                     seq, dim, ffn_dim) ||
            !h3_gpu_geglu_f32(gpu, tgpu->ffn_h, tgpu->ffn_g, tgpu->ffn_h,
                              (uint32_t)wide) ||
            !h3_gpu_linear_bf16w_f32(gpu, tgpu->branch, tgpu->ffn_h, dev->fc2,
                                     seq, ffn_dim, dim) ||
            !h3_gpu_add_f32(gpu, tgpu->x, tgpu->x, tgpu->branch,
                            (uint32_t)plane)) {
            text_fail(error, error_size, "umt5 GPU FFN failed: %s",
                      h3_gpu_error(gpu));
            return 0;
        }

        if (probe_after && layer + 1 == probe_after && probe_out) {
            if (!h3_gpu_submit(gpu) ||
                !h3_gpu_tensor_read_f32(tgpu->x, probe_out, plane)) {
                text_fail(error, error_size,
                          "umt5 GPU probe readback failed: %s",
                          h3_gpu_error(gpu));
                return 0;
            }
        }
    }

    h3_gpu_tensor *out_dev = h3_gpu_tensor_new_f32(gpu, plane);
    if (!out_dev ||
        !h3_gpu_rms_norm_f32(gpu, out_dev, tgpu->x, tgpu->final_norm, seq, dim,
                             config->norm_eps) ||
        !h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_f32(out_dev, out, plane)) {
        h3_gpu_tensor_free(out_dev);
        text_fail(error, error_size, "umt5 GPU final norm failed: %s",
                  h3_gpu_error(gpu));
        return 0;
    }
    h3_gpu_tensor_free(out_dev);
    tgpu->last_forward_gpu = 1;
    return 1;
}

/* --- serve context cache ---------------------------------------------------- */
/* One encoder output is seq x dim F32 (8 MiB at the serve window), so a
 * repeated token sequence can skip the ~62 s forward entirely.  Entries are
 * self-describing (magic, seq, dim) and published by rename, like the h3
 * text cache; the payload is stored as raw F32 bytes, so a hit reproduces the
 * stored plane bit for bit. */

/* key = sha256("wan22-text-cache-1|identity|seq|dim|ids") as hex. */
void h3_wan22_text_cache_key(const char *identity, const int32_t *ids,
                             uint32_t seq, uint32_t dim, char hex[65]) {
    char header[320];
    int length = snprintf(header, sizeof(header),
                          "wan22-text-cache-1|identity=%s|seq=%u|dim=%u|ids=",
                          identity && *identity ? identity : "none", seq, dim);
    h3_sha256_ctx ctx;
    h3_sha256_init(&ctx);
    if (length > 0) {
        size_t used = (size_t)length < sizeof(header) ? (size_t)length
                                                      : sizeof(header) - 1;
        h3_sha256_update(&ctx, header, used);
    }
    if (ids && seq) h3_sha256_update(&ctx, ids, (size_t)seq * sizeof(*ids));
    uint8_t digest[H3_SHA256_DIGEST_SIZE];
    h3_sha256_final(&ctx, digest);
    static const char digits[] = "0123456789abcdef";
    for (unsigned index = 0; index < H3_SHA256_DIGEST_SIZE; index++) {
        hex[index * 2] = digits[digest[index] >> 4];
        hex[index * 2 + 1] = digits[digest[index] & 0x0fu];
    }
    hex[H3_SHA256_DIGEST_SIZE * 2] = 0;
}

/* mkdir -p, so the leaf of a cache directory does not have to exist yet. */
static int text_cache_mkdir_p(const char *path) {
    char copy[1024];
    size_t length = strlen(path);
    if (!length || length >= sizeof(copy)) return 0;
    memcpy(copy, path, length + 1);
    for (char *cursor = copy + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = 0;
        if (mkdir(copy, 0755) != 0 && errno != EEXIST) return 0;
        *cursor = '/';
    }
    return mkdir(copy, 0755) == 0 || errno == EEXIST;
}

int h3_wan22_text_cache_init(h3_wan22_text_cache *cache, const char *dir) {
    if (!cache) return 0;
    cache->dir[0] = 0;
    if (!dir || !*dir) return 1;                 /* explicitly disabled */
    if (!text_cache_mkdir_p(dir)) return 0;
    snprintf(cache->dir, sizeof(cache->dir), "%s", dir);
    return 1;
}

static int text_cache_path(const h3_wan22_text_cache *cache, const char *key,
                           const char *suffix, char *path, size_t capacity) {
    if (!cache || !cache->dir[0] || !key) return 0;
    int written = snprintf(path, capacity, "%s/%s%s", cache->dir, key, suffix);
    return written > 0 && (size_t)written < capacity;
}

int h3_wan22_text_cache_load(const h3_wan22_text_cache *cache,
                             const char *identity, const int32_t *ids,
                             uint32_t seq, uint32_t dim, float *out) {
    if (!cache || !cache->dir[0] || !out || !seq || !dim) return 0;
    char key[65], path[1100];
    h3_wan22_text_cache_key(identity, ids, seq, dim, key);
    if (!text_cache_path(cache, key, ".bin", path, sizeof(path))) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    char magic[8];
    uint64_t entry_seq = 0, entry_dim = 0;
    size_t count = (size_t)seq * dim;
    int ok = fread(magic, 1, sizeof(magic), file) == sizeof(magic) &&
        memcmp(magic, H3_WAN22_TEXT_CACHE_MAGIC, sizeof(magic)) == 0 &&
        fread(&entry_seq, sizeof(entry_seq), 1, file) == 1 &&
        fread(&entry_dim, sizeof(entry_dim), 1, file) == 1 &&
        entry_seq == seq && entry_dim == dim &&
        fread(out, sizeof(*out), count, file) == count;
    fclose(file);
    return ok;
}

/* Atomic publish: write <key>.bin.tmp-<pid>, rename it over the final name,
 * then write the sibling manifest.  A crash mid-write leaves a temporary
 * file behind, never a truncated entry. */
int h3_wan22_text_cache_store(const h3_wan22_text_cache *cache,
                              const char *identity, const int32_t *ids,
                              uint32_t seq, uint32_t dim,
                              const float *values) {
    if (!cache || !cache->dir[0] || !values || !seq || !dim) return 0;
    char key[65], final_path[1100], temp_path[1150];
    h3_wan22_text_cache_key(identity, ids, seq, dim, key);
    if (!text_cache_path(cache, key, ".bin", final_path, sizeof(final_path)))
        return 0;
    int written = snprintf(temp_path, sizeof(temp_path), "%s.tmp-%ld",
                           final_path, (long)getpid());
    if (written < 0 || (size_t)written >= sizeof(temp_path)) return 0;
    FILE *file = fopen(temp_path, "wb");
    if (!file) return 0;
    uint64_t entry_seq = seq, entry_dim = dim;
    size_t count = (size_t)seq * dim;
    int ok = fwrite(H3_WAN22_TEXT_CACHE_MAGIC, 1, 8, file) == 8 &&
        fwrite(&entry_seq, sizeof(entry_seq), 1, file) == 1 &&
        fwrite(&entry_dim, sizeof(entry_dim), 1, file) == 1 &&
        fwrite(values, sizeof(*values), count, file) == count;
    if (fclose(file) != 0) ok = 0;
    if (!ok || rename(temp_path, final_path) != 0) {
        unlink(temp_path);
        return 0;
    }
    char manifest_path[1100], manifest_temp[1150];
    if (snprintf(manifest_path, sizeof(manifest_path), "%s/%s.json",
                 cache->dir, key) < (int)sizeof(manifest_path) &&
        snprintf(manifest_temp, sizeof(manifest_temp), "%s.tmp-%ld",
                 manifest_path, (long)getpid()) <
            (int)sizeof(manifest_temp)) {
        FILE *manifest = fopen(manifest_temp, "w");
        if (manifest) {
            fprintf(manifest,
                    "{\"key\":\"%s\",\"seq\":%llu,\"dim\":%llu,"
                    "\"bytes\":%llu,\"identity\":\"%s\"}\n",
                    key, (unsigned long long)seq, (unsigned long long)dim,
                    (unsigned long long)count * sizeof(*values),
                    identity && *identity ? identity : "");
            if (fclose(manifest) == 0)
                (void)rename(manifest_temp, manifest_path);
            else unlink(manifest_temp);
        }
    }
    return 1;
}

/* --- text_embedding MLP ------------------------------------------------------ */

int h3_wan22_text_mlp_load(const h3_weight_store *store,
                           uint32_t in_dim, uint32_t hidden_dim,
                           uint32_t out_dim, h3_wan22_text_mlp *mlp,
                           char *error, size_t error_size) {
    mlp->in = in_dim;
    mlp->hidden = hidden_dim;
    mlp->out = out_dim;
    if (!h3_wan22_linear_load_bf16(store, "text_embedding.0.weight",
                                   hidden_dim, in_dim, 1, &mlp->fc0,
                                   error, error_size)) {
        return 0;
    }
    if (!h3_wan22_linear_load_bf16(store, "text_embedding.2.weight",
                                   out_dim, hidden_dim, 1, &mlp->fc2,
                                   error, error_size)) {
        h3_wan22_text_mlp_free(mlp);
        return 0;
    }
    return 1;
}

void h3_wan22_text_mlp_free(h3_wan22_text_mlp *mlp) {
    h3_wan22_linear_free(&mlp->fc0);
    h3_wan22_linear_free(&mlp->fc2);
}

int h3_wan22_text_mlp_forward(const h3_wan22_text_mlp *mlp,
                              const float *input, uint32_t rows,
                              float *output) {
    float *hidden = malloc((size_t)rows * mlp->hidden * sizeof(float));
    if (!hidden) return 0;
    h3_wan22_linear_forward(&mlp->fc0, input, rows, hidden);
    for (size_t index = 0; index < (size_t)rows * mlp->hidden; index++) {
        float value = hidden[index];
        float inner = 0.7978845608028654f *
                      (value + 0.044715f * value * value * value);
        hidden[index] = 0.5f * value * (1.0f + tanhf(inner));
    }
    h3_wan22_linear_forward(&mlp->fc2, hidden, rows, output);
    free(hidden);
    return 1;
}
