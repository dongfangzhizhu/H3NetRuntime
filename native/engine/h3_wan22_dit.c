#include "h3_wan22_dit.h"
#include "h3_gpu.h"

#include <math.h>
#include <stdarg.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- resume / reserve trace (H3_WAN22_FWD_TRACE=1) ------------------------
 * The pipeline prints one line per forward (h3_wan22_pipeline.c); these
 * helpers split a resume into its weight re-upload and scratch rebuild
 * halves and report the stream-ordered pool counters, so a slow resume is
 * attributed to upload bandwidth or pool growth instead of guessed at.
 * Pure instrumentation: the resume path itself is unchanged. */
static double wan22_dit_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int wan22_dit_trace(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("H3_WAN22_FWD_TRACE");
        cached = value && *value && strcmp(value, "0") != 0;
    }
    return cached;
}

/* --- small helpers --------------------------------------------------------- */

static float wan_bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t wan_bf16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}


static void wan_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

h3_wan22_config h3_wan22_default_config(void) {
    h3_wan22_config config;
    config.dim = 3072;
    config.heads = 24;
    config.head_dim = 128;
    config.ffn_dim = 14336;
    config.freq_dim = 256;
    config.text_len = 512;
    config.norm_eps = 1e-6f;
    config.rope_axes[0] = 44;
    config.rope_axes[1] = 42;
    config.rope_axes[2] = 42;
    config.rope_theta = 10000.0f;
    return config;
}

/* Read a plain BF16 tensor flattened to `count` elements. */
static int read_bf16_flat(const h3_weight_store *store, const char *name,
                          uint64_t count, uint16_t *out,
                          char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        wan_fail(error, error_size, "required Wan 2.2 weight is absent: %s", name);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(tensor) != count) {
        wan_fail(error, error_size,
                 "weight %s is %s[%d] with %llu elements, expected BF16 with %llu",
                 name, h3_dtype_name(tensor->dtype), tensor->ndim,
                 (unsigned long long)h3_st_tensor_elements(tensor),
                 (unsigned long long)count);
        return 0;
    }
    return h3_st_read_data(header, tensor, out, (size_t)count * 2,
                           error, error_size);
}

/* Read a plain F32 tensor flattened to `count` elements. */
static int read_f32_flat(const h3_weight_store *store, const char *name,
                         uint64_t count, float *out,
                         char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        wan_fail(error, error_size, "required Wan 2.2 weight is absent: %s", name);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) != count) {
        wan_fail(error, error_size,
                 "weight %s is %s[%d] with %llu elements, expected F32 with %llu",
                 name, h3_dtype_name(tensor->dtype), tensor->ndim,
                 (unsigned long long)h3_st_tensor_elements(tensor),
                 (unsigned long long)count);
        return 0;
    }
    return h3_st_read_data(header, tensor, out, (size_t)count * 4,
                           error, error_size);
}

/* "<base>.weight" -> "<base>.bias". */
static void bias_tensor_name(const char *weight_name, char *out, size_t size) {
    static const char suffix[] = ".weight";
    size_t length = strlen(weight_name);
    size_t keep = length;
    if (length >= sizeof(suffix) - 1 &&
        strcmp(weight_name + length - (sizeof(suffix) - 1), suffix) == 0) {
        keep = length - (sizeof(suffix) - 1);
    }
    snprintf(out, size, "%.*s.bias", (int)keep, weight_name);
}

/* --- Linear ---------------------------------------------------------------- */

int h3_wan22_linear_load_quanto(const h3_weight_store *store, const char *name,
                                uint32_t out, uint32_t in, int has_bias,
                                h3_wan22_linear *linear,
                                char *error, size_t error_size) {
    linear->gpu_weight = NULL;
    linear->gpu_bias = NULL;
    linear->out = out;
    linear->in = in;
    linear->weight = NULL;
    linear->weight_f32 = NULL;
    linear->bias = NULL;
    linear->weight = h3_quanto_load_bf16(store, name, out, in, error, error_size);
    if (!linear->weight) return 0;
    if (has_bias) {
        char bias_name[224];
        bias_tensor_name(name, bias_name, sizeof(bias_name));
        linear->bias = malloc((size_t)out * sizeof(float));
        uint16_t *staged = linear->bias ? malloc((size_t)out * 2) : NULL;
        if (!staged ||
            !read_bf16_flat(store, bias_name, out, staged, error, error_size)) {
            free(staged);
            h3_wan22_linear_free(linear);
            return 0;
        }
        for (uint32_t index = 0; index < out; index++) {
            linear->bias[index] = wan_bf16_to_f32(staged[index]);
        }
        free(staged);
    }
    return 1;
}

int h3_wan22_linear_load_bf16(const h3_weight_store *store, const char *name,
                              uint32_t out, uint32_t in, int has_bias,
                              h3_wan22_linear *linear,
                              char *error, size_t error_size) {
    linear->gpu_weight = NULL;
    linear->gpu_bias = NULL;
    linear->out = out;
    linear->in = in;
    linear->weight = NULL;
    linear->weight_f32 = NULL;
    linear->bias = NULL;
    linear->weight = malloc((size_t)out * in * 2);
    if (!linear->weight) {
        wan_fail(error, error_size, "cannot allocate linear weight %s", name);
        return 0;
    }
    if (!read_bf16_flat(store, name, (uint64_t)out * in, linear->weight,
                        error, error_size)) {
        h3_wan22_linear_free(linear);
        return 0;
    }
    if (has_bias) {
        char bias_name[224];
        bias_tensor_name(name, bias_name, sizeof(bias_name));
        linear->bias = malloc((size_t)out * sizeof(float));
        uint16_t *staged = linear->bias ? malloc((size_t)out * 2) : NULL;
        if (!staged ||
            !read_bf16_flat(store, bias_name, out, staged, error, error_size)) {
            free(staged);
            h3_wan22_linear_free(linear);
            return 0;
        }
        for (uint32_t index = 0; index < out; index++) {
            linear->bias[index] = wan_bf16_to_f32(staged[index]);
        }
        free(staged);
    }
    return 1;
}

int h3_wan22_linear_load_f32(const h3_weight_store *store, const char *name,
                             uint32_t out, uint32_t in, int has_bias,
                             h3_wan22_linear *linear,
                             char *error, size_t error_size) {
    linear->gpu_weight = NULL;
    linear->gpu_bias = NULL;
    linear->out = out;
    linear->in = in;
    linear->weight = NULL;
    linear->weight_f32 = NULL;
    linear->bias = NULL;
    linear->weight_f32 = malloc((size_t)out * in * sizeof(float));
    if (!linear->weight_f32) {
        wan_fail(error, error_size, "cannot allocate linear weight %s", name);
        return 0;
    }
    if (!read_f32_flat(store, name, (uint64_t)out * in, linear->weight_f32,
                       error, error_size)) {
        h3_wan22_linear_free(linear);
        return 0;
    }
    if (has_bias) {
        char bias_name[224];
        bias_tensor_name(name, bias_name, sizeof(bias_name));
        linear->bias = malloc((size_t)out * sizeof(float));
        if (!linear->bias ||
            !read_f32_flat(store, bias_name, out, linear->bias,
                           error, error_size)) {
            h3_wan22_linear_free(linear);
            return 0;
        }
    }
    return 1;
}

void h3_wan22_linear_free(h3_wan22_linear *linear) {
    free(linear->weight);
    free(linear->weight_f32);
    free(linear->bias);
    linear->weight = NULL;
    linear->weight_f32 = NULL;
    linear->bias = NULL;
    if (linear->gpu_weight) h3_gpu_tensor_free(linear->gpu_weight);
    if (linear->gpu_bias) h3_gpu_tensor_free(linear->gpu_bias);
    linear->gpu_weight = NULL;
    linear->gpu_bias = NULL;
}

/* out[rows, out] = input[rows, in] @ weight^T + bias.  Weight rows are
 * contiguous in the input dimension, so the inner loop walks the input
 * vector and the weight row together. */
int h3_wan22_linear_forward(const h3_wan22_linear *linear,
                            const float *input, uint32_t rows, float *output) {
    uint32_t in = linear->in;
    uint32_t out = linear->out;
    const float *weight_f32 = linear->weight_f32;
    const uint16_t *weight = linear->weight;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            const float *x = input + (size_t)row64 * in;
            float *y = output + (size_t)row64 * out;
            for (uint32_t o = 0; o < out; o++) {
                float sum = linear->bias ? linear->bias[o] : 0.0f;
                if (weight_f32) {
                    const float *w = weight_f32 + (size_t)o * in;
                    for (uint32_t c = 0; c < in; c++) sum += x[c] * w[c];
                } else {
                    const uint16_t *w = weight + (size_t)o * in;
                    for (uint32_t c = 0; c < in; c++) {
                        sum += x[c] * wan_bf16_to_f32(w[c]);
                    }
                }
                y[o] = sum;
            }
        }
    }
    return 1;
}

/* --- RMSNorm (whole row) ---------------------------------------------------- */

int h3_wan22_rms_load(const h3_weight_store *store, const char *name,
                      uint32_t width, float eps, h3_wan22_rms *rms,
                      char *error, size_t error_size) {
    rms->width = width;
    rms->eps = eps;
    rms->weight = malloc((size_t)width * 2);
    if (!rms->weight) {
        wan_fail(error, error_size, "cannot allocate RMSNorm weight %s", name);
        return 0;
    }
    if (!read_bf16_flat(store, name, width, rms->weight, error, error_size)) {
        h3_wan22_rms_free(rms);
        return 0;
    }
    return 1;
}

void h3_wan22_rms_free(h3_wan22_rms *rms) {
    free(rms->weight);
    rms->weight = NULL;
    if (rms->gpu_weight) h3_gpu_tensor_free(rms->gpu_weight);
    rms->gpu_weight = NULL;
}

int h3_wan22_rms_forward(const h3_wan22_rms *rms, const float *input,
                         uint32_t rows, float *output) {
    float scale = 1.0f / (float)rms->width;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            const float *x = input + (size_t)row64 * rms->width;
            float *y = output + (size_t)row64 * rms->width;
            float sum = 0.0f;
            for (uint32_t c = 0; c < rms->width; c++) sum += x[c] * x[c];
            float inverse = 1.0f / sqrtf(sum * scale + rms->eps);
            for (uint32_t c = 0; c < rms->width; c++) {
                y[c] = x[c] * inverse * wan_bf16_to_f32(rms->weight[c]);
            }
        }
    }
    return 1;
}

/* --- LayerNorm -------------------------------------------------------------- */

int h3_wan22_ln_load(const h3_weight_store *store, const char *name,
                     uint32_t width, float eps, int affine, h3_wan22_ln *ln,
                     char *error, size_t error_size) {
    ln->width = width;
    ln->eps = eps;
    ln->affine = affine;
    ln->weight = NULL;
    ln->bias = NULL;
    if (!affine) return 1;
    char weight_name[224];
    char bias_name[224];
    snprintf(weight_name, sizeof(weight_name), "%s.weight", name);
    snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
    ln->weight = malloc((size_t)width * sizeof(float));
    ln->bias = malloc((size_t)width * sizeof(float));
    if (!ln->weight || !ln->bias ||
        !read_f32_flat(store, weight_name, width, ln->weight, error, error_size) ||
        !read_f32_flat(store, bias_name, width, ln->bias, error, error_size)) {
        h3_wan22_ln_free(ln);
        return 0;
    }
    return 1;
}

void h3_wan22_ln_free(h3_wan22_ln *ln) {
    free(ln->weight);
    free(ln->bias);
    ln->weight = NULL;
    ln->bias = NULL;
    if (ln->gpu_weight) h3_gpu_tensor_free(ln->gpu_weight);
    if (ln->gpu_bias) h3_gpu_tensor_free(ln->gpu_bias);
    ln->gpu_weight = NULL;
    ln->gpu_bias = NULL;
}

int h3_wan22_ln_forward(const h3_wan22_ln *ln, const float *input,
                        uint32_t rows, float *output) {
    float scale = 1.0f / (float)ln->width;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            const float *x = input + (size_t)row64 * ln->width;
            float *y = output + (size_t)row64 * ln->width;
            float mean = 0.0f;
            for (uint32_t c = 0; c < ln->width; c++) mean += x[c];
            mean *= scale;
            float variance = 0.0f;
            for (uint32_t c = 0; c < ln->width; c++) {
                float centered = x[c] - mean;
                variance += centered * centered;
            }
            variance *= scale;
            float inverse = 1.0f / sqrtf(variance + ln->eps);
            for (uint32_t c = 0; c < ln->width; c++) {
                y[c] = (x[c] - mean) * inverse;
            }
            if (ln->affine) {
                for (uint32_t c = 0; c < ln->width; c++) {
                    y[c] = y[c] * ln->weight[c] + ln->bias[c];
                }
            }
        }
    }
    return 1;
}

/* --- RoPE ------------------------------------------------------------------- */

/* RoPE over one [rows, heads, head_dim] tensor: interleaved pairs
 * (x[2i], x[2i+1]) rotated by (cos[i], sin[i]) from the [rows, head_dim/2]
 * tables. */
static void apply_rope(float *tensor, uint32_t rows, uint32_t heads,
                       uint32_t head_dim, const float *cos_table,
                       const float *sin_table) {
    uint32_t half = head_dim / 2;
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            uint32_t row = (uint32_t)row64;
            const float *cos_row = cos_table + (size_t)row * half;
            const float *sin_row = sin_table + (size_t)row * half;
            float *tokens = tensor + (size_t)row * heads * head_dim;
            for (uint32_t head = 0; head < heads; head++) {
                float *x = tokens + (size_t)head * head_dim;
                for (uint32_t pair = 0; pair < half; pair++) {
                    float x0 = x[2 * pair];
                    float x1 = x[2 * pair + 1];
                    float c = cos_row[pair];
                    float s = sin_row[pair];
                    x[2 * pair] = x0 * c - x1 * s;
                    x[2 * pair + 1] = x1 * c + x0 * s;
                }
            }
        }
    }
}

int h3_wan22_rope_tables(const h3_wan22_config *config, uint32_t frames,
                         uint32_t grid_h, uint32_t grid_w,
                         float *cos, float *sin) {
    uint32_t half = config->head_dim / 2;
    uint32_t offset = 0;
    for (int axis = 0; axis < 3; axis++) {
        uint32_t axis_dim = config->rope_axes[axis];
        uint32_t pairs = axis_dim / 2;
        for (uint32_t pair = 0; pair < pairs; pair++) {
            /* freq = theta^(-2p / axis_dim), float32 like
             * posemb_layers.get_1d_rotary_pos_embed. */
            float exponent = -(float)(2 * pair) / (float)axis_dim;
            float frequency = powf(config->rope_theta, exponent);
            uint64_t token = 0;
            for (uint32_t f = 0; f < frames; f++) {
                for (uint32_t h = 0; h < grid_h; h++) {
                    for (uint32_t w = 0; w < grid_w; w++, token++) {
                        float position = axis == 0 ? (float)f
                                        : axis == 1 ? (float)h
                                                    : (float)w;
                        float angle = position * frequency;
                        cos[token * half + offset + pair] = cosf(angle);
                        sin[token * half + offset + pair] = sinf(angle);
                    }
                }
            }
        }
        offset += pairs;
    }
    return offset == half;
}

/* --- SDPA ------------------------------------------------------------------- */

/* Full bidirectional SDPA: q [q_seq, heads, head_dim],
 * k/v [kv_seq, heads, head_dim], output [q_seq, heads, head_dim].
 * scores is a per-head scratch of kv_seq floats (heads slices). */
static void sdpa(const float *q, const float *k, const float *v,
                 uint32_t q_seq, uint32_t kv_seq, uint32_t heads,
                 uint32_t head_dim, float *output, float *scores) {
    float scale = 1.0f / sqrtf((float)head_dim);
    {
        int64_t head64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (head64 = 0; head64 < (int64_t)heads; head64++) {
            uint32_t head = (uint32_t)head64;
            float *head_scores = scores + (size_t)head * kv_seq;
            for (uint32_t i = 0; i < q_seq; i++) {
                const float *q_row = q + ((size_t)i * heads + head) * head_dim;
                float maximum = -INFINITY;
                for (uint32_t j = 0; j < kv_seq; j++) {
                    const float *k_row = k + ((size_t)j * heads + head) * head_dim;
                    float dot = 0.0f;
                    for (uint32_t c = 0; c < head_dim; c++) dot += q_row[c] * k_row[c];
                    dot *= scale;
                    head_scores[j] = dot;
                    if (dot > maximum) maximum = dot;
                }
                float sum = 0.0f;
                for (uint32_t j = 0; j < kv_seq; j++) {
                    head_scores[j] = expf(head_scores[j] - maximum);
                    sum += head_scores[j];
                }
                float inverse = 1.0f / sum;
                float *out_row = output + ((size_t)i * heads + head) * head_dim;
                memset(out_row, 0, head_dim * sizeof(float));
                for (uint32_t j = 0; j < kv_seq; j++) {
                    const float *v_row = v + ((size_t)j * heads + head) * head_dim;
                    float weight = head_scores[j] * inverse;
                    for (uint32_t c = 0; c < head_dim; c++) out_row[c] += weight * v_row[c];
                }
            }
        }
    }
}

/* --- Block ------------------------------------------------------------------ */

struct h3_wan22_block {
    h3_wan22_config config;
    float *modulation;        /* F32 [6 * dim], widened from BF16 parameter */
    struct h3_gpu_tensor *gpu_modulation; /* per-block adaLN table (GPU) */
    h3_wan22_linear q, k, v, o;        /* self_attn */
    h3_wan22_linear cq, ck, cv, co;    /* cross_attn */
    h3_wan22_rms norm_q, norm_k;       /* whole-row QK RMSNorm, self */
    h3_wan22_rms cnorm_q, cnorm_k;     /* whole-row QK RMSNorm, cross */
    h3_wan22_ln norm1, norm2;          /* affine-free LayerNorm */
    h3_wan22_ln norm3;                 /* LayerNorm with affine (F32) */
    h3_wan22_linear ffn0, ffn2;        /* GELU-tanh FFN */
};

h3_wan22_block *h3_wan22_block_load(const h3_weight_store *store,
                                    uint32_t index,
                                    const h3_wan22_config *config,
                                    char *error, size_t error_size) {
    h3_wan22_block *block = calloc(1, sizeof(h3_wan22_block));
    if (!block) {
        wan_fail(error, error_size, "cannot allocate Wan 2.2 block %u", index);
        return NULL;
    }
    block->config = *config;
    char prefix[96];
    snprintf(prefix, sizeof(prefix), "blocks.%u", index);
    char name[224];
#define WAN_NAME(buffer, format, ...) snprintf(buffer, sizeof(buffer), format, __VA_ARGS__)

    /* modulation parameter [1, 6, dim], BF16 -> F32. */
    WAN_NAME(name, "%s.modulation", prefix);
    block->modulation = malloc((size_t)config->dim * 6 * sizeof(float));
    uint16_t *staged = block->modulation ?
                       malloc((size_t)config->dim * 6 * 2) : NULL;
    if (!staged || !read_bf16_flat(store, name, (uint64_t)config->dim * 6,
                                  staged, error, error_size)) {
        free(staged);
        goto failure;
    }
    for (uint32_t element = 0; element < config->dim * 6; element++) {
        block->modulation[element] = wan_bf16_to_f32(staged[element]);
    }
    free(staged);

    WAN_NAME(name, "%s.self_attn.q.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->q, error, error_size)) goto failure;
    WAN_NAME(name, "%s.self_attn.k.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->k, error, error_size)) goto failure;
    WAN_NAME(name, "%s.self_attn.v.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->v, error, error_size)) goto failure;
    WAN_NAME(name, "%s.self_attn.o.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->o, error, error_size)) goto failure;
    WAN_NAME(name, "%s.self_attn.norm_q.weight", prefix);
    if (!h3_wan22_rms_load(store, name, config->dim, config->norm_eps,
                           &block->norm_q, error, error_size)) goto failure;
    WAN_NAME(name, "%s.self_attn.norm_k.weight", prefix);
    if (!h3_wan22_rms_load(store, name, config->dim, config->norm_eps,
                           &block->norm_k, error, error_size)) goto failure;

    WAN_NAME(name, "%s.cross_attn.q.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->cq, error, error_size)) goto failure;
    WAN_NAME(name, "%s.cross_attn.k.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->ck, error, error_size)) goto failure;
    WAN_NAME(name, "%s.cross_attn.v.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->cv, error, error_size)) goto failure;
    WAN_NAME(name, "%s.cross_attn.o.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->dim, 1,
                                     &block->co, error, error_size)) goto failure;
    WAN_NAME(name, "%s.cross_attn.norm_q.weight", prefix);
    if (!h3_wan22_rms_load(store, name, config->dim, config->norm_eps,
                           &block->cnorm_q, error, error_size)) goto failure;
    WAN_NAME(name, "%s.cross_attn.norm_k.weight", prefix);
    if (!h3_wan22_rms_load(store, name, config->dim, config->norm_eps,
                           &block->cnorm_k, error, error_size)) goto failure;

    WAN_NAME(name, "%s.norm1", prefix);
    if (!h3_wan22_ln_load(store, name, config->dim, config->norm_eps, 0,
                          &block->norm1, error, error_size)) goto failure;
    WAN_NAME(name, "%s.norm2", prefix);
    if (!h3_wan22_ln_load(store, name, config->dim, config->norm_eps, 0,
                          &block->norm2, error, error_size)) goto failure;
    WAN_NAME(name, "%s.norm3", prefix);
    if (!h3_wan22_ln_load(store, name, config->dim, config->norm_eps, 1,
                          &block->norm3, error, error_size)) goto failure;

    WAN_NAME(name, "%s.ffn.0.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->ffn_dim, config->dim, 1,
                                     &block->ffn0, error, error_size)) goto failure;
    WAN_NAME(name, "%s.ffn.2.weight", prefix);
    if (!h3_wan22_linear_load_quanto(store, name, config->dim, config->ffn_dim, 1,
                                     &block->ffn2, error, error_size)) goto failure;
#undef WAN_NAME
    return block;

failure:
    h3_wan22_block_free(block);
    return NULL;
}

void h3_wan22_block_free(h3_wan22_block *block) {
    if (!block) return;
    free(block->modulation);
    h3_gpu_tensor_free(block->gpu_modulation);
    h3_wan22_linear_free(&block->q);
    h3_wan22_linear_free(&block->k);
    h3_wan22_linear_free(&block->v);
    h3_wan22_linear_free(&block->o);
    h3_wan22_linear_free(&block->cq);
    h3_wan22_linear_free(&block->ck);
    h3_wan22_linear_free(&block->cv);
    h3_wan22_linear_free(&block->co);
    h3_wan22_rms_free(&block->norm_q);
    h3_wan22_rms_free(&block->norm_k);
    h3_wan22_rms_free(&block->cnorm_q);
    h3_wan22_rms_free(&block->cnorm_k);
    h3_wan22_ln_free(&block->norm1);
    h3_wan22_ln_free(&block->norm2);
    h3_wan22_ln_free(&block->norm3);
    h3_wan22_linear_free(&block->ffn0);
    h3_wan22_linear_free(&block->ffn2);
    free(block);
}

/* --- block-internal stage dumps -----------------------------------------
 * WAN22_DUMP_STAGE=1 plus WAN22_DUMP_BLOCK_PATH=<prefix> writes the MSA
 * stages of the FIRST block of the first forward as F32 [seq][dim]:
 *   stage_normed  after norm1 + adaLN (the input every linear sees)
 *   stage_q       q after the whole-row QK RMSNorm and RoPE
 *   stage_attn    the self-attention output
 *   stage_branch  the o-projection output, i.e. what the gate multiplies
 * The CPU path writes the same four, so diffing GPU against CPU names the
 * stage that is wrong rather than the block that is wrong.  Only the first
 * block is dumped on purpose: 4 x 85 MB per block would be 10 GB a run. */
static int wan22_stage_dump_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("WAN22_DUMP_STAGE");
        enabled = (value && *value && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled;
}

static void wan22_dump_stage_f32(const char *name, const float *plane,
                                 size_t count) {
    if (!wan22_stage_dump_enabled() || !plane) return;
    const char *prefix = getenv("WAN22_DUMP_BLOCK_PATH");
    if (!prefix || !*prefix) return;
    char path[512];
    snprintf(path, sizeof(path), "%s.stage_%s.bin", prefix, name);
    FILE *out = fopen(path, "wb");
    if (!out) return;
    fwrite(plane, sizeof(float), count, out);
    fclose(out);
    fprintf(stderr, "h3 wan22: stage %s -> %s\n", name, path);
}

static void wan22_dump_stage_bf16(const char *name, h3_gpu_tensor *tensor,
                                  size_t count) {
    if (!wan22_stage_dump_enabled() || !tensor) return;
    float *host = malloc(count * sizeof(float));
    uint16_t *raw = malloc(count * sizeof(uint16_t));
    if (!host || !raw) {
        free(host);
        free(raw);
        return;
    }
    if (h3_gpu_tensor_read_bf16(tensor, raw, count)) {
        for (size_t i = 0; i < count; i++) {
            const uint32_t bits = (uint32_t)raw[i] << 16u;
            float value;
            memcpy(&value, &bits, sizeof(value));
            host[i] = value;
        }
        wan22_dump_stage_f32(name, host, count);
    }
    free(host);
    free(raw);
}

/* Only the first block of the first forward: the point is to see whether the
 * divergence starts at the norm, the projection, the attention or the gate,
 * not to trace every block. */
static int wan22_stage_dump_this_block(void) {
    static int blocks_seen;
    if (!wan22_stage_dump_enabled()) return 0;
    return blocks_seen++ == 0;
}

int h3_wan22_block_forward(const h3_wan22_block *block, float *x,
                           uint32_t seq, const float *e0,
                           const float *context, uint32_t ctx_seq,
                           const float *rope_cos, const float *rope_sin,
                           char *error, size_t error_size) {
    const h3_wan22_config *config = &block->config;
    uint32_t dim = config->dim;
    uint32_t ffn_dim = config->ffn_dim;
    uint32_t heads = config->heads;
    uint32_t head_dim = config->head_dim;

    if (!e0 || !context || !ctx_seq) {
        wan_fail(error, error_size,
                 "Wan 2.2 block forward needs e0 and a context");
        return 0;
    }

    /* e = (modulation + e0).chunk(6): six [dim] chunks
     * shift_msa / scale_msa / gate_msa / shift_mlp / scale_mlp / gate_mlp. */
    float *mod = malloc((size_t)dim * 6 * sizeof(float));
    if (!mod) {
        wan_fail(error, error_size, "cannot allocate modulation buffer");
        return 0;
    }
    for (uint32_t chunk = 0; chunk < 6; chunk++) {
        for (uint32_t c = 0; c < dim; c++) {
            mod[(size_t)chunk * dim + c] =
                block->modulation[(size_t)chunk * dim + c] +
                e0[(size_t)chunk * dim + c];
        }
    }
    const float *shift_msa = mod;
    const float *scale_msa = mod + (size_t)dim;
    const float *gate_msa = mod + 2 * (size_t)dim;
    const float *shift_mlp = mod + 3 * (size_t)dim;
    const float *scale_mlp = mod + 4 * (size_t)dim;
    const float *gate_mlp = mod + 5 * (size_t)dim;

    size_t plane = (size_t)seq * dim;
    size_t ctx_plane = (size_t)ctx_seq * dim;
    float *normed = malloc(plane * sizeof(float));
    float *q = malloc(plane * sizeof(float));
    float *k = malloc(plane * sizeof(float));
    float *v = malloc(plane * sizeof(float));
    float *attn = malloc(plane * sizeof(float));
    float *branch = malloc(plane * sizeof(float));
    float *ck = malloc(ctx_plane * sizeof(float));
    float *cv = malloc(ctx_plane * sizeof(float));
    uint32_t scores_len = seq > ctx_seq ? seq : ctx_seq;
    float *scores = malloc((size_t)heads * scores_len * sizeof(float));
    float *ffn_a = malloc((size_t)seq * ffn_dim * sizeof(float));
    if (!normed || !q || !k || !v || !attn || !branch || !ck || !cv ||
        !scores || !ffn_a) {
        wan_fail(error, error_size, "cannot allocate Wan 2.2 block buffers");
        free(mod); free(normed); free(q); free(k); free(v); free(attn);
        free(branch); free(ck); free(cv); free(scores); free(ffn_a);
        return 0;
    }

    /* --- self-attention ---------------------------------------------------- */
    h3_wan22_ln_forward(&block->norm1, x, seq, normed);
    for (size_t index = 0; index < plane; index++) {
        normed[index] = normed[index] * (1.0f + scale_msa[index % dim]) +
                        shift_msa[index % dim];
    }
    h3_wan22_linear_forward(&block->q, normed, seq, q);
    h3_wan22_linear_forward(&block->k, normed, seq, k);
    h3_wan22_linear_forward(&block->v, normed, seq, v);
    const int dump_stages = wan22_stage_dump_this_block();
    if (dump_stages) wan22_dump_stage_f32("normed", normed, plane);
    /* Whole-row QK RMSNorm (over all 3072 dims), then RoPE. */
    h3_wan22_rms_forward(&block->norm_q, q, seq, q);
    h3_wan22_rms_forward(&block->norm_k, k, seq, k);
    apply_rope(q, seq, heads, head_dim, rope_cos, rope_sin);
    apply_rope(k, seq, heads, head_dim, rope_cos, rope_sin);
    if (dump_stages) wan22_dump_stage_f32("q", q, plane);
    /* k and v are what separates "the scores are wrong" from "the kernel that
     * consumes the scores is wrong": q already matches the device one up to
     * the rope column order, so a remaining attention gap has to come from
     * one of these two. */
    if (dump_stages) {
        wan22_dump_stage_f32("k", k, plane);
        wan22_dump_stage_f32("v", v, plane);
    }
    sdpa(q, k, v, seq, seq, heads, head_dim, attn, scores);
    if (dump_stages) wan22_dump_stage_f32("attn", attn, plane);
    h3_wan22_linear_forward(&block->o, attn, seq, branch);
    if (dump_stages) wan22_dump_stage_f32("branch", branch, plane);
    /* x += y * gate_msa (addcmul semantics). */
    for (size_t index = 0; index < plane; index++) {
        x[index] += branch[index] * gate_msa[index % dim];
    }

    /* --- cross-attention (text, no RoPE, no modulation) -------------------- */
    h3_wan22_ln_forward(&block->norm3, x, seq, normed);
    h3_wan22_linear_forward(&block->cq, normed, seq, q);
    h3_wan22_rms_forward(&block->cnorm_q, q, seq, q);
    h3_wan22_linear_forward(&block->ck, context, ctx_seq, ck);
    h3_wan22_rms_forward(&block->cnorm_k, ck, ctx_seq, ck);
    h3_wan22_linear_forward(&block->cv, context, ctx_seq, cv);
    sdpa(q, ck, cv, seq, ctx_seq, heads, head_dim, attn, scores);
    h3_wan22_linear_forward(&block->co, attn, seq, branch);
    if (dump_stages) wan22_dump_stage_f32("xattn", branch, plane);
    for (size_t index = 0; index < plane; index++) x[index] += branch[index];
    if (dump_stages) wan22_dump_stage_f32("x2", x, plane);

    /* --- FFN (GELU tanh) ---------------------------------------------------- */
    h3_wan22_ln_forward(&block->norm2, x, seq, normed);
    for (size_t index = 0; index < plane; index++) {
        normed[index] = normed[index] * (1.0f + scale_mlp[index % dim]) +
                        shift_mlp[index % dim];
    }
    h3_wan22_linear_forward(&block->ffn0, normed, seq, ffn_a);
    /* GELU(approximate="tanh"): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). */
    for (size_t index = 0; index < (size_t)seq * ffn_dim; index++) {
        float value = ffn_a[index];
        float inner = 0.7978845608028654f *
                      (value + 0.044715f * value * value * value);
        ffn_a[index] = 0.5f * value * (1.0f + tanhf(inner));
    }
    h3_wan22_linear_forward(&block->ffn2, ffn_a, seq, branch);
    if (dump_stages) wan22_dump_stage_f32("ffn", branch, plane);
    for (size_t index = 0; index < plane; index++) {
        x[index] += branch[index] * gate_mlp[index % dim];
    }

    free(mod);
    free(normed);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(branch);
    free(ck);
    free(cv);
    free(scores);
    free(ffn_a);
    return 1;
}

/* --- timestep chain --------------------------------------------------------- */

void h3_wan22_timestep_embedding(float t, uint32_t dim, float *out) {
    uint32_t half = dim / 2;
    /* sinusoid = outer(t, 10000^(-arange(half)/half)); [cos | sin]. */
    for (uint32_t index = 0; index < half; index++) {
        float exponent = -(float)index / (float)half;
        float argument = t * powf(10000.0f, exponent);
        out[index] = cosf(argument);
        out[half + index] = sinf(argument);
    }
}

int h3_wan22_time_embedder_load(const h3_weight_store *store,
                                const h3_wan22_config *config,
                                h3_wan22_time_embedder *embedder,
                                char *error, size_t error_size) {
    if (!h3_wan22_linear_load_f32(store, "time_embedding.0.weight",
                                  config->dim, config->freq_dim, 1,
                                  &embedder->fc0, error, error_size)) {
        return 0;
    }
    if (!h3_wan22_linear_load_f32(store, "time_embedding.2.weight",
                                  config->dim, config->dim, 1,
                                  &embedder->fc1, error, error_size)) {
        h3_wan22_time_embedder_free(embedder);
        return 0;
    }
    return 1;
}

void h3_wan22_time_embedder_free(h3_wan22_time_embedder *embedder) {
    h3_wan22_linear_free(&embedder->fc0);
    h3_wan22_linear_free(&embedder->fc1);
}

int h3_wan22_time_embedder_forward(const h3_wan22_time_embedder *embedder,
                                   const float *input, uint32_t rows,
                                   float *output) {
    uint32_t dim = embedder->fc1.in;
    float *hidden = malloc((size_t)rows * dim * sizeof(float));
    if (!hidden) return 0;
    h3_wan22_linear_forward(&embedder->fc0, input, rows, hidden);
    for (size_t index = 0; index < (size_t)rows * dim; index++) {
        float value = hidden[index];
        hidden[index] = value / (1.0f + expf(-value)); /* SiLU */
    }
    h3_wan22_linear_forward(&embedder->fc1, hidden, rows, output);
    free(hidden);
    return 1;
}

/* --- head -------------------------------------------------------------------- */

int h3_wan22_head_load(const h3_weight_store *store,
                       const h3_wan22_config *config, uint32_t out,
                       h3_wan22_head *head, char *error, size_t error_size) {
    head->dim = config->dim;
    head->out = out;
    head->norm_eps = config->norm_eps;
    head->shift_scale = NULL;
    head->shift_scale = malloc((size_t)config->dim * 2 * sizeof(float));
    if (!head->shift_scale ||
        !read_f32_flat(store, "head.modulation", (uint64_t)config->dim * 2,
                       head->shift_scale, error, error_size)) {
        h3_wan22_head_free(head);
        return 0;
    }
    if (!h3_wan22_linear_load_f32(store, "head.head.weight", out, config->dim, 1,
                                  &head->out_proj, error, error_size)) {
        h3_wan22_head_free(head);
        return 0;
    }
    return 1;
}

void h3_wan22_head_free(h3_wan22_head *head) {
    free(head->shift_scale);
    head->shift_scale = NULL;
    h3_wan22_linear_free(&head->out_proj);
}

/* e is the time_embedding output [rows, dim] (broadcast per row when a
 * single timestep): shift = modulation[0] + e, scale = modulation[1] + e. */
int h3_wan22_head_forward(const h3_wan22_head *head, const float *x,
                          uint32_t rows, const float *e, float *output) {
    uint32_t dim = head->dim;
    const float *shift = head->shift_scale;
    const float *scale = head->shift_scale + (size_t)dim;
    float *normed = malloc((size_t)rows * dim * sizeof(float));
    if (!normed) return 0;
    h3_wan22_ln ln = { dim, head->norm_eps, 0, NULL, NULL };
    h3_wan22_ln_forward(&ln, x, rows, normed);
    /* shift/scale chunks each get the shared e vector added (batch 1). */
    {
        int64_t row64;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row64 = 0; row64 < (int64_t)rows; row64++) {
            float *y = normed + (size_t)row64 * dim;
            for (uint32_t c = 0; c < dim; c++) {
                y[c] = y[c] * (1.0f + scale[c] + e[c]) + shift[c] + e[c];
            }
        }
    }
    h3_wan22_linear_forward(&head->out_proj, normed, rows, output);
    free(normed);
    return 1;
}

/* ------------------------------------------------------------------ */
/* Optional GPU forward path (WAN22_GPU=1).                            */
/*                                                                     */
/* The 30 blocks run as BF16 kernels with weights resident on the       */
/* device; the patch embedding, timestep chain and Head stay on the     */
/* host in F32 (same numbers as the CPU path).  Differences vs the      */
/* Z-Image GPU path:                                                    */
/*   - norm1/norm2 are affine-free LAYER norms (mean/variance), fused   */
/*     with the adaLN through h3_gpu_layer_norm_adaln_bf16 (the stock   */
/*     h3_gpu_adaln_bf16 fuses an RMSNorm);                             */
/*   - the QK RMSNorm spans the whole 3072-wide row (permutation        */
/*     invariant, so it composes with the RoPE weight permutation);      */
/*   - cross_attn attends over a separate-length context through        */
/*     h3_gpu_sdpa_flash_bf16 and has no modulation (norm3 is an        */
/*     affine LayerNorm, no shift/scale/gate).                          */
/* ------------------------------------------------------------------ */

#define WAN22_MOD_SLOTS 6
#define WAN22_MOD_SHIFT_MSA 0
#define WAN22_MOD_SCALE_MSA 1
#define WAN22_MOD_GATE_MSA 2
#define WAN22_MOD_SHIFT_MLP 3
#define WAN22_MOD_SCALE_MLP 4
#define WAN22_MOD_GATE_MLP 5
#define WAN22_MOD_GATE_ADD 6 /* ones row: plain residual add */

struct h3_wan22_gpu {
    h3_gpu *gpu;
    h3_wan22_config config;
    uint32_t seq;     /* reserved latent-token rows */
    uint32_t ctx_seq; /* reserved context rows */
    h3_gpu_tensor *x;         /* BF16 [seq, dim] block-chain carrier */
    h3_gpu_tensor *x_f32;     /* F32 residual carrier (block chain) */
    h3_gpu_tensor *normed;    /* BF16 [seq, dim] */
    h3_gpu_tensor *q;         /* BF16 [seq, dim] */
    h3_gpu_tensor *k;         /* BF16 [seq, dim] */
    h3_gpu_tensor *v;         /* BF16 [seq, dim] */
    h3_gpu_tensor *attn;      /* BF16 [seq, dim] */
    h3_gpu_tensor *branch;    /* BF16 [seq, dim] */
    h3_gpu_tensor *ffn_a;     /* BF16 [seq, ffn_dim] */
    h3_gpu_tensor *ctx;       /* BF16 [ctx_seq, dim] embedded text */
    h3_gpu_tensor *ck;        /* BF16 [ctx_seq, dim] */
    h3_gpu_tensor *cv;        /* BF16 [ctx_seq, dim] */
    h3_gpu_tensor *rope_cos;  /* F32 [seq, head_dim / 2] */
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *row_map;   /* U32 [seq], all zero */
    h3_gpu_tensor *modulation; /* BF16 [WAN22_MOD_SLOTS, dim] */
    h3_gpu_tensor *norm_ones;  /* BF16 [dim] */
    h3_gpu_tensor *norm_zeros; /* BF16 [dim] */
    /* Set by h3_wan22_gpu_state_release: the weights and scratch above are
     * gone from the device and the context refuses to run until a resume
     * rebuilds them from the host BF16 copies. */
    int released;
};

h3_wan22_gpu *h3_wan22_gpu_create(const h3_wan22_config *config,
                                  char *error, size_t error_size) {
    h3_wan22_gpu *wgu = calloc(1, sizeof(*wgu));
    if (!wgu) {
        snprintf(error, error_size, "cannot allocate Wan 2.2 GPU context");
        return NULL;
    }
    wgu->config = *config;
    wgu->gpu = h3_gpu_create(NULL, error, error_size);
    if (!wgu->gpu) {
        free(wgu);
        return NULL;
    }
    uint32_t dim = config->dim;
    uint16_t *ones = malloc((size_t)dim * 2);
    uint16_t *zeros = malloc((size_t)dim * 2);
    if (!ones || !zeros) {
        free(ones);
        free(zeros);
        h3_wan22_gpu_free(wgu);
        snprintf(error, error_size, "cannot stage Wan 2.2 norm constants");
        return NULL;
    }
    for (uint32_t c = 0; c < dim; c++) {
        ones[c] = wan_bf16_bits(1.0f);
        zeros[c] = wan_bf16_bits(0.0f);
    }
    wgu->norm_ones = h3_gpu_tensor_from_bf16(wgu->gpu, ones, dim);
    wgu->norm_zeros = h3_gpu_tensor_from_bf16(wgu->gpu, zeros, dim);
    wgu->modulation =
        h3_gpu_tensor_new_bf16(wgu->gpu, (size_t)(WAN22_MOD_SLOTS + 1) * dim);
    free(ones);
    free(zeros);
    if (!wgu->norm_ones || !wgu->norm_zeros || !wgu->modulation) {
        snprintf(error, error_size,
                 "cannot allocate Wan 2.2 GPU norm tensors: %s",
                 h3_gpu_error(wgu->gpu));
        h3_wan22_gpu_free(wgu);
        return NULL;
    }
    return wgu;
}

void h3_wan22_gpu_free(h3_wan22_gpu *wgu) {
    if (!wgu) return;
    h3_gpu_tensor_free(wgu->x);
    h3_gpu_tensor_free(wgu->x_f32);
    h3_gpu_tensor_free(wgu->normed);
    h3_gpu_tensor_free(wgu->q);
    h3_gpu_tensor_free(wgu->k);
    h3_gpu_tensor_free(wgu->v);
    h3_gpu_tensor_free(wgu->attn);
    h3_gpu_tensor_free(wgu->branch);
    h3_gpu_tensor_free(wgu->ffn_a);
    h3_gpu_tensor_free(wgu->ctx);
    h3_gpu_tensor_free(wgu->ck);
    h3_gpu_tensor_free(wgu->cv);
    h3_gpu_tensor_free(wgu->rope_cos);
    h3_gpu_tensor_free(wgu->rope_sin);
    h3_gpu_tensor_free(wgu->row_map);
    h3_gpu_tensor_free(wgu->modulation);
    h3_gpu_tensor_free(wgu->norm_ones);
    h3_gpu_tensor_free(wgu->norm_zeros);
    if (wgu->gpu) h3_gpu_free(wgu->gpu);
    free(wgu);
}

int h3_wan22_gpu_reserve(h3_wan22_gpu *wgu, uint32_t seq, uint32_t ctx_seq,
                         char *error, size_t error_size) {
    if (seq <= wgu->seq && ctx_seq <= wgu->ctx_seq) return 1;
    /* Trace: only the growth path costs anything; a reserve that fits the
     * current geometry is the early return above. */
    const int trace = wan22_dit_trace();
    h3_gpu_mem_stats pool_before = {0};
    if (trace) h3_gpu_mem_query(wgu->gpu, &pool_before);
    const double trace_started = trace ? wan22_dit_now() : 0.0;
    if (seq > wgu->seq) {
        h3_gpu_tensor_free(wgu->x);
        h3_gpu_tensor_free(wgu->x_f32);
        h3_gpu_tensor_free(wgu->normed);
        h3_gpu_tensor_free(wgu->q);
        h3_gpu_tensor_free(wgu->k);
        h3_gpu_tensor_free(wgu->v);
        h3_gpu_tensor_free(wgu->attn);
        h3_gpu_tensor_free(wgu->branch);
        h3_gpu_tensor_free(wgu->ffn_a);
        h3_gpu_tensor_free(wgu->rope_cos);
        h3_gpu_tensor_free(wgu->rope_sin);
        h3_gpu_tensor_free(wgu->row_map);
        size_t plane = (size_t)seq * wgu->config.dim;
        size_t wide = (size_t)seq * wgu->config.ffn_dim;
        size_t half = wgu->config.head_dim / 2;
        wgu->x = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->normed = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->x_f32 = h3_gpu_tensor_new_f32(wgu->gpu, plane);
        wgu->q = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->k = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->v = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->attn = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->branch = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->ffn_a = h3_gpu_tensor_new_bf16(wgu->gpu, wide);
        wgu->rope_cos = h3_gpu_tensor_new_f32(wgu->gpu, (size_t)seq * half);
        wgu->rope_sin = h3_gpu_tensor_new_f32(wgu->gpu, (size_t)seq * half);
        uint32_t *map = calloc(seq, sizeof(uint32_t));
        wgu->row_map = map ? h3_gpu_tensor_from_u32(wgu->gpu, map, seq) : NULL;
        free(map);

        /* No [heads, seq, max(seq,ctx)] F32 scores scratch: both attention
         * calls now use the flash SDPA (online softmax), so the largest
         * scratch here scales linearly with seq. The old scores tensor made
         * the 720p reserve need ~71.6 GiB (24*27280*27280*4) and fail. */
        if (!wgu->x || !wgu->x_f32 || !wgu->normed || !wgu->q || !wgu->k ||
            !wgu->v || !wgu->attn || !wgu->branch || !wgu->ffn_a ||
            !wgu->rope_cos || !wgu->rope_sin || !wgu->row_map) {
            snprintf(error, error_size,
                     "cannot allocate Wan 2.2 GPU scratch: %s",
                     h3_gpu_error(wgu->gpu));
            return 0;
        }
        wgu->seq = seq;
    }
    if (ctx_seq > wgu->ctx_seq) {
        h3_gpu_tensor_free(wgu->ctx);
        h3_gpu_tensor_free(wgu->ck);
        h3_gpu_tensor_free(wgu->cv);
        size_t plane = (size_t)ctx_seq * wgu->config.dim;
        wgu->ctx = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->ck = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        wgu->cv = h3_gpu_tensor_new_bf16(wgu->gpu, plane);
        if (!wgu->ctx || !wgu->ck || !wgu->cv) {
            snprintf(error, error_size,
                     "cannot allocate Wan 2.2 GPU context scratch: %s",
                     h3_gpu_error(wgu->gpu));
            return 0;
        }
        wgu->ctx_seq = ctx_seq;
    }
    if (trace) {
        h3_gpu_mem_stats pool_after = {0};
        h3_gpu_mem_query(wgu->gpu, &pool_after);
        fprintf(stderr,
                "wan22: reserve grew to seq=%u ctx=%u in %.3fs "
                "mallocs=%llu pool used %.2f->%.2f GiB reserved %.2f->%.2f "
                "GiB\n",
                wgu->seq, wgu->ctx_seq, wan22_dit_now() - trace_started,
                (unsigned long long)(pool_after.mallocs - pool_before.mallocs),
                (double)pool_before.used_current / (double)(1u << 30),
                (double)pool_after.used_current / (double)(1u << 30),
                (double)pool_before.reserved_current / (double)(1u << 30),
                (double)pool_after.reserved_current / (double)(1u << 30));
        fflush(stderr);
    }
    return 1;
}

h3_gpu_tensor *h3_wan22_gpu_modulation(h3_wan22_gpu *wgu) {
    return wgu->modulation;
}
h3_gpu_tensor *h3_wan22_gpu_rope_cos(h3_wan22_gpu *wgu) {
    return wgu->rope_cos;
}
h3_gpu_tensor *h3_wan22_gpu_rope_sin(h3_wan22_gpu *wgu) {
    return wgu->rope_sin;
}
h3_gpu_tensor *h3_wan22_gpu_x(h3_wan22_gpu *wgu) { return wgu->x_f32; }
h3_gpu_tensor *h3_wan22_gpu_row_map(h3_wan22_gpu *wgu) {
    return wgu->row_map;
}
h3_gpu_tensor *h3_wan22_gpu_normed(h3_wan22_gpu *wgu) {
    return wgu->normed;
}
h3_gpu_tensor *h3_wan22_gpu_q(h3_wan22_gpu *wgu) { return wgu->q; }
h3_gpu_tensor *h3_wan22_gpu_k(h3_wan22_gpu *wgu) { return wgu->k; }
h3_gpu_tensor *h3_wan22_gpu_v(h3_wan22_gpu *wgu) { return wgu->v; }
h3_gpu_tensor *h3_wan22_gpu_attn(h3_wan22_gpu *wgu) { return wgu->attn; }

h3_gpu *h3_wan22_gpu_handle(h3_wan22_gpu *wgu) {
    return wgu->gpu;
}

h3_gpu_tensor *h3_wan22_gpu_upload_f32_bf16(h3_wan22_gpu *wgu,
                                            const float *values,
                                            size_t count) {
    uint16_t *staged = malloc(count * 2);
    if (!staged) return NULL;
    for (size_t i = 0; i < count; i++) staged[i] = wan_bf16_bits(values[i]);
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(wgu->gpu, staged, count);
    free(staged);
    return tensor;
}

int h3_wan22_gpu_read_f32(h3_wan22_gpu *wgu, h3_gpu_tensor *tensor,
                          float *out, size_t count) {
    (void)wgu;
    if (h3_gpu_tensor_read_f32(tensor, out, count)) {
        return 1;
    }
    uint16_t *staged = malloc(count * 2);
    if (!staged) return 0;
    if (!h3_gpu_tensor_read_bf16(tensor, staged, count)) {
        free(staged);
        return 0;
    }
    for (size_t i = 0; i < count; i++) {
        out[i] = wan_bf16_to_f32(staged[i]);
    }
    free(staged);
    return 1;
}

/* Wan's bf16 widening helper (bits) reused from the CPU path. */
/* Allocate a BF16 tensor and fill it through the pinned staging path.
 * h3_gpu_tensor_from_bf16 copies from pageable host memory, which runs at
 * ~1.8 GB/s on this box; the DiT resume re-uploads ~9.8 GiB of weights
 * through it, so the pinned path (a raw byte copy -- identical bits, the
 * release/resume round trip stays bit-exact) takes the re-upload to PCIe
 * speed. */
static h3_gpu_tensor *wan22_tensor_upload_bf16(h3_gpu *gpu,
                                                const uint16_t *values,
                                                size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!tensor) return NULL;
    if (!h3_gpu_upload_bf16_pinned(gpu, tensor, values, elements)) {
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

/* Upload a Linear; rope_permute applies the even-dimensions-first row
 * permutation per head so the stock half-split RoPE kernel reproduces
 * Wan's interleaved (2i, 2i+1) pairing. */
static uint32_t wan22_rope_source_dim(uint32_t d, uint32_t head_dim) {
    uint32_t half = head_dim / 2;
    return d < half ? 2 * d : 2 * (d - half) + 1;
}

/* Row permutation used by the RoPE weight layout: device row d reads source
 * row perm(d).  Every tensor indexed by the *output* dimension of the q/k
 * projection has to be permuted with the matrix, otherwise the device applies
 * the wrong per-dimension scale to the right value: the permutation is only
 * transparent for the projection itself. */
static uint16_t *wan22_rope_permute_rows(const uint16_t *source, uint32_t out,
                                         uint32_t in, uint32_t head_dim) {
    uint16_t *staged = malloc((size_t)out * in * 2);
    if (!staged) return NULL;
    uint32_t heads = out / head_dim;
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            uint32_t from = wan22_rope_source_dim(d, head_dim);
            memcpy(staged + ((size_t)head * head_dim + d) * in,
                   source + ((size_t)head * head_dim + from) * in,
                   (size_t)in * 2);
        }
    }
    return staged;
}

static int wan22_linear_upload_gpu(h3_wan22_linear *linear, h3_wan22_gpu *wgu,
                                   int rope_permute, char *error,
                                   size_t error_size) {
    uint32_t out = linear->out;
    uint32_t in = linear->in;
    uint32_t head_dim = wgu->config.head_dim;
    const uint16_t *source = linear->weight;
    uint16_t *staged = NULL;
    if (rope_permute) {
        staged = wan22_rope_permute_rows(linear->weight, out, in, head_dim);
        if (!staged) {
            snprintf(error, error_size, "cannot stage permuted Q/K weight");
            return 0;
        }
        source = staged;
    }
    if (linear->gpu_weight) h3_gpu_tensor_free(linear->gpu_weight);
    linear->gpu_weight = wan22_tensor_upload_bf16(wgu->gpu, source,
                                                  (size_t)out * in);
    free(staged);
    if (!linear->gpu_weight) {
        snprintf(error, error_size, "cannot upload %s: %s", "linear weight",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    if (linear->bias) {
        uint16_t *bias = malloc((size_t)out * 2);
        if (!bias) {
            snprintf(error, error_size, "cannot stage linear bias");
            return 0;
        }
        for (uint32_t c = 0; c < out; c++) {
            bias[c] = wan_bf16_bits(linear->bias[c]);
        }
        if (rope_permute) {
            /* Same permutation as the matrix: device output row d is source
             * row perm(d), so its bias has to be source row perm(d) too. */
            uint16_t *permuted =
                wan22_rope_permute_rows(bias, out, 1, head_dim);
            if (!permuted) {
                free(bias);
                snprintf(error, error_size, "cannot stage permuted Q/K bias");
                return 0;
            }
            memcpy(bias, permuted, (size_t)out * 2);
            free(permuted);
        }
        if (linear->gpu_bias) h3_gpu_tensor_free(linear->gpu_bias);
        linear->gpu_bias = wan22_tensor_upload_bf16(wgu->gpu, bias, out);
        free(bias);
        if (!linear->gpu_bias) {
            snprintf(error, error_size, "cannot upload linear bias: %s",
                     h3_gpu_error(wgu->gpu));
            return 0;
        }
    }
    return 1;
}

static int wan22_rms_upload_gpu(h3_wan22_rms *rms, h3_wan22_gpu *wgu,
                                int rope_permute, char *error,
                                size_t error_size) {
    uint16_t *staged = NULL;
    const uint16_t *source = rms->weight;
    if (rope_permute) {
        /* The QK RMSNorm scales each dimension, so it is NOT permutation
         * invariant even though its row norm is: device dim d holds source
         * dim perm(d) and has to be scaled by that source dim's weight. */
        staged = wan22_rope_permute_rows(rms->weight, rms->width, 1,
                                         wgu->config.head_dim);
        if (!staged) {
            snprintf(error, error_size, "cannot stage permuted rms weight");
            return 0;
        }
        source = staged;
    }
    if (rms->gpu_weight) h3_gpu_tensor_free(rms->gpu_weight);
    rms->gpu_weight = wan22_tensor_upload_bf16(wgu->gpu, source, rms->width);
    free(staged);
    if (!rms->gpu_weight) {
        snprintf(error, error_size, "cannot upload rms weight: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    return 1;
}

static int wan22_ln_upload_gpu(h3_wan22_ln *ln, h3_wan22_gpu *wgu,
                               char *error, size_t error_size) {
    uint16_t *weight = malloc((size_t)ln->width * 2);
    uint16_t *bias = malloc((size_t)ln->width * 2);
    if (!weight || !bias) {
        free(weight);
        free(bias);
        snprintf(error, error_size, "cannot stage layernorm params");
        return 0;
    }
    for (uint32_t c = 0; c < ln->width; c++) {
        weight[c] = wan_bf16_bits(ln->weight ? ln->weight[c] : 1.0f);
        bias[c] = wan_bf16_bits(ln->bias ? ln->bias[c] : 0.0f);
    }
    if (ln->gpu_weight) h3_gpu_tensor_free(ln->gpu_weight);
    if (ln->gpu_bias) h3_gpu_tensor_free(ln->gpu_bias);
    ln->gpu_weight = wan22_tensor_upload_bf16(wgu->gpu, weight, ln->width);
    ln->gpu_bias = wan22_tensor_upload_bf16(wgu->gpu, bias, ln->width);
    free(weight);
    free(bias);
    if (!ln->gpu_weight || !ln->gpu_bias) {
        snprintf(error, error_size, "cannot upload layernorm params: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    return 1;
}

h3_gpu_tensor *h3_wan22_block_gpu_norm_q(h3_wan22_block *block) {
    return block->norm_q.gpu_weight;
}
h3_gpu_tensor *h3_wan22_block_gpu_norm_k(h3_wan22_block *block) {
    return block->norm_k.gpu_weight;
}
int h3_wan22_block_upload_gpu(h3_wan22_block *block, h3_wan22_gpu *wgu,
                              char *error, size_t error_size) {
    /* Self-attn: q/k permuted for the half-split RoPE kernel. */
    if (!wan22_linear_upload_gpu(&block->q, wgu, 1, error, error_size) ||
        !wan22_linear_upload_gpu(&block->k, wgu, 1, error, error_size) ||
        !wan22_linear_upload_gpu(&block->v, wgu, 0, error, error_size) ||
        !wan22_linear_upload_gpu(&block->o, wgu, 0, error, error_size) ||
        !wan22_rms_upload_gpu(&block->norm_q, wgu, 1, error, error_size) ||
        !wan22_rms_upload_gpu(&block->norm_k, wgu, 1, error, error_size) ||
        /* Cross-attn: no RoPE, no permutation. */
        !wan22_linear_upload_gpu(&block->cq, wgu, 0, error, error_size) ||
        !wan22_linear_upload_gpu(&block->ck, wgu, 0, error, error_size) ||
        !wan22_linear_upload_gpu(&block->cv, wgu, 0, error, error_size) ||
        !wan22_linear_upload_gpu(&block->co, wgu, 0, error, error_size) ||
        !wan22_rms_upload_gpu(&block->cnorm_q, wgu, 0, error, error_size) ||
        !wan22_rms_upload_gpu(&block->cnorm_k, wgu, 0, error, error_size) ||
        /* norm1/norm2 affine-free LN (ones/zeros staged), norm3 affine. */
        !wan22_ln_upload_gpu(&block->norm1, wgu, error, error_size) ||
        !wan22_ln_upload_gpu(&block->norm2, wgu, error, error_size) ||
        !wan22_ln_upload_gpu(&block->norm3, wgu, error, error_size) ||
        !wan22_linear_upload_gpu(&block->ffn0, wgu, 0, error, error_size) ||
        !wan22_linear_upload_gpu(&block->ffn2, wgu, 0, error, error_size)) {
        return 0;
    }
    /* Per-block adaLN table: slots 0-5 = raw block params (BF16, exact),
     * slot 6 = the constant 1.0 cross-attention residual gate. e0 is added
     * in F32 inside the LN-adaLN / gate kernels. */
    {
        uint32_t dim = wgu->config.dim;
        uint16_t *table = malloc((size_t)(WAN22_MOD_SLOTS + 1) * dim * 2);
        if (!table) {
            snprintf(error, error_size, "cannot stage block modulation");
            return 0;
        }
        for (uint32_t i = 0; i < WAN22_MOD_SLOTS * dim; i++) {
            table[i] = wan_bf16_bits(block->modulation[i]);
        }
        for (uint32_t c = 0; c < dim; c++) {
            table[(size_t)WAN22_MOD_SLOTS * dim + c] = 0x3f80u;
        }
        block->gpu_modulation = wan22_tensor_upload_bf16(
            wgu->gpu, table, (size_t)(WAN22_MOD_SLOTS + 1) * dim);
        free(table);
        if (!block->gpu_modulation) {
            snprintf(error, error_size,
                     "cannot upload block modulation: %s",
                     h3_gpu_error(wgu->gpu));
            return 0;
        }
    }
    return 1;
}

/* --- device-state residency (swap the DiT out before the VAE decode) -----
 * A 24 GB card cannot hold the BF16 DiT (~9.8 GiB) and the streaming VAE
 * decode (peak live 15.2 GiB / 21.3 GiB reserved at 720p) at the same time,
 * so the pipeline drops the DiT device state at the phase boundary.  The
 * host BF16 weights are what the CPU reference path runs from, so they are
 * already there and resume costs one upload, not a checkpoint re-read.
 */
static void wan22_linear_release_gpu(h3_wan22_linear *linear) {
    h3_gpu_tensor_free(linear->gpu_weight);
    h3_gpu_tensor_free(linear->gpu_bias);
    linear->gpu_weight = NULL;
    linear->gpu_bias = NULL;
}

static void wan22_rms_release_gpu(h3_wan22_rms *rms) {
    h3_gpu_tensor_free(rms->gpu_weight);
    rms->gpu_weight = NULL;
}

static void wan22_ln_release_gpu(h3_wan22_ln *ln) {
    h3_gpu_tensor_free(ln->gpu_weight);
    h3_gpu_tensor_free(ln->gpu_bias);
    ln->gpu_weight = NULL;
    ln->gpu_bias = NULL;
}

void h3_wan22_block_release_gpu(h3_wan22_block *block) {
    if (!block) return;
    h3_gpu_tensor_free(block->gpu_modulation);
    block->gpu_modulation = NULL;
    wan22_linear_release_gpu(&block->q);
    wan22_linear_release_gpu(&block->k);
    wan22_linear_release_gpu(&block->v);
    wan22_linear_release_gpu(&block->o);
    wan22_linear_release_gpu(&block->cq);
    wan22_linear_release_gpu(&block->ck);
    wan22_linear_release_gpu(&block->cv);
    wan22_linear_release_gpu(&block->co);
    wan22_rms_release_gpu(&block->norm_q);
    wan22_rms_release_gpu(&block->norm_k);
    wan22_rms_release_gpu(&block->cnorm_q);
    wan22_rms_release_gpu(&block->cnorm_k);
    wan22_ln_release_gpu(&block->norm1);
    wan22_ln_release_gpu(&block->norm2);
    wan22_ln_release_gpu(&block->norm3);
    wan22_linear_release_gpu(&block->ffn0);
    wan22_linear_release_gpu(&block->ffn2);
}

/* Free the [seq, ...] scratch a reserve built.  The tiny norm constants and
 * the e0 carrier stay: they are three dim-sized rows. */
static void wan22_gpu_release_scratch(h3_wan22_gpu *wgu) {
    h3_gpu_tensor_free(wgu->x);
    h3_gpu_tensor_free(wgu->x_f32);
    h3_gpu_tensor_free(wgu->normed);
    h3_gpu_tensor_free(wgu->q);
    h3_gpu_tensor_free(wgu->k);
    h3_gpu_tensor_free(wgu->v);
    h3_gpu_tensor_free(wgu->attn);
    h3_gpu_tensor_free(wgu->branch);
    h3_gpu_tensor_free(wgu->ffn_a);
    h3_gpu_tensor_free(wgu->ctx);
    h3_gpu_tensor_free(wgu->ck);
    h3_gpu_tensor_free(wgu->cv);
    h3_gpu_tensor_free(wgu->rope_cos);
    h3_gpu_tensor_free(wgu->rope_sin);
    h3_gpu_tensor_free(wgu->row_map);
    wgu->x = wgu->x_f32 = wgu->normed = NULL;
    wgu->q = wgu->k = wgu->v = wgu->attn = wgu->branch = NULL;
    wgu->ffn_a = wgu->ctx = wgu->ck = wgu->cv = NULL;
    wgu->rope_cos = wgu->rope_sin = wgu->row_map = NULL;
    /* Reserve must rebuild these on the next forward. */
    wgu->seq = 0;
    wgu->ctx_seq = 0;
}

int h3_wan22_gpu_state_resident(const h3_wan22_gpu *wgu) {
    return wgu && !wgu->released;
}

int h3_wan22_gpu_state_release(h3_wan22_block **blocks, uint32_t block_count,
                               h3_wan22_gpu *wgu, char *error,
                               size_t error_size) {
    (void)error;
    (void)error_size;
    if (!wgu) return 0;
    if (wgu->released) return 1;
    /* The blocks go back to the allocator's pool, which the VAE's own
     * context (a different stream) then hands out: cudaFreeAsync only makes
     * a block reusable once the freeing stream has retired, so drain it
     * here.  One fence per generation is not worth avoiding. */
    h3_gpu_submit(wgu->gpu);
    for (uint32_t i = 0; i < block_count; i++) {
        h3_wan22_block_release_gpu(blocks[i]);
    }
    wan22_gpu_release_scratch(wgu);
    wgu->released = 1;
    return 1;
}

int h3_wan22_gpu_state_resume(h3_wan22_block **blocks, uint32_t block_count,
                              h3_wan22_gpu *wgu, uint32_t seq, uint32_t ctx_seq,
                              char *error, size_t error_size) {
    if (!wgu) {
        snprintf(error, error_size, "no Wan 2.2 GPU context to resume");
        return 0;
    }
    /* Trace: one pool snapshot before anything, so the split below can
     * report how much of the resume was pool growth (real cudaMalloc inside
     * cudaMallocAsync) versus host-to-device copy time. */
    const int trace = wan22_dit_trace();
    h3_gpu_mem_stats pool_before = {0};
    if (trace) h3_gpu_mem_query(wgu->gpu, &pool_before);
    double weights_seconds = 0.0, scratch_seconds = 0.0;
    if (wgu->released) {
        const double started = trace ? wan22_dit_now() : 0.0;
        for (uint32_t i = 0; i < block_count; i++) {
            if (!h3_wan22_block_upload_gpu(blocks[i], wgu, error,
                                           error_size)) {
                return 0;
            }
        }
        wgu->released = 0;
        if (trace) weights_seconds = wan22_dit_now() - started;
    }
    if (seq != 0) {
        const double started = trace ? wan22_dit_now() : 0.0;
        if (!h3_wan22_gpu_reserve(wgu, seq, ctx_seq, error, error_size))
            return 0;
        if (trace) scratch_seconds = wan22_dit_now() - started;
    }
    if (trace) {
        h3_gpu_mem_stats pool_after = {0};
        h3_gpu_mem_query(wgu->gpu, &pool_after);
        fprintf(stderr,
                "wan22: resume split weights=%.3fs scratch=%.3fs "
                "mallocs=%llu pool used %.2f->%.2f GiB reserved %.2f->%.2f "
                "GiB\n",
                weights_seconds, scratch_seconds,
                (unsigned long long)(pool_after.mallocs - pool_before.mallocs),
                (double)pool_before.used_current / (double)(1u << 30),
                (double)pool_after.used_current / (double)(1u << 30),
                (double)pool_before.reserved_current / (double)(1u << 30),
                (double)pool_after.reserved_current / (double)(1u << 30));
        fflush(stderr);
    }
    return 1;
}

/* Per-step e0 upload into the shared [6, dim] carrier; per-block tables
 * and this e0 are summed in F32 inside the kernels. */
int h3_wan22_gpu_set_e0_gpu(h3_wan22_gpu *wgu, const float *e0,
                            char *error, size_t error_size) {
    uint32_t dim = wgu->config.dim;
    uint16_t *staged = malloc((size_t)WAN22_MOD_SLOTS * dim * 2);
    if (!staged) {
        snprintf(error, error_size, "cannot stage e0");
        return 0;
    }
    for (uint32_t i = 0; i < WAN22_MOD_SLOTS * dim; i++) {
        staged[i] = wan_bf16_bits(e0[i]);
    }
    int ok = h3_gpu_tensor_write_bf16(wgu->modulation, staged,
                                      (size_t)WAN22_MOD_SLOTS * dim);
    free(staged);
    if (!ok) {
        snprintf(error, error_size, "cannot upload e0: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    return 1;
}

int h3_wan22_block_forward_gpu(
    h3_wan22_block *block, h3_wan22_gpu *wgu, h3_gpu_tensor *x_f32,
    uint32_t seq, const h3_gpu_tensor *context, uint32_t ctx_seq,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    char *error, size_t error_size) {
    if (wgu->released) {
        snprintf(error, error_size,
                 "Wan 2.2 DiT device state was released; call "
                 "h3_wan22_gpu_state_resume before forwarding");
        return 0;
    }
    const h3_wan22_config *config = &wgu->config;
    uint32_t dim = config->dim;
    uint32_t ffn = config->ffn_dim;
    uint32_t heads = config->heads;
    uint32_t head_dim = config->head_dim;
    float eps = config->norm_eps;
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    /* --- self-attention -------------------------------------------------- */
    const int dump_stages = wan22_stage_dump_this_block();
    if (!h3_gpu_layer_norm_adaln_f32in_bf16(
            wgu->gpu, wgu->normed, x_f32, block->norm1.gpu_weight,
            block->gpu_modulation, wgu->modulation,
            wgu->row_map, seq, dim, WAN22_MOD_SLOTS + 1, WAN22_MOD_SHIFT_MSA,
            WAN22_MOD_SCALE_MSA, eps) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->q, wgu->normed, block->q.gpu_weight,
                            block->q.gpu_bias, seq, dim, dim) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->k, wgu->normed, block->k.gpu_weight,
                            block->k.gpu_bias, seq, dim, dim) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->v, wgu->normed, block->v.gpu_weight,
                            block->v.gpu_bias, seq, dim, dim)) {
        snprintf(error, error_size, "Wan 2.2 GPU self-attention failed: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    if (dump_stages)
        wan22_dump_stage_bf16("normed", wgu->normed, (size_t)seq * dim);
    if (!h3_gpu_rms_norm_bf16(wgu->gpu, wgu->q, wgu->q,
                              block->norm_q.gpu_weight, seq, dim, eps) ||
        !h3_gpu_rms_norm_bf16(wgu->gpu, wgu->k, wgu->k,
                              block->norm_k.gpu_weight, seq, dim, eps) ||
        !h3_gpu_rope_text_bf16(wgu->gpu, wgu->q, wgu->k, rope_cos, rope_sin,
                               seq, heads, heads, head_dim)) {
        snprintf(error, error_size, "Wan 2.2 GPU self-attention failed: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    if (dump_stages)
        wan22_dump_stage_bf16("q", wgu->q, (size_t)seq * dim);
    if (dump_stages) {
        wan22_dump_stage_bf16("k", wgu->k, (size_t)seq * dim);
        wan22_dump_stage_bf16("v", wgu->v, (size_t)seq * dim);
    }
    /* Flash SDPA: online softmax, no materialized scores. Keeps the 720p
     * self-attention (seq=27280) free of the ~71.6 GiB F32 score tensor the
     * materialized paths would need. */
    if (!h3_gpu_sdpa_flash_bf16(wgu->gpu, wgu->attn, wgu->q, wgu->k, wgu->v,
                                seq, seq, heads, head_dim, attn_scale)) {
        snprintf(error, error_size, "Wan 2.2 GPU self-attention failed: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    if (dump_stages)
        wan22_dump_stage_bf16("attn", wgu->attn, (size_t)seq * dim);
    if (!h3_gpu_linear_bf16(wgu->gpu, wgu->branch, wgu->attn,
                            block->o.gpu_weight, block->o.gpu_bias, seq, dim,
                            dim)) {
        snprintf(error, error_size, "Wan 2.2 GPU self-attention failed: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    if (dump_stages)
        wan22_dump_stage_bf16("branch", wgu->branch, (size_t)seq * dim);
    if (!h3_gpu_gate_f32res_bf16(wgu->gpu, x_f32, wgu->branch,
                                 block->gpu_modulation, wgu->modulation,
                                 WAN22_MOD_SLOTS, wgu->row_map, seq, dim,
                                 WAN22_MOD_SLOTS + 1, WAN22_MOD_GATE_MSA)) {
        snprintf(error, error_size, "Wan 2.2 GPU self-attention failed: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }

    /* --- cross-attention (affine LN, whole-row QK RMS, no RoPE/gate) ----- */
    if (!h3_gpu_layer_norm_f32in_bf16(wgu->gpu, wgu->normed, x_f32,
                                      block->norm3.gpu_weight,
                                      block->norm3.gpu_bias,
                                      seq, dim, eps) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->q, wgu->normed, block->cq.gpu_weight,
                            block->cq.gpu_bias, seq, dim, dim) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->ck, context, block->ck.gpu_weight,
                            block->ck.gpu_bias, ctx_seq, dim, dim) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->cv, context, block->cv.gpu_weight,
                            block->cv.gpu_bias, ctx_seq, dim, dim) ||
        !h3_gpu_rms_norm_bf16(wgu->gpu, wgu->q, wgu->q,
                              block->cnorm_q.gpu_weight, seq, dim, eps) ||
        !h3_gpu_rms_norm_bf16(wgu->gpu, wgu->ck, wgu->ck,
                              block->cnorm_k.gpu_weight, ctx_seq, dim, eps) ||
        !h3_gpu_sdpa_flash_bf16(wgu->gpu, wgu->attn, wgu->q, wgu->ck, wgu->cv,
                                seq, ctx_seq, heads, head_dim, attn_scale) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->branch, wgu->attn,
                            block->co.gpu_weight, block->co.gpu_bias, seq, dim,
                            dim) ||
        !h3_gpu_gate_f32res_bf16(wgu->gpu, x_f32, wgu->branch,
                                 block->gpu_modulation, NULL, 0,
                                 wgu->row_map, seq, dim, WAN22_MOD_SLOTS + 1,
                                 WAN22_MOD_GATE_ADD)) {
        snprintf(error, error_size, "Wan 2.2 GPU cross-attention failed: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }

    /* --- FFN -------------------------------------------------------------- */
    if (!h3_gpu_layer_norm_adaln_f32in_bf16(
            wgu->gpu, wgu->normed, x_f32, block->norm2.gpu_weight,
            block->gpu_modulation, wgu->modulation,
            wgu->row_map, seq, dim, WAN22_MOD_SLOTS + 1, WAN22_MOD_SHIFT_MLP,
            WAN22_MOD_SCALE_MLP, eps) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->ffn_a, wgu->normed,
                            block->ffn0.gpu_weight, block->ffn0.gpu_bias, seq,
                            dim, ffn) ||
        !h3_gpu_gelu_bf16(wgu->gpu, wgu->ffn_a, wgu->ffn_a,
                          (size_t)seq * ffn, 1) ||
        !h3_gpu_linear_bf16(wgu->gpu, wgu->branch, wgu->ffn_a,
                            block->ffn2.gpu_weight, block->ffn2.gpu_bias, seq,
                            ffn, dim) ||
        !h3_gpu_gate_f32res_bf16(wgu->gpu, x_f32, wgu->branch,
                                 block->gpu_modulation, wgu->modulation,
                                 WAN22_MOD_SLOTS, wgu->row_map, seq, dim,
                                 WAN22_MOD_SLOTS + 1, WAN22_MOD_GATE_MLP)) {
        snprintf(error, error_size, "Wan 2.2 GPU FFN failed: %s",
                 h3_gpu_error(wgu->gpu));
        return 0;
    }
    return 1;
}
