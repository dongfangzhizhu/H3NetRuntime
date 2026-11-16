/* Wan 2.2 T2V 5B Phase 2: CausalConv3d VAE decoder (CPU fp32).
 * See h3_wan22_vae.h for the architecture and the whole-sequence temporal
 * equivalence argument against the reference's feat_cache streaming. */
#include "h3_wan22_vae.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VAE_Z_DIM 48
#define VAE_NUM_BLOCKS 4
/* dims[i] -> dims[i+1] per upsample block (vae2_2.py Decoder3d). */
static const uint32_t VAE_DIMS[VAE_NUM_BLOCKS + 1] = {
    1024, 1024, 1024, 512, 256};
/* WanVAE_ reverses temperal_downsample=[False, True, True], so blocks 0 and
 * 1 end in an upsample3d Resample and block 2 in an upsample2d one. */
static const uint8_t VAE_T_UP[VAE_NUM_BLOCKS] = {1, 1, 0, 0};
static const uint8_t VAE_UP_FLAG[VAE_NUM_BLOCKS] = {1, 1, 1, 0};
#define VAE_RES_BLOCKS 3 /* num_res_blocks + 1 per upsample block */

#define VAE_NORM_EPS 1e-12f

typedef struct {
    uint32_t in;
    uint32_t out;
    uint8_t kt;
    uint8_t ks;
    float *weight;
    float *bias;
} vae_conv;

typedef struct {
    uint32_t in;
    uint32_t out;
    float *gamma0; /* RMS_norm(in), residual.0 */
    vae_conv a;    /* residual.2: (in -> out), kt3 ks3 */
    float *gamma1; /* RMS_norm(out), residual.3 */
    vae_conv b;    /* residual.6: (out -> out), kt3 ks3 */
    int has_shortcut;
    vae_conv sc; /* shortcut: (in -> out), kt1 ks1 */
} vae_resblock;

typedef struct {
    uint32_t dim;
    float *gamma;  /* RMS_norm(dim), channel-first */
    vae_conv qkv;  /* to_qkv: (dim -> 3*dim), kt1 ks1 */
    vae_conv proj; /* proj: (dim -> dim), kt1 ks1 */
} vae_attn;

typedef struct {
    uint32_t in;
    uint32_t out;
    vae_resblock res[VAE_RES_BLOCKS];
    int up_flag;        /* the block ends in a Resample */
    int t_up;           /* the Resample is upsample3d (temporal) */
    vae_conv spatial;   /* resample.1: (out -> out), kt1 ks3 */
    vae_conv time_conv; /* (out -> 2*out), kt3 ks1; only when t_up */
} vae_up;

/* GPU decode branch (WAN22_VAE_GPU=1), implemented at the end of this file.
 * The device mirror is built lazily on the first GPU decode; a GPU failure
 * is reported, never silently swapped for the CPU path. */
struct wan22_vae_gpu_state;
static int wan22_vae_gpu_requested(void);
static struct wan22_vae_gpu_state *wan22_vae_gpu_create(
    const h3_wan22_vae_decoder *decoder, char *error, size_t error_size);
static void wan22_vae_gpu_free(struct wan22_vae_gpu_state *state);
static int wan22_vae_decode_gpu(h3_wan22_vae_decoder *decoder,
                                struct wan22_vae_gpu_state *state,
                                const float *z, uint32_t frames, uint32_t h,
                                uint32_t w, float *out, char *error,
                                size_t error_size);
static int wan22_vae_stream_requested(uint32_t frames);
static int wan22_vae_decode_gpu_stream(h3_wan22_vae_decoder *decoder,
                                       struct wan22_vae_gpu_state *state,
                                       const float *z, uint32_t frames,
                                       uint32_t h, uint32_t w, float *out,
                                       char *error, size_t error_size);

struct h3_wan22_vae_decoder {
    vae_conv conv1; /* (48 -> 1024), kt3 ks3 */
    vae_conv conv2; /* (48 -> 48), kt1 ks1 (WanVAE_.conv2) */
    vae_resblock middle[2];
    vae_attn attn;
    vae_up ups[VAE_NUM_BLOCKS];
    float *head_gamma;
    vae_conv head; /* (256 -> 12), kt3 ks3 */
    float mean[VAE_Z_DIM];
    float std[VAE_Z_DIM];
    /* Device weight mirrors + CUDA context, built lazily on the first
     * WAN22_VAE_GPU=1 decode. */
    struct wan22_vae_gpu_state *gpu;
};

static void conv_free(vae_conv *conv) {
    free(conv->weight);
    free(conv->bias);
    memset(conv, 0, sizeof(*conv));
}

void h3_wan22_vae_decoder_free(h3_wan22_vae_decoder *decoder) {
    if (!decoder) return;
    wan22_vae_gpu_free(decoder->gpu);
    conv_free(&decoder->conv1);
    conv_free(&decoder->conv2);
    for (int i = 0; i < 2; i++) {
        vae_resblock *rb = &decoder->middle[i];
        free(rb->gamma0);
        free(rb->gamma1);
        conv_free(&rb->a);
        conv_free(&rb->b);
    }
    free(decoder->attn.gamma);
    conv_free(&decoder->attn.qkv);
    conv_free(&decoder->attn.proj);
    for (int i = 0; i < VAE_NUM_BLOCKS; i++) {
        vae_up *up = &decoder->ups[i];
        for (int j = 0; j < VAE_RES_BLOCKS; j++) {
            vae_resblock *rb = &up->res[j];
            free(rb->gamma0);
            free(rb->gamma1);
            conv_free(&rb->a);
            conv_free(&rb->b);
            if (rb->has_shortcut) conv_free(&rb->sc);
        }
        conv_free(&up->spatial);
        if (up->t_up) conv_free(&up->time_conv);
    }
    free(decoder->head_gamma);
    conv_free(&decoder->head);
    free(decoder);
}

static int read_tensor(const h3_st_header *header, const char *name,
                       uint64_t count, float **values, char *error,
                       size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    if (!tensor) {
        snprintf(error, error_size, "missing VAE tensor %s", name);
        return 0;
    }
    if (h3_st_tensor_elements(tensor) != count) {
        snprintf(error, error_size,
                 "VAE tensor %s has %llu elements, want %llu", name,
                 (unsigned long long)h3_st_tensor_elements(tensor),
                 (unsigned long long)count);
        return 0;
    }
    return h3_st_read_f32(header, tensor, NULL, NULL, values, error,
                          error_size);
}

static int read_conv(const h3_st_header *header, const char *prefix,
                     uint32_t in, uint32_t out, uint8_t kt, uint8_t ks,
                     int has_bias, vae_conv *conv, char *error,
                     size_t error_size) {
    char name[192];
    conv->in = in;
    conv->out = out;
    conv->kt = kt;
    conv->ks = ks;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (!read_tensor(header, name, (uint64_t)out * in * kt * ks * ks,
                     &conv->weight, error, error_size)) {
        return 0;
    }
    if (has_bias) {
        snprintf(name, sizeof(name), "%s.bias", prefix);
        if (!read_tensor(header, name, out, &conv->bias, error, error_size)) {
            return 0;
        }
    }
    return 1;
}

static int read_resblock(const h3_st_header *header, const char *prefix,
                         uint32_t in, uint32_t out, vae_resblock *rb,
                         char *error, size_t error_size) {
    char name[224];
    memset(rb, 0, sizeof(*rb));
    rb->in = in;
    rb->out = out;
    snprintf(name, sizeof(name), "%s.residual.0.gamma", prefix);
    if (!read_tensor(header, name, in, &rb->gamma0, error, error_size)) {
        return 0;
    }
    snprintf(name, sizeof(name), "%s.residual.2", prefix);
    if (!read_conv(header, name, in, out, 3, 3, 1, &rb->a, error,
                   error_size)) {
        return 0;
    }
    snprintf(name, sizeof(name), "%s.residual.3.gamma", prefix);
    if (!read_tensor(header, name, out, &rb->gamma1, error, error_size)) {
        return 0;
    }
    snprintf(name, sizeof(name), "%s.residual.6", prefix);
    if (!read_conv(header, name, out, out, 3, 3, 1, &rb->b, error,
                   error_size)) {
        return 0;
    }
    rb->has_shortcut = in != out;
    if (rb->has_shortcut) {
        snprintf(name, sizeof(name), "%s.shortcut", prefix);
        if (!read_conv(header, name, in, out, 1, 1, 1, &rb->sc, error,
                       error_size)) {
            return 0;
        }
    }
    return 1;
}

h3_wan22_vae_decoder *h3_wan22_vae_decoder_load(const char *vae_path,
                                                char *error,
                                                size_t error_size) {
    h3_st_header header;
    if (!h3_st_read_header(vae_path, &header, error, error_size)) {
        return NULL;
    }
    h3_wan22_vae_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        snprintf(error, error_size, "out of host memory (vae decoder)");
        h3_st_free_header(&header);
        return NULL;
    }
    /* 48 per-channel constants from Wan2_2_VAE (vae2_2.py). */
    static const float kMean[VAE_Z_DIM] = {
        -0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f,
        0.1838f,  0.1557f,  -0.1382f, 0.0542f,  0.2813f,  0.0891f,
        0.1570f,  -0.0098f, 0.0375f,  -0.1825f, -0.2246f, -0.1207f,
        -0.0698f, 0.5109f,  0.2665f,  -0.2108f, -0.2158f, 0.2502f,
        -0.2055f, -0.0322f, 0.1109f,  0.1567f,  -0.0729f, 0.0899f,
        -0.2799f, -0.1230f, -0.0313f, -0.1649f, 0.0117f,  0.0723f,
        -0.2839f, -0.2083f, -0.0520f, 0.3748f,  0.0152f,  0.1957f,
        0.1433f,  -0.2944f, 0.3573f,  -0.0548f, -0.1681f, -0.0667f};
    static const float kStd[VAE_Z_DIM] = {
        0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f,
        0.5013f, 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f,
        0.4978f, 0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f,
        0.8986f, 0.5659f, 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f,
        0.4999f, 0.6866f, 0.4093f, 0.5709f, 0.6065f, 0.6415f, 0.4944f,
        0.5726f, 1.2042f, 0.5458f, 1.6887f, 0.3971f, 1.0600f, 0.3943f,
        0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f};
    for (int c = 0; c < VAE_Z_DIM; c++) {
        decoder->mean[c] = kMean[c];
        decoder->std[c] = kStd[c];
    }

    char name[224];
    int ok = 1;
    ok = ok && read_conv(&header, "decoder.conv1", VAE_Z_DIM, 1024, 3, 3, 1,
                         &decoder->conv1, error, error_size);
    ok = ok && read_conv(&header, "conv2", VAE_Z_DIM, VAE_Z_DIM, 1, 1, 1,
                         &decoder->conv2, error, error_size);
    ok = ok && read_resblock(&header, "decoder.middle.0", 1024, 1024,
                             &decoder->middle[0], error, error_size);
    ok = ok && read_resblock(&header, "decoder.middle.2", 1024, 1024,
                             &decoder->middle[1], error, error_size);
    decoder->attn.dim = 1024;
    ok = ok && read_tensor(&header, "decoder.middle.1.norm.gamma", 1024,
                           &decoder->attn.gamma, error, error_size);
    ok = ok && read_conv(&header, "decoder.middle.1.to_qkv", 1024, 3072, 1, 1,
                         1, &decoder->attn.qkv, error, error_size);
    ok = ok && read_conv(&header, "decoder.middle.1.proj", 1024, 1024, 1, 1, 1,
                         &decoder->attn.proj, error, error_size);
    for (int i = 0; ok && i < VAE_NUM_BLOCKS; i++) {
        vae_up *up = &decoder->ups[i];
        uint32_t in = VAE_DIMS[i];
        uint32_t out = VAE_DIMS[i + 1];
        up->in = in;
        up->out = out;
        up->t_up = VAE_T_UP[i];
        up->up_flag = VAE_UP_FLAG[i];
        for (int j = 0; ok && j < VAE_RES_BLOCKS; j++) {
            snprintf(name, sizeof(name), "decoder.upsamples.%d.upsamples.%d",
                     i, j);
            ok = read_resblock(&header, name, j == 0 ? in : out, out,
                               &up->res[j], error, error_size);
        }
        if (ok && up->up_flag) {
            snprintf(name, sizeof(name),
                     "decoder.upsamples.%d.upsamples.3.resample.1", i);
            ok = read_conv(&header, name, out, out, 1, 3, 1, &up->spatial,
                           error, error_size);
        }
        if (ok && up->t_up) {
            snprintf(name, sizeof(name),
                     "decoder.upsamples.%d.upsamples.3.time_conv", i);
            ok = read_conv(&header, name, out, out * 2, 3, 1, 1,
                           &up->time_conv, error, error_size);
        }
    }
    ok = ok && read_tensor(&header, "decoder.head.0.gamma", 256,
                           &decoder->head_gamma, error, error_size);
    ok = ok && read_conv(&header, "decoder.head.2", 256, 12, 3, 3, 1,
                         &decoder->head, error, error_size);
    h3_st_free_header(&header);
    if (!ok) {
        h3_wan22_vae_decoder_free(decoder);
        return NULL;
    }
    return decoder;
}

static float *alloc_act(size_t count) {
    return malloc(count * sizeof(float));
}

/* Causal convolution.  Output frames equal input frames; a kt=3 conv reads
 * input frames t-2, t-1, t (missing leading frames act as zero padding).
 * kt=1 layers (2d resample convs, 1x1x1 convs) are time independent.
 * The input must be a contiguous [in_ch][t][h][w] tensor. */
static void conv_forward(const vae_conv *conv, const float *in, float *out,
                         uint32_t t, uint32_t h, uint32_t w) {
    const uint32_t in_ch = conv->in;
    const uint32_t out_ch = conv->out;
    const uint32_t kt = conv->kt;
    const uint32_t ks = conv->ks;
    const size_t plane = (size_t)h * w;
    const int64_t total = (int64_t)out_ch * t;
    {
        int64_t unit;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (unit = 0; unit < total; unit++) {
            const uint32_t o = (uint32_t)(unit / t);
            const uint32_t ti = (uint32_t)(unit % t);
            float *out_row = out + ((size_t)o * t + ti) * plane;
            const float bias = conv->bias ? conv->bias[o] : 0.0f;
            for (size_t index = 0; index < plane; index++) {
                out_row[index] = bias;
            }
            const float *weight_o =
                conv->weight + (size_t)o * in_ch * kt * ks * ks;
            for (uint32_t step = 0; step < kt; step++) {
                int64_t ts = (int64_t)ti + (int64_t)step - ((int64_t)kt - 1);
                if (ts < 0) continue;
                const size_t frame = (size_t)ts * plane;
                for (uint32_t c = 0; c < in_ch; c++) {
                    const float *wc =
                        weight_o + ((size_t)c * kt + step) * ks * ks;
                    if (ks == 1) {
                        const float *in_row = in + (size_t)c * t * plane + frame;
                        const float wc0 = wc[0];
                        for (size_t index = 0; index < plane; index++) {
                            out_row[index] += wc0 * in_row[index];
                        }
                        continue;
                    }
                    const size_t in_frame = (size_t)c * t * plane + frame;
                    for (uint32_t ky = 0; ky < 3; ky++) {
                        const float w0 = wc[ky * 3 + 0];
                        const float w1 = wc[ky * 3 + 1];
                        const float w2 = wc[ky * 3 + 2];
                        uint32_t y_begin = ky == 0 ? 1u : 0u;
                        uint32_t y_end = (ky == 2 && h > 1) ? h - 1 : h;
                        for (uint32_t y = y_begin; y < y_end; y++) {
                            const float *in_row =
                                in + in_frame + ((size_t)y + ky - 1) * w;
                            float *acc = out_row + (size_t)y * w;
                            if (w > 1) {
                                acc[0] += w1 * in_row[0] + w2 * in_row[1];
                                for (uint32_t x = 1; x < w - 1; x++) {
                                    acc[x] += w0 * in_row[x - 1] +
                                              w1 * in_row[x] +
                                              w2 * in_row[x + 1];
                                }
                                acc[w - 1] += w0 * in_row[w - 2] +
                                              w1 * in_row[w - 1];
                            } else {
                                acc[0] += w1 * in_row[0];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* RMS_norm (vae2_2.py RMS_norm, channel_first): x / max(|x|_2, eps) *
 * sqrt(dim) * gamma.  The VAE norms carry no bias.  Row r = f * h + y;
 * element (ch, f, y, x) sits at ch * t * plane + r * w + x. */
static void rms_forward(const float *gamma, const float *in, float *out,
                        uint32_t c, uint32_t t, uint32_t h, uint32_t w) {
    const size_t plane = (size_t)h * w;
    const float scale = sqrtf((float)c);
    const int64_t rows = (int64_t)t * h;
    {
        int64_t row;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (row = 0; row < rows; row++) {
            const float *in_row = in + (size_t)row * w;
            float *out_row = out + (size_t)row * w;
            for (uint32_t index = 0; index < w; index++) {
                double sum2 = 0.0;
                for (uint32_t ch = 0; ch < c; ch++) {
                    double value = in_row[(size_t)ch * t * plane + index];
                    sum2 += value * value;
                }
                float denom = (float)sqrt(sum2);
                if (denom < VAE_NORM_EPS) denom = VAE_NORM_EPS;
                const float inv = scale / denom;
                for (uint32_t ch = 0; ch < c; ch++) {
                    const size_t idx = (size_t)ch * t * plane + index;
                    out_row[idx] = in_row[idx] * inv * gamma[ch];
                }
            }
        }
    }
}

static void silu_forward(const float *in, float *out, size_t count) {
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)count; index++) {
            const float value = in[index];
            out[index] = value / (1.0f + expf(-value));
        }
    }
}

static void add_forward(const float *left, const float *right, float *out,
                        size_t count) {
    {
        int64_t index;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (index = 0; index < (int64_t)count; index++) {
            out[index] = left[index] + right[index];
        }
    }
}

/* Copy one frame block [C][H][W] into tensor[c][f] for every channel of a
 * [C][T][H][W] tensor (frames are strided per channel). */
static void scatter_frame(float *tensor, uint32_t c, uint32_t t_full,
                          const float *frame_block, uint32_t f,
                          size_t plane_full) {
    for (uint32_t ch = 0; ch < c; ch++) {
        memcpy(tensor + (size_t)ch * t_full * plane_full +
                   (size_t)f * plane_full,
               frame_block + (size_t)ch * plane_full,
               plane_full * sizeof(float));
    }
}

/* AttentionBlock: channel RMSNorm -> 1x1 qkv -> single-head spatial
 * self-attention per frame (scale 1/sqrt(dim)) -> 1x1 proj -> residual. */
static int attn_forward(const vae_attn *attn, float *x, uint32_t t,
                        uint32_t h, uint32_t w) {
    const uint32_t c = attn->dim;
    const size_t tokens = (size_t)h * w;
    const size_t plane = tokens;
    float *normed = alloc_act((size_t)c * t * plane);
    float *qkv = alloc_act((size_t)c * 3 * t * plane);
    float *proj_in = alloc_act((size_t)c * t * plane);
    if (!normed || !qkv || !proj_in) {
        free(normed);
        free(qkv);
        free(proj_in);
        return 0;
    }
    rms_forward(attn->gamma, x, normed, c, t, h, w);
    conv_forward(&attn->qkv, normed, qkv, t, h, w);
    const float scale = 1.0f / sqrtf((float)c);
    {
        int64_t f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (f = 0; f < (int64_t)t; f++) {
            float *q = malloc(tokens * c * sizeof(float));
            float *k = malloc(tokens * c * sizeof(float));
            float *v = malloc(tokens * c * sizeof(float));
            float *probs = malloc(tokens * tokens * sizeof(float));
            if (q && k && v && probs) {
                for (uint32_t ch = 0; ch < c; ch++) {
                    const float *src =
                        qkv + (size_t)ch * t * plane + (size_t)f * plane;
                    const float *src_k =
                        qkv + ((size_t)c + ch) * t * plane + (size_t)f * plane;
                    const float *src_v =
                        qkv + ((size_t)2 * c + ch) * t * plane +
                        (size_t)f * plane;
                    for (size_t i = 0; i < tokens; i++) {
                        q[i * c + ch] = src[i];
                        k[i * c + ch] = src_k[i];
                        v[i * c + ch] = src_v[i];
                    }
                }
                for (size_t i = 0; i < tokens; i++) {
                    const float *q_row = q + i * c;
                    double max_score = -1e30;
                    for (size_t j = 0; j < tokens; j++) {
                        const float *k_row = k + j * c;
                        double dot = 0.0;
                        for (uint32_t ch = 0; ch < c; ch++) {
                            dot += (double)q_row[ch] * k_row[ch];
                        }
                        double score = dot * scale;
                        probs[i * tokens + j] = (float)score;
                        if (score > max_score) max_score = score;
                    }
                    double sum = 0.0;
                    for (size_t j = 0; j < tokens; j++) {
                        double value =
                            exp((double)probs[i * tokens + j] - max_score);
                        probs[i * tokens + j] = (float)value;
                        sum += value;
                    }
                    const double inv = 1.0 / sum;
                    float *out_row = proj_in + (size_t)f * plane + i;
                    for (uint32_t ch = 0; ch < c; ch++) {
                        out_row[(size_t)ch * t * plane] = 0.0f;
                    }
                    for (size_t j = 0; j < tokens; j++) {
                        const float weight =
                            (float)((double)probs[i * tokens + j] * inv);
                        const float *v_row = v + j * c;
                        for (uint32_t ch = 0; ch < c; ch++) {
                            out_row[(size_t)ch * t * plane] += weight * v_row[ch];
                        }
                    }
                }
            }
            free(q);
            free(k);
            free(v);
            free(probs);
        }
    }
    float *proj_out = alloc_act((size_t)c * t * plane);
    if (!proj_out) {
        free(normed);
        free(qkv);
        free(proj_in);
        return 0;
    }
    conv_forward(&attn->proj, proj_in, proj_out, t, h, w);
    const size_t count = (size_t)c * t * plane;
    add_forward(x, proj_out, x, count);
    free(normed);
    free(qkv);
    free(proj_in);
    free(proj_out);
    return 1;
}

/* DupUp3d shortcut (vae2_2.py): channel repeat_interleave(repeats), then
 * each input frame expands to ft consecutive output frames and fs x fs
 * spatial pixels.  drop_first trims the first output frame, matching the
 * reference's first_chunk x[:, :, factor_t - 1:] trim. */
static void dupup_forward(const float *in, float *out, uint32_t in_ch,
                          uint32_t out_ch, uint32_t ft, uint32_t fs,
                          int drop_first, uint32_t t, uint32_t h,
                          uint32_t w) {
    const uint32_t repeats = out_ch * ft * fs * fs / in_ch;
    const uint32_t t2 = t * ft - (drop_first ? 1u : 0u);
    const uint32_t h2 = h * fs;
    const uint32_t w2 = w * fs;
    {
        int64_t o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (o = 0; o < (int64_t)out_ch; o++) {
            for (uint32_t a = 0; a < ft; a++) {
                for (uint32_t b = 0; b < fs; b++) {
                    for (uint32_t d = 0; d < fs; d++) {
                        uint32_t flat = (o * ft + a) * fs * fs + b * fs + d;
                        uint32_t c_in = flat / repeats;
                        for (uint32_t ti = 0; ti < t; ti++) {
                            uint32_t to = ti * ft + a;
                            if (drop_first) {
                                if (to == 0) continue;
                                to -= 1;
                            }
                            const float *src =
                                in + (((size_t)c_in * t + ti) * h) * w;
                            float *dst = out +
                                         (((size_t)o * t2 + to) * h2) * w2 +
                                         (size_t)b * w2 + d;
                            for (uint32_t y = 0; y < h; y++) {
                                for (uint32_t xx = 0; xx < w; xx++) {
                                    dst[(size_t)y * fs * w2 + (size_t)xx * fs] =
                                        src[(size_t)y * w + xx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Nearest-neighbour x2 spatial upsample (nn.Upsample nearest-exact at
 * scale 2: each input pixel repeats into a 2x2 block).  src addresses one
 * frame per channel with the given per-channel stride (frames of a
 * [C][T][H][W] tensor are strided, not contiguous); out is a contiguous
 * [C][2H][2W] single-frame block. */
static void nearest2x_forward(const float *src, size_t src_stride,
                              float *out, uint32_t c, uint32_t h,
                              uint32_t w) {
    const uint32_t h2 = h * 2;
    const uint32_t w2 = w * 2;
    {
        int64_t o;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (o = 0; o < (int64_t)c; o++) {
            const float *channel = src + (size_t)o * src_stride;
            float *dst = out + (size_t)o * h2 * w2;
            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t xx = 0; xx < w; xx++) {
                    const float value = channel[(size_t)y * w + xx];
                    float *block = dst + (size_t)y * 2 * w2 + (size_t)xx * 2;
                    block[0] = value;
                    block[1] = value;
                    block[w2] = value;
                    block[w2 + 1] = value;
                }
            }
        }
    }
}

/* unpatchify (vae2_2.py, patch_size 2): out[c][f][2h+q][2w+r] =
 * in[c*4 + r*2 + q][f][h][w]. */
static void unpatchify_forward(const float *in, float *out, uint32_t t,
                               uint32_t h, uint32_t w) {
    const uint32_t h2 = h * 2;
    const uint32_t w2 = w * 2;
    for (uint32_t ch = 0; ch < 3; ch++) {
        for (uint32_t r = 0; r < 2; r++) {
            for (uint32_t q = 0; q < 2; q++) {
                const uint32_t src = ch * 4 + r * 2 + q;
                {
                    int64_t f;
                    int64_t y;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
                    for (f = 0; f < (int64_t)t; f++) {
                        for (y = 0; y < (int64_t)h; y++) {
                            const float *in_row =
                                in + (((size_t)src * t + f) * h + y) * w;
                            float *out_row =
                                out + (((size_t)ch * t + f) * h2 +
                                       (size_t)y * 2 + q) *
                                          w2;
                            for (uint32_t xx = 0; xx < w; xx++) {
                                out_row[(size_t)xx * 2 + r] = in_row[xx];
                            }
                        }
                    }
                }
            }
        }
    }
}

static int resblock_forward(const vae_resblock *rb, const float *x,
                            float *out, uint32_t t, uint32_t h, uint32_t w) {
    const size_t in_count = (size_t)rb->in * t * h * w;
    const size_t out_count = (size_t)rb->out * t * h * w;
    float *shortcut = alloc_act(out_count);
    float *hidden = alloc_act(in_count);
    float *main = alloc_act(out_count);
    if (!shortcut || !hidden || !main) {
        free(shortcut);
        free(hidden);
        free(main);
        return 0;
    }
    if (rb->has_shortcut) {
        conv_forward(&rb->sc, x, shortcut, t, h, w);
    } else {
        memcpy(shortcut, x, out_count * sizeof(float));
    }
    rms_forward(rb->gamma0, x, hidden, rb->in, t, h, w);
    silu_forward(hidden, hidden, in_count);
    conv_forward(&rb->a, hidden, main, t, h, w);
    rms_forward(rb->gamma1, main, main, rb->out, t, h, w);
    silu_forward(main, main, out_count);
    conv_forward(&rb->b, main, out, t, h, w);
    add_forward(out, shortcut, out, out_count);
    free(shortcut);
    free(hidden);
    free(main);
    return 1;
}

/* One upsample block.  x: [in][T][H][W]; returns [out][T2][H2][W2]
 * (T2 = 2T-1 when t_up, else T; H2 = 2H when up_flag, else H). */
static float *up_block_forward(const vae_up *up, const float *x, uint32_t t,
                               uint32_t h, uint32_t w) {
    const uint32_t in = up->in;
    const uint32_t out = up->out;
    float *act = alloc_act((size_t)in * t * h * w);
    if (!act) return NULL;
    memcpy(act, x, (size_t)in * t * h * w * sizeof(float));
    for (int j = 0; j < VAE_RES_BLOCKS; j++) {
        float *next = alloc_act((size_t)out * t * h * w);
        if (!next || !resblock_forward(&up->res[j], act, next, t, h, w)) {
            free(act);
            free(next);
            return NULL;
        }
        free(act);
        act = next;
    }
    if (!up->up_flag) {
        return act; /* block 3: residual blocks only */
    }
    const uint32_t h2 = h * 2;
    const uint32_t w2 = w * 2;
    const uint32_t t_main = up->t_up ? 2 * t - 1 : t;
    float *main = alloc_act((size_t)out * t_main * h2 * w2);
    float *patch = alloc_act((size_t)out * h2 * w2);
    float *patch2 = alloc_act((size_t)out * h2 * w2);
    if (!main || !patch || !patch2) {
        free(act);
        free(main);
        free(patch);
        free(patch2);
        return NULL;
    }
    if (up->t_up) {
        /* Temporal path: y = causal time_conv over R[1:]; output frame 0
         * comes from R[0] spatially upsampled only (the reference skips
         * the time_conv for the first chunk), frames 1.. from the
         * channel-interleaved time_conv output. */
        float *y = NULL;
        if (t > 1) {
            /* R[1:] is not contiguous in [C][T][H][W]: gather the trailing
             * t-1 frames per channel first. */
            float *slice = alloc_act((size_t)out * (t - 1) * h * w);
            y = alloc_act((size_t)out * 2 * (t - 1) * h * w);
            if (!slice || !y) {
                free(slice);
                free(act);
                free(main);
                free(patch);
                free(patch2);
                return NULL;
            }
            for (uint32_t c = 0; c < out; c++) {
                memcpy(slice + (size_t)c * (t - 1) * h * w,
                       act + (size_t)c * t * h * w + h * w,
                       (size_t)(t - 1) * h * w * sizeof(float));
            }
            conv_forward(&up->time_conv, slice, y, t - 1, h, w);
            free(slice);
        }
        nearest2x_forward(act, (size_t)t * h * w, patch, out, h, w);
        conv_forward(&up->spatial, patch, patch2, 1, h2, w2);
        scatter_frame(main, out, t_main, patch2, 0, (size_t)h2 * w2);
        for (uint32_t j = 0; t > 1 && j < 2 * (t - 1); j++) {
            const size_t half = (size_t)(j % 2) * out * (t - 1) * h * w;
            const size_t index = (size_t)(j / 2) * h * w;
            nearest2x_forward(y + half + index, (size_t)(t - 1) * h * w,
                              patch, out, h, w);
            conv_forward(&up->spatial, patch, patch2, 1, h2, w2);
            scatter_frame(main, out, t_main, patch2, j + 1,
                          (size_t)h2 * w2);
        }
        free(y);
    } else {
        for (uint32_t f = 0; f < t; f++) {
            nearest2x_forward(act + (size_t)f * h * w, (size_t)t * h * w,
                              patch, out, h, w);
            conv_forward(&up->spatial, patch, patch2, 1, h2, w2);
            scatter_frame(main, out, t_main, patch2, f, (size_t)h2 * w2);
        }
    }
    float *shortcut = alloc_act((size_t)out * t_main * h2 * w2);
    if (!shortcut) {
        free(act);
        free(main);
        free(patch);
        free(patch2);
        return NULL;
    }
    dupup_forward(x, shortcut, in, out, up->t_up ? 2u : 1u, 2u,
                  up->t_up ? 1 : 0, t, h, w);
    add_forward(main, shortcut, main, (size_t)out * t_main * h2 * w2);
    free(act);
    free(patch);
    free(patch2);
    free(shortcut);
    return main;
}

int h3_wan22_vae_decode(const h3_wan22_vae_decoder *decoder, const float *z,
                        uint32_t frames, uint32_t h, uint32_t w, float *out,
                        char *error, size_t error_size) {
    if (!decoder || !z || !out || frames == 0 || h == 0 || w == 0) {
        snprintf(error, error_size, "invalid vae decode arguments");
        return 0;
    }
    if (wan22_vae_gpu_requested()) {
        h3_wan22_vae_decoder *mutable_decoder = (h3_wan22_vae_decoder *)decoder;
        if (!mutable_decoder->gpu) {
            mutable_decoder->gpu = wan22_vae_gpu_create(decoder, error,
                                                        error_size);
            if (!mutable_decoder->gpu) return 0;
        }
        if (wan22_vae_stream_requested(frames))
            return wan22_vae_decode_gpu_stream(mutable_decoder,
                                               mutable_decoder->gpu, z, frames,
                                               h, w, out, error, error_size);
        return wan22_vae_decode_gpu(mutable_decoder, mutable_decoder->gpu, z,
                                    frames, h, w, out, error, error_size);
    }
    const size_t latent_count = (size_t)VAE_Z_DIM * frames * h * w;
    float *act = alloc_act(latent_count);
    if (!act) {
        snprintf(error, error_size, "out of host memory (vae decode)");
        return 0;
    }
    /* WanVAE_.decode: z = z / scale[1] + scale[0] with scale[1] = 1/std,
     * i.e. z * std + mean per channel. */
    const size_t plane = (size_t)frames * h * w;
    for (uint32_t c = 0; c < VAE_Z_DIM; c++) {
        const float mean = decoder->mean[c];
        const float std = decoder->std[c];
        const float *src = z + (size_t)c * plane;
        float *dst = act + (size_t)c * plane;
        for (size_t i = 0; i < plane; i++) {
            dst[i] = src[i] * std + mean;
        }
    }
    float *next = alloc_act(latent_count);
    if (!next) {
        free(act);
        snprintf(error, error_size, "out of host memory (vae decode)");
        return 0;
    }
    conv_forward(&decoder->conv2, act, next, frames, h, w);
    free(act);
    act = next;
    /* conv1: 48 -> 1024 channels. */
    next = alloc_act((size_t)1024 * frames * h * w);
    if (!next) {
        free(act);
        snprintf(error, error_size, "out of host memory (vae decode)");
        return 0;
    }
    conv_forward(&decoder->conv1, act, next, frames, h, w);
    free(act);
    act = next;

    for (int i = 0; i < 2; i++) {
        next = alloc_act((size_t)1024 * frames * h * w);
        if (!next || !resblock_forward(&decoder->middle[i], act, next, frames,
                                       h, w)) {
            free(act);
            free(next);
            snprintf(error, error_size, "middle resblock failed");
            return 0;
        }
        free(act);
        act = next;
        if (i == 0) {
            if (!attn_forward(&decoder->attn, act, frames, h, w)) {
                free(act);
                snprintf(error, error_size, "attention block failed");
                return 0;
            }
        }
    }

    uint32_t t_cur = frames;
    uint32_t h_cur = h;
    uint32_t w_cur = w;
    uint32_t c_cur = 1024;
    for (int i = 0; i < VAE_NUM_BLOCKS; i++) {
        float *block_out =
            up_block_forward(&decoder->ups[i], act, t_cur, h_cur, w_cur);
        free(act);
        if (!block_out) {
            snprintf(error, error_size, "upsample block %d failed", i);
            return 0;
        }
        act = block_out;
        if (decoder->ups[i].t_up) t_cur = 2 * t_cur - 1;
        if (decoder->ups[i].up_flag) {
            h_cur *= 2;
            w_cur *= 2;
        }
        c_cur = decoder->ups[i].out;
    }

    float *head_in = alloc_act((size_t)c_cur * t_cur * h_cur * w_cur);
    float *head_out = alloc_act((size_t)12 * t_cur * h_cur * w_cur);
    if (!head_in || !head_out) {
        free(act);
        free(head_in);
        free(head_out);
        snprintf(error, error_size, "out of host memory (vae head)");
        return 0;
    }
    rms_forward(decoder->head_gamma, act, head_in, 256, t_cur, h_cur, w_cur);
    silu_forward(head_in, head_in, (size_t)256 * t_cur * h_cur * w_cur);
    conv_forward(&decoder->head, head_in, head_out, t_cur, h_cur, w_cur);
    unpatchify_forward(head_out, out, t_cur, h_cur, w_cur);
    free(act);
    free(head_in);
    free(head_out);
    return 1;
}

/* --- GPU decode branch (WAN22_VAE_GPU=1) ---------------------------------- */
/* Mirror of the CPU path above on the device: same op order, fp32 end to
 * end, device weight mirrors built lazily on the first GPU decode. A GPU
 * failure is reported, never silently swapped for the CPU path (the
 * regression lesson from 8eea2ff). Buffers come from the stream-ordered
 * pool and are freed after their last reader, so stages hand memory to
 * each other instead of holding every role at once. */

#include "h3_gpu.h"

typedef struct {
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
    /* WAN22_CONV_FAST: the same weight packed into the k-major [k][O] layout
     * the tiled conv3d reads.  The row-major weight is dropped once the pack
     * succeeds, so the decode's weight residency is unchanged; NULL here
     * means the reference kernel runs instead. */
    h3_gpu_tensor *weight_packed;
    uint32_t out;
    uint32_t in;
    uint8_t kt;
    uint8_t ks;
} wgpu_conv;

typedef struct {
    uint32_t in;
    uint32_t out;
    h3_gpu_tensor *gamma0;
    wgpu_conv a;
    h3_gpu_tensor *gamma1;
    wgpu_conv b;
    int has_shortcut;
    wgpu_conv sc;
} wgpu_resblock;

typedef struct {
    h3_gpu_tensor *gamma;
    wgpu_conv qkv;
    wgpu_conv proj;
} wgpu_attn;

typedef struct {
    uint32_t in;
    uint32_t out;
    wgpu_resblock res[VAE_RES_BLOCKS];
    int up_flag;
    int t_up;
    wgpu_conv spatial;
    wgpu_conv time_conv;
} wgpu_up;

struct wan22_vae_gpu_state {
    h3_gpu *gpu;
    wgpu_conv conv1;
    wgpu_conv conv2;
    wgpu_resblock middle[2];
    wgpu_attn attn;
    wgpu_up ups[VAE_NUM_BLOCKS];
    h3_gpu_tensor *head_gamma;
    wgpu_conv head;
};

static int wan22_vae_gpu_requested(void) {
    const char *value = getenv("WAN22_VAE_GPU");
    return value && value[0] == '1';
}

/* WAN22_CONV_FAST=0 falls back to the reference (memory-latency bound) causal
 * conv3d kernels; the tiled path is the default. */
static int wan22_conv_fast(void) {
    const char *value = getenv("WAN22_CONV_FAST");
    return !(value && value[0] == '0');
}

static void wgpu_conv_free(wgpu_conv *conv) {
    h3_gpu_tensor_free(conv->weight);
    h3_gpu_tensor_free(conv->weight_packed);
    h3_gpu_tensor_free(conv->bias);
    memset(conv, 0, sizeof(*conv));
}

static void wgpu_resblock_free(wgpu_resblock *rb) {
    h3_gpu_tensor_free(rb->gamma0);
    h3_gpu_tensor_free(rb->gamma1);
    wgpu_conv_free(&rb->a);
    wgpu_conv_free(&rb->b);
    if (rb->has_shortcut) wgpu_conv_free(&rb->sc);
    memset(rb, 0, sizeof(*rb));
}

static void wan22_vae_gpu_free(struct wan22_vae_gpu_state *state) {
    if (!state) return;
    wgpu_conv_free(&state->conv1);
    wgpu_conv_free(&state->conv2);
    for (int i = 0; i < 2; i++) wgpu_resblock_free(&state->middle[i]);
    h3_gpu_tensor_free(state->attn.gamma);
    wgpu_conv_free(&state->attn.qkv);
    wgpu_conv_free(&state->attn.proj);
    for (int i = 0; i < VAE_NUM_BLOCKS; i++) {
        wgpu_up *up = &state->ups[i];
        for (int j = 0; j < VAE_RES_BLOCKS; j++)
            wgpu_resblock_free(&up->res[j]);
        wgpu_conv_free(&up->spatial);
        if (up->t_up) wgpu_conv_free(&up->time_conv);
    }
    h3_gpu_tensor_free(state->head_gamma);
    wgpu_conv_free(&state->head);
    h3_gpu_free(state->gpu);
    free(state);
}

/* Host-side pack of [O][C][kt][ks][ks] into the k-major [k][O] layout the
 * tiled conv3d reads (k = ((c*kt + step)*ks + ky)*ks + kx).  The tiled path
 * only ever needs the packed layout, so packing here costs one weight upload
 * instead of two (row-major, then a device permute) and never has the old and
 * new copies of a decoder's weights device-resident at the same time - the
 * decode's 21 GiB budget is tight enough that the transient matters. */
static float *conv_pack_host(const vae_conv *src) {
    const size_t kk = (size_t)src->kt * src->ks * src->ks;
    const size_t ic = src->in, oc = src->out;
    float *packed = malloc(oc * ic * kk * sizeof(float));
    if (!packed) return NULL;
    const size_t cblk = 4u, oblk = 16u;
    for (size_t c0 = 0; c0 < ic; c0 += cblk) {
        const size_t cn = ic - c0 < cblk ? ic - c0 : cblk;
        for (size_t o0 = 0; o0 < oc; o0 += oblk) {
            const size_t on = oc - o0 < oblk ? oc - o0 : oblk;
            for (size_t t = 0; t < cn * kk; t++) {
                float *dst = packed + (c0 * kk + t) * oc + o0;
                for (size_t o = 0; o < on; o++)
                    dst[o] = src->weight[((o0 + o) * ic + c0) * kk + t];
            }
        }
    }
    return packed;
}

static int wgpu_conv_load(h3_gpu *gpu, wgpu_conv *dst, const vae_conv *src) {
    dst->out = src->out;
    dst->in = src->in;
    dst->kt = src->kt;
    dst->ks = src->ks;
    const size_t weight_count =
        (size_t)src->out * src->in * src->kt * src->ks * src->ks;
    if (wan22_conv_fast()) {
        float *packed = conv_pack_host(src);
        if (!packed) return 0;
        dst->weight_packed = h3_gpu_tensor_from_f32(gpu, packed, weight_count);
        free(packed);
        if (!dst->weight_packed) return 0;
    } else {
        dst->weight = h3_gpu_tensor_from_f32(gpu, src->weight, weight_count);
        if (!dst->weight) return 0;
    }
    if (src->bias) {
        dst->bias = h3_gpu_tensor_from_f32(gpu, src->bias, src->out);
        if (!dst->bias) return 0;
    }
    return 1;
}

static int wgpu_resblock_load(h3_gpu *gpu, wgpu_resblock *dst,
                              const vae_resblock *src) {
    dst->in = src->in;
    dst->out = src->out;
    dst->has_shortcut = src->has_shortcut;
    dst->gamma0 = h3_gpu_tensor_from_f32(gpu, src->gamma0, src->in);
    dst->gamma1 = h3_gpu_tensor_from_f32(gpu, src->gamma1, src->out);
    return dst->gamma0 && dst->gamma1 &&
           wgpu_conv_load(gpu, &dst->a, &src->a) &&
           wgpu_conv_load(gpu, &dst->b, &src->b) &&
           (!src->has_shortcut || wgpu_conv_load(gpu, &dst->sc, &src->sc));
}

static struct wan22_vae_gpu_state *wan22_vae_gpu_create(
    const h3_wan22_vae_decoder *decoder, char *error, size_t error_size) {
    struct wan22_vae_gpu_state *state = calloc(1, sizeof(*state));
    if (!state) {
        snprintf(error, error_size, "out of host memory (wan22 vae gpu)");
        return NULL;
    }
    state->gpu = h3_gpu_create(NULL, error, error_size);
    if (!state->gpu) {
        free(state);
        return NULL;
    }
    h3_gpu *gpu = state->gpu;
    int ok = wgpu_conv_load(gpu, &state->conv1, &decoder->conv1) &&
             wgpu_conv_load(gpu, &state->conv2, &decoder->conv2);
    for (int i = 0; ok && i < 2; i++)
        ok = wgpu_resblock_load(gpu, &state->middle[i], &decoder->middle[i]);
    if (ok) {
        state->attn.gamma =
            h3_gpu_tensor_from_f32(gpu, decoder->attn.gamma, decoder->attn.dim);
        ok = state->attn.gamma &&
             wgpu_conv_load(gpu, &state->attn.qkv, &decoder->attn.qkv) &&
             wgpu_conv_load(gpu, &state->attn.proj, &decoder->attn.proj);
    }
    for (int i = 0; ok && i < VAE_NUM_BLOCKS; i++) {
        wgpu_up *dst = &state->ups[i];
        const vae_up *src = &decoder->ups[i];
        dst->in = src->in;
        dst->out = src->out;
        dst->up_flag = src->up_flag;
        dst->t_up = src->t_up;
        for (int j = 0; ok && j < VAE_RES_BLOCKS; j++)
            ok = wgpu_resblock_load(gpu, &dst->res[j], &src->res[j]);
        if (ok && src->up_flag)
            ok = wgpu_conv_load(gpu, &dst->spatial, &src->spatial);
        if (ok && src->t_up)
            ok = wgpu_conv_load(gpu, &dst->time_conv, &src->time_conv);
    }
    if (ok) {
        state->head_gamma =
            h3_gpu_tensor_from_f32(gpu, decoder->head_gamma, 256);
        ok = state->head_gamma && wgpu_conv_load(gpu, &state->head, &decoder->head);
    }
    if (!ok) {
        if (error && error_size && !error[0])
            snprintf(error, error_size,
                     "wan22 vae gpu weight upload failed: %s",
                     h3_gpu_error(gpu));
        wan22_vae_gpu_free(state);
        return NULL;
    }
    return state;
}

static int wg_check(int result, const char *what, h3_gpu *gpu, char *error,
                    size_t error_size) {
    if (!result)
        snprintf(error, error_size, "wan22 vae gpu decode failed: %s: %s",
                 what, h3_gpu_error(gpu));
    return result;
}

static int wg_conv(h3_gpu *gpu, h3_gpu_tensor *output,
                   const h3_gpu_tensor *input, const wgpu_conv *conv,
                   uint32_t t, uint32_t h, uint32_t w) {
    if (conv->weight_packed)
        return h3_gpu_cconv3d_packed_f32(
            gpu, output, input, conv->weight_packed, conv->bias, conv->out,
            conv->in, t, h, w, conv->kt, conv->ks);
    return h3_gpu_cconv3d_f32(gpu, output, input, conv->weight, conv->bias,
                              conv->out, conv->in, t, h, w, conv->kt,
                              conv->ks);
}

/* ResidualBlock: RMS-SiLU -> conv a -> RMS-SiLU -> conv b, plus the
 * (1x1x1) shortcut or the identity. x is read-only; out is written. */
static int wg_resblock_forward(h3_gpu *gpu, const wgpu_resblock *rb,
                               h3_gpu_tensor *x, h3_gpu_tensor *out,
                               uint32_t t, uint32_t h, uint32_t w,
                               char *error, size_t error_size) {
    const size_t in_count = (size_t)rb->in * t * h * w;
    const size_t out_count = (size_t)rb->out * t * h * w;
    h3_gpu_tensor *shortcut = NULL;
    h3_gpu_tensor *hidden = NULL;
    h3_gpu_tensor *main = NULL;
    h3_gpu_tensor *main2 = NULL;
    int ok = 1;
    if (rb->has_shortcut) {
        shortcut = h3_gpu_tensor_new_f32(gpu, out_count);
        ok = shortcut && wg_conv(gpu, shortcut, x, &rb->sc, t, h, w);
        if (!ok) wg_check(0, "resblock.shortcut", gpu, error, error_size);
    }
    if (ok) {
        hidden = h3_gpu_tensor_new_f32(gpu, in_count);
        ok = hidden && wg_check(
            h3_gpu_rms_norm_silu_f32(gpu, hidden, x, rb->gamma0, rb->in, t, h,
                                     w, 1),
            "resblock.norm1", gpu, error, error_size);
    }
    if (ok) {
        main = h3_gpu_tensor_new_f32(gpu, out_count);
        ok = main && wg_check(wg_conv(gpu, main, hidden, &rb->a, t, h, w),
                              "resblock.conv_a", gpu, error, error_size);
    }
    if (ok) {
        main2 = h3_gpu_tensor_new_f32(gpu, out_count);
        ok = main2 && wg_check(
            h3_gpu_rms_norm_silu_f32(gpu, main2, main, rb->gamma1, rb->out, t,
                                     h, w, 1),
            "resblock.norm2", gpu, error, error_size);
    }
    if (ok) {
        ok = wg_check(wg_conv(gpu, out, main2, &rb->b, t, h, w),
                      "resblock.conv_b", gpu, error, error_size);
    }
    if (ok) {
        const h3_gpu_tensor *residual = rb->has_shortcut ? shortcut : x;
        ok = wg_check(
            h3_gpu_add_f32(gpu, out, out, residual, (uint32_t)out_count),
            "resblock.add", gpu, error, error_size);
    }
    h3_gpu_tensor_free(shortcut);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(main);
    h3_gpu_tensor_free(main2);
    return ok;
}

/* AttentionBlock, in place on x [c][t][h][w]: channel RMSNorm -> 1x1x1 qkv
 * -> per-frame single-head spatial SDPA -> 1x1x1 proj -> residual. */
static int wg_attention_forward(h3_gpu *gpu, const wgpu_attn *attn,
                                h3_gpu_tensor *x, uint32_t t, uint32_t h,
                                uint32_t w, char *error, size_t error_size) {
    const uint32_t c = attn->qkv.in;
    const size_t plane = (size_t)h * w;
    const size_t count = (size_t)c * t * plane;
    h3_gpu_tensor *normed = h3_gpu_tensor_new_f32(gpu, count);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_f32(gpu, count * 3u);
    h3_gpu_tensor *q = h3_gpu_tensor_new_f32(gpu, count);
    h3_gpu_tensor *k = h3_gpu_tensor_new_f32(gpu, count);
    h3_gpu_tensor *v = h3_gpu_tensor_new_f32(gpu, count);
    h3_gpu_tensor *scores =
        h3_gpu_tensor_new_f32(gpu, (size_t)t * plane * plane);
    h3_gpu_tensor *attended = h3_gpu_tensor_new_f32(gpu, count);
    h3_gpu_tensor *proj_in = h3_gpu_tensor_new_f32(gpu, count);
    h3_gpu_tensor *proj_out = h3_gpu_tensor_new_f32(gpu, count);
    int ok = normed && qkv && q && k && v && scores && attended && proj_in &&
             proj_out;
    if (ok) {
        ok = wg_check(
            h3_gpu_rms_norm_silu_f32(gpu, normed, x, attn->gamma, c, t, h, w,
                                     0),
            "attn.norm", gpu, error, error_size);
    }
    if (ok) {
        ok = wg_check(wg_conv(gpu, qkv, normed, &attn->qkv, t, h, w),
                      "attn.qkv", gpu, error, error_size);
    }
    if (ok) {
        ok = wg_check(
            h3_gpu_vae_qkv_pack_f32(gpu, q, k, v, qkv, c, t, (uint32_t)plane),
            "attn.pack", gpu, error, error_size);
    }
    if (ok) {
        ok = wg_check(h3_gpu_vae_sdpa3d_f32(gpu, attended, q, k, v, scores, t,
                                            (uint32_t)plane, c,
                                            1.0f / sqrtf((float)c)),
                      "attn.sdpa", gpu, error, error_size);
    }
    if (ok) {
        ok = wg_check(
            h3_gpu_vae_attn_unpack_f32(gpu, proj_in, attended, c, t,
                                       (uint32_t)plane),
            "attn.unpack", gpu, error, error_size);
    }
    if (ok) {
        ok = wg_check(wg_conv(gpu, proj_out, proj_in, &attn->proj, t, h, w),
                      "attn.proj", gpu, error, error_size);
    }
    if (ok) {
        ok = wg_check(h3_gpu_add_f32(gpu, x, x, proj_out, (uint32_t)count),
                      "attn.add", gpu, error, error_size);
    }
    h3_gpu_tensor_free(normed);
    h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(k);
    h3_gpu_tensor_free(v);
    h3_gpu_tensor_free(scores);
    h3_gpu_tensor_free(attended);
    h3_gpu_tensor_free(proj_in);
    h3_gpu_tensor_free(proj_out);
    return ok;
}

/* One upsample block; consumes x and returns [out][t_main][h2][w2]
 * (t_main = 2t-1 when t_up, else t; h2 = 2h when up_flag, else h). */
static h3_gpu_tensor *wg_up_block_forward(h3_gpu *gpu, const wgpu_up *up,
                                          h3_gpu_tensor *x, uint32_t t,
                                          uint32_t h, uint32_t w, char *error,
                                          size_t error_size) {
    h3_gpu_tensor *act = x; /* borrowed until the first resblock output */
    int act_owned = 0;
    for (int j = 0; j < VAE_RES_BLOCKS; j++) {
        h3_gpu_tensor *next =
            h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t * h * w);
        if (!next || !wg_resblock_forward(gpu, &up->res[j], act, next, t, h, w,
                                          error, error_size)) {
            h3_gpu_tensor_free(next);
            if (act_owned) h3_gpu_tensor_free(act);
            h3_gpu_tensor_free(x);
            return NULL;
        }
        if (act_owned) h3_gpu_tensor_free(act);
        act = next;
        act_owned = 1;
    }
    if (!up->up_flag) {
        h3_gpu_tensor_free(x);
        return act; /* block 3: residual blocks only */
    }
    const uint32_t h2 = h * 2u;
    const uint32_t w2 = w * 2u;
    const uint32_t t_main =
        up->t_up ? (t > 1 ? 2u * t - 1u : 1u) : t;
    h3_gpu_tensor *src = act; /* [out][t_main][h][w] going into nearest2x */
    h3_gpu_tensor *owned_src = NULL;
    if (up->t_up && t > 1) {
        /* Temporal path: y = causal time_conv over the full R (its frame
         * (j+1)/2 equals the slice conv R[1:] frame j/2); frame 0 keeps the
         * spatial-only path, frames 1.. interleave the two channel halves. */
        h3_gpu_tensor *shifted =
            h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t * h * w);
        h3_gpu_tensor *y =
            h3_gpu_tensor_new_f32(gpu, (size_t)up->out * 2u * t * h * w);
        h3_gpu_tensor *merged =
            h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h * w);
        if (!shifted || !y || !merged ||
            !wg_check(h3_gpu_frame_shift_f32(gpu, shifted, act, up->out, t,
                                             (uint32_t)(h * w)),
                      "up.frame_shift", gpu, error, error_size) ||
            !wg_check(wg_conv(gpu, y, shifted, &up->time_conv, t, h, w),
                      "up.time_conv", gpu, error, error_size) ||
            !wg_check(h3_gpu_vae_temporal_merge_f32(
                          gpu, merged, act, y, up->out, t, (uint32_t)(h * w)),
                      "up.temporal_merge", gpu, error, error_size)) {
            h3_gpu_tensor_free(shifted);
            h3_gpu_tensor_free(y);
            h3_gpu_tensor_free(merged);
            h3_gpu_tensor_free(act);
            h3_gpu_tensor_free(x);
            return NULL;
        }
        h3_gpu_tensor_free(shifted);
        h3_gpu_tensor_free(y);
        h3_gpu_tensor_free(act);
        act = NULL; /* consumed by the merge */
        src = owned_src = merged;
    }
    h3_gpu_tensor *upsampled =
        h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h2 * w2);
    if (!upsampled ||
        !wg_check(h3_gpu_upsample_nearest2x_ncthw_f32(gpu, upsampled, src,
                                                      up->out, t_main, h, w),
                  "up.upsample", gpu, error, error_size)) {
        h3_gpu_tensor_free(upsampled);
        h3_gpu_tensor_free(owned_src);
        h3_gpu_tensor_free(act);
        h3_gpu_tensor_free(x);
        return NULL;
    }
    h3_gpu_tensor_free(owned_src);
    h3_gpu_tensor_free(act);
    h3_gpu_tensor *main =
        h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h2 * w2);
    if (!main ||
        !wg_check(wg_conv(gpu, main, upsampled, &up->spatial, t_main, h2, w2),
                  "up.spatial", gpu, error, error_size)) {
        h3_gpu_tensor_free(upsampled);
        h3_gpu_tensor_free(main);
        h3_gpu_tensor_free(x);
        return NULL;
    }
    h3_gpu_tensor_free(upsampled);
    /* DupUp3d shortcut from the block input. */
    h3_gpu_tensor *shortcut =
        h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h2 * w2);
    if (!shortcut ||
        !wg_check(h3_gpu_dupup_f32(gpu, shortcut, x, up->in, up->out,
                                   up->t_up ? 2u : 1u, 2u, up->t_up, t, h, w),
                  "up.dupup", gpu, error, error_size)) {
        h3_gpu_tensor_free(shortcut);
        h3_gpu_tensor_free(main);
        h3_gpu_tensor_free(x);
        return NULL;
    }
    h3_gpu_tensor_free(x);
    if (!wg_check(h3_gpu_add_f32(gpu, main, main, shortcut,
                                 (uint32_t)((size_t)up->out * t_main * h2 * w2)),
                  "up.add", gpu, error, error_size)) {
        h3_gpu_tensor_free(shortcut);
        h3_gpu_tensor_free(main);
        return NULL;
    }
    h3_gpu_tensor_free(shortcut);
    return main;
}

static int wan22_vae_decode_gpu(h3_wan22_vae_decoder *decoder,
                                struct wan22_vae_gpu_state *state,
                                const float *z, uint32_t frames, uint32_t h,
                                uint32_t w, float *out, char *error,
                                size_t error_size) {
    h3_gpu *gpu = state->gpu;
    const size_t plane = (size_t)frames * h * w;
    h3_gpu_tensor *act = NULL;
    h3_gpu_tensor *next = NULL;
    h3_gpu_tensor *out_dev = NULL;
    int ok = 1;
    /* WanVAE_.decode: z * std + mean per channel, applied on the host (48 x
     * plane elements, negligible). */
    float *staged = malloc(VAE_Z_DIM * plane * sizeof(float));
    if (!staged) {
        snprintf(error, error_size, "out of host memory (wan22 vae gpu decode)");
        return 0;
    }
    for (uint32_t c = 0; c < VAE_Z_DIM; c++) {
        const float mean = decoder->mean[c];
        const float std = decoder->std[c];
        const float *src = z + (size_t)c * plane;
        float *dst = staged + (size_t)c * plane;
        for (size_t i = 0; i < plane; i++) dst[i] = src[i] * std + mean;
    }
    act = h3_gpu_tensor_new_f32(gpu, VAE_Z_DIM * plane);
    ok = act && h3_gpu_tensor_write_f32(act, staged, VAE_Z_DIM * plane);
    if (!ok) wg_check(0, "latent upload", gpu, error, error_size);
    free(staged);
    if (ok) ok = wg_check(h3_gpu_begin(gpu), "begin", gpu, error, error_size);

    if (ok) {
        next = h3_gpu_tensor_new_f32(gpu, VAE_Z_DIM * plane);
        ok = next && wg_check(wg_conv(gpu, next, act, &state->conv2, frames, h,
                                      w),
                              "conv2", gpu, error, error_size);
        h3_gpu_tensor_free(act);
        act = next;
    }
    if (ok) {
        next = h3_gpu_tensor_new_f32(gpu, 1024 * plane);
        ok = next && wg_check(wg_conv(gpu, next, act, &state->conv1, frames, h,
                                      w),
                              "conv1", gpu, error, error_size);
        h3_gpu_tensor_free(act);
        act = next;
    }
    for (int i = 0; ok && i < 2; i++) {
        next = h3_gpu_tensor_new_f32(gpu, 1024 * plane);
        ok = next && wg_resblock_forward(gpu, &state->middle[i], act, next,
                                         frames, h, w, error, error_size);
        h3_gpu_tensor_free(act);
        act = next;
        if (ok && i == 0) {
            ok = wg_attention_forward(gpu, &state->attn, act, frames, h, w,
                                      error, error_size);
        }
    }
    uint32_t t_cur = frames;
    uint32_t h_cur = h;
    uint32_t w_cur = w;
    for (int i = 0; ok && i < VAE_NUM_BLOCKS; i++) {
        h3_gpu_tensor *block_out =
            wg_up_block_forward(gpu, &state->ups[i], act, t_cur, h_cur, w_cur,
                                error, error_size);
        act = block_out;
        if (!act) {
            ok = 0;
            break;
        }
        if (decoder->ups[i].t_up) t_cur = 2u * t_cur - 1u;
        if (decoder->ups[i].up_flag) {
            h_cur *= 2u;
            w_cur *= 2u;
        }
    }
    if (ok) {
        h3_gpu_tensor *head_in =
            h3_gpu_tensor_new_f32(gpu, (size_t)256 * t_cur * h_cur * w_cur);
        h3_gpu_tensor *head_out =
            h3_gpu_tensor_new_f32(gpu, (size_t)12 * t_cur * h_cur * w_cur);
        out_dev =
            h3_gpu_tensor_new_f32(gpu, (size_t)12 * t_cur * h_cur * w_cur);
        ok = head_in && head_out && out_dev;
        if (ok) {
            ok = wg_check(h3_gpu_rms_norm_silu_f32(
                              gpu, head_in, act, state->head_gamma, 256,
                              t_cur, h_cur, w_cur, 1),
                          "head.norm", gpu, error, error_size);
        }
        if (ok) {
            ok = wg_check(wg_conv(gpu, head_out, head_in, &state->head, t_cur,
                                  h_cur, w_cur),
                          "head.conv", gpu, error, error_size);
        }
        if (ok) {
            ok = wg_check(h3_gpu_unpatchify_f32(gpu, out_dev, head_out, t_cur,
                                                h_cur, w_cur),
                          "head.unpatchify", gpu, error, error_size);
        }
        h3_gpu_tensor_free(head_in);
        h3_gpu_tensor_free(head_out);
    }
    if (ok) ok = wg_check(h3_gpu_submit(gpu), "submit", gpu, error, error_size);
    if (ok && !h3_gpu_tensor_read_f32(out_dev, out,
                                      (size_t)12 * t_cur * h_cur * w_cur)) {
        snprintf(error, error_size, "wan22 vae gpu download failed: %s",
                 h3_gpu_error(gpu));
        ok = 0;
    }
    h3_gpu_tensor_free(act);
    h3_gpu_tensor_free(out_dev);
    return ok;
}

/* --- streaming GPU decode (feat_cache) ------------------------------------
 * The whole-sequence decode above is equivalent to the reference's per-frame
 * feat_cache streaming, and the decoder is causal: output frame f depends on
 * latent frames [~f/4 - 37, ~f/4 + 1].  Restart-chunked decoding therefore
 * needs a ~37 latent-frame halo per chunk, whose activations alone exceed a
 * 24 GB card at 720p.  This path instead carries the decoder state across
 * groups of G latent frames: every kt=3 temporal conv keeps the last two
 * frames of its input sequence in a device tail and reads the tail instead
 * of zero-padding for taps before the group.  Tail frames are exact copies
 * of previously produced tensors, so each group's outputs are bit-identical
 * to the single-segment decode while activation memory scales with G, not
 * the clip length.  Tails are updated only after all same-group readers,
 * so a conv always sees the previous group's state. */

typedef struct {
    h3_gpu_tensor *tail;      /* current [ch][2][plane], last two frames */
    h3_gpu_tensor *tail_next; /* scratch target of the next update */
    uint32_t ch;
    size_t plane;             /* h * w at this role's resolution */
    uint32_t frames_seen;     /* global frames already folded in */
} wg_tail;


static int wg_tail_create(h3_gpu *gpu, wg_tail *t, uint32_t ch, size_t plane) {
    t->ch = ch;
    t->plane = plane;
    t->frames_seen = 0u;
    t->tail = h3_gpu_tensor_new_f32(gpu, (size_t)ch * 2u * plane);
    t->tail_next = h3_gpu_tensor_new_f32(gpu, (size_t)ch * 2u * plane);
    return t->tail != NULL && t->tail_next != NULL;
}

static void wg_tail_free(wg_tail *t) {
    h3_gpu_tensor_free(t->tail);
    h3_gpu_tensor_free(t->tail_next);
    memset(t, 0, sizeof(*t));
}

/* Keep the last two frames of the global input sequence per channel.  Slots
 * reaching before the sequence start stay zero (matching the kernel's zero
 * pad); the update reads the previous tail and the group tensor and writes
 * the scratch tensor, then the two are swapped. */
static int wg_tail_update_ex(h3_gpu *gpu, wg_tail *t, const h3_gpu_tensor *out,
                             uint32_t t_g, int reset);

static int wg_tail_update(h3_gpu *gpu, wg_tail *t, const h3_gpu_tensor *out,
                          uint32_t t_g) {
    return wg_tail_update_ex(gpu, t, out, t_g, 0);
}

/* reset=1 starts the tracked sequence fresh (the time_conv's shifted input
 * begins with a zero placeholder at the clip start). */
static int wg_tail_update_ex(h3_gpu *gpu, wg_tail *t, const h3_gpu_tensor *out,
                             uint32_t t_g, int reset) {
    const uint32_t seen = reset ? 0u : t->frames_seen;
    const int ok = h3_gpu_tail_update_f32(gpu, t->tail_next, t->tail, out,
                                          t->ch, t_g, t->plane, seen);
    if (ok) {
        h3_gpu_tensor *swap = t->tail;
        t->tail = t->tail_next;
        t->tail_next = swap;
        t->frames_seen = seen + t_g;
    }
    return ok;
}

static int wg_conv_stream(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input, const wg_tail *tail,
                          const wgpu_conv *conv, uint32_t t, uint32_t h,
                          uint32_t w, int first_group, char *error,
                          size_t error_size) {
    if (first_group || conv->kt < 3u)
        return wg_check(wg_conv(gpu, output, input, conv, t, h, w),
                        "stream conv", gpu, error, error_size);
    if (conv->weight_packed)
        return wg_check(h3_gpu_cconv3d_packed_cached_f32(
                            gpu, output, input, tail->tail, 2u,
                            conv->weight_packed, conv->bias, conv->out,
                            conv->in, t, h, w, conv->kt, conv->ks),
                        "stream cached conv", gpu, error, error_size);
    return wg_check(h3_gpu_cconv3d_cached_f32(
                        gpu, output, input, tail->tail, 2u, conv->weight,
                        conv->bias, conv->out, conv->in, t, h, w, conv->kt,
                        conv->ks),
                    "stream cached conv", gpu, error, error_size);
}

/* ResidualBlock with carried conv state; tail_a / tail_b are read by the
 * convs (previous group's normed inputs) and refreshed after each conv. */
static int wg_resblock_stream(h3_gpu *gpu, const wgpu_resblock *rb,
                              h3_gpu_tensor *x, h3_gpu_tensor *out,
                              wg_tail *tail_a, wg_tail *tail_b, uint32_t t,
                              uint32_t h, uint32_t w, int first_group,
                              char *error, size_t error_size) {
    const size_t in_count = (size_t)rb->in * t * h * w;
    const size_t out_count = (size_t)rb->out * t * h * w;
    h3_gpu_tensor *shortcut = NULL;
    h3_gpu_tensor *hidden = NULL;
    h3_gpu_tensor *main = NULL;
    h3_gpu_tensor *main2 = NULL;
    int ok = 1;
    if (rb->has_shortcut) {
        shortcut = h3_gpu_tensor_new_f32(gpu, out_count);
        ok = shortcut && wg_conv(gpu, shortcut, x, &rb->sc, t, h, w);
        if (!ok)
            wg_check(0, "stream resblock.shortcut", gpu, error, error_size);
    }
    if (ok) {
        hidden = h3_gpu_tensor_new_f32(gpu, in_count);
        ok = hidden && wg_check(
            h3_gpu_rms_norm_silu_f32(gpu, hidden, x, rb->gamma0, rb->in, t, h,
                                     w, 1),
            "stream resblock.norm1", gpu, error, error_size);
    }
    if (ok) {
        main = h3_gpu_tensor_new_f32(gpu, out_count);
        ok = main && wg_conv_stream(gpu, main, hidden, tail_a, &rb->a, t, h,
                                    w, first_group, error, error_size);
        if (!ok)
            wg_check(0, "stream resblock.conv_a", gpu, error, error_size);
    }
    if (ok) ok = wg_tail_update(gpu, tail_a, hidden, t);
    h3_gpu_tensor_free(hidden);
    hidden = NULL;
    if (ok) {
        main2 = h3_gpu_tensor_new_f32(gpu, out_count);
        ok = main2 && wg_check(
            h3_gpu_rms_norm_silu_f32(gpu, main2, main, rb->gamma1, rb->out, t,
                                     h, w, 1),
            "stream resblock.norm2", gpu, error, error_size);
    }
    h3_gpu_tensor_free(main);
    main = NULL;
    if (ok) {
        ok = wg_conv_stream(gpu, out, main2, tail_b, &rb->b, t, h, w,
                            first_group, error, error_size);
        if (!ok)
            wg_check(0, "stream resblock.conv_b", gpu, error, error_size);
    }
    if (ok) ok = wg_tail_update(gpu, tail_b, main2, t);
    h3_gpu_tensor_free(main2);
    main2 = NULL;
    if (ok) {
        const h3_gpu_tensor *residual = rb->has_shortcut ? shortcut : x;
        ok = wg_check(
            h3_gpu_add_f32(gpu, out, out, residual, (uint32_t)out_count),
            "stream resblock.add", gpu, error, error_size);
    }
    h3_gpu_tensor_free(shortcut);
    return ok;
}

/* One upsample block over a group.  Returns [out][t_main][h2][w2] and sets
 * t_main (2t-1 for the first group's t_up block, 2t for continuations, t
 * without t_up).  res_tails are the six per-conv tails; tail_r is the
 * pre-upsample output tail consumed by this block's own time_conv (NULL for
 * blocks whose consumers live outside). */
static h3_gpu_tensor *wg_up_block_stream(h3_gpu *gpu, const wgpu_up *up,
                                         h3_gpu_tensor *x,
                                         wg_tail **res_tails, wg_tail *tail_r,
                                         uint32_t t, uint32_t h, uint32_t w,
                                         int first_group, uint32_t *t_out,
                                         char *error, size_t error_size) {
    h3_gpu_tensor *act = x;
    for (int j = 0; j < VAE_RES_BLOCKS; j++) {
        h3_gpu_tensor *next =
            h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t * h * w);
        if (!next || !wg_resblock_stream(gpu, &up->res[j], act, next,
                                         res_tails[2 * j], res_tails[2 * j + 1],
                                         t, h, w, first_group, error,
                                         error_size)) {
            h3_gpu_tensor_free(next);
            if (act != x) h3_gpu_tensor_free(act);
            h3_gpu_tensor_free(x);
            return NULL;
        }
        if (act != x) h3_gpu_tensor_free(act);
        act = next;
    }
    if (!up->up_flag) {
        *t_out = t;
        /* Block 3 has no resample and returns here, above the resample path
         * that consumes and frees the block's input for the other blocks. The
         * group loop only frees the tensor the last block returns, so x used
         * to leak once per group, sized by block 2's spatial output: 128 MiB
         * for the 512p first group, 512 MiB for the rest, 1.9 GiB at 720p.
         * That growth is what drove the pool into the 24 GiB ceiling. */
        h3_gpu_tensor_free(x);
        return act; /* block 3: residual blocks only */
    }
    uint32_t t_main = t;
    h3_gpu_tensor *merged = NULL;
    if (up->t_up && !(first_group && t == 1u)) {
        /* Temporal resample.  First group: time_conv over the frame-shifted
         * group with the plain zero pad (reference first_chunk); the carried
         * tail tracks the shifted sequence, which gains [0, R[1..t-1]].
         * Later groups: time_conv over the group with the carried shifted
         * tail (the group's frames extend it by R[]); outputs continue the
         * global interleave at frame 2*g0 - 1. */
        h3_gpu_tensor *y =
            h3_gpu_tensor_new_f32(gpu, (size_t)up->out * 2u * t * h * w);
        h3_gpu_tensor *shifted = NULL;
        int ok = y != NULL;
        if (ok && first_group) {
            shifted =
                h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t * h * w);
            ok = shifted &&
                 wg_check(h3_gpu_frame_shift_f32(gpu, shifted, act, up->out,
                                                 t, (size_t)(h * w)),
                          "stream up.frame_shift", gpu, error, error_size);
        }
        if (ok) {
            if (first_group) {
                ok = wg_check(wg_conv(gpu, y, shifted, &up->time_conv, t, h,
                                      w),
                              "stream up.time_conv", gpu, error, error_size);
            } else if (up->time_conv.weight_packed) {
                ok = wg_check(
                    h3_gpu_cconv3d_packed_cached_f32(
                        gpu, y, act, tail_r->tail, 2u,
                        up->time_conv.weight_packed, up->time_conv.bias,
                        up->time_conv.out, up->time_conv.in, t, h, w, 3u,
                        1u),
                    "stream up.time_conv cached", gpu, error, error_size);
            } else {
                ok = wg_check(
                    h3_gpu_cconv3d_cached_f32(
                        gpu, y, act, tail_r->tail, 2u, up->time_conv.weight,
                        up->time_conv.bias, up->time_conv.out,
                        up->time_conv.in, t, h, w, 3u, 1u),
                    "stream up.time_conv cached", gpu, error, error_size);
            }
        }
        /* the time_conv above read the previous shifted-sequence tail;
         * refresh it with this group's shifted-sequence frames */
        if (ok && tail_r) {
            ok = first_group
                     ? wg_tail_update_ex(gpu, tail_r, shifted, t, 1)
                     : wg_tail_update_ex(gpu, tail_r, act, t, 0);
        }
        h3_gpu_tensor_free(shifted);
        if (ok && first_group) {
            t_main = 2u * t - 1u;
            merged = h3_gpu_tensor_new_f32(gpu,
                                           (size_t)up->out * t_main * h * w);
            ok = merged &&
                 wg_check(h3_gpu_vae_temporal_merge_f32(
                              gpu, merged, act, y, up->out, t,
                              (size_t)(h * w)),
                          "stream up.merge", gpu, error, error_size);
        } else if (ok) {
            t_main = 2u * t;
            merged =
                h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h * w);
            ok = merged &&
                 wg_check(h3_gpu_vae_temporal_merge_cont_f32(
                              gpu, merged, y, up->out, t, (size_t)(h * w)),
                          "stream up.merge cont", gpu, error, error_size);
        }
        h3_gpu_tensor_free(y);
        if (!ok) {
            h3_gpu_tensor_free(merged);
            h3_gpu_tensor_free(act);
            h3_gpu_tensor_free(x);
            return NULL;
        }
        h3_gpu_tensor_free(act);
        act = merged;
    } else if (up->t_up) {
        t_main = 1u; /* first group, single latent frame: no time_conv */
        if (tail_r && !wg_tail_update_ex(gpu, tail_r, act, 0u, 1)) {
            h3_gpu_tensor_free(act);
            h3_gpu_tensor_free(x);
            return NULL;
        }
    }
    const uint32_t h2 = h * 2u;
    const uint32_t w2 = w * 2u;
    h3_gpu_tensor *upsampled =
        h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h2 * w2);
    if (!upsampled ||
        !wg_check(h3_gpu_upsample_nearest2x_ncthw_f32(gpu, upsampled, act,
                                                      up->out, t_main, h, w),
                  "stream up.upsample", gpu, error, error_size)) {
        h3_gpu_tensor_free(upsampled);
        h3_gpu_tensor_free(act);
        h3_gpu_tensor_free(x);
        return NULL;
    }
    h3_gpu_tensor_free(act);
    h3_gpu_tensor *main =
        h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h2 * w2);
    if (!main ||
        !wg_check(wg_conv(gpu, main, upsampled, &up->spatial, t_main, h2, w2),
                  "stream up.spatial", gpu, error, error_size)) {
        h3_gpu_tensor_free(upsampled);
        h3_gpu_tensor_free(main);
        h3_gpu_tensor_free(x);
        return NULL;
    }
    h3_gpu_tensor_free(upsampled);
    h3_gpu_tensor *shortcut =
        h3_gpu_tensor_new_f32(gpu, (size_t)up->out * t_main * h2 * w2);
    if (!shortcut ||
        !wg_check(h3_gpu_dupup_f32(gpu, shortcut, x, up->in, up->out,
                                   up->t_up ? 2u : 1u, 2u,
                                   up->t_up ? (first_group ? 1 : 0) : 0, t, h,
                                   w),
                  "stream up.dupup", gpu, error, error_size)) {
        h3_gpu_tensor_free(shortcut);
        h3_gpu_tensor_free(main);
        h3_gpu_tensor_free(x);
        return NULL;
    }
    h3_gpu_tensor_free(x);
    if (!wg_check(h3_gpu_add_f32(
                      gpu, main, main, shortcut,
                      (uint32_t)((size_t)up->out * t_main * h2 * w2)),
                  "stream up.add", gpu, error, error_size)) {
        h3_gpu_tensor_free(shortcut);
        h3_gpu_tensor_free(main);
        return NULL;
    }
    h3_gpu_tensor_free(shortcut);
    *t_out = t_main;
    return main;
}

static int wan22_vae_stream_requested(uint32_t frames) {
    const char *forced = getenv("WAN22_VAE_STREAM");
    if (forced && forced[0] == '1') return 1;
    if (forced && forced[0] == '0') return 0;
    /* The single-segment decode holds several activations per video frame
     * that scale with the latent area; past a handful of latent frames that
     * no longer fits a 24 GB card at 720p, so hand longer clips to the
     * streaming path (whose memory is group-bounded). */
    uint32_t threshold = 4u;
    const char *env = getenv("WAN22_VAE_STREAM_T");
    if (env && env[0]) threshold = (uint32_t)atoi(env);
    return frames > threshold;
}

/* Diagnostic for the group loop: H3_WAN22_STREAM_PROFILE=1 prints one line per
 * group with wall time, the conv share of it (needs H3_PROFILE for the event
 * pair accounting), the bytes the group allocated, and the pool's used /
 * reserved watermarks. That is enough to tell a slow allocator from slow
 * kernels without a profiler. */
static int wan22_vae_stream_profile(void) {
    const char *env = getenv("H3_WAN22_STREAM_PROFILE");
    return env && env[0] == '1';
}

static double wan22_vae_stream_now(void) {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t wan22_vae_stream_group_size(void) {
    const char *env = getenv("WAN22_VAE_STREAM_G");
    if (env && env[0]) {
        const uint32_t g = (uint32_t)atoi(env);
        if (g) return g;
    }
    return 1u;
}

static int wan22_vae_decode_gpu_stream(h3_wan22_vae_decoder *decoder,
                                       struct wan22_vae_gpu_state *state,
                                       const float *z, uint32_t frames,
                                       uint32_t h, uint32_t w, float *out,
                                       char *error, size_t error_size) {
    h3_gpu *gpu = state->gpu;
    const uint32_t G = wan22_vae_stream_group_size();
    const size_t plane_latent = (size_t)h * w;
    const size_t out_plane = (size_t)3u * 16u * h * 16u * w;
    int ok = 1;

    /* Carried state: the last two input frames of every kt=3 temporal conv.
     * Stage-boundary tails are shared between producers and consumers. */
    wg_tail t_conv2out, t_mid00, t_mid01, t_mid10, t_mid11;
    wg_tail t_b0[6], t_R0, t_A1;
    wg_tail t_b1[5], t_R1, t_A2;
    wg_tail t_b2[5], t_B2;
    wg_tail t_b3[5], t_B3;
    wg_tail *b0_heads[6] = {&t_b0[0], &t_b0[1], &t_b0[2],
                            &t_b0[3], &t_b0[4], &t_b0[5]};
    wg_tail *b1_heads[6] = {&t_A1,    &t_b1[0], &t_b1[1],
                            &t_b1[2], &t_b1[3], &t_b1[4]};
    wg_tail *b2_heads[6] = {&t_A2,    &t_b2[0], &t_b2[1],
                            &t_b2[2], &t_b2[3], &t_b2[4]};
    wg_tail *b3_heads[6] = {&t_B2,    &t_b3[0], &t_b3[1],
                            &t_b3[2], &t_b3[3], &t_b3[4]};
#define WG_TAIL(v, ch, pl)                                              \
    do {                                                                \
        if (ok && !wg_tail_create(gpu, &(v), (ch), (pl))) {             \
            snprintf(error, error_size,                                 \
                     "out of device memory (wan22 vae stream tails)");  \
            ok = 0;                                                     \
        }                                                               \
    } while (0)
    WG_TAIL(t_conv2out, 48u, plane_latent);
    WG_TAIL(t_mid00, 1024u, plane_latent);
    WG_TAIL(t_mid01, 1024u, plane_latent);
    WG_TAIL(t_mid10, 1024u, plane_latent);
    WG_TAIL(t_mid11, 1024u, plane_latent);
    for (int i = 0; i < 6; i++) WG_TAIL(t_b0[i], 1024u, plane_latent);
    WG_TAIL(t_R0, 1024u, plane_latent);
    WG_TAIL(t_A1, 1024u, plane_latent * 4u);
    for (int i = 0; i < 5; i++) WG_TAIL(t_b1[i], 1024u, plane_latent * 4u);
    WG_TAIL(t_R1, 1024u, plane_latent * 4u);
    WG_TAIL(t_A2, 1024u, plane_latent * 16u);
    for (int i = 0; i < 5; i++) WG_TAIL(t_b2[i], 512u, plane_latent * 16u);
    WG_TAIL(t_B2, 512u, plane_latent * 64u);
    for (int i = 0; i < 5; i++) WG_TAIL(t_b3[i], 256u, plane_latent * 64u);
    WG_TAIL(t_B3, 256u, plane_latent * 64u);
#undef WG_TAIL

    float *stage =
        ok ? malloc((size_t)VAE_Z_DIM * G * plane_latent * sizeof(float))
           : NULL;
    if (ok && !stage) {
        snprintf(error, error_size, "out of host memory (wan22 vae stream)");
        ok = 0;
    }
    if (ok)
        ok = wg_check(h3_gpu_begin(gpu), "stream begin", gpu, error,
                      error_size);

    h3_gpu_tensor *zg = NULL;
    h3_gpu_tensor *act = NULL;
    const int profile = wan22_vae_stream_profile();
    for (uint32_t g0 = 0; ok && g0 < frames; g0 += G) {
        const uint32_t t_g = (frames - g0 < G) ? (frames - g0) : G;
        const int first = (g0 == 0);
        h3_gpu_stats stats_start;
        h3_gpu_mem_stats mem_start;
        const double group_start = profile ? wan22_vae_stream_now() : 0.0;
        if (profile) {
            h3_gpu_get_stats(gpu, &stats_start);
            h3_gpu_mem_query(gpu, &mem_start);
        }

        /* gather and normalize the group's latents [48][t_g][h][w] */
        for (uint32_t c = 0; c < VAE_Z_DIM; c++) {
            const float mean = decoder->mean[c];
            const float stdv = decoder->std[c];
            for (uint32_t i = 0; i < t_g; i++) {
                const float *src =
                    z + ((size_t)c * frames + (size_t)g0 + i) * plane_latent;
                float *dst = stage + ((size_t)c * t_g + i) * plane_latent;
                for (size_t p = 0; p < plane_latent; p++)
                    dst[p] = src[p] * stdv + mean;
            }
        }
        zg = h3_gpu_tensor_new_f32(gpu,
                                   (size_t)VAE_Z_DIM * t_g * plane_latent);
        ok = zg && h3_gpu_tensor_write_f32(
                       zg, stage, (size_t)VAE_Z_DIM * t_g * plane_latent);
        if (!ok) wg_check(0, "stream z upload", gpu, error, error_size);

        /* conv2 (kt=1) -> conv1 (kt=3, carries conv2's output tail) */
        if (ok) {
            h3_gpu_tensor *c2out = h3_gpu_tensor_new_f32(
                gpu, (size_t)VAE_Z_DIM * t_g * plane_latent);
            ok = c2out && wg_conv(gpu, c2out, zg, &state->conv2, t_g, h, w);
            if (!ok)
                wg_check(0, "stream conv2", gpu, error, error_size);
            if (ok) {
                act = h3_gpu_tensor_new_f32(gpu,
                                            (size_t)1024u * t_g * plane_latent);
                ok = act && wg_conv_stream(gpu, act, c2out, &t_conv2out,
                                           &state->conv1, t_g, h, w, first,
                                           error, error_size);
                if (ok) ok = wg_tail_update(gpu, &t_conv2out, c2out, t_g);
            }
            h3_gpu_tensor_free(c2out);
            h3_gpu_tensor_free(zg);
            zg = NULL;
        }
        /* middle: resblock -> attention -> resblock */
        for (int i = 0; ok && i < 2; i++) {
            h3_gpu_tensor *next =
                h3_gpu_tensor_new_f32(gpu, (size_t)1024u * t_g * plane_latent);
            ok = next &&
                 wg_resblock_stream(gpu, &state->middle[i], act, next,
                                    i == 0 ? &t_mid00 : &t_mid10,
                                    i == 0 ? &t_mid01 : &t_mid11, t_g, h, w,
                                    first, error, error_size);
            h3_gpu_tensor_free(act);
            act = next;
            if (ok && i == 0) {
                ok = wg_check(wg_attention_forward(gpu, &state->attn, act,
                                                   t_g, h, w, error,
                                                   error_size),
                              "stream attn", gpu, error, error_size);
            }
        }
        /* upsample blocks; the b0/b1 time_conv tails are consumed inside
         * their blocks, the rest after their consumers ran */
        uint32_t t_a1 = 0;
        uint32_t t_a2 = 0;
        if (ok) {
            act = wg_up_block_stream(gpu, &state->ups[0], act, b0_heads, &t_R0,
                                     t_g, h, w, first, &t_a1, error,
                                     error_size);
        }
        if (ok) {
            act = wg_up_block_stream(gpu, &state->ups[1], act, b1_heads, &t_R1,
                                     t_a1, h * 2u, w * 2u, first, &t_a2,
                                     error, error_size);
        }
        if (ok) {
            act = wg_up_block_stream(gpu, &state->ups[2], act, b2_heads, NULL,
                                     t_a2, h * 4u, w * 4u, first, &t_a2,
                                     error, error_size);
        }
        if (ok) {
            act = wg_up_block_stream(gpu, &state->ups[3], act, b3_heads, NULL,
                                     t_a2, h * 8u, w * 8u, first, &t_a2,
                                     error, error_size);
        }
        /* head: RMS-SiLU -> causal conv (carries B3's tail) -> unpatchify */
        h3_gpu_tensor *out_dev = NULL;
        if (ok) {
            h3_gpu_tensor *head_in = h3_gpu_tensor_new_f32(
                gpu, (size_t)256u * t_a2 * 64u * plane_latent);
            h3_gpu_tensor *head_out = h3_gpu_tensor_new_f32(
                gpu, (size_t)12u * t_a2 * 64u * plane_latent);
            out_dev = h3_gpu_tensor_new_f32(gpu, (size_t)t_a2 * out_plane);
            ok = head_in && head_out && out_dev;
            if (ok) {
                ok = wg_check(h3_gpu_rms_norm_silu_f32(
                                  gpu, head_in, act, state->head_gamma, 256,
                                  t_a2, h * 8u, w * 8u, 1),
                              "stream head.norm", gpu, error, error_size);
            }
            if (ok) {
                ok = wg_conv_stream(gpu, head_out, head_in, &t_B3,
                                    &state->head, t_a2, h * 8u, w * 8u, first,
                                    error, error_size);
            }
            if (ok) ok = wg_tail_update(gpu, &t_B3, head_in, t_a2);
            if (ok) {
                ok = wg_check(h3_gpu_unpatchify_f32(gpu, out_dev, head_out,
                                                    t_a2, h * 8u, w * 8u),
                              "stream head.unpatchify", gpu, error,
                              error_size);
            }
            h3_gpu_tensor_free(head_in);
            h3_gpu_tensor_free(head_out);
        }
        if (ok) ok = wg_check(h3_gpu_submit(gpu), "stream submit", gpu, error,
                              error_size);
        if (ok) {
            /* the output is channel-major [3][4T-3][H][W]: each channel's
             * group frames land at its own frame offset */
            const size_t frame_base = first ? 0u : (size_t)4u * g0 - 3u;
            const size_t hw_out = (size_t)16u * h * 16u * w;
            const size_t video_frames = 4u * frames - 3u;
            for (uint32_t c = 0; ok && c < 3u; c++) {
                if (!h3_gpu_tensor_read_f32_range(
                        out_dev, (size_t)c * t_a2 * hw_out,
                        out + (size_t)c * video_frames * hw_out +
                            frame_base * hw_out,
                        (size_t)t_a2 * hw_out)) {
                    snprintf(error, error_size,
                             "wan22 vae stream download failed: %s",
                             h3_gpu_error(gpu));
                    ok = 0;
                }
            }
        }
        if (profile) {
            h3_gpu_stats stats_now;
            h3_gpu_mem_stats mem_now;
            const double wall = wan22_vae_stream_now() - group_start;
            /* H3_PROFILE makes the conv ops emit event pairs; the mark flushes
             * them into gpu_conv_seconds so the delta below is meaningful. */
            h3_gpu_profile_mark(gpu, "vae stream group");
            h3_gpu_get_stats(gpu, &stats_now);
            h3_gpu_mem_query(gpu, &mem_now);
            const double conv = stats_now.gpu_conv_seconds -
                                stats_start.gpu_conv_seconds;
            fprintf(stderr,
                    "[vae-stream] g=%3u t=%2u wall=%7.3fs conv=%7.3fs "
                    "rest=%7.3fs alloc=%7.1fMiB mallocs=%-6llu fails=%-4llu "
                    "live=%7.1f pool used=%7.1f reserved=%7.1f "
                    "high=%7.1f devfree=%7.1fMiB\n",
                    g0, t_g, wall, conv, wall - conv,
                    (double)(stats_now.allocated_bytes -
                             stats_start.allocated_bytes) /
                        (1024.0 * 1024.0),
                    (unsigned long long)(mem_now.mallocs - mem_start.mallocs),
                    (unsigned long long)(mem_now.malloc_fails -
                                         mem_start.malloc_fails),
                    (double)stats_now.live_bytes / (1024.0 * 1024.0),
                    (double)mem_now.used_current / (1024.0 * 1024.0),
                    (double)mem_now.reserved_current / (1024.0 * 1024.0),
                    (double)mem_now.reserved_high / (1024.0 * 1024.0),
                    (double)mem_now.device_free / (1024.0 * 1024.0));
            fflush(stderr);
        }
        h3_gpu_tensor_free(out_dev);
        h3_gpu_tensor_free(act);
        act = NULL;
    }

    free(stage);
    wg_tail_free(&t_conv2out);
    wg_tail_free(&t_mid00);
    wg_tail_free(&t_mid01);
    wg_tail_free(&t_mid10);
    wg_tail_free(&t_mid11);
    for (int i = 0; i < 6; i++) wg_tail_free(&t_b0[i]);
    wg_tail_free(&t_R0);
    wg_tail_free(&t_A1);
    for (int i = 0; i < 5; i++) wg_tail_free(&t_b1[i]);
    wg_tail_free(&t_R1);
    wg_tail_free(&t_A2);
    for (int i = 0; i < 5; i++) wg_tail_free(&t_b2[i]);
    wg_tail_free(&t_B2);
    for (int i = 0; i < 5; i++) wg_tail_free(&t_b3[i]);
    wg_tail_free(&t_B3);
    return ok;
}
