/* Wan 2.2 T2V 5B Phase 2/3a: pipeline assembly (text_embedding -> DiT ->
 * Euler -> VAE).  See h3_wan22_pipeline.h. */
#include "h3_wan22_pipeline.h"

#include "h3_quanto.h"
#include "h3_wan22_text.h"
#include "h3_weights.h"
#include "h3_gpu.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WAN22_BLOCKS 30
#define WAN22_DIM 3072
#define WAN22_PATCH_VEC 192 /* 48 * 2 * 2 */
#define WAN22_LATENT_CH 48

/* CLOCK_MONOTONIC seconds: the resume timing below and the per-forward
 * trace (H3_WAN22_FWD_TRACE=1) share one clock. */
static double wan22_pipeline_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Per-forward trace (H3_WAN22_FWD_TRACE=1): one stderr line per DiT forward
 * with the host-side wall of every stage.  A forward that follows a decode
 * pays the resume (weight re-upload) and the scratch rebuild inside its
 * first stages, and the very first forward after load pays the scratch
 * reserve; the trace separates those from steady-state compute so the
 * first-forward premium is attributed instead of guessed.  Pure
 * instrumentation -- nothing in the math changes. */
static int wan22_fwd_trace(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("H3_WAN22_FWD_TRACE");
        cached = value && *value && strcmp(value, "0") != 0;
    }
    return cached;
}

struct h3_wan22_pipeline {
    h3_wan22_config config;
    h3_weight_store *store;
    h3_wan22_linear patch_embed;     /* F32 [3072, 192] + bias */
    h3_wan22_time_embedder embedder; /* F32 time_embedding */
    h3_wan22_linear time_projection; /* F32 [18432, 3072] */
    h3_wan22_block *blocks[WAN22_BLOCKS];
    h3_wan22_head head;
    h3_wan22_text_mlp text_embedding; /* 4096 -> 3072 -> 3072, BF16 */
    h3_wan22_vae_decoder *vae;
    /* optional GPU forward path (WAN22_GPU=1), NULL stays on the CPU */
    h3_wan22_gpu *gpu;
    /* Latent/context geometry of the last GPU forward: the decode releases
     * the DiT device state (see h3_wan22_pipeline_decode), so a later
     * forward has to rebuild it for the same shape. */
    uint32_t gpu_seq;
    uint32_t gpu_ctx_seq;
    uint64_t fwd_index; /* H3_WAN22_FWD_TRACE forward counter */
};

void h3_wan22_pipeline_free(h3_wan22_pipeline *pipeline) {
    if (!pipeline) return;
    for (int i = 0; i < WAN22_BLOCKS; i++) {
        h3_wan22_block_free(pipeline->blocks[i]);
    }
    h3_wan22_linear_free(&pipeline->patch_embed);
    h3_wan22_time_embedder_free(&pipeline->embedder);
    h3_wan22_linear_free(&pipeline->time_projection);
    h3_wan22_head_free(&pipeline->head);
    h3_wan22_text_mlp_free(&pipeline->text_embedding);
    if (pipeline->vae) h3_wan22_vae_decoder_free(pipeline->vae);
    if (pipeline->gpu) h3_wan22_gpu_free(pipeline->gpu);
    if (pipeline->store) h3_weight_store_free(pipeline->store);
    free(pipeline);
}

/* One-time device warmup, run once after the weights are uploaded.  The
 * first forward of a process pays three one-time costs that a 720p request
 * would otherwise absorb inside its dit phase: the driver stages the first
 * pageable host-to-device copy (measured +1.6 s on the upload stage), and the
 * first BF16 GEMM and flash-SDPA launches load their kernel modules and run
 * the library's per-shape setup (measured +0.5 s on the blocks stage).  A few
 * tiny launches move all of that to load time.  The tensors are zero-filled
 * and freed here, so no real data and no math is involved. */
static void wan22_gpu_warmup(h3_gpu *gpu, const h3_wan22_config *config) {
    const uint32_t dim = config->dim;
    const uint32_t rows = 8; /* a real launch, small enough to be free */
    uint16_t *zeros_bf16 = calloc((size_t)rows * dim, 2);
    float *zeros_f32 = calloc((size_t)rows * dim, sizeof(float));
    uint16_t *zeros_weight = calloc((size_t)dim * dim, 2);
    h3_gpu_tensor *act = zeros_bf16
                             ? h3_gpu_tensor_from_bf16(gpu, zeros_bf16,
                                                       (size_t)rows * dim)
                             : NULL;
    h3_gpu_tensor *weight =
        zeros_weight ? h3_gpu_tensor_from_bf16(gpu, zeros_weight,
                                               (size_t)dim * dim)
                     : NULL;
    h3_gpu_tensor *carrier = zeros_f32
                                 ? h3_gpu_tensor_new_f32(gpu,
                                                         (size_t)rows * dim)
                                 : NULL;
    if (act && weight && carrier &&
        h3_gpu_tensor_write_f32(carrier, zeros_f32, (size_t)rows * dim)) {
        h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * dim);
        if (out) {
            h3_gpu_linear_bf16(gpu, out, act, weight, NULL, rows, dim, dim);
            h3_gpu_sdpa_flash_bf16(gpu, out, act, act, act, rows, rows,
                                   config->heads, config->head_dim,
                                   1.0f / sqrtf((float)config->head_dim));
            h3_gpu_tensor_free(out);
        }
    }
    h3_gpu_tensor_free(act);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(carrier);
    free(zeros_bf16);
    free(zeros_f32);
    free(zeros_weight);
}

h3_wan22_pipeline *h3_wan22_pipeline_load(const char *dit_ckpt_dir,
                                          const char *vae_path,
                                          const h3_wan22_config *config,
                                          char *error, size_t error_size) {
    h3_wan22_pipeline *pipeline = calloc(1, sizeof(*pipeline));
    if (!pipeline) {
        snprintf(error, error_size, "out of host memory (pipeline)");
        return NULL;
    }
    pipeline->config = config ? *config : h3_wan22_default_config();
    pipeline->store = h3_weight_store_open(dit_ckpt_dir, error, error_size);
    if (!pipeline->store) {
        h3_wan22_pipeline_free(pipeline);
        return NULL;
    }
    if (!h3_wan22_linear_load_f32(pipeline->store, "patch_embedding.weight",
                                  WAN22_DIM, WAN22_PATCH_VEC, 1,
                                  &pipeline->patch_embed, error,
                                  error_size)) {
        h3_wan22_pipeline_free(pipeline);
        return NULL;
    }
    if (!h3_wan22_time_embedder_load(pipeline->store, &pipeline->config,
                                     &pipeline->embedder, error,
                                     error_size)) {
        h3_wan22_pipeline_free(pipeline);
        return NULL;
    }
    if (!h3_wan22_linear_load_f32(pipeline->store, "time_projection.1.weight",
                                  WAN22_DIM * 6, WAN22_DIM, 1,
                                  &pipeline->time_projection, error,
                                  error_size)) {
        h3_wan22_pipeline_free(pipeline);
        return NULL;
    }
    for (int i = 0; i < WAN22_BLOCKS; i++) {
        pipeline->blocks[i] =
            h3_wan22_block_load(pipeline->store, i, &pipeline->config, error,
                                error_size);
        if (!pipeline->blocks[i]) {
            h3_wan22_pipeline_free(pipeline);
            return NULL;
        }
    }
    if (!h3_wan22_head_load(pipeline->store, &pipeline->config,
                            WAN22_PATCH_VEC, &pipeline->head, error,
                            error_size)) {
        h3_wan22_pipeline_free(pipeline);
        return NULL;
    }
    if (!h3_wan22_text_mlp_load(pipeline->store, 4096, WAN22_DIM, WAN22_DIM,
                                &pipeline->text_embedding, error,
                                error_size)) {
        h3_wan22_pipeline_free(pipeline);
        return NULL;
    }
    pipeline->vae = h3_wan22_vae_decoder_load(vae_path, error, error_size);
    if (!pipeline->vae) {
        h3_wan22_pipeline_free(pipeline);
        return NULL;
    }
    /* Optional GPU path: WAN22_GPU=1 uploads the 30 blocks (~9.8 GB BF16);
     * 0 / unset keeps the CPU reference path.  A CUDA failure falls back to
     * the CPU path with a note on stderr. */
    const char *gpu_switch = getenv("WAN22_GPU");
    if (gpu_switch && strcmp(gpu_switch, "1") == 0) {
        char gpu_error[256] = {0};
        pipeline->gpu = h3_wan22_gpu_create(&pipeline->config, gpu_error,
                                            sizeof(gpu_error));
        if (!pipeline->gpu) {
            fprintf(stderr, "wan22: GPU unavailable (%s), CPU path\n",
                    gpu_error);
        } else {
            int ok = 1;
            for (int i = 0; ok && i < WAN22_BLOCKS; i++) {
                ok = h3_wan22_block_upload_gpu(pipeline->blocks[i],
                                               pipeline->gpu, error,
                                               error_size);
            }
            if (!ok) {
                fprintf(stderr, "wan22: GPU upload failed (%s), CPU path\n",
                        error);
                h3_wan22_gpu_free(pipeline->gpu);
                pipeline->gpu = NULL;
            } else {
                wan22_gpu_warmup(h3_wan22_gpu_handle(pipeline->gpu),
                                 &pipeline->config);
            }
        }
    }
    return pipeline;
}

/* --- block-level residual dumps -----------------------------------------
 * WAN22_DUMP_BLOCK=<-1,0,1,5,10> plus WAN22_DUMP_BLOCK_PATH=<prefix> writes
 * the F32 residual stream after each listed transformer block as raw F32
 * [seq][dim] to "<prefix>.<n>.bin"; -1 means "the patched input, before
 * block 0".  The CPU and the GPU path share that layout, so the same request
 * run twice (WAN22_GPU=1 and WAN22_GPU=0) and diffed per block says where the
 * relative error stops looking like rounding and starts looking like a wrong
 * answer.  A list rather than one index because a full-size CPU forward takes
 * tens of minutes; dumping the first few blocks is enough to see the slope. */
static const unsigned char *wan22_block_dump_mask(void) {
    static unsigned char mask[WAN22_BLOCKS + 1];
    static int parsed;
    if (parsed) return mask;
    parsed = 1;
    const char *value = getenv("WAN22_DUMP_BLOCK");
    if (!value || !*value) return mask;
    const char *cursor = value;
    while (*cursor) {
        char *end = NULL;
        const long index = strtol(cursor, &end, 10);
        if (end == cursor) break;
        if (index >= -1 && index < WAN22_BLOCKS) mask[index + 1] = 1;
        cursor = end;
        while (*cursor == ',' || *cursor == ' ') cursor++;
    }
    return mask;
}

static int wan22_block_dump_wanted(int block) {
    if (block < -1 || block >= WAN22_BLOCKS) return 0;
    return wan22_block_dump_mask()[block + 1] != 0;
}

/* Which CFG branch the next forward belongs to.  Only the residual dump cares,
 * and only the sampler knows: the two branches share one code path, one context
 * shape (both padded to 512 rows) and one geometry, so without this the uncond
 * forward silently overwrites the cond dumps and "block 0 is 7.8% off" turns
 * out to be "cond was compared against uncond". */
static const char *wan22_forward_branch = "cond";

/* WAN22_DUMP_CTX=<prefix> writes "<prefix>.<branch>.raw.bin" (the assembled
 * 4096-dim context the text_embedding MLP sees) and "<prefix>.<branch>.te.bin"
 * (its output, the 3072-dim rows cross-attention actually keys on). */
static void wan22_dump_context_if_requested(const float *raw, const float *te,
                                            uint32_t ctx_seq) {
    const char *prefix = getenv("WAN22_DUMP_CTX");
    if (!prefix || !*prefix) return;
    const char *branch = getenv("WAN22_DUMP_BRANCH");
    const char *tag = (branch && *branch) ? branch : wan22_forward_branch;
    char path[512];
    snprintf(path, sizeof(path), "%s.%s.raw.bin", prefix, tag);
    FILE *out = fopen(path, "wb");
    if (out) {
        fwrite(raw, sizeof(float), (size_t)ctx_seq * 4096u, out);
        fclose(out);
        fprintf(stderr, "h3 wan22: context raw -> %s\n", path);
    }
    snprintf(path, sizeof(path), "%s.%s.te.bin", prefix, tag);
    out = fopen(path, "wb");
    if (out) {
        fwrite(te, sizeof(float), (size_t)ctx_seq * WAN22_DIM, out);
        fclose(out);
        fprintf(stderr, "h3 wan22: context te -> %s\n", path);
    }
}

static void wan22_dump_residual_if_requested(const float *plane, size_t count,
                                             int block) {
    if (!wan22_block_dump_wanted(block)) return;
    const char *prefix = getenv("WAN22_DUMP_BLOCK_PATH");
    if (!prefix || !*prefix) return;
    char path[512];
    /* WAN22_DUMP_BRANCH=<cond|uncond> tags the file and drops the other
     * branch; unset keeps the historical untagged name. */
    const char *branch = getenv("WAN22_DUMP_BRANCH");
    if (branch && *branch) {
        if (strcmp(branch, wan22_forward_branch) != 0) return;
        snprintf(path, sizeof(path), "%s.%s.%d.bin", prefix,
                 wan22_forward_branch, block);
    } else {
        snprintf(path, sizeof(path), "%s.%d.bin", prefix, block);
    }
    FILE *out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "h3 wan22: cannot write %s\n", path);
        return;
    }
    if (fwrite(plane, sizeof(float), count, out) != count) {
        fprintf(stderr, "h3 wan22: warning: %s is short\n", path);
    }
    fclose(out);
    fprintf(stderr, "h3 wan22: residual after block %d -> %s\n", block, path);
}

/* DiT unpatchify (model.py WanModel.unpatchify, patch_size (1, 2, 2)):
 * out[c][f][2h+q][2w+r] = tokens[f * gh * gw + h * gw + w][q*96 + r*48 + c].
 * tokens: [seq, 192]; output [48, T, h, w] with the caller's full h/w
 * strides.  The reference unpatchify writes grid * patch_size = 2*gh x
 * 2*gw rows/cols; with an odd latent h or w the tail rows/cols of the
 * buffer are left to zero_dropped_velocity() below. */
static void dit_unpatchify(const float *tokens, float *out, uint32_t frames,
                           uint32_t gh, uint32_t gw, uint32_t h,
                           uint32_t w) {
    for (uint32_t f = 0; f < frames; f++) {
        for (uint32_t y = 0; y < gh; y++) {
            for (uint32_t x = 0; x < gw; x++) {
                const float *vec =
                    tokens +
                    ((size_t)f * gh + y) * gw * WAN22_PATCH_VEC +
                    (size_t)x * WAN22_PATCH_VEC;
                for (uint32_t q = 0; q < 2; q++) {
                    for (uint32_t r = 0; r < 2; r++) {
                        float *dst =
                            out + (size_t)f * h * w +
                                   ((size_t)y * 2 + q) * w + x * 2 + r;
                        const float *src = vec + (size_t)q * 96 + r * 48;
                        for (uint32_t c = 0; c < WAN22_LATENT_CH; c++) {
                            dst[(size_t)c * frames * h * w] = src[c];
                        }
                    }
                }
            }
        }
    }
}

/* Odd latent h/w: the reference patch_embedding is Conv3d(k=(1,2,2),
 * stride=(1,2,2), padding=0) (Wan2GP models/wan/modules/model.py:1165),
 * so the last latent row/col is dropped (floor), never padded, and
 * unpatchify (model.py:2173) writes only 2*gh x 2*gw rows/cols.  gh=h/2
 * and gw=w/2 below already floor, so the patch/unpatchify math is the
 * reference's; zero_dropped_velocity() zeroes the velocity rows/cols
 * past 2*gh/2*gw so the dropped region has a defined value.  Zero
 * velocity makes the Euler update a no-op there, i.e. those latent rows
 * stay frozen at their initial noise exactly as if the grid were the
 * even part -- the caller can crop to 2*gh x 2*gw rows/cols before the
 * VAE decode for a result identical to running the even grid.  WanGP
 * itself never feeds an odd grid here (it floors the video canvas to
 * vae_block_size first, wan_handler.py:326 + resolutions.py:204, and
 * get_rotary_pos_embed asserts even latent dims, posemb_layers.py:518);
 * this path pins the DiT-level semantics for direct callers such as a
 * 720p request. */
static void zero_dropped_velocity(float *velocity, uint32_t frames,
                                  uint32_t gh, uint32_t gw, uint32_t h,
                                  uint32_t w) {
    for (uint32_t f = 0; f < frames; f++) {
        for (uint32_t y = 2 * gh; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                for (uint32_t c = 0; c < WAN22_LATENT_CH; c++) {
                    velocity[((size_t)c * frames + f) * h * w +
                             (size_t)y * w + x] = 0.0f;
                }
            }
        }
        for (uint32_t y = 0; y < h && y < 2 * gh; y++) {
            for (uint32_t x = 2 * gw; x < w; x++) {
                for (uint32_t c = 0; c < WAN22_LATENT_CH; c++) {
                    velocity[((size_t)c * frames + f) * h * w +
                             (size_t)y * w + x] = 0.0f;
                }
            }
        }
    }
}

int h3_wan22_pipeline_text_embedding(h3_wan22_pipeline *pipeline,
                                     const float *context, uint32_t ctx_seq,
                                     float *out) {
    if (!pipeline || !context || !out) return 0;
    return h3_wan22_text_mlp_forward(&pipeline->text_embedding, context,
                                     ctx_seq, out);
}

int h3_wan22_context_assemble(float *plane, uint32_t ctx_seq, uint32_t dim,
                              const float *source, uint32_t rows,
                              uint32_t window) {
    if (!plane || !source || dim == 0) return 0;
    if (rows > window || window > ctx_seq) return 0;
    memcpy(plane, source, (size_t)rows * dim * sizeof(float));
    /* Covers both the encoder's pad-token activations (rows..window) and the
     * tail WanGP pads (window..ctx_seq) in one pass. */
    memset(plane + (size_t)rows * dim, 0,
           ((size_t)ctx_seq - rows) * dim * sizeof(float));
    return 1;
}

int h3_wan22_pipeline_forward(h3_wan22_pipeline *pipeline,
                              const float *latents, uint32_t frames,
                              uint32_t h, uint32_t w, float t,
                              const float *context, uint32_t ctx_seq,
                              float *velocity, char *error,
                              size_t error_size) {
    /* Odd latent h/w are accepted: gh=h/2 (gw=w/2) floors, exactly the
     * reference Conv3d stride-2 padding-0 patch_embedding (Wan2GP
     * models/wan/modules/model.py:1165) -- the tail row/col is dropped,
     * not padded -- and the RoPE grid below uses the same post-conv grid
     * the reference derives from grid_sizes=x.shape[2:]. */
    if (!pipeline || !latents || !context || !velocity || frames == 0 ||
        h == 0 || w == 0) {
        snprintf(error, error_size, "invalid pipeline forward arguments");
        return 0;
    }
    const uint32_t gh = h / 2;
    const uint32_t gw = w / 2;
    const size_t seq = (size_t)frames * gh * gw;

    /* H3_WAN22_FWD_TRACE stage stamps (host wall; the blocks are submitted
     * asynchronously, so their GPU time lands in the tail/readback stage). */
    const int fwd_trace = wan22_fwd_trace();
    double trace_started = fwd_trace ? wan22_pipeline_now() : 0.0;
    double trace_patched = 0.0, trace_resumed = 0.0, trace_reserved = 0.0;
    double trace_uploaded = 0.0, trace_blocks = 0.0;
    int fwd_resumed = 0;

    /* patch tokens: vec[c*4 + q*2 + r] = latents[c, f, 2y+q, 2x+r]; the
     * latent layout is [C][T][H][W], so channel c sits at
     * c * frames * h * w. */
    float *tokens = malloc(seq * WAN22_PATCH_VEC * sizeof(float));
    float *x = malloc(seq * WAN22_DIM * sizeof(float));
    float *e0 = malloc((size_t)WAN22_DIM * 6 * sizeof(float));
    float *t_freq = malloc(256 * sizeof(float));
    float *e = malloc(WAN22_DIM * sizeof(float));
    float *head_out = malloc(seq * WAN22_PATCH_VEC * sizeof(float));
    float *rope_cos = malloc(seq * 64 * sizeof(float));
    float *rope_sin = malloc(seq * 64 * sizeof(float));
    if (!tokens || !x || !e0 || !t_freq || !e || !head_out || !rope_cos ||
        !rope_sin) {
        free(tokens);
        free(x);
        free(e0);
        free(t_freq);
        free(e);
        free(head_out);
        free(rope_cos);
        free(rope_sin);
        snprintf(error, error_size, "out of host memory (pipeline forward)");
        return 0;
    }
    for (uint32_t f = 0; f < frames; f++) {
        for (uint32_t y = 0; y < gh; y++) {
            for (uint32_t xx = 0; xx < gw; xx++) {
                float *vec =
                    tokens +
                    ((size_t)f * gh + y) * gw * WAN22_PATCH_VEC +
                    (size_t)xx * WAN22_PATCH_VEC;
                for (uint32_t q = 0; q < 2; q++) {
                    for (uint32_t r = 0; r < 2; r++) {
                        const float *src =
                            latents + ((size_t)f * h + y * 2 + q) * w +
                                       xx * 2 + r;
                        float *dst = vec + (size_t)q * 2 + r;
                        for (uint32_t c = 0; c < WAN22_LATENT_CH; c++) {
                            dst[(size_t)c * 4] =
                                src[(size_t)c * frames * h * w];
                        }
                    }
                }
            }
        }
    }
    if (!h3_wan22_linear_forward(&pipeline->patch_embed, tokens,
                                 (uint32_t)seq, x)) {
        snprintf(error, error_size, "patch_embedding failed");
        goto fail;
    }
    free(tokens);
    tokens = NULL;
    if (fwd_trace) trace_patched = wan22_pipeline_now();

    if (pipeline->gpu) {
        /* --- GPU path: blocks on the device, patch/time/head on host ---
         * The residual carrier x_f32 stays in F32 on the device across the
         * 30 blocks (matches the fp32 reference chain); only the per-block
         * inputs are staged as BF16. */
        h3_wan22_gpu *wgu = pipeline->gpu;
        pipeline->gpu_seq = (uint32_t)seq;
        pipeline->gpu_ctx_seq = ctx_seq;
        /* A decode may have handed the device memory back: rebuild the
         * weights (one upload from the host BF16 copies) before reserving. */
        fwd_resumed = !h3_wan22_gpu_state_resident(wgu);
        if (fwd_resumed &&
            !h3_wan22_pipeline_resume_dit(pipeline, error, error_size)) {
            goto fail;
        }
        if (fwd_trace) trace_resumed = wan22_pipeline_now();
        if (!h3_wan22_gpu_reserve(wgu, (uint32_t)seq, ctx_seq, error,
                                  error_size)) {
            goto fail;
        }
        if (fwd_trace) trace_reserved = wan22_pipeline_now();
        h3_gpu_tensor *x_gpu = h3_wan22_gpu_x(wgu); /* F32 carrier */
        if (!h3_gpu_tensor_write_f32(x_gpu, x, seq * WAN22_DIM)) {
            snprintf(error, error_size, "cannot upload x: %s",
                     h3_gpu_error(h3_wan22_gpu_handle(wgu)));
            goto fail;
        }
        /* RoPE pair tables (F32) straight from the CPU generator. */
        h3_wan22_rope_tables(&pipeline->config, frames, gh, gw, rope_cos,
                             rope_sin);
        if (!h3_gpu_tensor_write_f32(h3_wan22_gpu_rope_cos(wgu), rope_cos,
                                     seq * 64) ||
            !h3_gpu_tensor_write_f32(h3_wan22_gpu_rope_sin(wgu), rope_sin,
                                     seq * 64)) {
            snprintf(error, error_size, "cannot upload rope: %s",
                     h3_gpu_error(h3_wan22_gpu_handle(wgu)));
            goto fail;
        }
        /* Embedded text context as BF16. */
        float *ctx3072 = malloc((size_t)ctx_seq * WAN22_DIM * sizeof(float));
        if (!ctx3072) {
            snprintf(error, error_size, "out of host memory (GPU ctx)");
            goto fail;
        }
        if (!h3_wan22_text_mlp_forward(&pipeline->text_embedding, context,
                                       ctx_seq, ctx3072)) {
            free(ctx3072);
            snprintf(error, error_size, "text_embedding failed");
            goto fail;
        }
        /* Same dump as the host path, before the BF16 upload: the context is
         * the one input the two CFG branches do not share, so it has to be
         * comparable on the path that actually serves requests. */
        wan22_dump_context_if_requested(context, ctx3072, ctx_seq);
        h3_gpu_tensor *ctx_gpu = h3_wan22_gpu_upload_f32_bf16(
            wgu, ctx3072, (size_t)ctx_seq * WAN22_DIM);
        free(ctx3072);
        if (!ctx_gpu) {
            snprintf(error, error_size, "cannot upload context: %s",
                     h3_gpu_error(h3_wan22_gpu_handle(wgu)));
            goto fail;
        }
        /* Timestep chain on the host (F32): e for the Head, e0 for adaLN. */
        h3_wan22_timestep_embedding(t, 256, t_freq);
        if (!h3_wan22_time_embedder_forward(&pipeline->embedder, t_freq, 1,
                                            e)) {
            snprintf(error, error_size, "time_embedding failed");
            h3_gpu_tensor_free(ctx_gpu);
            goto fail;
        }
        float *e_silu = malloc(WAN22_DIM * sizeof(float));
        if (!e_silu) {
            h3_gpu_tensor_free(ctx_gpu);
            snprintf(error, error_size, "out of host memory (pipeline)");
            goto fail;
        }
        for (uint32_t i = 0; i < WAN22_DIM; i++) {
            const float value = e[i];
            e_silu[i] = value / (1.0f + expf(-value));
        }
        h3_wan22_linear_forward(&pipeline->time_projection, e_silu, 1, e0);
        free(e_silu);
        if (!h3_wan22_gpu_set_e0_gpu(wgu, e0, error, error_size)) {
            h3_gpu_tensor_free(ctx_gpu);
            goto fail;
        }
        if (fwd_trace) trace_uploaded = wan22_pipeline_now();
        if (wan22_block_dump_wanted(-1)) {
            float *snapshot = malloc((size_t)seq * WAN22_DIM * sizeof(float));
            if (snapshot &&
                h3_wan22_gpu_read_f32(wgu, x_gpu, snapshot,
                                      (size_t)seq * WAN22_DIM)) {
                wan22_dump_residual_if_requested(snapshot,
                                                 (uint32_t)seq * WAN22_DIM, -1);
            }
            free(snapshot);
        }
        for (int i = 0; i < WAN22_BLOCKS; i++) {
            if (!h3_wan22_block_forward_gpu(pipeline->blocks[i], wgu, x_gpu,
                                            (uint32_t)seq, ctx_gpu, ctx_seq,
                                            h3_wan22_gpu_rope_cos(wgu),
                                            h3_wan22_gpu_rope_sin(wgu),
                                            error, error_size)) {
                h3_gpu_tensor_free(ctx_gpu);
                goto fail;
            }
            if (wan22_block_dump_wanted(i)) {
                float *snapshot =
                    malloc((size_t)seq * WAN22_DIM * sizeof(float));
                if (snapshot &&
                    h3_wan22_gpu_read_f32(wgu, x_gpu, snapshot,
                                          (size_t)seq * WAN22_DIM)) {
                    wan22_dump_residual_if_requested(snapshot,
                                                     (uint32_t)seq * WAN22_DIM,
                                                     i);
                }
                free(snapshot);
            }
        }
        if (fwd_trace) trace_blocks = wan22_pipeline_now();
        h3_gpu_tensor_free(ctx_gpu);
        /* Read the hidden state back and run the F32 Head on the host. */
        if (!h3_wan22_gpu_read_f32(wgu, x_gpu, x, seq * WAN22_DIM)) {
            snprintf(error, error_size, "cannot read back x: %s",
                     h3_gpu_error(h3_wan22_gpu_handle(wgu)));
            goto fail;
        }
        if (!h3_wan22_head_forward(&pipeline->head, x, (uint32_t)seq, e,
                                   head_out)) {
            snprintf(error, error_size, "head failed");
            goto fail;
        }
        dit_unpatchify(head_out, velocity, frames, gh, gw, h, w);
        zero_dropped_velocity(velocity, frames, gh, gw, h, w);
        if (fwd_trace) {
            pipeline->fwd_index++;
            const double finished = wan22_pipeline_now();
            fprintf(stderr,
                    "wan22: fwd#%llu seq=%u ctx=%u%s patch=%.3f resume=%.3f "
                    "reserve=%.3f upload=%.3f blocks=%.3f tail=%.3f "
                    "total=%.3f\n",
                    (unsigned long long)pipeline->fwd_index, (unsigned)seq,
                    ctx_seq, fwd_resumed ? " RESUMED" : "",
                    trace_patched - trace_started, trace_resumed - trace_patched,
                    trace_reserved - trace_resumed,
                    trace_uploaded - trace_reserved, trace_blocks - trace_uploaded,
                    finished - trace_blocks, finished - trace_started);
            fflush(stderr);
            /* With H3_PROFILE=1 this additionally reports the exclusive GPU
             * seconds of the linear / SDPA ops this forward issued. */
            h3_gpu_profile_mark(h3_wan22_gpu_handle(wgu), "wan22 fwd");
        }
        free(tokens);
        free(x);
        free(e0);
        free(t_freq);
        free(e);
        free(head_out);
        free(rope_cos);
        free(rope_sin);
        return 1;
    }

    /* timestep chain: e = time_embedding(sinusoid(t)); the Head modulates
     * with the raw e, while e0 = time_projection(silu(e)) (the Linear needs
     * distinct input/output buffers). */
    h3_wan22_timestep_embedding(t, 256, t_freq);
    if (!h3_wan22_time_embedder_forward(&pipeline->embedder, t_freq, 1, e)) {
        snprintf(error, error_size, "time_embedding failed");
        goto fail;
    }
    {
        float *e_silu = malloc(WAN22_DIM * sizeof(float));
        if (!e_silu) {
            snprintf(error, error_size, "out of host memory (pipeline)");
            goto fail;
        }
        for (uint32_t i = 0; i < WAN22_DIM; i++) {
            const float value = e[i];
            e_silu[i] = value / (1.0f + expf(-value));
        }
        h3_wan22_linear_forward(&pipeline->time_projection, e_silu, 1, e0);
        free(e_silu);
    }

    if (!h3_wan22_rope_tables(&pipeline->config, frames, gh, gw, rope_cos,
                              rope_sin)) {
        snprintf(error, error_size, "rope table failure");
        goto fail;
    }

    {
        float *ctx = malloc((size_t)ctx_seq * WAN22_DIM * sizeof(float));
        if (!ctx) {
            snprintf(error, error_size, "out of host memory (text_embedding)");
            goto fail;
        }
        if (!h3_wan22_text_mlp_forward(&pipeline->text_embedding, context,
                                       ctx_seq, ctx)) {
            free(ctx);
            snprintf(error, error_size, "text_embedding failed");
            goto fail;
        }
        /* WAN22_DUMP_CTX=<prefix> writes the assembled context and the
         * text_embedding output for the current branch.  The cond branch can
         * agree to 1e-4 while uncond is 2.4% off from block 0, and the context
         * is then the only input the two branches do not share -- so the one
         * thing worth diffing is the context itself, not the block code. */
        wan22_dump_context_if_requested(context, ctx, ctx_seq);
        wan22_dump_residual_if_requested(x, (uint32_t)seq * WAN22_DIM, -1);
        for (int i = 0; i < WAN22_BLOCKS; i++) {
            if (!h3_wan22_block_forward(pipeline->blocks[i], x,
                                        (uint32_t)seq, e0, ctx, ctx_seq,
                                        rope_cos, rope_sin, error,
                                        error_size)) {
                free(ctx);
                goto fail;
            }
            wan22_dump_residual_if_requested(x, (uint32_t)seq * WAN22_DIM, i);
        }
        free(ctx);
    }

    if (!h3_wan22_head_forward(&pipeline->head, x, (uint32_t)seq, e,
                               head_out)) {
        snprintf(error, error_size, "head failed");
        goto fail;
    }
    dit_unpatchify(head_out, velocity, frames, gh, gw, h, w);
    zero_dropped_velocity(velocity, frames, gh, gw, h, w);

    free(tokens);
    free(x);
    free(e0);
    free(t_freq);
    free(e);
    free(head_out);
    free(rope_cos);
    free(rope_sin);
    return 1;

fail:
    free(tokens);
    free(x);
    free(e0);
    free(t_freq);
    free(e);
    free(head_out);
    free(rope_cos);
    free(rope_sin);
    return 0;
}

/* EulerScheduler.set_timesteps (WanGP shared/utils/euler_scheduler.py):
 * linspace(1000, 1, steps) ++ [0], each value transformed with
 * shift*t/(1+(shift-1)*t) in 1/1000 units, and the transformed trailing 0
 * dropped -- so the schedule ends at transform(1) = ~4.98, not at 0, and the
 * last step's dt is that value.  The reference fixture
 * tests/fixtures/wan22_pipeline/timesteps.bin pins these numbers. */
static uint32_t euler_timesteps(float *out, uint32_t steps, float shift) {
    if (steps == 0) return 0;
    for (uint32_t i = 0; i < steps; i++) {
        const float raw = steps == 1
                              ? 1000.0f
                              : 1000.0f + (1.0f - 1000.0f) *
                                        (float)i / (float)(steps - 1);
        const float u = raw / 1000.0f;
        out[i] = shift * u / (1.0f + (shift - 1.0f) * u) * 1000.0f;
    }
    return steps;
}

/* Value trace (WAN22_SAMPLE_TRACE=1): RMS and mean of the latents and of both
 * velocity branches per step.  This is the cheapest way to tell "the model is
 * not moving the latent" (rms(v) ~ 0) from "the latent is moving but decodes
 * to noise" (rms(v) healthy, rms(x) not shrinking).  Pure instrumentation. */
static void wan22_trace_stats(const char *label, uint32_t step, float t,
                              const float *x, size_t count, const float *v,
                              const float *vu) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("WAN22_SAMPLE_TRACE");
        cached = value && *value && strcmp(value, "0") != 0;
    }
    if (!cached) return;
    double sx = 0.0, sxx = 0.0, sv = 0.0, svv = 0.0, svu = 0.0, svuvu = 0.0;
    double sd = 0.0, svvuvu = 0.0;
    for (size_t i = 0; i < count; i++) {
        sx += x[i];
        sxx += (double)x[i] * x[i];
        if (v) { sv += v[i]; svv += (double)v[i] * v[i]; }
        if (vu) { svu += vu[i]; svuvu += (double)vu[i] * vu[i]; }
        if (v && vu) {
            const double delta = (double)v[i] - (double)vu[i];
            sd += delta * delta;
            svvuvu += (double)v[i] * (double)vu[i];
        }
    }
    const double n = (double)count;
    /* dv is the term CFG actually amplifies: rms(v_cond - v_uncond).  Without
     * it "v and vu have the same rms" says nothing -- two branches that agree
     * in magnitude can still be very different vectors -- and it is the one
     * number that separates "the text is steering" from "guidance 5 is a
     * no-op and the sample is effectively unconditional". */
    fprintf(stderr,
            "wan22 trace %-9s step %u t=%7.2f x[rms %.4f mean %+.4f] "
            "v[rms %.4f mean %+.4f] vu[rms %.4f mean %+.4f] "
            "dv[rms %.4f cos %+.4f]\n",
            label, step, t, sqrt(sxx / n), sx / n,
            v ? sqrt(svv / n) : 0.0, v ? sv / n : 0.0,
            vu ? sqrt(svuvu / n) : 0.0, vu ? svu / n : 0.0,
            (v && vu) ? sqrt(sd / n) : 0.0,
            (v && vu && svv > 0.0 && svuvu > 0.0)
                ? svvuvu / sqrt(svv * svuvu) : 0.0);
    fflush(stderr);
}

/* One CFG step: both branches at timestep t, then the guided mix WanGP hands to
 * scheduler.step (v_uncond + guide * (v_cond - v_uncond)).  Both samplers
 * integrate this same field; only the integrator differs. */
static int wan22_cfg_velocity(h3_wan22_pipeline *pipeline, const float *latents,
                              uint32_t frames, uint32_t h, uint32_t w, float t,
                              const float *context_cond, uint32_t ctx_cond_len,
                              const float *context_uncond,
                              uint32_t ctx_uncond_len, float guide,
                              float *v_cond, float *v_uncond, float *mixed,
                              size_t latent_count, char *error,
                              size_t error_size) {
    wan22_forward_branch = "cond";
    if (!h3_wan22_pipeline_forward(pipeline, latents, frames, h, w, t,
                                   context_cond, ctx_cond_len, v_cond, error,
                                   error_size)) {
        return 0;
    }
    wan22_forward_branch = "uncond";
    if (!h3_wan22_pipeline_forward(pipeline, latents, frames, h, w, t,
                                   context_uncond, ctx_uncond_len, v_uncond,
                                   error, error_size)) {
        return 0;
    }
    for (size_t i = 0; i < latent_count; i++) {
        mixed[i] = v_uncond[i] + guide * (v_cond[i] - v_uncond[i]);
    }
    return 1;
}

/* WAN22_DUMP_VEL=<path> writes the CFG-mixed velocity of a chosen step as raw
 * F32 [48][frames][h][w].  The reference script writes the same array as
 * vel_step<N>.bin, so a numeric comparison at the geometry that actually fails
 * (31 latent frames, 30x30) no longer needs a fixture: run the reference with
 * ATTN_SDPA=1 and diff. */
static void wan22_maybe_dump_velocity(uint32_t step, const float *mixed,
                                      size_t latent_count) {
    const char *dump_vel = getenv("WAN22_DUMP_VEL");
    const char *dump_step = getenv("WAN22_DUMP_VEL_STEP");
    if (!dump_vel || !*dump_vel) return;
    const uint32_t wanted = dump_step && *dump_step
                                ? (uint32_t)strtoul(dump_step, NULL, 10)
                                : 0u;
    if (step != wanted) return;
    FILE *out = fopen(dump_vel, "wb");
    if (!out) {
        fprintf(stderr, "h3 wan22: cannot write %s: %s\n", dump_vel,
                strerror(errno));
        return;
    }
    if (fwrite(mixed, sizeof(float), latent_count, out) != latent_count) {
        fprintf(stderr, "h3 wan22: warning: %s is short\n", dump_vel);
    }
    fclose(out);
}

int h3_wan22_pipeline_sample(h3_wan22_pipeline *pipeline, float *latents,
                             uint32_t frames, uint32_t h, uint32_t w,
                             const float *context_cond, uint32_t ctx_cond_len,
                             const float *context_uncond,
                             uint32_t ctx_uncond_len, uint32_t steps,
                             float shift, float guide_scale,
                             h3_wan22_solver_kind solver, char *error,
                             size_t error_size) {
    const size_t latent_count = (size_t)WAN22_LATENT_CH * frames * h * w;
    float *timesteps = malloc(((size_t)steps + 1) * sizeof(float));
    float *v_cond = malloc(latent_count * sizeof(float));
    float *v_uncond = malloc(latent_count * sizeof(float));
    float *mixed = malloc(latent_count * sizeof(float));
    h3_wan22_unipc *unipc = NULL;
    if (!timesteps || !v_cond || !v_uncond || !mixed) {
        free(timesteps);
        free(v_cond);
        free(v_uncond);
        free(mixed);
        snprintf(error, error_size, "out of host memory (sampling)");
        return 0;
    }

    uint32_t count;
    if (solver == H3_WAN22_SOLVER_UNIPC) {
        unipc = h3_wan22_unipc_create(steps, shift, latent_count, error,
                                      error_size);
        if (!unipc) {
            free(timesteps);
            free(v_cond);
            free(v_uncond);
            free(mixed);
            return 0;
        }
        count = h3_wan22_unipc_step_count(unipc);
        h3_wan22_unipc_timesteps(unipc, timesteps);
    } else {
        count = euler_timesteps(timesteps, steps, shift);
    }

    wan22_trace_stats("init", 0, timesteps[0], latents, latent_count, NULL, NULL);
    for (uint32_t step = 0; step < count; step++) {
        if (!wan22_cfg_velocity(pipeline, latents, frames, h, w, timesteps[step],
                                context_cond, ctx_cond_len, context_uncond,
                                ctx_uncond_len, guide_scale, v_cond, v_uncond,
                                mixed, latent_count, error, error_size)) {
            goto fail;
        }
        wan22_maybe_dump_velocity(step, mixed, latent_count);
        if (unipc) {
            h3_wan22_unipc_step(unipc, latents, mixed);
        } else {
            for (size_t i = 0; i < latent_count; i++) {
                /* EulerScheduler.step: dt is the gap to the next timestep, and
                 * the final step's dt is the last timestep itself (the
                 * transformed trailing 0 that set_timesteps dropped). */
                const float dt =
                    step + 1 < count
                        ? (timesteps[step] - timesteps[step + 1]) / 1000.0f
                        : timesteps[step] / 1000.0f;
                latents[i] -= mixed[i] * dt;
            }
        }
        wan22_trace_stats("after", step, timesteps[step], latents, latent_count,
                          v_cond, v_uncond);
    }
    h3_wan22_unipc_free(unipc);
    free(timesteps);
    free(v_cond);
    free(v_uncond);
    free(mixed);
    return 1;

fail:
    h3_wan22_unipc_free(unipc);
    free(timesteps);
    free(v_cond);
    free(v_uncond);
    free(mixed);
    return 0;
}

int h3_wan22_pipeline_release_dit(h3_wan22_pipeline *pipeline, char *error,
                                  size_t error_size) {
    if (!pipeline->gpu) return 1;
    return h3_wan22_gpu_state_release(pipeline->blocks, WAN22_BLOCKS,
                                      pipeline->gpu, error, error_size);
}

const char *h3_wan22_pipeline_dit_device(const h3_wan22_pipeline *pipeline) {
    if (!pipeline || !pipeline->gpu) return "cpu";
    return h3_wan22_gpu_state_resident(pipeline->gpu) ? "gpu" : "released";
}

/* Times one resume: the upload is the only per-request cost the release /
 * resume trade-off adds, so log it.  Serve warms the state eagerly right
 * after the decode (h3_serve.c), so the cost lands in that request's vae
 * phase instead of the next request's first forward; a caller that does not
 * warm still gets the lazy resume inside its first forward. */
int h3_wan22_pipeline_resume_dit(h3_wan22_pipeline *pipeline, char *error,
                                 size_t error_size) {
    if (!pipeline->gpu) return 1;
    const int reupload = !h3_wan22_gpu_state_resident(pipeline->gpu);
    const double started = reupload ? wan22_pipeline_now() : 0.0;
    if (!h3_wan22_gpu_state_resume(pipeline->blocks, WAN22_BLOCKS,
                                   pipeline->gpu, pipeline->gpu_seq,
                                   pipeline->gpu_ctx_seq, error,
                                   error_size)) {
        return 0;
    }
    if (reupload) {
        h3_gpu_stats stats = {0};
        h3_gpu_get_stats(h3_wan22_gpu_handle(pipeline->gpu), &stats);
        fprintf(stderr,
                "wan22: DiT device state resumed (%.2fs, %.2f GiB live)\n",
                wan22_pipeline_now() - started,
                (double)stats.live_bytes / (double)(1u << 30));
        fflush(stderr);
    }
    return 1;
}

/* The DiT and the VAE cannot be device-resident together on a 24 GB card:
 * the BF16 DiT is ~9.8 GiB and the streaming VAE decode peaks at 15.2 GiB
 * live / 21.3 GiB reserved at 720p.  The device state is therefore dropped
 * at the phase boundary; the next forward re-uploads it lazily (see
 * h3_wan22_pipeline_forward), which is what makes a second request on the
 * same pipeline work.  H3_WAN22_FREE_DIT_BEFORE_DECODE=0 restores the old
 * keep-it-resident behaviour (only useful when the VAE runs on the CPU).
 */
static int wan22_free_dit_before_decode(void) {
    const char *value = getenv("H3_WAN22_FREE_DIT_BEFORE_DECODE");
    return !(value && strcmp(value, "0") == 0);
}

int h3_wan22_pipeline_decode(h3_wan22_pipeline *pipeline,
                             const float *latents, uint32_t frames,
                             uint32_t h, uint32_t w, float *out,
                             char *error, size_t error_size) {
    h3_gpu_stats before = {0}, after = {0};
    const int release = pipeline->gpu && wan22_free_dit_before_decode();
    if (release) {
        h3_gpu_get_stats(h3_wan22_gpu_handle(pipeline->gpu), &before);
        if (!h3_wan22_pipeline_release_dit(pipeline, error, error_size)) {
            return 0;
        }
        h3_gpu_get_stats(h3_wan22_gpu_handle(pipeline->gpu), &after);
        fprintf(stderr,
                "wan22: DiT device state released before the VAE decode "
                "(%.2f -> %.2f GiB live)\n",
                (double)before.live_bytes / (double)(1u << 30),
                (double)after.live_bytes / (double)(1u << 30));
        fflush(stderr);
    }
    return h3_wan22_vae_decode(pipeline->vae, latents, frames, h, w, out,
                               error, error_size);
}
