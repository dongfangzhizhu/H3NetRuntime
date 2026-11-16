#ifndef H3_PDD_HEADS_H
#define H3_PDD_HEADS_H

#include "h3_gpu.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

/* Per-step output heads of the PDD 8-step release
 * (minimax_h3_fl2va_pdd_8step_heads_bf16.safetensors):
 *   pdd_heads.video_out.weight BF16 [32, 96, 5376]   .bias BF16 [32, 96]
 *   pdd_heads.audio_out.weight BF16 [32, 32, 5376]   .bias BF16 [32, 32]
 *
 * The 32 trained heads live on the fine grid; a sampling step mixes them with
 * the plan row the PDD sampler derives from the coarse schedule, so the head
 * that runs at step s is
 *     W_eff(s) = sum_n plan[s][n] * W[n]     b_eff(s) = sum_n plan[s][n] * b[n]
 * with every plan row summing to 1. Fusing all steps once at load time costs
 * 8 * 128 * 5376 multiply-adds (trivial) and 11 MB of device memory, and keeps
 * the per-step path a plain BF16 linear. */
enum {
    H3_PDD_HEADS_TRAINED = 32,
    H3_PDD_VIDEO_DIM = 96,
    H3_PDD_AUDIO_DIM = 32,
    H3_PDD_HIDDEN = 5376,
};

typedef struct h3_pdd_heads h3_pdd_heads;

/* Non-zero when the store holds a complete set of PDD output heads. */
int h3_pdd_heads_available(const h3_weight_store *store);

/* Read the trained heads into host memory. trained_out (optional) receives
 * H3_PDD_HEADS_TRAINED. The heads only reach the device through
 * h3_pdd_heads_fuse_steps(), so loading alone costs no VRAM. */
h3_pdd_heads *h3_pdd_heads_load(const h3_weight_store *store, h3_gpu *gpu,
                                uint32_t *trained_out, char *error,
                                size_t error_size);
void h3_pdd_heads_free(h3_pdd_heads *heads);

/* Number of trained heads and, after fusion, the number of steps available. */
uint32_t h3_pdd_heads_trained(const h3_pdd_heads *heads);
uint32_t h3_pdd_heads_steps(const h3_pdd_heads *heads);

/* Mix the trained heads into one head per coarse step. video_plans and
 * audio_plans are row-major [coarse_steps][fine_steps] as produced by
 * h3_pdd_plans_build(); fine_steps must equal H3_PDD_HEADS_TRAINED, and every
 * row must sum to 1 (checked, so a transposed plan is rejected instead of
 * silently producing a wrong head). */
int h3_pdd_heads_fuse_steps(h3_pdd_heads *heads, const float *video_plans,
                            const float *audio_plans, uint32_t coarse_steps,
                            uint32_t fine_steps, char *error,
                            size_t error_size);

/* Fused head for one step; NULL when the head is not fused or the step is out
 * of range. h3_pdd_heads_video_step / _audio_step return the projection, and
 * the matching *_bias accessor returns its bias. */
const h3_gpu_tensor *h3_pdd_heads_video_step(const h3_pdd_heads *heads,
                                             uint32_t step);
const h3_gpu_tensor *h3_pdd_heads_audio_step(const h3_pdd_heads *heads,
                                             uint32_t step);
const h3_gpu_tensor *h3_pdd_heads_video_bias(const h3_pdd_heads *heads,
                                             uint32_t step);
const h3_gpu_tensor *h3_pdd_heads_audio_bias(const h3_pdd_heads *heads,
                                             uint32_t step);

#endif
