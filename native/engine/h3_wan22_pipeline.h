#ifndef H3_WAN22_PIPELINE_H
#define H3_WAN22_PIPELINE_H

#include "h3_wan22_dit.h"
#include "h3_wan22_solver.h"
#include "h3_wan22_vae.h"

#include <stddef.h>
#include <stdint.h>

/* Wan 2.2 T2V 5B Phase 2: full pipeline assembly (CPU fp32 reference path).
 *
 *   text (umt5-xxl, h3_wan22_text)  ->  context [seq, 4096]
 *   text_embedding MLP (in this module, DiT checkpoint BF16)
 *   patch_embedding (Conv3d 48 -> 3072, k = stride = (1, 2, 2), F32)
 *   timestep chain: sinusoidal(256) -> time_embedding (F32) -> e,
 *                   SiLU -> time_projection (F32) -> e0 [6, 3072]
 *   30 x h3_wan22_block_forward (cross-attn on the embedded context)
 *   Head (h3_wan22_head) -> [seq, 192] -> unpatchify -> velocity
 *   [48, T, h, w]
 *   Euler sampling (flow matching), shift-transformed timesteps, CFG
 *   noise_pred = uncond + guide_scale * (cond - uncond)
 *   VAE decode (h3_wan22_vae) -> pixels, caller clamps to [-1, 1]
 *
 * The text encoder is deliberately NOT owned here: callers encode their
 * prompts with h3_wan22_text (and can free it) before loading this module,
 * which keeps the umt5 (~11 GB) and DiT+VAE (~12 GB) residents disjoint.
 *
 * Token layout matches models/wan/modules/model.py: patchify flattens
 * (f, h, w) f-major; the patch vector index is (c, ph, pw) = c*4 + ph*2 + pw
 * for the 1x2x2 patches on both the input (patch_embedding weight is stored
 * [3072, 48, 1, 2, 2]) and the Head output (192 = 48 * 2 * 2, index
 * q*96 + r*48 + ch).
 */

typedef struct h3_wan22_pipeline h3_wan22_pipeline;

/* dit_ckpt_dir: single-file directory (symlink) with the quanto qint8 DiT
 * checkpoint; vae_path: the Wan2.2_VAE.safetensors file. */
h3_wan22_pipeline *h3_wan22_pipeline_load(const char *dit_ckpt_dir,
                                          const char *vae_path,
                                          const h3_wan22_config *config,
                                          char *error, size_t error_size);
void h3_wan22_pipeline_free(h3_wan22_pipeline *pipeline);

/* One DiT forward.  latents [48, T, h, w] fp32 (h, w even), t the raw
 * (transformed) timestep, context [ctx_seq, 4096] raw encoder output
 * (text_embedding is applied inside), velocity [48, T, h, w] output. */
int h3_wan22_pipeline_forward(h3_wan22_pipeline *pipeline,
                              const float *latents, uint32_t frames,
                              uint32_t h, uint32_t w, float t,
                              const float *context, uint32_t ctx_seq,
                              float *velocity, char *error,
                              size_t error_size);

/* Flow-matching sampling in place: latents [48, T, h, w] go from noise to
 * the denoised x0.  contexts[0] = cond, contexts[1] = uncond,
 * each [ctx_len, 4096] with its own row count: WanGP truncates every
 * context to that prompt's real token count (models/wan/modules/t5.py)
 * and the two CFG branches are evaluated separately, so their lengths
 * differ and neither is padded to the 512-token encoder window.  steps
 * >= 1, shift 5, guide_scale 5 per the WanGP defaults.
 *
 * solver picks the integrator: H3_WAN22_SOLVER_UNIPC is what WanGP runs for
 * this checkpoint (sample_solver = "unipc"), H3_WAN22_SOLVER_EULER is
 * EulerScheduler.set_timesteps + step (shared/utils/euler_scheduler.py), kept
 * because the 121-frame acceptance run and the fixture gates were measured on
 * it.  The two use different schedules: Euler ends at transform(1) ~ 4.98 with
 * the trailing zero dropped, UniPC ends at sigma 0. */
int h3_wan22_pipeline_sample(h3_wan22_pipeline *pipeline, float *latents,
                             uint32_t frames, uint32_t h, uint32_t w,
                             const float *context_cond, uint32_t ctx_cond_len,
                             const float *context_uncond,
                             uint32_t ctx_uncond_len, uint32_t steps,
                             float shift, float guide_scale,
                             h3_wan22_solver_kind solver, char *error,
                             size_t error_size);

/* text_embedding MLP forward (exposed for the numerics gate):
 * context [ctx_seq, 4096] raw encoder output -> [ctx_seq, 3072]. */
int h3_wan22_pipeline_text_embedding(h3_wan22_pipeline *pipeline,
                                     const float *context,
                                     uint32_t ctx_seq, float *out);

/* Assemble one CFG branch's context plane the way WanGP does it: the branch's
 * real `rows` encoder rows first, zeros for everything up to `ctx_seq`.
 *
 * This is not a formatting detail.  WanGP truncates a prompt's context to its
 * real token count and then pads it straight back to model.text_len with
 * zeros (models/wan/any2video.py:600-604, text_len = 512), so the DiT always
 * cross-attends over 512 rows.  Feeding the raw per-branch lengths instead
 * (4 rows for "a dancing bear", 126 for the default negative prompt) makes
 * every wan22 video come out as noise -- see PLAN_RTX4090.md 5.4.
 *
 * `plane` must hold ctx_seq * dim floats and `source` at least rows * dim.
 * Returns 0 if rows > window > ctx_seq, 1 otherwise. */
int h3_wan22_context_assemble(float *plane, uint32_t ctx_seq, uint32_t dim,
                              const float *source, uint32_t rows,
                              uint32_t window);

/* Drop / rebuild the DiT device state (BF16 weights + scratch, ~12 GiB at
 * 720p).  Both are no-ops on the CPU path and idempotent on the GPU path;
 * the decode drops the state itself and the next forward rebuilds it, so
 * these exist for callers that place the phase boundary elsewhere (or for
 * the release/resume bit-exactness gate). */
int h3_wan22_pipeline_release_dit(h3_wan22_pipeline *pipeline, char *error,
                                  size_t error_size);
int h3_wan22_pipeline_resume_dit(h3_wan22_pipeline *pipeline, char *error,
                                 size_t error_size);

/* Where the DiT runs right now, for the serve health report:
 *   "cpu"      the reference path (GPU state never allocated)
 *   "gpu"      BF16 weights + scratch device-resident
 *   "released" GPU path, but the state was dropped for the VAE decode; the
 *              next forward re-uploads it (one ~9.8 GiB PCIe transfer). */
const char *h3_wan22_pipeline_dit_device(const h3_wan22_pipeline *pipeline);

/* VAE decode: latents [48, T, h, w] -> pixels [3, 4T-3, 16h, 16w] fp32
 * (pre clamp).
 *
 * With the GPU DiT path on, this releases the DiT device state first
 * (~9.8 GiB of BF16 weights + scratch) because the streaming VAE decode
 * peaks above what is left on a 24 GB card; the next forward resumes it.
 * H3_WAN22_FREE_DIT_BEFORE_DECODE=0 keeps the old behaviour. */
int h3_wan22_pipeline_decode(h3_wan22_pipeline *pipeline,
                             const float *latents, uint32_t frames,
                             uint32_t h, uint32_t w, float *out,
                             char *error, size_t error_size);

#endif
