#ifndef H3_WAN22_SOLVER_H
#define H3_WAN22_SOLVER_H

#include <stddef.h>
#include <stdint.h>

/* Wan 2.2 samplers.
 *
 * WanGP runs the wan family with sample_solver = "unipc"
 * (models/wan/wan_handler.py:1225), so an engine that only has Euler is not
 * running the same sampling chain as the reference -- it integrates the same
 * velocity field with a first-order method and calls it parity.
 *
 * This file ports the one piece that is pure arithmetic:
 * shared/utils/fm_solvers_unipc.py, FlowUniPCMultistepScheduler, with the
 * configuration any2video.py:523 builds (solver_order 2, solver_type "bh2",
 * predict_x0, flow_prediction, lower_order_final, final_sigmas_type "zero",
 * no thresholding, no corrector disabled).  It owns no model and no device
 * state, so it can be gated against the real Python class on a synthetic
 * velocity field (tests/test_wan22_unipc.c, tools/ref_unipc.py).
 */

typedef enum h3_wan22_solver_kind {
    H3_WAN22_SOLVER_EULER = 0,
    H3_WAN22_SOLVER_UNIPC = 1
} h3_wan22_solver_kind;

/* "euler" / "unipc", case-insensitive.  Returns 0 when the name is not one of
 * them, so a typo in a request is a 400 and not a silent fallback. */
int h3_wan22_solver_parse(const char *name, h3_wan22_solver_kind *out);
const char *h3_wan22_solver_name(h3_wan22_solver_kind kind);

/* The integrator a request gets when it names none.  It lives here rather than
 * as a literal at the call site so that "the default is the one WanGP runs" is
 * one value that a test can pin (tests/test_serve_base64.c) and that
 * tools/wan22_solver_parity.sh can verify end to end. */
h3_wan22_solver_kind h3_wan22_solver_default(void);

typedef struct h3_wan22_unipc h3_wan22_unipc;

/* steps >= 1, count = element count of one latent.  Allocates the history the
 * multistep update needs (three copies of the latent plus the schedule). */
h3_wan22_unipc *h3_wan22_unipc_create(uint32_t steps, float shift,
                                      size_t count, char *error,
                                      size_t error_size);
void h3_wan22_unipc_free(h3_wan22_unipc *solver);

/* Number of model evaluations, i.e. how many timesteps the loop runs. */
uint32_t h3_wan22_unipc_step_count(const h3_wan22_unipc *solver);

/* Timesteps as the model sees them: sigmas[:-1] * 1000 truncated to integers,
 * because the reference scheduler stores timesteps as int64 and hands those to
 * the DiT while keeping the unrounded sigmas for its own arithmetic. */
void h3_wan22_unipc_timesteps(const h3_wan22_unipc *solver, float *out);

/* The sigma schedule, steps + 1 values, the last one 0 (final_sigmas_type
 * "zero").  Exposed for the gate; the sampler itself is the only user. */
void h3_wan22_unipc_sigmas(const h3_wan22_unipc *solver, float *out);

/* One predictor-corrector step, in place.  velocity is the CFG-mixed model
 * output for the current timestep. */
void h3_wan22_unipc_step(h3_wan22_unipc *solver, float *latents,
                         const float *velocity);

#endif
