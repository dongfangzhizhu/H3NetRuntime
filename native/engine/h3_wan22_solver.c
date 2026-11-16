#include "h3_wan22_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Port of shared/utils/fm_solvers_unipc.py (WanGP), which is diffusers'
 * UniPCMultistepScheduler specialised for flow matching.  The configuration is
 * the one any2video.py builds for wan: solver_order 2, solver_type "bh2",
 * predict_x0 = True, prediction_type "flow_prediction", lower_order_final =
 * True, final_sigmas_type "zero", thresholding off, disable_corrector empty.
 *
 * Everything below keeps the reference's variable names, including the ones
 * that read oddly here (m0/m1 are x0 predictions, not velocities; hh is -h
 * because predict_x0 flips the sign of the exponential integral), because the
 * only way to audit a port is to be able to put the two side by side.
 */

struct h3_wan22_unipc {
    uint32_t steps;
    size_t count;

    float *sigmas;      /* [steps + 1], last is 0 */
    float *timesteps;   /* [steps], truncated as the model sees them */

    float *x0;          /* this step's x0 prediction, before the history shift */
    float *d1;          /* the one difference vector a 2nd-order step needs */
    float *m0;          /* model_outputs[-2]: x0 of two steps ago */
    float *m1;          /* model_outputs[-1]: x0 of the previous step */
    float *last_sample; /* x before the previous predictor */

    uint32_t index;
    uint32_t lower_order_nums;
    uint32_t this_order;
    int have_last;
};

int h3_wan22_solver_parse(const char *name, h3_wan22_solver_kind *out) {
    if (!name || !out) return 0;
    if (strcmp(name, "euler") == 0 || strcmp(name, "EULER") == 0) {
        *out = H3_WAN22_SOLVER_EULER;
        return 1;
    }
    if (strcmp(name, "unipc") == 0 || strcmp(name, "UNIPC") == 0 ||
        strcmp(name, "uniPC") == 0) {
        *out = H3_WAN22_SOLVER_UNIPC;
        return 1;
    }
    return 0;
}

const char *h3_wan22_solver_name(h3_wan22_solver_kind kind) {
    return kind == H3_WAN22_SOLVER_UNIPC ? "unipc" : "euler";
}

h3_wan22_solver_kind h3_wan22_solver_default(void) {
    return H3_WAN22_SOLVER_UNIPC;
}

/* lambda = log(alpha) - log(sigma) with alpha = 1 - sigma.  sigma reaches 0 on
 * the last entry of the schedule, where this is +inf -- deliberately: the
 * predictor then collapses to x0, which is what final_sigmas_type "zero"
 * means.  The arithmetic downstream stays finite because lower_order_final
 * makes that step first order (see the guard in unipc_predict). */
static double sigma_lambda(double sigma) {
    return log(1.0 - sigma) - log(sigma);
}

/* Solve R rhos = b for the 2x2 the second-order corrector builds, with partial
 * pivoting so it matches torch.linalg.solve (LU) rather than an ad-hoc
 * division that would blow up when rk is large. */
static void solve2(const double r[2][2], const double b[2], double out[2]) {
    double a[2][2] = {{r[0][0], r[0][1]}, {r[1][0], r[1][1]}};
    double y[2] = {b[0], b[1]};
    if (fabs(a[1][0]) > fabs(a[0][0])) {
        const double t0 = a[0][0], t1 = a[0][1];
        a[0][0] = a[1][0];
        a[0][1] = a[1][1];
        a[1][0] = t0;
        a[1][1] = t1;
        const double ty = y[0];
        y[0] = y[1];
        y[1] = ty;
    }
    if (a[0][0] == 0.0) {
        out[0] = 0.0;
        out[1] = 0.0;
        return;
    }
    const double factor = a[1][0] / a[0][0];
    const double pivot = a[1][1] - factor * a[0][1];
    out[1] = pivot != 0.0 ? (y[1] - factor * y[0]) / pivot : 0.0;
    out[0] = (y[0] - a[0][1] * out[1]) / a[0][0];
}

h3_wan22_unipc *h3_wan22_unipc_create(uint32_t steps, float shift,
                                      size_t count, char *error,
                                      size_t error_size) {
    if (steps < 1 || count == 0) {
        if (error) snprintf(error, error_size, "unipc: bad geometry");
        return NULL;
    }
    h3_wan22_unipc *solver = calloc(1, sizeof(*solver));
    if (!solver) {
        if (error) snprintf(error, error_size, "unipc: out of memory");
        return NULL;
    }
    solver->steps = steps;
    solver->count = count;
    solver->sigmas = malloc(((size_t)steps + 1) * sizeof(float));
    solver->timesteps = malloc((size_t)steps * sizeof(float));
    solver->x0 = malloc(count * sizeof(float));
    solver->d1 = malloc(count * sizeof(float));
    solver->m0 = malloc(count * sizeof(float));
    solver->m1 = malloc(count * sizeof(float));
    solver->last_sample = malloc(count * sizeof(float));
    if (!solver->sigmas || !solver->timesteps || !solver->x0 || !solver->d1 ||
        !solver->m0 || !solver->m1 || !solver->last_sample) {
        h3_wan22_unipc_free(solver);
        if (error) snprintf(error, error_size, "unipc: out of memory");
        return NULL;
    }

    /* set_timesteps: linspace(sigma_max, sigma_min, steps + 1)[:-1] with
     * sigma_max = 0.999, sigma_min = 0 (1 - 1/1000 after the constructor's
     * 1..1/1000 ramp is reversed and subtracted from one), then the shift,
     * then a trailing 0 for the final sigma.  The interval count is `steps`,
     * not steps - 1: the schedule is sampled at steps + 1 points and the last
     * one is dropped, which is what makes the first eight of nine values
     * 0.999 ... 0.1249 rather than 0.999 ... 0. */
    const double sigma_max = 0.999, sigma_min = 0.0;
    for (uint32_t i = 0; i < steps; i++) {
        const double raw = sigma_max + (sigma_min - sigma_max) *
                                            (double)i / (double)steps;
        const double shifted =
            (double)shift * raw / (1.0 + ((double)shift - 1.0) * raw);
        solver->sigmas[i] = (float)shifted;
        solver->timesteps[i] = (float)(long long)(shifted * 1000.0);
    }
    solver->sigmas[steps] = 0.0f;
    return solver;
}

void h3_wan22_unipc_free(h3_wan22_unipc *solver) {
    if (!solver) return;
    free(solver->sigmas);
    free(solver->timesteps);
    free(solver->x0);
    free(solver->d1);
    free(solver->m0);
    free(solver->m1);
    free(solver->last_sample);
    free(solver);
}

uint32_t h3_wan22_unipc_step_count(const h3_wan22_unipc *solver) {
    return solver ? solver->steps : 0;
}

void h3_wan22_unipc_timesteps(const h3_wan22_unipc *solver, float *out) {
    if (solver && out) memcpy(out, solver->timesteps,
                              (size_t)solver->steps * sizeof(float));
}

void h3_wan22_unipc_sigmas(const h3_wan22_unipc *solver, float *out) {
    if (solver && out) memcpy(out, solver->sigmas,
                              ((size_t)solver->steps + 1) * sizeof(float));
}

/* multistep_uni_c_bh_update, predict_x0 branch.  Rewrites x in place. */
static void unipc_correct(h3_wan22_unipc *solver, float *x) {
    const uint32_t order = solver->this_order ? solver->this_order : 1;
    const size_t count = solver->count;
    const double sigma_t = solver->sigmas[solver->index];
    const double sigma_s0 = solver->sigmas[solver->index - 1];
    const double alpha_t = 1.0 - sigma_t;
    const double alpha_s0 = 1.0 - sigma_s0;
    (void)alpha_s0;
    const double lambda_t = sigma_lambda(sigma_t);
    const double lambda_s0 = sigma_lambda(sigma_s0);
    const double h = lambda_t - lambda_s0;

    double rks[2] = {0.0, 1.0};
    int terms = 0;
    for (uint32_t i = 1; i < order; i++) {
        const uint32_t si = solver->index - (i + 1);
        const float *mi = i == 1 ? solver->m0 : solver->m1;
        const double lambda_si = sigma_lambda(solver->sigmas[si]);
        const double rk = (lambda_si - lambda_s0) / h;
        rks[i - 1] = rk;
        if (terms == 0) {
            /* One difference vector is all a 2nd-order corrector needs; write
             * it into d1 and reuse it below. */
            const double inverse = rk != 0.0 ? 1.0 / rk : 0.0;
            for (size_t k = 0; k < count; k++) {
                solver->d1[k] = (float)(((double)mi[k] - (double)solver->m1[k]) *
                                        inverse);
            }
            terms = 1;
        }
    }

    const double hh = -h;                     /* predict_x0 */
    const double h_phi_1 = expm1(hh);
    const double b_h = expm1(hh);             /* solver_type "bh2" */
    double h_phi_k = h_phi_1 / hh - 1.0;
    double factorial_i = 1.0;
    double r[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double b[2] = {0.0, 0.0};
    for (uint32_t i = 1; i <= order; i++) {
        for (uint32_t j = 0; j < order; j++) {
            r[i - 1][j] = pow(rks[j], (double)(i - 1));
        }
        b[i - 1] = h_phi_k * factorial_i / b_h;
        factorial_i *= (double)(i + 1);
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
    }

    double rhos[2] = {0.0, 0.0};
    if (order == 1) {
        rhos[0] = 0.5;                        /* simplified 1st order */
    } else {
        solve2(r, b, rhos);
    }

    for (size_t k = 0; k < count; k++) {
        const double x_t = sigma_t / sigma_s0 * (double)solver->last_sample[k] -
                           alpha_t * h_phi_1 * (double)solver->m1[k];
        double corr_res = 0.0;
        if (terms > 0) corr_res = rhos[0] * (double)solver->d1[k];
        const double d1_t = (double)solver->x0[k] - (double)solver->m1[k];
        x[k] = (float)(x_t - alpha_t * b_h * (corr_res + rhos[order - 1] * d1_t));
    }
}

/* multistep_uni_p_bh_update, predict_x0 branch.  Rewrites x in place. */
static void unipc_predict(h3_wan22_unipc *solver, float *x, uint32_t order) {
    const size_t count = solver->count;
    const double sigma_t = solver->sigmas[solver->index + 1];
    const double sigma_s0 = solver->sigmas[solver->index];
    const double alpha_t = 1.0 - sigma_t;
    const double lambda_t = sigma_lambda(sigma_t);
    const double lambda_s0 = sigma_lambda(sigma_s0);
    const double h = lambda_t - lambda_s0;

    double rks[2] = {0.0, 1.0};
    int terms = 0;
    for (uint32_t i = 1; i < order; i++) {
        const uint32_t si = solver->index - i;
        const float *mi = i == 1 ? solver->m0 : solver->m1;
        const double lambda_si = sigma_lambda(solver->sigmas[si]);
        const double rk = (lambda_si - lambda_s0) / h;
        rks[i - 1] = rk;
        if (terms == 0) {
            const double inverse = rk != 0.0 ? 1.0 / rk : 0.0;
            for (size_t k = 0; k < count; k++) {
                solver->d1[k] = (float)(((double)mi[k] - (double)solver->m1[k]) *
                                        inverse);
            }
            terms = 1;
        }
    }

    const double hh = -h;
    const double h_phi_1 = expm1(hh);
    const double b_h = expm1(hh);
    double h_phi_k = h_phi_1 / hh - 1.0;
    double factorial_i = 1.0;
    double r[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    double b[2] = {0.0, 0.0};
    for (uint32_t i = 1; i <= order; i++) {
        for (uint32_t j = 0; j < order; j++) {
            r[i - 1][j] = pow(rks[j], (double)(i - 1));
        }
        b[i - 1] = h_phi_k * factorial_i / b_h;
        factorial_i *= (double)(i + 1);
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i;
    }

    double rhos[2] = {0.0, 0.0};
    if (order == 1) {
        /* No D1 terms, so the coefficients never enter the update; keep the
         * branch explicit rather than pretending to solve a 0x0 system. */
    } else if (order == 2) {
        rhos[0] = 0.5;                        /* simplified 2nd order */
    } else {
        solve2(r, b, rhos);
    }

    for (size_t k = 0; k < count; k++) {
        const double x_t = sigma_t / sigma_s0 * (double)x[k] -
                           alpha_t * h_phi_1 * (double)solver->m1[k];
        double pred_res = 0.0;
        if (order == 2 && terms > 0) pred_res = 0.5 * (double)solver->d1[k];
        x[k] = (float)(x_t - alpha_t * b_h * pred_res);
    }
}

void h3_wan22_unipc_step(h3_wan22_unipc *solver, float *latents,
                         const float *velocity) {
    const size_t count = solver->count;
    const double sigma_t = solver->sigmas[solver->index];

    /* convert_model_output: x0 = sample - sigma_t * model_output. */
    for (size_t k = 0; k < count; k++) {
        solver->x0[k] = (float)((double)latents[k] -
                                sigma_t * (double)velocity[k]);
    }

    if (solver->index > 0 && solver->have_last) {
        unipc_correct(solver, latents);
    }

    memcpy(solver->m0, solver->m1, count * sizeof(float));
    memcpy(solver->m1, solver->x0, count * sizeof(float));

    uint32_t order = solver->steps - solver->index;   /* lower_order_final */
    if (order > 2) order = 2;
    if (order > solver->lower_order_nums + 1) order = solver->lower_order_nums + 1;
    if (order < 1) order = 1;
    solver->this_order = order;

    memcpy(solver->last_sample, latents, count * sizeof(float));
    solver->have_last = 1;

    unipc_predict(solver, latents, order);

    if (solver->lower_order_nums < 2) solver->lower_order_nums++;
    solver->index++;
}
