#ifndef H3_HOST_H
#define H3_HOST_H

#include <stddef.h>
#include <stdint.h>

#define H3_CANVAS_MULTIPLE 32
#define H3_MAX_PIXELS (768 * 1344)
#define H3_FPS 24
#define H3_AUDIO_LATENT_FPS 40
#define H3_VAE_SPATIAL_RATIO 16
#define H3_VIDEO_SIGMA_SHIFT 12.0
#define H3_AUDIO_SIGMA_SHIFT 3.0
#define H3_MAX_STEPS 1000
/* PDD (progressive denoise distillation): the sampler takes 8 coarse steps
 * while the network is evaluated on a 32-step fine grid, so every coarse step
 * needs the share of each fine interval it covers. */
#define H3_PDD_NUM_STEPS 32
#define H3_PDD_BLOCK_SIZE 4


typedef struct {
    int frame_count;
    int video_t;
    int audio_t;
} h3_temporal_shape;

typedef struct {
    double t;
    double h;
    double w;
} h3_position;

typedef enum {
    H3_SEG_TEXT,
    H3_SEG_COND,
    H3_SEG_REF_IMAGE,
    H3_SEG_REF_AUDIO,
    H3_SEG_AUDIO,
    H3_SEG_VIDEO
} h3_segment_kind;

typedef struct {
    size_t start;
    size_t stop;
    h3_segment_kind kind;
} h3_segment;

typedef enum {
    H3_LAYOUT_REF_IMAGE,
    H3_LAYOUT_REF_AUDIO,
    H3_LAYOUT_REF_VIDEO
} h3_layout_ref_kind;

typedef struct {
    h3_layout_ref_kind kind;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
} h3_layout_ref;

typedef struct {
    int text_len;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    int frame_count;
    const int *keyframes;
    size_t keyframe_count;
    const h3_layout_ref *references;
    size_t reference_count;
} h3_layout_spec;

typedef struct {
    size_t seq_len;
    h3_segment *segments;
    size_t segment_count;
    h3_position *positions;
    size_t img_cond_rows;
    size_t img_target_rows;
    size_t audio_cond_rows;
    size_t audio_target_rows;
    int signature[5];
} h3_layout;

typedef struct {
    int steps;
    float video[H3_MAX_STEPS + 1];
    float audio[H3_MAX_STEPS + 1];
} h3_sigma_schedule;

typedef struct {
    uint64_t state;
    uint64_t increment;
    float spare;
    int has_spare;
} h3_rng;

int h3_align_frame_count(int requested);
int h3_video_latent_t(int frame_count);
int h3_video_encoder_latent_t(int frame_count);
h3_temporal_shape h3_temporal(int requested_frames);
void h3_latent_canvas(int width, int height, int *latent_w, int *latent_h);
int h3_adapt_canvas(int width, int height, int *adapted_w, int *adapted_h);
/* Ref2VA image sizing is down-only and aspect preserving. max_short_edge=0
 * matches the output pixel area; a positive value selects a short-edge cap. */
int h3_reference_image_canvas(int width, int height,
                              int target_width, int target_height,
                              int max_short_edge,
                              int *adapted_w, int *adapted_h);
/* Ref2VA video references use the normal target-style canvas, except that a
 * smaller source is never enlarged. */
int h3_reference_video_canvas(int width, int height,
                              int *adapted_w, int *adapted_h);

double h3_time_shift_sigma(double sigma, double from_shift, double to_shift);
double h3_time_shift_slope(double sigma, double from_shift, double to_shift);
int h3_schedule_build(int steps, h3_sigma_schedule *schedule);
/* Released linear base grid: evaluations model forwards plus terminal zero. */
int h3_serving_schedule_build(int evaluations, h3_sigma_schedule *schedule);
/* HyperFlow Flash's own 4/6/8-step interval grid (video shift 12, audio 3).
 * A checkpoint that carries hyperflow_sigmas must use this instead of the
 * uniform grid: its distilled weights were trained on these exact knots. */
int h3_hyperflow_schedule_build(int steps, h3_sigma_schedule *schedule);
/* PDD's 8-step schedule (shift 12 video, 3 audio), fixed at
 * H3_PDD_NUM_STEPS / H3_PDD_BLOCK_SIZE steps. */
int h3_pdd_schedule_build(h3_sigma_schedule *schedule);
/* Per-step overlap of the coarse schedule with the fine grid the network is
 * evaluated on, row-major [coarse_steps][num_steps] float, every row summing
 * to 1. The channel is named explicitly by the caller -- pass sigmas->video
 * with H3_VIDEO_SIGMA_SHIFT or sigmas->audio with H3_AUDIO_SIGMA_SHIFT -- so
 * nothing is inferred from the shift value. `coarse_sigmas` holds
 * coarse_steps + 1 values (terminal zero included) and must be strictly
 * descending. Returns 0 on a non-descending grid, on a step that covers no
 * fine interval, or on inconsistent arguments. */
int h3_pdd_plans_build(const float *coarse_sigmas, uint32_t coarse_steps,
                       double shift, uint32_t num_steps, uint32_t block_size,
                       float *plans);

int h3_layout_build(const h3_layout_spec *spec, h3_layout *layout,
                    char *error, size_t error_size);
void h3_layout_free(h3_layout *layout);
const char *h3_segment_name(h3_segment_kind kind);

void h3_rng_seed(h3_rng *rng, uint64_t seed);
uint32_t h3_rng_u32(h3_rng *rng);
float h3_rng_normal(h3_rng *rng);
void h3_rng_fill_normal(h3_rng *rng, float *values, size_t count);

/* Resize interleaved RGB24 frames with Accelerate/vImage high-quality
 * resampling. The caller owns *output. Identity geometry still returns an
 * independent copy. */
int h3_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output);

int h3_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, size_t count,
                const float *sigmas, int step, int total_steps);
int h3_euler_velocity_step(float *sample, const float *velocity, size_t count,
                           float sigma, float sigma_next);

/* The root directory for engine scratch files.  Paths built from it are handed
 * to fopen, to the PNG writer and (for video) to ffmpeg as an argv entry, so it
 * has to be a path the operating system itself accepts: /tmp on POSIX, %TEMP%
 * on Windows.  Never NULL; the returned pointer stays valid for the process. */
const char *h3_temp_dir(void);

#endif
