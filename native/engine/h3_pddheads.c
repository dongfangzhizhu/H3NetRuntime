#include "h3_pddheads.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PLAN_ROW_SUM_TOLERANCE_MILLI = 1,
};

struct h3_pdd_heads {
    h3_gpu *gpu;              /* borrowed */
    uint32_t trained;         /* heads in the file (fine grid) */
    uint32_t steps;           /* fused steps, 0 until fused */
    uint16_t *video_weight;   /* [trained][H3_PDD_VIDEO_DIM][hidden] */
    uint16_t *video_bias;     /* [trained][H3_PDD_VIDEO_DIM] */
    uint16_t *audio_weight;   /* [trained][H3_PDD_AUDIO_DIM][hidden] */
    uint16_t *audio_bias;     /* [trained][H3_PDD_AUDIO_DIM] */
    h3_gpu_tensor **fused_video_weight;
    h3_gpu_tensor **fused_video_bias;
    h3_gpu_tensor **fused_audio_weight;
    h3_gpu_tensor **fused_audio_bias;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float f32_from_bf16(uint16_t bits) {
    union { uint32_t word; float value; } converted;
    converted.word = (uint32_t)bits << 16u;
    return converted.value;
}

static uint16_t bf16_from_f32(float value) {
    union { uint32_t word; float value; } converted;
    converted.value = value;
    converted.word += 0x7fffu + ((converted.word >> 16u) & 1u);
    return (uint16_t)(converted.word >> 16u);
}

static const h3_st_tensor *find_head(const h3_weight_store *store,
                                     const char *name, int ndim,
                                     const uint64_t *shape,
                                     const h3_st_header **header,
                                     char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_weight_find(store, name, header);
    if (!tensor) {
        fail(error, error_size, "PDD heads are missing %s", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != ndim) {
        fail(error, error_size, "%s has dtype/rank %s/%d, expected BF16/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim, ndim);
        return NULL;
    }
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size,
                 "%s shape mismatch at dimension %d ([%llu], expected %llu)",
                 name, dimension, (unsigned long long)tensor->shape[dimension],
                 (unsigned long long)shape[dimension]);
            return NULL;
        }
    }
    return tensor;
}

static void *read_head(const h3_st_header *header, const h3_st_tensor *tensor,
                       const char *name, char *error, size_t error_size) {
    uint64_t elements = h3_st_tensor_elements(tensor);
    if (!elements || elements > SIZE_MAX / sizeof(uint16_t)) {
        fail(error, error_size, "%s is too large", name);
        return NULL;
    }
    size_t bytes = (size_t)elements * sizeof(uint16_t);
    uint16_t *host = malloc(bytes);
    if (!host) {
        fail(error, error_size, "out of memory reading %s", name);
        return NULL;
    }
    if (!h3_st_read_data(header, tensor, host, bytes, error, error_size)) {
        free(host);
        return NULL;
    }
    return host;
}

int h3_pdd_heads_available(const h3_weight_store *store) {
    static const char *const names[] = {
        "pdd_heads.video_out.weight", "pdd_heads.video_out.bias",
        "pdd_heads.audio_out.weight", "pdd_heads.audio_out.bias",
    };
    for (size_t index = 0; index < sizeof(names) / sizeof(names[0]); index++)
        if (!h3_weight_find(store, names[index], NULL)) return 0;
    return 1;
}

h3_pdd_heads *h3_pdd_heads_load(const h3_weight_store *store, h3_gpu *gpu,
                                uint32_t *trained_out, char *error,
                                size_t error_size) {
    if (trained_out) *trained_out = 0;
    if (!store || !gpu) {
        fail(error, error_size, "PDD heads need a weight store and a GPU");
        return NULL;
    }
    uint64_t video_weight_shape[] = {H3_PDD_HEADS_TRAINED, H3_PDD_VIDEO_DIM,
                                     H3_PDD_HIDDEN};
    uint64_t video_bias_shape[] = {H3_PDD_HEADS_TRAINED, H3_PDD_VIDEO_DIM};
    uint64_t audio_weight_shape[] = {H3_PDD_HEADS_TRAINED, H3_PDD_AUDIO_DIM,
                                     H3_PDD_HIDDEN};
    uint64_t audio_bias_shape[] = {H3_PDD_HEADS_TRAINED, H3_PDD_AUDIO_DIM};
    const h3_st_header *video_weight_header = NULL, *video_bias_header = NULL;
    const h3_st_header *audio_weight_header = NULL, *audio_bias_header = NULL;
    const h3_st_tensor *video_weight = find_head(
        store, "pdd_heads.video_out.weight", 3, video_weight_shape,
        &video_weight_header, error, error_size);
    if (!video_weight) return NULL;
    const h3_st_tensor *video_bias = find_head(
        store, "pdd_heads.video_out.bias", 2, video_bias_shape,
        &video_bias_header, error, error_size);
    if (!video_bias) return NULL;
    const h3_st_tensor *audio_weight = find_head(
        store, "pdd_heads.audio_out.weight", 3, audio_weight_shape,
        &audio_weight_header, error, error_size);
    if (!audio_weight) return NULL;
    const h3_st_tensor *audio_bias = find_head(
        store, "pdd_heads.audio_out.bias", 2, audio_bias_shape,
        &audio_bias_header, error, error_size);
    if (!audio_bias) return NULL;

    h3_pdd_heads *heads = calloc(1, sizeof(*heads));
    if (!heads) {
        fail(error, error_size, "out of memory creating PDD heads");
        return NULL;
    }
    heads->gpu = gpu;
    heads->trained = H3_PDD_HEADS_TRAINED;
    heads->video_weight = read_head(video_weight_header, video_weight,
                                    "pdd_heads.video_out.weight", error,
                                    error_size);
    heads->video_bias = read_head(video_bias_header, video_bias,
                                  "pdd_heads.video_out.bias", error,
                                  error_size);
    heads->audio_weight = read_head(audio_weight_header, audio_weight,
                                    "pdd_heads.audio_out.weight", error,
                                    error_size);
    heads->audio_bias = read_head(audio_bias_header, audio_bias,
                                  "pdd_heads.audio_out.bias", error,
                                  error_size);
    if (!heads->video_weight || !heads->video_bias || !heads->audio_weight ||
        !heads->audio_bias) {
        h3_pdd_heads_free(heads);
        return NULL;
    }
    if (trained_out) *trained_out = heads->trained;
    return heads;
}

void h3_pdd_heads_free(h3_pdd_heads *heads) {
    if (!heads) return;
    for (uint32_t step = 0; step < heads->steps; step++) {
        h3_gpu_tensor_free(heads->fused_video_weight[step]);
        h3_gpu_tensor_free(heads->fused_video_bias[step]);
        h3_gpu_tensor_free(heads->fused_audio_weight[step]);
        h3_gpu_tensor_free(heads->fused_audio_bias[step]);
    }
    free(heads->fused_video_weight);
    free(heads->fused_video_bias);
    free(heads->fused_audio_weight);
    free(heads->fused_audio_bias);
    free(heads->video_weight);
    free(heads->video_bias);
    free(heads->audio_weight);
    free(heads->audio_bias);
    free(heads);
}

uint32_t h3_pdd_heads_trained(const h3_pdd_heads *heads) {
    return heads ? heads->trained : 0;
}

uint32_t h3_pdd_heads_steps(const h3_pdd_heads *heads) {
    return heads ? heads->steps : 0;
}

/* One fused projection: mix `trained` host BF16 heads into a fresh device BF16
 * tensor. Accumulation is float32 with a single BF16 rounding at the end, which
 * is strictly better than accumulating in BF16 and keeps the step-to-step
 * difference driven by the plan rather than by rounding. */
static h3_gpu_tensor *fuse_matrix(h3_gpu *gpu, const uint16_t *heads,
                                  uint32_t trained, uint32_t rows,
                                  uint32_t columns, const float *plan,
                                  char *error, size_t error_size) {
    size_t row_elements = (size_t)rows * columns;
    float *sum = calloc(row_elements, sizeof(float));
    uint16_t *quantized = malloc(row_elements * sizeof(uint16_t));
    if (!sum || !quantized) {
        fail(error, error_size, "out of memory fusing a PDD head");
        free(sum);
        free(quantized);
        return NULL;
    }
    for (uint32_t index = 0; index < trained; index++) {
        float coefficient = plan[index];
        if (coefficient == 0.0f) continue;
        const uint16_t *source = heads + (size_t)index * row_elements;
        for (size_t element = 0; element < row_elements; element++)
            sum[element] += coefficient * f32_from_bf16(source[element]);
    }
    for (size_t element = 0; element < row_elements; element++)
        quantized[element] = bf16_from_f32(sum[element]);
    h3_gpu_tensor *result = h3_gpu_tensor_from_bf16(gpu, quantized,
                                                   row_elements);
    if (!result)
        fail(error, error_size, "cannot upload a fused PDD head: %s",
             h3_gpu_error(gpu));
    free(sum);
    free(quantized);
    return result;
}

static h3_gpu_tensor *fuse_vector(h3_gpu *gpu, const uint16_t *heads,
                                  uint32_t trained, uint32_t rows,
                                  const float *plan, char *error,
                                  size_t error_size) {
    float *sum = calloc(rows, sizeof(float));
    uint16_t *quantized = malloc((size_t)rows * sizeof(uint16_t));
    if (!sum || !quantized) {
        fail(error, error_size, "out of memory fusing a PDD bias");
        free(sum);
        free(quantized);
        return NULL;
    }
    for (uint32_t index = 0; index < trained; index++) {
        float coefficient = plan[index];
        if (coefficient == 0.0f) continue;
        const uint16_t *source = heads + (size_t)index * rows;
        for (uint32_t row = 0; row < rows; row++)
            sum[row] += coefficient * f32_from_bf16(source[row]);
    }
    for (uint32_t row = 0; row < rows; row++)
        quantized[row] = bf16_from_f32(sum[row]);
    h3_gpu_tensor *result = h3_gpu_tensor_from_bf16(gpu, quantized, rows);
    if (!result)
        fail(error, error_size, "cannot upload a fused PDD bias: %s",
             h3_gpu_error(gpu));
    free(sum);
    free(quantized);
    return result;
}

int h3_pdd_heads_fuse_steps(h3_pdd_heads *heads, const float *video_plans,
                            const float *audio_plans, uint32_t coarse_steps,
                            uint32_t fine_steps, char *error,
                            size_t error_size) {
    if (!heads || !video_plans || !audio_plans || !coarse_steps) {
        fail(error, error_size, "PDD fusion needs plans and a step count");
        return 0;
    }
    if (fine_steps != heads->trained) {
        fail(error, error_size,
             "plans cover %u fine steps but the heads have %u", fine_steps,
             heads->trained);
        return 0;
    }
    if (heads->steps) {
        fail(error, error_size, "PDD heads are already fused");
        return 0;
    }
    for (uint32_t step = 0; step < coarse_steps; step++) {
        double total = 0.0;
        for (uint32_t fine = 0; fine < fine_steps; fine++)
            total += video_plans[(size_t)step * fine_steps + fine];
        if (fabs(total - 1.0) > 1e-3) {
            fail(error, error_size,
                 "video plan row %u sums to %.4f, expected 1 (transposed plans?)",
                 step, total);
            return 0;
        }
        total = 0.0;
        for (uint32_t fine = 0; fine < fine_steps; fine++)
            total += audio_plans[(size_t)step * fine_steps + fine];
        if (fabs(total - 1.0) > 1e-3) {
            fail(error, error_size,
                 "audio plan row %u sums to %.4f, expected 1 (transposed plans?)",
                 step, total);
            return 0;
        }
    }
    heads->fused_video_weight = calloc(coarse_steps, sizeof(h3_gpu_tensor *));
    heads->fused_video_bias = calloc(coarse_steps, sizeof(h3_gpu_tensor *));
    heads->fused_audio_weight = calloc(coarse_steps, sizeof(h3_gpu_tensor *));
    heads->fused_audio_bias = calloc(coarse_steps, sizeof(h3_gpu_tensor *));
    if (!heads->fused_video_weight || !heads->fused_video_bias ||
        !heads->fused_audio_weight || !heads->fused_audio_bias) {
        fail(error, error_size, "out of memory fusing PDD heads");
        return 0;
    }
    for (uint32_t step = 0; step < coarse_steps; step++) {
        const float *video_plan = video_plans + (size_t)step * fine_steps;
        const float *audio_plan = audio_plans + (size_t)step * fine_steps;
        heads->fused_video_weight[step] = fuse_matrix(
            heads->gpu, heads->video_weight, heads->trained, H3_PDD_VIDEO_DIM,
            H3_PDD_HIDDEN, video_plan, error, error_size);
        heads->fused_video_bias[step] = fuse_vector(
            heads->gpu, heads->video_bias, heads->trained, H3_PDD_VIDEO_DIM,
            video_plan, error, error_size);
        heads->fused_audio_weight[step] = fuse_matrix(
            heads->gpu, heads->audio_weight, heads->trained, H3_PDD_AUDIO_DIM,
            H3_PDD_HIDDEN, audio_plan, error, error_size);
        heads->fused_audio_bias[step] = fuse_vector(
            heads->gpu, heads->audio_bias, heads->trained, H3_PDD_AUDIO_DIM,
            audio_plan, error, error_size);
        if (!heads->fused_video_weight[step] ||
            !heads->fused_video_bias[step] ||
            !heads->fused_audio_weight[step] || !heads->fused_audio_bias[step]) {
            heads->steps = step;
            return 0;
        }
    }
    heads->steps = coarse_steps;
    return 1;
}

const h3_gpu_tensor *h3_pdd_heads_video_step(const h3_pdd_heads *heads,
                                             uint32_t step) {
    return heads && step < heads->steps ? heads->fused_video_weight[step] : NULL;
}

const h3_gpu_tensor *h3_pdd_heads_audio_step(const h3_pdd_heads *heads,
                                             uint32_t step) {
    return heads && step < heads->steps ? heads->fused_audio_weight[step] : NULL;
}

const h3_gpu_tensor *h3_pdd_heads_video_bias(const h3_pdd_heads *heads,
                                             uint32_t step) {
    return heads && step < heads->steps ? heads->fused_video_bias[step] : NULL;
}

const h3_gpu_tensor *h3_pdd_heads_audio_bias(const h3_pdd_heads *heads,
                                             uint32_t step) {
    return heads && step < heads->steps ? heads->fused_audio_bias[step] : NULL;
}
