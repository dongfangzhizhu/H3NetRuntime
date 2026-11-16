#include "h3_dit_schedule.h"

#include "h3_lora.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    BLOCK_OUTPUT = H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN,
    FINAL_OUTPUT = 2 * H3_DIT_HIDDEN
};

/* The pruned release replaces the 256-wide sinusoidal time embedder with a
 * rank-8 curve: adaln_t_table[grid][rank] holds PCA components of the
 * modulation curve, and each block projects it with a [output, rank] matrix.
 * The table is already mean-centred and the mean is folded into the bias, so
 * the runtime path is lerp -> small GEMV -> no SiLU. */
enum { ADALN_CLASSIC = 0, ADALN_CURVE = 1 };

struct h3_dit_schedule {
    h3_gpu *gpu;
    int steps;
    int curve;
    uint32_t rank;
    uint32_t grid;
    uint32_t time_rows;
    uint32_t *video_rows;
    uint32_t *audio_rows;
    uint32_t *visual_condition_rows;
    uint32_t *audio_condition_rows;
    h3_gpu_tensor *blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static h3_gpu_tensor *weight_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t width,
                                    char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t rows,
                                    uint64_t columns, char *error,
                                    size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_1d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t width,
                                     char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_2d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t rows,
                                     uint64_t columns, char *error,
                                     size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

/* The block modulation tables depend only on the sigma schedule and the two
 * condition flags, never on the prompt, yet producing them reads every block's
 * adaln_proj weight — 24.2 GiB of the transformer's 61.7 GiB. The tables
 * themselves are around 400 MiB, so they are worth keeping on disk. */
/* Version 2 adds rank/grid to the key so the classic 2688-wide family and the
 * rank-8 curve family can never share a table.  Version 3 exists because
 * version 2 entries carry no adaptor provenance: a table computed with the PDD
 * LoRA folded in and one computed without it hash to the same key, so whichever
 * ran first was served to the other (§24). */
#define ADALN_CACHE_VERSION 3u

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t blocks;
    uint32_t time_rows;
    uint32_t block_output;
    uint32_t final_output;
    uint32_t reserved;
    uint64_t key;
} adaln_cache_header;

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes) {
    const unsigned char *cursor = data;
    for (size_t index = 0; index < bytes; index++) {
        hash ^= cursor[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

/* Identifies the checkpoint by the shard that holds the first block's
 * projection, so swapping weights misses the cache instead of reusing it. */
static uint64_t adaln_cache_key(const h3_weight_store *weights,
                                const float *features, const float *times,
                                const float *card, size_t card_bytes,
                                uint32_t time_rows, uint32_t rank,
                                uint32_t grid, int curve,
                                int visual_condition, int audio_condition) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor =
        h3_weight_find(weights, "blocks.0.adaln_proj.linear.weight", &header);
    if (!tensor || !header || !header->path) return 0;
    struct stat status;
    if (stat(header->path, &status) != 0) return 0;
    uint64_t hash = 14695981039346656037ull;
    uint32_t shape[] = {ADALN_CACHE_VERSION, H3_DIT_BLOCKS,       time_rows,
                        BLOCK_OUTPUT,        FINAL_OUTPUT,        TIME_INPUT,
                        TIME_HIDDEN,         (uint32_t)!!visual_condition,
                        (uint32_t)!!audio_condition, (uint32_t)curve,
                        rank,                grid};
    hash = hash_bytes(hash, shape, sizeof(shape));
    hash = hash_bytes(hash, header->path, strlen(header->path));
    uint64_t identity[] = {(uint64_t)status.st_size, (uint64_t)status.st_mtime};
    hash = hash_bytes(hash, identity, sizeof(identity));
    if (curve) {
        /* Same timesteps must give the same curve rows. The shard identity is
         * only path+size+mtime, so the table bytes themselves are hashed too:
         * two curve checkpoints dropped on the same path cannot collide. */
        if (times)
            hash = hash_bytes(hash, times,
                              (size_t)time_rows * sizeof(*times));
        if (card && card_bytes) hash = hash_bytes(hash, card, card_bytes);
    } else if (features) {
        hash = hash_bytes(hash, features,
                          (size_t)time_rows * TIME_INPUT * sizeof(*features));
    }
    return hash ? hash : 1;
}

static int adaln_cache_path(char *path, size_t size, uint64_t key) {
    const char *setting = getenv("H3_ADALN_CACHE");
    if (setting && (!strcmp(setting, "0") || !strcmp(setting, "off")))
        return 0;
    char root[768];
    if (setting && *setting) {
        snprintf(root, sizeof(root), "%s", setting);
    } else {
        const char *base = getenv("XDG_CACHE_HOME");
        if (base && *base) {
            snprintf(root, sizeof(root), "%s/h3-spark", base);
        } else {
            const char *home = getenv("HOME");
            if (!home || !*home) return 0;
            snprintf(root, sizeof(root), "%s/.cache/h3-spark", home);
        }
        char parent[768];
        snprintf(parent, sizeof(parent), "%s", root);
        char *slash = strrchr(parent, '/');
        if (slash) {
            *slash = '\0';
            mkdir(parent, 0777);
        }
    }
    if (mkdir(root, 0777) != 0 && errno != EEXIST) return 0;
    int written = snprintf(path, size, "%s/adaln-%016llx.bin", root,
                           (unsigned long long)key);
    return written > 0 && (size_t)written < size;
}

static int adaln_cache_load(h3_dit_schedule *schedule, h3_gpu *gpu,
                            const char *path, uint64_t key) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    adaln_cache_header header;
    int ok = fread(&header, sizeof(header), 1, file) == 1 &&
             !memcmp(header.magic, "H3ADALN", 8) &&
             header.version == ADALN_CACHE_VERSION && header.key == key &&
             header.blocks == H3_DIT_BLOCKS &&
             header.time_rows == schedule->time_rows &&
             header.block_output == BLOCK_OUTPUT &&
             header.final_output == FINAL_OUTPUT &&
             header.reserved ==
                 (uint32_t)(schedule->curve ? schedule->rank : 0);
    fclose(file);
    if (!ok) return 0;
    size_t block_elements = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    size_t final_elements = (size_t)schedule->time_rows * FINAL_OUTPUT;
    uint64_t offset = sizeof(header);
    char error[256];
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        schedule->blocks[block] = h3_gpu_tensor_new_bf16(gpu, block_elements);
        if (!schedule->blocks[block] ||
            !h3_gpu_tensor_read_file_bf16(schedule->blocks[block], path, offset,
                                          block_elements, error,
                                          sizeof(error)))
            return 0;
        offset += block_elements * sizeof(uint16_t);
    }
    schedule->final = h3_gpu_tensor_new_bf16(gpu, final_elements);
    return schedule->final &&
           h3_gpu_tensor_read_file_bf16(schedule->final, path, offset,
                                        final_elements, error, sizeof(error)) &&
           h3_gpu_submit(gpu);
}

/* Best effort: a run that cannot write the cache still has its tables. */
static void adaln_cache_store(const h3_dit_schedule *schedule,
                              const char *path, uint64_t key) {
    size_t block_elements = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    size_t final_elements = (size_t)schedule->time_rows * FINAL_OUTPUT;
    uint16_t *host = malloc(block_elements * sizeof(*host));
    if (!host) return;
    char temporary[1056];
    snprintf(temporary, sizeof(temporary), "%s.%d", path, (int)getpid());
    FILE *file = fopen(temporary, "wb");
    if (!file) {
        free(host);
        return;
    }
    adaln_cache_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "H3ADALN", 8);
    header.version = ADALN_CACHE_VERSION;
    header.blocks = H3_DIT_BLOCKS;
    header.time_rows = schedule->time_rows;
    header.block_output = BLOCK_OUTPUT;
    header.final_output = FINAL_OUTPUT;
    header.reserved = (uint32_t)(schedule->curve ? schedule->rank : 0);
    header.key = key;
    int ok = fwrite(&header, sizeof(header), 1, file) == 1;
    for (unsigned block = 0; ok && block < H3_DIT_BLOCKS; block++) {
        ok = h3_gpu_tensor_read_bf16(schedule->blocks[block], host,
                                     block_elements) &&
             fwrite(host, sizeof(*host), block_elements, file) ==
                 block_elements;
    }
    if (ok)
        ok = h3_gpu_tensor_read_bf16(schedule->final, host, final_elements) &&
             fwrite(host, sizeof(*host), final_elements, file) ==
                 final_elements;
    ok = fclose(file) == 0 && ok;
    free(host);
    if (ok && rename(temporary, path) == 0) return;
    unlink(temporary);
}

/* Emits the per-step row indices plus the per-row timestep. The 256-wide
 * sin/cos features are only built for the classic embedder; the curve family
 * consumes the timesteps directly. */
static int prepare_rows(h3_dit_schedule *schedule,
                        const h3_sigma_schedule *sigmas,
                        int visual_condition, int audio_condition, int curve,
                        float **times_out, float **features_out, char *error,
                        size_t error_size) {
    schedule->steps = sigmas->steps;
    schedule->video_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->video_rows));
    schedule->audio_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->audio_rows));
    if (visual_condition)
        schedule->visual_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->visual_condition_rows));
    if (audio_condition)
        schedule->audio_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->audio_condition_rows));
    if (!schedule->video_rows || !schedule->audio_rows ||
        (visual_condition && !schedule->visual_condition_rows) ||
        (audio_condition && !schedule->audio_condition_rows)) {
        fail(error, error_size, "out of memory allocating timestep row maps");
        return 0;
    }
    uint32_t count = 0;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (video == audio) {
            schedule->video_rows[step] = count;
            schedule->audio_rows[step] = count++;
        } else if (video < audio) {
            schedule->video_rows[step] = count++;
            schedule->audio_rows[step] = count++;
        } else {
            schedule->audio_rows[step] = count++;
            schedule->video_rows[step] = count++;
        }
    }
    uint32_t visual_condition_row = UINT32_MAX;
    uint32_t audio_condition_row = UINT32_MAX;
    if (visual_condition) visual_condition_row = count++;
    if (audio_condition) audio_condition_row = count++;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (visual_condition)
            schedule->visual_condition_rows[step] = video >= 0.999f ?
                schedule->video_rows[step] : visual_condition_row;
        if (audio_condition)
            schedule->audio_condition_rows[step] = audio >= 1.0f ?
                schedule->audio_rows[step] : audio_condition_row;
    }
    schedule->time_rows = count;
    if (!count || count > UINT32_MAX / TIME_INPUT) {
        fail(error, error_size, "invalid number of timestep rows");
        return 0;
    }
    float *times = calloc(count, sizeof(*times));
    float *features = curve ? NULL :
        malloc((size_t)count * TIME_INPUT * sizeof(*features));
    if (!times || (!curve && !features)) {
        free(times);
        free(features);
        fail(error, error_size, "out of memory allocating timestep features");
        return 0;
    }
    for (int step = 0; step < sigmas->steps; step++) {
        times[schedule->video_rows[step]] = 1.0f - sigmas->video[step];
        times[schedule->audio_rows[step]] = 1.0f - sigmas->audio[step];
    }
    if (visual_condition) times[visual_condition_row] = 0.999f;
    if (audio_condition) times[audio_condition_row] = 1.0f;
    for (uint32_t row = 0; row < count && !curve; row++) {
        for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }
    *times_out = times;
    *features_out = features;
    return 1;
}

/* Detects the rank-8 curve family from the checkpoint itself: the block
 * projection's input width is the discriminator, because a curve checkpoint
 * ships no time_embedder.* at all. */
static int adaln_curve_probe(const h3_weight_store *weights, int *curve,
                             uint32_t *rank, uint32_t *grid, char *error,
                             size_t error_size) {
    const h3_st_tensor *block = h3_weight_find(
        weights, "blocks.0.adaln_proj.linear.weight", NULL);
    if (!block || block->ndim != 2 || block->shape[0] != BLOCK_OUTPUT ||
        !block->shape[1]) {
        fail(error, error_size,
             "blocks.0.adaln_proj.linear.weight is absent or not [%u, rank]",
             BLOCK_OUTPUT);
        return 0;
    }
    *curve = ADALN_CLASSIC;
    *rank = (uint32_t)block->shape[1];
    *grid = 0;
    if (*rank == H3_DIT_TIME_DIM) return 1;
    const h3_st_tensor *table = h3_weight_find(weights, "adaln_t_table", NULL);
    if (!table) {
        fail(error, error_size,
             "AdaLN projection wants %u inputs but the checkpoint has no "
             "adaln_t_table; it needs either time_embedder.* or the curve",
             *rank);
        return 0;
    }
    if (table->dtype != H3_DTYPE_F32 || table->ndim != 2 ||
        table->shape[1] != *rank || table->shape[0] < 2) {
        fail(error, error_size, "adaln_t_table must be F32 [grid >= 2, %u]",
             *rank);
        return 0;
    }
    *curve = ADALN_CURVE;
    *grid = (uint32_t)table->shape[0];
    return 1;
}

/* Round-to-nearest-even float32 -> bfloat16, matching the device cast. */
static uint16_t bf16_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0xfffu + lsb;
    return (uint16_t)(bits >> 16);
}

/* The whole table is 33 KiB at rank 8, so precompute reads it once and both
 * hashes it into the cache key and lerps from the same copy. */
static float *curve_table_read(const h3_weight_store *weights, uint32_t grid,
                               uint32_t rank, char *error,
                               size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *table =
        h3_weight_find(weights, "adaln_t_table", &header);
    if (!table || !header) {
        fail(error, error_size, "adaln_t_table disappeared");
        return NULL;
    }
    size_t elements = (size_t)grid * rank;
    float *values = malloc(elements * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory reading adaln_t_table");
        return NULL;
    }
    if (!h3_st_read_data(header, table, values, elements * sizeof(*values),
                         error, error_size)) {
        free(values);
        return NULL;
    }
    return values;
}

/* t -> curve position -> lerp. The table covers t in [0, 1] and its last row is
 * the t == 1 endpoint, so the lower index stops one row short. */
static float *curve_timesteps(const float *values, const float *times,
                              uint32_t rows, uint32_t rank, uint32_t grid,
                              char *error, size_t error_size) {
    float *temb = malloc((size_t)rows * rank * sizeof(*temb));
    if (!temb) {
        fail(error, error_size, "out of memory building curve timesteps");
        return NULL;
    }
    for (uint32_t row = 0; row < rows; row++) {
        double time = times[row];
        if (!(time > 0.0)) time = 0.0;
        if (time > 1.0) time = 1.0;
        double position = time * (double)(grid - 1);
        uint32_t lower = (uint32_t)floor(position);
        if (lower > grid - 2) lower = grid - 2;
        double fraction = position - (double)lower;
        for (uint32_t index = 0; index < rank; index++) {
            double low = values[(size_t)lower * rank + index];
            double high = values[(size_t)(lower + 1) * rank + index];
            temb[(size_t)row * rank + index] =
                (float)(low + fraction * (high - low));
        }
    }
    return temb;
}

/* mod[row][col] = temb[row] . W[col] + bias[col], rounded straight to BF16.
 * 17 x 96768 x 8 is ~13 M multiply-adds per block, so the host does it without
 * a launch, and the ~3 MiB projection is never resident. */
static h3_gpu_tensor *curve_modulation(const h3_weight_store *weights,
                                       h3_gpu *gpu, const char *weight_name,
                                       const char *bias_name,
                                       const float *temb, uint32_t rows,
                                       uint32_t rank, uint32_t output,
                                       h3_lora *lora, unsigned block,
                                       char *error, size_t error_size) {
    const h3_st_header *weight_header = NULL, *bias_header = NULL;
    const h3_st_tensor *weight =
        h3_weight_find(weights, weight_name, &weight_header);
    const h3_st_tensor *bias = h3_weight_find(weights, bias_name, &bias_header);
    if (!weight || !weight_header || weight->dtype != H3_DTYPE_F32 ||
        weight->ndim != 2 || weight->shape[0] != output ||
        weight->shape[1] != rank) {
        fail(error, error_size, "%s must be F32 [%u, %u]", weight_name, output,
             rank);
        return NULL;
    }
    if (!bias || !bias_header || bias->dtype != H3_DTYPE_F32 ||
        bias->ndim != 1 || bias->shape[0] != output) {
        fail(error, error_size, "%s must be F32 [%u]", bias_name, output);
        return NULL;
    }
    size_t weight_elements = (size_t)output * rank;
    float *matrix = malloc(weight_elements * sizeof(*matrix));
    float *offset = malloc((size_t)output * sizeof(*offset));
    uint16_t *table = malloc((size_t)rows * output * sizeof(*table));
    h3_gpu_tensor *result = NULL;
    if (!matrix || !offset || !table) {
        fail(error, error_size, "out of memory building %s", weight_name);
        goto done;
    }
    if (!h3_st_read_data(weight_header, weight, matrix,
                         weight_elements * sizeof(*matrix), error,
                         error_size) ||
        !h3_st_read_data(bias_header, bias, offset,
                         (size_t)output * sizeof(*offset), error, error_size))
        goto done;
    /* The PDD adaptor also shifts the modulation; fold it into the rank-8
     * projection before the table is materialized. */
    if (lora && h3_lora_adaln_delta(lora, block, matrix, offset, output, rank,
                                    error, error_size) < 0)
        goto done;
    for (uint32_t row = 0; row < rows; row++) {
        const float *source = temb + (size_t)row * rank;
        uint16_t *destination = table + (size_t)row * output;
        for (uint32_t column = 0; column < output; column++) {
            const float *line = matrix + (size_t)column * rank;
            float sum = offset[column];
            for (uint32_t index = 0; index < rank; index++)
                sum += source[index] * line[index];
            destination[column] = bf16_from_f32(sum);
        }
    }
    result = h3_gpu_tensor_from_bf16(gpu, table, (size_t)rows * output);
    if (!result && (!error || !*error))
        fail(error, error_size, "cannot upload %s modulation: %s", weight_name,
             h3_gpu_error(gpu));
done:
    free(matrix);
    free(offset);
    free(table);
    return result;
}

static h3_gpu_tensor *time_embeddings(const h3_weight_store *weights,
                                      h3_gpu *gpu, uint32_t rows,
                                      const float *features, char *error,
                                      size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, features, (size_t)rows * TIME_INPUT);
    h3_gpu_tensor *in_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT,
        error, error_size);
    h3_gpu_tensor *in_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_in.bias", TIME_HIDDEN, error, error_size);
    h3_gpu_tensor *out_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_out.weight", H3_DIT_TIME_DIM, TIME_HIDDEN,
        error, error_size);
    h3_gpu_tensor *out_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_out.bias", H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *silu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *result = NULL;
    h3_gpu_tensor *all[] = {input, in_w, in_b, out_w, out_b, hidden,
                            activated, output, bf16, silu};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate timestep tensors: %s",
                     h3_gpu_error(gpu));
            goto cleanup;
        }
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep embedding") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                       TIME_INPUT, TIME_HIDDEN),
                error, error_size, "timestep input projection") ||
        !gpu_op(gpu, h3_gpu_silu_f32(gpu, activated, hidden,
                                     rows * TIME_HIDDEN),
                error, error_size, "timestep SiLU") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, output, activated, out_w, out_b,
                                       rows, TIME_HIDDEN, H3_DIT_TIME_DIM),
                error, error_size, "timestep output projection") ||
        !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(
                        gpu, bf16, output, rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep BF16 cast") ||
        !gpu_op(gpu, h3_gpu_silu_bf16(gpu, silu, bf16,
                                      rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep AdaLN SiLU") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep embedding")) {
        goto cleanup;
    }
    result = silu;
    silu = NULL;
cleanup:
    free_tensor(&input);
    free_tensor(&in_w);
    free_tensor(&in_b);
    free_tensor(&out_w);
    free_tensor(&out_b);
    free_tensor(&hidden);
    free_tensor(&activated);
    free_tensor(&output);
    free_tensor(&bf16);
    free_tensor(&silu);
    return result;
}

h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition, h3_lora *lora,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !sigmas || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid AdaLN schedule arguments");
        return NULL;
    }
    h3_dit_schedule *schedule = calloc(1, sizeof(*schedule));
    if (!schedule) {
        fail(error, error_size, "out of memory creating AdaLN schedule");
        return NULL;
    }
    schedule->gpu = gpu;
    float *features = NULL, *times = NULL, *temb = NULL, *card = NULL;
    if (!adaln_curve_probe(weights, &schedule->curve, &schedule->rank,
                           &schedule->grid, error, error_size)) goto failed;
    if (!prepare_rows(schedule, sigmas, visual_condition, audio_condition,
                      schedule->curve, &times, &features, error, error_size))
        goto failed;
    if (schedule->curve) {
        card = curve_table_read(weights, schedule->grid, schedule->rank, error,
                                error_size);
        if (!card) goto failed;
    }
    uint64_t cache_key = adaln_cache_key(
        weights, features, times, card,
        card ? (size_t)schedule->grid * schedule->rank * sizeof(*card) : 0,
        schedule->time_rows, schedule->rank, schedule->grid, schedule->curve,
        visual_condition, audio_condition);
    char cache_file[1024];
    /* An adaptor rewrites every block's modulation (h3_lora_adaln_delta), and
     * the tables are keyed on the checkpoint and the timesteps alone.  Keying
     * on the adaptor as well would work, but then the same key would have to
     * mean two different things depending on who wrote it first, and a stale
     * entry is invisible: the video still renders.  So a LoRA load neither
     * reads nor writes the cache -- the 50 folds it recomputes cost a few
     * seconds against a load measured in minutes. */
    int cacheable = !lora && cache_key &&
        adaln_cache_path(cache_file, sizeof(cache_file), cache_key);
    if (lora && cache_key && !getenv("H3_QUIET"))
        fprintf(stderr,
                "h3: AdaLN cache not used: the PDD adaptor changes every "
                "block's modulation\n");
    if (cacheable && adaln_cache_load(schedule, gpu, cache_file, cache_key)) {
        free(features);
        free(times);
        free(card);
        if (progress)
            progress((int)H3_DIT_BLOCKS, (int)H3_DIT_BLOCKS, progress_opaque);
        return schedule;
    }
    if (cacheable) {
        /* A partial load leaves tensors behind that the recompute would leak. */
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
            free_tensor(&schedule->blocks[block]);
        free_tensor(&schedule->final);
    }
    h3_lora_reset_adaln(lora);
    h3_gpu_tensor *time = NULL;
    if (schedule->curve) {
        temb = curve_timesteps(card, times, schedule->time_rows,
                               schedule->rank, schedule->grid, error,
                               error_size);
        if (!temb) goto failed;
        free(card);
        card = NULL;
    } else {
        time = time_embeddings(weights, gpu, schedule->time_rows, features,
                                error, error_size);
        if (!time) goto failed;
    }
    free(features);
    features = NULL;
    if (schedule->curve && getenv("H3_PROFILE"))
        fprintf(stderr,
                "h3: AdaLN curve: rank %u, grid %u, %u timestep rows, "
                "%.1f MiB of F32 projections\n",
                schedule->rank, schedule->grid, schedule->time_rows,
                (double)H3_DIT_BLOCKS * BLOCK_OUTPUT *
                    (double)(schedule->rank + 1) * 4.0 /
                    (1024.0 * 1024.0));

    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        char weight_name[128], bias_name[128], operation[128];
        snprintf(weight_name, sizeof(weight_name),
                 "blocks.%u.adaln_proj.linear.weight", block);
        snprintf(bias_name, sizeof(bias_name),
                 "blocks.%u.adaln_proj.linear.bias", block);
        if (schedule->curve) {
            schedule->blocks[block] = curve_modulation(
                weights, gpu, weight_name, bias_name, temb,
                schedule->time_rows, schedule->rank, BLOCK_OUTPUT, lora, block,
                error, error_size);
            if (!schedule->blocks[block]) goto failed;
            if (progress)
                progress((int)block + 1, (int)H3_DIT_BLOCKS, progress_opaque);
            continue;
        }
        h3_gpu_tensor *weight = weight_bf16_2d(
            weights, gpu, weight_name, BLOCK_OUTPUT, H3_DIT_TIME_DIM,
            error, error_size);
        h3_gpu_tensor *bias = weight_bf16_1d(
            weights, gpu, bias_name, BLOCK_OUTPUT, error, error_size);
        schedule->blocks[block] = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);
        if (!weight || !bias || !schedule->blocks[block]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate AdaLN block %u: %s",
                     block, h3_gpu_error(gpu));
            free_tensor(&weight);
            free_tensor(&bias);
            h3_gpu_tensor_free(time);
            goto failed;
        }
        snprintf(operation, sizeof(operation), "AdaLN block %u", block);
        int ok = gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_linear_bf16(
                gpu, schedule->blocks[block], time, weight, bias,
                schedule->time_rows, H3_DIT_TIME_DIM, BLOCK_OUTPUT),
                error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation);
        free_tensor(&weight);
        free_tensor(&bias);
        if (!ok) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
        if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                               progress_opaque);
    }

    if (schedule->curve) {
        schedule->final = curve_modulation(
            weights, gpu, "final_layer.adaln_proj.linear.weight",
            "final_layer.adaln_proj.linear.bias", temb, schedule->time_rows,
            schedule->rank, FINAL_OUTPUT, NULL, 0, error, error_size);
        if (!schedule->final) goto failed;
        free(times);
        times = NULL;
        free(temb);
        temb = NULL;
        if (cacheable) adaln_cache_store(schedule, cache_file, cache_key);
        return schedule;
    }
    h3_gpu_tensor *final_w = weight_bf16_2d(
        weights, gpu, "final_layer.adaln_proj.linear.weight",
        FINAL_OUTPUT, H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *final_b = weight_bf16_1d(
        weights, gpu, "final_layer.adaln_proj.linear.bias",
        FINAL_OUTPUT, error, error_size);
    schedule->final = h3_gpu_tensor_new_bf16(
        gpu, (size_t)schedule->time_rows * FINAL_OUTPUT);
    if (!final_w || !final_b || !schedule->final ||
        !gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin final AdaLN") ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, schedule->final, time, final_w, final_b, schedule->time_rows,
            H3_DIT_TIME_DIM, FINAL_OUTPUT), error, error_size,
            "final AdaLN projection") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit final AdaLN")) {
        if ((!error || !*error) && (!final_w || !final_b || !schedule->final))
            fail(error, error_size, "cannot allocate final AdaLN tensors: %s",
                 h3_gpu_error(gpu));
        free_tensor(&final_w);
        free_tensor(&final_b);
        h3_gpu_tensor_free(time);
        goto failed;
    }
    free_tensor(&final_w);
    free_tensor(&final_b);
    h3_gpu_tensor_free(time);
    free(times);
    free(temb);
    if (cacheable) adaln_cache_store(schedule, cache_file, cache_key);
    return schedule;

failed:
    free(features);
    free(times);
    free(temb);
    free(card);
    h3_dit_schedule_free(schedule);
    return NULL;
}

void h3_dit_schedule_free(h3_dit_schedule *schedule) {
    if (!schedule) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_gpu_tensor_free(schedule->blocks[block]);
    h3_gpu_tensor_free(schedule->final);
    free(schedule->video_rows);
    free(schedule->audio_rows);
    free(schedule->visual_condition_rows);
    free(schedule->audio_condition_rows);
    free(schedule);
}

int h3_dit_schedule_steps(const h3_dit_schedule *schedule) {
    return schedule ? schedule->steps : 0;
}

uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule) {
    return schedule ? schedule->time_rows : 0;
}

uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->video_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->audio_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->visual_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->visual_condition_rows[step] :
        UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->audio_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->audio_condition_rows[step] :
        UINT32_MAX;
}

const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block) {
    return schedule && block < H3_DIT_BLOCKS ? schedule->blocks[block] : NULL;
}

double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block) {
    if (!schedule || block >= H3_DIT_BLOCKS || !schedule->blocks[block])
        return -1.0;
    size_t count = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                             count)) {
        free(values);
        return -1.0;
    }
    double total = 0.0;
    size_t samples = 0;
    for (uint32_t row = 0; row < schedule->time_rows; row++)
        for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
            for (uint32_t slot = 2; slot <= 5; slot += 3) {
                size_t base = ((size_t)row * H3_DIT_MODALITIES *
                               H3_DIT_ADALN_SLOTS +
                               (size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                              H3_DIT_HIDDEN;
                for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++) {
                    uint32_t bits = (uint32_t)values[base + column] << 16;
                    float value;
                    memcpy(&value, &bits, sizeof(value));
                    total += fabs((double)value);
                }
                samples += H3_DIT_HIDDEN;
            }
    free(values);
    return samples ? total / (double)samples : -1.0;
}

void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count) {
    if (!schedule || !active_blocks || count != H3_DIT_BLOCKS) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (active_blocks[block]) continue;
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
}

const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule) {
    return schedule ? schedule->final : NULL;
}

int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count) {
    if (!schedule || step < 0 || step >= schedule->steps || !layout || !rows ||
        row_count != layout->seq_len || !layout->segments ||
        (text_tags && text_tag_count != (size_t)layout->signature[0])) return 0;
    size_t text_index = 0;
    for (size_t seg_index = 0; seg_index < layout->segment_count; seg_index++) {
        const h3_segment *segment = &layout->segments[seg_index];
        if (segment->start > segment->stop || segment->stop > row_count)
            return 0;
        uint32_t time_row;
        uint32_t tag;
        switch (segment->kind) {
        case H3_SEG_TEXT:
            time_row = schedule->video_rows[step];
            for (size_t row = segment->start; row < segment->stop; row++) {
                uint32_t text_tag = text_tags ? text_tags[text_index] : 1u;
                if (text_tag >= H3_DIT_MODALITIES) return 0;
                rows[row] = time_row * H3_DIT_MODALITIES + text_tag;
                text_index++;
            }
            continue;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE:
            if (!schedule->visual_condition_rows) return 0;
            time_row = schedule->visual_condition_rows[step];
            tag = 0;
            break;
        case H3_SEG_REF_AUDIO:
            if (!schedule->audio_condition_rows) return 0;
            time_row = schedule->audio_condition_rows[step];
            tag = 2;
            break;
        case H3_SEG_AUDIO:
            time_row = schedule->audio_rows[step];
            tag = 2;
            break;
        case H3_SEG_VIDEO:
            time_row = schedule->video_rows[step];
            tag = 0;
            break;
        default:
            return 0;
        }
        uint32_t modulation = time_row * H3_DIT_MODALITIES + tag;
        for (size_t row = segment->start; row < segment->stop; row++)
            rows[row] = modulation;
    }
    return text_index == (size_t)layout->signature[0];
}
