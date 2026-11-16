/* PDD 8-step LoRA merge for the pruned FL2VA DiT.
 *
 * The adaptor file holds 362 rank-64 modules with no alpha tensors, so the
 * scale is alpha/rank = 1. The merge happens once at load time: dequantize the
 * INT8 rows, add up @ down, requantize each row against a fresh scale. Nothing
 * survives into the denoise loop.
 *
 * Two placement rules matter and both are covered by tools/ref_lora_merge.py:
 *   - ff.net.0.proj's lora_up rows are reordered (chunk(2)[::-1]) because the
 *     checkpoint stores fc1 as [gate; up].
 *   - qkv rows live in per-head triples in h3-spark but the adaptors use the
 *     file's contiguous [q][k][v] thirds, so each channel is merged into the
 *     strided head blocks its rows belong to.
 */
#include "h3_lora.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HIDDEN = 5376,
    INNER = 7168,
    FFN = 14336,
    HEADS = 56,
    HEAD_DIM = 128,
    RANK = 64,
    MERGE_CHUNK = 2048
};

struct h3_lora {
    h3_st_header header;
    char *path;
    double seconds;
    /* Adaptors merged into the INT8 codes or the refiner weights. */
    unsigned modules;
    /* AdaLN adaptors folded into the current modulation schedule.  Kept apart
     * from `modules` because the schedule can be rebuilt (a rebind with a new
     * layout) and folding is then repeated for the same 50 adaptors: an
     * accumulator would report 412/362 the second time round. */
    unsigned adaln;
    /* Adaptors the file holds, counted at open: every module carries one
     * `*.lora_down.weight`.  `modules` alone cannot tell "merged everything"
     * from "merged the subset this build knows about", and a half-merged
     * adaptor still produces a plausible video. */
    unsigned indexed;
    /* adaln_affine_map, loaded on first use; affine_state is 0 (untried),
     * 1 (loaded) or -1 (failed, do not retry). */
    float *affine;
    int affine_state;
};

enum { ADALN_CURVE = 2688, ADALN_MAP_ROWS = 9 };

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static float bf16_value(uint16_t bits) {
    uint32_t raw = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static double now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

h3_lora *h3_lora_open(const char *path, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path) {
        fail(error, error_size, "LoRA path is required");
        return NULL;
    }
    h3_lora *lora = calloc(1, sizeof(*lora));
    if (!lora) {
        fail(error, error_size, "out of memory opening the LoRA");
        return NULL;
    }
    if (!h3_st_read_header(path, &lora->header, error, error_size)) {
        free(lora);
        return NULL;
    }
    lora->path = malloc(strlen(path) + 1);
    if (!lora->path) {
        h3_st_free_header(&lora->header);
        free(lora);
        fail(error, error_size, "out of memory opening the LoRA");
        return NULL;
    }
    memcpy(lora->path, path, strlen(path) + 1);
    /* Count the adaptors the file actually holds, so "merged N/M" has a
     * denominator that comes from the file rather than from a constant in this
     * source file.  A build that knows about fewer modules than the adaptor has
     * (a new projection, a refiner that gained one) then reports the gap
     * instead of quietly merging a subset. */
    static const char suffix[] = ".lora_down.weight";
    const size_t suffix_length = sizeof(suffix) - 1;
    for (size_t index = 0; index < lora->header.tensor_count; index++) {
        const char *name = lora->header.tensors[index].name;
        size_t length = name ? strlen(name) : 0;
        if (length > suffix_length &&
            memcmp(name + length - suffix_length, suffix, suffix_length) == 0)
            lora->indexed++;
    }
    return lora;
}

void h3_lora_free(h3_lora *lora) {
    if (!lora) return;
    h3_st_free_header(&lora->header);
    free(lora->affine);
    free(lora->path);
    free(lora);
}

/* Where the AdaLN affine map can be, in the order it should be looked for.
 *
 * The map ships with the Wan2GP adaptor *package* rather than with the adaptor
 * file, so its location depends on how the adaptor was installed. Two layouts
 * are in use:
 *
 *   - beside the adaptor: <adaptor dir>/lora_affine_maps/fl2va_rank8.sft.
 *     A model pack carries the pair, which is what lets a pack move to another
 *     drive or machine without an environment variable;
 *   - the Wan2GP tree the adaptor was published in, where the map sits at
 *     <root>/models/minimax_h3/lora_affine_maps/ and the adaptor at
 *     <root>/loras/minimax_h3/ — hence two levels up from the adaptor.
 *
 * Pure string work so tests/test_h3.c can pin the order without a fixture. */
int h3_lora_affine_candidates(const char *lora_path,
                              char paths[][H3_LORA_PATH_MAX], int max) {
    if (!lora_path || !paths || max <= 0) return 0;
    /* Both separators, because H3_LORA on Windows is often a path pasted from
     * Explorer. A POSIX file name may legally contain a backslash, but then the
     * candidate simply does not load and the next one is tried. */
    const char *slash = strrchr(lora_path, '/');
    const char *backslash = strrchr(lora_path, '\\');
    if (backslash && (!slash || backslash > slash)) slash = backslash;
    if (!slash) return 0;
    int directory = (int)(slash - lora_path);
    int count = 0;
    if (count < max) {
        snprintf(paths[count], H3_LORA_PATH_MAX,
                 "%.*s/lora_affine_maps/fl2va_rank8.sft", directory, lora_path);
        count++;
    }
    if (count < max) {
        snprintf(paths[count], H3_LORA_PATH_MAX,
                 "%.*s/../../models/minimax_h3/lora_affine_maps/"
                 "fl2va_rank8.sft", directory, lora_path);
        count++;
    }
    return count;
}

static int load_affine(h3_lora *lora, char *error, size_t error_size) {
    if (lora->affine_state > 0) return 1;
    if (lora->affine_state < 0) {
        fail(error, error_size,
             "the AdaLN affine map could not be loaded earlier; set "
             "H3_LORA_AFFINE to fl2va_rank8.sft");
        return 0;
    }
    lora->affine_state = -1;

    /* H3_LORA_AFFINE is authoritative: an explicit path that does not load is
     * reported instead of silently falling through to a discovered one. */
    char candidates[2][H3_LORA_PATH_MAX];
    int candidate_count = 0;
    const char *env = getenv("H3_LORA_AFFINE");
    if (env && *env) {
        snprintf(candidates[0], sizeof(candidates[0]), "%s", env);
        candidate_count = 1;
    } else {
        candidate_count = h3_lora_affine_candidates(lora->path, candidates, 2);
        if (candidate_count == 0) {
            fail(error, error_size,
                 "cannot resolve the AdaLN affine map next to %s; set "
                 "H3_LORA_AFFINE", lora->path);
            return 0;
        }
    }

    h3_st_header header;
    int found = -1;
    for (int index = 0; index < candidate_count; index++) {
        if (h3_st_read_header(candidates[index], &header, error, error_size)) {
            found = index;
            break;
        }
    }
    if (found < 0) {
        /* The per-candidate reason is only the last errno; the list of places
         * actually tried is what makes this diagnosable. */
        char tried[2048];
        size_t used = 0;
        tried[0] = '\0';
        for (int index = 0; index < candidate_count && used < sizeof(tried);
             index++) {
            int wrote = snprintf(tried + used, sizeof(tried) - used, "%s%s",
                                 index ? ", " : "", candidates[index]);
            if (wrote < 0 || (size_t)wrote >= sizeof(tried) - used) break;
            used += (size_t)wrote;
        }
        fail(error, error_size,
             "the AdaLN affine map is missing (tried %s); set H3_LORA_AFFINE "
             "to fl2va_rank8.sft", tried);
        return 0;
    }
    const char *path = candidates[found];
    const h3_st_tensor *map = h3_st_find(&header, "adaln_affine_map");
    if (!map || map->dtype != H3_DTYPE_F32 || map->ndim != 2 ||
        map->shape[0] != ADALN_MAP_ROWS || map->shape[1] != ADALN_CURVE) {
        h3_st_free_header(&header);
        fail(error, error_size,
             "%s: adaln_affine_map must be F32 [%d, %d]", path,
             (int)ADALN_MAP_ROWS, (int)ADALN_CURVE);
        return 0;
    }
    size_t elements = (size_t)ADALN_MAP_ROWS * ADALN_CURVE;
    float *affine = malloc(elements * sizeof(*affine));
    if (!affine) {
        h3_st_free_header(&header);
        fail(error, error_size, "out of memory reading the AdaLN affine map");
        return 0;
    }
    if (!h3_st_read_data(&header, map, affine, elements * sizeof(*affine),
                         error, error_size)) {
        h3_st_free_header(&header);
        free(affine);
        return 0;
    }
    h3_st_free_header(&header);
    lora->affine = affine;
    lora->affine_state = 1;
    return 1;
}

int h3_lora_adaln_delta(h3_lora *lora, unsigned block, float *matrix,
                        float *offset, uint32_t output, uint32_t rank,
                        char *error, size_t error_size) {
    if (!lora || !matrix || !offset) return 0;
    if (rank != 8) {
        fail(error, error_size,
             "the AdaLN adaptor fold expects a rank-8 curve, got %u", rank);
        return -1;
    }
    char name[192];
    snprintf(name, sizeof(name),
             "transformer_blocks.%u.adaln_proj.linear.lora_down.weight",
             block);
    const h3_st_tensor *down = h3_st_find(&lora->header, name);
    if (!down) {
        if (getenv("H3_LORA_TRACE"))
            fprintf(stderr, "h3 lora: no tensor %s (header has %zu)\n", name,
                    lora->header.tensor_count);
        return 0;
    }
    if (down->dtype != H3_DTYPE_BF16 || down->ndim != 2 ||
        down->shape[0] != RANK || down->shape[1] != ADALN_CURVE) {
        fail(error, error_size, "%s must be BF16 [%d, %d]", name, (int)RANK,
             (int)ADALN_CURVE);
        return -1;
    }
    snprintf(name, sizeof(name),
             "transformer_blocks.%u.adaln_proj.linear.lora_up.weight", block);
    const h3_st_tensor *up = h3_st_find(&lora->header, name);
    if (!up || up->dtype != H3_DTYPE_BF16 || up->ndim != 2 ||
        up->shape[0] != output || up->shape[1] != RANK) {
        fail(error, error_size, "%s must be BF16 [%u, %d]", name, output,
             (int)RANK);
        return -1;
    }
    if (!load_affine(lora, error, error_size)) return -1;

    uint16_t *a = malloc((size_t)RANK * ADALN_CURVE * sizeof(*a));
    uint16_t *b = malloc((size_t)output * RANK * sizeof(*b));
    float mapped[64][ADALN_MAP_ROWS];
    if (!a || !b) {
        free(a);
        free(b);
        fail(error, error_size, "out of memory folding the AdaLN adaptor");
        return -1;
    }
    int ok = h3_st_read_data(&lora->header, down, a,
                             (size_t)RANK * ADALN_CURVE * sizeof(*a), error,
                             error_size) &&
             h3_st_read_data(&lora->header, up, b,
                             (size_t)output * RANK * sizeof(*b), error,
                             error_size);
    if (ok) {
        for (uint32_t r = 0; r < RANK; r++) {
            const uint16_t *row = a + (size_t)r * ADALN_CURVE;
            for (uint32_t j = 0; j < ADALN_MAP_ROWS; j++) {
                const float *line = lora->affine + (size_t)j * ADALN_CURVE;
                double sum = 0.0;
                for (uint32_t k = 0; k < ADALN_CURVE; k++)
                    sum += (double)bf16_value(row[k]) * (double)line[k];
                mapped[r][j] = (float)sum;
            }
        }
        /* H3_LORA_SKIP suppresses the delta in merge_chunks() but used to
         * leave these folds applying it, so the control produced a model with
         * 312 of the 362 adaptors removed and 50 still in place -- neither the
         * base model nor the adapted one, and nothing said so (§25).  The fold
         * still runs and still counts, exactly like the block path: what the
         * switch removes is the add. */
        const int skip = getenv("H3_LORA_SKIP") != NULL;
        for (uint32_t o = 0; o < output; o++) {
            const uint16_t *row = b + (size_t)o * RANK;
            float delta[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float delta_bias = 0.0f;
            for (uint32_t r = 0; r < RANK; r++) {
                float weight = bf16_value(row[r]);
                if (weight == 0.0f) continue;
                for (uint32_t j = 0; j < 8; j++) delta[j] += weight * mapped[r][j];
                delta_bias += weight * mapped[r][8];
            }
            if (skip) continue;
            for (uint32_t j = 0; j < 8; j++) matrix[(size_t)o * rank + j] += delta[j];
            offset[o] += delta_bias;
        }
        lora->adaln++;
    }
    free(a);
    free(b);
    return ok ? 1 : -1;
}

double h3_lora_seconds(const h3_lora *lora) {
    return lora ? lora->seconds : 0.0;
}

unsigned h3_lora_modules_merged(const h3_lora *lora) {
    return lora ? lora->modules + lora->adaln : 0;
}

unsigned h3_lora_modules_indexed(const h3_lora *lora) {
    return lora ? lora->indexed : 0;
}

void h3_lora_reset_adaln(h3_lora *lora) {
    if (lora) lora->adaln = 0;
}

static const h3_st_tensor *find_bf16(const h3_lora *lora, const char *name,
                                     uint64_t rows, uint64_t columns,
                                     char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(&lora->header, name);
    if (!tensor) {
        fail(error, error_size, "LoRA tensor is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2) {
        fail(error, error_size, "LoRA tensor %s is %s/%d, expected BF16/2", name,
             h3_dtype_name(tensor->dtype), tensor->ndim);
        return NULL;
    }
    if (tensor->shape[0] != rows || tensor->shape[1] != columns) {
        fail(error, error_size,
             "LoRA tensor %s has shape [%llu, %llu], expected [%llu, %llu]",
             name, (unsigned long long)tensor->shape[0],
             (unsigned long long)tensor->shape[1], (unsigned long long)rows,
             (unsigned long long)columns);
        return NULL;
    }
    return tensor;
}

static uint16_t *read_matrix(const h3_lora *lora, const char *name,
                             uint64_t rows, uint64_t columns, char *error,
                             size_t error_size) {
    const h3_st_tensor *tensor =
        find_bf16(lora, name, rows, columns, error, error_size);
    if (!tensor) return NULL;
    size_t elements = (size_t)(rows * columns);
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values) {
        fail(error, error_size, "out of memory reading %s", name);
        return NULL;
    }
    if (!h3_st_read_data(&lora->header, tensor, values,
                         elements * sizeof(*values), error, error_size)) {
        free(values);
        return NULL;
    }
    return values;
}

/* [rows, columns] -> [columns, rows], so the adaptor can feed h3_gpu_linear_bf16
 * whose weight is stored the other way round. */
static uint16_t *transpose(const uint16_t *source, uint32_t rows,
                           uint32_t columns) {
    uint16_t *flipped = malloc((size_t)rows * columns * sizeof(*flipped));
    if (!flipped) return NULL;
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t column = 0; column < columns; column++)
            flipped[(size_t)column * rows + row] = source[(size_t)row * columns + column];
    return flipped;
}

typedef struct {
    h3_lora *lora;
    h3_gpu *gpu;
    int skip;
    char *error;
    size_t error_size;
} merge_context;

/* Chunked delta = up[rows, RANK] @ down[RANK, columns], merged as it is
 * produced. qkv_layout places each 128-row head block at (head * 3 + channel) *
 * HEAD_DIM because the in-memory qkv order is per-head triples. */
static int merge_chunks(merge_context *context, const uint16_t *up,
                        const uint16_t *down_transposed, uint32_t rows,
                        uint32_t columns, uint32_t channel, int qkv_layout,
                        h3_gpu_tensor *weight, h3_gpu_tensor *scales,
                        int bf16_weight) {
    h3_gpu *gpu = context->gpu;
    uint32_t chunk = rows < MERGE_CHUNK ? rows : MERGE_CHUNK;
    h3_gpu_tensor *down = h3_gpu_tensor_from_bf16(
        gpu, down_transposed, (size_t)columns * RANK);
    h3_gpu_tensor *delta = context->skip ? NULL :
        h3_gpu_tensor_new_bf16(gpu, (size_t)chunk * columns);
    if (!down || (!context->skip && !delta)) {
        fail(context->error, context->error_size,
             "cannot allocate the LoRA delta tile: %s", h3_gpu_error(gpu));
        h3_gpu_tensor_free(down);
        h3_gpu_tensor_free(delta);
        return 0;
    }
    int ok = 1;
    for (uint32_t start = 0; start < rows && ok; start += chunk) {
        uint32_t count = rows - start < chunk ? rows - start : chunk;
        h3_gpu_tensor *up_chunk = h3_gpu_tensor_from_bf16(
            gpu, up + (size_t)start * RANK, (size_t)count * RANK);
        if (!up_chunk) {
            fail(context->error, context->error_size,
                 "cannot upload the LoRA up chunk: %s", h3_gpu_error(gpu));
            ok = 0;
            break;
        }
        if (!context->skip)
            ok = h3_gpu_linear_bf16(gpu, delta, up_chunk, down, NULL, count,
                                    RANK, columns);
        /* The INT8 DiT weights are stored in ConvRot space
         * (W_rot = W @ H, 256-column groups), so a delta computed in plain
         * space has to be rotated before it can be added. Merging it as-is
         * mixes two bases and is silently wrong. The BF16 token-refiner
         * weights are not rotated. */
        if (ok && !context->skip && !bf16_weight)
            ok = h3_gpu_convrot_rotate_bf16(gpu, delta, delta, count, columns);
        /* H3_LORA_SKIP leaves `delta` NULL, so the add must not be issued at
         * all: the BF16 kernel rejects a null delta with "invalid LoRA BF16
         * merge request" and the run dies at the token refiner -- which is how
         * the switch turned out to be unusable end to end (§25). The INT8 path
         * tolerated it and wrote zeros, which is why the merge gate never
         * noticed. */
        if (ok && !context->skip && qkv_layout) {
            for (uint32_t block = 0; block < count / HEAD_DIM && ok; block++) {
                uint32_t head = (start + block * HEAD_DIM) / HEAD_DIM;
                uint32_t destination = (head * 3u + channel) * HEAD_DIM;
                ok = bf16_weight
                    ? h3_gpu_lora_add_bf16(gpu, weight, delta, destination,
                                           block * HEAD_DIM, HEAD_DIM, columns)
                    : h3_gpu_lora_merge_i8(gpu, weight, scales, delta,
                                           destination, block * HEAD_DIM,
                                           HEAD_DIM, columns);
            }
        } else if (ok && !context->skip) {
            ok = bf16_weight
                ? h3_gpu_lora_add_bf16(gpu, weight, delta, start, 0, count,
                                       columns)
                : h3_gpu_lora_merge_i8(gpu, weight, scales, delta, start, 0,
                                       count, columns);
        }
        h3_gpu_tensor_free(up_chunk);
    }
    h3_gpu_tensor_free(down);
    h3_gpu_tensor_free(delta);
    if (!ok && (!context->error || !*context->error))
        fail(context->error, context->error_size, "LoRA merge kernel failed: %s",
             h3_gpu_error(gpu));
    return ok;
}

static int merge_single(merge_context *context, const char *prefix,
                        const char *module, uint32_t rows, uint32_t columns,
                        int gate_up, h3_gpu_tensor *weight,
                        h3_gpu_tensor *scales, int bf16_weight) {
    char name[192];
    snprintf(name, sizeof(name), "%s%s.lora_up.weight", prefix, module);
    uint16_t *up = read_matrix(context->lora, name, rows, RANK, context->error,
                               context->error_size);
    if (!up) return 0;
    snprintf(name, sizeof(name), "%s%s.lora_down.weight", prefix, module);
    uint16_t *down = read_matrix(context->lora, name, RANK, columns,
                                 context->error, context->error_size);
    if (!down) {
        free(up);
        return 0;
    }
    /* Wan2GP feeds ff.net.0.proj through chunk(2)[::-1]; the checkpoint's fc1 is
     * [gate; up], so the adaptor halves are swapped to match. */
    if (gate_up) {
        size_t half = (size_t)(rows / 2) * RANK;
        uint16_t *head = malloc(half * sizeof(*head));
        if (!head) {
            free(up);
            free(down);
            fail(context->error, context->error_size,
                 "out of memory swapping the fc1 adaptor");
            return 0;
        }
        memcpy(head, up, half * sizeof(*head));
        memcpy(up, up + half, half * sizeof(*head));
        memcpy(up + half, head, half * sizeof(*head));
        free(head);
    }
    uint16_t *down_transposed = transpose(down, RANK, columns);
    free(down);
    if (!down_transposed) {
        free(up);
        fail(context->error, context->error_size,
             "out of memory transposing the LoRA down matrix");
        return 0;
    }
    int ok = merge_chunks(context, up, down_transposed, rows, columns, 0, 0,
                          weight, scales, bf16_weight);
    free(up);
    free(down_transposed);
    if (ok) context->lora->modules++;
    return ok;
}

static int merge_qkv(merge_context *context, const char *prefix,
                     h3_gpu_tensor *weight, h3_gpu_tensor *scales,
                     int bf16_weight) {
    static const char *modules[3] = {"attn.to_q", "attn.to_k", "attn.to_v"};
    char name[192];
    for (uint32_t channel = 0; channel < 3; channel++) {
        snprintf(name, sizeof(name), "%s%s.lora_up.weight", prefix,
                 modules[channel]);
        uint16_t *up = read_matrix(context->lora, name, INNER, RANK,
                                   context->error, context->error_size);
        if (!up) return 0;
        snprintf(name, sizeof(name), "%s%s.lora_down.weight", prefix,
                 modules[channel]);
        uint16_t *down = read_matrix(context->lora, name, RANK, HIDDEN,
                                     context->error, context->error_size);
        if (!down) {
            free(up);
            return 0;
        }
        uint16_t *down_transposed = transpose(down, RANK, HIDDEN);
        free(down);
        if (!down_transposed) {
            free(up);
            fail(context->error, context->error_size,
                 "out of memory transposing a qkv adaptor");
            return 0;
        }
        int ok = merge_chunks(context, up, down_transposed, INNER, HIDDEN,
                              channel, 1, weight, scales, bf16_weight);
        free(up);
        free(down_transposed);
        if (!ok) return 0;
        context->lora->modules++;
    }
    return 1;
}

int h3_lora_merge_block_int8(h3_lora *lora, h3_gpu *gpu, unsigned block,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *qkv_scales,
                             h3_gpu_tensor *out, h3_gpu_tensor *out_scales,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *fc1_scales,
                             h3_gpu_tensor *fc2, h3_gpu_tensor *fc2_scales,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!lora || !gpu || !qkv || !qkv_scales || !out || !out_scales || !fc1 ||
        !fc1_scales || !fc2 || !fc2_scales) {
        fail(error, error_size, "invalid LoRA merge arguments for block %u",
             block);
        return 0;
    }
    merge_context context = {lora, gpu, getenv("H3_LORA_SKIP") != NULL, error,
                             error_size};
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "transformer_blocks.%u.", block);
    double started = now();
    int ok = merge_qkv(&context, prefix, qkv, qkv_scales, 0) &&
             merge_single(&context, prefix, "attn.to_out.0", HIDDEN, INNER, 0,
                          out, out_scales, 0) &&
             merge_single(&context, prefix, "ff.net.0.proj", FFN * 2, HIDDEN,
                          1, fc1, fc1_scales, 0) &&
             merge_single(&context, prefix, "ff.net.2", HIDDEN, FFN, 0, fc2,
                          fc2_scales, 0);
    lora->seconds += now() - started;
    return ok;
}

int h3_lora_merge_refiner(h3_lora *lora, h3_gpu *gpu, unsigned index,
                          h3_gpu_tensor *qkv, h3_gpu_tensor *out,
                          h3_gpu_tensor *fc1, h3_gpu_tensor *fc2, char *error,
                          size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!lora || !gpu || !qkv || !out || !fc1 || !fc2) {
        fail(error, error_size, "invalid LoRA merge arguments for refiner %u",
             index);
        return 0;
    }
    merge_context context = {lora, gpu, getenv("H3_LORA_SKIP") != NULL, error,
                             error_size};
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "token_refiner.refiner_blocks.%u.", index);
    double started = now();
    int ok = merge_qkv(&context, prefix, qkv, NULL, 1) &&
             merge_single(&context, prefix, "attn.to_out.0", HIDDEN, INNER, 0,
                          out, NULL, 1) &&
             merge_single(&context, prefix, "ff.net.0.proj", FFN * 2, HIDDEN,
                          1, fc1, NULL, 1) &&
             merge_single(&context, prefix, "ff.net.2", HIDDEN, FFN, 0, fc2,
                          NULL, 1);
    lora->seconds += now() - started;
    return ok;
}
