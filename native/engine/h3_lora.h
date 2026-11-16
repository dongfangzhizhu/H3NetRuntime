/* PDD LoRA merge for the pruned FL2VA checkpoint. */
#ifndef H3_LORA_H
#define H3_LORA_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct h3_lora h3_lora;

/* Opens the adaptor file and indexes its 362 modules. Every failure names the
 * tensor that is missing or malformed. */
h3_lora *h3_lora_open(const char *path, char *error, size_t error_size);
void h3_lora_free(h3_lora *lora);

/* Merges one DiT block's four INT8 projections in place: the INT8 codes are
 * dequantized per row, the adaptor product is added, and each row is
 * requantized against a fresh scale. */
int h3_lora_merge_block_int8(h3_lora *lora, h3_gpu *gpu, unsigned block,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *qkv_scales,
                             h3_gpu_tensor *out, h3_gpu_tensor *out_scales,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *fc1_scales,
                             h3_gpu_tensor *fc2, h3_gpu_tensor *fc2_scales,
                             char *error, size_t error_size);

/* The token refiner stays BF16, so its adaptors are added without requantizing. */
int h3_lora_merge_refiner(h3_lora *lora, h3_gpu *gpu, unsigned index,
                          h3_gpu_tensor *qkv, h3_gpu_tensor *out,
                          h3_gpu_tensor *fc1, h3_gpu_tensor *fc2,
                          char *error, size_t error_size);

/* Folds the rank-64 AdaLN adaptor into the rank-8 curve projection. The
 * adaptor was trained against the 2688-wide modulation curve, so it is mapped
 * back through adaln_affine_map (Wan2GP's lora_affine_maps package, resolved
 * from H3_LORA_AFFINE or from the adaptor file's sibling path):
 *   mapped = A @ M.T ; A8 = mapped[:, :8] ; b64 = mapped[:, 8]
 *   W8 += B @ A8 ; b8 += B @ b64
 * matrix is [output][rank] row-major, offset is [output]; both are updated in
 * place. Returns 1 when this block has an adaptor and it was applied, 0 when
 * the block has none, -1 on error. */
int h3_lora_adaln_delta(h3_lora *lora, unsigned block, float *matrix,
                        float *offset, uint32_t output, uint32_t rank,
                        char *error, size_t error_size);

/* Longest path the affine-map lookup will build. */
#define H3_LORA_PATH_MAX 1024

/* Fills paths[0..n) with the places the AdaLN affine map may live, in the order
 * the loader tries them: beside the adaptor first (a pack carries the pair),
 * then the Wan2GP tree the adaptor was published in. Returns n, 0 when the
 * adaptor path has no directory component. */
int h3_lora_affine_candidates(const char *lora_path,
                              char paths[][H3_LORA_PATH_MAX], int max);

double h3_lora_seconds(const h3_lora *lora);

/* Adaptors merged so far, and adaptors the file holds (counted at open from
 * its `*.lora_down.weight` tensors).  A merge that stops short of the file --
 * a projection this build does not know about -- is otherwise invisible: the
 * video still looks like a video.  h3_dit.c logs both numbers. */
unsigned h3_lora_modules_merged(const h3_lora *lora);
unsigned h3_lora_modules_indexed(const h3_lora *lora);

/* Called by h3_dit_schedule_precompute before it rebuilds the modulation
 * tables: the AdaLN count belongs to the schedule being built, not to the
 * adaptor's lifetime, or a second build would report twice as many. */
void h3_lora_reset_adaln(h3_lora *lora);
#endif
