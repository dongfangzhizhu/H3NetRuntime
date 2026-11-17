#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

/* Upper bound on the transformer depth any checkpoint may declare; the real
 * depth is read from the file, so this only sizes fixed arrays. */
#define H3_WEIGHT_MAX_BLOCKS 64u

typedef struct h3_weight_store h3_weight_store;

/* Which Nunchaku SVDQuant release a store holds.  The int4 and fp4 packs have
 * the same tensor names, so nothing else tells them apart. */
typedef enum {
    H3_SVDQ_NONE = 0,
    H3_SVDQ_INT4,
    H3_SVDQ_FP4
} h3_svdq_kind;

/* Open every safetensors header of a component without reading tensor
 * payloads.  `path` is either a directory of shards or a single
 * `.safetensors` file, which is what a pack selected out of a directory of
 * packs resolves to. */
h3_weight_store *h3_weight_store_open(const char *path,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Identify a Nunchaku SVDQuant pack: the low-rank width (the second dimension
 * of any `proj_up`) and the code format, which is the dtype of the matching
 * `wscales` -- BF16 for int4, float8 for fp4.  Both come from the tensors, not
 * from `__metadata__`, so a repack that dropped the metadata still identifies.
 * Returns H3_SVDQ_NONE and leaves *rank at 0 for a store that is not one. */
h3_svdq_kind h3_weight_svdq_identify(const h3_weight_store *store,
                                     uint32_t *rank);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

/* True when the store reads a single quantised .gguf instead of safetensors
 * shards. Its tensors have no BF16 payload and no byte offset. */
int h3_weight_is_gguf(const h3_weight_store *store);

/* Depth of the transformer in this store, counted from the contiguous
 * blocks.N.norm1.weight names.  The unpruned release has 50 blocks; HyperFlow
 * Flash drops whole blocks and renumbers the survivors (fl2va 47, ref2va 49),
 * so nothing may assume 50. Returns 0 when the store holds no blocks. */
unsigned h3_weight_block_count(const h3_weight_store *store);

/* Read a small F32 vector (latents_mean / latents_std) straight out of the
 * checkpoint in `directory`, for the VAEs that ship their latent
 * normalization as tensors instead of a config.json. */
int h3_weight_read_f32_vector(const char *directory, const char *name,
                              float *values, size_t count,
                              char *error, size_t error_size);

/* Read one weight into an already-allocated BF16 device tensor. Safetensors
 * takes the zero-copy path straight into the destination; a GGUF tensor is
 * dequantised on the host and uploaded, which is what lets the Qwen prefetch
 * lanes read a quantised text encoder without knowing the container. */
int h3_weight_read_into_bf16(const h3_weight_store *store, const char *name,
                             h3_gpu_tensor *destination, size_t elements,
                             char *error, size_t error_size);

/* INT8 ConvRot checkpoints (Comfy/WanGP `int8_convrot`) store the DiT linears as
 * "<name>" I8 [rows, cols] plus a per-output-row F32 scale sidecar
 * "<name minus .weight>.weight_scale" of [rows, 1] or [rows].  There is no BF16
 * copy in the file, so these weights are read straight into device memory and the
 * scale is used as-is by the INT8 GEMM; the ConvRot Hadamard rotation lives on the
 * activation side and is not applied here. */
h3_gpu_tensor *h3_weight_load_i8_convrot(const h3_weight_store *store, h3_gpu *gpu,
                                         const char *name, uint64_t rows,
                                         uint64_t cols, h3_gpu_tensor **scales,
                                         char *error, size_t error_size);

/* Non-zero when the named tensor exists and is INT8, i.e. the checkpoint stores
 * quantized DiT weights instead of BF16 ones. */
int h3_weight_is_int8(const h3_weight_store *store, const char *name);
/* ConvRot checkpoints store qkv rows as three contiguous thirds [q][k][v], but
 * h3 consumes them as per-head triples [h0q, h0k, h0v, h1q, ...]. These entries
 * read the payload in destination-block order (one 128-row block at a time, no
 * second pass over the tensor) and return the per-head layout; the per-row
 * scales follow the same permutation. Rotation runs along K, so only the row
 * order changes. `rows` must be 3 * heads * head_dim. */
h3_gpu_tensor *h3_weight_load_qkv_perhead(const h3_weight_store *store, h3_gpu *gpu,
                                          const char *name, uint64_t rows,
                                          uint64_t cols, uint32_t head_dim,
                                          h3_gpu_tensor **scales, char *error,
                                          size_t error_size);
h3_gpu_tensor *h3_weight_load_qkv_perhead_bf16(
                                          const h3_weight_store *store, h3_gpu *gpu,
                                          const char *name, uint64_t rows,
                                          uint64_t cols, uint32_t head_dim,
                                          char *error, size_t error_size);

#endif
