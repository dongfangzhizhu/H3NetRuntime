#ifndef H3_GGUF_H
#define H3_GGUF_H

/* GGUF (llama.cpp) checkpoint reader, used by the text and vision encoders.
 *
 * The released MiniMax-H3 text encoder ships as a Q4_K_M `.gguf`
 * (`qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf`), not as a safetensors directory, so
 * the engine has to read the container itself: header, tensor table, and the
 * k-quant block formats.  Tensors are presented to the rest of the engine in
 * torch layout (shape reversed, `model.language_model.*` naming) and are
 * dequantised to BF16 on read, which keeps every downstream consumer on the
 * same code path it already uses for a BF16 checkpoint.
 *
 * Deliberately read-only and dependency-free: no mmap, no allocator hooks, one
 * file descriptor and pread() so several prefetch lanes can share it.
 */

#include <stddef.h>
#include <stdint.h>

typedef struct h3_gguf h3_gguf;

/* Tensor payload formats this reader can convert. */
typedef enum {
    H3_GGUF_F32 = 0,
    H3_GGUF_F16 = 1,
    H3_GGUF_Q4_0 = 2,
    H3_GGUF_Q4_1 = 3,
    H3_GGUF_Q5_0 = 6,
    H3_GGUF_Q5_1 = 7,
    H3_GGUF_Q8_0 = 8,
    H3_GGUF_Q2_K = 10,
    H3_GGUF_Q3_K = 11,
    H3_GGUF_Q4_K = 12,
    H3_GGUF_Q5_K = 13,
    H3_GGUF_Q6_K = 14,
    H3_GGUF_BF16 = 30,
    H3_GGUF_UNKNOWN = 255
} h3_gguf_type;

h3_gguf *h3_gguf_open(const char *path, char *error, size_t error_size);
void h3_gguf_free(h3_gguf *gguf);

size_t h3_gguf_tensor_count(const h3_gguf *gguf);
/* Index by canonical name ("model.layers.0.self_attn.q_proj.weight",
 * "visual.blocks.0..."), or -1.  GGUF files written by ComfyUI keep the HF
 * names, llama.cpp conversions use blk.*, so every lookup is canonicalised
 * first and either spelling finds the tensor. */
int h3_gguf_find(const h3_gguf *gguf, const char *name);

const char *h3_gguf_tensor_name(const h3_gguf *gguf, size_t index);
h3_gguf_type h3_gguf_tensor_type(const h3_gguf *gguf, size_t index);
const char *h3_gguf_type_name(h3_gguf_type type);
int h3_gguf_tensor_rank(const h3_gguf *gguf, size_t index);
/* Shape in torch order: 2D dims are swapped, 5D shapes come from the
 * `comfy.gguf.orig_shape.<name>` metadata the ComfyUI converter writes. */
const uint64_t *h3_gguf_tensor_shape(const h3_gguf *gguf, size_t index);
uint64_t h3_gguf_tensor_elements(const h3_gguf *gguf, size_t index);

/* Convert one tensor into BF16 / F32. `out` holds elements() values. */
int h3_gguf_read_bf16(const h3_gguf *gguf, size_t index, uint16_t *out,
                      char *error, size_t error_size);
int h3_gguf_read_f32(const h3_gguf *gguf, size_t index, float *out,
                     char *error, size_t error_size);

/* Canonical form of a checkpoint tensor name; exposed for the store's
 * diagnostics and for tests. */
void h3_gguf_canonical_name(const char *name, char *out, size_t size);

#endif
