/* h3rt - Python-free native runtime for the MiniMax-H3 / HyperFlow H3 Flash
 * pipeline formerly driven by ComfyUI_MiniMax-H3-V8.
 *
 * This is the stable C ABI consumed by .NET through P/Invoke.  It is a pure C
 * contract: no C++ types, no STL, no callbacks that outlive the call, and no
 * Python objects.  Strings are UTF-8 and are passed as JSON documents so the
 * ABI does not have to change when a new generation knob is added.
 *
 * Ownership rules
 *   - handles: created by h3rt_create(), released by h3rt_destroy()
 *   - strings returned in char** / char*: owned by the caller, released with
 *     h3rt_free_string()
 *   - const char* returned by h3rt_last_error()/h3rt_version(): borrowed
 *   - callback pointers: only valid for the duration of the callback
 */
#ifndef H3RT_H
#define H3RT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(H3RT_BUILDING_DLL)
#    define H3RT_API __declspec(dllexport)
#  else
#    define H3RT_API __declspec(dllimport)
#  endif
#  define H3RT_CALL __cdecl
#else
#  if defined(__GNUC__)
#    define H3RT_API __attribute__((visibility("default")))
#  else
#    define H3RT_API
#  endif
#  define H3RT_CALL
#endif

#define H3RT_VERSION "1.0.0"
#define H3RT_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3rt_handle h3rt_handle;

enum h3rt_status {
    H3RT_OK = 0,
    H3RT_ERR_INVALID_ARGUMENT = -1,
    H3RT_ERR_INVALID_JSON = -2,
    H3RT_ERR_MODEL = -3,
    H3RT_ERR_DEVICE = -4,
    H3RT_ERR_GENERATE = -5,
    H3RT_ERR_BUSY = -6,
    H3RT_ERR_CANCELLED = -7,
    H3RT_ERR_OUT_OF_MEMORY = -8,
    H3RT_ERR_UNSUPPORTED = -9,
    H3RT_ERR_THREAD = -10
};

/* One decoded RGB frame.  `rgb` points to engine-owned memory and is valid
 * only inside the callback.  Copy it if the caller needs it afterwards. */
typedef struct {
    int width;
    int height;
    int stride;
    const uint8_t *rgb;
    int frame_index;
    int frame_count;
    int denoise_step;      /* >= 0 only for intermediate denoise previews */
    int denoise_steps;
} h3rt_frame;

/* Returning non-zero from a callback aborts the generation. */
typedef int (H3RT_CALL *h3rt_progress_cb)(const char *phase_utf8,
                                          int completed, int total,
                                          void *opaque);
typedef int (H3RT_CALL *h3rt_frame_cb)(const h3rt_frame *frame, void *opaque);

typedef struct {
    h3rt_progress_cb on_progress;
    h3rt_frame_cb on_frame;
    void *opaque;
} h3rt_callbacks;

/* --- library ---------------------------------------------------------- */
H3RT_API const char *H3RT_CALL h3rt_version(void);
H3RT_API int H3RT_CALL h3rt_abi_version(void);
H3RT_API const char *H3RT_CALL h3rt_status_text(int status);
/* JSON with runtime/engine/build details. Free with h3rt_free_string(). */
H3RT_API char *H3RT_CALL h3rt_build_info(void);
H3RT_API void H3RT_CALL h3rt_free_string(char *text);

/* --- lifecycle -------------------------------------------------------- */
/* options_json (UTF-8, may be NULL):
 *   { "model_dir": "D:/models/pdd8",   MiniMax-H3 pack (FL2VA/ ...)
 *     "config": "D:/.../h3.config.json", optional path-table file
 *     "models": { ... },               optional per-component overrides
 *     "pack_cache_dir": "...",         where composed link packs live
 *     "text_cache_dir": "...",         optional on-disk prompt cache
 *     "cache_enabled": true,           reuse resident weights / embeddings
 *     "keep_dit": true }               keep the DiT mapped between calls
 *
 * Precedence for the model directory: model_dir -> config.h3_dir ->
 * "~/h3-models/pdd8" (the last one is skipped when "models" is used, because
 * the components may come from anywhere).
 *
 * "models" names each weight component independently.  Every value is either a
 * directory holding that component's files, or, except for the tokenizer, a
 * single .safetensors that is linked in as model.safetensors (any small
 * sidecar the same slot has under model_dir, such as config.json or
 * pdd_heads.safetensors, is linked next to it):
 *   { "text_encoder": "...",          -> FL2VA/text_encoder
 *     "transformer": "...",           -> FL2VA/transformer
 *     "video_vae": "...",             -> FL2VA/video_vae/source
 *     "audio_vae": "...",             -> FL2VA/audio_vae
 *     "tokenizer": "...",             -> FL2VA/tokenizer
 *     "pdd_lora": "...",              -> FL2VA/pdd_lora.safetensors
 *     "ref2va_transformer": "...",    -> Ref2VA/transformer
 *     "ref2va_text_encoder": "...",   -> Ref2VA/text_encoder
 *     "ref2va_video_vae": "...",      -> Ref2VA/video_vae/source
 *     "ref2va_audio_vae": "...",      -> Ref2VA/audio_vae
 *     "ref2va_tokenizer": "..." }     -> Ref2VA/tokenizer
 *
 * The engine reads one pack root, so h3rt materialises that layout as links
 * (junctions on Windows, symlinks on Linux) inside a content-addressed
 * directory under pack_cache_dir, and loads that.  Components left out fall
 * back to the same slot under model_dir.  When every component already sits at
 * its standard location no links are created and model_dir is used directly.
 * A manifest of the links is written to <pack>/h3rt-pack.json.
 *
 * `error` may be NULL; when it is not, it receives the failure message even
 * though no handle was produced.
 */
H3RT_API int H3RT_CALL h3rt_create(const char *options_json,
                                   h3rt_handle **out_handle,
                                   char *error, size_t error_size);
H3RT_API void H3RT_CALL h3rt_destroy(h3rt_handle *handle);
/* Thread-local snapshot of the last error on this handle (borrowed). */
H3RT_API const char *H3RT_CALL h3rt_last_error(h3rt_handle *handle);

/* --- introspection ---------------------------------------------------- */
/* All three return malloc'd JSON; free with h3rt_free_string(). */
H3RT_API char *H3RT_CALL h3rt_device_info(h3rt_handle *handle);
H3RT_API char *H3RT_CALL h3rt_model_info(h3rt_handle *handle);
H3RT_API char *H3RT_CALL h3rt_cache_info(h3rt_handle *handle);
H3RT_API int H3RT_CALL h3rt_cache_clear(h3rt_handle *handle);
H3RT_API int H3RT_CALL h3rt_set_text_cache_dir(h3rt_handle *handle,
                                                const char *dir);

/* --- performance ------------------------------------------------------ */
/* Warm the denoiser before the first request: pays the DiT weight read once
 * so the first user-visible generation only pays the VAEs.  request_json uses
 * the same fields as h3rt_generate but only the layout-derived ones matter
 * (width/height/frames/steps/pdd8/dit_layers/denoise_reuse/core_reuse).
 * Returns H3RT_OK when the DiT is resident afterwards.  Safe to ignore a
 * failure: the next h3rt_generate loads it anyway. */
H3RT_API int H3RT_CALL h3rt_preload(h3rt_handle *handle,
                                    const char *request_json,
                                    char *error, size_t error_size);

/* --- generation ------------------------------------------------------- */
/* request_json (UTF-8) accepts the ComfyUI-V8-equivalent fields documented
 * in README.md.  On success *out_json receives a malloc'd JSON result:
 *   { "status":"ok", "model":"h3", "output":"...", "width":864,
 *     "height":480, "frames":56, "fps":25, "sample_rate":32000,
 *     "seed":42, "seed_string":"42", "elapsed_ms":12345.6,
 *     "frames_per_second":4.5,
 *     "bytes":1234567, "content_type":"video/mp4", "pdd8":true,
 *     "engine":"0.2.3",
 *     "phases":[{"phase":"DiT","at_ms":120.5}, ...] }
 * On failure *out_json is NULL and h3rt_last_error() has the message.
 *
 * One generation at a time per handle (H3RT_ERR_BUSY otherwise).  The engine
 * runs on an internal 8 MiB-stack thread, so a .NET thread-pool thread can
 * safely call this.  callbacks may be NULL.
 */
H3RT_API int H3RT_CALL h3rt_generate(h3rt_handle *handle,
                                     const char *request_json,
                                     const h3rt_callbacks *callbacks,
                                     char **out_json);
/* Cooperative cancellation.  The running generation aborts at the next
 * progress/frame callback; there is no forced thread termination. */
H3RT_API int H3RT_CALL h3rt_cancel(h3rt_handle *handle);
H3RT_API int H3RT_CALL h3rt_is_cancelled(h3rt_handle *handle);

#ifdef __cplusplus
}
#endif
#endif /* H3RT_H */
