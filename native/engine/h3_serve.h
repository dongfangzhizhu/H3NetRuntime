#ifndef H3_SERVE_H
#define H3_SERVE_H

#include "h3_config.h"

/* Options for the resident HTTP service (see h3_serve.c). */
typedef struct {
    const char *model_dir;    /* legacy single-pack alias for -d */
    const char *model_type;   /* optional "h3"/"zimage"/"wan22": preload */
    /* Pack directories. NULL/empty falls through to config->* then to the
     * built-in "~/h3-models/<model>". */
    const char *h3_dir;       /* MiniMax-H3 pack */
    const char *zimage_dir;   /* Z-Image pack */
    /* The Nunchaku DiT repacks for the same Z-Image model: a directory of
     * svdq-*.safetensors (one pack each) or a single .safetensors, not a pack.
     * The VAE and text encoder stay in zimage_dir.  Empty disables the
     * "nunchaku" checkpoint tier. */
    const char *zimage_nunchaku_dir;
    const char *wan22_dir;    /* Wan 2.2 pack */
    const char *qwen21_dir;   /* Qwen-Image 2.1 pack */
    /* Qwen-Image 2.1 adapters (PEFT/Diffusers .safetensors). Default is
     * <qwen21_dir>/loras; a request names one by file name only. */
    const char *qwen21_lora_dir;
    /* Paths from the config file, or NULL when no file was read. */
    const h3_config *config;
    const char *host;         /* default 127.0.0.1 */
    int port;                 /* default 8321 */
    int keep_dit;             /* default 1: retain the DiT between requests */
    int preload;              /* default 1: warm the startup model's DiT */
    const char *text_cache_dir; /* default ~/.cache/h3-text */
    const char *wan22_text_cache_dir; /* default <text_cache_dir>/wan22 */
    const char *qwen21_text_cache_dir; /* default <text_cache_dir>/qwen21 */
} h3_serve_options;

/* Serve until SIGINT/SIGTERM. No model is loaded unless --model-type (or the
 * legacy -d) names one; /generate then brings models into residence on
 * demand. Returns 0 after a clean shutdown and 1 when the startup model, the
 * cache directory, or the listening socket could not be prepared. */
int h3_serve_run(const h3_serve_options *options);

#endif
