#ifndef H3_INTERNAL_H
#define H3_INTERNAL_H

#include "h3.h"
#include "h3_host.h"

#include <stdarg.h>

struct h3_dit;
struct h3_video_vae_decoder;

struct h3_ctx {
    char *model_dir;
    char error[512];
    h3_device_info device;
    h3_model_info model;
    int cache_enabled;
    char *conditioning_key;
    size_t conditioning_tokens;
    size_t conditioning_width;
    uint16_t *conditioning_values;
    uint8_t *conditioning_tags;
    float *conditioning_video_rows;
    size_t conditioning_video_elements;
    float *conditioning_audio_rows;
    size_t conditioning_audio_elements;
    h3_layout_ref *conditioning_references;
    size_t conditioning_reference_count;
    int conditioning_present;
    char *dit_key;
    /* Weight-structural key of the retained DiT: prompt- and layout-free, so
     * any request with the same checkpoint/LoRA/sigma grid/block count rebinds
     * the layout state instead of reloading the weights. dit_key stays the
     * full request key that identifies the identical-request fast path. */
    char *dit_weight_key;
    struct h3_dit *dit;
    char *video_decoder_key;
    struct h3_video_vae_decoder *video_decoder;
    /* Directory for the on-disk text embedding cache; NULL disables it. */
    char *text_cache_dir;
    /* Cumulative DiT weight loads served by this context (resident-service
     * statistic: a rebind or an identical-request reuse does not count). */
    unsigned long long dit_loads;
};

void h3_set_error(h3_ctx *ctx, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

#endif
