/* h3rt - in-process, Python-free MiniMax-H3 runtime.
 *
 * This translation unit is the only code added to the vendored h3-spark.c
 * engine.  It turns h3.h's C API into a small, stable, JSON-driven ABI that
 * .NET can P/Invoke without a Python interpreter, without ComfyUI, and
 * without an HTTP round trip.
 */
#include "h3rt.h"

#include "h3.h"
#include "h3_config.h"
#include "h3_host.h"
#include "cJSON.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#if defined(_WIN32)
#include <process.h>
/* The Win32 compatibility header is force-included into every TU and must not
 * pull in <windows.h> (it breaks nvcc).  Declare the two kernel32 entry points
 * we need for the big-stack worker instead of including the whole SDK header. */
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(void *handle,
                                                                  unsigned long ms);
__declspec(dllimport) int __stdcall CloseHandle(void *handle);
#define H3RT_WAIT_INFINITE 0xFFFFFFFFul
#define H3RT_THREAD_RET unsigned __stdcall
#define H3RT_THREAD_RETURN return 0
#else
#define H3RT_THREAD_RET void *
#define H3RT_THREAD_RETURN return NULL
#endif

#ifndef H3_SM_ARCH
#define H3_SM_ARCH 0
#endif
#define H3RT_STRINGIFY_INNER(x) #x
#define H3RT_STRINGIFY(x) H3RT_STRINGIFY_INNER(x)

#define H3RT_ERROR_CAPACITY 1024
#define H3RT_PATH_CAPACITY 1024
#define H3RT_MAX_REFERENCES 16
#define H3RT_WORKER_STACK (8u * 1024u * 1024u)
#define H3RT_ENGINE_REVISION "a474eca"

/* h3_platform.h maps __thread to __declspec(thread) on MSVC. */
static __thread char h3rt_tls_error[H3RT_ERROR_CAPACITY];

struct h3rt_handle {
    h3_ctx *ctx;
    char model_dir[H3RT_PATH_CAPACITY];
    char text_cache_dir[H3RT_PATH_CAPACITY];
    int keep_dit;
    int cache_enabled;
    pthread_mutex_t api_mutex;    /* serialises generation and destruction */
    pthread_mutex_t error_mutex;
    char error[H3RT_ERROR_CAPACITY];
    long cancel;                  /* accessed with __atomic_* builtins */
    int busy;                     /* guarded by api_mutex */
    unsigned long long job_sequence;
};

#define H3RT_MAX_PHASES 64
typedef struct {
    char phase[64];
    double at_ms;
} h3rt_phase_mark;

typedef struct {
    h3rt_handle *handle;
    h3rt_callbacks user;
    h3_params params;
    const char *prompt;
    char output_path[H3RT_PATH_CAPACITY];
    int width;
    int height;
    int frames;
    int fps;
    int sample_rate;
    uint64_t seed;
    double elapsed_ms;
    double started_ms;
    h3rt_phase_mark phases[H3RT_MAX_PHASES];
    int phase_count;
    int cancelled;
    int ok;
    char error[H3RT_ERROR_CAPACITY];
} h3rt_generate_task;

/* ------------------------------------------------------------------ utils */

static void h3rt_set_error(h3rt_handle *handle, const char *fmt, ...) {
    if (!handle) return;
    va_list ap;
    pthread_mutex_lock(&handle->error_mutex);
    va_start(ap, fmt);
    vsnprintf(handle->error, sizeof(handle->error), fmt, ap);
    va_end(ap);
    pthread_mutex_unlock(&handle->error_mutex);
}

static void h3rt_write_create_error(char *error, size_t error_size,
                                    const char *fmt, ...) {
    if (!error || !error_size) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_size, fmt, ap);
    va_end(ap);
}

static const cJSON *h3rt_item(const cJSON *object, const char *key) {
    if (!object) return NULL;
    return cJSON_GetObjectItemCaseSensitive(object, key);
}

static const char *h3rt_str(const cJSON *object, const char *key,
                            const char *fallback) {
    const cJSON *item = h3rt_item(object, key);
    if (item && cJSON_IsString(item) && item->valuestring) return item->valuestring;
    return fallback;
}

static int h3rt_int(const cJSON *object, const char *key, int fallback) {
    const cJSON *item = h3rt_item(object, key);
    if (!item || !cJSON_IsNumber(item)) return fallback;
    double value = item->valuedouble;
    if (value < -2147483648.0 || value > 2147483647.0) return fallback;
    return (int)value;
}

static int h3rt_bool(const cJSON *object, const char *key, int fallback) {
    const cJSON *item = h3rt_item(object, key);
    if (!item) return fallback;
    if (cJSON_IsBool(item)) return cJSON_IsTrue(item) ? 1 : 0;
    if (cJSON_IsNumber(item)) return item->valuedouble != 0.0 ? 1 : 0;
    return fallback;
}

static uint64_t h3rt_seed(const cJSON *object, const char *key,
                          uint64_t fallback) {
    const cJSON *item = h3rt_item(object, key);
    if (!item) return fallback;
    if (cJSON_IsString(item) && item->valuestring)
        return strtoull(item->valuestring, NULL, 10);
    if (cJSON_IsNumber(item) && item->valuedouble >= 0.0)
        return (uint64_t)item->valuedouble;
    return fallback;
}

static double h3rt_now_ms(void) {
    struct timespec ts;
    memset(&ts, 0, sizeof(ts));
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0.0;
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static double h3rt_file_size(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0.0;
    double size = 0.0;
    if (fseek(file, 0, SEEK_END) == 0) {
        long position = ftell(file);
        if (position > 0) size = (double)position;
    }
    fclose(file);
    return size;
}

static char *h3rt_json_to_string(cJSON *value) {
    char *text = cJSON_PrintUnformatted(value);
    cJSON_Delete(value);
    return text;
}

/* ------------------------------------------------------------- callbacks */

static int h3rt_progress_bridge(const char *phase, int completed, int total,
                                void *opaque) {
    h3rt_generate_task *task = (h3rt_generate_task *)opaque;
    if (!task) return 0;
    if (__atomic_load_n(&task->handle->cancel, __ATOMIC_SEQ_CST)) {
        task->cancelled = 1;
        return 1;
    }
    /* Record phase transitions for the result's timing breakdown.  The
     * callback fires often; only a new phase name costs a strcmp. */
    if (task->phase_count < H3RT_MAX_PHASES) {
        const char *name = phase ? phase : "";
        if (task->phase_count == 0 ||
            strcmp(task->phases[task->phase_count - 1].phase, name) != 0) {
            snprintf(task->phases[task->phase_count].phase,
                     sizeof(task->phases[task->phase_count].phase), "%s", name);
            task->phases[task->phase_count].at_ms =
                h3rt_now_ms() - task->started_ms;
            task->phase_count++;
        }
    }
    if (task->user.on_progress) {
        int result = task->user.on_progress(phase, completed, total,
                                            task->user.opaque);
        if (result) task->cancelled = 1;
        return result;
    }
    return 0;
}

static int h3rt_frame_bridge(const h3_frame *frame, void *opaque) {
    h3rt_generate_task *task = (h3rt_generate_task *)opaque;
    if (!task || !frame) return 0;
    if (__atomic_load_n(&task->handle->cancel, __ATOMIC_SEQ_CST)) {
        task->cancelled = 1;
        return 1;
    }
    if (!task->user.on_frame) return 0;
    h3rt_frame out;
    memset(&out, 0, sizeof(out));
    out.width = frame->width;
    out.height = frame->height;
    out.stride = frame->stride;
    out.rgb = frame->rgb;
    out.frame_index = frame->frame_index;
    out.frame_count = frame->frame_count;
    out.denoise_step = frame->denoise_step;
    out.denoise_steps = frame->denoise_steps;
    int result = task->user.on_frame(&out, task->user.opaque);
    if (result) task->cancelled = 1;
    return result;
}

static H3RT_THREAD_RET h3rt_generate_worker(void *argument) {
    h3rt_generate_task *task = (h3rt_generate_task *)argument;
    task->started_ms = h3rt_now_ms();
    double started = task->started_ms;
    h3_result *result = h3_generate(task->handle->ctx, task->prompt,
                                    &task->params);
    task->elapsed_ms = h3rt_now_ms() - started;
    if (!result) {
        const char *message = h3_last_error(task->handle->ctx);
        snprintf(task->error, sizeof(task->error), "%s",
                 message && *message ? message : "generation failed");
        H3RT_THREAD_RETURN;
    }
    task->width = result->width;
    task->height = result->height;
    task->frames = result->frames;
    task->fps = result->fps;
    task->sample_rate = result->sample_rate;
    task->seed = result->seed;
    task->ok = 1;
    h3_result_free(result);
    H3RT_THREAD_RETURN;
}

/* The engine was written against an 8 MiB process stack.  A .NET thread-pool
 * thread has 1 MiB, so generation always runs on our own thread with a known
 * stack size; the calling thread only waits. */
static int h3rt_run_worker(h3rt_generate_task *task) {
#if defined(_WIN32)
    unsigned thread_id = 0;
    uintptr_t thread = _beginthreadex(NULL, H3RT_WORKER_STACK,
                                      h3rt_generate_worker, task, 0,
                                      &thread_id);
    if (!thread) return 0;
    WaitForSingleObject((void *)thread, H3RT_WAIT_INFINITE);
    CloseHandle((void *)thread);
    return 1;
#else
    pthread_attr_t attr;
    pthread_t thread;
    if (pthread_attr_init(&attr) != 0) return 0;
    (void)pthread_attr_setstacksize(&attr, H3RT_WORKER_STACK);
    int rc = pthread_create(&thread, &attr, h3rt_generate_worker, task);
    pthread_attr_destroy(&attr);
    if (rc != 0) return 0;
    pthread_join(thread, NULL);
    return 1;
#endif
}

/* ------------------------------------------------------------ references */

static int h3rt_reference_kind(const char *name, int fallback) {
    if (!name || !*name) return fallback;
    if (!strcmp(name, "image")) return H3_REFERENCE_IMAGE;
    if (!strcmp(name, "video")) return H3_REFERENCE_VIDEO;
    if (!strcmp(name, "audio")) return H3_REFERENCE_AUDIO;
    if (!strcmp(name, "video_audio") || !strcmp(name, "video+audio"))
        return H3_REFERENCE_VIDEO_AUDIO;
    return fallback;
}

static int h3rt_add_reference(h3_reference *refs, size_t *count, size_t max,
                              const char *kind_name, int numeric_kind,
                              const char *path, const char *audio_path,
                              int include_audio,
                              char *error, size_t error_size) {
    if (!path || !*path) {
        snprintf(error, error_size, "a reference is missing its path");
        return 0;
    }
    if (*count >= max) {
        snprintf(error, error_size, "at most %zu references are supported", max);
        return 0;
    }
    h3_reference *ref = &refs[*count];
    memset(ref, 0, sizeof(*ref));
    int kind = numeric_kind;
    if (kind_name) {
        if (!strcmp(kind_name, "image")) kind = H3_REFERENCE_IMAGE;
        else if (!strcmp(kind_name, "video")) kind = H3_REFERENCE_VIDEO;
        else if (!strcmp(kind_name, "audio")) kind = H3_REFERENCE_AUDIO;
        else if (!strcmp(kind_name, "video_audio") ||
                 !strcmp(kind_name, "video+audio"))
            kind = H3_REFERENCE_VIDEO_AUDIO;
        else {
            snprintf(error, error_size,
                     "unknown reference kind \"%s\" (image, video, audio or "
                     "video_audio)", kind_name);
            return 0;
        }
    }
    if (kind < H3_REFERENCE_IMAGE || kind > H3_REFERENCE_VIDEO_AUDIO) {
        snprintf(error, error_size, "a reference has an invalid kind");
        return 0;
    }
    ref->kind = (h3_reference_kind)kind;
    ref->path = path;
    ref->audio_path = audio_path;
    ref->include_embedded_audio = include_audio;
    (*count)++;
    return 1;
}

static int h3rt_collect_references(const cJSON *request, h3_reference *refs,
                                   size_t *count, size_t max,
                                   char *error, size_t error_size) {
    *count = 0;
    static const struct {
        const char *field;
        int kind;
    } singular[] = {
        {"reference_image", H3_REFERENCE_IMAGE},
        {"reference_video", H3_REFERENCE_VIDEO},
        {"reference_audio", H3_REFERENCE_AUDIO},
        {"reference_video_audio", H3_REFERENCE_VIDEO_AUDIO}
    };
    static const struct {
        const char *field;
        int kind;
    } plural[] = {
        {"reference_images", H3_REFERENCE_IMAGE},
        {"reference_videos", H3_REFERENCE_VIDEO},
        {"reference_audios", H3_REFERENCE_AUDIO}
    };
    for (size_t i = 0; i < sizeof(singular) / sizeof(singular[0]); i++) {
        const char *path = h3rt_str(request, singular[i].field, NULL);
        if (path && !h3rt_add_reference(refs, count, max, NULL,
                                        singular[i].kind, path, NULL, 0,
                                        error, error_size)) return 0;
    }
    for (size_t i = 0; i < sizeof(plural) / sizeof(plural[0]); i++) {
        const cJSON *array = h3rt_item(request, plural[i].field);
        if (!array || !cJSON_IsArray(array)) continue;
        int size = cJSON_GetArraySize(array);
        for (int j = 0; j < size; j++) {
            const cJSON *entry = cJSON_GetArrayItem(array, j);
            if (!cJSON_IsString(entry) || !entry->valuestring) {
                snprintf(error, error_size, "%s must contain file paths",
                         plural[i].field);
                return 0;
            }
            if (!h3rt_add_reference(refs, count, max, NULL, plural[i].kind,
                                    entry->valuestring, NULL, 0,
                                    error, error_size)) return 0;
        }
    }
    const cJSON *array = h3rt_item(request, "references");
    if (array && cJSON_IsArray(array)) {
        int size = cJSON_GetArraySize(array);
        for (int j = 0; j < size; j++) {
            const cJSON *entry = cJSON_GetArrayItem(array, j);
            if (!cJSON_IsObject(entry)) {
                snprintf(error, error_size,
                         "references[%d] must be an object", j);
                return 0;
            }
            const char *kind_name = h3rt_str(entry, "kind", NULL);
            int numeric_kind = h3rt_int(entry, "kind", 0);
            const char *path = h3rt_str(entry, "path", NULL);
            if (!path) path = h3rt_str(entry, "file", NULL);
            const char *audio_path = h3rt_str(entry, "audio_path", NULL);
            int include_audio = h3rt_bool(entry, "include_embedded_audio", 0);
            if (!h3rt_add_reference(refs, count, max, kind_name,
                                    numeric_kind, path, audio_path,
                                    include_audio, error, error_size))
                return 0;
        }
    } else if (array) {
        snprintf(error, error_size, "references must be an array");
        return 0;
    }
    return 1;
}

/* --------------------------------------------------------------- public */

const char *h3rt_version(void) { return H3RT_VERSION; }

int h3rt_abi_version(void) { return H3RT_ABI_VERSION; }

const char *h3rt_status_text(int status) {
    switch (status) {
        case H3RT_OK: return "ok";
        case H3RT_ERR_INVALID_ARGUMENT: return "invalid argument";
        case H3RT_ERR_INVALID_JSON: return "invalid JSON";
        case H3RT_ERR_MODEL: return "model error";
        case H3RT_ERR_DEVICE: return "device error";
        case H3RT_ERR_GENERATE: return "generation failed";
        case H3RT_ERR_BUSY: return "a generation is already running";
        case H3RT_ERR_CANCELLED: return "cancelled";
        case H3RT_ERR_OUT_OF_MEMORY: return "out of memory";
        case H3RT_ERR_UNSUPPORTED: return "unsupported request";
        case H3RT_ERR_THREAD: return "worker thread failed";
        default: return "unknown status";
    }
}

void h3rt_free_string(char *text) { free(text); }

char *h3rt_build_info(void) {
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;
    cJSON_AddStringToObject(json, "runtime_version", H3RT_VERSION);
    cJSON_AddStringToObject(json, "engine_version", H3_VERSION);
    cJSON_AddStringToObject(json, "engine_revision", H3RT_ENGINE_REVISION);
    cJSON_AddNumberToObject(json, "abi_version", H3RT_ABI_VERSION);
    cJSON_AddStringToObject(json, "platform",
#if defined(_WIN32)
                            "windows"
#else
                            "linux"
#endif
    );
    cJSON_AddStringToObject(json, "cuda_arch", H3RT_STRINGIFY(H3_SM_ARCH));
    cJSON_AddStringToObject(json, "model", "MiniMax-H3 / HyperFlow H3 Flash");
    cJSON_AddStringToObject(json, "output", "video/mp4 (video + audio)");
    cJSON_AddBoolToObject(json, "python_required", 0);
    cJSON_AddBoolToObject(json, "comfyui_required", 0);
    cJSON_AddStringToObject(json, "callbacks", "progress, decoded RGB frames");
    cJSON_AddStringToObject(json, "external_tools", "ffmpeg (mpeg muxing)");
    return h3rt_json_to_string(json);
}

int h3rt_create(const char *options_json, h3rt_handle **out_handle,
                char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!out_handle) {
        h3rt_write_create_error(error, error_size,
                                "out_handle must not be NULL");
        return H3RT_ERR_INVALID_ARGUMENT;
    }
    *out_handle = NULL;
    cJSON *options = NULL;
    if (options_json && *options_json) {
        options = cJSON_Parse(options_json);
        if (!options || !cJSON_IsObject(options)) {
            cJSON_Delete(options);
            h3rt_write_create_error(error, error_size,
                                    "options must be a JSON object");
            return H3RT_ERR_INVALID_JSON;
        }
    }

    h3rt_handle *handle = (h3rt_handle *)calloc(1, sizeof(*handle));
    if (!handle) {
        cJSON_Delete(options);
        h3rt_write_create_error(error, error_size, "out of memory");
        return H3RT_ERR_OUT_OF_MEMORY;
    }
    pthread_mutex_init(&handle->api_mutex, NULL);
    pthread_mutex_init(&handle->error_mutex, NULL);
    handle->keep_dit = h3rt_bool(options, "keep_dit", 1);
    handle->cache_enabled = h3rt_bool(options, "cache_enabled", 1);

    h3_config config;
    memset(&config, 0, sizeof(config));
    const char *config_path = h3rt_str(options, "config", NULL);
    if (config_path && *config_path) {
        char config_error[512] = "";
        if (h3_config_load(config_path, &config, config_error,
                           sizeof(config_error)) < 0) {
            h3rt_write_create_error(error, error_size,
                                    "cannot read config: %s", config_error);
            cJSON_Delete(options);
            pthread_mutex_destroy(&handle->api_mutex);
            pthread_mutex_destroy(&handle->error_mutex);
            free(handle);
            return H3RT_ERR_MODEL;
        }
    }

    const char *model_dir = h3rt_str(options, "model_dir", NULL);
    if (!model_dir || !*model_dir) model_dir = h3rt_str(options, "h3_dir", NULL);
    char resolved[H3RT_PATH_CAPACITY];
    const char *source = model_dir && *model_dir ? model_dir :
                         config.h3_dir[0] ? config.h3_dir :
                         "~/h3-models/pdd8";
    if (source[0] == '~') {
        char expanded[H3RT_PATH_CAPACITY];
        if (h3_config_expand_home(source, expanded, sizeof(expanded))) {
            snprintf(resolved, sizeof(resolved), "%s", expanded);
        } else {
            snprintf(resolved, sizeof(resolved), "%s", source);
        }
    } else {
        snprintf(resolved, sizeof(resolved), "%s", source);
    }
    snprintf(handle->model_dir, sizeof(handle->model_dir), "%s", resolved);

    h3_ctx *ctx = h3_load_dir(handle->model_dir);
    if (!ctx) {
        const char *message = h3_last_error(NULL);
        h3rt_write_create_error(error, error_size,
                                "cannot load model pack '%s': %s", resolved,
                                message && *message ? message :
                                "FL2VA/transformer/config.json not found");
        cJSON_Delete(options);
        pthread_mutex_destroy(&handle->api_mutex);
        pthread_mutex_destroy(&handle->error_mutex);
        free(handle);
        return H3RT_ERR_MODEL;
    }
    handle->ctx = ctx;
    h3_cache_set_enabled(ctx, handle->cache_enabled);

    const char *cache_dir = h3rt_str(options, "text_cache_dir", NULL);
    if (!cache_dir || !*cache_dir)
        cache_dir = config.text_cache_dir[0] ? config.text_cache_dir : NULL;
    if (cache_dir && *cache_dir && h3_set_text_cache_dir(ctx, cache_dir))
        snprintf(handle->text_cache_dir, sizeof(handle->text_cache_dir),
                 "%s", cache_dir);

    cJSON_Delete(options);
    *out_handle = handle;
    return H3RT_OK;
}

void h3rt_destroy(h3rt_handle *handle) {
    if (!handle) return;
    /* generate() holds api_mutex for its whole duration, so this waits for an
     * in-flight job instead of pulling the context out from under it. */
    pthread_mutex_lock(&handle->api_mutex);
    if (handle->ctx) h3_free(handle->ctx);
    handle->ctx = NULL;
    pthread_mutex_unlock(&handle->api_mutex);
    pthread_mutex_destroy(&handle->api_mutex);
    pthread_mutex_destroy(&handle->error_mutex);
    free(handle);
}

const char *h3rt_last_error(h3rt_handle *handle) {
    if (!handle) return "invalid handle";
    pthread_mutex_lock(&handle->error_mutex);
    snprintf(h3rt_tls_error, sizeof(h3rt_tls_error), "%s", handle->error);
    pthread_mutex_unlock(&handle->error_mutex);
    return h3rt_tls_error;
}

char *h3rt_device_info(h3rt_handle *handle) {
    if (!handle || !handle->ctx) return NULL;
    const h3_device_info *device = h3_device(handle->ctx);
    if (!device) return NULL;
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;
    cJSON_AddStringToObject(json, "name", device->name);
    cJSON_AddStringToObject(json, "architecture", device->architecture);
    cJSON_AddNumberToObject(json, "physical_memory",
                            (double)device->physical_memory);
    cJSON_AddNumberToObject(json, "recommended_working_set",
                            (double)device->recommended_working_set);
    cJSON_AddNumberToObject(json, "max_buffer_length",
                            (double)device->max_buffer_length);
    cJSON_AddNumberToObject(json, "compute_capability",
                            device->apple_gpu_family);
    cJSON_AddNumberToObject(json, "build_arch", device->build_arch);
    cJSON_AddBoolToObject(json, "arch_matches_build",
                          device->build_arch == 0 ||
                          device->build_arch == device->apple_gpu_family);
    cJSON_AddBoolToObject(json, "unified_memory", device->unified_memory);
    cJSON_AddBoolToObject(json, "metal4_or_blackwell", device->metal4);
    cJSON_AddStringToObject(json, "model_dir", handle->model_dir);
    return h3rt_json_to_string(json);
}

char *h3rt_model_info(h3rt_handle *handle) {
    if (!handle || !handle->ctx) return NULL;
    const h3_model_info *model = h3_model(handle->ctx);
    if (!model) return NULL;
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;
    const char *names[] = {"text_encoder", "fl2va_transformer",
                           "ref2va_transformer", "video_vae", "audio_vae"};
    const h3_component_info *parts[] = {
        &model->text_encoder, &model->fl2va_transformer,
        &model->ref2va_transformer, &model->video_vae, &model->audio_vae};
    for (size_t i = 0; i < 5; i++) {
        cJSON *part = cJSON_CreateObject();
        if (!part) { cJSON_Delete(json); return NULL; }
        cJSON_AddNumberToObject(part, "bytes", (double)parts[i]->bytes);
        cJSON_AddNumberToObject(part, "tensor_bytes",
                                (double)parts[i]->tensor_bytes);
        cJSON_AddNumberToObject(part, "files", (double)parts[i]->files);
        cJSON_AddNumberToObject(part, "tensors", (double)parts[i]->tensors);
        cJSON_AddItemToObject(json, names[i], part);
    }
    cJSON_AddStringToObject(json, "model_dir", handle->model_dir);
    cJSON_AddBoolToObject(json, "ref2va_available",
                          model->ref2va_transformer.files > 0);
    return h3rt_json_to_string(json);
}

char *h3rt_cache_info(h3rt_handle *handle) {
    if (!handle || !handle->ctx) return NULL;
    h3_cache_info info;
    memset(&info, 0, sizeof(info));
    h3_cache_get_info(handle->ctx, &info);
    cJSON *json = cJSON_CreateObject();
    if (!json) return NULL;
    cJSON_AddNumberToObject(json, "embedding_entries",
                            (double)info.embedding_entries);
    cJSON_AddNumberToObject(json, "embedding_bytes",
                            (double)info.embedding_bytes);
    cJSON_AddBoolToObject(json, "prepared_dit", info.prepared_dit);
    cJSON_AddBoolToObject(json, "video_decoder", info.video_decoder);
    cJSON_AddNumberToObject(json, "dit_loads", (double)info.dit_loads);
    cJSON_AddStringToObject(json, "text_cache_dir", handle->text_cache_dir);
    return h3rt_json_to_string(json);
}

int h3rt_cache_clear(h3rt_handle *handle) {
    if (!handle || !handle->ctx) return H3RT_ERR_INVALID_ARGUMENT;
    h3_cache_clear(handle->ctx);
    return H3RT_OK;
}

int h3rt_set_text_cache_dir(h3rt_handle *handle, const char *dir) {
    if (!handle || !handle->ctx || !dir || !*dir)
        return H3RT_ERR_INVALID_ARGUMENT;
    if (!h3_set_text_cache_dir(handle->ctx, dir)) {
        h3rt_set_error(handle, "text cache directory '%s' is not usable", dir);
        return H3RT_ERR_INVALID_ARGUMENT;
    }
    snprintf(handle->text_cache_dir, sizeof(handle->text_cache_dir), "%s", dir);
    return H3RT_OK;
}

static int h3rt_model_supported(const char *model) {
    if (!model || !*model) return 1;
    static const char *accepted[] = {"h3", "minimax-h3", "minimax_h3",
                                     "hyperflow", "hyperflow-h3",
                                     "hyperflow_h3", "fl2va", "ref2va"};
    for (size_t i = 0; i < sizeof(accepted) / sizeof(accepted[0]); i++)
        if (!strcmp(model, accepted[i])) return 1;
    return 0;
}

/* Shared by generate and preload: the engine knobs that do not depend on a
 * prompt.  h3_params is 39 fields; keeping the mapping in one place is what
 * stops preload from warming a different layout than the first request uses. */
static void h3rt_fill_params(const cJSON *request, int default_keep_dit,
                             h3_params *out) {
    h3_params params = H3_PARAMS_DEFAULT;
    params.width = h3rt_int(request, "width", H3_DEFAULT_WIDTH);
    params.height = h3rt_int(request, "height", H3_DEFAULT_HEIGHT);
    params.frames = h3rt_int(request, "frames", H3_DEFAULT_FRAMES);
    params.steps = h3rt_int(request, "steps", H3_DEFAULT_STEPS);
    params.seed = h3rt_seed(request, "seed", 42);
    params.pdd8 = h3rt_bool(request, "pdd8", 1);
    params.denoise_reuse = h3rt_int(request, "denoise_reuse", 1);
    params.dit_layers = h3rt_int(request, "dit_layers", H3_DEFAULT_DIT_LAYERS);
    params.core_reuse = h3rt_int(request, "core_reuse", 1);
    if (params.pdd8) {
        /* The 8-step distilled release evaluates every step. */
        params.steps = 8;
        params.denoise_reuse = 1;
    }
    params.token_reduction = h3rt_bool(request, "token_reduction", 0);
    params.sol_attn = h3rt_bool(request, "sol_attn", 0);
    params.fbc = h3rt_bool(request, "fbc", 0);
    params.use_int8_row_fc2 = h3rt_bool(request, "use_int8_row_fc2", 0);
    params.use_reference_rope = h3rt_bool(request, "use_reference_rope", 0);
    params.ssd_streaming = h3rt_bool(request, "ssd_streaming", 0);
    params.render_width = h3rt_int(request, "render_width", 0);
    params.render_height = h3rt_int(request, "render_height", 0);
    params.preview_denoise = h3rt_bool(request, "preview_denoise", 0);
    params.keep_dit = h3rt_bool(request, "keep_dit", default_keep_dit);
    params.first_frame = h3rt_str(request, "first_frame", NULL);
    params.last_frame = h3rt_str(request, "last_frame", NULL);
    const char *size_mode = h3rt_str(request, "reference_image_size", NULL);
    params.reference_image_size = size_mode && !strcmp(size_mode, "max") ?
                                  H3_REFERENCE_IMAGE_MAX :
                                  H3_REFERENCE_IMAGE_MATCH;
    *out = params;
}

int h3rt_preload(h3rt_handle *handle, const char *request_json,
                 char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!handle || !handle->ctx) return H3RT_ERR_INVALID_ARGUMENT;

    cJSON *request = NULL;
    if (request_json && *request_json) {
        request = cJSON_Parse(request_json);
        if (!request || !cJSON_IsObject(request)) {
            cJSON_Delete(request);
            h3rt_set_error(handle, "request must be a JSON object");
            if (error && error_size)
                snprintf(error, error_size, "request must be a JSON object");
            return H3RT_ERR_INVALID_JSON;
        }
    }
    h3_params params;
    h3rt_fill_params(request, handle->keep_dit, &params);
    cJSON_Delete(request);

    char detail[512] = "";
    pthread_mutex_lock(&handle->api_mutex);
    if (handle->busy) {
        pthread_mutex_unlock(&handle->api_mutex);
        h3rt_set_error(handle, "a generation is already running on this handle");
        if (error && error_size)
            snprintf(error, error_size, "a generation is already running");
        return H3RT_ERR_BUSY;
    }
    handle->busy = 1;
    int ok = h3_dit_preload(handle->ctx, &params, detail, sizeof(detail));
    handle->busy = 0;
    pthread_mutex_unlock(&handle->api_mutex);

    if (!ok) {
        if (!detail[0]) {
            const char *message = h3_last_error(handle->ctx);
            snprintf(detail, sizeof(detail), "%s",
                     message && *message ? message : "preload failed");
        }
        h3rt_set_error(handle, "%s", detail);
        if (error && error_size) snprintf(error, error_size, "%s", detail);
        return H3RT_ERR_GENERATE;
    }
    return H3RT_OK;
}

int h3rt_generate(h3rt_handle *handle, const char *request_json,
                  const h3rt_callbacks *callbacks, char **out_json) {
    if (out_json) *out_json = NULL;
    if (!handle || !handle->ctx || !request_json || !*request_json || !out_json)
        return H3RT_ERR_INVALID_ARGUMENT;

    cJSON *request = cJSON_Parse(request_json);
    if (!request || !cJSON_IsObject(request)) {
        cJSON_Delete(request);
        h3rt_set_error(handle, "request must be a JSON object");
        return H3RT_ERR_INVALID_JSON;
    }
    const char *model = h3rt_str(request, "model", "h3");
    if (!h3rt_model_supported(model)) {
        h3rt_set_error(handle,
                       "model \"%s\" is not supported by h3rt; this runtime "
                       "serves the MiniMax-H3 / HyperFlow H3 Flash pipeline",
                       model);
        cJSON_Delete(request);
        return H3RT_ERR_UNSUPPORTED;
    }
    const char *prompt = h3rt_str(request, "prompt", NULL);
    if (!prompt || !*prompt) {
        h3rt_set_error(handle, "prompt is required");
        cJSON_Delete(request);
        return H3RT_ERR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&handle->api_mutex);
    if (handle->busy) {
        pthread_mutex_unlock(&handle->api_mutex);
        h3rt_set_error(handle, "a generation is already running on this handle");
        cJSON_Delete(request);
        return H3RT_ERR_BUSY;
    }
    handle->busy = 1;
    __atomic_store_n(&handle->cancel, 0, __ATOMIC_SEQ_CST);

    int status = H3RT_OK;
    h3rt_generate_task task;
    memset(&task, 0, sizeof(task));
    task.handle = handle;
    task.prompt = prompt;
    if (callbacks) task.user = *callbacks;

    h3_params params;
    h3rt_fill_params(request, handle->keep_dit, &params);

    h3_reference refs[H3RT_MAX_REFERENCES];
    size_t ref_count = 0;
    char ref_error[256] = "";
    if (!h3rt_collect_references(request, refs, &ref_count,
                                 H3RT_MAX_REFERENCES, ref_error,
                                 sizeof(ref_error))) {
        h3rt_set_error(handle, "%s", ref_error);
        status = H3RT_ERR_INVALID_ARGUMENT;
    }

    if (status == H3RT_OK) {
        params.references = ref_count ? refs : NULL;
        params.reference_count = ref_count;

        const char *output = h3rt_str(request, "output", NULL);
        if (!output || !*output) output = h3rt_str(request, "output_path", NULL);
        if (output && *output) {
            snprintf(task.output_path, sizeof(task.output_path), "%s", output);
        } else {
            snprintf(task.output_path, sizeof(task.output_path),
                     "%s/h3rt-%ld-%llu.mp4", h3_temp_dir(), (long)getpid(),
                     ++handle->job_sequence);
        }
        params.output_path = task.output_path;
        params.on_progress = h3rt_progress_bridge;
        params.on_frame = h3rt_frame_bridge;
        params.callback_opaque = &task;
        task.params = params;

        if (!h3rt_run_worker(&task)) {
            h3rt_set_error(handle, "cannot start the generation thread");
            status = H3RT_ERR_THREAD;
        } else if (task.cancelled ||
                   __atomic_load_n(&handle->cancel, __ATOMIC_SEQ_CST)) {
            h3rt_set_error(handle, "generation cancelled by the caller");
            status = H3RT_ERR_CANCELLED;
        } else if (!task.ok) {
            h3rt_set_error(handle, "%s", task.error[0] ? task.error :
                           "generation failed");
            status = H3RT_ERR_GENERATE;
        } else {
            cJSON *result = cJSON_CreateObject();
            if (!result) {
                h3rt_set_error(handle, "out of memory building the result");
                status = H3RT_ERR_OUT_OF_MEMORY;
            } else {
                char seed_text[32];
                snprintf(seed_text, sizeof(seed_text), "%llu",
                         (unsigned long long)task.seed);
                cJSON_AddStringToObject(result, "status", "ok");
                cJSON_AddStringToObject(result, "model", "h3");
                cJSON_AddStringToObject(result, "engine", H3_VERSION);
                cJSON_AddStringToObject(result, "output", task.output_path);
                cJSON_AddStringToObject(result, "content_type", "video/mp4");
                cJSON_AddNumberToObject(result, "width", task.width);
                cJSON_AddNumberToObject(result, "height", task.height);
                cJSON_AddNumberToObject(result, "frames", task.frames);
                cJSON_AddNumberToObject(result, "fps", task.fps);
                cJSON_AddNumberToObject(result, "sample_rate", task.sample_rate);
                cJSON_AddNumberToObject(result, "seed", (double)task.seed);
                cJSON_AddStringToObject(result, "seed_string", seed_text);
                cJSON_AddNumberToObject(result, "elapsed_ms", task.elapsed_ms);
                cJSON_AddNumberToObject(result, "frames_per_second",
                                        task.elapsed_ms > 0.0
                                            ? (double)task.frames * 1000.0 /
                                                  task.elapsed_ms
                                            : 0.0);
                cJSON_AddNumberToObject(result, "bytes",
                                        h3rt_file_size(task.output_path));
                cJSON_AddBoolToObject(result, "pdd8", params.pdd8);
                cJSON_AddBoolToObject(result, "cache_dir_configured",
                                      handle->text_cache_dir[0] != '\0');
                cJSON *phases = cJSON_AddArrayToObject(result, "phases");
                if (phases) {
                    for (int i = 0; i < task.phase_count; i++) {
                        cJSON *mark = cJSON_CreateObject();
                        if (!mark) break;
                        cJSON_AddStringToObject(mark, "phase", task.phases[i].phase);
                        cJSON_AddNumberToObject(mark, "at_ms",
                                                task.phases[i].at_ms);
                        cJSON_AddItemToArray(phases, mark);
                    }
                }
                *out_json = h3rt_json_to_string(result);
                if (!*out_json) {
                    h3rt_set_error(handle, "out of memory printing the result");
                    status = H3RT_ERR_OUT_OF_MEMORY;
                }
            }
        }
    }

    handle->busy = 0;
    pthread_mutex_unlock(&handle->api_mutex);
    cJSON_Delete(request);
    return status;
}

int h3rt_cancel(h3rt_handle *handle) {
    if (!handle) return H3RT_ERR_INVALID_ARGUMENT;
    __atomic_store_n(&handle->cancel, 1, __ATOMIC_SEQ_CST);
    return H3RT_OK;
}

int h3rt_is_cancelled(h3rt_handle *handle) {
    if (!handle) return 0;
    return __atomic_load_n(&handle->cancel, __ATOMIC_SEQ_CST) ? 1 : 0;
}
