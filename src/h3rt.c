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
#include "h3rt_links.h"
#include "cJSON.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include <dirent.h>
#include <sys/stat.h>

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
#include <unistd.h> /* getpid() for unique temp output names */
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
    char model_dir[H3RT_PATH_CAPACITY];   /* pack the caller asked for */
    char pack_dir[H3RT_PATH_CAPACITY];    /* pack the engine actually reads */
    char text_cache_dir[H3RT_PATH_CAPACITY];
    int composed;                         /* 1 when pack_dir is a link pack */
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

/* Bytes in a file.  ftell() returns a 32-bit long on Windows, which truncates
 * a 15.8 GB checkpoint, so use the 64-bit variants there. */
static double h3rt_file_size(const char *path) {
#if defined(_WIN32)
    FILE *file = fopen(path, "rb");
    if (!file) return 0.0;
    double size = 0.0;
    if (_fseeki64(file, 0, SEEK_END) == 0) {
        long long position = _ftelli64(file);
        if (position > 0) size = (double)position;
    }
    fclose(file);
    return size;
#else
    struct stat info;
    if (stat(path, &info) != 0 || info.st_size <= 0) return 0.0;
    return (double)info.st_size;
#endif
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
            /* One line per phase transition: a run takes minutes, and this is
             * what makes "it is still in the text encoder" visible in a
             * console without a callback. */
            const char *quiet = getenv("H3RT_QUIET");
            if (!quiet || !*quiet || !strcmp(quiet, "0"))
                fprintf(stderr, "[h3rt] %6.1fs  %s (%d/%d)\n",
                        (double)(h3rt_now_ms() - task->started_ms) / 1000.0,
                        name, completed, total);
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

/* ------------------------------------------------------------ model paths */

/* Every weight component the engine resolves under the pack root.  A caller
 * that keeps its checkpoints in separate directories (the normal case after
 * downloading a DiT, a text encoder and two VAEs from different releases)
 * names them in options.models and h3rt links them into that layout.
 *
 * `accepts_parent` means the slot may also be given one level up: the video
 * VAE is read from FL2VA/video_vae/source, but users naturally point at the
 * FL2VA/video_vae directory that contains it. */
typedef struct {
    const char *key;         /* options.models key */
    const char *slot;        /* path inside the pack root */
    int is_directory;        /* 0 = a single file */
    int accepts_parent;      /* the slot may be given one level up */
    int required;            /* the engine cannot start without it */
    const char *hint;        /* what to tell the user when it is missing */
} h3rt_slot;

static const h3rt_slot h3rt_slots[] = {
    {"text_encoder", "FL2VA/text_encoder", 1, 0, 1, "text encoder directory"},
    {"transformer", "FL2VA/transformer", 1, 0, 1, "FL2VA DiT directory"},
    {"video_vae", "FL2VA/video_vae/source", 1, 1, 1, "video VAE weights"},
    {"audio_vae", "FL2VA/audio_vae", 1, 0, 1, "audio VAE weights"},
    {"tokenizer", "FL2VA/tokenizer", 1, 0, 1, "tokenizer files"},
    {"pdd_lora", "FL2VA/pdd_lora.safetensors", 0, 0, 0, NULL},
    {"ref2va_transformer", "Ref2VA/transformer", 1, 0, 0, NULL},
    {"ref2va_text_encoder", "Ref2VA/text_encoder", 1, 0, 0, NULL},
    {"ref2va_video_vae", "Ref2VA/video_vae/source", 1, 1, 0, NULL},
    {"ref2va_audio_vae", "Ref2VA/audio_vae", 1, 0, 0, NULL},
    {"ref2va_tokenizer", "Ref2VA/tokenizer", 1, 0, 0, NULL}
};
#define H3RT_SLOT_COUNT (sizeof(h3rt_slots) / sizeof(h3rt_slots[0]))

typedef struct {
    const char *target;                    /* resolved source, NULL when absent */
    char storage[H3RT_PATH_CAPACITY];
} h3rt_resolved_slot;

static void h3rt_join_path(char *out, size_t size, const char *left,
                           const char *right) {
    if (!left || !*left) {
        snprintf(out, size, "%s", right ? right : "");
        return;
    }
    size_t length = strlen(left);
    int needs_separator = left[length - 1] != '/' && left[length - 1] != '\\';
    snprintf(out, size, "%s%s%s", left, needs_separator ? "/" : "",
             right ? right : "");
}

static int h3rt_same_path(const char *left, const char *right) {
    if (!left || !right) return 0;
    size_t a = strlen(left), b = strlen(right);
    while (a > 0 && (left[a - 1] == '/' || left[a - 1] == '\\')) a--;
    while (b > 0 && (right[b - 1] == '/' || right[b - 1] == '\\')) b--;
    if (a != b) return 0;
    for (size_t i = 0; i < a; i++) {
        char x = left[i] == '\\' ? '/' : left[i];
        char y = right[i] == '\\' ? '/' : right[i];
#if defined(_WIN32)
        if (x >= 'A' && x <= 'Z') x = (char)(x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z') y = (char)(y - 'A' + 'a');
#endif
        if (x != y) return 0;
    }
    return 1;
}

/* FNV-1a over the resolved layout: the link pack is content-addressed by the
 * paths it was built from, so a second process (or a second run) reuses the
 * same directory instead of accumulating one per start. */
static unsigned long long h3rt_fnv1a(const char *text,
                                     unsigned long long hash) {
    const unsigned char *cursor = (const unsigned char *)text;
    for (; cursor && *cursor; cursor++) {
        hash ^= (unsigned long long)*cursor;
        hash *= 1099511628211ull;
    }
    hash ^= 0x1full;
    hash *= 1099511628211ull;
    return hash;
}

/* Links every entry of `source_dir` that `target_dir` does not already have.
 * Used when a component is given as a bare .safetensors: the weights come from
 * the caller, the small sidecars (config.json, pdd_heads.safetensors) come from
 * the pack root. */
static int h3rt_fill_from_directory(const char *source_dir,
                                    const char *target_dir, char *error,
                                    size_t error_size) {
    if (!h3rt_is_directory(source_dir)) return 1;
    DIR *stream = opendir(source_dir);
    if (!stream) return 1; /* nothing to copy is not a failure */
    int ok = 1;
    struct dirent *entry;
    while (ok && (entry = readdir(stream))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char source[H3RT_PATH_CAPACITY];
        char target[H3RT_PATH_CAPACITY];
        h3rt_join_path(source, sizeof(source), source_dir, entry->d_name);
        h3rt_join_path(target, sizeof(target), target_dir, entry->d_name);
        if (h3rt_path_exists(target)) continue;
        if (!h3rt_link_path(target, source, h3rt_is_directory(source), error,
                            error_size))
            ok = 0;
    }
    closedir(stream);
    return ok;
}

/* Case-insensitive "does text end with suffix", without strcasecmp (which the
 * Win32 compatibility layer does not expose). */
static int h3rt_ends_with_ci(const char *text, const char *suffix) {
    size_t text_len = strlen(text);
    size_t suffix_len = strlen(suffix);
    if (suffix_len > text_len) return 0;
    const char *tail = text + (text_len - suffix_len);
    for (size_t i = 0; i < suffix_len; i++) {
        char a = tail[i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
        if (a != b) return 0;
    }
    return 1;
}

static void h3rt_parent_dir(char *out, size_t size, const char *path) {
    snprintf(out, size, "%s", path);
    char *cut = strrchr(out, '/');
#if defined(_WIN32)
    char *backslash = strrchr(out, '\\');
    if (backslash && (!cut || backslash > cut)) cut = backslash;
#endif
    if (cut) *cut = '\0';
}

/* Copies the small metadata files that sit next to a bare weight file into the
 * slot, so pointing "models" at one downloaded .safetensors is enough: the
 * transformer slot needs config.json and may want pdd_heads.safetensors, and
 * both normally live in the same directory as the weights.  Big shards are
 * never pulled in, only the names the engine actually reads. */
static int h3rt_link_sidecars(const char *source_dir, const char *target_dir,
                              char *error, size_t error_size) {
    if (!h3rt_is_directory(source_dir)) return 1;
    DIR *stream = opendir(source_dir);
    if (!stream) return 1;
    int ok = 1;
    struct dirent *entry;
    while (ok && (entry = readdir(stream))) {
        const char *name = entry->d_name;
        if (!strcmp(name, ".") || !strcmp(name, "..")) continue;
        if (!h3rt_ends_with_ci(name, ".json") &&
            !h3rt_ends_with_ci(name, "pdd_heads.safetensors"))
            continue;
        char source[H3RT_PATH_CAPACITY];
        char target[H3RT_PATH_CAPACITY];
        h3rt_join_path(source, sizeof(source), source_dir, name);
        h3rt_join_path(target, sizeof(target), target_dir, name);
        if (h3rt_path_exists(target)) continue;
        if (h3rt_is_directory(source)) continue;
        if (!h3rt_link_path(target, source, 0, error, error_size)) ok = 0;
    }
    closedir(stream);
    return ok;
}

/* Human-readable component report for the console.
 *
 * Every "why is it loading the wrong thing" question starts with the resolved
 * paths, and they are only knowable inside the engine: the pack root, the link
 * pack the composer produced, and which slot each weight actually came from.
 * Printed unconditionally (stderr, so a JSON stdout stays clean); H3RT_QUIET=1
 * silences it. */
static void h3rt_report_paths(const char *model_dir, const char *pack_dir,
                              int composed) {
    const char *quiet = getenv("H3RT_QUIET");
    if (quiet && *quiet && strcmp(quiet, "0") != 0) return;
    fprintf(stderr, "[h3rt] model_dir : %s\n",
            model_dir && *model_dir ? model_dir : "(none)");
    fprintf(stderr, "[h3rt] pack_dir  : %s%s\n", pack_dir,
            composed ? "  (composed link pack)" : "");
    for (size_t i = 0; i < H3RT_SLOT_COUNT; i++) {
        char path[H3RT_PATH_CAPACITY];
        h3rt_join_path(path, sizeof(path), pack_dir, h3rt_slots[i].slot);
        if (!h3rt_path_exists(path)) {
            if (h3rt_slots[i].required)
                fprintf(stderr, "[h3rt]   %-20s MISSING  %s\n",
                        h3rt_slots[i].key, path);
            continue;
        }
        if (h3rt_is_directory(path)) {
            unsigned files = 0;
            unsigned long long bytes = 0;
            DIR *stream = opendir(path);
            if (stream) {
                struct dirent *entry;
                while ((entry = readdir(stream))) {
                    if (!strcmp(entry->d_name, ".") ||
                        !strcmp(entry->d_name, ".."))
                        continue;
                    char child[H3RT_PATH_CAPACITY];
                    h3rt_join_path(child, sizeof(child), path, entry->d_name);
                    if (h3rt_is_directory(child)) continue;
                    files++;
                    bytes += (unsigned long long)h3rt_file_size(child);
                }
                closedir(stream);
            }
            fprintf(stderr, "[h3rt]   %-20s %s  (%u file%s, %.2f GB)\n",
                    h3rt_slots[i].key, path, files, files == 1 ? "" : "s",
                    (double)bytes / 1073741824.0);
        } else {
            fprintf(stderr, "[h3rt]   %-20s %s  (%.2f GB)\n",
                    h3rt_slots[i].key, path,
                    (double)h3rt_file_size(path) / 1073741824.0);
        }
    }
    fflush(stderr);
}

/* ------------------------------------------------------------ tokenizer */

/* The tokenizer JSON is the one input the ComfyUI model download does not
 * ship: CLIPLoaderGGUF fetches it from the HuggingFace repo, while the engine
 * needs the file on disk.  A user who only has the five released weight files
 * therefore still needs this 7 MiB text file, and making them hunt for it is
 * the single most common "it will not start" report.  So: look for it in every
 * place it can reasonably be, including next to the native library, where the
 * packaging step drops a copy.
 *
 * Returns 1 and fills `out` with the directory holding tokenizer.json. */
static int h3rt_tokenizer_in(const char *dir, char *out, size_t size) {
    if (!dir || !*dir) return 0;
    char probe[H3RT_PATH_CAPACITY];
    h3rt_join_path(probe, sizeof(probe), dir, "tokenizer.json");
    if (!h3rt_path_exists(probe)) return 0;
    snprintf(out, size, "%s", dir);
    return 1;
}

/* Directory holding the h3rt shared library itself.  A tokenizer shipped next
 * to it (runtimes/<rid>/native/tokenizer/) is found with no configuration.
 *
 * `which`: 0 = this library, 1 = the host executable.  The address of a
 * function inside the library is not a valid HMODULE, so the module handle has
 * to come from GetModuleHandleExA(FROM_ADDRESS). */
static int h3rt_binary_dir(int which, char *out, size_t size) {
#if defined(_WIN32)
    __declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(
        void *module, char *name, unsigned long count);
    __declspec(dllimport) int __stdcall GetModuleHandleExA(
        unsigned long flags, const void *name, void **module);
    void *module = NULL;
    if (which == 0) {
        /* GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | _UNCHANGED_REFCOUNT */
        if (!GetModuleHandleExA(0x00000004ul | 0x00000002ul,
                                (const void *)&h3rt_binary_dir, &module))
            return 0;
    }
    char buffer[H3RT_PATH_CAPACITY];
    unsigned long length = GetModuleFileNameA(module, buffer,
                                             (unsigned long)sizeof(buffer));
    if (length == 0 || length >= sizeof(buffer)) return 0;
    while (length > 0) {
        char c = buffer[length - 1];
        if (c == '/' || c == '\\') {
            buffer[length - 1] = '\0';
            break;
        }
        length--;
    }
    if (!buffer[0]) return 0;
    snprintf(out, size, "%s", buffer);
    return 1;
#else
    /* /proc/self/maps lines look like
     *   7f2b4c000000-7f2b4c021000 r-xp 00000000 fd:01 1234567 /path/libh3rt.so
     * The path is the sixth whitespace-separated field. */
    const char *want = which == 0 ? "libh3rt.so" : NULL;
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) return 0;
    char line[4096];
    int found = 0;
    while (!found && fgets(line, sizeof(line), maps)) {
        char *scan = line;
        int field = 0;
        while (field < 5 && *scan) {
            while (*scan == ' ' || *scan == '\t') scan++;
            while (*scan && *scan != ' ' && *scan != '\t') scan++;
            field++;
        }
        while (*scan == ' ' || *scan == '\t') scan++;
        char *path = *scan ? scan : NULL;
        if (!path) continue;
        char *newline = strchr(path, '\n');
        if (newline) *newline = '\0';
        size_t length = strlen(path);
        if (!want) {
            /* The host executable is /proc/self/exe; handled below. */
            break;
        }
        if (length < strlen(want)) continue;
        if (strcmp(path + length - strlen(want), want) != 0) continue;
        char *slash = strrchr(path, '/');
        if (!slash) continue;
        *slash = '\0';
        snprintf(out, size, "%s", path);
        found = 1;
    }
    fclose(maps);
    if (found) return 1;
    if (which == 1) {
        char target[H3RT_PATH_CAPACITY];
        ssize_t length = readlink("/proc/self/exe", target,
                                  sizeof(target) - 1);
        if (length <= 0) return 0;
        target[length] = '\0';
        char *slash = strrchr(target, '/');
        if (!slash) return 0;
        *slash = '\0';
        snprintf(out, size, "%s", target);
        return 1;
    }
    return 0;
#endif
}

/* Search chain for tokenizer.json.  `report` (optional) receives a one-line
 * summary of what was tried, which is what the create error quotes. */
static int h3rt_find_tokenizer(const char *model_dir, char *out, size_t size,
                               char *report, size_t report_size) {
    if (report && report_size) report[0] = '\0';
    char candidate[H3RT_PATH_CAPACITY];

    /* 1. Explicit override: H3RT_TOKENIZER_DIR, either the directory itself or
     *    a path straight to tokenizer.json. */
    const char *env = getenv("H3RT_TOKENIZER_DIR");
    if (env && *env) {
        snprintf(candidate, sizeof(candidate), "%s", env);
        if (h3rt_ends_with_ci(candidate, "tokenizer.json")) {
            char parent[H3RT_PATH_CAPACITY];
            snprintf(parent, sizeof(parent), "%s", candidate);
            char *cut = strrchr(parent, '/');
            char *cut_win = strrchr(parent, '\\');
            if (cut_win && (!cut || cut_win > cut)) cut = cut_win;
            if (cut) {
                *cut = '\0';
                if (h3rt_tokenizer_in(parent, out, size)) return 1;
            }
        } else if (h3rt_tokenizer_in(candidate, out, size)) {
            return 1;
        }
        if (report && report_size)
            snprintf(report, report_size, "H3RT_TOKENIZER_DIR=%s has no "
                     "tokenizer.json; ", env);
    }

    /* 2. Layouts a user actually points at. */
    static const char *const relatives[] = {
        "tokenizer",                  /* ComfyUI models/tokenizer, or a pack root */
        "text_encoders/tokenizer",    /* ComfyUI-GGUF convention */
        "FL2VA/tokenizer",            /* an existing H3 pack */
        "../tokenizer"                /* next to the pack root */
    };
    for (size_t index = 0; index < sizeof(relatives) / sizeof(relatives[0]);
         index++) {
        if (!model_dir || !*model_dir) break;
        h3rt_join_path(candidate, sizeof(candidate), model_dir,
                       relatives[index]);
        if (h3rt_tokenizer_in(candidate, out, size)) return 1;
    }

    /* 3. Shipped next to the native library (runtimes/<rid>/native/tokenizer,
     *    dist/win-x64/tokenizer, ...), or next to the host executable, which
     *    is the same folder in a normal package layout. */
    char module[H3RT_PATH_CAPACITY] = "";
    for (int which = 0; which < 2; which++) {
        char root[H3RT_PATH_CAPACITY];
        if (!h3rt_binary_dir(which, root, sizeof(root))) continue;
        if (which == 0) snprintf(module, sizeof(module), "%s", root);
        static const char *const near_binary[] = {
            "tokenizer",
            "../tokenizer",
            "../../tokenizer",
            "../../../tokenizer",
            "../../../../tokenizer",
            "../share/h3rt/tokenizer"
        };
        for (size_t index = 0;
             index < sizeof(near_binary) / sizeof(near_binary[0]); index++) {
            h3rt_join_path(candidate, sizeof(candidate), root,
                           near_binary[index]);
            if (h3rt_tokenizer_in(candidate, out, size)) return 1;
        }
    }

    /* 4. The engine's own default model root, which is where a pre-existing
     *    H3 install keeps the tokenizer. */
    char home[H3RT_PATH_CAPACITY];
    if (h3_config_expand_home("~/h3-models/pdd8/FL2VA/tokenizer", home,
                              sizeof(home)) &&
        h3rt_tokenizer_in(home, out, size))
        return 1;

    if (report && report_size) {
        size_t used = strlen(report);
        snprintf(report + used, report_size - used,
                 "looked for tokenizer.json in <%s>/{tokenizer,"
                 "text_encoders/tokenizer,FL2VA/tokenizer}, next to %s, and in "
                 "~/h3-models/pdd8/FL2VA/tokenizer; set H3RT_TOKENIZER_DIR or "
                 "models.tokenizer to override",
                 model_dir && *model_dir ? model_dir : "model_dir",
                 module[0] ? module : "the native library");
    }
    return 0;
}

/* --------------------------------------------------- ComfyUI model layout */

/* ComfyUI's `models/` directory: diffusion_models/, text_encoders/, vae/.
 * That is where the five released files of ComfyUI_MiniMax-H3-V8 live, and it
 * is what a user points --model-dir at. */
static int h3rt_is_comfy_layout(const char *dir) {
    static const char *const parts[] = {"diffusion_models", "text_encoders", "vae"};
    for (size_t index = 0; index < sizeof(parts) / sizeof(parts[0]); index++) {
        char probe[H3RT_PATH_CAPACITY];
        h3rt_join_path(probe, sizeof(probe), dir, parts[index]);
        if (!h3rt_is_directory(probe)) return 0;
    }
    return 1;
}

/* First file in `dir` whose lower-cased name contains every needle and ends
 * with `extension` (pass "" for any). */
static int h3rt_pick_file(const char *dir, const char *const *needles,
                          int needle_count, const char *extension,
                          char *out, size_t size) {
    DIR *stream = opendir(dir);
    if (!stream) return 0;
    char best[H3RT_PATH_CAPACITY];
    best[0] = '\0';
    struct dirent *entry;
    while ((entry = readdir(stream))) {
        if (entry->d_name[0] == '.') continue;
        if (extension && *extension &&
            !h3rt_ends_with_ci(entry->d_name, extension))
            continue;
        int matches = 1;
        for (int needle = 0; needle < needle_count; needle++) {
            /* Case-insensitive substring test, without strcasestr. */
            size_t need = strlen(needles[needle]);
            size_t have = strlen(entry->d_name);
            int found = 0;
            for (size_t at = 0; at + need <= have && !found; at++) {
                size_t step = 0;
                while (step < need) {
                    char a = entry->d_name[at + step];
                    char b = needles[needle][step];
                    if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
                    if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
                    if (a != b) break;
                    step++;
                }
                if (step == need) found = 1;
            }
            if (!found) { matches = 0; break; }
        }
        if (!matches) continue;
        char candidate[H3RT_PATH_CAPACITY];
        h3rt_join_path(candidate, sizeof(candidate), dir, entry->d_name);
        if (!h3rt_is_directory(candidate)) {
            snprintf(best, sizeof(best), "%s", candidate);
            break;
        }
    }
    closedir(stream);
    if (!best[0]) return 0;
    snprintf(out, size, "%s", best);
    return 1;
}

/* Fills `models` with what the ComfyUI layout holds.  Missing optional pieces
 * are simply not added, so the composer reports exactly what is absent. */
static void h3rt_derive_comfy_models(const char *dir, cJSON *models) {
    char path[H3RT_PATH_CAPACITY];
    char diffusion[H3RT_PATH_CAPACITY];
    char encoders[H3RT_PATH_CAPACITY];
    char vae[H3RT_PATH_CAPACITY];
    h3rt_join_path(diffusion, sizeof(diffusion), dir, "diffusion_models");
    h3rt_join_path(encoders, sizeof(encoders), dir, "text_encoders");
    h3rt_join_path(vae, sizeof(vae), dir, "vae");

    static const char *const fl2va[] = {"fl2va"};
    static const char *const ref2va[] = {"ref2va"};
    static const char *const sol_vae[] = {"sol_vae"};
    static const char *const video_vae[] = {"video_vae"};
    static const char *const audio_vae[] = {"audio_vae"};
    static const char *const gguf[] = {"gguf"};
    static const char *const qwen[] = {"qwen"};

    if (h3rt_pick_file(diffusion, fl2va, 1, ".safetensors", path, sizeof(path)))
        cJSON_AddStringToObject(models, "transformer", path);
    if (h3rt_pick_file(diffusion, ref2va, 1, ".safetensors", path, sizeof(path)))
        cJSON_AddStringToObject(models, "ref2va_transformer", path);
    if (h3rt_pick_file(encoders, qwen, 1, ".gguf", path, sizeof(path)))
        cJSON_AddStringToObject(models, "text_encoder", path);
    else if (h3rt_pick_file(encoders, gguf, 0, ".gguf", path, sizeof(path)))
        cJSON_AddStringToObject(models, "text_encoder", path);
    else if (h3rt_pick_file(encoders, gguf, 0, ".safetensors", path, sizeof(path)))
        cJSON_AddStringToObject(models, "text_encoder", path);
    if (h3rt_pick_file(vae, sol_vae, 1, ".safetensors", path, sizeof(path)) ||
        h3rt_pick_file(vae, video_vae, 1, ".safetensors", path, sizeof(path)))
        cJSON_AddStringToObject(models, "video_vae", path);
    if (h3rt_pick_file(vae, audio_vae, 1, ".safetensors", path, sizeof(path)))
        cJSON_AddStringToObject(models, "audio_vae", path);
    /* The tokenizer is not part of the ComfyUI model download; search the
     * usual places (including next to the native library) instead of failing
     * with "models.tokenizer is required". */
    char tokenizer[H3RT_PATH_CAPACITY];
    if (h3rt_find_tokenizer(dir, tokenizer, sizeof(tokenizer), NULL, 0))
        cJSON_AddStringToObject(models, "tokenizer", tokenizer);
}

/* Builds `<pack>/FL2VA/...` out of the per-component paths in `models`.
 *
 * Returns 1 when `out_pack` holds a freshly resolved link pack, 0 when the
 * base directory already has the requested layout (nothing to do), and -1 with
 * a message in `error` when a component is missing or cannot be linked. */
static int h3rt_compose_pack(const char *base, const cJSON *models,
                             const char *cache_dir, char *out_pack,
                             size_t out_pack_size, char *error,
                             size_t error_size) {
    h3rt_resolved_slot *slots = (h3rt_resolved_slot *)calloc(
        H3RT_SLOT_COUNT, sizeof(*slots));
    if (!slots) {
        snprintf(error, error_size, "out of memory resolving model paths");
        return -1;
    }

    int override = 0;
    int status = 1;
    unsigned long long hash = 1469598103934665603ull;
    hash = h3rt_fnv1a(base ? base : "", hash);

    for (size_t i = 0; i < H3RT_SLOT_COUNT; i++) {
        const char *given = h3rt_str(models, h3rt_slots[i].key, NULL);
        char standard[H3RT_PATH_CAPACITY];
        standard[0] = '\0';
        if (base && *base) h3rt_join_path(standard, sizeof(standard), base,
                                          h3rt_slots[i].slot);

        if (given && *given) {
            snprintf(slots[i].storage, sizeof(slots[i].storage), "%s", given);
            /* Without a pack root there is no standard layout to match, so the
             * link pack is the only way the engine can see these components. */
            if (!standard[0] || !h3rt_same_path(given, standard)) override = 1;
        } else if (standard[0]) {
            snprintf(slots[i].storage, sizeof(slots[i].storage), "%s", standard);
        } else {
            continue;
        }

        if (!h3rt_path_exists(slots[i].storage)) {
            if (given && *given) {
                snprintf(error, error_size,
                         "models.%s points at '%s', which does not exist",
                         h3rt_slots[i].key, slots[i].storage);
                status = -1;
                break;
            }
            continue; /* the base pack simply does not have this slot */
        }
        slots[i].target = slots[i].storage;
        hash = h3rt_fnv1a(slots[i].storage, hash);
    }

    if (status == 1 && !override) {
        /* Every component is already where the engine expects it. */
        free(slots);
        return 0;
    }

    for (size_t i = 0; status == 1 && i < H3RT_SLOT_COUNT; i++) {
        if (!slots[i].target && h3rt_slots[i].required) {
            snprintf(error, error_size,
                     "models.%s is required: point it at the %s, or set "
                     "model_dir to a pack that already has FL2VA/%s",
                     h3rt_slots[i].key, h3rt_slots[i].hint,
                     strchr(h3rt_slots[i].slot, '/') + 1);
            status = -1;
        }
    }

    char pack[H3RT_PATH_CAPACITY];
    if (status == 1) {
        int written = snprintf(pack, sizeof(pack), "%s/h3rt-pack-%016llx",
                               cache_dir, hash);
        if (written < 0 || (size_t)written >= sizeof(pack)) {
            snprintf(error, error_size,
                     "pack_cache_dir is too long (%zu bytes); use a shorter "
                     "directory", strlen(cache_dir));
            status = -1;
        } else if (!h3rt_make_dirs(pack)) {
            snprintf(error, error_size,
                     "cannot create the link pack '%s'; set pack_cache_dir to "
                     "a writable directory", pack);
            status = -1;
        }
    }

    for (size_t i = 0; status == 1 && i < H3RT_SLOT_COUNT; i++) {
        if (!slots[i].target) continue;
        char link[H3RT_PATH_CAPACITY];
        const char *slot = h3rt_slots[i].slot;
        int is_directory = h3rt_slots[i].is_directory;

        /* A component given as a single weight file: the slot becomes a
         * directory holding that file as model.safetensors (or model.gguf, for
         * the Q4_K_M text encoder), plus the metadata files that live next to
         * it (config.json, pdd_heads.safetensors) and whatever the pack root
         * already had for the same slot. */
        if (is_directory && !h3rt_is_directory(slots[i].target)) {
            int is_gguf = h3rt_ends_with_ci(slots[i].target, ".gguf");
            h3rt_join_path(link, sizeof(link), pack, slot);
            if (!h3rt_make_dirs(link)) {
                snprintf(error, error_size, "cannot create the '%s' slot",
                         slot);
                status = -1;
                break;
            }
            char weight[H3RT_PATH_CAPACITY];
            h3rt_join_path(weight, sizeof(weight), link,
                           is_gguf ? "model.gguf" : "model.safetensors");
            if (!h3rt_path_exists(weight) &&
                !h3rt_link_path(weight, slots[i].target, 0, error,
                                error_size)) {
                status = -1;
                break;
            }
            /* A GGUF carries everything in one file; pulling the pack root's
             * safetensors in as well would make the reader prefer that. */
            if (is_gguf) continue;
            char weight_dir[H3RT_PATH_CAPACITY];
            h3rt_parent_dir(weight_dir, sizeof(weight_dir), slots[i].target);
            if (!h3rt_link_sidecars(weight_dir, link, error, error_size)) {
                status = -1;
                break;
            }
            if (base && *base) {
                char base_slot[H3RT_PATH_CAPACITY];
                h3rt_join_path(base_slot, sizeof(base_slot), base, slot);
                if (!h3rt_fill_from_directory(base_slot, link, error,
                                              error_size))
                    status = -1;
            }
            /* The loader treats transformer/config.json as a presence marker,
             * but the ComfyUI diffusion model is a bare .safetensors. Write a
             * minimal one so that download works without a hand-made pack. */
            if (status == 1 && strstr(slot, "transformer")) {
                char config[H3RT_PATH_CAPACITY];
                h3rt_join_path(config, sizeof(config), link, "config.json");
                if (!h3rt_path_exists(config)) {
                    FILE *file = fopen(config, "wb");
                    if (file) {
                        fputs("{\n  \"architectures\": [\"MiniMaxH3FL2VA\"],\n"
                              "  \"model_type\": \"minimax_h3_fl2va\"\n}\n",
                              file);
                        fclose(file);
                    }
                }
            }
            continue;
        }

        if (h3rt_slots[i].accepts_parent && is_directory) {
            char nested[H3RT_PATH_CAPACITY];
            h3rt_join_path(nested, sizeof(nested), slots[i].target, "source");
            if (h3rt_is_directory(nested)) {
                /* Given FL2VA/video_vae, link that whole directory. */
                char trimmed[H3RT_PATH_CAPACITY];
                snprintf(trimmed, sizeof(trimmed), "%s", slot);
                char *cut = strrchr(trimmed, '/');
                if (cut) *cut = '\0';
                slot = trimmed;
            }
        }
        h3rt_join_path(link, sizeof(link), pack, slot);
        if (h3rt_path_exists(link)) continue; /* same hash, same target */
        if (!h3rt_link_path(link, slots[i].target, is_directory, error,
                            error_size)) {
            status = -1;
            break;
        }
    }

    if (status == 1) {
        /* A manifest turns "why is it looking in AppData" into one file to
         * open; the engine never reads it. */
        cJSON *manifest = cJSON_CreateObject();
        if (manifest) {
            cJSON_AddStringToObject(manifest, "pack", pack);
            cJSON_AddStringToObject(manifest, "base", base ? base : "");
            cJSON *entries = cJSON_AddObjectToObject(manifest, "slots");
            for (size_t i = 0; entries && i < H3RT_SLOT_COUNT; i++) {
                if (slots[i].target)
                    cJSON_AddStringToObject(entries, h3rt_slots[i].key,
                                            slots[i].target);
            }
            char *text = cJSON_Print(manifest);
            cJSON_Delete(manifest);
            if (text) {
                char manifest_path[H3RT_PATH_CAPACITY];
                h3rt_join_path(manifest_path, sizeof(manifest_path), pack,
                               "h3rt-pack.json");
                FILE *file = fopen(manifest_path, "wb");
                if (file) {
                    fputs(text, file);
                    fclose(file);
                }
                free(text);
            }
        }
        snprintf(out_pack, out_pack_size, "%s", pack);
    }

    free(slots);
    return status;
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
    cJSON_AddBoolToObject(json, "per_component_model_paths", 1);
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

    const cJSON *models = h3rt_item(options, "models");
    int has_models = models && cJSON_IsObject(models) &&
                     cJSON_GetArraySize(models) > 0;

    const char *model_dir = h3rt_str(options, "model_dir", NULL);
    if (!model_dir || !*model_dir) model_dir = h3rt_str(options, "h3_dir", NULL);
    char resolved[H3RT_PATH_CAPACITY];
    resolved[0] = '\0';
    /* With per-component paths the pack root is optional: everything the
     * engine needs may come from options.models. */
    const char *source = model_dir && *model_dir ? model_dir :
                         config.h3_dir[0] ? config.h3_dir :
                         has_models ? NULL :
                         "~/h3-models/pdd8";
    if (source) {
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
    }
    snprintf(handle->model_dir, sizeof(handle->model_dir), "%s", resolved);

    /* Pointed at a ComfyUI models/ directory: derive the five components from
     * its subdirectories instead of asking the user to list them. */
    if (!has_models && options && resolved[0] && h3rt_is_comfy_layout(resolved)) {
        cJSON *derived = cJSON_CreateObject();
        if (derived) {
            h3rt_derive_comfy_models(resolved, derived);
            if (cJSON_GetArraySize(derived) > 0) {
                cJSON_AddItemToObject(options, "models", derived);
                models = cJSON_GetObjectItemCaseSensitive(options, "models");
                has_models = 1;
                fprintf(stderr, "[h3rt] ComfyUI 模型目录：%s\n", resolved);
            } else {
                cJSON_Delete(derived);
            }
        }
    }

    /* A caller that listed the weights itself usually omits the tokenizer: it
     * is not a weight file.  Fill it in from the search chain so the composer
     * never reports a missing component that the runtime could find alone. */
    if (has_models && !h3rt_str(models, "tokenizer", NULL)) {
        char tokenizer[H3RT_PATH_CAPACITY];
        char searched[512];
        if (h3rt_find_tokenizer(resolved, tokenizer, sizeof(tokenizer), searched,
                                sizeof(searched))) {
            cJSON_AddStringToObject((cJSON *)models, "tokenizer", tokenizer);
        } else {
            const char *quiet = getenv("H3RT_QUIET");
            if (!quiet || !*quiet || strcmp(quiet, "0") == 0)
                fprintf(stderr, "[h3rt] tokenizer: %s\n", searched);
        }
    }

    char pack_dir[H3RT_PATH_CAPACITY];
    snprintf(pack_dir, sizeof(pack_dir), "%s", resolved);
    if (has_models) {
        const char *cache_option = h3rt_str(options, "pack_cache_dir", NULL);
        char cache_dir[H3RT_PATH_CAPACITY];
        if (cache_option && *cache_option) {
            if (cache_option[0] == '~') {
                char expanded[H3RT_PATH_CAPACITY];
                if (h3_config_expand_home(cache_option, expanded,
                                          sizeof(expanded)))
                    snprintf(cache_dir, sizeof(cache_dir), "%s", expanded);
                else
                    snprintf(cache_dir, sizeof(cache_dir), "%s", cache_option);
            } else {
                snprintf(cache_dir, sizeof(cache_dir), "%s", cache_option);
            }
        } else {
            /* Prefer a cache on the same volume as the weights: a hard link
             * cannot cross volumes, and falling back to a copy of a 20 GB DiT
             * is not an acceptable surprise.  A read-only model share cannot
             * hold the cache, so fall back to the temp directory. */
            int placed = 0;
            if (resolved[0]) {
                snprintf(cache_dir, sizeof(cache_dir), "%s/.h3rt-packs",
                         resolved);
                placed = h3rt_make_dirs(cache_dir);
            }
            if (!placed)
                snprintf(cache_dir, sizeof(cache_dir), "%s/h3rt-packs",
                         h3_temp_dir());
        }
        char compose_error[H3RT_ERROR_CAPACITY] = "";
        int composed = h3rt_compose_pack(resolved, models, cache_dir, pack_dir,
                                         sizeof(pack_dir), compose_error,
                                         sizeof(compose_error));
        if (composed < 0) {
            h3rt_write_create_error(error, error_size, "cannot resolve the "
                                    "model components: %s", compose_error);
            cJSON_Delete(options);
            pthread_mutex_destroy(&handle->api_mutex);
            pthread_mutex_destroy(&handle->error_mutex);
            free(handle);
            return H3RT_ERR_MODEL;
        }
        handle->composed = composed;
    }
    if (!pack_dir[0]) {
        h3rt_write_create_error(error, error_size,
                                "no model_dir and no models.* paths given");
        cJSON_Delete(options);
        pthread_mutex_destroy(&handle->api_mutex);
        pthread_mutex_destroy(&handle->error_mutex);
        free(handle);
        return H3RT_ERR_MODEL;
    }
    snprintf(handle->pack_dir, sizeof(handle->pack_dir), "%s", pack_dir);
    h3rt_report_paths(handle->model_dir, handle->pack_dir, handle->composed);

    h3_ctx *ctx = h3_load_dir(handle->pack_dir);
    if (!ctx) {
        const char *message = h3_last_error(NULL);
        if (handle->composed) {
            h3rt_write_create_error(
                error, error_size,
                "cannot load the model from the composed pack '%s': %s "
                "(the link manifest is %s/h3rt-pack.json)",
                handle->pack_dir,
                message && *message ? message : "FL2VA/transformer/config.json "
                                                "not found",
                handle->pack_dir);
        } else {
            h3rt_write_create_error(error, error_size,
                                    "cannot load model pack '%s': %s",
                                    handle->pack_dir,
                                    message && *message ? message :
                                    "FL2VA/transformer/config.json not found");
        }
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
    cJSON_AddStringToObject(json, "pack_dir", handle->pack_dir);
    cJSON_AddBoolToObject(json, "composed_pack", handle->composed != 0);
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
    cJSON_AddStringToObject(json, "pack_dir", handle->pack_dir);
    cJSON_AddBoolToObject(json, "composed_pack", handle->composed != 0);
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
