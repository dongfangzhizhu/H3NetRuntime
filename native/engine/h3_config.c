#include "h3_config.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "third_party/cJSON.h"

enum { H3_CONFIG_FILE_MAX = 1 << 20 };

/* One row per key the engine understands. Adding a path means adding a field to
 * h3_config and a row here; the loader, the test table and h3_config_field()
 * all follow from that. */
typedef struct {
    const char *key;
    size_t offset;
} config_field;

#define CONFIG_FIELD(name) {#name, offsetof(h3_config, name)}

static const config_field config_fields[] = {
    CONFIG_FIELD(h3_dir),
    CONFIG_FIELD(zimage_dir),
    CONFIG_FIELD(zimage_nunchaku_dir),
    CONFIG_FIELD(wan22_dir),
    CONFIG_FIELD(qwen21_dir),
    CONFIG_FIELD(qwen21_lora_dir),
    CONFIG_FIELD(text_cache_dir),
    CONFIG_FIELD(wan22_text_cache_dir),
    CONFIG_FIELD(qwen21_text_cache_dir),
};

enum { CONFIG_FIELD_COUNT =
           (int)(sizeof(config_fields) / sizeof(config_fields[0])) };

const char *h3_home_dir(void) {
    static const char *resolved;
    if (resolved) return resolved;
    const char *home = getenv("HOME");
#if defined(_WIN32)
    if (!home || !*home) home = getenv("USERPROFILE");
#endif
    resolved = home && *home ? home : "";
    return resolved;
}

static char *config_slot(h3_config *config, int index) {
    return (char *)config + config_fields[index].offset;
}

static int find_field(const char *key) {
    for (int index = 0; index < CONFIG_FIELD_COUNT; index++)
        if (!strcmp(config_fields[index].key, key)) return index;
    return -1;
}

const char *h3_program_dir(void) {
    static char directory[H3_CONFIG_PATH_MAX];
    static int state; /* 0 = untried, 1 = found, -1 = unavailable */
    if (state) return state > 0 ? directory : "";
    state = -1;
#if defined(_WIN32)
    wchar_t wide[H3_CONFIG_PATH_MAX];
    DWORD units = GetModuleFileNameW(NULL, wide,
                                     (DWORD)(sizeof(wide) / sizeof(wide[0])));
    if (units > 0 && units < sizeof(wide) / sizeof(wide[0])) {
        int bytes = WideCharToMultiByte(CP_UTF8, 0, wide, (int)units, directory,
                                        (int)sizeof(directory) - 1, NULL, NULL);
        if (bytes > 0) {
            directory[bytes] = '\0';
            char *slash = strrchr(directory, '\\');
            char *forward = strrchr(directory, '/');
            if (forward && (!slash || forward > slash)) slash = forward;
            if (slash && slash != directory) {
                *slash = '\0';
                state = 1;
                /* Forward slashes on the way out.  Windows accepts either in
                 * every path API the engine uses, but this string is what
                 * `h3 paths` prints as h3_config_source, and a documented
                 * machine-readable line should not change shape per platform --
                 * `D:\...\bin/h3.config.json` is a mixed separator no consumer
                 * wants to special-case. */
                for (char *cursor = directory; *cursor; cursor++)
                    if (*cursor == '\\') *cursor = '/';
            }
        }
    }
#else
    /* /proc/self/exe is the only portable-enough answer on Linux: argv[0] is
     * whatever the launcher typed, which may be a bare name on PATH. */
    char link[H3_CONFIG_PATH_MAX * 2];
    ssize_t length = readlink("/proc/self/exe", link, sizeof(link) - 1);
    if (length > 0) {
        link[length] = '\0';
        char *slash = strrchr(link, '/');
        if (slash && slash != link) {
            *slash = '\0';
            if (snprintf(directory, sizeof(directory), "%s", link) > 0) state = 1;
        }
    }
#endif
    return state > 0 ? directory : "";
}

int h3_config_expand_home(const char *path, char *out, size_t capacity) {
    if (!path || !out || !capacity) return 0;
    if (path[0] == '~' &&
        (path[1] == '\0' || path[1] == '/' || path[1] == '\\')) {
        const char *home = h3_home_dir();
        if (*home) {
            int written = snprintf(out, capacity, "%s%s", home, path + 1);
            return written > 0 && (size_t)written < capacity;
        }
    }
    int written = snprintf(out, capacity, "%s", path);
    return written > 0 && (size_t)written < capacity;
}

int h3_config_search_paths(char paths[][H3_CONFIG_PATH_MAX], int max) {
    if (!paths || max <= 0) return 0;
    int count = 0;
    /* The executable's own directory first: that is the one location a
     * double-click, a shortcut and a scheduled task agree on, and it is where a
     * dist/ directory keeps its config. */
    const char *program = h3_program_dir();
    if (*program && count < max) {
        snprintf(paths[count], H3_CONFIG_PATH_MAX, "%s/h3.config.json", program);
        count++;
    }
    if (count < max) {
        snprintf(paths[count], H3_CONFIG_PATH_MAX, "%s", "h3.config.json");
        count++;
    }
    const char *home = h3_home_dir();
    if (*home && count < max) {
        snprintf(paths[count], H3_CONFIG_PATH_MAX, "%s/.config/h3/config.json",
                 home);
        count++;
    }
    return count;
}

static void report(char *error, size_t error_size, const char *origin,
                   const char *key, const char *detail) {
    if (!error || !error_size) return;
    if (key)
        snprintf(error, error_size, "%s: \"%s\": %s", origin, key, detail);
    else
        snprintf(error, error_size, "%s: %s", origin, detail);
}

/* Applies one JSON object. Top-level objects contain the platform sections,
 * which are objects rather than paths and are skipped here; the section itself
 * is applied by the caller. */
static int apply_object(const cJSON *object, h3_config *config,
                        const char *origin, int top_level, char *error,
                        size_t error_size) {
    const cJSON *item = NULL;
    cJSON_ArrayForEach(item, object) {
        const char *key = item->string;
        if (!key) continue;
        if (top_level &&
            (!strcmp(key, "windows") || !strcmp(key, "linux"))) {
            if (!cJSON_IsObject(item)) {
                report(error, error_size, origin, key,
                       "must be an object of paths");
                return -1;
            }
            continue;
        }
        /* "_comment" and "$schema" are documentation, not settings. */
        if (key[0] == '_' || key[0] == '$') continue;
        int index = find_field(key);
        if (index < 0) {
            fprintf(stderr,
                    "h3: %s: unknown key \"%s\" (ignored; known keys: "
                    "h3_dir, zimage_dir, zimage_nunchaku_dir, wan22_dir, "
                    "qwen21_dir, qwen21_lora_dir, text_cache_dir, "
                    "wan22_text_cache_dir, qwen21_text_cache_dir)\n",
                    origin, key);
            continue;
        }
        if (!cJSON_IsString(item) || !item->valuestring) {
            report(error, error_size, origin, key, "must be a string path");
            return -1;
        }
        /* An empty string means "not set here" so a platform section can leave
         * a shared default alone without repeating it. */
        if (!*item->valuestring) continue;
        if (!h3_config_expand_home(item->valuestring,
                                   config_slot(config, index),
                         H3_CONFIG_PATH_MAX)) {
            report(error, error_size, origin, key, "path is too long");
            return -1;
        }
    }
    return 0;
}

static const char *platform_section(void) {
#if defined(_WIN32)
    return "windows";
#else
    return "linux";
#endif
}

static int file_exists(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    fclose(file);
    return 1;
}

static char *read_file(const char *path, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        report(error, error_size, path, NULL, "cannot be opened");
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        report(error, error_size, path, NULL, "cannot be sized");
        return NULL;
    }
    long size = ftell(file);
    if (size < 0 || size > H3_CONFIG_FILE_MAX) {
        fclose(file);
        report(error, error_size, path, NULL, "is empty or larger than 1 MiB");
        return NULL;
    }
    rewind(file);
    char *text = malloc((size_t)size + 1);
    if (!text) {
        fclose(file);
        report(error, error_size, path, NULL, "out of memory");
        return NULL;
    }
    size_t read = fread(text, 1, (size_t)size, file);
    fclose(file);
    if (read != (size_t)size) {
        free(text);
        report(error, error_size, path, NULL, "cannot be read");
        return NULL;
    }
    text[read] = '\0';
    return text;
}

int h3_config_load(const char *explicit_path, h3_config *config, char *error,
                   size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!config) {
        report(error, error_size, "config", NULL, "no destination");
        return -1;
    }
    memset(config, 0, sizeof(*config));

    char path[H3_CONFIG_PATH_MAX * 2];
    if (explicit_path && *explicit_path) {
        if (!h3_config_expand_home(explicit_path, path, sizeof(path))) {
            report(error, error_size, "config", NULL, "path is too long");
            return -1;
        }
        if (!file_exists(path)) {
            /* An explicitly named file that is not there is a mistake worth
             * stopping for; a missing default location is normal. */
            report(error, error_size, path, NULL,
                   "not found (given by --config or H3_CONFIG)");
            return -1;
        }
    } else {
        char candidates[3][H3_CONFIG_PATH_MAX];
        int count = h3_config_search_paths(candidates, 3);
        path[0] = '\0';
        for (int index = 0; index < count; index++) {
            if (!file_exists(candidates[index])) continue;
            snprintf(path, sizeof(path), "%s", candidates[index]);
            break;
        }
        if (!path[0]) return 0;
    }

    char *text = read_file(path, error, error_size);
    if (!text) return -1;
    cJSON *root = cJSON_Parse(text);
    free(text);
    if (!root) {
        report(error, error_size, path, NULL, "is not valid JSON");
        return -1;
    }
    if (!cJSON_IsObject(root)) {
        cJSON_Delete(root);
        report(error, error_size, path, NULL, "must contain a JSON object");
        return -1;
    }
    snprintf(config->source, sizeof(config->source), "%s", path);

    int status = apply_object(root, config, path, 1, error, error_size);
    if (status == 0) {
        const cJSON *section = cJSON_GetObjectItemCaseSensitive(
            root, platform_section());
        if (section &&
            apply_object(section, config, path, 0, error, error_size) != 0)
            status = -1;
    }
    cJSON_Delete(root);
    if (status != 0) {
        config->source[0] = '\0';
        return -1;
    }
    return 1;
}

const char *h3_config_field(const h3_config *config, const char *key) {
    if (!config || !key) return "";
    int index = find_field(key);
    if (index < 0) return "";
    return (const char *)config + config_fields[index].offset;
}

int h3_config_set(h3_config *config, const char *key, const char *value) {
    if (!config || !key) return 0;
    int index = find_field(key);
    if (index < 0) return 0;
    char *slot = (char *)config + config_fields[index].offset;
    const char *text = value ? value : "";
    if (strlen(text) >= H3_CONFIG_PATH_MAX) return 0;
    snprintf(slot, H3_CONFIG_PATH_MAX, "%s", text);
    return 1;
}

/* A shell assignment the caller can `eval`.  Single quotes keep a path with
 * spaces intact; an embedded quote is the one case that needs the closing /
 * escaped / reopening dance, which is the same idiom everywhere. */
static void print_assignment(FILE *out, const char *key, const char *value) {
    fputs(key, out);
    fputc('=', out);
    fputc('\'', out);
    for (const char *cursor = value; *cursor; cursor++) {
        if (*cursor == '\'') fputs("'\\''", out);
        else fputc(*cursor, out);
    }
    fputs("'\n", out);
}

void h3_config_print(const h3_config *config, FILE *out) {
    if (!config || !out) return;
    /* The file the values came from, so a script can say which one it read
     * instead of guessing why a path is not the one it expected. */
    print_assignment(out, "h3_config_source",
                     config->source[0] ? config->source : "");
    for (size_t index = 0; index < CONFIG_FIELD_COUNT; index++) {
        const char *value = (const char *)config + config_fields[index].offset;
        /* Unset keys are omitted rather than printed empty: "not configured"
         * and "configured as the empty string" mean the same thing here, and
         * an absent line is what a `: "${zimage_dir:?}"` guard wants. */
        if (!value[0]) continue;
        print_assignment(out, config_fields[index].key, value);
    }
}
