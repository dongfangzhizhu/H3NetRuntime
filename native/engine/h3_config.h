/* Model paths from a file instead of from the build or the shell.
 *
 * The engine used to resolve every pack as "~/h3-models/<model>" unless a flag
 * said otherwise, which works on one machine and forces every launcher to
 * repeat four absolute paths. A config file makes the paths data: the same
 * checkout runs on Linux and Windows with a different set of directories, and
 * the flags stay available for one-off overrides.
 *
 * Precedence, highest first:
 *   1. command line (--h3-dir, --qwen21-dir, ...)
 *   2. environment (H3_TEXT_CACHE_DIR, H3_WAN22_TEXT_CACHE_DIR, ...)
 *   3. this config file: the platform section, then the top-level keys
 *   4. the built-in default
 *
 * The file is JSON. Top-level keys are defaults for every platform; a
 * "windows" or "linux" object overrides them on that platform:
 *
 *   {
 *     "text_cache_dir": "/srv/h3-cache",
 *     "linux":   { "h3_dir": "/home/spider/h3-models/pdd8" },
 *     "windows": { "h3_dir": "D:/AI/convert/minimax/h3-models/pdd8" }
 *   }
 *
 * Unknown keys are reported rather than ignored: a typo ("h3_dirr") would
 * otherwise look exactly like "the setting does not work".
 */
#ifndef H3_CONFIG_H
#define H3_CONFIG_H

#include <stddef.h>
/* h3_config_print() takes a FILE*; MSVC's headers pull stdio in transitively and
 * gcc's do not, which is how this header built for a day on one platform and
 * failed on the other.  A header names what it uses. */
#include <stdio.h>

/* Longest path a config value can hold. */
#define H3_CONFIG_PATH_MAX 1024

typedef struct {
    char h3_dir[H3_CONFIG_PATH_MAX];
    char zimage_dir[H3_CONFIG_PATH_MAX];
    /* The Nunchaku DiT repacks for the same model: a directory of
     * svdq-*.safetensors (one pack each, which is how the rank 32 and rank 256
     * releases coexist) or a single .safetensors.  They are repacks of the DiT
     * only, so the VAE and text encoder keep coming from zimage_dir.  Empty
     * means the nunchaku checkpoint tier is unavailable, and a request that
     * asks for it is refused instead of quietly running the bf16 pack. */
    char zimage_nunchaku_dir[H3_CONFIG_PATH_MAX];
    char wan22_dir[H3_CONFIG_PATH_MAX];
    char qwen21_dir[H3_CONFIG_PATH_MAX];
    char qwen21_lora_dir[H3_CONFIG_PATH_MAX];
    char text_cache_dir[H3_CONFIG_PATH_MAX];
    char wan22_text_cache_dir[H3_CONFIG_PATH_MAX];
    char qwen21_text_cache_dir[H3_CONFIG_PATH_MAX];
    /* The file the values came from; empty when no config file was found. */
    char source[H3_CONFIG_PATH_MAX];
} h3_config;

/* Reads the first file that exists among h3_config_search_paths(), or
 * explicit_path (a --config flag or H3_CONFIG, an error when it does not
 * exist). Returns 1 when a file was read, 0 when none exists (not an error),
 * -1 when the file exists but cannot be used. "~" in a value is expanded with
 * h3_home_dir(). */
int h3_config_load(const char *explicit_path, h3_config *config, char *error,
                   size_t error_size);

/* The value for a key ("h3_dir", "qwen21_lora_dir", ...) or "" when unset.
 * Exists so tests and callers can drive the table by name instead of repeating
 * the field list a third time. */
const char *h3_config_field(const h3_config *config, const char *key);

/* Sets one key by name, for a caller that has the key as a string (the
 * `h3 paths` overrides). Returns 0 for an unknown key or a value too long for
 * the slot, so a typo is reported instead of silently doing nothing. */
int h3_config_set(h3_config *config, const char *key, const char *value);

/* $HOME, or %USERPROFILE% on Windows when HOME is unset; "" when neither is.
 * The engine needs this because a default pack path of "~/h3-models/pdd8" is
 * meaningless on a Windows box where only USERPROFILE is set. */
const char *h3_home_dir(void);

/* Directory holding the running executable, without a trailing separator; ""
 * when it cannot be determined.
 *
 * This is the anchor for the config file, not the working directory: a
 * double-clicked launcher, a scheduled task and a shell started somewhere else
 * all have different cwds, but the directory the binary sits in is the same.
 * On Windows that is also where a `dist/` copy puts its h3.config.json. */
const char *h3_program_dir(void);

/* Expands a leading "~", "~/" or "~\\" with h3_home_dir(); anything else is
 * copied verbatim. Returns 0 when the result does not fit the buffer, so the
 * caller reports a too-long path instead of using a truncated one. */
int h3_config_expand_home(const char *path, char *out, size_t capacity);

/* The config files that would be tried, in order, without touching the disk:
 * <program dir>/h3.config.json, then ./h3.config.json, then
 * <home>/.config/h3/config.json. Exposed so tests can pin the order. */
int h3_config_search_paths(char paths[][H3_CONFIG_PATH_MAX], int max);

/* Writes the resolved paths as `key='value'` lines, one per key, skipping the
 * ones that are unset, prefixed by `h3_config_source='<file>'`.  This is what
 * `h3 paths` prints and what tools/gates.sh evals: a gate runner has to know
 * where the packs are, and the alternative — a second JSON parser in shell —
 * is a second place for the key list to go stale.
 *
 * Values are single-quoted, so `eval "$(h3 paths)"` is safe for paths with
 * spaces; a caller that only wants one value can still read it with
 * `h3 paths | sed -n 's/^wan22_dir=//p'` and strip the quotes. */
void h3_config_print(const h3_config *config, FILE *out);

#endif
