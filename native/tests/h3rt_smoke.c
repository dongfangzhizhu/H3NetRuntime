/* C ABI smoke test.  Needs no model and no GPU unless a model directory is
 * passed as argv[1] (then it also exercises device/model introspection). */
#include "h3rt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void check(int condition, const char *what) {
    printf("[%s] %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) failures++;
}

static void expect_status(int actual, int expected, const char *what) {
    printf("[%s] %s (status=%d %s)\n",
           actual == expected ? "PASS" : "FAIL", what, actual,
           h3rt_status_text(actual));
    if (actual != expected) failures++;
}

int main(int argc, char **argv) {
    printf("h3rt version %s, ABI %d\n", h3rt_version(), h3rt_abi_version());
    check(h3rt_abi_version() == H3RT_ABI_VERSION, "ABI version matches header");
    check(strcmp(h3rt_version(), H3RT_VERSION) == 0, "version string matches");

    char *info = h3rt_build_info();
    check(info != NULL, "h3rt_build_info returns JSON");
    if (info) {
        check(strstr(info, "\"python_required\":false") != NULL ||
              strstr(info, "\"python_required\":0") != NULL,
              "build info declares no Python dependency");
        printf("  %s\n", info);
        h3rt_free_string(info);
    }

    char error[512] = "";
    h3rt_handle *handle = NULL;
    expect_status(h3rt_create("{not json", &handle, error, sizeof(error)),
                  H3RT_ERR_INVALID_JSON, "malformed options JSON rejected");
    check(handle == NULL, "no handle leaked on parse failure");
    printf("  error: %s\n", error);

    error[0] = '\0';
    expect_status(h3rt_create("{\"model_dir\":\"C:/h3rt/definitely/missing\"}",
                              &handle, error, sizeof(error)),
                  H3RT_ERR_MODEL, "missing model pack rejected");
    check(handle == NULL, "no handle leaked on model failure");
    printf("  error: %s\n", error);

    expect_status(h3rt_create(NULL, NULL, error, sizeof(error)),
                  H3RT_ERR_INVALID_ARGUMENT, "NULL out_handle rejected");

    char *out = (char *)0x1; /* must be set to NULL even on early failure */
    expect_status(h3rt_generate(NULL, "{\"prompt\":\"x\"}", NULL, &out),
                  H3RT_ERR_INVALID_ARGUMENT, "generate without a handle rejected");
    check(out == NULL, "out_json cleared on argument failure");

    if (argc > 1) {
        /* A pack path is the common case; a JSON object is used verbatim so
         * the per-component path options can be exercised too. */
        static char options[8192];
        if (argv[1][0] == '{')
            snprintf(options, sizeof(options), "%s", argv[1]);
        else
            snprintf(options, sizeof(options), "{\"model_dir\":\"%s\"}", argv[1]);
        error[0] = '\0';
        int rc = h3rt_create(options, &handle, error, sizeof(error));
        printf("h3rt_create(%s) = %d %s\n", options, rc,
               h3rt_status_text(rc));
        if (rc != H3RT_OK) {
            printf("  error: %s\n", error);
            failures++;
        } else {
            char *device = h3rt_device_info(handle);
            char *model = h3rt_model_info(handle);
            char *cache = h3rt_cache_info(handle);
            check(device && model && cache, "introspection returns JSON");
            printf("  device: %s\n", device ? device : "(null)");
            printf("  model:  %s\n", model ? model : "(null)");
            printf("  cache:  %s\n", cache ? cache : "(null)");
            h3rt_free_string(device);
            h3rt_free_string(model);
            h3rt_free_string(cache);
            expect_status(h3rt_cache_clear(handle), H3RT_OK,
                          "h3rt_cache_clear");
            expect_status(h3rt_cancel(handle), H3RT_OK, "h3rt_cancel");
            expect_status(h3rt_is_cancelled(handle), 1,
                          "h3rt_is_cancelled after cancel");
            h3rt_destroy(handle);
        }
    } else {
        printf("(pass a MiniMax-H3 pack path as argv[1] to exercise model "
               "introspection)\n");
    }

    printf("\n%s (%d failure(s))\n", failures ? "SMOKE FAILED" : "SMOKE OK",
           failures);
    return failures ? 1 : 0;
}
