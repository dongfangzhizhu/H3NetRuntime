#include "h3rt_links.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(_WIN32)
/* The CMake build force-includes platform/win32/h3_platform.h into every
 * translation unit, and that header redefines off_t for the engine's 64-bit
 * safetensors seeks.  windows.h declares functions that take off_t, so the
 * macro is hidden for the duration of the include and restored afterwards. */
#  ifdef off_t
#    undef off_t
#    define H3RT_RESTORE_OFF_T 1
#  endif
#  include <windows.h>
#  include <winioctl.h>
#  ifdef H3RT_RESTORE_OFF_T
#    define off_t int64_t
#  endif
#else
#  include <errno.h>
#  include <unistd.h>
#endif

int h3rt_make_dirs(const char *path) {
    if (!path || !*path) return 0;
    size_t length = strlen(path);
    char *buffer = (char *)malloc(length + 1);
    if (!buffer) return 0;
    memcpy(buffer, path, length + 1);
    /* Create every prefix. Both separators are accepted so a caller may hand
     * in a Windows path on Linux (and the reverse) without normalising. */
    for (size_t i = 1; i <= length; i++) {
        if (i != length && buffer[i] != '/' && buffer[i] != '\\') continue;
        char saved = buffer[i];
        buffer[i] = '\0';
        /* A drive prefix ("D:") or a UNC root is not a directory on its own. */
        size_t prefix = strlen(buffer);
        int skip = prefix == 0 || (prefix == 2 && buffer[1] == ':');
        if (!skip) mkdir(buffer, 0777);
        buffer[i] = saved;
    }
    free(buffer);
    return h3rt_is_directory(path);
}

int h3rt_path_exists(const char *path) {
    if (!path || !*path) return 0;
    struct stat info;
    if (stat(path, &info) == 0) return 1;
#if defined(_WIN32)
    /* A junction whose target is gone still stats as missing; the entry itself
     * is what the caller asked about. */
    DWORD attributes = GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES;
#else
    return lstat(path, &info) == 0;
#endif
}

int h3rt_is_directory(const char *path) {
    if (!path || !*path) return 0;
    struct stat info;
    if (stat(path, &info) != 0) return 0;
#if defined(_WIN32)
    return (info.st_mode & _S_IFDIR) != 0;
#else
    return S_ISDIR(info.st_mode);
#endif
}

#if defined(_WIN32)

#define H3RT_MOUNT_POINT_TAG 0xA0000003u /* IO_REPARSE_TAG_MOUNT_POINT */

typedef struct {
    ULONG tag;
    USHORT data_length;
    USHORT reserved;
    USHORT substitute_offset;
    USHORT substitute_length;
    USHORT print_offset;
    USHORT print_length;
    WCHAR path[1];
} h3rt_mount_point;

static wchar_t *h3rt_wide(const char *utf8) {
    if (!utf8) return NULL;
    int length = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    if (length <= 0) return NULL;
    wchar_t *out = (wchar_t *)malloc(sizeof(wchar_t) * (size_t)length);
    if (!out) return NULL;
    if (MultiByteToWideChar(CP_UTF8, 0, utf8, -1, out, length) <= 0) {
        free(out);
        return NULL;
    }
    return out;
}

/* Absolute form of `path` without a trailing separator, so a junction target
 * is never drive-relative. */
static wchar_t *h3rt_absolute_wide(const char *utf8) {
    wchar_t *raw = h3rt_wide(utf8);
    if (!raw) return NULL;
    DWORD needed = GetFullPathNameW(raw, 0, NULL, NULL);
    if (needed == 0) {
        free(raw);
        return NULL;
    }
    wchar_t *full = (wchar_t *)malloc(sizeof(wchar_t) * (size_t)(needed + 1));
    if (!full) {
        free(raw);
        return NULL;
    }
    DWORD written = GetFullPathNameW(raw, needed, full, NULL);
    free(raw);
    if (written == 0 || written >= needed) {
        free(full);
        return NULL;
    }
    while (written > 0 && (full[written - 1] == L'\\' || full[written - 1] == L'/'))
        full[--written] = L'\0';
    return full;
}

static int h3rt_create_junction(const wchar_t *link, const wchar_t *target) {
    if (!CreateDirectoryW(link, NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        return 0;
    HANDLE handle = CreateFileW(link, GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                                FILE_FLAG_BACKUP_SEMANTICS |
                                    FILE_FLAG_OPEN_REPARSE_POINT,
                                NULL);
    if (handle == INVALID_HANDLE_VALUE) {
        /* Leave nothing behind, or the symlink fallback sees a busy name. */
        RemoveDirectoryW(link);
        return 0;
    }

    /* The substitute name is an NT path: \??\ plus the local absolute path.
     * The printable name is what Explorer shows, so it stays the plain path. */
    size_t target_chars = wcslen(target);
    size_t substitute_chars = target_chars + 4;
    size_t path_chars = substitute_chars + 1 + target_chars + 1;
    size_t path_bytes = path_chars * sizeof(WCHAR);
    size_t data_bytes = 8 + path_bytes;
    size_t total = 8 + data_bytes;

    unsigned char *buffer = (unsigned char *)calloc(1, total);
    if (!buffer) {
        CloseHandle(handle);
        RemoveDirectoryW(link);
        return 0;
    }
    h3rt_mount_point *point = (h3rt_mount_point *)buffer;
    point->tag = H3RT_MOUNT_POINT_TAG;
    point->data_length = (USHORT)data_bytes;
    point->substitute_offset = 0;
    point->substitute_length = (USHORT)(substitute_chars * sizeof(WCHAR));
    point->print_offset = (USHORT)((substitute_chars + 1) * sizeof(WCHAR));
    point->print_length = (USHORT)(target_chars * sizeof(WCHAR));
    memcpy(point->path, L"\\??\\", 4 * sizeof(WCHAR));
    memcpy(point->path + 4, target, target_chars * sizeof(WCHAR));
    memcpy(point->path + substitute_chars + 1, target,
           target_chars * sizeof(WCHAR));

    DWORD returned = 0;
    BOOL ok = DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, buffer,
                              (DWORD)total, NULL, 0, &returned, NULL);
    free(buffer);
    CloseHandle(handle);
    if (!ok) RemoveDirectoryW(link);
    return ok ? 1 : 0;
}

static int h3rt_create_directory_link(const wchar_t *link, const wchar_t *target) {
    /* A junction needs no privilege at all, so it is the primary form: it
     * works on a locked-down account, in a service, and inside CI.  A
     * symbolic link (Developer Mode or SeCreateSymbolicLinkPrivilege) is the
     * fallback, which is also the only form that can point at a UNC path. */
    if (h3rt_create_junction(link, target)) return 1;
    return CreateSymbolicLinkW(link, target,
                               SYMBOLIC_LINK_FLAG_DIRECTORY | 0x2u) ? 1 : 0;
}

static int h3rt_create_file_link(const wchar_t *link, const wchar_t *target,
                                 char *error, size_t error_size) {
    /* A hard link is free and indistinguishable from the original, but it
     * cannot cross a volume; a 1.3 GiB PDD adaptor copy is the last resort. */
    if (CreateHardLinkW(link, target, NULL)) return 1;
    if (CreateSymbolicLinkW(link, target, 0x2u)) return 1;
    if (CopyFileW(target, link, FALSE)) return 1;
    snprintf(error, error_size,
             "cannot link '%ls' (hard link, symlink and copy all failed)",
             target);
    return 0;
}

int h3rt_link_path(const char *link_path, const char *target, int is_directory,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    wchar_t *link = h3rt_absolute_wide(link_path);
    wchar_t *source = h3rt_absolute_wide(target);
    if (!link || !source) {
        free(link);
        free(source);
        snprintf(error, error_size, "cannot resolve the link paths");
        return 0;
    }
    /* A stale entry from an interrupted run would make every link call fail. */
    if (h3rt_path_exists(link_path)) h3rt_remove_link(link_path);

    char parent[1024];
    snprintf(parent, sizeof(parent), "%s", link_path);
    char *cut = strrchr(parent, '/');
    char *cut2 = strrchr(parent, '\\');
    if (cut2 && (!cut || cut2 > cut)) cut = cut2;
    if (cut) {
        *cut = '\0';
        h3rt_make_dirs(parent);
    }

    int ok = is_directory ? h3rt_create_directory_link(link, source)
                          : h3rt_create_file_link(link, source, error, error_size);
    if (!ok && error && error_size && !error[0]) {
        DWORD code = GetLastError();
        snprintf(error, error_size,
                 "cannot link '%s' to '%s' (win32 error %lu)", link_path,
                 target, (unsigned long)code);
    }
    free(link);
    free(source);
    return ok;
}

void h3rt_remove_link(const char *path) {
    if (!path || !*path) return;
    wchar_t *wide = h3rt_absolute_wide(path);
    if (!wide) return;
    DWORD attributes = GetFileAttributesW(wide);
    if (attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_DIRECTORY))
        RemoveDirectoryW(wide); /* removes the reparse point, not the target */
    else
        DeleteFileW(wide);
    free(wide);
}

#else /* POSIX */

int h3rt_link_path(const char *link_path, const char *target, int is_directory,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    (void)is_directory; /* a symlink is a symlink either way */
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s", link_path);
    char *cut = strrchr(parent, '/');
    if (cut) {
        *cut = '\0';
        h3rt_make_dirs(parent);
    }
    if (h3rt_path_exists(link_path)) h3rt_remove_link(link_path);
    if (symlink(target, link_path) == 0) return 1;
    snprintf(error, error_size, "cannot symlink '%s' to '%s': %s", link_path,
             target, strerror(errno));
    return 0;
}

void h3rt_remove_link(const char *path) {
    if (!path || !*path) return;
    if (unlink(path) == 0) return;
    rmdir(path);
}

#endif
