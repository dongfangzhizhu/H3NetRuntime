/* Directory/file links used to assemble a "virtual pack" out of component
 * paths that live in different directories.
 *
 * The engine resolves every weight component relative to one pack root
 * (FL2VA/transformer, FL2VA/text_encoder, ...), so per-component paths only
 * work if the standard layout exists on disk.  Rather than teach the engine a
 * second path table, h3rt materialises that layout as links:
 *
 *   Windows  directory -> junction (no privilege) or symlink
 *            file      -> hard link, then symlink, then copy
 *   Linux    symlink
 *
 * Junctions need no administrator rights, which is what makes the Windows path
 * usable from a plain `dotnet run`; symbolic links are only tried when the
 * target is on another volume (hard links cannot cross one) or when the
 * junction call fails.
 */
#ifndef H3RT_LINKS_H
#define H3RT_LINKS_H

#include <stddef.h>

/* Creates `path` and every missing parent. Returns 1 on success, 1 when the
 * directory already exists, and 0 when it cannot be created. */
int h3rt_make_dirs(const char *path);

/* 1 when anything (file, directory, link) is reachable at `path`. */
int h3rt_path_exists(const char *path);

/* 1 when `path` is a directory (following links). */
int h3rt_is_directory(const char *path);

/* Links `link_path` to `target`. `link_path` must not exist yet; its parent
 * directory is created when missing. On success returns 1. On failure returns
 * 0 and writes a human-readable reason into `error`. */
int h3rt_link_path(const char *link_path, const char *target, int is_directory,
                   char *error, size_t error_size);

/* Best-effort removal of a single link/directory entry (never recursive). */
void h3rt_remove_link(const char *path);

#endif /* H3RT_LINKS_H */
