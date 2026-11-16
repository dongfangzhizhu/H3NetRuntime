/* <sys/socket.h> over Winsock2.
 *
 * The engine stores sockets in `h3_socket` (see h3_serve.c), which is `int` on
 * POSIX and a 64-bit handle here; the wrappers below translate the POSIX
 * return conventions (-1 + errno) that the server's error paths are written
 * against. */
#ifndef H3_WIN_SYS_SOCKET_H
#define H3_WIN_SYS_SOCKET_H

#include "h3_platform.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#ifdef __cplusplus
extern "C" {
#endif

/* No SIGPIPE on Windows, so nothing to suppress. */
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

/* Winsock has no EINTR-style partial-write signal; the wrappers map
 * WSAGetLastError() onto errno so the server's `errno == EINTR` loops work. */
int h3_win_socket_startup(void);
int h3_win_socket_last_error(void);

/* SOCKET is pointer-width on Win64, and the server keeps its sockets in
 * intptr_t-typed variables, so the wrappers take the same width. */
ssize_t h3_win_recv(intptr_t fd, void *buffer, size_t bytes, int flags);
ssize_t h3_win_send(intptr_t fd, const void *buffer, size_t bytes, int flags);
int h3_win_socket_close(intptr_t fd);
int h3_win_socket_shutdown_write(intptr_t fd);

#define recv(fd, buffer, bytes, flags) \
    h3_win_recv((intptr_t)(fd), (buffer), (bytes), (flags))
#define send(fd, buffer, bytes, flags) \
    h3_win_send((intptr_t)(fd), (buffer), (bytes), (flags))
#define closesocket(fd) h3_win_socket_close((intptr_t)(fd))

#ifdef __cplusplus
}
#endif

#endif /* H3_WIN_SYS_SOCKET_H */
