/* POSIX names the engine expects from <unistd.h>.
 *
 * Most of them (read, write, open, close, unlink, access, dup, getcwd) already
 * exist in the UCRT under their POSIX spelling once _CRT_DECLARE_NONSTDC_NAMES
 * is on, so this header only has to pull in the right CRT header; the handful
 * that are genuinely missing are declared in h3_platform.h. */
#ifndef H3_WIN_UNISTD_H
#define H3_WIN_UNISTD_H

#include "h3_platform.h"

#include <io.h>
#include <direct.h>
#include <process.h>

#endif /* H3_WIN_UNISTD_H */
