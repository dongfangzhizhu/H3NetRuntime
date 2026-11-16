#ifndef H3_HASH_H
#define H3_HASH_H

#include <stddef.h>
#include <stdint.h>

#define H3_SHA256_DIGEST_SIZE 32u
#define H3_SHA256_HEX_SIZE 65u

typedef struct {
    uint32_t state[8];
    uint64_t bits;
    uint8_t buffer[64];
    size_t buffered;
} h3_sha256_ctx;

void h3_sha256_init(h3_sha256_ctx *ctx);
void h3_sha256_update(h3_sha256_ctx *ctx, const void *data, size_t length);
void h3_sha256_final(h3_sha256_ctx *ctx, uint8_t digest[32]);
void h3_sha256_hex(const void *data, size_t length, char hex[65]);

/* Hash a whole file in bounded memory. Returns 0 when it cannot be read. */
int h3_sha256_file(const char *path, char hex[65]);

#endif
