#include "h3_wan22_tokenizer.h"

#include "third_party/cJSON.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNIGRAM_UNK_PENALTY 10.0

struct h3_wan22_tokenizer {
    /* model vocabulary: pieces + log scores, hash-indexed for prefix
     * lookups during the Viterbi walk. */
    char **pieces;                 /* [piece_count] UTF-8 strings */
    uint32_t *piece_lengths;
    float *piece_scores;
    uint32_t piece_count;
    uint32_t max_piece_length;
    float min_score;
    int32_t unk_id;

    /* added tokens (specials + extra_ids), matched on raw text. */
    char **added;
    uint32_t *added_lengths;
    int32_t *added_ids;
    uint32_t added_count;

    int32_t eos_id;                /* </s> from the post-processor */
    int32_t pad_id;                /* <pad> */

    /* open-addressing hash set: piece bytes -> piece index (+1). */
    uint32_t *table;
    uint32_t table_mask;
};

static void tok_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static uint64_t tok_hash(const char *bytes, uint32_t length) {
    uint64_t hash = 1469598103934665603ULL;    /* FNV-1a 64 */
    for (uint32_t index = 0; index < length; index++) {
        hash ^= (unsigned char)bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void tok_put(h3_wan22_tokenizer *tokenizer, uint32_t index) {
    const char *bytes = tokenizer->pieces[index];
    uint32_t length = tokenizer->piece_lengths[index];
    uint32_t slot = (uint32_t)tok_hash(bytes, length) & tokenizer->table_mask;
    while (tokenizer->table[slot]) {
        slot = (slot + 1) & tokenizer->table_mask;
    }
    tokenizer->table[slot] = index + 1;
}

/* UTF-8 character length from a leading byte (4 on malformed input, so the
 * walk always advances). */
static uint32_t tok_char_length(const char *text, uint32_t remaining) {
    unsigned char lead = (unsigned char)text[0];
    uint32_t want = lead < 0x80 ? 1
                  : (lead & 0xE0) == 0xC0 ? 2
                  : (lead & 0xF0) == 0xE0 ? 3
                  : (lead & 0xF8) == 0xF0 ? 4
                  : 1;
    return want <= remaining ? want : remaining ? remaining : 1;
}

static char *tok_strdup(const char *bytes, uint32_t length) {
    char *copy = malloc((size_t)length + 1);
    if (!copy) return NULL;
    memcpy(copy, bytes, length);
    copy[length] = 0;
    return copy;
}

h3_wan22_tokenizer *h3_wan22_tokenizer_load(const char *tokenizer_json_path,
                                            char *error, size_t error_size) {
    FILE *file = fopen(tokenizer_json_path, "rb");
    if (!file) {
        tok_fail(error, error_size, "cannot open %s", tokenizer_json_path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        tok_fail(error, error_size, "cannot seek %s", tokenizer_json_path);
        return NULL;
    }
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *json = malloc((size_t)file_size + 1);
    if (!json || fread(json, 1, (size_t)file_size, file) != (size_t)file_size) {
        fclose(file);
        free(json);
        tok_fail(error, error_size, "cannot read %s", tokenizer_json_path);
        return NULL;
    }
    fclose(file);
    json[file_size] = 0;

    cJSON *root = cJSON_Parse(json);
    free(json);
    if (!root) {
        tok_fail(error, error_size, "cannot parse %s", tokenizer_json_path);
        return NULL;
    }

    h3_wan22_tokenizer *tokenizer = calloc(1, sizeof(h3_wan22_tokenizer));
    if (!tokenizer) {
        cJSON_Delete(root);
        tok_fail(error, error_size, "cannot allocate tokenizer");
        return NULL;
    }

    cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");
    cJSON *vocab = model ? cJSON_GetObjectItemCaseSensitive(model, "vocab")
                         : NULL;
    cJSON *unk = model ? cJSON_GetObjectItemCaseSensitive(model, "unk_id")
                       : NULL;
    if (!cJSON_IsArray(vocab) || !cJSON_IsNumber(unk)) {
        tok_fail(error, error_size, "tokenizer.json has no unigram vocab");
        goto failure;
    }
    tokenizer->piece_count = (uint32_t)cJSON_GetArraySize(vocab);
    tokenizer->unk_id = (int32_t)unk->valuedouble;
    tokenizer->pieces = malloc((size_t)tokenizer->piece_count *
                               sizeof(char *));
    tokenizer->piece_lengths = malloc((size_t)tokenizer->piece_count *
                                      sizeof(uint32_t));
    tokenizer->piece_scores = malloc((size_t)tokenizer->piece_count *
                                     sizeof(float));
    uint32_t table_size = 1;
    while (table_size < tokenizer->piece_count * 2) table_size <<= 1;
    tokenizer->table = calloc(table_size, sizeof(uint32_t));
    tokenizer->table_mask = table_size - 1;
    if (!tokenizer->pieces || !tokenizer->piece_lengths ||
        !tokenizer->piece_scores || !tokenizer->table) {
        tok_fail(error, error_size, "cannot allocate unigram vocab");
        goto failure;
    }

    tokenizer->min_score = 0.0f;
    uint32_t piece_index = 0;
    cJSON *entry = NULL;
    cJSON_ArrayForEach(entry, vocab) {
        cJSON *piece = cJSON_GetArrayItem(entry, 0);
        cJSON *score = cJSON_GetArrayItem(entry, 1);
        if (!cJSON_IsString(piece) || !cJSON_IsNumber(score)) {
            tok_fail(error, error_size, "bad vocab entry %u", piece_index);
            goto failure;
        }
        const char *bytes = piece->valuestring;
        uint32_t length = (uint32_t)strlen(bytes);
        tokenizer->pieces[piece_index] = tok_strdup(bytes, length);
        tokenizer->piece_lengths[piece_index] = length;
        tokenizer->piece_scores[piece_index] = (float)score->valuedouble;
        if (piece_index == 0 ||
            tokenizer->piece_scores[piece_index] < tokenizer->min_score) {
            tokenizer->min_score = tokenizer->piece_scores[piece_index];
        }
        if (length > tokenizer->max_piece_length) {
            tokenizer->max_piece_length = length;
        }
        tok_put(tokenizer, piece_index);
        piece_index++;
    }

    /* added tokens (matched on the raw prompt before normalization). */
    cJSON *added = cJSON_GetObjectItemCaseSensitive(root, "added_tokens");
    tokenizer->added_count =
        cJSON_IsArray(added) ? (uint32_t)cJSON_GetArraySize(added) : 0;
    tokenizer->added = malloc((size_t)tokenizer->added_count * sizeof(char *));
    tokenizer->added_lengths = malloc((size_t)tokenizer->added_count *
                                      sizeof(uint32_t));
    tokenizer->added_ids = malloc((size_t)tokenizer->added_count *
                                  sizeof(int32_t));
    if (!tokenizer->added || !tokenizer->added_lengths ||
        !tokenizer->added_ids) {
        tok_fail(error, error_size, "cannot allocate added tokens");
        goto failure;
    }
    uint32_t added_index = 0;
    cJSON *added_entry = NULL;
    cJSON_ArrayForEach(added_entry, added) {
        cJSON *content = cJSON_GetObjectItemCaseSensitive(added_entry,
                                                          "content");
        cJSON *id = cJSON_GetObjectItemCaseSensitive(added_entry, "id");
        if (!cJSON_IsString(content) || !cJSON_IsNumber(id)) {
            tok_fail(error, error_size, "bad added token %u", added_index);
            goto failure;
        }
        uint32_t length = (uint32_t)strlen(content->valuestring);
        tokenizer->added[added_index] = tok_strdup(content->valuestring,
                                                   length);
        tokenizer->added_lengths[added_index] = length;
        tokenizer->added_ids[added_index] = (int32_t)id->valuedouble;
        added_index++;
    }

    /* post-processor: TemplateProcessing single = "A </s>"; pad token is
     * the added token with the smallest id (<pad> = 0). */
    tokenizer->eos_id = -1;
    for (uint32_t index = 0; index < tokenizer->added_count; index++) {
        if (strcmp(tokenizer->added[index], "</s>") == 0) {
            tokenizer->eos_id = tokenizer->added_ids[index];
        }
    }
    tokenizer->pad_id = 0;
    cJSON_Delete(root);
    if (tokenizer->eos_id < 0) {
        tok_fail(error, error_size, "no </s> added token found");
        goto failure;
    }
    return tokenizer;

failure:
    cJSON_Delete(root);
    h3_wan22_tokenizer_free(tokenizer);
    return NULL;
}

void h3_wan22_tokenizer_free(h3_wan22_tokenizer *tokenizer) {
    if (!tokenizer) return;
    if (tokenizer->pieces) {
        for (uint32_t index = 0; index < tokenizer->piece_count; index++) {
            free(tokenizer->pieces[index]);
        }
        free(tokenizer->pieces);
    }
    free(tokenizer->piece_lengths);
    free(tokenizer->piece_scores);
    if (tokenizer->added) {
        for (uint32_t index = 0; index < tokenizer->added_count; index++) {
            free(tokenizer->added[index]);
        }
        free(tokenizer->added);
    }
    free(tokenizer->added_lengths);
    free(tokenizer->added_ids);
    free(tokenizer->table);
    free(tokenizer);
}

/* Python's `\s` for str patterns: the ASCII controls, the Unicode space
 * separators, and the two line separators.  WanGP's whitespace_clean is
 * literally `re.sub(r"\s+", " ", text)`, so U+3000 is a space there and has to
 * be one here too.
 *
 * There is deliberately NO fullwidth fold (U+FF01..U+FF5E -> ASCII).  WanGP
 * builds this tokenizer with clean='whitespace' (t5.py:708), not
 * basic_clean, so ftfy never runs; folding anyway rewrites the prompt before
 * the vocabulary sees it -- `，` (U+FF0C, id 356) becomes `,` (id 275) in every
 * Chinese prompt.  English prompts contain no fullwidth characters, which is
 * why every English-only comparison came out at 1e-6 while the Chinese
 * negative prompt was 5-10% off per row. */
static int tok_is_space(uint32_t codepoint) {
    if (codepoint == 0x20 || (codepoint >= 0x09 && codepoint <= 0x0D)) return 1;
    if (codepoint == 0x85 || codepoint == 0xA0) return 1;
    if (codepoint == 0x1680) return 1;
    if (codepoint >= 0x2000 && codepoint <= 0x200A) return 1;
    if (codepoint == 0x2028 || codepoint == 0x2029) return 1;
    if (codepoint == 0x202F || codepoint == 0x205F) return 1;
    if (codepoint == 0x3000) return 1;
    return 0;
}

/* whitespace clean (tokenizers.py): \s+ -> single space, strip. */
static uint32_t tok_whitespace_clean(const char *text, char *out) {
    uint32_t out_index = 0;
    const char *cursor = text;
    while (*cursor) {
        const unsigned char lead = (unsigned char)*cursor;
        uint32_t want = lead < 0x80 ? 1
                      : (lead & 0xE0) == 0xC0 ? 2
                      : (lead & 0xF0) == 0xE0 ? 3
                      : (lead & 0xF8) == 0xF0 ? 4
                      : 1;
        uint32_t codepoint = lead;
        int valid = want == 1;
        if (want > 1) {
            codepoint = lead & (0x7F >> (want - 1));
            valid = 1;
            for (uint32_t extra = 1; extra < want; extra++) {
                const unsigned char cont = (unsigned char)cursor[extra];
                if ((cont & 0xC0) != 0x80) {
                    valid = 0;
                    break;
                }
                codepoint = (codepoint << 6) | (cont & 0x3F);
            }
        }
        if (valid && tok_is_space(codepoint)) {
            while (*cursor) {
                const unsigned char next_lead = (unsigned char)*cursor;
                uint32_t next_want = next_lead < 0x80 ? 1
                                   : (next_lead & 0xE0) == 0xC0 ? 2
                                   : (next_lead & 0xF0) == 0xE0 ? 3
                                   : (next_lead & 0xF8) == 0xF0 ? 4
                                   : 1;
                uint32_t next = next_lead;
                int next_valid = next_want == 1;
                if (next_want > 1) {
                    next = next_lead & (0x7F >> (next_want - 1));
                    next_valid = 1;
                    for (uint32_t extra = 1; extra < next_want; extra++) {
                        const unsigned char cont =
                            (unsigned char)cursor[extra];
                        if ((cont & 0xC0) != 0x80) {
                            next_valid = 0;
                            break;
                        }
                        next = (next << 6) | (cont & 0x3F);
                    }
                }
                if (!(next_valid && tok_is_space(next))) break;
                cursor += next_want;
            }
            out[out_index++] = ' ';
        } else {
            memcpy(out + out_index, cursor, want);
            out_index += want;
            cursor += want;
        }
    }
    while (out_index && out[out_index - 1] == ' ') out_index--;
    uint32_t start = 0;
    while (start < out_index && out[start] == ' ') start++;
    if (start) memmove(out, out + start, out_index - start);
    out[out_index - start] = 0;
    return out_index - start;
}

/* Unigram Viterbi over one metaspace segment.  best_score is F64 like HF
 * tokenizers; unknown characters map to one <unk> each. */
static int tok_viterbi(const h3_wan22_tokenizer *tokenizer,
                       const char *segment, uint32_t length,
                       int32_t *ids, uint32_t *id_count,
                       char *error, size_t error_size) {
    double *best = malloc(((size_t)length + 1) * sizeof(double));
    uint32_t *back_token = malloc((size_t)length * sizeof(uint32_t));
    uint32_t *back_length = malloc((size_t)length * sizeof(uint32_t));
    if (!best || !back_token || !back_length) {
        free(best); free(back_token); free(back_length);
        tok_fail(error, error_size, "cannot allocate viterbi buffers");
        return 0;
    }
    for (uint32_t index = 0; index <= length; index++) {
        best[index] = -INFINITY;
    }
    best[0] = 0.0;
    const float unk_score = tokenizer->min_score - (float)UNIGRAM_UNK_PENALTY;

    for (uint32_t start = 0; start < length; start++) {
        if (best[start] == -INFINITY) continue;
        uint32_t limit = length - start;
        if (limit > tokenizer->max_piece_length) {
            limit = tokenizer->max_piece_length;
        }
        int matched = 0;
        /* FNV-1a is appendable, so each prefix extends the running hash. */
        uint64_t hash = 1469598103934665603ULL;
        for (uint32_t span = 1; span <= limit; span++) {
            hash ^= (unsigned char)segment[start + span - 1];
            hash *= 1099511628211ULL;
            /* inline lookup with the running hash */
            uint32_t slot = (uint32_t)hash & tokenizer->table_mask;
            while (tokenizer->table[slot]) {
                uint32_t index = tokenizer->table[slot] - 1;
                if (tokenizer->piece_lengths[index] == span &&
                    memcmp(tokenizer->pieces[index],
                           segment + start, span) == 0) {
                    matched = 1;
                    double candidate =
                        best[start] + (double)tokenizer->piece_scores[index];
                    if (best[start + span] == -INFINITY ||
                        candidate > best[start + span]) {
                        best[start + span] = candidate;
                        back_token[start + span - 1] = index;
                        back_length[start + span - 1] = span;
                    }
                    break;
                }
                slot = (slot + 1) & tokenizer->table_mask;
            }
        }
        if (!matched) {
            uint32_t char_len = tok_char_length(segment + start,
                                                length - start);
            double candidate = best[start] + (double)unk_score;
            if (best[start + char_len] == -INFINITY ||
                candidate > best[start + char_len]) {
                best[start + char_len] = candidate;
                back_token[start + char_len - 1] = UINT32_MAX;  /* <unk> */
                back_length[start + char_len - 1] = char_len;
            }
        }
    }

    if (best[length] == -INFINITY) {
        /* empty segment: no tokens */
        free(best); free(back_token); free(back_length);
        *id_count = 0;
        return 1;
    }

    /* count tokens by walking the back-pointers */
    uint32_t count = 0;
    uint32_t end = length;
    while (end > 0) {
        uint32_t span = back_length[end - 1];
        count++;
        end -= span;
    }
    if (count > *id_count) {
        free(best); free(back_token); free(back_length);
        tok_fail(error, error_size, "segment needs %u ids", count);
        return 0;
    }
    uint32_t write = count;
    end = length;
    while (end > 0) {
        uint32_t span = back_length[end - 1];
        uint32_t token = back_token[end - 1];
        write--;
        ids[write] = token == UINT32_MAX
                         ? tokenizer->unk_id
                         : (int32_t)token;
        end -= span;
    }
    free(best); free(back_token); free(back_length);
    *id_count = count;
    return 1;
}

int h3_wan22_tokenizer_encode(const h3_wan22_tokenizer *tokenizer,
                              const char *text, uint32_t text_len,
                              int32_t *ids, uint8_t *mask,
                              char *error, size_t error_size) {
    if (!tokenizer || !ids || !mask || !text_len) {
        tok_fail(error, error_size, "tokenizer_encode: bad arguments");
        return -1;
    }

    size_t raw_length = strlen(text);
    char *clean = malloc(raw_length + 2);
    /* metaspace: every space becomes the 3-byte U+2581, plus the prefix */
    char *segment = malloc(strlen(text) * 3 + 4);
    int32_t *content = malloc((size_t)text_len * sizeof(int32_t));
    int32_t *piece_ids = malloc((size_t)text_len * sizeof(int32_t));
    if (!clean || !segment || !content || !piece_ids) {
        free(clean); free(segment); free(content); free(piece_ids);
        tok_fail(error, error_size, "cannot allocate encode buffers");
        return -1;
    }
    uint32_t clean_length = tok_whitespace_clean(text, clean);

    uint32_t content_count = 0;
    uint32_t cursor = 0;
    while (cursor < clean_length && content_count < text_len) {
        /* longest added-token match on the raw (cleaned) text */
        int32_t matched_id = -1;
        uint32_t matched_length = 0;
        for (uint32_t index = 0; index < tokenizer->added_count; index++) {
            uint32_t length = tokenizer->added_lengths[index];
            if (length > matched_length &&
                cursor + length <= clean_length &&
                memcmp(clean + cursor, tokenizer->added[index], length) == 0) {
                matched_id = tokenizer->added_ids[index];
                matched_length = length;
            }
        }
        if (matched_id >= 0) {
            if (content_count < text_len) {
                content[content_count++] = matched_id;
            }
            cursor += matched_length;
            continue;
        }
        /* one raw-text segment up to the next added token */
        uint32_t segment_start = cursor;
        uint32_t segment_end = cursor;
        while (segment_end < clean_length) {
            int is_added = 0;
            for (uint32_t index = 0; index < tokenizer->added_count; index++) {
                uint32_t length = tokenizer->added_lengths[index];
                if (segment_end + length <= clean_length &&
                    memcmp(clean + segment_end, tokenizer->added[index],
                           length) == 0) {
                    is_added = 1;
                    break;
                }
            }
            if (is_added) break;
            segment_end++;
        }
        cursor = segment_end;

        /* Metaspace pre-tokenizer (verified against transformers 4.57 /
         * tokenizers 0.22): replace ' ' with U+2581, prepend U+2581 only
         * when the result does not already start with one, then split into
         * word chunks at every U+2581 (separator kept with the following
         * chunk, empty chunks dropped) and unigram-encode each chunk. */
        static const char prefix[] = "\xe2\x96\x81";
        uint32_t write = 0;
        for (uint32_t index = segment_start; index < segment_end; index++) {
            if (clean[index] == ' ') {
                memcpy(segment + write, prefix, 3);
                write += 3;
            } else {
                segment[write++] = clean[index];
            }
        }
        segment[write] = 0;
        if (write == 0) continue;
        if (memcmp(segment, prefix, 3) != 0) {
            memmove(segment + 3, segment, write);
            memcpy(segment, prefix, 3);
            write += 3;
        }

        uint32_t chunk_start = 0;
        uint32_t position = 0;
        while (position < write && content_count < text_len) {
            if (position > 0 &&
                memcmp(segment + position, prefix, 3) == 0) {
                uint32_t length = position - chunk_start;
                if (length) {
                    uint32_t capacity = text_len - content_count;
                    uint32_t piece_count = capacity;
                    if (!tok_viterbi(tokenizer, segment + chunk_start,
                                     length, piece_ids, &piece_count,
                                     error, error_size)) {
                        free(clean); free(segment); free(content);
                        free(piece_ids);
                        return -1;
                    }
                    if (piece_count > capacity) piece_count = capacity;
                    memcpy(content + content_count, piece_ids,
                           (size_t)piece_count * sizeof(int32_t));
                    content_count += piece_count;
                }
                chunk_start = position;
                position += 3;
            } else {
                position++;
            }
        }
        uint32_t length = write - chunk_start;
        if (length && content_count < text_len) {
            uint32_t capacity = text_len - content_count;
            uint32_t piece_count = capacity;
            if (!tok_viterbi(tokenizer, segment + chunk_start, length,
                             piece_ids, &piece_count, error, error_size)) {
                free(clean); free(segment); free(content); free(piece_ids);
                return -1;
            }
            if (piece_count > capacity) piece_count = capacity;
            memcpy(content + content_count, piece_ids,
                   (size_t)piece_count * sizeof(int32_t));
            content_count += piece_count;
        }
    }
    free(clean);
    free(segment);
    free(piece_ids);

    /* post-processor </s>, truncation keeps room for it (HF subtracts the
     * post-processor's added tokens from the truncation length). */
    if (content_count >= text_len) {
        content_count = text_len - 1;
    }
    content[content_count++] = tokenizer->eos_id;

    for (uint32_t index = 0; index < text_len; index++) {
        if (index < content_count) {
            ids[index] = content[index];
            mask[index] = 1;
        } else {
            ids[index] = tokenizer->pad_id;
            mask[index] = 0;
        }
    }
    free(content);
    return (int)content_count;
}
