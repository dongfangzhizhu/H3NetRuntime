#ifndef H3_WAN22_TOKENIZER_H
#define H3_WAN22_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/* Unigram (sentencepiece) tokenizer for the umt5-xxl text encoder, loaded
 * straight from tokenizer.json (Wan2GP/ckpts/umt5-xxl/tokenizer.json).
 *
 * The pipeline mirrors Huggingface tokenizers + the WanGP HuggingfaceTokenizer
 * wrapper (models/wan/modules/tokenizers.py, clean = 'whitespace'):
 *
 *   1. whitespace clean: \s+ -> ' ', strip.  `\s` is Python's Unicode set, so
 *      U+3000 and the other space separators count; fullwidth punctuation
 *      does NOT fold (WanGP builds this tokenizer with clean='whitespace',
 *      t5.py:708, so ftfy never runs — folding `，` to `,` silently changes
 *      every Chinese prompt's token ids)
 *   2. split on the 304 added tokens (<pad>..</s>..<extra_id_299>),
 *      longest match first, matched on the raw text
 *   3. per segment: tokenizer.json normalizer (collapse ' {2,}' -> ' ',
 *      already done) then the Metaspace pre-tokenizer: prepend U+2581 and
 *      replace ' ' with U+2581
 *   4. Unigram Viterbi over the 256300-piece vocabulary (byte_fallback is
 *      off; unmatched characters become <unk> id 3, one UTF-8 char at a
 *      time, with score min_score - 10.0)
 *   5. TemplateProcessing appends </s> (id 1)
 *   6. truncate to text_len (special token included, like HF truncation
 *      with the post-processor's added tokens) and pad with <pad> (id 0)
 */
typedef struct h3_wan22_tokenizer h3_wan22_tokenizer;

h3_wan22_tokenizer *h3_wan22_tokenizer_load(const char *tokenizer_json_path,
                                            char *error, size_t error_size);
void h3_wan22_tokenizer_free(h3_wan22_tokenizer *tokenizer);

/* Encode one prompt into ids int32[text_len] + mask uint8[text_len]
 * (1 = real token, 0 = padding).  Returns the number of real tokens
 * (content + </s>, i.e. the mask popcount), or -1 on error. */
int h3_wan22_tokenizer_encode(const h3_wan22_tokenizer *tokenizer,
                              const char *text, uint32_t text_len,
                              int32_t *ids, uint8_t *mask,
                              char *error, size_t error_size);

#endif
