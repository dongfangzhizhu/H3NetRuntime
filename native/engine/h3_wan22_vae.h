#ifndef H3_WAN22_VAE_H
#define H3_WAN22_VAE_H

#include <stddef.h>
#include <stdint.h>

/* Wan 2.2 T2V 5B Phase 2: CausalConv3d VAE decoder, CPU fp32 reference path.
 *
 * Architecture (Wan2GP models/wan/modules/vae2_2.py, Wan2.2 VAE config):
 * Decoder3d(dec_dim 256, z_dim 48, dim_mult [1,2,4,4], num_res_blocks 2).
 * WanVAE_ reverses temperal_downsample=[False,True,True], so the decoder's
 * upsample chain is [True, True, False]: blocks 0 and 1 end in an
 * upsample3d Resample (temporal CausalConv3d + nearest x2 + Conv2d) and
 * block 2 in an upsample2d one; block 3 has no resample.
 *
 *   dims      = [1024, 1024, 1024, 512, 256]
 *   conv1     : CausalConv3d(48 -> 1024, 3x3x3)
 *   middle    : ResBlock(1024) -> AttentionBlock(1024) -> ResBlock(1024)
 *   block 0   : 3x ResBlock(1024) + up3d  (time_conv 1024->2048, spatial x2)
 *   block 1   : 3x ResBlock(1024) + up3d  (time_conv 1024->2048, spatial x2)
 *   block 2   : 3x ResBlock(1024->512) + up2d (spatial x2)
 *   block 3   : 3x ResBlock(512->256)
 *   head      : RMS -> SiLU -> CausalConv3d(256 -> 12) -> unpatchify(2x2)
 *
 * Temporal compression is 4x (T latent frames -> 4T-3 video frames),
 * spatial 16x (8x from the resample chain + 2x from unpatchify).
 *
 * This implementation decodes the whole latent clip in one pass with
 * explicit causal padding instead of the reference's per-frame feat_cache
 * streaming; both are exactly equivalent:
 *   - every plain CausalConv3d (conv1, the residual convs, the head conv)
 *     accumulates the last two input frames per conv in feat_cache, so
 *     streaming output frame t equals the whole-sequence causal conv at t
 *     (a 1-frame cache at chunk 0 plus the remaining zero pad reproduces
 *     the 2-frame front pad);
 *   - in an upsample3d Resample the time_conv skips its output for chunk 0
 *     (the "Rep" cache state) and never sees the chunk-0 input frame, so
 *     the whole-sequence equivalent is
 *         y = causal time_conv over R[1:]          (2-frame zero front pad)
 *         main = [spatial_up(R[0])] ++ spatial_up(interleave(y))
 *         shortcut = DupUp3d(x)[:, :, 1:]          (first frame dropped)
 *     which is what this decoder computes.
 *
 * All math is fp32 (the checkpoint is fp32); convolutions run as OpenMP
 * stencil loops over output channels like h3_zimage_vae.c.  Weights are
 * read directly from Wan2.2_VAE.safetensors with h3_st (tensor names
 * "decoder.*" and "conv2.*").
 */

typedef struct h3_wan22_vae_decoder h3_wan22_vae_decoder;

/* Load the decoder half (decoder.* + conv2.*) of a Wan 2.2 VAE checkpoint
 * (single safetensors file, fp32). */
h3_wan22_vae_decoder *h3_wan22_vae_decoder_load(const char *vae_path,
                                                char *error,
                                                size_t error_size);
void h3_wan22_vae_decoder_free(h3_wan22_vae_decoder *decoder);

/* Decode raw latents (pre-scale, [48, T, h, w] channel-major fp32) into
 * pixels [3, 4T-3, 16h, 16w] fp32 (pre-clamp; the caller clamps to
 * [-1, 1]).  Applies the 48 per-channel mean/std constants internally. */
int h3_wan22_vae_decode(const h3_wan22_vae_decoder *decoder,
                        const float *z, uint32_t frames, uint32_t h,
                        uint32_t w, float *out,
                        char *error, size_t error_size);

#endif
