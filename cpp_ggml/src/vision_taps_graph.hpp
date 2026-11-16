// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Qwen3-VL ViT forward with pre-merger taps for the pixel decoder.
//
// The graph replicates llama.cpp mtmd's qwen3vl block sequence (pin
// 620d2225) exactly — dual conv patch embed (kt=2 split), learned position
// embedding resized bilinear/align_corners, pre-LN, per-block
// ln1 -> qkv -> vision m-rope -> attention -> residual, ln2 -> ffn ->
// residual — but instead of the final merger it EXPOSES the raw block
// outputs at the released feature layers (5/11/17/23) as taps copied into
// dedicated leaves (the single-graph + ictx-leaf tap contract from the
// sibling integrations; interior view taps corrupt gallocr reuse).
//
// Token order at the taps: (block_y, block_x, intra_y, intra_x) — the same
// order Qwen's processor feeds the mergers, so the decoder consumes them
// with the standard un-shuffle (feature_bridge.premerge_tokens_to_2d).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace wwv {

struct VisionTapsParams {
    int32_t grid_w = 0;      // post-resize pixel width  (ViT input)
    int32_t grid_h = 0;      // post-resize pixel height (ViT input)
    int32_t patch_size = 16;
    int32_t merge_size = 2;  // spatial merge (2 convs pair along x AND y)
    int32_t heads = 16;      // ViT attention heads (validated against n_embd)
    float eps = 1e-6f;
};

struct VisionTapsResult {
    // Tap outputs, one per requested layer, each ne = (C, T) with tokens in
    // (block_y, block_x, intra_y, intra_x) order; T = grid_w * grid_h /
    // (patch * merge)^2 * (patch*merge / patch)^2 ... i.e. the FULL
    // patch-grid token count (blocks run pre-merger at full resolution).
    std::vector<ggml_tensor*> taps;
    int64_t n_tokens = 0;      // full patch-grid token count
    int64_t token_grid_w = 0;  // full patch grid width  (nx / patch)
    int64_t token_grid_h = 0;  // full patch grid height (ny / patch)
    int64_t n_embd = 0;
};

// Builds the forward graph. `taps_layers` lists block indexes to tap
// (0-based). Weights come from the loader tensors already registered in
// `wctx` under llama.cpp mmproj names (v.patch_embd.*, v.blk.N.*, ...).
// Returns nullptr on invalid arguments; *err carries the reason.
ggml_cgraph* build_vision_taps_graph(
    ggml_context* ctx0, const VisionTapsParams& params,
    const std::vector<int32_t>& taps_layers,
    const std::vector<ggml_tensor*>& tap_leaves,
    ggml_tensor* pixels,     // (W, H, 3, 1) f32, Qwen-normalized
    ggml_tensor* positions,  // (4 * n_pos) i32, m-rope sections
    const std::vector<ggml_tensor*>& patch_embd,  // 2 conv kernels
    ggml_tensor* patch_bias,
    ggml_tensor* position_embd,  // stored learned pos grid (C, W0, H0)
    ggml_tensor* pre_ln_w, ggml_tensor* pre_ln_b,  // nullable
    const std::vector<int>& layer_indexes,         // per-block weight index
    // Per-block weight accessor: {ln1_w, ln1_b, qkv_w, qkv_b, o_w, o_b,
    //                              ln2_w, ln2_b, ff_up_w, ff_up_b,
    //                              ff_gate_w, ff_gate_b, ff_down_w,
    //                              ff_down_b}
    const std::vector<std::vector<ggml_tensor*>>& block_weights,
    VisionTapsResult* out, std::string* err);

// Host-side m-rope position ids for one image: 4 sections of n_pos entries
// each, (block_y, block_x, intra_y, intra_x) token order. Returns 4 *
// (pw*ph) i32 values where pw = pixel_w / patch, ph = pixel_h / patch.
std::vector<int32_t> vision_mrope_positions(int32_t pixel_w, int32_t pixel_h,
                                            int32_t patch_size,
                                            int32_t merge_size);

// Host-side un-shuffle of a tapped token block into a row-major map
// (feature_bridge.premerge_tokens_to_2d). ggml tensors are 4-D and the
// shuffle is a 5-axis permutation, so this runs after tap read-back (same
// host-glue pattern as the sibling integrations' HFRE stage).
//   tokens:  (E, T) f32, T = ((by*Wb + bx)*m + iy)*m + ix
//   out_map: (E, W', H') f32 row-major, W' = m*Wb, H' = m*Hb
//   map_w/map_h are the BLOCK grid (W/m, H/m).
void unshuffle_tokens_host(const float* tokens, int64_t n_embd,
                           int64_t n_tokens, int64_t map_w, int64_t map_h,
                           int32_t merge_size, float* out_map);

}  // namespace wwv
