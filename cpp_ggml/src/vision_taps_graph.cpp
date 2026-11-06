// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Single-graph contract: every tap is ggml_cpy-ed into a dedicated leaf
// tensor allocated in the input context and expanded inside the SAME graph,
// so no allocator can re-point tap memory (the cross-graph expand trap from
// the sibling integrations never applies here: one graph, one gallocr).
//
// The (block_y, block_x, intra_y, intra_x) -> row-major un-shuffle is done
// on the host after tap read-back (the same host-glue pattern as the
// sibling integrations' HFRE stage): ggml tensors are 4-D, and the un-shuffle
// is a 5-axis permutation, so an in-graph chain cannot express it.

#include "vision_taps_graph.hpp"

#include <cmath>
#include <cstring>

#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace wwv {

namespace {

ggml_tensor* layer_norm(ggml_context* ctx0, ggml_tensor* x, ggml_tensor* w,
                        ggml_tensor* b, float eps) {
    ggml_tensor* cur = ggml_norm(ctx0, x, eps);
    if (w != nullptr) cur = ggml_mul(ctx0, cur, w);
    if (b != nullptr) cur = ggml_add(ctx0, cur, b);
    return cur;
}

// Full attention with vision m-rope, following the mtmd qwen3vl sequence.
ggml_tensor* vision_attention(ggml_context* ctx0, ggml_tensor* qkv_out,
                              ggml_tensor* o_w, ggml_tensor* o_b,
                              int64_t n_embd, int64_t n_head, int64_t n_pos,
                              float kq_scale, ggml_tensor* positions) {
    const int64_t d_head = n_embd / n_head;
    ggml_tensor* q =
        ggml_view_3d(ctx0, qkv_out, d_head, n_head, n_pos,
                     ggml_row_size(qkv_out->type, d_head), qkv_out->nb[1], 0);
    ggml_tensor* k = ggml_view_3d(
        ctx0, qkv_out, d_head, n_head, n_pos, ggml_row_size(qkv_out->type, d_head),
        qkv_out->nb[1], ggml_row_size(qkv_out->type, n_embd));
    ggml_tensor* v = ggml_view_3d(
        ctx0, qkv_out, d_head, n_head, n_pos, ggml_row_size(qkv_out->type, d_head),
        qkv_out->nb[1], ggml_row_size(qkv_out->type, 2 * n_embd));

    int32_t sections[4] = {static_cast<int32_t>(d_head / 4),
                           static_cast<int32_t>(d_head / 4),
                           static_cast<int32_t>(d_head / 4),
                           static_cast<int32_t>(d_head / 4)};
    q = ggml_rope_multi(ctx0, q, positions, nullptr, d_head / 2, sections,
                        GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
    k = ggml_rope_multi(ctx0, k, positions, nullptr, d_head / 2, sections,
                        GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

    // flash_attn_ext wants (hd, T, nh); the views above are (hd, nh, T).
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    ggml_tensor* kq =
        ggml_flash_attn_ext(ctx0, q, k, v, nullptr, kq_scale, 0.0f, 0.0f);
    kq = ggml_permute(ctx0, kq, 0, 2, 1, 3);
    ggml_tensor* flat = ggml_cont_2d(ctx0, kq, n_embd, n_pos);
    ggml_tensor* cur = ggml_mul_mat(ctx0, o_w, flat);
    if (o_b != nullptr) cur = ggml_add(ctx0, cur, o_b);
    return cur;
}

}  // namespace

std::vector<int32_t> vision_mrope_positions(int32_t pixel_w, int32_t pixel_h,
                                            int32_t patch_size,
                                            int32_t merge_size) {
    const int pw = pixel_w / patch_size;
    const int ph = pixel_h / patch_size;
    const int n_pos = pw * ph;
    std::vector<int32_t> positions(static_cast<size_t>(n_pos) * 4, 0);
    int ptr = 0;
    for (int y = 0; y < ph; y += merge_size) {
        for (int x = 0; x < pw; x += merge_size) {
            for (int dy = 0; dy < merge_size; ++dy) {
                for (int dx = 0; dx < merge_size; ++dx) {
                    positions[static_cast<size_t>(ptr)] = y + dy;
                    positions[static_cast<size_t>(n_pos) + ptr] = x + dx;
                    positions[static_cast<size_t>(2 * n_pos) + ptr] = y + dy;
                    positions[static_cast<size_t>(3 * n_pos) + ptr] = x + dx;
                    ++ptr;
                }
            }
        }
    }
    return positions;
}

void unshuffle_tokens_host(const float* tokens, int64_t n_embd, int64_t n_tokens,
                           int64_t map_w, int64_t map_h, int32_t merge_size,
                           float* out_map) {
    // tokens: (E, T), T = ((by*Wb + bx)*m + iy)*m + ix (by slowest, ix fastest)
    // out_map: (E, W', H') row-major, x fastest; W' = m*Wb, H' = m*Hb.
    const int64_t m = merge_size;
    const int64_t wb = map_w;      // blocks per row of the map
    const int64_t hb = map_h;
    const int64_t w = map_w * m;   // full map width (post-merge grid)
    for (int64_t c = 0; c < n_embd; ++c) {
        const float* src = tokens + static_cast<size_t>(c) * n_tokens;
        for (int64_t by = 0; by < hb; ++by) {
            for (int64_t bx = 0; bx < wb; ++bx) {
                for (int64_t iy = 0; iy < m; ++iy) {
                    for (int64_t ix = 0; ix < m; ++ix) {
                        const int64_t t =
                            ((by * wb + bx) * m + iy) * m + ix;
                        const int64_t dst = (by * m + iy) * w + bx * m + ix;
                        out_map[static_cast<size_t>(c) * (w * map_h * m) +
                                dst] = src[t];
                    }
                }
            }
        }
    }
}

ggml_cgraph* build_vision_taps_graph(
    ggml_context* ctx0, const VisionTapsParams& params,
    const std::vector<int32_t>& taps_layers,
    const std::vector<ggml_tensor*>& tap_leaves, ggml_tensor* pixels,
    ggml_tensor* positions, const std::vector<ggml_tensor*>& patch_embd,
    ggml_tensor* patch_bias, ggml_tensor* position_embd, ggml_tensor* pre_ln_w,
    ggml_tensor* pre_ln_b, const std::vector<int>& layer_indexes,
    const std::vector<std::vector<ggml_tensor*>>& block_weights,
    VisionTapsResult* out, std::string* err) {
    const int64_t p = params.patch_size;
    const int64_t m = params.merge_size;
    if (params.grid_w % (p * m) != 0 || params.grid_h % (p * m) != 0) {
        if (err != nullptr) {
            *err = "smart-resize grid not divisible by patch*merge";
        }
        return nullptr;
    }
    if (patch_embd.size() < 2 || patch_embd[0] == nullptr ||
        patch_embd[1] == nullptr) {
        if (err != nullptr) {
            *err = "qwen3vl requires the dual patch-embed convs";
        }
        return nullptr;
    }
    if (tap_leaves.size() != taps_layers.size()) {
        if (err != nullptr) *err = "tap leaf count mismatch";
        return nullptr;
    }
    const int64_t npx = params.grid_w / p;  // full patch grid
    const int64_t npy = params.grid_h / p;
    const int64_t n_embd = patch_embd[0]->ne[3];
    if (params.heads <= 0 || n_embd % params.heads != 0) {
        if (err != nullptr) *err = "invalid head count";
        return nullptr;
    }

    // Patch embed: the kt=2 temporal conv split into two 2D convs, summed,
    // then the 2x2 spatial merge (qwen3vl.cpp build_inp_with_temporal_merge
    // + spatial-merge block): [x, y, c, b] -> [c, x, y, b] -> merged tokens
    // in (by, bx, iy, ix) order.
    ggml_tensor* inp = ggml_add(
        ctx0,
        ggml_conv_2d(ctx0, patch_embd[0], pixels, static_cast<int>(p),
                     static_cast<int>(p), 0, 0, 1, 1),
        ggml_conv_2d(ctx0, patch_embd[1], pixels, static_cast<int>(p),
                     static_cast<int>(p), 0, 0, 1, 1));
    inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 2, 0, 3));
    inp = ggml_cont_4d(ctx0, inp, n_embd * 2, npx / 2, npy, 1);
    inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, npx / 2, 2, npy / 2);
    inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
    inp = ggml_cont_3d(ctx0, inp, n_embd, npx * npy, 1);
    if (patch_bias != nullptr) inp = ggml_add(ctx0, inp, patch_bias);

    // Learned position embedding, resized bilinear/align_corners to the
    // target patch grid (clip_graph::resize_position_embeddings), then the
    // same 2x2 merge shuffle so the token order matches the input.
    ggml_tensor* pos = position_embd;  // (C, PW*PH) flat token table
    const int64_t n_per_side =
        static_cast<int64_t>(std::sqrt(static_cast<double>(pos->ne[1])));
    if (n_per_side * n_per_side != pos->ne[1]) {
        if (err != nullptr) *err = "position embedding is not square";
        return nullptr;
    }
    if (n_per_side != npx || n_per_side != npy) {
        pos = ggml_reshape_3d(ctx0, pos, n_embd, n_per_side, n_per_side);
        pos = ggml_cont(ctx0, ggml_permute(ctx0, pos, 2, 0, 1, 3));
        pos = ggml_interpolate(ctx0, pos, npy, npx, n_embd, 1,
                               GGML_SCALE_MODE_BILINEAR |
                                   GGML_SCALE_FLAG_ALIGN_CORNERS);
        pos = ggml_permute(ctx0, pos, 1, 2, 0, 3);
        pos = ggml_cont_2d(ctx0, pos, n_embd, npx * npy);
    }
    {
        ggml_tensor* cur = ggml_cont_4d(ctx0, pos, n_embd * 2, npx / 2, npy, 1);
        cur = ggml_reshape_4d(ctx0, cur, n_embd * 2, npx / 2, 2, npy / 2);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_3d(ctx0, cur, n_embd, npx * npy, 1);
        inp = ggml_add(ctx0, inp, cur);
    }

    ggml_tensor* inp_l = inp;
    if (pre_ln_w != nullptr) {
        inp_l = layer_norm(ctx0, inp_l, pre_ln_w, pre_ln_b, params.eps);
    }

    const int64_t n_pos = npx * npy;
    const float kq_scale =
        1.0f / std::sqrt(static_cast<float>(n_embd / params.heads));

    // real-size graphs exceed the default 2048-node limit
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);
    const size_t n_blocks = block_weights.size();
    for (size_t il = 0; il < n_blocks; ++il) {
        const auto& w = block_weights[il];  // 14 slots, see the header.
        ggml_tensor* cur = inp_l;
        cur = layer_norm(ctx0, cur, w[0], w[1], params.eps);
        cur = ggml_mul_mat(ctx0, w[2], cur);
        if (w[3] != nullptr) cur = ggml_add(ctx0, cur, w[3]);
        cur = vision_attention(ctx0, cur, w[4], w[5], n_embd, params.heads,
                               n_pos, kq_scale, positions);
        cur = ggml_add(ctx0, cur, inp_l);
        inp_l = cur;
        cur = layer_norm(ctx0, cur, w[6], w[7], params.eps);
        // FFN: up (+ gate with SiLU? qwen3vl vision uses GELU gate per the
        // mtmd ffn_op), then down.
        ggml_tensor* up = ggml_mul_mat(ctx0, w[8], cur);
        if (w[9] != nullptr) up = ggml_add(ctx0, up, w[9]);
        if (w[10] != nullptr) {
            ggml_tensor* gate = ggml_mul_mat(ctx0, w[10], cur);
            if (w[11] != nullptr) gate = ggml_add(ctx0, gate, w[11]);
            gate = ggml_gelu(ctx0, gate);
            up = ggml_mul(ctx0, up, gate);
        } else {
            up = ggml_gelu(ctx0, up);
        }
        cur = ggml_mul_mat(ctx0, w[12], up);
        if (w[13] != nullptr) cur = ggml_add(ctx0, cur, w[13]);
        cur = ggml_add(ctx0, inp_l, cur);
        inp_l = cur;

        for (size_t ti = 0; ti < taps_layers.size(); ++ti) {
            if (taps_layers[ti] == static_cast<int32_t>(il)) {
                ggml_build_forward_expand(gf, cur);
                ggml_tensor* tap = ggml_cpy(ctx0, cur, tap_leaves[ti]);
                ggml_build_forward_expand(gf, tap);
            }
        }
    }

    out->taps.assign(tap_leaves.begin(), tap_leaves.end());
    out->n_tokens = n_pos;
    out->token_grid_w = npx;
    out->token_grid_h = npy;
    out->n_embd = n_embd;
    return gf;
}

}  // namespace wwv
