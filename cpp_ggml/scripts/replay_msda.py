#!/usr/bin/env python3
"""MSDA bisect: replay the torch cross-attention with dumped stage inputs.

Feeds torch dumps (tap_memory / tap_tfm_n1_0 / query position / boxes01) and
the f32 decoder GGUF weights through the exact msda.py formulas, verifies the
replay against the dumped tap_tfm_cr0, then writes the same intermediate
quantities the C++ msda_layer taps emit (msda_value / msda_offs / msda_px /
msda_py / msda_att / msda_acc / msda_agg) so compare can pinpoint the first
divergent op.

Slot convention (both sides): slot = h*(L*P) + l*P + p + HLP*q.
ggml flat == torch flat whenever ne == reversed(torch shape).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cpp_ggml" / "third_party" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

HEADS, LEVELS, POINTS = 8, 4, 4
D = 192


def load_stage(d: Path, name: str) -> np.ndarray:
    shape = tuple(int(x) for x in (d / f"{name}.shape").read_text().split(","))
    return np.frombuffer((d / f"{name}.torch.bin").read_bytes(),
                         dtype=np.float32).reshape(shape).copy()


def main() -> int:
    ref = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wwv-ref")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/wwv-msda-ref")
    gguf_path = REPO / "cpp_ggml" / "models" / "gguf" / "wevisone-mask-decoder-f32.gguf"
    out.mkdir(parents=True, exist_ok=True)

    memory = torch.from_numpy(load_stage(ref, "tap_memory"))          # (S_tok, 192)
    target = torch.from_numpy(load_stage(ref, "tap_tfm_n1_0"))        # (6, 192)
    position = torch.from_numpy(load_stage(ref, "tap_query_position"))  # (6, 192)
    boxes = torch.from_numpy(load_stage(ref, "boxes01"))              # (6, 4)
    cr0_ref = torch.from_numpy(load_stage(ref, "tap_tfm_cr0"))        # (6, 192)

    gr = gguf.GGUFReader(str(gguf_path))
    w = {}
    for t in gr.tensors:
        if ".cross_attn." in t.name:
            w[t.name] = torch.from_numpy(
                np.frombuffer(t.data if isinstance(t.data, np.ndarray)
                              else np.asarray(t.data), dtype=np.float32).copy()
            ).reshape(tuple(reversed(t.shape)))  # gguf ne -> torch shape
    base = "transformer.layers.0.cross_attn."

    b = 1
    query = (target + position)[None]            # (1, 6, 192)
    value = w[base + "value_proj.weight"] @ memory.T + w[base + "value_proj.bias"][:, None]
    # value (192, S_tok) == ggml msda_value flat
    value_torch = value.T[None]                  # (1, S_tok, 192)
    offsets = (w[base + "sampling_offsets.weight"] @ query[0].T
               + w[base + "sampling_offsets.bias"][:, None]).T  # (6, 256)
    att_raw = (w[base + "attention_weights.weight"] @ query[0].T
               + w[base + "attention_weights.bias"][:, None]).T  # (6, 128)

    offs_v = offsets.view(b, 6, HEADS, LEVELS, POINTS, 2)
    weights = att_raw.view(b, 6, HEADS, LEVELS * POINTS).softmax(-1).view(
        b, 6, HEADS, LEVELS, POINTS)

    # reference boxes: cxcywh per mask_decoder.py _boxes_xyxy_to_cxcywh
    bx = boxes.clamp(0.0, 1.0)
    x1, y1, x2, y2 = bx.unbind(-1)
    ref_pts = torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5,
                           (x2 - x1).clamp_min(1e-4),
                           (y2 - y1).clamp_min(1e-4)], dim=-1)
    ref4 = ref_pts[None, :, None].expand(b, -1, LEVELS, -1)  # (1, 6, 4, 4)

    locations = (ref4[:, :, None, :, None, :2]
                 + offs_v / float(POINTS) * ref4[:, :, None, :, None, 2:] * 0.5)
    # locations (1, 6, H, L, P, 2); grid = loc*2 - 1
    grids = locations.mul(2.0).sub(1.0)

    # same-scale levels: one (H, W) grid for every level (grid_hw dump)
    gh, gw_ = (int(v) for v in load_stage(ref, "grid_hw").reshape(-1))
    spatial_shapes = [(gh, gw_)] * LEVELS
    split_sizes = [int(h) * int(w) for h, w in spatial_shapes]

    value_v = value_torch.view(b, -1, HEADS, D // HEADS)
    value_levels = value_v.split(split_sizes, dim=1)
    sampled_per_level = []
    for level, (height, width) in enumerate(spatial_shapes):
        value_l = value_levels[level].permute(0, 2, 3, 1).reshape(
            b * HEADS, D // HEADS, int(height), int(width))
        grid_l = grids[:, :, :, level].permute(0, 2, 1, 3, 4).reshape(
            b * HEADS, 6, POINTS, 2)
        sampled = F.grid_sample(value_l.float(), grid_l.float(),
                                mode="bilinear", padding_mode="zeros",
                                align_corners=False)
        sampled_per_level.append(sampled)
    sampled = torch.stack(sampled_per_level, dim=-2)      # (8, 24, 6, 4, 4)
    sampled = sampled.flatten(-2)                          # (8, 24, 6, L*P)
    weights_p = weights.permute(0, 2, 1, 3, 4).reshape(
        b * HEADS, 1, 6, LEVELS * POINTS)
    agg = (sampled * weights_p.float()).sum(-1)            # (8, 24, 6)
    output = agg.reshape(b, HEADS * (D // HEADS), 6).transpose(1, 2)[0]
    res = (w[base + "output_proj.weight"] @ output.T
           + w[base + "output_proj.bias"][:, None]).T

    # slot-ordered intermediates (slot = h*LP + l*P + p + HLP*q).
    # locations (1, q, H, L, P, 2) -> [..., k].reshape(q, H, L, P).ravel()
    # gives flat = q*HLP + h*LP + l*P + p — the slot order itself.
    px = (locations[..., 0].reshape(6, HEADS, LEVELS, POINTS).reshape(-1)
          * 96.0 - 0.5)
    py = (locations[..., 1].reshape(6, HEADS, LEVELS, POINTS).reshape(-1)
          * 64.0 - 0.5)

    def save(name, t: torch.Tensor):
        t.detach().cpu().numpy().tofile(out / f"{name}.torch.bin")
        (out / f"{name}.shape").write_text(
            ",".join(str(x) for x in t.shape))

    save("msda_value", value.T)          # (S_tok, 192) == ggml (192, S_tok)
    save("msda_offs", offsets)           # (6, 256) == ggml (256, 6)
    save("msda_px", px)                  # (768,)
    save("msda_py", py)
    save("msda_locx", locations[..., 0].reshape(6, HEADS, LEVELS, POINTS)
         .reshape(-1))                   # loc_x before the W scale
    refw_slots = (ref4[0, :, :, 2][:, None, :, None]
                  .expand(6, HEADS, LEVELS, POINTS).reshape(-1))
    save("msda_refw", refw_slots)        # (768,)
    # gather-side intermediates (slot order): floor corner index, combined
    # slot weight (att * bilinear * zeros-mask), and the level-0 gather.
    att_slot_t = weights.view(1, 6, HEADS, LEVELS * POINTS).reshape(-1)
    ix_t = torch.floor(px); iy_t = torch.floor(py)
    fx_t = px - ix_t; fy_t = py - iy_t

    def corner_mask(xv, yv):
        mx = ((xv + 0.5) > 0) & ((96.0 - 0.5 - xv) > 0)
        my = ((yv + 0.5) > 0) & ((64.0 - 0.5 - yv) > 0)
        return (mx & my).float()
    idx_nw_t = (iy_t.clamp(0, 63) * 96 + ix_t.clamp(0, 95))
    w_nw_s_t = (att_slot_t * (1 - fx_t) * (1 - fy_t) * corner_mask(ix_t, iy_t))
    save("msda_idxw", idx_nw_t.float())          # (768,)
    save("msda_wnw", w_nw_s_t)                   # (768,)
    save("msda_g0", value.T[:6144][idx_nw_t.long()])  # (768, 192) == ggml (192, 768)
    save("msda_idxne", ((iy_t.clamp(0, 63) * 96
                         + (ix_t + 1).clamp(0, 95)).float()))
    save("msda_idxsw", (((iy_t + 1).clamp(0, 63) * 96
                         + ix_t.clamp(0, 95)).float()))
    save("msda_idxse", (((iy_t + 1).clamp(0, 63) * 96
                         + (ix_t + 1).clamp(0, 95)).float()))
    w_ne_s_t = att_slot_t * fx_t * (1 - fy_t) * corner_mask(ix_t + 1, iy_t)
    w_sw_s_t = att_slot_t * (1 - fx_t) * fy_t * corner_mask(ix_t, iy_t + 1)
    w_se_s_t = att_slot_t * fx_t * fy_t * corner_mask(ix_t + 1, iy_t + 1)
    save("msda_wne", w_ne_s_t)
    save("msda_wsw", w_sw_s_t)
    save("msda_wse", w_se_s_t)
    vl0 = value.T[:6144]                          # (6144, 192) level-0 value
    for l in range(LEVELS):
        vll = value.T[l * 6144:(l + 1) * 6144]
        lvl_ref = (vll[idx_nw_t.long()] * w_nw_s_t[:, None]
                   + vll[(iy_t.clamp(0, 63) * 96 + (ix_t + 1).clamp(0, 95)).long()]
                   * w_ne_s_t[:, None]
                   + vll[((iy_t + 1).clamp(0, 63) * 96 + ix_t.clamp(0, 95)).long()]
                   * w_sw_s_t[:, None]
                   + vll[((iy_t + 1).clamp(0, 63) * 96 + (ix_t + 1).clamp(0, 95)).long()]
                   * w_se_s_t[:, None])
        save(f"msda_lvl{l}", lvl_ref)             # (768, 192)
    # msda_att: ggml att_p ne = (16, 8, 6), flat = pl + 16*h + 128*q — the
    # torch-equal order is (q, h, lp) via the reversed-shape identity.
    save("msda_att", weights.view(1, 6, HEADS, LEVELS * POINTS).reshape(-1))
    # msda_acc: ggml acc (192, 768) with slot = f + 128*q, f = h*16 + l*4 + p
    # and the d axis per slot head-major. acc is the PER-SLOT weighted sample
    # (bilinear corners summed per level, then over levels) — i.e. sampled
    # collapsed over (l, p) per slot, NOT the fully-summed agg.
    acc_slots = torch.zeros(6 * HEADS * LEVELS * POINTS, D)
    for hh in range(HEADS):
        for l in range(LEVELS):
            for p in range(POINTS):
                for q in range(6):
                    slot = hh * 16 + l * 4 + p + 128 * q
                    acc_slots[slot, hh * 24:(hh + 1) * 24] = (
                        sampled[hh, :, q, l * POINTS + p]
                        * weights[0, q, hh, l, p])
    save("msda_acc", acc_slots)          # (768, 192) == ggml (192, 768)
    save("msda_agg", output)             # (6, 192) == ggml (192, 6)
    save("msda_cr0", res)

    err = (res - cr0_ref).abs().max().item()
    print(f"replay vs tap_tfm_cr0: max_abs={err:.3e} "
          f"{'OK' if err < 5e-3 else 'MISMATCH'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
