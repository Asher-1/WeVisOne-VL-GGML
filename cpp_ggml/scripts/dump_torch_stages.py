#!/usr/bin/env python3
"""Dump torch reference tensors for the C++ parity gates.

Stages (matching dump sites in the C++ graphs):
  vit_l{5,11,17,23}  pre-merger ViT block outputs (merger-input order)
  t_seg              LM hidden states at <OBJ> positions (teacher-forced)
  enc_l{0..3}        SameScaleMSDAEncoder projected maps
  memory             encoder memory (4*Hg*Wg, 192)
  queries            DINOMaskDecoder output
  pixel_feature      AdditivePixelFusion output (64, S/2, S/2)
  coarse             DynamicMaskPredictor logits (Q, S/2, S/2)
  logits_s2/s1       alpha cascade outputs

Usage:
  python3 cpp_ggml/scripts/dump_torch_stages.py \
      --model tencent/WeVisOne-VL-4B --image demo/examples/images/sofa.jpg \
      --task segmentation --query "person" --out /tmp/wwv-ref
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def save(out: Path, name: str, t: torch.Tensor | np.ndarray) -> None:
    arr = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else t
    (out / f"{name}.bin").write_bytes(np.ascontiguousarray(arr).tobytes())
    (out / f"{name}.shape").write_text(",".join(map(str, arr.shape)))
    print(f"  {name}: {tuple(arr.shape)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tencent/WeVisOne-VL-2B")
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", default="matting")
    ap.add_argument("--query", default="person")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cnn-size", type=int, default=0,
                    help="override decoder cnn_image_size (parity gate at "
                         "reduced S; semantics are size-invariant)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import numpy as np
    import torch
    from PIL import Image
    from wevisone_vl import feature_bridge
    from wevisone_vl.engine import WeVisOneVL, _pil_to_float_rgb
    from wevisone_vl.prompts import build_prompt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, t):
        arr = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=np.float32)
        (out / f"{name}.torch.bin").write_bytes(np.ascontiguousarray(arr).tobytes())
        (out / f"{name}.shape").write_text(",".join(map(str, arr.shape)))
        print(f"  {name}: {tuple(arr.shape)}")

    engine = WeVisOneVL(args.model, device="cpu", dtype="float32")
    if args.cnn_size:
        engine.cnn_size = args.cnn_size
    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.task, args.query, image_size=image.size)
    save("prompt_txt", np.frombuffer(prompt.encode(), dtype=np.uint8))

    # ---- stage 1: generation ----
    response = engine.generate_text(image, prompt)
    save("response_txt", np.frombuffer(response.encode(), dtype=np.uint8))
    print(f"response: {response[:120]}")

    # ---- stage 2: teacher-forced decoder pass with captures ----
    from wevisone_vl.engine import BOX_RE, COORD_RE, SEG_RE  # noqa
    boxes = []
    for body in BOX_RE.findall(response):
        for prefix in SEG_RE.split(body)[:-1]:
            nums = [int(v) for v in COORD_RE.findall(prefix)]
            if len(nums) >= 4:
                x1, y1, x2, y2 = nums[-4:]
                if all(0 <= v <= 1000 for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                    boxes.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    if not boxes:
        print("no <OBJ> boxes in the response — nothing to dump")
        return 1

    messages = engine._messages(image, prompt) + [
        {"role": "assistant", "content": response}]
    teacher = engine._inputs(messages, image, generation=False)
    decoder_image = image.convert("RGB").resize(
        (engine.cnn_size, engine.cnn_size), resample=Image.Resampling.BICUBIC)
    cnn_image = feature_bridge.assert_cnn_rgb(
        _pil_to_float_rgb(decoder_image).to(engine.device, dtype=torch.float32),
        cnn_image_size=engine.cnn_size)

    old_use_cache = getattr(engine.model.config, "use_cache", True)
    engine.model.config.use_cache = False
    try:
        with feature_bridge.VisualFeatureCapture(
                engine.model, feature_layers=engine.cfg.vision_feature_layers
        ) as visual_capture:
            with feature_bridge.LastHiddenCapture(engine.model) as hidden_capture:
                engine.model(**teacher, output_hidden_states=False,
                             return_dict=True, use_cache=False)
    finally:
        engine.model.config.use_cache = old_use_cache

    from wevisone_vl.engine import extract_t_seg, premerge_grid_hw
    hidden = hidden_capture.hidden
    t_seg = feature_bridge.extract_t_seg(hidden, teacher["input_ids"],
                                         engine.seg_token_id)[0]
    grid_thw = teacher["image_grid_thw"]
    grid_hw = feature_bridge.premerge_grid_hw(grid_thw)[0]
    save("t_seg", t_seg)
    save("boxes01", torch.tensor(boxes, dtype=torch.float32))
    save("cnn_image", cnn_image[0])
    save("grid_hw", torch.tensor(grid_hw, dtype=torch.float32))
    save("merge_size", torch.tensor(engine.spatial_merge_size))

    # raw pre-merger tokens in merger-input order (matches the C++ taps)
    for idx in engine.cfg.vision_feature_layers:
        feat = visual_capture.premerge_features.get(int(idx))
        save(f"vit_tap{idx}", feat[0])

    visual_maps = feature_bridge.per_sample_premerge_feature_maps(
        visual_capture.premerge_features, engine.cfg.vision_feature_layers,
        grid_thw, engine.spatial_merge_size)[0]
    for li, m in enumerate(visual_maps):
        save(f"level{li}", m[0])  # (C, H, W) row-major == C++ level layout

    # ---- stage 3: decoder forward ----
    boxes_t = torch.tensor(boxes, device=engine.device, dtype=torch.float32)
    t_seg_f = t_seg[:len(boxes)].to(engine.device, dtype=torch.float32)
    visual_maps = [m.to(engine.device, dtype=torch.float32) for m in visual_maps]
    engine.mask_decoder.to(device=engine.device, dtype=torch.float32).eval()
    # stage-boundary hooks for the C++ bisect (names match the C++ taps)
    tap_out = {}
    def hook(name):
        def fn(mod, inp, outp):
            tap_out[name] = outp
        return fn
    dec = engine.mask_decoder
    handles = [dec.visual_encoder.register_forward_hook(hook("memory"))]
    for i, proj in enumerate(dec.visual_encoder.input_proj):
        handles.append(proj.register_forward_hook(hook(f"enc_proj{i}")))
    handles.append(dec.query_builder.register_forward_hook(hook("query_builder")))
    for i, layer in enumerate(dec.transformer.layers):
        handles.append(layer.register_forward_hook(hook(f"tfm{i}")))
    handles.append(dec.pixel_fusion.register_forward_hook(hook("pixel_fusion")))
    def enc_hook(module, inp, outp):
        tap_out["enc_in_sum"] = inp[0] + inp[1]
    handles.append(dec.visual_encoder.encoder.register_forward_hook(enc_hook))
    l0 = dec.visual_encoder.encoder.layers[0]
    handles.append(l0.depthwise.register_forward_hook(hook("enc_dw0")))
    handles.append(l0.pointwise.register_forward_hook(hook("enc_pw0")))
    handles.append(l0.norm1.register_forward_hook(hook("enc_n1_0")))
    handles.append(l0.norm2.register_forward_hook(hook("enc_n2_0")))
    dl0 = dec.transformer.layers[0] if hasattr(dec, "transformer") else None
    if dl0 is not None:
        handles.append(dl0.self_attn.register_forward_hook(hook("tfm_sa0")))
        handles.append(dl0.norm1.register_forward_hook(hook("tfm_n1_0")))
        handles.append(dl0.cross_attn.register_forward_hook(hook("tfm_cr0")))
        handles.append(dl0.norm2.register_forward_hook(hook("tfm_n2_0")))
    handles.append(dec.mask_predictor.kernel.register_forward_hook(
        hook("kernel_params")))
    with torch.inference_mode():
        dout = engine.mask_decoder(
            image=cnn_image, visual_features_2d=visual_maps, t_seg=t_seg_f,
            boxes01=boxes_t[:len(boxes)], grid_hw=grid_hw)
    for h in handles:
        h.remove()
    if "enc_in_sum" in tap_out:
        save("tap_enc_in_sum", tap_out["enc_in_sum"])
    for k in ("enc_dw0", "enc_pw0", "enc_n1_0", "enc_n2_0",
              "tfm_sa0", "tfm_n1_0", "tfm_cr0", "tfm_n2_0"):
        if k in tap_out:
            v = tap_out[k]
            if isinstance(v, tuple):
                v = v[0]
            save(f"tap_{k}", v)
    if isinstance(tap_out.get("memory"), tuple):
        save("tap_memory", tap_out["memory"][1][0])
    else:
        save("tap_memory", tap_out["memory"])
    if isinstance(tap_out.get("query_builder"), tuple):
        save("tap_query_content", tap_out["query_builder"][0])
        save("tap_query_position", tap_out["query_builder"][1])
    else:
        save("tap_query_content", tap_out["query_builder"])
    for i in range(len(dec.transformer.layers)):
        t = tap_out[f"tfm{i}"]
        save(f"tap_tfm{i}", t[0] if isinstance(t, tuple) else t)
    for i, proj in enumerate(dec.visual_encoder.input_proj):
        t = tap_out[f"enc_proj{i}"]
        save(f"tap_enc_proj{i}", t[0] if isinstance(t, tuple) else t)
    pf = tap_out["pixel_fusion"]
    save("tap_pixel_fusion", pf[0] if isinstance(pf, tuple) else pf)
    save("tap_kernel_params", tap_out["kernel_params"])
    save("coarse", dout["mask_logits_coarse"])
    save("logits_s2", dout["mask_logits_aux"])
    save("logits_s1", dout["mask_logits"])
    print("dump complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
