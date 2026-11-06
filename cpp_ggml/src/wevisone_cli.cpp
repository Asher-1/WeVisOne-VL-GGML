// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Full-chain CLI: wevisone-cli run --image x.jpg --task detection
//                    --query "sofa" [--device cuda] [--precision f16]
//                    [--models-dir DIR] [--out result.png] [--json out.json]
// Also: `wevisone-cli info` prints the compiled backend surface.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstring>
#include <string>

#include "image_prep.hpp"
#include "pipeline_common.hpp"
#include "render.hpp"
#include "wevisone/wevisone_capi.h"
#include "wevisone_runtime.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <thread>
#include "gguf_loader.hpp"
#include "mask_decoder_graph.hpp"

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: wevisone-cli run --image IMG --task TASK [--query Q]\n"
                 "       [--device cuda|vulkan|cpu] [--precision f16|q8_0|q4_K|f32]\n"
                 "       [--decoder-precision f32|f16|q8_0] [--models-dir DIR]\n"
                 "       [--max-new-tokens N] [--threads N] [--out PNG] [--json JSON]\n"
                 "       wevisone-cli info\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "selftest") {
        // Decoder-graph execution test with a synthetic GGUF
        // (make_synth_decoder_gguf.py): proves shapes, allocator and
        // compute run end-to-end without the released checkpoint.
        if (argc < 3) {
            std::fprintf(stderr,
                "usage: wevisone-cli selftest <decoder.gguf> [--vit N] [--llm N]"
                " [--grid N] [--q N]\n");
            return 1;
        }
        int32_t vit_arg = 64;
        int32_t llm_arg = 64;
        int32_t grid_arg = 8;
        int32_t q_arg = 3;
        int32_t cnn_arg = 0;  // 0 = keep the GGUF cfg size
        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            const auto nx = [&]() { return i + 1 < argc ? std::atoi(argv[++i]) : 0; };
            if (a == "--vit") vit_arg = nx();
            else if (a == "--llm") llm_arg = nx();
            else if (a == "--grid") grid_arg = nx();
            else if (a == "--q") q_arg = nx();
            else if (a == "--cnn") cnn_arg = nx();
        }
        wwv::GgufFile gf;
        std::string err;
        if (!gf.load(argv[2], &err)) {
            std::fprintf(stderr, "load: %s\n", err.c_str());
            return 1;
        }
        wwv::DecoderConfig cfg;
        cfg.cnn_image_size = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.cnn_image_size", 128));
        cfg.hidden_dim = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.hidden_dim", 32));
        cfg.mask_feature_dim = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.mask_feature_dim", 16));
        cfg.num_fourier_freqs = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.num_fourier_freqs", 8));
        cfg.msda_encoder_layers = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.msda_encoder_layers", 2));
        cfg.transformer_layers = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.transformer_layers", 2));
        cfg.transformer_heads = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.transformer_heads", 4));
        cfg.transformer_ffn_dim = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.transformer_ffn_dim", 64));
        cfg.group_norm_groups = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.group_norm_groups", 8));
        cfg.stem_mid_dim = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.stem_mid_dim", 16));
        cfg.alpha_cascade_dim = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.alpha_cascade_dim", 8));
        cfg.alpha_cascade_blocks = static_cast<int32_t>(gf.i32("wevisone.mask_decoder.alpha_cascade_blocks", 1));

        const int32_t q = q_arg;
        if (cnn_arg > 0) cfg.cnn_image_size = cnn_arg;
        const int64_t S = cfg.cnn_image_size;
        const int32_t grid = grid_arg;
        const int32_t vit_c = vit_arg;
        const int32_t llm = llm_arg;

        ggml_init_params ip{(8192ull + 256ull * q) << 20, nullptr, false};
        ggml_context* ctx = ggml_init(ip);
        auto rnd = [&](ggml_tensor* t) {
            float* d = static_cast<float*>(t->data);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) {
                d[i] = static_cast<float>((i * 2654435761u % 1000) / 100000.0f - 0.005f);
            }
            return t;
        };
        // CPU selftest: the context owns host memory (no_alloc=false), so
        // every tensor writes its bytes directly — no staging, no gallocr.
        wwv::MaskDecoderInputs din;
        auto rnd_fill = [&](ggml_tensor* t) {
            float* d = static_cast<float*>(t->data);
            for (int64_t i = 0; i < ggml_nelements(t); ++i) {
                d[i] = static_cast<float>((i * 2654435761u % 1000) / 100000.0f - 0.005f);
            }
            return t;
        };
        din.cnn_image = rnd_fill(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, S, 3));
        for (int l = 0; l < 4; ++l) {
            din.levels[l] = rnd_fill(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, grid, grid, vit_c));
        }
        din.t_seg = rnd_fill(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, llm, q));
        din.boxes01 = rnd_fill(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, q));

        auto wt = [&](const char* name) -> ggml_tensor* {
            const auto* ti = gf.tensor(name);
            if (ti == nullptr) return nullptr;
            const float* f = gf.tensor_f32(name);
            if (f == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor(
                ctx, GGML_TYPE_F32, ti->n_dims,
                reinterpret_cast<const int64_t*>(ti->ne));
            std::memcpy(t->data, f,
                        sizeof(float) * static_cast<size_t>(ggml_nelements(t)));
            return t;
        };
        auto wts = [&](const std::string& n) { return wt(n.c_str()); };
        wwv::MaskDecoderWeights W;
        for (int l = 0; l < 4; ++l) {
            W.input_proj_w[l] = wts("visual_encoder.input_proj." + std::to_string(l) + ".0.weight");
            W.input_proj_gn_w[l] = wts("visual_encoder.input_proj." + std::to_string(l) + ".1.weight");
            W.input_proj_gn_b[l] = wts("visual_encoder.input_proj." + std::to_string(l) + ".1.bias");
        }
        W.level_embed = wts("visual_encoder.level_embed");
        for (int li = 0; li < cfg.msda_encoder_layers; ++li) {
            const std::string ps =
                "visual_encoder.encoder.layers." + std::to_string(li) + ".";
            std::array<ggml_tensor*, 10> lw{};
            lw[0] = wts(ps + "depthwise.weight");
            lw[1] = wts(ps + "pointwise.weight");
            lw[2] = wts(ps + "norm1.weight");
            lw[3] = wts(ps + "norm1.bias");
            lw[4] = wts(ps + "linear1.weight");
            lw[5] = wts(ps + "linear1.bias");
            lw[6] = wts(ps + "linear2.weight");
            lw[7] = wts(ps + "linear2.bias");
            lw[8] = wts(ps + "norm2.weight");
            lw[9] = wts(ps + "norm2.bias");
            W.encoder_layers.push_back(lw);
        }
        W.content_proj_w = wts("query_builder.content_proj.weight");
        W.content_proj_b = wts("query_builder.content_proj.bias");
        W.content_norm_w = wts("query_builder.content_norm.weight");
        W.content_norm_b = wts("query_builder.content_norm.bias");
        W.box_mlp_w1 = wts("query_builder.box_position.mlp.0.weight");
        W.box_mlp_b1 = wts("query_builder.box_position.mlp.0.bias");
        W.box_mlp_w2 = wts("query_builder.box_position.mlp.2.weight");
        W.box_mlp_b2 = wts("query_builder.box_position.mlp.2.bias");
        W.position_norm_w = wts("query_builder.position_norm.weight");
        W.position_norm_b = wts("query_builder.position_norm.bias");
        {
            ggml_tensor* bf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.num_fourier_freqs, 1);
            float* d = static_cast<float*>(bf->data);
            for (int i = 0; i < cfg.num_fourier_freqs; ++i) d[i] = std::pow(2.0f, (float)i);
            W.box_freqs = bf;
        }
        for (int li = 0; li < cfg.transformer_layers; ++li) {
            const std::string ps = "transformer.layers." + std::to_string(li) + ".";
            std::array<ggml_tensor*, 20> tw{};
            tw[0] = wts(ps + "self_attn.in_proj_weight");
            tw[1] = wts(ps + "self_attn.in_proj_bias");
            tw[2] = wts(ps + "self_attn.out_proj.weight");
            tw[3] = wts(ps + "self_attn.out_proj.bias");
            tw[4] = wts(ps + "norm1.weight");
            tw[5] = wts(ps + "norm1.bias");
            tw[6] = wts(ps + "cross_attn.value_proj.weight");
            tw[7] = wts(ps + "cross_attn.value_proj.bias");
            tw[8] = wts(ps + "cross_attn.sampling_offsets.weight");
            tw[9] = wts(ps + "cross_attn.sampling_offsets.bias");
            tw[10] = wts(ps + "cross_attn.attention_weights.weight");
            tw[11] = wts(ps + "cross_attn.attention_weights.bias");
            tw[12] = wts(ps + "cross_attn.output_proj.weight");
            tw[13] = wts(ps + "cross_attn.output_proj.bias");
            tw[14] = wts(ps + "norm2.weight");
            tw[15] = wts(ps + "norm2.bias");
            tw[16] = wts(ps + "linear1.weight");
            tw[17] = wts(ps + "linear1.bias");
            tw[18] = wts(ps + "linear2.weight");
            tw[19] = wts(ps + "linear2.bias");
            W.transformer_layers.push_back(tw);
            W.transformer_norms.emplace_back(wts(ps + "norm3.weight"),
                                             wts(ps + "norm3.bias"));
        }
        for (int l = 0; l < 4; ++l) {
            W.fpn_lateral_w[l] = wts("pixel_fusion.fpn.lateral." + std::to_string(l) + ".weight");
        }
        for (int i = 0; i < 3; ++i) {
            const std::string ps = "pixel_fusion.fpn.output." + std::to_string(i) + ".";
            std::array<ggml_tensor*, 6> ow{};
            ow[0] = wts(ps + "0.weight");
            ow[1] = wts(ps + "1.weight");
            ow[2] = wts(ps + "1.bias");
            W.fpn_output.push_back(ow);
        }
        W.up_conv0_w = wts("pixel_fusion.upsample.0.conv.weight");
        W.up_conv0_b = wts("pixel_fusion.upsample.0.conv.bias");
        W.up_conv1_w = wts("pixel_fusion.upsample.1.conv.weight");
        W.up_conv1_b = wts("pixel_fusion.upsample.1.conv.bias");
        W.s2_up_conv_w = wts("pixel_fusion.s2_upsample.conv.weight");
        W.s2_up_conv_b = wts("pixel_fusion.s2_upsample.conv.bias");
        W.stem_conv0_w = wts("pixel_fusion.cnn.net.0.weight");
        W.stem_gn0_w = wts("pixel_fusion.cnn.net.1.weight");
        W.stem_gn0_b = wts("pixel_fusion.cnn.net.1.bias");
        W.stem_conv1_w = wts("pixel_fusion.cnn.net.3.weight");
        W.stem_gn1_w = wts("pixel_fusion.cnn.net.4.weight");
        W.stem_gn1_b = wts("pixel_fusion.cnn.net.4.bias");
        W.fpn_proj_w = wts("pixel_fusion.fpn_proj.weight");
        W.cnn_proj_w = wts("pixel_fusion.cnn_proj.weight");
        W.smooth_gn_w = wts("pixel_fusion.smooth.0.weight");
        W.smooth_gn_b = wts("pixel_fusion.smooth.0.bias");
        W.smooth_conv_w = wts("pixel_fusion.smooth.2.weight");
        W.kernel_w1 = wts("mask_predictor.kernel.0.weight");
        W.kernel_b1 = wts("mask_predictor.kernel.0.bias");
        W.kernel_w2 = wts("mask_predictor.kernel.2.weight");
        W.kernel_b2 = wts("mask_predictor.kernel.2.bias");
        const std::string hr = "highres_refine.";
        W.cascade_semantic_w = wts(hr + "semantic.0.weight");
        W.cascade_semantic_gn_w = wts(hr + "semantic.1.weight");
        W.cascade_semantic_gn_b = wts(hr + "semantic.1.bias");
        W.rgb_s4_w = wts(hr + "rgb_s4.0.weight");
        W.rgb_s4_gn_w = wts(hr + "rgb_s4.1.weight");
        W.rgb_s4_gn_b = wts(hr + "rgb_s4.1.bias");
        W.rgb_s2_w = wts(hr + "rgb_s2.0.weight");
        W.rgb_s2_gn_w = wts(hr + "rgb_s2.1.weight");
        W.rgb_s2_gn_b = wts(hr + "rgb_s2.1.bias");
        W.rgb_s1_w = wts(hr + "rgb_s1.0.weight");
        W.rgb_s1_gn_w = wts(hr + "rgb_s1.1.weight");
        W.rgb_s1_gn_b = wts(hr + "rgb_s1.1.bias");
        W.coarse_s4_w = wts(hr + "coarse_s4.0.weight");
        W.coarse_s4_gn_w = wts(hr + "coarse_s4.1.weight");
        W.coarse_s4_gn_b = wts(hr + "coarse_s4.1.bias");
        W.coarse_s2_w = wts(hr + "coarse_s2.0.weight");
        W.coarse_s2_gn_w = wts(hr + "coarse_s2.1.weight");
        W.coarse_s2_gn_b = wts(hr + "coarse_s2.1.bias");
        W.previous_s2_w = wts(hr + "previous_s2.0.weight");
        W.previous_s2_gn_w = wts(hr + "previous_s2.1.weight");
        W.previous_s2_gn_b = wts(hr + "previous_s2.1.bias");
        W.previous_s1_w = wts(hr + "previous_s1.0.weight");
        W.previous_s1_gn_w = wts(hr + "previous_s1.1.weight");
        W.previous_s1_gn_b = wts(hr + "previous_s1.1.bias");
        W.fuse_s4_w = wts(hr + "fuse_s4.weight");
        W.fuse_s2_w = wts(hr + "fuse_s2.weight");
        W.fuse_s1_w = wts(hr + "fuse_s1.weight");
        W.up_s4_s2_w = wts(hr + "up_s4_to_s2.weight");
        W.up_s2_s1_w = wts(hr + "up_s2_to_s1.weight");
        W.query_s4_w = wts(hr + "query_s4.weight");
        W.query_s4_b = wts(hr + "query_s4.bias");
        W.query_s2_w = wts(hr + "query_s2.weight");
        W.query_s2_b = wts(hr + "query_s2.bias");
        W.query_s1_w = wts(hr + "query_s1.weight");
        W.query_s1_b = wts(hr + "query_s1.bias");
        W.output_s4_w = wts(hr + "output_s4.weight");
        W.output_s4_b = wts(hr + "output_s4.bias");
        W.output_s2_w = wts(hr + "output_s2.weight");
        W.output_s2_b = wts(hr + "output_s2.bias");
        W.output_s1_w = wts(hr + "output_s1.weight");
        W.output_s1_b = wts(hr + "output_s1.bias");
        const char* stacks[3] = {"blocks_s4", "blocks_s2", "blocks_s1"};
        for (int st = 0; st < 3; ++st) {
            std::array<ggml_tensor*, 12> stack{};
            for (int b = 0; b < cfg.alpha_cascade_blocks; ++b) {
                const std::string ps = hr + stacks[st] + "." + std::to_string(b) + ".net.";
                stack[static_cast<size_t>(b) * 6 + 0] = wts(ps + "0.weight");
                stack[static_cast<size_t>(b) * 6 + 1] = wts(ps + "0.bias");
                stack[static_cast<size_t>(b) * 6 + 2] = wts(ps + "2.weight");
                stack[static_cast<size_t>(b) * 6 + 3] = wts(ps + "3.weight");
                stack[static_cast<size_t>(b) * 6 + 4] = wts(ps + "3.bias");
                stack[static_cast<size_t>(b) * 6 + 5] = wts(ps + "5.weight");
            }
            W.cascade_blocks.push_back(stack);
        }
        // host sine pos leaf
        {
            W.encoder_pos_leaf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hidden_dim, 4 * grid * grid);
            float* d = static_cast<float*>(W.encoder_pos_leaf->data);
            for (int64_t i = 0; i < ggml_nelements(W.encoder_pos_leaf); ++i) {
                d[i] = static_cast<float>((i % 97) / 97.0f) * 0.01f;
            }
        }

        wwv::MaskDecoderOutputs dout;
        ggml_cgraph* dg = wwv::build_mask_decoder_graph(
            ctx, cfg, q, grid, grid, vit_c, llm, W, din, &dout, &err,
            nullptr, nullptr);
        if (dg == nullptr) {
            std::fprintf(stderr, "graph build failed: %s\n", err.c_str());
            return 1;
        }
        // single upload pass over the staged host buffers + the builder's
        // f32 constant leaves
        const auto t0 = std::chrono::steady_clock::now();
        ggml_cplan cplan = ggml_graph_plan(dg, static_cast<int>(std::thread::hardware_concurrency()), nullptr);
        std::vector<uint8_t> plan_buf(cplan.work_size);
        cplan.work_data = plan_buf.data();
        ggml_status st = ggml_graph_compute(dg, &cplan);
        const auto t1c = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph compute failed\n");
            return 1;
        }
        std::vector<float> out(static_cast<size_t>(S) * S);
        std::memcpy(out.data(), dout.mask_logits_s1->data, out.size() * 4);
        double lo = 1e30;
        double hi = -1e30;
        double sum = 0;
        for (float v : out) {
            lo = std::min<double>(lo, v);
            hi = std::max<double>(hi, v);
            sum += v;
        }
        const double ms = std::chrono::duration<double, std::milli>(t1c - t0).count();
        std::printf("SELFTEST PASS: q=%d grid=%dx%d S=%lld logits[min=%.4f max=%.4f mean=%.5f] %.1f ms\n",
                    q, grid, grid, (long long)S, lo, hi, sum / out.size(), ms);
        ggml_free(ctx);
        return 0;
    }
    if (cmd == "info") {
        std::printf("wevisone-cli (WeVisOne-VL ggml port)\n");
        std::printf("abi: %d\n", wevisone_abi_version());
        std::printf("built for exactly one GPU backend (see CMakeLists.txt)\n");
        return 0;
    }
    if (cmd != "run") {
        usage();
        return 1;
    }
    wwv::RuntimeOptions opts;
    std::string image_path;
    std::string task = "detection";
    std::string query;
    std::string out_png;
    std::string out_json;
    for (int i = 2; i < argc; ++i) {
        const auto next = [&]() -> std::string {
            return i + 1 < argc ? argv[++i] : "";
        };
        const std::string a = argv[i];
        if (a == "--image") image_path = next();
        else if (a == "--task") task = next();
        else if (a == "--query") query = next();
        else if (a == "--device") opts.device = next();
        else if (a == "--precision") opts.precision = next();
        else if (a == "--decoder-precision") opts.decoder_precision = next();
        else if (a == "--models-dir") opts.models_dir = next();
        else if (a == "--vlm-gguf") opts.vlm_gguf = next();
        else if (a == "--mmproj-gguf") opts.mmproj_gguf = next();
        else if (a == "--decoder-gguf") opts.decoder_gguf = next();
        else if (a == "--max-new-tokens") opts.max_new_tokens = std::atoi(next().c_str());
        else if (a == "--threads") opts.threads = std::atoi(next().c_str());
        else if (a == "--out") out_png = next();
        else if (a == "--json") out_json = next();
        else if (a == "--dump-dir") opts.dump_dir = next();
        else if (a == "--no-taps") opts.no_taps = true;
        else if (a == "--taps-filter") opts.taps_filter = next();
        else if (a == "--stages-in") opts.stages_dir = next();
        else if (a == "--stages-in") opts.stages_dir = next();
        else { usage(); return 1; }
    }
    if (image_path.empty()) {
        usage();
        return 1;
    }

    std::string err;
    auto rt = wwv::WevisoneRuntime::create(opts, &err);
    if (rt == nullptr) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }

    wwv::RgbImage rgb;
    if (!wwv::load_path_rgb(image_path, &rgb, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    wwv_image_view view{};
    view.data = rgb.pixels.data();
    view.width = rgb.width;
    view.height = rgb.height;
    view.row_stride_bytes = static_cast<size_t>(rgb.width) * 3;
    view.format = WWV_IMAGE_RGB8;

    wwv::RunResult result;
    if (!rt->run_image(view, task, query, nullptr, 0, &result)) {
        std::fprintf(stderr, "run failed: %s\n", rt->last_error());
        return 1;
    }
    std::printf("answer: %s\n", result.answer.c_str());
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const wwv::ResultBox& b = result.boxes[i];
        std::printf("box[%zu]: %.1f %.1f %.1f %.1f\n", i, b.x1, b.y1, b.x2, b.y2);
    }
    std::printf("timings: pre %.1fms infer %.1fms post %.1fms e2e %.1fms\n",
                result.timings.preprocess_ms, result.timings.inference_ms,
                result.timings.postprocess_ms, result.timings.e2e_ms);
    if (!out_png.empty()) {
        const bool ok = wwv::render_result_png(
            view, reinterpret_cast<const float*>(result.boxes.data()),
            static_cast<int32_t>(result.boxes.size()), nullptr, 0, 0, out_png);
        std::printf("png: %s (%s)\n", out_png.c_str(), ok ? "ok" : "FAIL");
    }
    if (!out_json.empty()) {
        // Box coordinates are normalized 0-1 xyxy to match the python
        // engine payload (the parity/accuracy diff compares them directly).
        std::string json = "{\"answer\": \"" + wwv::json_escape(result.answer) +
                           "\", \"box_count\": " +
                           std::to_string(result.boxes.size()) +
                           ", \"backend\": \"cpp\"";
        json += ", \"box_coords\": [";
        for (size_t i = 0; i < result.boxes.size(); ++i) {
            const wwv::ResultBox& b = result.boxes[i];
            json += (i ? ", [" : "[") +
                    std::to_string(b.x1 / static_cast<float>(view.width)) + ", " +
                    std::to_string(b.y1 / static_cast<float>(view.height)) + ", " +
                    std::to_string(b.x2 / static_cast<float>(view.width)) + ", " +
                    std::to_string(b.y2 / static_cast<float>(view.height)) + "]";
        }
        json += "]";
        json += std::string(", \"timings\": {\"pre_ms\": ") +
                std::to_string(result.timings.preprocess_ms) +
                ", \"infer_ms\": " +
                std::to_string(result.timings.inference_ms) +
                ", \"post_ms\": " +
                std::to_string(result.timings.postprocess_ms) +
                ", \"e2e_ms\": " +
                std::to_string(result.timings.e2e_ms) + "}";
        json += "}";
        wwv::write_file_bytes(out_json, json.data(), json.size());
    }
    return 0;
}
