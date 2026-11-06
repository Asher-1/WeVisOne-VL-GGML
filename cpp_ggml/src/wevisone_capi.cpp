// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// C API implementation over WevisoneRuntime (thin shell; all validation and
// error mapping lives here so the C ABI never sees C++ exceptions).

#include "wevisone/wevisone_capi.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "image_prep.hpp"
#include "pipeline_common.hpp"
#include "render.hpp"
#include "wevisone_runtime.hpp"

#define WWV_LOG_INFO(...) ::wwv::logf(wwv::LogLevel::kInfo, __VA_ARGS__)

namespace wwv {

namespace {

constexpr int kAbiVersion = 1;

int map_rt_error(const std::string& err) {
    if (err.find("not compiled") != std::string::npos ||
        err.find("cannot hold") != std::string::npos) {
        return -3;  // resources
    }
    if (err.find("invalid") != std::string::npos ||
        err.find("unsupported") != std::string::npos) {
        return -1;  // contract
    }
    return -2;  // runtime
}

}  // namespace

}  // namespace wwv

// The C API declares wevisone_options / wevisone_ctx as opaque typedefs;
// the concrete layouts live here at global scope and never cross the ABI.
struct wevisone_options {
    wwv::RuntimeOptions rt;
};

struct wevisone_ctx {
    std::unique_ptr<wwv::WevisoneRuntime> rt;
    wwv::RunResult last;
};

using namespace wwv;  // NOLINT(google-build-using-namespace) — C shim only

extern "C" {

int wevisone_abi_version(void) { return kAbiVersion; }

wevisone_options* wevisone_options_new(void) { return new wevisone_options(); }

void wevisone_options_free(wevisone_options* opts) { delete opts; }

void wevisone_options_set_device(wevisone_options* opts, const char* device) {
    if (opts != nullptr && device != nullptr) opts->rt.device = device;
}

void wevisone_options_set_threads(wevisone_options* opts, int n_threads) {
    if (opts != nullptr) opts->rt.threads = n_threads;
}

void wevisone_options_set_precision(wevisone_options* opts,
                                    const char* precision) {
    if (opts != nullptr && precision != nullptr) opts->rt.precision = precision;
}

void wevisone_options_set_decoder_precision(wevisone_options* opts,
                                            const char* precision) {
    if (opts != nullptr && precision != nullptr) {
        opts->rt.decoder_precision = precision;
    }
}

void wevisone_options_set_models_dir(wevisone_options* opts, const char* dir) {
    if (opts != nullptr && dir != nullptr) opts->rt.models_dir = dir;
}

void wevisone_options_set_vlm_gguf(wevisone_options* opts, const char* path) {
    if (opts != nullptr && path != nullptr) opts->rt.vlm_gguf = path;
}

void wevisone_options_set_decoder_gguf(wevisone_options* opts,
                                       const char* path) {
    if (opts != nullptr && path != nullptr) opts->rt.decoder_gguf = path;
}

void wevisone_options_set_max_new_tokens(wevisone_options* opts,
                                         int n_tokens) {
    if (opts != nullptr) opts->rt.max_new_tokens = n_tokens;
}

void wevisone_options_set_decode_masks(wevisone_options* opts, int enable) {
    if (opts != nullptr) opts->rt.decode_masks = enable != 0;
}

void wevisone_options_set_log_level(wevisone_options* opts, int log_level) {
    if (opts != nullptr) opts->rt.log_level = log_level;
}

void wevisone_options_set_dump_dir(wevisone_options* opts, const char* dir) {
    if (opts != nullptr && dir != nullptr) opts->rt.dump_dir = dir;
}

wevisone_ctx* wevisone_load_opts(const wevisone_options* opts) {
    if (opts == nullptr) return nullptr;
    std::string err;
    auto ctx = new wevisone_ctx();
    ctx->rt = WevisoneRuntime::create(opts->rt, &err);
    if (ctx->rt == nullptr) {
        WWV_LOG_INFO("load failed: %s", err.c_str());
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void wevisone_free(wevisone_ctx* ctx) { delete ctx; }

int wevisone_is_ready(const wevisone_ctx* ctx) {
    return ctx != nullptr && ctx->rt != nullptr && ctx->rt->is_ready() ? 1 : 0;
}

const char* wevisone_last_error(const wevisone_ctx* ctx) {
    return ctx == nullptr || ctx->rt == nullptr ? nullptr
                                                : ctx->rt->last_error();
}

void wevisone_free_buffer(void* p) { std::free(p); }

int wevisone_load_path_rgb(const char* image_path, uint8_t** out_rgb,
                           int32_t* out_width, int32_t* out_height) {
    if (image_path == nullptr || out_rgb == nullptr || out_width == nullptr ||
        out_height == nullptr) {
        return -1;
    }
    RgbImage img;
    std::string err;
    if (!load_path_rgb(image_path, &img, &err)) return -1;
    const size_t n = static_cast<size_t>(img.width) * img.height * 3;
    *out_rgb = static_cast<uint8_t*>(std::malloc(n));
    if (*out_rgb == nullptr) return -1;
    std::memcpy(*out_rgb, img.pixels.data(), n);
    *out_width = img.width;
    *out_height = img.height;
    return 0;
}

int wevisone_run_image(wevisone_ctx* ctx, const wwv_image_view* img,
                       const char* task, const char* query,
                       const float* ref_boxes, int32_t n_ref_boxes) {
    if (ctx == nullptr || ctx->rt == nullptr || img == nullptr ||
        task == nullptr) {
        return -1;
    }
    if (img->data == nullptr || img->width <= 0 || img->height <= 0) {
        return -1;
    }
    RunResult res;
    if (!ctx->rt->run_image(*img, task, query != nullptr ? query : "",
                            ref_boxes, n_ref_boxes, &res)) {
        return map_rt_error(ctx->rt->last_error());
    }
    ctx->last = std::move(res);
    return 0;
}

const char* wevisone_result_answer(const wevisone_ctx* ctx) {
    return ctx == nullptr ? nullptr : ctx->last.answer.c_str();
}

char* wevisone_result_json(const wevisone_ctx* ctx) {
    if (ctx == nullptr) return nullptr;
    std::ostringstream json;
    json << "{\"answer\": \"" << json_escape(ctx->last.answer) << "\", ";
    json << "\"boxes\": [";
    for (size_t i = 0; i < ctx->last.boxes.size(); ++i) {
        const ResultBox& b = ctx->last.boxes[i];
        json << (i > 0 ? ", " : "") << "[" << b.x1 << ", " << b.y1 << ", "
             << b.x2 << ", " << b.y2 << "]";
    }
    json << "], ";
    json << "\"has_masks\": " << (ctx->last.masks_decoded ? "true" : "false")
         << ", ";
    json << "\"prompt_tokens\": " << ctx->last.prompt_tokens << ", ";
    json << "\"completion_tokens\": " << ctx->last.completion_tokens << "}";
    const std::string s = json.str();
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

int32_t wevisone_result_box_count(const wevisone_ctx* ctx) {
    return ctx == nullptr
               ? 0
               : static_cast<int32_t>(ctx->last.boxes.size());
}

int wevisone_result_box(const wevisone_ctx* ctx, int32_t index, float* out) {
    if (ctx == nullptr || out == nullptr || index < 0 ||
        index >= static_cast<int32_t>(ctx->last.boxes.size())) {
        return -1;
    }
    const ResultBox& b = ctx->last.boxes[static_cast<size_t>(index)];
    out[0] = b.x1;
    out[1] = b.y1;
    out[2] = b.x2;
    out[3] = b.y2;
    return 0;
}

int wevisone_result_has_masks(const wevisone_ctx* ctx) {
    return ctx != nullptr && ctx->last.masks_decoded ? 1 : 0;
}

int32_t wevisone_result_matte_count(const wevisone_ctx* ctx) {
    return wevisone_result_has_masks(ctx) != 0
               ? wevisone_result_box_count(ctx)
               : 0;
}

int wevisone_result_matte_size(const wevisone_ctx* ctx, int32_t index,
                               int32_t* out_width, int32_t* out_height) {
    if (ctx == nullptr || out_width == nullptr || out_height == nullptr ||
        index != 0 || !ctx->last.masks_decoded) {
        return -1;
    }
    *out_width = ctx->last.matte_width;
    *out_height = ctx->last.matte_height;
    return 0;
}

const float* wevisone_result_matte_data(const wevisone_ctx* ctx,
                                        int32_t index) {
    if (ctx == nullptr || index != 0 || !ctx->last.masks_decoded) return nullptr;
    return ctx->last.matte.data();
}

int32_t wevisone_result_prompt_tokens(const wevisone_ctx* ctx) {
    return ctx == nullptr ? 0 : ctx->last.prompt_tokens;
}

int32_t wevisone_result_completion_tokens(const wevisone_ctx* ctx) {
    return ctx == nullptr ? 0 : ctx->last.completion_tokens;
}

int wevisone_result_render_png(const wevisone_ctx* ctx,
                               const wwv_image_view* img,
                               const char* out_png_path) {
    if (ctx == nullptr || img == nullptr || out_png_path == nullptr) return -1;
    const bool ok = render_result_png(
        *img,
        reinterpret_cast<const float*>(ctx->last.boxes.data()),
        static_cast<int32_t>(ctx->last.boxes.size()),
        ctx->last.masks_decoded ? ctx->last.matte.data() : nullptr,
        ctx->last.matte_width, ctx->last.matte_height, out_png_path);
    return ok ? 0 : -2;
}

int wevisone_last_pipeline_timings(const wevisone_ctx* ctx,
                                   wwv_pipeline_timings* out) {
    if (ctx == nullptr || out == nullptr) return -1;
    *out = ctx->last.timings;
    return 0;
}

const char* wevisone_context_device(const wevisone_ctx* ctx) {
    return ctx == nullptr || ctx->rt == nullptr
               ? nullptr
               : ctx->rt->resolved_device().c_str();
}

int wevisone_context_threads(const wevisone_ctx* ctx) {
    return ctx == nullptr || ctx->rt == nullptr
               ? 0
               : ctx->rt->resolved_threads();
}

const char* wevisone_context_precision(const wevisone_ctx* ctx) {
    return ctx == nullptr || ctx->rt == nullptr
               ? nullptr
               : ctx->rt->resolved_precision().c_str();
}

char* wevisone_info_json(wevisone_ctx* ctx) {
    if (ctx == nullptr || ctx->rt == nullptr) return nullptr;
    std::ostringstream json;
    json << "{\"device\": \"" << json_escape(ctx->rt->resolved_device())
         << "\", \"threads\": " << ctx->rt->resolved_threads()
         << ", \"precision\": \"" << ctx->rt->resolved_precision() << "\", "
         << "\"ready\": " << (ctx->rt->is_ready() ? "true" : "false") << "}";
    const std::string s = json.str();
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

int wevisone_warmup_backend(wevisone_ctx* ctx) {
    if (ctx == nullptr || ctx->rt == nullptr) return -1;
    return ctx->rt->warmup_backend() ? 0 : -2;
}

void wevisone_shutdown(void) {
    // Model and context frees happen through context destruction; ggml
    // backends self-report. Kept for AICore lifecycle symmetry.
}

}  // extern "C"
