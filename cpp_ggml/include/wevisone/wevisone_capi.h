// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// WeVisOne-VL C API: full-chain unified visual perception over the in-tree
// ggml port in cpp_ggml/src (llama.cpp vendored ggml @ LLAMA 0.7.0-dev /
// GGML 0.27.0).
//
// Pipeline: Qwen3-VL ViT + LLM (llama.cpp qwen3vl arch) -> structured answer
// "<box><x1><y1><x2><y2><OBJ></box>" (coordinate tokens <0>..<1000>, "None"
// rejects absent targets) -> optional 9.6M-parameter fp32 pixel decoder
// (MSDA encoder + DINO transformer + dynamic-kernel mask head + full-canvas
// alpha cascade) -> boxes / binary masks / continuous alpha mattes.
//
// Thread-safety: a context is single-threaded. Use one context per consumer,
// or serialize access externally.
//
// Weights: GGUF files produced by cpp_ggml/scripts — the converted 2B/4B
// VLM (mmproj + LLM, llama.cpp converter) and the mask_decoder sidecar
// (convert_mask_decoder_to_gguf.py).
//
// The C API never exposes ggml, llama, STL containers or exceptions across
// the boundary; errors are return codes plus a per-context last_error
// message, exactly like the AICore task APIs this header mirrors.
//
// No environment variables are read anywhere in this library; every knob is
// an explicit field of wevisone_options passed to wevisone_load_opts.
//
// This header is drop-in for include/aicore/wevisone_capi.h at integration
// time (rename "wevisone/wevisone_export.h" and the WWV_/wevisone_ prefixes
// to AICORE_/aicore_).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "wevisone/wevisone_export.h"
#include "wevisone/wevisone_image_view.h"
#include "wevisone/wevisone_pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the WeVisOne-VL C API (bump on breaking ABI
 *  changes). */
WWV_CAPI int wevisone_abi_version(void);

typedef struct wevisone_ctx wevisone_ctx;
typedef struct wevisone_options wevisone_options;

/** Creates a default options struct (device "cuda", threads 0 = backend
 *  default, precision "f16", decoder gguf precision "f32", models_dir
 *  "cpp_ggml/models/gguf", max new tokens 4096, matte decoding enabled).
 *  Release with wevisone_options_free. */
WWV_CAPI wevisone_options* wevisone_options_new(void);
/** Releases an options struct created by wevisone_options_new. */
WWV_CAPI void wevisone_options_free(wevisone_options* opts);
/** Selects the inference device: "cuda", "vulkan", "cpu". Exactly one GPU
 *  backend is compiled into the binary (see CMakeLists.txt). */
WWV_CAPI void wevisone_options_set_device(wevisone_options* opts,
                                          const char* device);
/** CPU thread count; <= 0 picks the backend default. */
WWV_CAPI void wevisone_options_set_threads(wevisone_options* opts,
                                           int n_threads);
/** VLM weight precision: "f16" (default), "q8_0", "q4_K" or "f32". Selects
 *  the <model>-vlm-<precision>.gguf files. */
WWV_CAPI void wevisone_options_set_precision(wevisone_options* opts,
                                             const char* precision);
/** Pixel-decoder weight precision: "f32" (default, the checkpoint dtype and
 *  the only parity-gated tier), "f16" or "q8_0". */
WWV_CAPI void wevisone_options_set_decoder_precision(
        wevisone_options* opts, const char* precision);
/** Directory holding the GGUF files. Default: "cpp_ggml/models/gguf". */
WWV_CAPI void wevisone_options_set_models_dir(wevisone_options* opts,
                                              const char* dir);
/** Explicit per-model overrides; empty (default) derives the filenames from
 *  models_dir + precision:
 *    wevisone-<size>-vlm-<precision>.gguf       (merged mmproj + LLM)
 *    wevisone-<size>-mask-decoder-<prec>.gguf
 *  where <size> is read from the VLM GGUF metadata. */
WWV_CAPI void wevisone_options_set_vlm_gguf(wevisone_options* opts,
                                            const char* path);
WWV_CAPI void wevisone_options_set_decoder_gguf(wevisone_options* opts,
                                                const char* path);
/** Maximum greedy generation length (default 4096). */
WWV_CAPI void wevisone_options_set_max_new_tokens(wevisone_options* opts,
                                                  int n_tokens);
/** Enables (default) or disables the pixel-decoder pass when the answer
 *  contains <OBJ> tokens. Disabling turns the context into a pure object-
 *  level detector (boxes / points / counts / OCR). */
WWV_CAPI void wevisone_options_set_decode_masks(wevisone_options* opts,
                                                int enable);
/** Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR (default 1). */
WWV_CAPI void wevisone_options_set_log_level(wevisone_options* opts,
                                             int log_level);
/** Debug: dump parity stage tensors as raw f32 files into this directory
 *  after every run call. The string is copied; empty (default) disables
 *  dumping. */
WWV_CAPI void wevisone_options_set_dump_dir(wevisone_options* opts,
                                            const char* dir);

/** Loads the VLM (mmproj + LLM) and, lazily, the pixel decoder. Returns a
 *  context whose is_ready() may still be 0 (device admission failure);
 *  inspect wevisone_last_error(). NULL only on invalid input or
 *  out-of-memory. */
WWV_CAPI wevisone_ctx* wevisone_load_opts(const wevisone_options* opts);
/** Releases a context; safe on NULL. Contexts must not be destroyed while a
 *  worker thread is inside a run call (serialize externally). */
WWV_CAPI void wevisone_free(wevisone_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model set. */
WWV_CAPI int wevisone_is_ready(const wevisone_ctx* ctx);
/** Returns the last error message of the context (NULL when none). */
WWV_CAPI const char* wevisone_last_error(const wevisone_ctx* ctx);
/** Releases any buffer returned by a wevisone_* function (JSON/RGB
 *  helpers). Safe on NULL. */
WWV_CAPI void wevisone_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel).
 *  Compatibility helper for path-driven callers; the typed hot path is
 *  wevisone_run_image with a borrowed wwv_image_view. Caller frees \p
 *  out_rgb with wevisone_free_buffer. */
WWV_CAPI int wevisone_load_path_rgb(const char* image_path, uint8_t** out_rgb,
                                    int32_t* out_width, int32_t* out_height);

/** ---- typed inference ----
 *
 *  One request = vision encode + LLM greedy decode (+ optional pixel
 *  decoder over every <OBJ>). \p img must be non-NULL with positive
 *  dimensions, a supported format (RGB8 native; RGBA8/BGRA8/GRAY8/BGR8 are
 *  converted) and row_stride_bytes >= width * channels (else -1).
 *
 *  \p task: "detection" | "referring" | "point" | "count_box" |
 *  "count_point" | "ocr" | "ocr_grounding" | "segmentation" | "matting" |
 *  "visual_prompt" | "freeform" (the released prompt family, prompts.py).
 *  \p query: newline-separated category list or a single referring phrase
 *  (UTF-8); ignored by "ocr" and "freeform".
 *  \p ref_boxes: borrowed [x1, y1, x2, y2] float quadruples in original-
 *  image pixels, used only by "visual_prompt" (the <ref>/<box> tokens).
 *
 *  Returns 0 on success, -1 on contract/validation errors, -2 on
 *  runtime/inference errors and -3 when the device cannot hold the models
 *  plus the graph workspace (see wevisone_last_error). On success the typed
 *  result getters below expose the answer until the next run call. */
WWV_CAPI int wevisone_run_image(wevisone_ctx* ctx, const wwv_image_view* img,
                                const char* task, const char* query,
                                const float* ref_boxes, int32_t n_ref_boxes);

/** ---- typed result of the most recent run (valid until the next run) ---- */

/** Model answer after <|im_end|>/stripping, before parsing. */
WWV_CAPI const char* wevisone_result_answer(const wevisone_ctx* ctx);
/** Full result as JSON (answer, boxes, has_masks, prompt_tokens,
 *  completion_tokens). Caller frees with wevisone_free_buffer. */
WWV_CAPI char* wevisone_result_json(const wevisone_ctx* ctx);
/** Number of parsed result boxes in original-image pixels (0 when the
 *  answer rejected the target). */
WWV_CAPI int32_t wevisone_result_box_count(const wevisone_ctx* ctx);
/** Box \p index in original-image pixels (out[4] = x1, y1, x2, y2).
 *  Returns 0 on success, -1 when out of range. */
WWV_CAPI int wevisone_result_box(const wevisone_ctx* ctx, int32_t index,
                                 float* out);
/** 1 when the answer carried <OBJ> tokens and the pixel decoder ran. */
WWV_CAPI int wevisone_result_has_masks(const wevisone_ctx* ctx);
/** Number of decoded mattes (== box count when masks were decoded). */
WWV_CAPI int32_t wevisone_result_matte_count(const wevisone_ctx* ctx);
/** Matte \p index dimensions. Returns 0 on success, -1 when out of range. */
WWV_CAPI int wevisone_result_matte_size(const wevisone_ctx* ctx,
                                        int32_t index, int32_t* out_width,
                                        int32_t* out_height);
/** Borrowed row-major f32 alpha of matte \p index in [0, 1] (w*h floats);
 *  valid until the next run call. NULL when out of range. */
WWV_CAPI const float* wevisone_result_matte_data(const wevisone_ctx* ctx,
                                                 int32_t index);
/** Token counts of the most recent run. */
WWV_CAPI int32_t wevisone_result_prompt_tokens(const wevisone_ctx* ctx);
WWV_CAPI int32_t wevisone_result_completion_tokens(const wevisone_ctx* ctx);

/** Renders the most recent result on \p img (borrowed view; the image is
 *  copied) in the official style: red width-3 rectangles for boxes plus,
 *  when mattes exist, a green alpha composite of the first matte. Returns 0
 *  on success, -1 on invalid arguments, -2 on write failure. */
WWV_CAPI int wevisone_result_render_png(const wevisone_ctx* ctx,
                                        const wwv_image_view* img,
                                        const char* out_png_path);

/** Per-stage wall-clock timings of the most recent run. Returns 0 on
 *  success, -1 when no inference has run yet. */
WWV_CAPI int wevisone_last_pipeline_timings(const wevisone_ctx* ctx,
                                            wwv_pipeline_timings* out);

/** Model introspection. */
/** Backend-RESOLVED device name (surfaces silent CPU fallbacks). Owned by
 *  the context; copy before freeing. */
WWV_CAPI const char* wevisone_context_device(const wevisone_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
WWV_CAPI int wevisone_context_threads(const wevisone_ctx* ctx);
/** VLM weight precision of the loaded context ("f16"/"q8_0"/"q4_K"/"f32"). */
WWV_CAPI const char* wevisone_context_precision(const wevisone_ctx* ctx);
/** Returns a JSON summary of the loaded models and the latest run. Caller
 *  frees with wevisone_free_buffer. */
WWV_CAPI char* wevisone_info_json(wevisone_ctx* ctx);

/** Warms up the backend for the context's device (compiles backend kernels
 *  so the first real request does not pay JIT costs). */
WWV_CAPI int wevisone_warmup_backend(wevisone_ctx* ctx);
/** Releases process-wide WeVisOne-VL runtime resources it owns (idempotent;
 *  never destroys live contexts). */
WWV_CAPI void wevisone_shutdown(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
