// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Torch-exact host preprocessing for the WeVisOne-VL pipeline.
//
// Two image paths leave this module:
//  1. the decoder CNN canvas: the ORIGINAL photo bicubic-resized to
//     cnn_image_size x cnn_image_size (PIL `resize(..., BICUBIC)` semantics =
//     torch upsample_bicubic2d with antialias=True, implemented tap-exactly —
//     the one FMA torch uses per tap is spelled std::fma, everything else
//     stays unfused; this file is compiled with -ffp-contract=off);
//  2. the ViT input: Qwen smart-resize to patch-grid multiples plus the
//     released Qwen2-VL normalization (per-channel mean/std).
//
// Qwen orders patches as [block_y, block_x, intra_y, intra_x]; the graph
// un-shuffles at the taps, so this module only produces dense row-major
// CHW pixel tensors.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "wevisone/wevisone_image_view.h"

namespace wwv {

// Decoded tightly-packed RGB image (HWC, 3 bytes per pixel).
struct RgbImage {
    int32_t width = 0;
    int32_t height = 0;
    std::vector<uint8_t> pixels;  // row-major RGB8, stride = width * 3
};

// Loads JPEG/PNG from disk through libjpeg/libpng when available (byte-
// identical to PIL), falling back to stb_image otherwise.
bool load_path_rgb(const std::string& path, RgbImage* out, std::string* err);

// Converts a borrowed image view (RGB8 native; RGBA8/BGRA8/BGR8/GRAY8
// converted) to a tightly-packed RgbImage. Returns false on invalid views.
bool view_to_rgb(const wwv_image_view& img, RgbImage* out, std::string* err);

// PIL-BICUBIC (= torch AA bicubic) resize to an exact square canvas.
// Output is CHW f32 in [0, 1] (row-major, 3 * size * size floats).
std::vector<float> bicubic_square_chw(const RgbImage& src, int32_t size);

// Qwen smart-resize: rounds both dimensions to multiples of
// factor = patch_size * merge_size while keeping the pixel count within
// [min_pixels, max_pixels]. Returns the resized CHW f32 tensor, normalized
// with the released Qwen2-VL mean/std, plus the post-resize dimensions.
struct VitInput {
    int32_t width = 0;    // after smart resize
    int32_t height = 0;
    std::vector<float> pixels;  // CHW f32, normalized
};

VitInput smart_resize_normalized(const RgbImage& src, int32_t patch_size,
                                 int32_t merge_size, int64_t min_pixels,
                                 int64_t max_pixels);

// The Qwen factor arithmetic alone (used by the runtime to compute grids).
void smart_resize_dims(int32_t width, int32_t height, int32_t factor,
                       int64_t min_pixels, int64_t max_pixels,
                       int32_t* out_width, int32_t* out_height);

// Released Qwen2-VL image normalization constants.
extern const float kQwenImageMean[3];
extern const float kQwenImageStd[3];

}  // namespace wwv
