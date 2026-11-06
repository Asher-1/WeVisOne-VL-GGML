// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Official-style PNG rendering (boxes + optional alpha matte composite).
#pragma once

#include <string>

#include "wevisone/wevisone_image_view.h"

namespace wwv {

// Draws red width-3 rectangles for the boxes and, when `matte` is non-null
// (row-major w*h floats in [0, 1]), composites the matte in green, then
// writes a PNG through stb_image_write.
bool render_result_png(const wwv_image_view& img, const float* boxes,
                       int32_t n_boxes, const float* matte, int32_t matte_w,
                       int32_t matte_h, const std::string& out_png_path);

}  // namespace wwv
