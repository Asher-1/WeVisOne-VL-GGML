// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Structured-answer parsing for the released WeVisOne-VL checkpoint
// (engine.py::parse_seg_boxes semantics): <box>...<OBJ>...</box> groups with
// coordinate tokens <0>..<1000>, "None" rejection, im_end truncation.
#pragma once

#include <string>
#include <vector>

#include "wwv_types.hpp"

namespace wwv {

struct ParsedAnswer {
    std::string answer;                // truncated at <|im_end|>
    std::vector<ResultBox> boxes_px;   // original-image pixels
    std::vector<bool> has_obj;         // per box: followed by <OBJ>
};

// Parses the model answer. boxes are scaled back by (width, height); the
// <box>/<OBJ> grammar matches engine.py exactly (last 4 coord tokens of a
// prefix, 0..1000 domain, x2 > x1, y2 > y1).
ParsedAnswer parse_answer(const std::string& raw, int32_t width,
                          int32_t height);

}  // namespace wwv
