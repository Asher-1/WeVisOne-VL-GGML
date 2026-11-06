// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "result_parser.hpp"

#include <cctype>
#include <cstdlib>

namespace wwv {

namespace {

// Finds "<token>" occurrences from `pos`; returns npos when absent.
size_t find_token(const std::string& s, size_t pos, const std::string& tok) {
    return s.find(tok, pos);
}

// Collects the coordinate tokens <n> inside [begin, end).
std::vector<int32_t> coords_between(const std::string& s, size_t begin,
                                    size_t end) {
    std::vector<int32_t> out;
    size_t p = begin;
    while (p < end) {
        const size_t lt = s.find('<', p);
        if (lt == std::string::npos || lt >= end) break;
        const size_t gt = s.find('>', lt + 1);
        if (gt == std::string::npos || gt > end) break;
        bool digits = gt > lt + 1;
        for (size_t i = lt + 1; i < gt && digits; ++i) {
            digits = std::isdigit(static_cast<unsigned char>(s[i])) != 0;
        }
        if (digits) out.push_back(std::atoi(s.c_str() + lt + 1));
        p = gt + 1;
    }
    return out;
}

}  // namespace

ParsedAnswer parse_answer(const std::string& raw, int32_t width,
                          int32_t height) {
    ParsedAnswer out;
    size_t cut = raw.find("<|im_end|>");
    if (cut == std::string::npos) cut = raw.find("<|endoftext|>");
    const std::string text =
        cut == std::string::npos ? raw : raw.substr(0, cut);
    out.answer = text;

    const float fw = static_cast<float>(width > 0 ? width : 1);
    const float fh = static_cast<float>(height > 0 ? height : 1);

    size_t pos = 0;
    while (true) {
        const size_t box = find_token(text, pos, "<box>");
        if (box == std::string::npos) break;
        size_t close = find_token(text, box, "</box>");
        const size_t body_end =
            close == std::string::npos ? text.size() : close;
        const std::string body =
            text.substr(box + 5, body_end - (box + 5));
        // engine.py: split on <OBJ>, every prefix carries one box.
        size_t seg = 0;
        while (true) {
            const size_t obj = body.find("<OBJ>", seg);
            const bool has = obj != std::string::npos;
            const size_t stop = has ? obj : body.size();
            const std::vector<int32_t> nums =
                coords_between(body, seg, stop);
            if (nums.size() >= 4) {
                const int32_t x1 = nums[nums.size() - 4];
                const int32_t y1 = nums[nums.size() - 3];
                const int32_t x2 = nums[nums.size() - 2];
                const int32_t y2 = nums[nums.size() - 1];
                if (x1 >= 0 && x1 <= 1000 && y1 >= 0 && y1 <= 1000 &&
                    x2 >= 0 && x2 <= 1000 && y2 >= 0 && y2 <= 1000 &&
                    x2 > x1 && y2 > y1) {
                    ResultBox rb;
                    rb.x1 = static_cast<float>(x1) / 1000.0f * fw;
                    rb.y1 = static_cast<float>(y1) / 1000.0f * fh;
                    rb.x2 = static_cast<float>(x2) / 1000.0f * fw;
                    rb.y2 = static_cast<float>(y2) / 1000.0f * fh;
                    out.boxes_px.push_back(rb);
                    out.has_obj.push_back(has);
                }
            }
            if (!has) break;
            seg = obj + 5;
        }
        pos = body_end;
        if (close == std::string::npos) break;
    }
    return out;
}

}  // namespace wwv
