// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Borrowed, row-stride-aware image view (== aicore_image_view). The C API
// never copies unless documented; callers keep ownership of the pixels.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "wevisone/wevisone_export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum wwv_image_format {
    WWV_IMAGE_RGB8 = 1,
    WWV_IMAGE_RGBA8 = 2,
    WWV_IMAGE_GRAY8 = 3,
    WWV_IMAGE_BGR8 = 4,
    WWV_IMAGE_BGRA8 = 5
} wwv_image_format;

typedef struct wwv_image_view {
    const uint8_t* data;
    int32_t width;
    int32_t height;
    size_t row_stride_bytes;
    wwv_image_format format;
} wwv_image_view;

#ifdef __cplusplus
} /* extern "C" */
#endif
