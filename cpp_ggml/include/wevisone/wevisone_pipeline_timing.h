// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Common wall-clock timing contract for every WeVisOne-VL pipeline stage,
// byte-compatible with aicore_pipeline_timings.
#pragma once

#include <stdint.h>

#include "wevisone/wevisone_export.h"

#define WWV_PIPELINE_TIMINGS_ABI_VERSION 1u

enum wwv_pipeline_timing_field {
    WWV_TIMING_PREPROCESS = 1u << 0,
    WWV_TIMING_INFERENCE = 1u << 1,
    WWV_TIMING_POSTPROCESS = 1u << 2,
    WWV_TIMING_SERIALIZATION = 1u << 3,
    WWV_TIMING_E2E = 1u << 4
};

// preprocess_ms:  input validation, decode/resize/normalize, ViT forward and
//                 prompt assembly.
// inference_ms:   LLM prefill + decode, optional teacher-forced hidden-state
//                 pass, pixel-decoder graph execution and readback.
// postprocess_ms: structured-answer parsing, box/matte extraction.
// serialization_ms: optional PNG rendering only.
// e2e_ms:         API entry until the native result is ready; excludes
//                 optional serialization.
typedef struct wwv_pipeline_timings {
    uint32_t abi_version;
    uint32_t valid_fields;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double serialization_ms;
    double e2e_ms;
} wwv_pipeline_timings;
