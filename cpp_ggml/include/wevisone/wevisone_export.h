// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// WeVisOne-VL C API export macro. Mirrors the AICore export contract: the
// wevisone task ships as a static archive here and as part of libAICore when
// integrated, so the macro stays neutral (empty) and the headers remain
// drop-in for include/aicore/ (rename wevisone/ -> aicore/, WWV_ -> AICORE_,
// wevisone_ -> aicore_).
#pragma once

#define WWV_CAPI
