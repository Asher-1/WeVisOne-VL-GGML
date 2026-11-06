cmake_minimum_required(VERSION 3.14)
project(wevisonevl LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Backends: forward a single GPU backend selection to ggml.
# ---------------------------------------------------------------------------
option(WWV_GGML_CUDA   "Enable CUDA backend in ggml"   OFF)
option(WWV_GGML_VULKAN "Enable Vulkan backend in ggml" OFF)
option(WWV_GGML_METAL  "Enable Metal backend in ggml"  OFF)
option(WWV_GGML_BUILD_CLI "Build the wevisone-cli binary" ON)

set(_wwv_gpu_backends 0)
foreach(_b WWV_GGML_CUDA WWV_GGML_VULKAN WWV_GGML_METAL)
    if(${_b})
        math(EXPR _wwv_gpu_backends "${_wwv_gpu_backends} + 1")
    endif()
endforeach()
if(_wwv_gpu_backends GREATER 1)
    message(FATAL_ERROR "Select at most one GPU backend per build directory")
endif()

if(WWV_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    enable_language(CUDA)
endif()
if(WWV_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(WWV_GGML_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "cpp_ggml/third_party/llama.cpp is missing — run "
        "git submodule update --init cpp_ggml/third_party/llama.cpp. It is "
        "the single ggml source (pinned at LLAMA 0.7.0-dev / GGML 0.27.0) "
        "and the only one this build links.")
endif()

# ---------------------------------------------------------------------------
# Single ggml source: llama.cpp's vendored copy (pin 620d2225 = LLAMA
# 0.7.0-dev + GGML 0.27.0). The pixel-decoder graphs (wevisonevision)
# compile against it, so the vision taps, MSDA decoder and the LLM share one
# runtime. ggml-source changes go through third_party/llama-patches/*.patch
# — never hand-edit the submodule (sync contract for other developers).
# Patches are replayed at configure time: add_subdirectory below reads the
# sources immediately, so a build-time custom command would be too late.
# ---------------------------------------------------------------------------
find_package(Patch REQUIRED)
file(GLOB WWV_LLAMA_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-patches/*.patch")
list(SORT WWV_LLAMA_PATCHES)
execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DPATCH_EXECUTABLE=${Patch_EXECUTABLE}
            -DLLAMA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
            "-DWWV_PATCH_FILES=${WWV_LLAMA_PATCHES}"
            -DSTAMP=${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/.wwv-patches-applied
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_patches.cmake
    RESULT_VARIABLE _wwv_llama_patch_rc)
if(NOT _wwv_llama_patch_rc EQUAL 0)
    message(FATAL_ERROR "llama.cpp patch replay failed (${_wwv_llama_patch_rc})")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${WWV_LLAMA_PATCHES})

# Must come BEFORE anything references the ggml target: llama.cpp brings its
# own vendored ggml (the only one this build links).
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp llama.cpp-sub EXCLUDE_FROM_ALL)
# mtmd (the supported VLM image/text chunk path) — self-contained target,
# added directly so LLAMA_BUILD_TOOLS stays OFF (no example binaries).
add_subdirectory(third_party/llama.cpp/tools/mtmd mtmd-sub EXCLUDE_FROM_ALL)
# quantize tool for the GGUF variant export (f16 -> q8_0 / q4_K_M)
add_subdirectory(third_party/llama.cpp/tools/quantize quantize-sub EXCLUDE_FROM_ALL)

# llama.cpp vendors stb_image at vendor/stb — reuse it (single source; a
# separate copy would drift).
set(WWV_STB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/vendor/stb")
# stb_image_write is not vendored by llama.cpp — the local third_party/stb
# copy (byte-identical to the sibling integrations) covers PNG output.
set(WWV_STB_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")

# ---------------------------------------------------------------------------
# wevisonevision: vision + pixel-decoder sub-chain.
#   gguf_loader          minimal GGUF reader (F32/F16/Q8_0)
#   image_prep           torch-exact host preprocessing (bicubic 1024 canvas,
#                        Qwen smart-resize grids)
#   vision_taps_graph    Qwen3-VL ViT forward with pre-merger taps at the
#                        released feature layers (5/11/17/23)
#   mask_decoder_graph   the 9.6M-parameter fp32 pixel decoder graph
#   result_parser        structured-answer parsing (<box>/<OBJ>/coords/None)
#   render               official-style PNG visualization (boxes + matte)
#   pipeline_common      shared helpers (logging, timing, file IO)
#   runtime              task runtime: llama.cpp VLM + decoder assembly
#   capi                 the C API implementation (thin shell)
# ---------------------------------------------------------------------------
add_library(wevisonevision STATIC
    src/gguf_loader.cpp
    src/stb_impl.cpp
    src/image_prep.cpp
    src/vision_taps_graph.cpp
    src/mask_decoder_graph.cpp
    src/result_parser.cpp
    src/render.cpp
    src/pipeline_common.cpp
    src/wevisone_runtime.cpp
    src/wevisone_capi.cpp
)
target_include_directories(wevisonevision PUBLIC src include)
target_include_directories(wevisonevision SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/tools/mtmd
    ${WWV_STB_DIR}
    ${WWV_STB_LOCAL}
)
target_link_libraries(wevisonevision PUBLIC ggml llama mtmd Threads::Threads)

# Host-side preprocessing must round exactly like torch's CPU kernels: the
# one FMA torch uses (AA bicubic taps) is spelled std::fma, every other
# mul+add has to stay unfused whatever -march the toolchain defaults to.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/image_prep.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# JPEG decode parity: PIL decodes through libjpeg (ISLOW IDCT + fancy
# upsampling); stb_image's IDCT/upsampler differ by +-1..2 LSB on real
# photos. With the system libjpeg(-turbo) the decoded RGB is byte-identical
# to Image.open(...).convert("RGB"); without it stb_image is the fallback.
find_package(JPEG)
if(JPEG_FOUND)
    target_compile_definitions(wevisonevision PRIVATE WWV_HAVE_LIBJPEG=1)
    target_link_libraries(wevisonevision PRIVATE JPEG::JPEG)
endif()

find_package(Threads REQUIRED)

# Pure-C smoke test of the WeVisOne-VL C API (AICore-contract surface).
add_executable(wevisone-capi-smoke tests/wevisone_capi_smoke.c)
target_include_directories(wevisone-capi-smoke PRIVATE include)
target_link_libraries(wevisone-capi-smoke PRIVATE wevisonevision)
set_target_properties(wevisone-capi-smoke PROPERTIES LINKER_LANGUAGE CXX)

# ---------------------------------------------------------------------------
# Full-chain CLI: image -> prompt -> LLM -> parse -> pixel decoder -> PNG.
# ---------------------------------------------------------------------------
if(WWV_GGML_BUILD_CLI)
    add_executable(wevisone-cli src/wevisone_cli.cpp)
    target_include_directories(wevisone-cli SYSTEM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
        ${WWV_STB_DIR}
        ${WWV_STB_LOCAL}
    )
    target_link_libraries(wevisone-cli PRIVATE wevisonevision)
endif()
