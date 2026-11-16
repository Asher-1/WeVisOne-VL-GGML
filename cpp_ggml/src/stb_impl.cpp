// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Single translation unit carrying the stb implementations (one per binary;
// llama.cpp vendors the headers, this repo owns the instantiations).

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO_MSC 1
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
