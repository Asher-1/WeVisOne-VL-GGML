# Vendored stb (single-file libraries)

Vendored, NOT a git submodule — deliberate, matching how llama.cpp itself
integrates stb (`llama.cpp/vendor/stb/`, same repo layout):

* stb's own distribution model is "drop the single header into your project";
  a submodule would pull the whole multi-library stb repo for two headers.
* Sync contract for other developers: these files are COMMITTED to this
  repository — a plain clone delivers byte-identical copies (no
  submodule init/update step), and `git diff` exposes any local change,
  same guarantee as the ggml-patches contract.
* Zero local modifications expected; if a change is ever required, apply it
  via a patch file in third_party/ggml-patches-style tooling and record it here.

Contents:
* stb_image.h       v2.30 — copied verbatim from llama.cpp/vendor/stb/ at our
                    llama.cpp pin (build b11644), so the decoder matches the
                    one llama.cpp's mtmd uses. Replaces the earlier v2.25 copy
                    taken from the local MVS/xatlas checkout.
* stb_image_write.h v1.16 — copied verbatim from the same MVS/xatlas checkout
                    (llama.cpp does not vendor a write header).
                    v1.14 (from the same checkout)

Update procedure: copy the newer header from llama.cpp/vendor/stb/ after
bumping the llama.cpp pin, or from https://github.com/nothings/stb when
network is available; update the version notes above in the same commit.
