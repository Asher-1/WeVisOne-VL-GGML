# Idempotent llama.cpp patch replay for the WeVisOne-VL integration.
#
# Contract (mirrors the VLX-Seek / ultralytics-ggml integrations):
#   * exactly the patches under third_party/llama-patches/ are applied, in
#     lexical order, against the pinned submodule checkout;
#   * the stamp file records the patch set; any change invalidates it;
#   * a stamp that is older than a patched file is also invalidated, so a
#     partial manual edit can never be mistaken for a successful replay.

set(stamp_content "")
foreach(p IN LISTS WWV_PATCH_FILES)
    file(SHA256 "${p}" h)
    get_filename_component(n "${p}" NAME)
    string(APPEND stamp_content "${n} ${h}\n")
endforeach()

set(need_replay TRUE)
if(EXISTS "${STAMP}")
    file(READ "${STAMP}" old_content)
    if(old_content STREQUAL stamp_content)
        set(need_replay FALSE)
    endif()
endif()

if(NOT WWV_PATCH_FILES)
    file(WRITE "${STAMP}" "")
    message(STATUS "No llama.cpp patches to apply.")
    return()
endif()

# A matching stamp is not proof: a `git checkout .` in the submodule reverts
# the sources but leaves the stamp. Verify every patch is still reverse-
# applicable.
if(NOT need_replay)
    foreach(p IN LISTS WWV_PATCH_FILES)
        execute_process(
            COMMAND "${PATCH_EXECUTABLE}" -p1 --reverse --silent
                    --dry-run --directory "${LLAMA_DIR}" -i "${p}"
            RESULT_VARIABLE rv
            ERROR_QUIET OUTPUT_QUIET
        )
        if(NOT rv EQUAL 0)
            set(need_replay TRUE)
        endif()
    endforeach()
endif()

if(need_replay)
    execute_process(
        COMMAND git -C "${LLAMA_DIR}" status --porcelain
        OUTPUT_VARIABLE gs
        ERROR_QUIET
    )
    # After a successful previous replay the submodule is intentionally dirty;
    # only refuse when a replay has never happened and the tree is dirty.
    if(gs AND NOT EXISTS "${STAMP}")
        message(FATAL_ERROR
            "llama.cpp submodule is dirty and no patch stamp exists. "
            "Refusing to replay patches on an unknown tree. Use a clean "
            "worktree checkout of the pinned revision.")
    endif()

    foreach(p IN LISTS WWV_PATCH_FILES)
        get_filename_component(n "${p}" NAME)
        message(STATUS "Applying ${n}")
        # Forward apply; tolerate already-applied state by reverse-checking.
        execute_process(
            COMMAND "${PATCH_EXECUTABLE}" -p1 --forward --silent
                    --directory "${LLAMA_DIR}" -i "${p}"
            RESULT_VARIABLE r1
            ERROR_QUIET OUTPUT_QUIET
        )
        if(NOT r1 EQUAL 0)
            execute_process(
                COMMAND "${PATCH_EXECUTABLE}" -p1 --reverse --silent
                        --dry-run --directory "${LLAMA_DIR}" -i "${p}"
                RESULT_VARIABLE r2
                ERROR_QUIET OUTPUT_QUIET
            )
            if(r2 EQUAL 0)
                message(STATUS "${n} already applied; skipping.")
            else()
                message(FATAL_ERROR "Patch ${n} failed to apply. "
                    "Reset third_party/llama.cpp to the pinned revision "
                    "and retry.")
            endif()
        endif()
    endforeach()
endif()

file(WRITE "${STAMP}" "${stamp_content}")
