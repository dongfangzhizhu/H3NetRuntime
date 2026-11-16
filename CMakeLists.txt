# h3rt - Python-free native runtime DLL/SO for .NET P/Invoke.
#
# Windows (MSVC + CUDA 13.x + Ninja), from a vcvars64 shell:
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DH3_CUDA_ARCH=89
#   cmake --build build --target h3rt
# Linux / WSL (gcc + nvcc):
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DH3_CUDA_ARCH=89
#   cmake --build build --target h3rt
#
# The engine sources are vendored under native/engine (see its ORIGIN.md).
# Point H3_ENGINE_DIR at an h3-spark.c checkout to build against a newer tree.
cmake_minimum_required(VERSION 3.25)
project(h3rt LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(H3_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/native/engine" CACHE PATH
    "Directory holding the vendored h3-spark.c sources")
set(H3_CUDA_ARCH "89" CACHE STRING "CUDA compute capability (89 = RTX 4090, 121 = GB10)")
set(CMAKE_CUDA_ARCHITECTURES "${H3_CUDA_ARCH}")
add_compile_definitions(H3_SM_ARCH=${H3_CUDA_ARCH})

if(NOT EXISTS "${H3_ENGINE_DIR}/h3.c")
    message(FATAL_ERROR "h3.c not found under H3_ENGINE_DIR='${H3_ENGINE_DIR}'")
endif()

set(E "${H3_ENGINE_DIR}")
set(H3_LIB_SOURCES
    h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c
    h3_dit_schedule.c h3_dit.c h3_lora.c h3_config.c h3_pddheads.c
    h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c
    h3_terminal.c h3_vision_encoder.c h3_multimodal.c h3_device.c
    h3_cuda.c h3_tokenizer.c h3_hash.c h3_quanto.c h3_nunchaku.c
    h3_zimage_dit.c h3_zimage_vae.c h3_zimage_pipeline.c h3_zimage_text.c
    h3_wan22_dit.c h3_wan22_text.c h3_wan22_tokenizer.c h3_wan22_vae.c
    h3_wan22_pipeline.c h3_wan22_solver.c
    h3_qwen21_dit.c h3_qwen21_text.c h3_qwen21_vae.c h3_qwen21_pipeline.c
    third_party/cJSON.c)
set(H3_CUDA_SOURCES h3_gpu.cu h3_gpu_stubs.c)
set(H3_WIN_SOURCES
    platform/win32/h3_win_posix.c platform/win32/h3_win_spawn.c
    platform/win32/h3_win_socket.c platform/win32/h3_win_thread.c
    platform/win32/h3_win_dir.c platform/win32/h3_win_getopt.c
    platform/win32/h3_win_unicode.c)
list(TRANSFORM H3_LIB_SOURCES PREPEND "${E}/")
list(TRANSFORM H3_CUDA_SOURCES PREPEND "${E}/")
list(TRANSFORM H3_WIN_SOURCES PREPEND "${E}/")

# --- the exported runtime --------------------------------------------------
add_library(h3rt SHARED
    ${H3_LIB_SOURCES}
    ${H3_CUDA_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/h3rt.c")
if(WIN32)
    target_sources(h3rt PRIVATE ${H3_WIN_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/src/h3rt.def")
endif()

target_include_directories(h3rt PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(h3rt PRIVATE
    "${E}" "${E}/third_party"
    "$<$<PLATFORM_ID:Windows>:${E}/platform/win32>"
    "$<$<PLATFORM_ID:Windows>:${E}/platform/win32/include>")
# cJSON defaults to __declspec(dllexport) on Windows; hide it so the .def file
# is the complete, stable export list of the DLL.
target_compile_definitions(h3rt PRIVATE H3RT_BUILDING_DLL=1 CJSON_HIDE_SYMBOLS=1)

# Mirrors upstream cmake/PlatformWin32.cmake.  The POSIX compatibility header
# is force-included into *every* translation unit, including src/h3rt.c.
if(WIN32)
    set(H3_PLATFORM_HEADER "${E}/platform/win32/h3_platform.h")
    target_compile_definitions(h3rt PRIVATE
        _CRT_SECURE_NO_WARNINGS _CRT_SECURE_NO_DEPRECATE
        _CRT_NONSTDC_NO_DEPRECATE WIN32_LEAN_AND_MEAN NOMINMAX
        _SCL_SECURE_NO_WARNINGS H3_WIN32=1)
    target_compile_options(h3rt PRIVATE
        "$<$<COMPILE_LANGUAGE:C>:/std:c11;/utf-8;/openmp:llvm;/W3;/FI${H3_PLATFORM_HEADER}>"
        "$<$<COMPILE_LANGUAGE:CXX>:/std:c++17;/utf-8;/openmp:llvm;/W3;/FI${H3_PLATFORM_HEADER}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O2;-Xcompiler=/utf-8,-Xcompiler=/openmp:llvm,-Xcompiler=/FI${H3_PLATFORM_HEADER}>"
        "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>")
    target_link_options(h3rt PRIVATE "/STACK:8388608")
else()
    target_compile_definitions(h3rt PRIVATE _GNU_SOURCE)
    target_compile_options(h3rt PRIVATE
        "$<$<COMPILE_LANGUAGE:C>:-O2;-Wall;-Wextra;-fopenmp>"
        "$<$<COMPILE_LANGUAGE:CXX>:-O2;-fopenmp>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O2;-Xcompiler=-fPIC>")
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(h3rt PRIVATE OpenMP::OpenMP_C)
    endif()
    find_library(ICUUC_LIBRARY NAMES icuuc)
    if(ICUUC_LIBRARY)
        target_link_libraries(h3rt PRIVATE "${ICUUC_LIBRARY}")
    endif()
    target_link_libraries(h3rt PRIVATE stdc++ m pthread dl)
endif()

find_package(CUDAToolkit REQUIRED)
target_link_libraries(h3rt PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
if(WIN32)
    target_link_libraries(h3rt PRIVATE ws2_32)
endif()
target_compile_options(h3rt PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")

set_target_properties(h3rt PROPERTIES
    OUTPUT_NAME "h3rt"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# --- C ABI smoke test ------------------------------------------------------
add_executable(h3rt_smoke "${CMAKE_CURRENT_SOURCE_DIR}/native/tests/h3rt_smoke.c")
target_include_directories(h3rt_smoke PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(h3rt_smoke PRIVATE h3rt)
set_target_properties(h3rt_smoke PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(WIN32)
    target_compile_definitions(h3rt_smoke PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_options(h3rt_smoke PRIVATE "/STACK:8388608")
endif()

message(STATUS "h3rt: engine='${H3_ENGINE_DIR}', CUDA arch sm_${H3_CUDA_ARCH}")
