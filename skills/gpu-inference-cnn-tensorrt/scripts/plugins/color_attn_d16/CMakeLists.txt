cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(color_attn_d16_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# The MMA (s8.u8 / FP8) instructions need sm_120; disable CMake's default arch
# so only the explicit -gencode below is used.
set(CMAKE_CUDA_ARCHITECTURES OFF)

set(TRT_LIB "" CACHE PATH "TensorRT library path")
set(TRT_INCLUDE "" CACHE PATH "TensorRT include path")
set(CUDA_INC_DIR "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" CACHE PATH "CUDA include path")
set(CUDA_LIB_DIR "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}" CACHE PATH "CUDA root")

if(NOT TRT_LIB OR NOT TRT_INCLUDE)
    message(FATAL_ERROR "Pass -DTRT_LIB=<TensorRT lib dir> and -DTRT_INCLUDE=<TensorRT include dir>")
endif()

find_library(NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
if(NOT NVINFER_LIB)
    find_library(NVINFER_LIB NAMES libnvinfer.so.10 HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64 REQUIRED)
endif()
find_library(CUDA_DRIVER_LIB cuda HINTS ${CUDA_LIB_DIR} PATH_SUFFIXES lib/stubs lib64/stubs REQUIRED)

add_library(color_attn_d16_plugin SHARED
    color_attn_d16_plugin.cu
    color_attn_d16_kernels.cu
)
target_include_directories(color_attn_d16_plugin PUBLIC ${TRT_INCLUDE} ${CUDA_INC_DIR})
target_link_libraries(color_attn_d16_plugin PRIVATE ${NVINFER_LIB} ${CUDA_DRIVER_LIB})
target_compile_options(color_attn_d16_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
