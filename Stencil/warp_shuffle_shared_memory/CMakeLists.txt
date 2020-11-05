cmake_minimum_required(VERSION 3.2)

set(CMAKE_C_COMPILER "icc")
set(CMAKE_CXX_COMPILER "icpc")

project(stencil LANGUAGES CXX CUDA)

include_directories(../../Utility)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CCBIN "-ccbin=${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CCBIN}")

add_executable(stencil GPU.cu)
target_link_libraries(stencil intlc)
set_property(TARGET stencil PROPERTY CUDA_SEPARABLE_COMPILATION ON)


