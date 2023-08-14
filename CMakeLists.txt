cmake_minimum_required(VERSION 3.26)
project(sandal CUDA)

set(CMAKE_CUDA_STANDARD 23)
SET(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0")
SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

add_executable(sandal main.cu fpng.cu
        pendulum.cu
        mandelbrot.cu
        mandelbrot.cuh
        fractal.cuh
        pendulum.cuh
)

set_target_properties(sandal PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)