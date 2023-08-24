cmake_minimum_required(VERSION 3.26)
project(sandal CUDA)

set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

add_executable(sandal main.cu fpng.cu
        pendulum.cu
        mandelbrot.cu
        mandelbrot.cuh
        fractal.cuh
        pendulum.cuh
        arbitrary.cu
        arbitrary.cuh
)

set_target_properties(sandal PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

target_link_libraries(sandal ${SDL2_LIBRARIES})