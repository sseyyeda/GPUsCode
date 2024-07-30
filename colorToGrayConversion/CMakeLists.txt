cmake_minimum_required(VERSION 3.10)
project(GrayscaleConversion)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -G -g")

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(GrayscaleConversion main.cpp grayscale.cu)
target_link_libraries(GrayscaleConversion ${OpenCV_LIBS})

