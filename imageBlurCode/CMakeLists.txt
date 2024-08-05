cmake_minimum_required(VERSION 3.10)
project(ImageBlurring)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS} include)

set(CMAKE_CXX_STANDARD 17)

# CPU executable
add_executable(image_blurring src/main.cpp src/blur_cpu.cpp)
target_link_libraries(image_blurring ${OpenCV_LIBS})

# GPU executable
cuda_add_executable(image_blurring_gpu src/main.cpp src/blur_cpu.cpp src/blur_gpu.cu)
target_compile_definitions(image_blurring_gpu PRIVATE USE_GPU)
target_link_libraries(image_blurring_gpu ${OpenCV_LIBS} ${CUDA_LIBRARIES})

