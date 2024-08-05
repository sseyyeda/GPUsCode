cmake_minimum_required(VERSION 3.10)
project(MatrixMultiplication)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

# CUDA settings
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# Add the CUDA executable
cuda_add_executable(main_gpu src/main.cu)
target_link_libraries(main_gpu OpenMP::OpenMP_CXX ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Ensure CUDA architecture compatibility (update as needed)
set_target_properties(main_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60"  # Adjust to your GPU architecture
)

