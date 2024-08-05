#include "blur_gpu.cuh"
#include <cuda_runtime.h>

#define CHECK_CUDA_ERRORS(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int rows, int cols, int channels, int blur_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int patch_size = 2 * blur_size + 1;
    int half_patch = patch_size * patch_size;
    int pixel_index = (y * cols + x) * channels;

    float sum[3] = {0.0f, 0.0f, 0.0f};

    for (int dy = -blur_size; dy <= blur_size; ++dy) {
        for (int dx = -blur_size; dx <= blur_size; ++dx) {
            int yy = min(max(y + dy, 0), rows - 1);
            int xx = min(max(x + dx, 0), cols - 1);
            int neighbor_index = (yy * cols + xx) * channels;

            for (int c = 0; c < channels; ++c) {
                sum[c] += input[neighbor_index + c];
            }
        }
    }

    for (int c = 0; c < channels; ++c) {
        output[pixel_index + c] = static_cast<unsigned char>(sum[c] / half_patch);
    }
}

void blur_image_gpu(const cv::Mat& input, cv::Mat& output, int blur_size) {
    int rows = input.rows;
    int cols = input.cols;
    int channels = input.channels();
    size_t img_size = rows * cols * channels * sizeof(unsigned char);

    unsigned char *d_input, *d_output;

    CHECK_CUDA_ERRORS(cudaMalloc(&d_input, img_size));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_output, img_size));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_input, input.ptr(), img_size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    blur_kernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols, channels, blur_size);
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    output.create(rows, cols, input.type());
    CHECK_CUDA_ERRORS(cudaMemcpy(output.ptr(), d_output, img_size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}
