#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

// Define CUDA error checking macro
#define CUDA_ERROR_CHECK(call)                                                \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define N 1024  // Input matrix size (N x N)
#define K 3     // Kernel size (K x K)

// Define constant memory size for kernel
__constant__ float d_kernelConst[K * K];

// CPU Convolution function
void cpuConvolution(const std::vector<float>& input, const std::vector<float>& kernel, std::vector<float>& output, int n, int k) {
    int pad = k / 2;
    for (int i = pad; i < n - pad; ++i) {
        for (int j = pad; j < n - pad; ++j) {
            float sum = 0.0f;
            for (int ki = -pad; ki <= pad; ++ki) {
                for (int kj = -pad; kj <= pad; ++kj) {
                    sum += input[(i + ki) * n + (j + kj)] * kernel[(ki + pad) * k + (kj + pad)];
                }
            }
            output[i * n + j] = sum;
        }
    }
}

// CUDA Convolution Kernel
__global__ void cudaConvolution(float* input, float* kernel, float* output, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = k / 2;

    if (i >= pad && i < n - pad && j >= pad && j < n - pad) {
        float sum = 0.0f;
        for (int ki = -pad; ki <= pad; ++ki) {
            for (int kj = -pad; kj <= pad; ++kj) {
                sum += input[(i + ki) * n + (j + kj)] * kernel[(ki + pad) * k + (kj + pad)];
            }
        }
        output[i * n + j] = sum;
    }
}

// CUDA Convolution Kernel using Constant Memory
__global__ void cudaConvolutionConstMem(float* input, float* output, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = k / 2;

    if (i >= pad && i < n - pad && j >= pad && j < n - pad) {
        float sum = 0.0f;
        for (int ki = -pad; ki <= pad; ++ki) {
            for (int kj = -pad; kj <= pad; ++kj) {
                sum += input[(i + ki) * n + (j + kj)] * d_kernelConst[(ki + pad) * k + (kj + pad)];
            }
        }
        output[i * n + j] = sum;
    }
}

int main() {
    int inputSize = N * N;
    int kernelSize = K * K;
    
    // Host memory allocation
    std::vector<float> h_input(inputSize);
    std::vector<float> h_kernel(kernelSize);
    std::vector<float> h_outputCPU(inputSize);
    std::vector<float> h_outputGPU(inputSize);
    std::vector<float> h_outputConstMem(inputSize);

    // Initialize input and kernel with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < inputSize; ++i) h_input[i] = dis(gen);
    for (int i = 0; i < kernelSize; ++i) h_kernel[i] = dis(gen);

    // CPU Convolution and time measurement
    auto start = std::chrono::high_resolution_clock::now();
    cpuConvolution(h_input, h_kernel, h_outputCPU, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = end - start;
    std::cout << "CPU Convolution Time: " << cpuDuration.count() << " ms\n";

    // Device memory allocation
    float *d_input, *d_kernel, *d_output;
    CUDA_ERROR_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_kernel, kernelSize * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&d_output, inputSize * sizeof(float)));

    // Copy data to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_input, h_input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    // Load kernel to constant memory
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(d_kernelConst, h_kernel.data(), kernelSize * sizeof(float)));

    // CUDA Convolution and time measurement
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t startEvent, stopEvent;
    CUDA_ERROR_CHECK(cudaEventCreate(&startEvent));
    CUDA_ERROR_CHECK(cudaEventCreate(&stopEvent));
    
    // Measure time for unoptimized CUDA kernel
    CUDA_ERROR_CHECK(cudaEventRecord(startEvent, 0));
    cudaConvolution<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_kernel, d_output, N, K);
    CUDA_ERROR_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stopEvent));
    float gpuDuration = 0;
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent));
    std::cout << "CUDA Convolution Time: " << gpuDuration << " ms\n";

    // Copy result back to host
    CUDA_ERROR_CHECK(cudaMemcpy(h_outputGPU.data(), d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Measure time for CUDA kernel with constant memory
    CUDA_ERROR_CHECK(cudaEventRecord(startEvent, 0));
    cudaConvolutionConstMem<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, K);
    CUDA_ERROR_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_ERROR_CHECK(cudaEventSynchronize(stopEvent));
    float constMemDuration = 0;
    CUDA_ERROR_CHECK(cudaEventElapsedTime(&constMemDuration, startEvent, stopEvent));
    std::cout << "CUDA Convolution with Constant Memory Time: " << constMemDuration << " ms\n";

    // Copy result back to host
    CUDA_ERROR_CHECK(cudaMemcpy(h_outputConstMem.data(), d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_ERROR_CHECK(cudaFree(d_input));
    CUDA_ERROR_CHECK(cudaFree(d_kernel));
    CUDA_ERROR_CHECK(cudaFree(d_output));
    CUDA_ERROR_CHECK(cudaEventDestroy(startEvent));
    CUDA_ERROR_CHECK(cudaEventDestroy(stopEvent));

    // Compare results between CPU and GPU
    bool matchGPU = true;
    for (int i = 0; i < inputSize; ++i) {
        if (fabs(h_outputCPU[i] - h_outputGPU[i]) > 1e-5) {
            matchGPU = false;
            break;
        }
    }
    std::cout << "Results match with CUDA kernel: " << (matchGPU ? "Yes" : "No") << "\n";

    bool matchConstMem = true;
    for (int i = 0; i < inputSize; ++i) {
        if (fabs(h_outputCPU[i] - h_outputConstMem[i]) > 1e-5) {
            matchConstMem = false;
            break;
        }
    }
    std::cout << "Results match with CUDA kernel using Constant Memory: " << (matchConstMem ? "Yes" : "No") << "\n";

    return 0;
}

