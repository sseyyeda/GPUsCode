#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

const int NUM_ITERATIONS = 100000;  // 1 million iterations
const int THREADS_PER_BLOCK = 512;
const int NUM_BLOCKS = (NUM_ITERATIONS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

// GPU kernel to print "helloworld" with thread ID
__global__ void printHelloWorldFromGPU(int num_iterations) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = threadID; i < num_iterations; i += gridDim.x * blockDim.x) {
        printf("helloworld from GPU, threadID: %d\n", threadID);
    }
}

int main() {
    // Measure time for CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        printf("helloworld from CPU: %d\n", i);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    // Measure time for GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    printHelloWorldFromGPU<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(NUM_ITERATIONS);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    printf("Time taken by GPU: %.3f ms\n", gpu_duration.count());
    printf("Time taken by CPU: %.3f ms\n", cpu_duration.count());

    return 0;
}
