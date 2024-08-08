#include <iostream>
#include <cuda.h>
#include <chrono>

// Global and constant memory declarations
__device__ int GlobalVar;
__device__ __constant__ int ConstVar;

// Kernel function to measure register access time
__global__ void accessRegister(int* d_output) {
    int regVar = threadIdx.x;
    for (int i = 0; i < 10000; ++i) {
        regVar += i;  // Perform operations to simulate workload
    }
    d_output[threadIdx.x] = regVar;
}

// Kernel function to measure local memory access time
__global__ void accessLocal(int* d_local, int* d_output) {
    int localVar = d_local[threadIdx.x];
    for (int i = 0; i < 10000; ++i) {
        localVar += i;  // Simulate workload
    }
    d_output[threadIdx.x] = localVar;
}

// Kernel function to measure shared memory access time
__global__ void accessShared(int* d_output) {
    extern __shared__ int SharedVar[];
    SharedVar[threadIdx.x] = threadIdx.x;
    int sharedVar = SharedVar[threadIdx.x];
    for (int i = 0; i < 10000; ++i) {
        sharedVar += i;  // Simulate workload
    }
    d_output[threadIdx.x] = sharedVar;
}

// Kernel function to measure global memory access time
__global__ void accessGlobal(int* d_output) {
    int globalVar = GlobalVar;
    for (int i = 0; i < 10000; ++i) {
        globalVar += i;  // Simulate workload
    }
    d_output[threadIdx.x] = globalVar;
}

// Kernel function to measure constant memory access time
__global__ void accessConstant(int* d_output) {
    int constantVar = ConstVar;
    for (int i = 0; i < 10000; ++i) {
        constantVar += i;  // Simulate workload
    }
    d_output[threadIdx.x] = constantVar;
}

// Utility function to measure execution time of a kernel
template <typename KernelFunction>
void measureKernelExecutionTime(KernelFunction kernel, int* d_input, int* d_output, size_t sharedMemSize, const char* label) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<1, 1024, sharedMemSize>>>(d_output);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << label << " execution time: " << elapsed.count() * 1e3 << " ms" << std::endl;
}

// Overloaded version for kernels with input arguments
template <typename KernelFunction>
void measureKernelExecutionTime(KernelFunction kernel, int* d_input, int* d_output, const char* label) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<1, 1024>>>(d_input, d_output);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << label << " execution time: " << elapsed.count() * 1e3 << " ms" << std::endl;
}

int main() {
    const int numThreads = 1024;
    const int dataSize = numThreads * sizeof(int);

    // Host variables
    int h_local[numThreads];

    // Initialize host variables
    for (int i = 0; i < numThreads; ++i) {
        h_local[i] = i;
    }

    // Device variables
    int *d_local, *d_output;
    cudaMalloc(&d_local, dataSize);
    cudaMalloc(&d_output, dataSize);

    cudaMemcpy(d_local, h_local, dataSize, cudaMemcpyHostToDevice);

    // Assign values to global and constant variables
    cudaMemcpyToSymbol(GlobalVar, &h_local[0], sizeof(int));
    cudaMemcpyToSymbol(ConstVar, &h_local[0], sizeof(int));

    // Measure and compare the execution times
    measureKernelExecutionTime(accessRegister, nullptr, d_output, 0, "Register");
    measureKernelExecutionTime(accessLocal, d_local, d_output, "Local");
    measureKernelExecutionTime(accessShared, nullptr, d_output, dataSize, "Shared");
    measureKernelExecutionTime(accessGlobal, nullptr, d_output, 0, "Global");
    measureKernelExecutionTime(accessConstant, nullptr, d_output, 0, "Constant");

    // Clean up
    cudaFree(d_local);
    cudaFree(d_output);

    return 0;
}

