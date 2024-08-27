#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define N 1024 * 1024 * 32  // Array size

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// Naive reduction kernel
__global__ void naive_reduction(const int *input, int *output, int size) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < size) {
        sdata[tid] = input[i] + ((i + blockDim.x < size) ? input[i + blockDim.x] : 0);
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Divergence-free reduction kernel
__global__ void divergence_free_reduction(const int *input, int *output, int size) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < size) {
        sdata[tid] = input[i] + ((i + blockDim.x < size) ? input[i + blockDim.x] : 0);
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int *vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Warp shuffle reduction kernel
__global__ void warp_shuffle_reduction(const int *input, int *output, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int sum = (i < size) ? input[i] + ((i + blockDim.x < size) ? input[i + blockDim.x] : 0) : 0;

    // Perform warp-level reduction within each warp
    for (unsigned int s = warpSize / 2; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
    }

    // Write the result of each warp to shared memory
    if (tid % warpSize == 0) {
        output[blockIdx.x * blockDim.x / warpSize + tid / warpSize] = sum;
    }

    // Synchronize to make sure all warps have written their results
    __syncthreads();

    // Final reduction within each block using the results from each warp
    if (tid < blockDim.x / warpSize) {
        sum = output[blockIdx.x * blockDim.x / warpSize + tid];
        for (unsigned int s = blockDim.x / (2 * warpSize); s > 0; s >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
        }

        // Only the first thread writes the final block result
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}


// CPU reduction function
int cpu_reduction(const std::vector<int>& data) {
    int sum = 0;
    for (int i = 0; i < data.size(); i++) {
        sum += data[i];
    }
    return sum;
}

// Final reduction on the GPU to sum up all block results
__global__ void final_reduction(int *input, int *output, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    *output = sum;
}

int main() {
    std::vector<int> data(N);

    // Initialize the data with random values
    for (int i = 0; i < N; i++) {
        data[i] = std::rand() % 100;
    }

    int *d_input, *d_output, *d_final_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, (N / 1024) * sizeof(int));
    cudaMalloc(&d_final_output, sizeof(int));
    cudaCheckError();

    cudaMemcpy(d_input, data.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();

    auto start = std::chrono::high_resolution_clock::now();
    int cpu_result = cpu_reduction(data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end - start;
    std::cout << "CPU reduction result: " << cpu_result << " Time: " << cpu_duration.count() << "s\n";

    int block_size = 1024;
    int grid_size = (N + block_size * 2 - 1) / (block_size * 2);

    // Naive GPU reduction
    start = std::chrono::high_resolution_clock::now();
    naive_reduction<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    final_reduction<<<1, 1>>>(d_output, d_final_output, grid_size);
    cudaDeviceSynchronize();
    cudaCheckError();
    int naive_gpu_result;
    cudaMemcpy(&naive_gpu_result, d_final_output, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_naive_duration = end - start;
    std::cout << "Naive GPU reduction result: " << naive_gpu_result << " Time: " << gpu_naive_duration.count() << "s\n";

    // Divergence-free GPU reduction
    start = std::chrono::high_resolution_clock::now();
    divergence_free_reduction<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    final_reduction<<<1, 1>>>(d_output, d_final_output, grid_size);
    cudaDeviceSynchronize();
    cudaCheckError();
    int divergence_free_gpu_result;
    cudaMemcpy(&divergence_free_gpu_result, d_final_output, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_div_free_duration = end - start;
    std::cout << "Divergence-free GPU reduction result: " << divergence_free_gpu_result << " Time: " << gpu_div_free_duration.count() << "s\n";

    // Warp shuffle GPU reduction
    start = std::chrono::high_resolution_clock::now();
    warp_shuffle_reduction<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    final_reduction<<<1, 1>>>(d_output, d_final_output, grid_size);
    cudaDeviceSynchronize();
    cudaCheckError();
    int warp_shuffle_gpu_result;
    cudaMemcpy(&warp_shuffle_gpu_result, d_final_output, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_warp_duration = end - start;
    std::cout << "Warp shuffle GPU reduction result: " << warp_shuffle_gpu_result << " Time: " << gpu_warp_duration.count() << "s\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_final_output);
    cudaCheckError();

    return 0;
}

