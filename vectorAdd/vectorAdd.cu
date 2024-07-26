#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

size_t getAvailableGPUMemory() {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "Error getting GPU memory info: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }
    return free_mem;
}

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>

size_t getAvailableCPUMemoryWindows() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return status.ullAvailPhys;
    } else {
        std::cerr << "Error getting CPU memory info." << std::endl;
        return 0;
    }
}
#else
#include <sys/sysinfo.h>

size_t getAvailableCPUMemoryLinux() {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    } else {
        std::cerr << "Error getting CPU memory info." << std::endl;
        return 0;
    }
}
#endif

size_t getAvailableCPUMemory() {
#ifdef _WIN32
    return getAvailableCPUMemoryWindows();
#else
    return getAvailableCPUMemoryLinux();
#endif
}

size_t getMaxArraySize() {
    size_t cpu_mem = getAvailableCPUMemory();
    size_t gpu_mem = getAvailableGPUMemory();

    if (cpu_mem == 0 || gpu_mem == 0) {
        std::cerr << "Error: Failed to retrieve memory info." << std::endl;
        return 0;
    }

    // Determine the minimum available memory
    size_t min_mem = std::min(cpu_mem, gpu_mem);

    // Since we need three arrays (A, B, C), each of the same size,
    // and they are of type float, each element takes 4 bytes.
    // So, divide the available memory by 3 * sizeof(float).
    size_t max_elements = min_mem / (3 * sizeof(float));

    return max_elements;
}

// Function to calculate the optimal block size
int getOptimalThreadsPerBlock(size_t problemSize) {
    int threadsPerBlock = 256;

    // Example heuristic based on problem size
    if (problemSize <= 1024) {
        threadsPerBlock = 256;
    } else if (problemSize <= 4096) {
        threadsPerBlock = 512;
    } else {
        threadsPerBlock = 1024; // Maximum value for many GPUs
    }

    // Ensure the threads per block is a multiple of the warp size (32)
    if (threadsPerBlock % 32 != 0) {
        threadsPerBlock = (threadsPerBlock / 32) * 32;
    }

    return threadsPerBlock;
}

// Kernel for vector addition
__global__ void vectorAddKernel(float *A, float *B, float *C, size_t N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Function to add vectors on the CPU
void vectorAddCPU(const float *A, const float *B, float *C, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t max_array_size = getMaxArraySize();
    if (max_array_size > 0) {
        std::cout << "Maximum number of elements for arrays A, B, and C: " << max_array_size << std::endl;
    } else {
        std::cerr << "Failed to determine the maximum array size." << std::endl;
        return 0;
    }

    int THREADS_PER_BLOCK = getOptimalThreadsPerBlock(max_array_size);
    int BLOCK_SIZE = (max_array_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate memory on the host
    float *h_A = new float[max_array_size];
    float *h_B = new float[max_array_size];
    float *h_C = new float[max_array_size];

    // Initialize host arrays
    for (size_t i = 0; i < max_array_size; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, max_array_size * sizeof(float));
    cudaMalloc(&d_B, max_array_size * sizeof(float));
    cudaMalloc(&d_C, max_array_size * sizeof(float));

    // Copy host arrays to device
    cudaMemcpy(d_A, h_A, max_array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, max_array_size * sizeof(float), cudaMemcpyHostToDevice);

    // Measure GPU time
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAddKernel<<<BLOCK_SIZE, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, max_array_size);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;
    std::cout << "Time taken by GPU: " << gpu_duration.count() << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C, d_C, max_array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure CPU time
    float *h_C_cpu = new float[max_array_size];
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, max_array_size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "Time taken by CPU: " << cpu_duration.count() << " ms" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;

    return 0;
}
