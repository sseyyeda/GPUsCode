#include <iostream>
#include <cuda_runtime.h>

__global__ void exampleKernel(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory for the block
    __shared__ int sharedData[256];

    if (idx < N) {
        // Load data into shared memory
        sharedData[threadIdx.x] = data[idx];
        
        // Barrier synchronization for all threads in the block
        __syncthreads();

        // Perform some operation (e.g., increment each element)
        sharedData[threadIdx.x] += 1;
        
        // Barrier synchronization for all threads in the block
        __syncthreads();

        // Write back to global memory
        data[idx] = sharedData[threadIdx.x];

        // Warp-level synchronization example
        int laneIdx = threadIdx.x % warpSize;
        if (laneIdx == 0) {
            // Only one thread in each warp prints the message
            printf("Warp %d reached this point\n", threadIdx.x / warpSize);
        }

        // Synchronize all threads in the warp
        __syncwarp();

        // Continue with other operations
    }
}

int main() {
    int N = 1024;
    int *data = new int[N];
    int *d_data;

    // Initialize data
    for (int i = 0; i < N; ++i) {
        data[i] = i;
    }

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with 4 blocks and 256 threads per block
    exampleKernel<<<4, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaMemcpy(data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first 10 results
    for (int i = 0; i < 10; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_data);
    delete[] data;

    return 0;
}

