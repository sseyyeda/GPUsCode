#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#define WIDTH 512
#define HEIGHT 512

// CPU Matrix Multiplication
void matrixMultiplyCPU(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float sum = 0.0f;
            for (int k = 0; k < colsA; k++) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
}

// GPU Matrix Multiplication without shared memory optimization
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

void matrixMultiplyGPU(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    float *d_A, *d_B, *d_C;
    size_t size_A = rowsA * colsA * sizeof(float);
    size_t size_B = colsA * colsB * sizeof(float);
    size_t size_C = rowsA * colsB * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// GPU Matrix Multiplication with shared memory optimization
__global__ void matrixMultiplyKernelShared(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    __shared__ float shared_A[16][16];
    __shared__ float shared_B[16][16];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < (colsA + 16 - 1) / 16; i++) {
        if (i * 16 + threadIdx.x < colsA && row < rowsA) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * colsA + i * 16 + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (i * 16 + threadIdx.y < colsA && col < colsB) {
            shared_B[threadIdx.y][threadIdx.x] = B[(i * 16 + threadIdx.y) * colsB + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < 16; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = sum;
    }
}

void matrixMultiplyGPUShared(float* A, float* B, float* C, int rowsA, int colsA, int colsB) {
    float *d_A, *d_B, *d_C;
    size_t size_A = rowsA * colsA * sizeof(float);
    size_t size_B = colsA * colsB * sizeof(float);
    size_t size_C = rowsA * colsB * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiplyKernelShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int rowsA = WIDTH, colsA = WIDTH, colsB = WIDTH;
    float *A = new float[rowsA * colsA];
    float *B = new float[colsA * colsB];
    float *C = new float[rowsA * colsB];

    // Initialize matrices with random values
    for (int i = 0; i < rowsA * colsA; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < colsA * colsB; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C, rowsA, colsA, colsB);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "CPU Time: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyGPU(A, B, C, rowsA, colsA, colsB);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "GPU Time (No Shared Memory): " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyGPUShared(A, B, C, rowsA, colsA, colsB);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "GPU Time (Shared Memory): " << duration.count() << " seconds" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

