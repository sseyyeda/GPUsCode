#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

// Helper function to initialize matrices
void initializeMatrix(std::vector<float> &matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// CPU Matrix Multiplication
void cpuMatrixMultiplication(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA Kernel for Matrix Multiplication without Shared Memory
__global__ void cudaMatrixMulNoShared(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA Kernel for Matrix Multiplication with Shared Memory
__global__ void cudaMatrixMulShared(float *A, float *B, float *C, int N, int tileSize) {
    extern __shared__ float sharedMem[];
    float* As = sharedMem;
    float* Bs = sharedMem + tileSize * tileSize;

    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + tileSize - 1) / tileSize; t++) {
        if (row < N && t * tileSize + threadIdx.x < N)
            As[threadIdx.y * tileSize + threadIdx.x] = A[row * N + t * tileSize + threadIdx.x];
        else
            As[threadIdx.y * tileSize + threadIdx.x] = 0.0;

        if (col < N && t * tileSize + threadIdx.y < N)
            Bs[threadIdx.y * tileSize + threadIdx.x] = B[(t * tileSize + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y * tileSize + threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < tileSize; k++) {
            sum += As[threadIdx.y * tileSize + k] * Bs[k * tileSize + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CUDA Kernel for Matrix Multiplication with Thread Coarsening
__global__ void cudaMatrixMulCoarsened(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int col = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            if (row + dy < N && col + dx < N) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[(row + dy) * N + k] * B[k * N + (col + dx)];
                }
                C[(row + dy) * N + (col + dx)] = sum;
            }
        }
    }
}

// CUDA Kernel for Tiled and Coarsened Matrix Multiplication
__global__ void cudaMatrixMulTiledCoarsened(float *A, float *B, float *C, int N, int tileSize) {
    extern __shared__ float sharedMem[];
    float* As = sharedMem;
    float* Bs = sharedMem + tileSize * tileSize;

    int row = blockIdx.y * tileSize * 2 + threadIdx.y;
    int col = blockIdx.x * tileSize * 2 + threadIdx.x;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int t = 0; t < (N + tileSize - 1) / tileSize; t++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (row + i * tileSize < N && t * tileSize + threadIdx.x < N)
                    As[(threadIdx.y + i * tileSize) * tileSize + threadIdx.x] = A[(row + i * tileSize) * N + t * tileSize + threadIdx.x];
                else
                    As[(threadIdx.y + i * tileSize) * tileSize + threadIdx.x] = 0.0;

                if (col + j * tileSize < N && t * tileSize + threadIdx.y < N)
                    Bs[(threadIdx.y + j * tileSize) * tileSize + threadIdx.x] = B[(t * tileSize + threadIdx.y) * N + col + j * tileSize];
                else
                    Bs[(threadIdx.y + j * tileSize) * tileSize + threadIdx.x] = 0.0;
            }
        }

        __syncthreads();

        for (int k = 0; k < tileSize; k++) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    sum[i * 2 + j] += As[(threadIdx.y + i * tileSize) * tileSize + k] * Bs[k * tileSize + threadIdx.x + j * tileSize];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (row + i * tileSize < N && col + j * tileSize < N) {
                C[(row + i * tileSize) * N + col + j * tileSize] = sum[i * 2 + j];
            }
        }
    }
}

// Function to compare runtimes
template <typename Func>
void timeExecution(Func f, const char* label) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << label << " execution time: " << elapsed.count() * 1e3 << " ms" << std::endl;
}

// Main Function
int main() {
    const int N = 1024;  // Matrix size
    size_t size = N * N * sizeof(float);

    // Allocate memory on the host
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N);

    // Initialize matrices
    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    // Query device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Determine the optimal tile size based on shared memory
    int maxTileSize = 32;
    int sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    int tileSize = sqrt(sharedMemPerBlock / 2 / sizeof(float));

    // Clamp the tile size to avoid exceeding warp size
    tileSize = (tileSize > maxTileSize) ? maxTileSize : tileSize;

    std::cout << "Determined tile size: " << tileSize << std::endl;

    // CPU Matrix Multiplication
    timeExecution([&] {
        cpuMatrixMultiplication(h_A, h_B, h_C, N);
    }, "CPU Matrix Multiplication");

    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // CUDA Matrix Multiplication without Shared Memory
    timeExecution([&] {
        dim3 threadsPerBlock(32, 32);
        dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);
        cudaMatrixMulNoShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    }, "CUDA Matrix Multiplication without Shared Memory");

    // CUDA Matrix Multiplication with Shared Memory
    timeExecution([&] {
        dim3 threadsPerBlock(tileSize, tileSize);
        dim3 blocksPerGrid((N + tileSize - 1) / tileSize, (N + tileSize - 1) / tileSize);
        int sharedMemSize = 2 * tileSize * tileSize * sizeof(float);
        cudaMatrixMulShared<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, N, tileSize);
        cudaDeviceSynchronize();
    }, "CUDA Matrix Multiplication with Shared Memory");

    // CUDA Matrix Multiplication with Thread Coarsening
    timeExecution([&] {
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((N + 31) / 32, (N + 31) / 32);
        cudaMatrixMulCoarsened<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    }, "CUDA Matrix Multiplication with Thread Coarsening");

    // CUDA Tiled and Coarsened Matrix Multiplication
    timeExecution([&] {
        dim3 threadsPerBlock(tileSize, tileSize);
        dim3 blocksPerGrid((N + 2 * tileSize - 1) / (2 * tileSize), (N + 2 * tileSize - 1) / (2 * tileSize));
        int sharedMemSize = 2 * tileSize * tileSize * sizeof(float);
        cudaMatrixMulTiledCoarsened<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_B, d_C, N, tileSize);
        cudaDeviceSynchronize();
    }, "CUDA Tiled and Coarsened Matrix Multiplication");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

