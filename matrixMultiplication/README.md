# Matrix Multiplication with CUDA

This project demonstrates matrix multiplication using CUDA, with implementations optimized and non-optimized. It includes a CPU-based matrix multiplication implementation for comparison. The project evaluates performance using CUDA's shared memory optimization.

## Project Structure

- **`CMakeLists.txt`**: CMake configuration file for building the project.
- **`src/main.cu`**: The main CUDA file containing matrix multiplication implementations, both with and without shared memory optimization.

## Building the Project

To compile and run the code, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/sseyyeda/GPUsCode.git
    cd GPUsCode/matrixMultiplication/
    ```

2. **Create a Build Directory:**
    ```bash
    mkdir build
    cd build
    ```

3. **Run CMake to Configure the Project:**
    ```bash
    cmake ..
    ```

4. **Build the Project:**
    ```bash
    make
    ```

   This will compile the code and generate an executable named `main_gpu`.

## Running the Code

After successful compilation, you can run the executable with:

```bash
./main_gpu
```

## Matrix Multiplication Implementations

### 1. **CPU Implementation**

The CPU-based matrix multiplication uses nested loops. It is used as a baseline for performance comparison.

### 2. **GPU Implementation (No Shared Memory Optimization)**

This CUDA implementation uses global memory for all operations. Each thread computes one element of the result matrix, which can lead to suboptimal performance due to global memory access latency.

**Function Signature:**
```cpp
void matrixMultiplyGPU(float* A, float* B, float* C, int rowsA, int colsA, int colsB);
```

### 3. **GPU Implementation (Shared Memory Optimized)**

This CUDA implementation uses shared memory to cache sub-matrices of `A` and `B`, reducing global memory accesses and improving performance.

**Function Signature:**
```cpp
void matrixMultiplyGPUShared(float* A, float* B, float* C, int rowsA, int colsA, int colsB);
```

**Shared Memory Optimization:**

Shared memory is used to store sub-matrices of `A` and `B` for each block, reducing global memory access and improving performance. Threads within a block load data into shared memory, perform computations, and then write results back to global memory.

## Sample Output

Here is a sample output from running the GPU matrix multiplication code:

```
CPU Time: 0.175485 seconds
GPU Time (No Shared Memory): 0.0927734 seconds
GPU Time (Shared Memory): 0.00257207 seconds
```

## Dependencies

- **CUDA Toolkit**: Ensure CUDA is installed and properly configured.
- **OpenCV**: Required for matrix generation and timing.
- **OpenMP**: For parallel processing on CPU (if used).


