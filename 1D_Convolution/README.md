# 1D Convolution Comparison

This repository contains a detailed comparison of 1D convolution implementations on both CPU and GPU, including a specialized CUDA kernel that leverages constant memory for performance optimization.

## Folder Structure

The folder `1D_Convolution` contains the following files:

- `1D_convolution.cu`: Source code implementing CPU and CUDA convolution functions, including an optimized CUDA kernel using constant memory.
- `README.md`: This file.

## Overview

### Convolution Implementations

1. **CPU Convolution**:
   - Implements a straightforward 1D convolution on the CPU. This serves as a baseline for performance comparison.

2. **CUDA Convolution**:
   - Executes 1D convolution on the GPU using global memory. This provides a comparison point for evaluating performance improvements achieved through CUDA.

3. **CUDA Convolution with Constant Memory**:
   - An optimized CUDA convolution kernel that utilizes constant memory to store the convolution kernel. This optimization aims to enhance performance by leveraging faster access times for constant memory.

## Code Explanation

### CPU Convolution Function

The `cpuConvolution` function performs a 1D convolution on the CPU. It iterates over each pixel of the input matrix, applies the kernel, and stores the result in the output matrix. This method is straightforward but may be slower compared to GPU implementations for large datasets due to the CPU's serial processing nature.

### CUDA Convolution Kernel

The `cudaConvolution` kernel function performs the convolution operation on the GPU using global memory. Each thread processes a single element of the output matrix, applying the kernel to the corresponding region of the input matrix.

### CUDA Convolution with Constant Memory

In the `cudaConvolutionWithConstantMemory` kernel, the convolution kernel is stored in constant memory. Constant memory is a special type of memory on the GPU that is optimized for read-only data that remains unchanged throughout kernel execution. Here’s a detailed look at its usage and benefits:

**Constant Memory Usage:**
- **Read-Only Data**: Constant memory is ideal for storing data that is read-only during kernel execution, such as fixed convolution kernels. This is because constant memory is cached and can be accessed faster compared to global memory.
- **Caching and Performance**: When all threads in a warp access the same data from constant memory, the data is cached, resulting in lower latency and higher throughput. This is particularly advantageous for kernels where the same kernel matrix is accessed by multiple threads.

**Benefits of Constant Memory:**
- **Faster Access**: Constant memory access times are faster due to caching, which can significantly improve performance for read-only data that is accessed by all threads.
- **Reduced Global Memory Traffic**: By storing frequently accessed but unchanging data in constant memory, the amount of global memory traffic is reduced, which can lead to better overall performance.

### Cloning the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/sseyyeda/GPUsCode.git
cd GPUsCode/1D_Convolution
```
## Compilation

To compile the CUDA code, use the following command:

```bash
nvcc -o 1D_convolution 1D_convolution.cu
```

## Execution

Run the compiled executable:

```bash
./1D_convolution
```

## Sample Output

The following sample output illustrates the execution times for different convolution implementations and verifies the correctness of the results:

```
CPU Convolution Time: 49.105 ms
CUDA Convolution Time: 0.580608 ms
CUDA Convolution with Constant Memory Time: 0.45744 ms
Results match with CUDA kernel: Yes
Results match with CUDA kernel using Constant Memory: Yes
```

## Key Points

- **CPU Convolution**: Computes the convolution on the CPU. Time measured for reference.
- **CUDA Convolution**: Computes the convolution on the GPU using global memory.
- **CUDA Convolution with Constant Memory**: Computes the convolution on the GPU using constant memory to store the kernel, which can be faster due to more efficient memory access patterns.

## Usage

1. **Compile** the code using the provided `nvcc` command.
2. **Run** the executable to see the performance and correctness results.
3. **Verify** that the results match between CPU and CUDA implementations, both with and without constant memory optimization.


