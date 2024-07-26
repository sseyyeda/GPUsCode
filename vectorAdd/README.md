# VECTORADD

Welcome to the `GPUsCode` repository! This repository contains various CUDA programs for GPU programming and examples of vector addition operations. This particular example demonstrates vector addition on both the CPU and GPU and compares their performance.

## Project Overview

The project includes a CUDA program to add two vectors on both the CPU and GPU. The purpose of this code is to compare the performance of vector addition operations on a GPU versus a CPU, taking into account available memory and optimal block size configuration for CUDA.

## Files

- `vector_addition.cu`: Contains the CUDA code for vector addition and performance measurement.

## Code Description

### `vector_addition.cu`

1. **Memory Measurement Functions:**
   - `getAvailableGPUMemory()`: Retrieves the available GPU memory.
   - `getAvailableCPUMemory()`: Retrieves the available CPU memory (Windows and Linux versions).

2. **Maximum Array Size Calculation:**
   - `getMaxArraySize()`: Determines the maximum size of the arrays that can be allocated based on the minimum of available CPU and GPU memory.

3. **Optimal Block Size Calculation:**
   - `getOptimalThreadsPerBlock()`: Determines an optimal number of threads per block based on the problem size.

4. **GPU Kernel:**
   - `vectorAddKernel()`: Performs vector addition on the GPU.

5. **CPU Vector Addition Function:**
   - `vectorAddCPU()`: Performs vector addition on the CPU.

6. **Main Function:**
   - Allocates memory for vectors on both the CPU and GPU.
   - Initializes vectors and performs vector addition on both the CPU and GPU.
   - Measures and prints the execution time for both operations.

## Compilation and Execution

To compile and run the code, follow these steps:

1. **Compile the Code:**
   ```bash
   nvcc -o vector_addition vector_addition.cu
   ```

2. **Run the Executable:**
   ```bash
   ./vector_addition
   ```

## Example Output

Here is an example output from running the code:

```
Maximum number of elements for arrays A, B, and C: [number]
Time taken by GPU: 109.195 ms
Time taken by CPU: 82.2941 ms
```

## Possible Reasons for Performance Differences

In this example, the GPU execution time is higher than the CPU execution time. This might occur due to several factors:

1. **Overhead of Kernel Launch:**
   - GPU operations involve overhead for launching kernels and transferring data between host and device. For small problem sizes, this overhead can outweigh the performance benefits of parallel processing on the GPU.

2. **Memory Transfer Time:**
   - The time taken to transfer data between the host and the GPU can be significant, especially for smaller datasets. This can result in longer overall execution times on the GPU compared to the CPU.

3. **Optimized CPU Code:**
   - Modern CPUs are highly optimized for certain operations and may have efficient implementations for vector operations, which can make them faster than a GPU for small-scale problems.

4. **GPU Configuration:**
   - The block size and number of threads per block might not be optimal for the problem size, affecting the performance. Tuning these parameters might improve GPU performance.

## Contribution

If you would like to contribute to this repository:
- Fork the repository and create a new branch for your changes.
- Make your changes and test them thoroughly.
- Submit a pull request with a detailed description of your changes.


