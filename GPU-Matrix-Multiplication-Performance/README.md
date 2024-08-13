# Matrix Multiplication Performance Comparison

This project compares the performance of matrix multiplication using various approaches on the CPU and GPU, including:

1. **CPU Matrix Multiplication**: Baseline matrix multiplication using standard C++.
2. **CUDA Matrix Multiplication without Shared Memory**: Basic CUDA implementation without memory optimization.
3. **CUDA Matrix Multiplication with Shared Memory**: Optimized CUDA implementation utilizing shared memory.
4. **CUDA Matrix Multiplication with Thread Coarsening**: CUDA implementation with thread coarsening to improve computational throughput.
5. **CUDA Tiled and Coarsened Matrix Multiplication**: A combination of tiling (using shared memory) and thread coarsening for maximum performance.

## Project Structure

- `matrix_mul_comparison.cu`: The main CUDA C++ source file containing implementations of the different matrix multiplication methods.
- `README.md`: This file, which explains the project, how to compile and run the code, and provides a sample output.

## Getting Started

### Prerequisites

- **CUDA Toolkit**: Ensure you have the CUDA Toolkit installed on your system.
- **C++ Compiler**: A C++ compiler with CUDA support (e.g., `nvcc`).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sseyyeda/GPUsCode.git
   cd GPUsCode/GPU-Matrix-Multiplication-Performance

### Compiling the Code

Use the NVIDIA CUDA Compiler `nvcc` to compile the `matrix_mul_comparison.cu` file:

```bash
nvcc -o matrix_mul_comparison matrix_mul_comparison.cu
```

### Running the Code

After compiling, run the executable:

```bash
./matrix_mul_comparison
```

### Expected Output

When you run the code, you should see output similar to the following, with execution times depending on your hardware configuration:

```
Determined tile size: 32
CPU Matrix Multiplication execution time: 6785.81 ms
CUDA Matrix Multiplication without Shared Memory execution time: 29.6475 ms
CUDA Matrix Multiplication with Shared Memory execution time: 17.3695 ms
CUDA Matrix Multiplication with Thread Coarsening execution time: 33.6202 ms
CUDA Tiled and Coarsened Matrix Multiplication execution time: 1.70057 ms
```

## Detailed Explanation

### Matrix Multiplication Approaches

#### 1. CPU Matrix Multiplication
- A straightforward implementation using nested loops.
- Performance is often significantly slower compared to GPU implementations due to the lack of parallelization.

#### 2. CUDA Matrix Multiplication without Shared Memory
- A basic CUDA implementation where each thread computes a single element of the resulting matrix.
- All data is read directly from global memory, which can lead to poor performance due to high memory latency.

#### 3. CUDA Matrix Multiplication with Shared Memory
- Utilizes shared memory to load tiles of the matrix into faster on-chip memory.
- This reduces global memory accesses, leading to better performance.

#### 4. CUDA Matrix Multiplication with Thread Coarsening
- Each thread computes multiple elements of the result matrix, reducing the number of active threads.
- This approach can increase computational throughput but may not always be optimal depending on the matrix size and hardware.

#### 5. CUDA Tiled and Coarsened Matrix Multiplication
- Combines both tiling (shared memory) and thread coarsening.
- This approach maximizes the efficiency of memory usage and computational throughput, often yielding the best performance.

### Performance Considerations

- **Tile Size Determination**: The tile size is dynamically determined based on the available shared memory on the GPU. This ensures that the code can run efficiently on different architectures without the need for recompilation.
- **Thread Coarsening**: The level of thread coarsening is chosen to balance the number of threads and the workload per thread, optimizing for different hardware configurations.

### Conclusion

The results demonstrate the significant performance gains that can be achieved by optimizing matrix multiplication on the GPU using shared memory, thread coarsening, and tiling. The CUDA Tiled and Coarsened Matrix Multiplication typically offers the best performance, as seen in the sample output.

