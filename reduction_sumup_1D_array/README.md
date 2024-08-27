# GPU Reduction Techniques

This repository contains CUDA code that implements and compares different reduction techniques for summing up array elements on both CPU and GPU. The techniques included are:

1. Naive GPU Reduction
2. Divergence-Free GPU Reduction
3. Warp Shuffle GPU Reduction

## Directory Structure

The code is located in the `reduction_sumup_1D_array` folder. 

## Techniques Explained

### 1. Naive GPU Reduction

The naive reduction technique involves each thread performing a reduction operation and then synchronizing to perform a final reduction. This method does not optimize for warp divergence or shared memory access patterns.

### 2. Divergence-Free GPU Reduction

This technique improves upon the naive approach by minimizing warp divergence. It reduces the need for synchronization by performing operations within each warp and then uses volatile shared memory for the final aggregation of results.

### 3. Warp Shuffle GPU Reduction

The warp shuffle reduction technique further optimizes the reduction by using warp shuffle operations to reduce the need for shared memory accesses and synchronization, making it highly efficient.

## Compilation and Execution

### Prerequisites

- NVIDIA CUDA Toolkit
- A CUDA-capable GPU

### Compilation

To compile the CUDA code, navigate to the `reduction_sumup_1D_array` folder and run:

```bash
nvcc -o reduction_sumup_1D_array reduction_sumup_1D_array.cu
```

### Running the Code

To run the compiled code, use:

```bash
./reduction_sumup_1D_array
```

## Expected Results

When you run the code, you should see output similar to the following:

```
CPU reduction result: 1661175516 Time: 0.10673s
Naive GPU reduction result: 1661175516 Time: 0.00848055s
Divergence-free GPU reduction result: 1661175516 Time: 0.00558506s
Warp shuffle GPU reduction result: 1661175516 Time: 0.00544538s
```

## Performance Comparison

### Timing Results

- **CPU Reduction**:
  - Result: `1661175516`
  - Time: `0.10673s`

- **Naive GPU Reduction**:
  - Result: `1661175516`
  - Time: `0.00848055s`

- **Divergence-Free GPU Reduction**:
  - Result: `1661175516`
  - Time: `0.00558506s`

- **Warp Shuffle GPU Reduction**:
  - Result: `1661175516`
  - Time: `0.00544538s`

### Speed-Up Calculations

1. **Naive GPU vs. CPU**:
   - Speed-Up = CPU Time / Naive GPU Time
   - Speed-Up = 0.10673 / 0.00848055 ≈ 12.6x
   - The naive GPU reduction is approximately `12.6x` faster than the CPU reduction.

2. **Divergence-Free GPU vs. CPU**:
   - Speed-Up = CPU Time / Divergence-Free GPU Time
   - Speed-Up = 0.10673 / 0.00558506 ≈ 19.1x
   - The divergence-free GPU reduction is approximately `19.1x` faster than the CPU reduction.

3. **Warp Shuffle GPU vs. CPU**:
   - Speed-Up = CPU Time / Warp Shuffle GPU Time
   - Speed-Up = 0.10673 / 0.00544538 ≈ 19.6x
   - The warp shuffle GPU reduction is approximately `19.6x` faster than the CPU reduction.

### Relative Performance of GPU Techniques

1. **Divergence-Free GPU vs. Naive GPU**:
   - Speed-Up = Naive GPU Time / Divergence-Free GPU Time
   - Speed-Up = 0.00848055 / 0.00558506 ≈ 1.5x
   - The divergence-free GPU reduction is about `1.5x` faster than the naive GPU reduction.

2. **Warp Shuffle GPU vs. Divergence-Free GPU**:
   - Speed-Up = Divergence-Free GPU Time / Warp Shuffle GPU Time
   - Speed-Up = 0.00558506 / 0.00544538 ≈ 1.03x
   - The warp shuffle GPU reduction is slightly faster than the divergence-free GPU reduction, approximately `3%` faster.

## Conclusion

- **Fastest Method**: Warp Shuffle GPU reduction (`0.00544538s`, approximately `19.6x` faster than CPU).
- **Second Fastest**: Divergence-Free GPU reduction (`0.00558506s`, approximately `19.1x` faster than CPU, `1.5x` faster than Naive GPU).
- **Third Fastest**: Naive GPU reduction (`0.00848055s`, approximately `12.6x` faster than CPU).
- **Slowest Method**: CPU reduction (`0.10673s`).
