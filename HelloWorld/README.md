# HelloWorld

This repository contains various CUDA programs for exploring and benchmarking GPU programming techniques.

## HelloWorld

The `HelloWorld` folder contains a simple CUDA program designed to compare the performance of CPU and GPU for printing the phrase "helloworld" multiple times. This example demonstrates basic GPU programming concepts and provides a benchmark to understand the performance differences between CPU and GPU for repetitive tasks.

### Purpose

The purpose of this program is to:
- Demonstrate how to write and run a basic CUDA program.
- Compare the execution time of CPU versus GPU for a simple task (printing "helloworld").
- Illustrate the use of CUDA kernel functions and how to manage thread execution on the GPU.

### Code

- `helloworld.cu`: This file contains the CUDA code that performs the following:
  - Prints "helloworld" from the CPU.
  - Uses GPU threads to print "helloworld" and includes thread IDs to differentiate between threads.
  - Measures and compares the time taken for both CPU and GPU executions.

### Compilation and Execution

To compile and run the CUDA program, follow these steps:

1. **Compile** the CUDA code using the NVIDIA CUDA Compiler (`nvcc`):
   ```bash
   nvcc -o helloworld helloworld.cu

2. Run the compiled executable:
  ./helloworld


