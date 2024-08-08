# CUDA Memory Access Comparison

This repository contains a CUDA program that demonstrates the different types of memory available in CUDA: register, local, shared, global, and constant memory. The program measures and compares the access times for each type of memory to help understand their performance characteristics.

## Memory Types in CUDA

### Register Memory
- **Scope**: Thread
- **Lifetime**: Grid
- **Description**: Registers are the fastest type of memory in CUDA. Each thread has its own set of registers, and they are not shared between threads. Registers are used to store frequently accessed variables.

### Local Memory
- **Scope**: Thread
- **Lifetime**: Grid
- **Description**: Local memory is private to each thread and is used when there are not enough registers to hold all variables. Local memory resides in device memory, so accessing it is slower than accessing registers.

### Shared Memory
- **Scope**: Block
- **Lifetime**: Grid
- **Description**: Shared memory is a fast, user-managed cache that is shared among threads within the same block. It is much faster than global memory but slower than registers.

### Global Memory
- **Scope**: Grid
- **Lifetime**: Application
- **Description**: Global memory is accessible by all threads in the grid. It is the slowest type of memory and should be used sparingly.

### Constant Memory
- **Scope**: Grid
- **Lifetime**: Application
- **Description**: Constant memory is read-only and cached. It is faster than global memory for read-only data when all threads read the same address.

## CUDA Variable Declaration Type Qualifiers

The following table summarizes the properties of different CUDA memory types based on their variable declaration type qualifiers:

| Variable Declaration                    | Memory   | Scope   | Lifetime     |
| ----------------------------------------| -------- | ------- | ------------ |
| Automatic variables other than arrays   | Register | Thread  | Grid         |
| Automatic array variables               | Local    | Thread  | Grid         |
| `__device__ __shared__ int SharedVar;`  | Shared   | Block   | Grid         |
| `__device__ int GlobalVar;`             | Global   | Grid    | Application  |
| `__device__ __constant__ int ConstVar;` | Constant | Grid    | Application  |

## Repository Structure

- `src/memory_access_time.cu`: The main CUDA program that measures and compares the access times for different memory types.
- `README.md`: Documentation file with instructions and explanations.

## Accessing the Repository

You can clone this repository to your local machine using the following command:

```bash
git clone https://github.com/sseyyeda/GPUsCode.git
cd GPUsCode/deviceMemoryEval/
```

## Compiling the Code

To compile the CUDA program, navigate to the directory where the code is located and use the following command:

```bash
cd src/
nvcc -o memory_access_time memory_access_time.cu
```

This will generate an executable named `memory_access_time`.

## Running the Program

After compiling, you can run the program with:

```bash
./memory_access_time
```

The output will display the kernel execution times for accessing each type of memory. You should observe that accessing shared memory is faster than global memory, and registers are the fastest of all.


