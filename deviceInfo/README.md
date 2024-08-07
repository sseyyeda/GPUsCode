# CUDA Device Information

This repository contains a simple CUDA program that retrieves and displays detailed information about the CUDA devices available on your system. The program uses CUDA APIs to gather information such as device name, total global memory, shared memory per block, warp size, and more.

## Requirements

- CUDA Toolkit installed
- NVIDIA GPU with CUDA support
- C++ compiler

## How to Compile and Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sseyyeda/GPUsCode.git
    cd GPUsCode/deviceInfo
    ```

2. **Compile the CUDA code:**
    ```bash
    nvcc -o device_info src/device_info.cu
    ```

3. **Run the executable:**
    ```bash
    ./device_info
    ```


### Explanation of CUDA APIs Used

The program uses several CUDA Runtime APIs to query and retrieve device information:

1. **cudaGetDeviceCount(int* count)**
    - This function returns in `count` the number of CUDA-capable devices present in the system.

2. **cudaGetDeviceProperties(cudaDeviceProp* prop, int device)**
    - This function returns in `prop` the properties of the CUDA device specified by `device`.

### Key Properties Retrieved

- **Device Name (`prop.name`):** The name of the CUDA device.
- **Total Global Memory (`prop.totalGlobalMem`):** The total amount of global memory available on the device.
- **Shared Memory Per Block (`prop.sharedMemPerBlock`):** The maximum amount of shared memory available to a thread block.
- **Registers Per Block (`prop.regsPerBlock`):** The maximum number of 32-bit registers available to a thread block.
- **Warp Size (`prop.warpSize`):** The warp size in threads.
- **Max Threads Per Block (`prop.maxThreadsPerBlock`):** The maximum number of threads per block.
- **Max Threads Dimension (`prop.maxThreadsDim`):** The maximum sizes of each dimension of a block.
- **Max Grid Size (`prop.maxGridSize`):** The maximum sizes of each dimension of a grid.
- **Clock Rate (`prop.clockRate`):** The clock frequency in kilohertz.
- **Total Constant Memory (`prop.totalConstMem`):** The total amount of constant memory available on the device.
- **Compute Capability (`prop.major` and `prop.minor`):** The major and minor revision numbers defining the compute capability of the device.
The amount of resources in each CUDA device SM is specified as part of the compute capability of the device. In general, the higher the compute capability level, the more resources are available in each SM. The compute capability of GPUs tends to increase from generation to generation. The Ampere A100 GPU has compute capability 8.0.
- **Multi-Processor Count (`prop.multiProcessorCount`):** The number of multiprocessors on the device.
- **Memory Clock Rate (`prop.memoryClockRate`):** The memory clock frequency in kilohertz.
- **Memory Bus Width (`prop.memoryBusWidth`):** The width of the memory bus in bits.
- **Peak Memory Bandwidth:** Calculated as `2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6` in gigabytes per second.


## Output

The program will display detailed information for each CUDA device available on your system. A sample output might look like:

```

Number of CUDA devices: 1
Device Number: 0
Device name: Quadro P620
Total global memory: 1976 MB
Shared memory per block: 48 KB
Registers per block: 65536
Warp size: 32
Max threads per block: 1024
Max threads dimension: [1024, 1024, 64]
Max grid size: [2147483647, 65535, 65535]
Clock rate: 1354 MHz
Total constant memory: 64 KB
Compute capability: 6.1
Multi-processor count: 4
Memory Clock Rate (KHz): 2505000
Memory Bus Width (bits): 128
Peak Memory Bandwidth (GB/s): 80.16
```
