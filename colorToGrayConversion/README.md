# Color to Grayscale Conversion

This project demonstrates the conversion of a color image to a grayscale image using both CPU and GPU. The GPU implementation leverages CUDA to accelerate the process, while the CPU implementation uses standard C++ with OpenCV.

## Description

The code reads a color image, converts it to grayscale using both CPU and GPU, and then compares the runtime performance of the two methods.

## Prerequisites

- CMake
- CUDA Toolkit
- OpenCV (installed with development files)

### Installation on Linux

**Install CMake:**
```bash
sudo apt update
sudo apt install cmake
```

**Install CUDA Toolkit:**
Follow the instructions on the [CUDA Toolkit Download page](https://developer.nvidia.com/cuda-downloads) to download and install the appropriate version for your system.

**Install OpenCV:**
```bash
sudo apt update
sudo apt install libopencv-dev
```

## File Structure

- `CMakeLists.txt`: CMake configuration file.
- `main.cpp`: The main C++ file containing both the CPU and GPU implementations for the grayscale conversion.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sseyyeda/GPUsCode.git
   cd GPUsCode/colorToGrayConversion
   ```

2. **Build the project:**
   ```bash
   mkdir -p build
   cd build
   cmake -D CMAKE_PREFIX_PATH=/usr ..
   make
   ```

3. **Run the executable:**
   ```bash
   ./colorToGrayConversion <path_to_image>
   ```

   Replace `<path_to_image>` with the path to the image you want to convert.

## Example Output

```
Time taken by CPU: 45.1234 ms
Time taken by GPU: 12.5678 ms
```

The output shows the time taken for the grayscale conversion by both the CPU and the GPU.

## Run the Executable

You can run the executable by providing the path to an image file as an argument:

```bash
./colorToGrayConversion ../path/to/your/image.jpg
```

