# Image Blurring with CUDA and OpenCV

This project demonstrates image blurring using both CPU and GPU implementations. The CPU implementation utilizes OpenCV's built-in `cv::blur` function, while the GPU implementation leverages CUDA for parallel processing.

## Repository

Clone the repository using:

```bash
git clone https://github.com/sseyyeda/GPUsCode.git
cd GPUsCode/imageBlurCode
```

## Project Structure

- `include/`
  - `blur_cpu.h`: Header file for the CPU blurring function.
  - `blur_gpu.cuh`: Header file for the GPU blurring kernel and function declarations.
- `src/`
  - `blur_cpu.cpp`: Implementation of the CPU blurring function.
  - `blur_gpu.cu`: Implementation of the GPU blurring kernel and function.
  - `main.cpp`: Main application file that handles image loading, blurring, and saving results.
- `CMakeLists.txt`: CMake configuration file for building the project.

## Dependencies

- OpenCV
- CUDA Toolkit

## Building the Project

To build the project, follow these steps:

1. **Create a Build Directory:**

   ```bash
   mkdir build
   cd build
   ```

2. **Generate Build Files with CMake:**

   ```bash
   cmake ..
   ```

3. **Compile the Project:**

   ```bash
   make
   ```

## Usage

To run the application, use the following command:

```bash
./image_blurring <image_path>
```

Replace `<image_path>` with the path to the image you want to process.

### CPU and GPU Processing

- If you want to use the GPU version of the blurring, ensure that the `USE_GPU` flag is enabled in your CMake configuration.
- Running the command above will produce output for both CPU and GPU versions of the blurring process. The results will be saved in the `images` directory:

  - `images/output_image_cpu.png`: Result from the CPU-based blurring.
  - `images/output_image_gpu.png`: Result from the GPU-based blurring (if GPU support is enabled).

### Parameters

- **`blur_size`**: This parameter controls the size of the blurring kernel. It defines the radius of the blur effect. The blur kernel will be a square of size `(2 * blur_size + 1) x (2 * blur_size + 1)`.

  For example, setting `blur_size` to `15` results in a `31 x 31` kernel, producing a strong blur effect. To achieve a more pronounced blur, you can increase the `blur_size` value. However, larger kernel sizes will require more computation and may result in longer processing times.

  Adjusting the `blur_size`:

  - Small `blur_size` (e.g., 1-5): Light blur, subtle effect.
  - Medium `blur_size` (e.g., 10-20): Noticeable blur, good for general use.
  - Large `blur_size` (e.g., 30+): Strong blur, suitable for heavy blurring.

## Output

The processed images will be saved in the `images` directory:

- `images/output_image_cpu.png`: Result from the CPU-based blurring.
- `images/output_image_gpu.png`: Result from the GPU-based blurring (if GPU support is enabled).

## Troubleshooting

- Ensure that CUDA and OpenCV are properly installed and configured on your system.
- Check for CUDA errors in the console output if you encounter issues during GPU processing.


