#include <opencv2/opencv.hpp>
#include "blur_cpu.h"
#ifdef USE_GPU
#include "blur_gpu.cuh"
#endif
#include <iostream>
#include <chrono>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    cv::Mat output_image_cpu, output_image_gpu;
    int blur_size = 15; // Increase blur size for more noticeable effect

    auto start = std::chrono::high_resolution_clock::now();
    blur_image_cpu(input_image, output_image_cpu, blur_size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> cpu_duration = end - start;
    std::cout << "CPU blur duration: " << cpu_duration.count() << " seconds" << std::endl;

#ifdef USE_GPU
    start = std::chrono::high_resolution_clock::now();
    blur_image_gpu(input_image, output_image_gpu, blur_size);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> gpu_duration = end - start;
    std::cout << "GPU blur duration: " << gpu_duration.count() << " seconds" << std::endl;
#endif

    // Ensure the output directory exists
    std::filesystem::create_directories("images");

    // Save the output images
    cv::imwrite("images/output_image_cpu.png", output_image_cpu);
    std::cout << "CPU output image saved as images/output_image_cpu.png" << std::endl;

#ifdef USE_GPU
    cv::imwrite("images/output_image_gpu.png", output_image_gpu);
    std::cout << "GPU output image saved as images/output_image_gpu.png" << std::endl;
#endif

    return 0;
}

