#include "../include/blur_cpu.h"

void blur_image_cpu(const cv::Mat& input, cv::Mat& output, int blur_size) {
    int kernel_size = 2 * blur_size + 1;
    cv::blur(input, output, cv::Size(kernel_size, kernel_size));
}

