#ifndef BLUR_GPU_CUH
#define BLUR_GPU_CUH

#include <opencv2/opencv.hpp>

void blur_image_gpu(const cv::Mat& input, cv::Mat& output, int blur_size);

#endif // BLUR_GPU_CUH

