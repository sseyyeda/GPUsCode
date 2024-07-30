#ifndef GRAYSCALE_CUH
#define GRAYSCALE_CUH

#include <opencv2/opencv.hpp>

void convertToGrayscaleGPU(const cv::Mat& input, cv::Mat& output);

#endif

