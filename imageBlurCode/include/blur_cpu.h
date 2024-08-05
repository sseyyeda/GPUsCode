#ifndef BLUR_CPU_H
#define BLUR_CPU_H

#include <opencv2/opencv.hpp>

void blur_image_cpu(const cv::Mat& input, cv::Mat& output, int blur_size);

#endif // BLUR_CPU_H

