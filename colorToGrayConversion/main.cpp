#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "grayscale.cuh"

using namespace cv;
using namespace std;

void convertToGrayscaleCPU(const Mat& input, Mat& output) {
    cvtColor(input, output, COLOR_BGR2GRAY);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./GrayscaleConversion <image_path>" << endl;
        return -1;
    }

    Mat colorImage = imread(argv[1], IMREAD_COLOR);
    if (colorImage.empty()) {
        cout << "Could not open or find the image." << endl;
        return -1;
    }

    Mat grayImageCPU, grayImageGPU;

    // CPU Grayscale Conversion
    auto start_cpu = chrono::high_resolution_clock::now();
    convertToGrayscaleCPU(colorImage, grayImageCPU);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_duration = end_cpu - start_cpu;

    // GPU Grayscale Conversion
    grayImageGPU.create(colorImage.rows, colorImage.cols, CV_8UC1);
    auto start_gpu = chrono::high_resolution_clock::now();
    convertToGrayscaleGPU(colorImage, grayImageGPU);
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> gpu_duration = end_gpu - start_gpu;

    // Displaying the time taken by CPU and GPU
    cout << "Time taken by CPU: " << cpu_duration.count() << " ms" << endl;
    cout << "Time taken by GPU: " << gpu_duration.count() << " ms" << endl;

    imwrite("gray_cpu.jpg", grayImageCPU);
    imwrite("gray_gpu.jpg", grayImageGPU);

    return 0;
}

