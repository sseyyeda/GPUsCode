#include "grayscale.cuh"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

__global__ void grayscaleKernel(uchar3* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = d_input[idx];
        d_output[idx] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
    }
}

void convertToGrayscaleGPU(const Mat& input, Mat& output) {
    int width = input.cols;
    int height = input.rows;
    size_t numPixels = width * height;

    uchar3* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, numPixels * sizeof(uchar3));
    cudaMalloc(&d_output, numPixels * sizeof(unsigned char));

    cudaMemcpy(d_input, input.ptr<uchar3>(), numPixels * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaMemcpy(output.ptr<unsigned char>(), d_output, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

