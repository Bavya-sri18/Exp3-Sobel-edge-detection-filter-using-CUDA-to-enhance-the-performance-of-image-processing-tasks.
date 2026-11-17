# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>AIM:</h3>
<h3>ENTER YOUR NAME: BAVYA SRI B</h3>
<h3>ENTER YOUR REGISTER NO: 212224230034</h3>
<h3>EX. NO<: 3/h3>
<h3>DATE: 17.11.2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height) {
    //Write your code here
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float Kx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    float Ky[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

    // only threads inside image will write results
    if ((x >= 3 / 2) && (x < (width - 3 / 2)) && (y >= 3 / 2) && (y < (height - 3 / 2))) {
        // Gradient in x-direction
        float Gx = 0;
        for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
            for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gx += fl * Kx[ky + 3 / 2][kx + 3 / 2];
            }
        }
        float Gx_abs = Gx < 0 ? -Gx : Gx;

        // Gradient in y-direction
        float Gy = 0;
        for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
            for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gy += fl * Ky[ky + 3 / 2][kx + 3 / 2];
            }
        }
        float Gy_abs = Gy < 0 ? -Gy : Gy;

        // Cap the value at 255 to prevent overflow
        float val = Gx_abs + Gy_abs;
        if (val > 255) val = 255;

        dstImage[(y * width + x)] = val;
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image as GRAYSCALE (Sobel requires 1 channel)
    Mat image = imread("/content/butterfly.jpg", IMREAD_GRAYSCALE);

    if (image.empty()) {
        printf("Error: Image not found. Check your path in the Colab folder.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));

    cudaEventRecord(start);

    // FIXED LINE: Added gridSize, blockSize, and corrected angle brackets
    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}

```

## OUTPUT:
SHOW YOUR OUTPUT HERE

## RESULT:
Thus the program has been executed by using CUDA to ________________.

Questions:

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:

Modified CUDA code with comments explaining your changes.
A report summarizing your findings, including graphs of execution times and a comparison of outputs.
Answers to the questions posed in the experiment.
Tools Required:

