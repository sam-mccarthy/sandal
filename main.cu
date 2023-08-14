#include <iostream>
#include <string>
#include "fpng.h"

const int SIZE = 4096;
const float PI = 3.14159265;

__global__ void CalculateFrame(float *theta1, float* theta2, float* velocity1, float* velocity2){
    const float m1 = 1;
    const float m2 = 1;
    const float M = m1 + m2;

    const float L1 = 1;
    const float L2 = 1;

    const float g = 9.81;
    const float timestep = 200;

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * SIZE + x;

    float t1 = theta1[index];
    float t2 = theta2[index];
    float v1 = velocity1[index];
    float v2 = velocity2[index];

    float delta = t1 - t2;

    float sin2 = sin(t2);
    float sin1 = sin(t1);
    float cosDelta = cos(delta);

    float dt1 = -sin(delta * (m2 * L1 * v1 * v1 * cosDelta + m2 * L2 * v2 * v2)) - g * (M * sin1 - m2 * sin2 * cosDelta);
    float dt2 = sin(delta * (M * L1 * v1 * v1 + m2 * L2 * v2 * v2 * cosDelta)) + g * (M * sin1 * cosDelta - M * sin2);

    float alpha = m1 + m2 * sin(delta) * sin(delta);
    dt1 /= L1 * alpha;
    dt2 /= L2 * alpha;

    velocity1[index] += dt1 / timestep;
    velocity2[index] += dt2 / timestep;

    theta1[index] += velocity1[index];
    theta2[index] += velocity2[index];
}

__global__ void DrawFrame(float *theta1, float *theta2, unsigned char* image){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * SIZE + x;

    unsigned int imageIndex = index * 4;

    auto red = (unsigned char)((sin(theta1[index]) + 1) * 127);
    auto green = (unsigned char)((sin(theta2[index]) + 1) * 127);

    image[imageIndex + 0] = red;
    image[imageIndex + 1] = green;
    image[imageIndex + 2] = 255;
    image[imageIndex + 3] = 255;
}

int main() {
    auto byteSize = sizeof(float) * size2;
    float* theta1Ptr;
    float* theta2Ptr;
    float* velocity1Ptr;
    float* velocity2Ptr;

    cudaMalloc((void **)&theta1Ptr, byteSize);
    cudaMalloc((void **)&theta2Ptr, byteSize);
    cudaMalloc((void **)&velocity1Ptr, byteSize);
    cudaMalloc((void **)&velocity2Ptr, byteSize);
    cudaMemcpy(theta1Ptr, theta1, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(theta2Ptr, theta2, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(theta1Ptr, theta1, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(theta2Ptr, theta2, byteSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(SIZE / threadsPerBlock.x, SIZE / threadsPerBlock.y);

    auto imgSize = sizeof(unsigned char) * size2 * 4;
    auto* image = new unsigned char[imgSize];

    unsigned char* imagePtr;
    cudaMalloc((void **)&imagePtr, imgSize);

    int i = 0;
    while(++i) {
        std::cout << "Calculating frame " << i;
        CalculateFrame<<<numBlocks, threadsPerBlock>>>(theta1Ptr, theta2Ptr, velocity1Ptr, velocity2Ptr);

        std::cout << "\nSaving frame\n";
        DrawFrame<<<numBlocks, threadsPerBlock>>>(theta1Ptr, theta2Ptr, imagePtr);
        cudaMemcpy(image, imagePtr, imgSize, cudaMemcpyDeviceToHost);

        fpng::fpng_encode_image_to_file(("frames/frame-" + std::to_string(i) + ".png").c_str(), image, SIZE, SIZE, 4);
    }

    cudaFree(theta1Ptr);
    cudaFree(theta2Ptr);
    cudaFree(velocity1Ptr);
    cudaFree(velocity2Ptr);
    cudaFree(imagePtr);

    return 0;
}
