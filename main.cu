#include <iostream>
#include <string>
#include "fpng.cuh"

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

