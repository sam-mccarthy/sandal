#include "pendulum.cuh"

const float PI = 3.14159265358979323;

__global__ void CalculateFrame(float* theta1, float* theta2, float* velocity1, float* velocity2){

}

__global__ void DrawFrame(float* theta1, float* theta2, unsigned char* image){

}

Pendulum::Pendulum(int sz, float lowerBound, float upperBound){
    size = sz;
    int size2 = size * size;
    theta1 = new float[size2];
    theta2 = new float[size2];

    velocity1 = new float[size2];
    velocity2 = new float[size2];

    int x = 0;
    int y = 0;
    for(int i = 0; i < size * size; i++){
        theta1[i] = ((float)x / size) * (upperBound - lowerBound) + lowerBound;
        theta2[i] = ((float)y / size) * (upperBound - lowerBound) + lowerBound;

        velocity1[i] = 0;
        velocity2[i] = 0;

        x++;
        if(x == size){
            x = 0;
            y++;
        }
    }
}

void Pendulum::RenderFrame(){

    auto byteSize = sizeof(float) * size2;
    float* theta1ptr;
    float* theta2ptr;

    float* velocity1ptr;
    float* velocity2ptr;

    cudaMalloc((void **)&theta1ptr, byteSize);
    cudaMalloc((void **)&theta2ptr, byteSize);

    cudaMalloc((void **)&velocity1ptr, byteSize);
    cudaMalloc((void **)&velocity2ptr, byteSize);

    cudaMemcpy(theta1ptr, theta1, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(theta2ptr, theta2, byteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(theta1ptr, theta1, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(theta2ptr, theta2, byteSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);


}

