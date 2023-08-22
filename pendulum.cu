#include <string>
#include "pendulum.cuh"
#include "fpng.cuh"

const float PI = 3.14159265358979323;

__global__ void CalculateFrame(float* theta1, float* theta2, float* velocity1, float* velocity2, unsigned char* image, int size, float m1, float m2, float L1, float L2, float g, float accuracy){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * size + x;

    float t1 = theta1[index];
    float t2 = theta2[index];

    float v1 = velocity1[index];
    float v2 = velocity2[index];

    float delta = t1 - t2;

    float sin1 = sin(t1);
    float sin2 = sin(t2);
    float cosDelta = cos(delta);

    float M = m1 + m2;
    float dt1 = -sin(delta * (m2 * L1 * v1 * v1 * cosDelta + m2 * L2 * v2 * v2)) - g * (M * sin1 - m2 * sin2 * cosDelta);
    float dt2 = sin(delta * (M * L1 * v1 * v1 + m2 * L2 * v2 * v2 * cosDelta)) + g * (M * sin1 * cosDelta - M * sin2);

    float alpha = m1 + m2 * sin(delta) * sin(delta);
    dt1 /= L1 * alpha;
    dt2 /= L2 * alpha;

    velocity1[index] += dt1 / accuracy;
    velocity2[index] += dt2 / accuracy;

    theta1[index] += velocity1[index];
    theta2[index] += velocity2[index];

    unsigned int imageIndex = index * 4;

    auto red = (unsigned char)((sin(theta1[index]) + 1) * 127);
    auto green = (unsigned char)((sin(theta2[index]) + 1) * 127);

    image[imageIndex + 0] = red;
    image[imageIndex + 1] = green;
    image[imageIndex + 2] = 255;
    image[imageIndex + 3] = 255;
}

Pendulum::Pendulum(int sz, float lowerBound, float upperBound){
    size = sz;
    int size2 = size * size;
    theta1 = new float[size2];
    theta2 = new float[size2];

    velocity1 = new float[size2];
    velocity2 = new float[size2];

    image = new unsigned char[size2 * 4];

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

    auto byteSize = sizeof(float) * size2;

    cudaMalloc((void **)&theta1ptr, byteSize);
    cudaMalloc((void **)&theta2ptr, byteSize);

    cudaMalloc((void **)&velocity1ptr, byteSize);
    cudaMalloc((void **)&velocity2ptr, byteSize);

    cudaMalloc((void **)&imageptr, size2 * 4);

    cudaMemcpy(theta1ptr, theta1, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(theta2ptr, theta2, byteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(theta1ptr, theta1, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(theta2ptr, theta2, byteSize, cudaMemcpyHostToDevice);
}

Pendulum::~Pendulum() {
    cudaFree(theta1ptr);
    cudaFree(theta2ptr);

    cudaFree(velocity1ptr);
    cudaFree(velocity2ptr);

    cudaFree(imageptr);

    delete[] theta1;
    delete[] theta2;

    delete[] velocity1;
    delete[] velocity2;

    delete[] image;
}

void Pendulum::RenderFrame(){
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);

    CalculateFrame<<<threadsPerBlock, numBlocks>>>(theta1ptr, theta2ptr, velocity1ptr, velocity2ptr, imageptr, size, m1, m2, L1, L2, g, accuracy);
    cudaMemcpy(imageptr, image, size * size * 4, cudaMemcpyDeviceToHost);
}

void Pendulum::SaveFrame(int i){
    auto name = "frames/frame-" + std::to_string(i) + ".png";
    fpng::fpng_encode_image_to_file(name.c_str(), image, size, size, 4);
}
