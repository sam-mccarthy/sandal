#include "mandelbrot.cuh"

__global__ void CalculateFrame(int width, int height, unsigned char* image, int maxIter, float scale, float panX, float panY, float* reference){
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = py * width + px;

    float dx0 = (float)px / width * (3.47f / scale) - 2 - panX;
    float dy0 = (float)py / height * (2.24f / scale) - 1.12f - panY;

    float x = 0;
    float y = 0;
    float x2 = 0;
    float y2 = 0;

    int iter = 0;
    while(x2 + y2 <= 4 && iter < maxIter){
        float Xx = reference[iter * 2];
        float Xy = reference[iter * 2 + 1];

        float oldX = x;

        x = 2 * (Xx * x - Xy * y) + x2 - y2 + dx0;
        y = 2 * (Xx * y + Xy * oldX + y * oldX) + dy0;

        x2 = x * x;
        y2 = y * y;

        iter++;
    }

    float color = iter;
    if(iter < maxIter){
        color++;

        float log_zn = log(x2 + y2) / 2;
        color -= log(log_zn / log(2)) / log(2);
    }

    color = color / maxIter * 255;

    unsigned int imageIndex = index * 4;
    auto cbyte = (unsigned char)color;

    image[imageIndex + 0] = cbyte;
    image[imageIndex + 1] = cbyte;
    image[imageIndex + 2] = cbyte;
    image[imageIndex + 3] = 255;
}

Mandelbrot::Mandelbrot(int w, int h, int iter) {
    width = w;
    height = h;
    maxIter = iter;

    image = new unsigned char[w * h * 4];
    cudaMalloc((void **)&imageptr, w * h * 4);
    cudaMalloc((void **)&referenceptr, maxIter * 2);

    reference = new float[iter * 2];
}

void Mandelbrot::CalculateReference(double x0, double y0) {
    double x = 0;
    double y = 0;
    double x2 = 0;
    double y2 = 0;

    int iter = 0;
    while(x2 + y2 <= 4 && iter < maxIter)
    {
        reference[iter * 2] = (float)x;
        reference[iter * 2 + 1] = (float)y;

        y = (x + x) * y + y0;
        x = x2 - y2 + x0;

        x2 = x * x;
        y2 = y * y;

        iter++;
    }

    cudaMemcpy(referenceptr, reference, iter * 2, cudaMemcpyHostToDevice);
}

void Mandelbrot::RenderFrame(float scale, float panX, float panY){
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    CalculateReference(panX, panY);
    CalculateFrame<<<threadsPerBlock, numBlocks>>>(width, height, imageptr, maxIter, scale, panX, panY, referenceptr);
    cudaMemcpy(imageptr, image, width * height * 4, cudaMemcpyDeviceToHost);
}

Mandelbrot::~Mandelbrot(){
    cudaFree(imageptr);

    delete[] image;
    delete[] reference;
}