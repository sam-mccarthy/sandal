#include "mandelbrot.cuh"

__global void CalculateFrame(int width, int height, int* iterations, double refX, double refY, double* reference){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = y * width + x;

    double x0 = ((double)x / width) * 3.47 - 2;
    double y0 = ((double)y / height) * 2.24 - 1.12;

    double x = 0;
    double y = 0;
    double x2 = 0;
    double y2 = 0;
    double w = 0;

    int iter = 0;
    while(x2 + y2 <= 4 && iter < 100){
        x = x2 - y2 + x0;
        y = w - x2 - y2 + y0;

        x2 = x * x;
        y2 = y * y;
        w = (x + y) * (x + y);
        iter++;
    }

}
