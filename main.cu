#include "mandelbrot.cuh"

int main(int argc, char* argv[]) {
    Mandelbrot bratwurst(1080, 1080, 16384);
    bratwurst.RenderSDL();
    return 0;
}