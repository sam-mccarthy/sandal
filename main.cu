#include "mandelbrot.cuh"

int main() {
    Mandelbrot bratwurst(4096, 4096, 16384);
    bratwurst.RenderFrame(1.4, -1.768667862837488812627419470, 0.001645580546820209430325900);
    bratwurst.SaveFrame();
    return 0;
}