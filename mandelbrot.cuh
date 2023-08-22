#ifndef PNGULUM_MANDELBROT_CUH
#define PNGULUM_MANDELBROT_CUH

class Mandelbrot {
    Mandelbrot(int w, int h, int iter);
    ~Mandelbrot();
    void RenderSDL();
    void RenderFrame(float scale, float panX, float panY);
    void SaveFrame();
private:
    void CalculateReference(double x0, double y0);

    unsigned char* image;
    unsigned char* imageptr;

    int width;
    int height;
    int maxIter;

    float* reference;
    float* referenceptr;
};


#endif //PNGULUM_MANDELBROT_CUH
