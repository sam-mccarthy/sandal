#ifndef PNGULUM_MANDELBROT_CUH
#define PNGULUM_MANDELBROT_CUH

#include <string>

class Mandelbrot {
public:
    Mandelbrot(int w, int h, int iter);
    ~Mandelbrot();
    void RenderSDL();
    void RenderFrame(double scale, double panX, double panY);
    void SaveFrame();
private:
    void CalculateReference(double x0, double y0, double scale);

    unsigned char* image;
    unsigned char* imageptr;

    int width;
    int height;
    int maxIter;

    float* reference;
    float* referenceptr;

    std::string filename;
};


#endif //PNGULUM_MANDELBROT_CUH
