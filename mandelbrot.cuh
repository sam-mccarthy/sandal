#ifndef PNGULUM_MANDELBROT_CUH
#define PNGULUM_MANDELBROT_CUH

#include <string>
#include <SDL.h>

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

    dim3 blockSize;
    dim3 gridSize;

    int width;
    int height;
    int maxIter;

    float* reference;
    float* referenceptr;

    std::string filename;

    void UpdateSDL(SDL_Texture *texture, SDL_Renderer *renderer, double scale, double panX, double panY);
    void SDLEventLoop(SDL_Texture *texture, SDL_Renderer *renderer);
};


#endif //PNGULUM_MANDELBROT_CUH
