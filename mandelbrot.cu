#include <format>
#include <iostream>

#include <SDL.h>

#include "mandelbrot.cuh"
#include "fpng.cuh"

__global__ void CalculateFrame(int width, int height, unsigned char* image, uint32_t maxIter, double scale, double panX, double panY){
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = py * width + px;

    if(index > width * height) return;

    float dx0 = ((double)px / width * 4 - 2) * scale + panX;
    float dy0 = ((double)py / height * 4 - 2) * scale + panY;

    float x = 0;
    float y = 0;
    float x2 = 0;
    float y2 = 0;

    int iter = 0;
    while(x2 + y2 <= 4 && iter < maxIter){
        y = (x + x) * y + dy0;
        x = x2 - y2 + dx0;

        x2 = x * x;
        y2 = y * y;

        iter++;
    }

    unsigned int imageIndex = index * 4;

    image[imageIndex + 0] = (unsigned char)(sin((double)iter) * 127 + 128);
    image[imageIndex + 1] = (unsigned char)(cos((double)iter) * 127 + 128);
    image[imageIndex + 2] = (unsigned char)(tan((double)iter) * 127 + 128);
    image[imageIndex + 3] = 255;
}

Mandelbrot::Mandelbrot(int w, int h, int iter) {
    width = w;
    height = h;
    maxIter = iter;

    blockSize = dim3(32, 32);
    gridSize = dim3((int)ceil((float)width / blockSize.x), (int)ceil((float)height / blockSize.y));

    image = new unsigned char[w * h * 4];
    cudaMalloc((void **)&imageptr, w * h * 4);
}

void Mandelbrot::RenderFrame(double scale, double panX, double panY){
    CalculateFrame<<<gridSize, blockSize>>>(width, height, imageptr, maxIter, scale, panX, panY);
    cudaMemcpy(image, imageptr, width * height * 4, cudaMemcpyDeviceToHost);

    filename = std::format("{}{}{}{}{}.png", width, height, scale, panX, panY);
}

void Mandelbrot::RenderSDL(){
    if(SDL_Init(SDL_INIT_VIDEO) < 0){
        std::cout << "SDL Init Failed.";
        return;
    }

    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_CreateWindowAndRenderer(width, height, 0, &window, &renderer);

    if(window == nullptr || renderer == nullptr){
        std::cout << "Window / renderer creation error.";
        return;
    }

    SDL_Surface* surface = SDL_GetWindowSurface(window);
    SDL_Texture* mandelTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);

    SDLEventLoop(mandelTexture, renderer);

    SDL_FreeSurface(surface);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Mandelbrot::SDLEventLoop(SDL_Texture* texture, SDL_Renderer* renderer){
    bool quit = false;

    double scale = 1;

    double panX = 0;
    double panY = 0;

    bool change = true;
    bool mousePressed = false;
    SDL_Event event;
    while(!quit){
        if(SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_q) {
                        scale *= 0.9;
                        change = true;
                    } else if (event.key.keysym.sym == SDLK_e) {
                        scale *= 1.1;
                        change = true;
                    }
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    mousePressed = true;
                    break;
                case SDL_MOUSEBUTTONUP:
                    mousePressed = false;
                    break;
                case SDL_MOUSEMOTION:
                    if (mousePressed) {
                        panX -= (float)event.motion.xrel / width / (scale / 4.0);
                        panY -= (float)event.motion.yrel / height / (scale / 4.0);
                        std::cout << panX << std::endl;
                        std::cout << panY << std::endl;
                        change = true;
                    }
                    break;
                case SDL_QUIT:
                    quit = true;
                    break;
                default:
                    break;
            }
        }

        if(change)
            UpdateSDL(texture, renderer, scale, panX, panY);

        change = false;
    }
}

void Mandelbrot::UpdateSDL(SDL_Texture* texture, SDL_Renderer* renderer, double scale, double panX, double panY){
    int pitch;
    void* pixels;

    RenderFrame(scale, panX, panY);

    SDL_LockTexture(texture, nullptr, &pixels, &pitch);
    memcpy(pixels, image, width * height * 4);
    SDL_UnlockTexture(texture);

    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
}

void Mandelbrot::SaveFrame(){
    fpng::fpng_encode_image_to_file(filename.c_str(), image, width, height, 4);
}

Mandelbrot::~Mandelbrot(){
    cudaFree(imageptr);

    delete[] image;
}