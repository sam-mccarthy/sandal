//
// Created by stiit on 8/13/2023.
//

#ifndef PNGULUM_PENDULUM_CUH
#define PNGULUM_PENDULUM_CUH

class Pendulum {
public:
    Pendulum(int size, float lowerBound, float upperBound);
    void RenderVideo(int frames);
    void RenderSDL();
private:
    void RenderFrame();

    int size;

    float* theta1;
    float* theta2;

    float* velocity1;
    float* velocity2;

    float* theta1ptr;
    float* theta2ptr;

    float* velocity1ptr;
    float* velocity2ptr;
};
#endif //PNGULUM_PENDULUM_CUH
