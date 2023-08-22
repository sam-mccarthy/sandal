#ifndef PNGULUM_PENDULUM_CUH
#define PNGULUM_PENDULUM_CUH

class Pendulum {
public:
    Pendulum(int size, float lowerBound, float upperBound);
    ~Pendulum();
    void RenderVideo(int frames);
    void RenderSDL();
    void RenderFrame();
    void SaveFrame();
private:
    int size;

    float* theta1;
    float* theta2;

    float* velocity1;
    float* velocity2;

    float* theta1ptr;
    float* theta2ptr;

    float* velocity1ptr;
    float* velocity2ptr;

    unsigned char* image;
    unsigned char* imageptr;

    float m1;
    float m2;
    float L1;
    float L2;

    float g;
    float accuracy;
};
#endif //PNGULUM_PENDULUM_CUH

