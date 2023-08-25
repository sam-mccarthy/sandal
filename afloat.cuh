#ifndef SANDAL_AFLOAT_CUH
#define SANDAL_AFLOAT_CUH

#include "aint.cuh"

class AFloat {
    AInt mantissa;
    AInt exponent;
public:
    __host__ __device__ AFloat();
    __host__ __device__ AFloat(AInt mant, AInt exp);
    __host__ __device__ ~AFloat();

    __host__ __device__ AFloat operator+(AFloat b);
    __host__ __device__ AFloat operator-(AFloat b);
    __host__ __device__ AFloat operator-(uint64_t b);
    __host__ __device__ AFloat operator*(AFloat b);
    __host__ __device__ AFloat operator/(AFloat b);
    __host__ __device__ bool operator<(uint64_t b);
};

#endif //SANDAL_AFLOAT_CUH
