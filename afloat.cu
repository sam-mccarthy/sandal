#include "afloat.cuh"

__host__ __device__ AFloat AFloat::operator+(AFloat b) {
    if(exponent <= b.exponent)
        return {mantissa + b.mantissa * AInt(10, b.exponent - exponent), exponent};
    else
        return {mantissa * AInt(10, b.exponent - exponent) + b.mantissa, b.exponent};
}

__host__ __device__ AFloat AFloat::operator-(AFloat b) {
    b.mantissa.negative = !b.mantissa.negative;
    return *this + b;
}

__host__ __device__ AFloat AFloat::operator-(uint64_t b) {
    AInt exp(exponent);
    exp.negative = !exp.negative;
    return {mantissa + AInt(b, exp), exponent};
}

__host__ __device__ AFloat AFloat::operator*(AFloat b) {
    return {mantissa * b.mantissa, exponent + b.exponent};
}

__host__ __device__ AFloat AFloat::operator/(AFloat b) {
    AFloat x(b);

    uint64_t expDec = x.mantissa.size + x.exponent;
    if(x < 1){
        x.exponent -= expDec;
    }

    for(int i = 0; i < 50; i++){
        x *= b * x - 2;
        x.mantissa.negative = !x.mantissa.negative;
    }

    return {mantissa / b.mantissa, exponent - b.exponent};
}

__host__ __device__ bool AFloat::operator<(uint64_t b){

}