#ifndef SANDAL_AINT_CUH
#define SANDAL_AINT_CUH

class AInt {
    __host__ __device__ void resize(uint64_t size);
public:
    uint64_t* bits;
    uint64_t size;
    bool negative;

    __host__ __device__ AInt();
    __host__ __device__ AInt(uint64_t num, uint64_t exponent);
    __host__ __device__ AInt(uint64_t num, AInt exponent);
    __host__ __device__ AInt(AInt const &victim);

    __host__ __device__ ~AInt();

    __host__ __device__ AInt operator+(AInt b);
    __host__ __device__ AInt operator+(uint64_t b);
    __host__ __device__ AInt operator+=(uint64_t b);
    __host__ __device__ AInt operator-(AInt b);
    __host__ __device__ AInt operator-(uint64_t b);
    __host__ __device__ AInt operator-=(uint64_t b);
    __host__ __device__ AInt operator*(AInt b);
};

#endif //SANDAL_AINT_CUH
