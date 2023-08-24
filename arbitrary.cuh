#ifndef SANDAL_ARBITRARY_CUH
#define SANDAL_ARBITRARY_CUH

class Arbitrary {
    uint32_t* sig_blocks;
    uint32_t* dec_blocks;
    uint32_t n_sig;
    uint32_t n_dec;
    bool negative;
public:
    __host__ __device__ Arbitrary();
    __host__ __device__ Arbitrary(Arbitrary const &victim);
    __host__ __device__ ~Arbitrary();
    __host__ __device__ void ExpandDec();
    __host__ __device__ void ExpandSig();
    __host__ __device__ void ExpandDec(uint32_t goal);
    __host__ __device__ void ExpandSig(uint32_t goal);
    __host__ __device__ Arbitrary operator+(Arbitrary b);
    __host__ __device__ Arbitrary operator-(Arbitrary b);
    __host__ __device__ Arbitrary operator*(Arbitrary b);
    __host__ __device__ Arbitrary operator/(Arbitrary b);
    __host__ __device__ void Print();
};

class AComplex {
    Arbitrary x;
    Arbitrary y;
public:
    __host__ __device__ AComplex();
    __host__ __device__ AComplex(Arbitrary x, Arbitrary y);
    __host__ __device__ AComplex operator+(AComplex b);
    __host__ __device__ AComplex operator-(AComplex b);
    __host__ __device__ AComplex operator*(AComplex b);
    __host__ __device__ AComplex operator/(AComplex b);
};

#endif //SANDAL_ARBITRARY_CUH
