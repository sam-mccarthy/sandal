#include "aint.cuh"

__host__ __device__ AInt AInt::operator+(AInt b) {
    AInt a(*this);
    if(a.size < b.size + 1)
        resize(b.size + 1);

    for(int i = 0; i < b.size; i++){
        if (negative == b.negative && __builtin_add_overflow(a.bits[i], b.bits[i], a.bits + i)) {
            int k = i;
            while(a.bits[++k] == UINT32_MAX) a.bits[k] = 0;
            a.bits[k]++;
        } else if (__builtin_sub_overflow(a.bits[i], b.bits[i], a.bits + i)) {
            int k = i;
            while (a.bits[++k] == 0 || k == a.size - 1);

            if(a.bits[k] == 0) {
                negative = !negative;
            } else {
                k = i;
                while (a.bits[++k] == 0) a.bits[k] = UINT64_MAX;
                a.bits[k]--;
            }
        }
    }

    return a;
}

__host__ __device__ AInt AInt::operator+(uint64_t b) {
    AInt a(*this);
    if(a.bits[a.size - 1] == UINT64_MAX)
        resize(a.size * 1.4);

    if (__builtin_add_overflow(a.bits[0], b, a.bits)) {
        int k = 0;
        while(a.bits[++k] == UINT32_MAX) a.bits[k] = 0;
        a.bits[k]++;
    }
}

__host__ __device__ AInt AInt::operator+=(uint64_t b) {
    if(bits[size - 1] == UINT64_MAX)
        resize(size * 1.4);

    if (__builtin_add_overflow(bits[0], b, bits)) {
        int k = 0;
        while(bits[++k] == UINT32_MAX) bits[k] = 0;
        bits[k]++;
    }

    return *this;
}

__host__ __device__ AInt AInt::operator-(AInt b) {
    AInt a(*this);
    b.negative = !b.negative;
    return a + b;
}

__host__ __device__ AInt AInt::operator-(uint64_t b) {
    AInt a(*this);
    if (__builtin_sub_overflow(a.bits[0], b, a.bits)) {
        int k = 0;
        while (a.bits[++k] == 0 || k == a.size - 1);

        if(a.bits[k] == 0) {
            negative = !negative;;
        } else {
            k = 0;
            while (a.bits[++k] == 0) a.bits[k] = UINT64_MAX;

            a.bits[k]--;
        }
    }

    return a;
}

__host__ __device__ AInt AInt::operator-=(uint64_t b) {
    if (__builtin_sub_overflow(bits[0], b, bits)) {
        int k = 0;
        while (bits[++k] == 0 || k == size - 1);

        if(bits[k] == 0) {
            negative = !negative;;
        } else {
            k = 0;
            while (bits[++k] == 0) bits[k] = UINT64_MAX;

            bits[k]--;
        }
    }

    return *this;
}

__host__ __device__ AInt AInt::operator*(AInt b){
    AInt a(*this);
    if(a.size < a.size + b.size + 1)
        resize(a.size + b.size + 1);

    a.negative = negative != b.negative;

    for(int i = 0; i < a.size; i++){
        for(int j = 0; j < b.size; j++) {
            unsigned __int128 additive = a.bits[i] * b.bits[i];
            if (__builtin_add_overflow(a.bits[i], additive & UINT64_MAX, a.bits + i)) {
                int k = i;
                while(a.bits[++k] == UINT32_MAX) a.bits[k] = 0;
                a.bits[k]++;
            }

            if (__builtin_add_overflow(a.bits[i + 1], (additive >> 64) & UINT64_MAX, a.bits + i + 1)) {
                int k = i + 1;
                while(a.bits[++k] == UINT32_MAX) a.bits[k] = 0;
                a.bits[k]++;
            }
        }
    }

    return a;
}