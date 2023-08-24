#include "arbitrary.cuh"

__host__ __device__ Arbitrary::Arbitrary() {

}

__host__ __device__ Arbitrary::Arbitrary(Arbitrary const &victim){

}

__host__ __device__ Arbitrary::~Arbitrary() {

}

__host__ __device__ Arbitrary Arbitrary::operator+(Arbitrary b) {
    Arbitrary a(*this);
    uint32_t carry = 0;
    for(uint32_t dec = b.n_dec - 1;; dec--){
        if(dec > a.n_dec)
            ExpandDec(dec);
        uint64_t block_a = a.dec_blocks[dec];
        uint64_t block_b = b.dec_blocks[dec];
        uint64_t sum = block_a + block_b;

        if(carry > 0){
            sum += carry;
            carry = 0;
        }

        if(sum > UINT32_MAX){
            sum -= UINT32_MAX;
            carry = 1;
        }

        a.dec_blocks[dec] = sum;

        if(dec == 0)
            break;
    }

    for(uint32_t sig = 0; sig < b.n_sig; sig++){
        if(sig > a.n_dec)
            ExpandSig(sig);
        uint64_t block_a = a.sig_blocks[sig];
        uint64_t block_b = b.sig_blocks[sig];
        uint64_t sum = block_a + block_b;

        if(carry > 0){
            sum += carry;
            carry = 0;
        }

        if(sum > UINT32_MAX){
            sum -= UINT32_MAX;
            carry = 1;
        }

        b.sig_blocks[sig] = sum;
    }

    if(carry > 0){
        int limit = a.n_sig;
        ExpandSig();
        a.sig_blocks[limit] = carry;
    }

    return a;
}

__host__ __device__ Arbitrary Arbitrary::operator-(Arbitrary b) {
    Arbitrary a(*this);
    uint32_t borrow = 0;
    for(uint32_t dec = b.n_dec - 1; dec >= 0; dec--){
        if(dec > a.n_dec)
            ExpandDec(dec);
        uint64_t block_a = a.dec_blocks[dec];
        uint64_t block_b = b.dec_blocks[dec];
        uint64_t diff;
        if(block_a > block_b)
            diff = block_a - block_b;
        else {
            diff = UINT32_MAX - block_b;
            borrow = 1;
        }

        if(borrow > 0){
            diff -= borrow;
            borrow = 0;
        }

        a.dec_blocks[dec] = diff;
    }

    for(int sig = 0; sig < b.n_sig; sig++){
        if(sig > a.n_sig)
            ExpandDec(sig);
        uint64_t block_a = a.sig_blocks[sig];
        uint64_t block_b = b.sig_blocks[sig];
        uint64_t diff;
        if(block_a > block_b)
            diff = block_a - block_b;
        else {
            diff = UINT32_MAX - block_b;
            borrow = 1;
        }

        if(borrow > 0){
            diff -= borrow;
            borrow = 0;
        }

        a.sig_blocks[sig] = diff;
    }

    if(borrow > 0){
        int limit = a.n_sig;
        ExpandSig();
        a.sig_blocks[limit] = carry;
    }

    return a;
}

__host__ __device__ Arbitrary Arbitrary::operator*(Arbitrary b) {
    return Arbitrary();
}

__host__ __device__ Arbitrary Arbitrary::operator/(Arbitrary b) {
    return Arbitrary();
}

__host__ __device__ void Arbitrary::Print() {

}

__host__ __device__ void Arbitrary::ExpandDec() {

}

__host__ __device__ void Arbitrary::ExpandSig() {

}

__host__ __device__ void Arbitrary::ExpandDec(int goal) {

}

__host__ __device__ void Arbitrary::ExpandSig(int goal) {

}

__host__ __device__ AComplex AComplex::operator+(AComplex b) {
    return {x + b.x, y + b.y};
}

__host__ __device__ AComplex AComplex::operator-(AComplex b) {
    return {x - b.x, y - b.y};
}

__host__ __device__ AComplex AComplex::operator*(AComplex b) {
    //(ax + iay)(bx + iby)
    //axbx + axiby + iay bx - ayby
    return {x * b.x - y * b.y, x * b.y + y * b.x};
}

__host__ __device__ AComplex AComplex::operator/(AComplex b) {
    //(ax + iay) * (bx - iby)
    //axbx - axiby + iaybx + ayby

    //(bx + iby) * (bx - iby)
    //bxbx + byby
    Arbitrary denom = b.x * b.x + b.y * b.y;
    return {(x * b.x + y * b.y) / denom, (x * b.y + y * b.x) / denom};
}
