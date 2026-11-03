/* CL-68: linear layer with bfloat16 products and a sequential bfloat16 running sum.
 * out[b,h] = sum_k bf16(x[b,k] * w[h,k]), rounding the running sum to bfloat16
 * after every addition (round to nearest, ties to even; same bit trick as cl58.bf16).
 * Compile without -ffast-math. */
#include <stdint.h>
#include <string.h>

static inline float bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u = (u + (((u >> 16) & 1u) + 0x7FFFu)) & 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

void seq_dot(const float *x, const float *w, float *out, long B, long H, long K) {
    for (long b = 0; b < B; b++) {
        const float *xb = x + b * K;
        for (long h = 0; h < H; h++) {
            const float *wh = w + h * K;
            float s = 0.0f;
            for (long k = 0; k < K; k++) s = bf16(s + bf16(xb[k] * wh[k]));
            out[b * H + h] = s;
        }
    }
}
