/* CL-67: sequential bfloat16 accumulation, bit-identical to cl62.accumulate(t, "seq").
 * Each row of t (rows x n, float32) is summed term by term; after every float32
 * addition the running sum is rounded to bfloat16 (round to nearest, ties to even),
 * using the same bit trick as cl58.bf16. Compile without -ffast-math. */
#include <stdint.h>
#include <string.h>

static inline float bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u = (u + (((u >> 16) & 1u) + 0x7FFFu)) & 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}

void seq_bf16(const float *t, float *out, long rows, long n) {
    for (long r = 0; r < rows; r++) {
        const float *row = t + r * n;
        float s = 0.0f;
        for (long j = 0; j < n; j++) s = bf16(s + row[j]);
        out[r] = s;
    }
}
