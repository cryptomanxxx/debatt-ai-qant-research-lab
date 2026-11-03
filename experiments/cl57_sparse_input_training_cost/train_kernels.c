/* CL-57 single-threaded per-sample SGD for a 784 -> H -> 10 ReLU MLP.
 *
 * W1T is D x H row-major (input-major), W2 is C x H row-major.
 * Both variants visit inputs in the same ascending order; sparse_input
 * skips inputs equal to zero, whose contributions and gradients are zero.
 * scratch: at least 2*H + C floats; iscratch: at least D ints.
 */
#include <math.h>
#include <string.h>

static int nonzeros(const float *x, int D, int *nz) {
    int n = 0;
    for (int i = 0; i < D; i++) if (x[i] != 0.0f) nz[n++] = i;
    return n;
}

static void hidden(const float *x, int D, int H, const float *W1T, const float *b1,
                   int sparse_input, const int *nz, int m, float *pre) {
    memcpy(pre, b1, sizeof(float) * H);
    int count = sparse_input ? m : D;
    for (int t = 0; t < count; t++) {
        int i = sparse_input ? nz[t] : t;
        const float xi = x[i];
        const float *col = W1T + (long)i * H;
        for (int j = 0; j < H; j++) pre[j] += xi * col[j];
    }
}

void train_epoch(const float *X, const long *y, const long *order, int n, int D, int H, int C,
                 float *W1T, float *b1, float *W2, float *b2, float lr, int sparse_input,
                 float *scratch, int *iscratch) {
    float *pre = scratch, *dh = scratch + H, *g = scratch + 2 * H;
    int *nz = iscratch;
    for (int s = 0; s < n; s++) {
        const float *x = X + order[s] * (long)D;
        const int label = (int)y[order[s]];
        int m = sparse_input ? nonzeros(x, D, nz) : D;
        hidden(x, D, H, W1T, b1, sparse_input, nz, m, pre);
        /* softmax over logits */
        float mx = -INFINITY;
        for (int c = 0; c < C; c++) {
            float v = b2[c];
            for (int j = 0; j < H; j++) v += W2[c * H + j] * (pre[j] > 0.0f ? pre[j] : 0.0f);
            g[c] = v;
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int c = 0; c < C; c++) { g[c] = expf(g[c] - mx); sum += g[c]; }
        for (int c = 0; c < C; c++) g[c] = g[c] / sum - (c == label ? 1.0f : 0.0f);
        /* backward into the hidden layer, using W2 before its update */
        for (int j = 0; j < H; j++) {
            float d = 0.0f;
            if (pre[j] > 0.0f) for (int c = 0; c < C; c++) d += W2[c * H + j] * g[c];
            dh[j] = d;
        }
        /* output layer update */
        for (int c = 0; c < C; c++) {
            const float gc = lr * g[c];
            float *row = W2 + c * H;
            for (int j = 0; j < H; j++) row[j] -= gc * (pre[j] > 0.0f ? pre[j] : 0.0f);
            b2[c] -= gc;
        }
        /* hidden layer update: only inputs visited by the forward pass */
        int count = sparse_input ? m : D;
        for (int t = 0; t < count; t++) {
            int i = sparse_input ? nz[t] : t;
            const float a = lr * x[i];
            float *col = W1T + (long)i * H;
            for (int j = 0; j < H; j++) col[j] -= a * dh[j];
        }
        for (int j = 0; j < H; j++) b1[j] -= lr * dh[j];
    }
}

void predict(const float *X, int n, int D, int H, int C, const float *W1T, const float *b1,
             const float *W2, const float *b2, float *scratch, int *iscratch, long *out) {
    float *pre = scratch;
    int *nz = iscratch;
    for (int s = 0; s < n; s++) {
        const float *x = X + (long)s * D;
        int m = nonzeros(x, D, nz);
        hidden(x, D, H, W1T, b1, 1, nz, m, pre);
        int best = 0;
        float bestv = 0.0f;
        for (int c = 0; c < C; c++) {
            float v = b2[c];
            for (int j = 0; j < H; j++) v += W2[c * H + j] * (pre[j] > 0.0f ? pre[j] : 0.0f);
            if (c == 0 || v > bestv) { bestv = v; best = c; }
        }
        out[s] = best;
    }
}
