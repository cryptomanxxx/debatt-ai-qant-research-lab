/* CL-54 single-threaded inference kernels (one sample at a time, float32).
 *
 * All kernels write the predicted class of each of n samples to out[].
 * Layouts: W1 is H x D row-major, W1T is D x H row-major (= W1 transposed),
 * W2 is C x H row-major, V is D x r row-major, VT is r x D row-major,
 * A is H x r row-major. scratch must hold at least 2*H + D floats, iscratch H + D ints.
 */
#include <string.h>

static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* Reorder idx[0..n) so that its first m entries index the m largest val[]. */
static void select_top(int *idx, int n, int m, const float *val) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float pivot = val[idx[(lo + hi) / 2]];
        int i = lo, j = hi;
        while (i <= j) {
            while (val[idx[i]] > pivot) i++;
            while (val[idx[j]] < pivot) j--;
            if (i <= j) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; i++; j--; }
        }
        if (m - 1 <= j) hi = j;
        else if (m - 1 >= i) lo = i;
        else break;
    }
}

static int argmax_logits(const float *b2, const float *W2, int C, int H,
                         const int *units, int nu, const float *h) {
    int best = 0;
    float bestv = 0.0f;
    for (int c = 0; c < C; c++) {
        float v = b2[c];
        for (int u = 0; u < nu; u++) v += W2[c * H + units[u]] * h[units[u]];
        if (c == 0 || v > bestv) { bestv = v; best = c; }
    }
    return best;
}

static int nonzeros(const float *x, int D, int *nz) {
    int n = 0;
    for (int i = 0; i < D; i++) if (x[i] != 0.0f) nz[n++] = i;
    return n;
}

void dense(const float *X, int n, int D, int H, int C, const float *W1, const float *b1,
           const float *W2, const float *b2, float *scratch, int *iscratch, int *out) {
    float *h = scratch;
    int *all = iscratch;
    for (int j = 0; j < H; j++) all[j] = j;
    for (int s = 0; s < n; s++) {
        const float *x = X + (long)s * D;
        for (int j = 0; j < H; j++) {
            float p = b1[j] + dot(W1 + (long)j * D, x, D);
            h[j] = p > 0.0f ? p : 0.0f;
        }
        out[s] = argmax_logits(b2, W2, C, H, all, H, h);
    }
}

void dense_sparse_input(const float *X, int n, int D, int H, int C, const float *W1T, const float *b1,
                        const float *W2, const float *b2, float *scratch, int *iscratch, int *out) {
    float *h = scratch;
    int *all = iscratch;
    int *nz = iscratch + H;
    for (int j = 0; j < H; j++) all[j] = j;
    for (int s = 0; s < n; s++) {
        const float *x = X + (long)s * D;
        int m = nonzeros(x, D, nz);
        memcpy(h, b1, sizeof(float) * H);
        for (int t = 0; t < m; t++) {
            const float xi = x[nz[t]];
            const float *col = W1T + (long)nz[t] * H;
            for (int j = 0; j < H; j++) h[j] += xi * col[j];
        }
        for (int j = 0; j < H; j++) if (h[j] < 0.0f) h[j] = 0.0f;
        out[s] = argmax_logits(b2, W2, C, H, all, H, h);
    }
}

/* kWTA with all H pre-activations computed; sparse_input selects the W1T path. */
void kwta_exact(const float *X, int n, int D, int H, int C, int k, int sparse_input,
                const float *W1, const float *W1T, const float *b1, const float *W2, const float *b2,
                float *scratch, int *iscratch, int *out) {
    float *pre = scratch, *h = scratch + H;
    int *idx = iscratch, *nz = iscratch + H;
    for (int s = 0; s < n; s++) {
        const float *x = X + (long)s * D;
        if (sparse_input) {
            int m = nonzeros(x, D, nz);
            memcpy(pre, b1, sizeof(float) * H);
            for (int t = 0; t < m; t++) {
                const float xi = x[nz[t]];
                const float *col = W1T + (long)nz[t] * H;
                for (int j = 0; j < H; j++) pre[j] += xi * col[j];
            }
        } else {
            for (int j = 0; j < H; j++) pre[j] = b1[j] + dot(W1 + (long)j * D, x, D);
        }
        for (int j = 0; j < H; j++) idx[j] = j;
        select_top(idx, H, k, pre);
        for (int u = 0; u < k; u++) h[idx[u]] = pre[idx[u]] > 0.0f ? pre[idx[u]] : 0.0f;
        out[s] = argmax_logits(b2, W2, C, H, idx, k, h);
    }
}

/* kWTA with low-rank screening: z = V^T x, score = b1 + A z, top-c candidates exact. */
void kwta_lowrank(const float *X, int n, int D, int H, int C, int k, int r, int c, int sparse_input,
                  const float *W1, const float *V, const float *VT, const float *A, const float *b1,
                  const float *W2, const float *b2, float *scratch, int *iscratch, int *out) {
    float *score = scratch, *h = scratch + H, *z = scratch + 2 * H;
    int *idx = iscratch, *nz = iscratch + H;
    for (int s = 0; s < n; s++) {
        const float *x = X + (long)s * D;
        int m = 0;
        if (sparse_input) {
            m = nonzeros(x, D, nz);
            for (int q = 0; q < r; q++) z[q] = 0.0f;
            for (int t = 0; t < m; t++) {
                const float xi = x[nz[t]];
                const float *row = V + (long)nz[t] * r;
                for (int q = 0; q < r; q++) z[q] += xi * row[q];
            }
        } else {
            for (int q = 0; q < r; q++) z[q] = dot(VT + (long)q * D, x, D);
        }
        for (int j = 0; j < H; j++) score[j] = b1[j] + dot(A + (long)j * r, z, r);
        for (int j = 0; j < H; j++) idx[j] = j;
        select_top(idx, H, c, score);
        /* exact pre-activations for the c candidates, stored in score[] */
        for (int u = 0; u < c; u++) {
            const float *w = W1 + (long)idx[u] * D;
            float p = b1[idx[u]];
            if (sparse_input) { for (int t = 0; t < m; t++) p += w[nz[t]] * x[nz[t]]; }
            else p += dot(w, x, D);
            score[idx[u]] = p;
        }
        select_top(idx, c, k, score);
        for (int u = 0; u < k; u++) h[idx[u]] = score[idx[u]] > 0.0f ? score[idx[u]] : 0.0f;
        out[s] = argmax_logits(b2, W2, C, H, idx, k, h);
    }
}
