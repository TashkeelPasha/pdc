/*
 * Shared radix-2 FFT kernel used by all three versions (seq / OpenMP / hybrid MPI).
 * PDC semester project - Hybrid MPI + OpenMP Parallel FFT for Large Speech Signals.
 * Group: M Tashkeel Pasha (23k-2014), M Kashan (23k-0043), Hamadullah (23k-0723).
 */
#ifndef FFT_KERNEL_H
#define FFT_KERNEL_H

#include <math.h>
#include <stddef.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { double re, im; } cplx;

static inline double now_sec(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Reorder x so element i moves to the bit-reversed index of i (in place). */
static inline void bit_reverse(cplx *x, size_t n) {
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { cplx t = x[i]; x[i] = x[j]; x[j] = t; }
    }
}

/* Serial iterative forward FFT. n must be a power of two.
 * Also used per-row by the parallel versions (each row is independent). */
static inline void fft_radix2(cplx *x, size_t n) {
    bit_reverse(x, n);
    for (size_t len = 2; len <= n; len <<= 1) {       /* one stage per doubling */
        double ang = -2.0 * M_PI / (double)len;
        size_t half = len >> 1;
        for (size_t start = 0; start < n; start += len) {
            for (size_t k = 0; k < half; k++) {       /* butterflies */
                double wr = cos(ang * (double)k), wi = sin(ang * (double)k);
                cplx a = x[start + k], b = x[start + k + half];
                double tr = b.re * wr - b.im * wi;
                double ti = b.re * wi + b.im * wr;
                x[start + k].re        = a.re + tr; x[start + k].im        = a.im + ti;
                x[start + k + half].re = a.re - tr; x[start + k + half].im = a.im - ti;
            }
        }
    }
}

#endif /* FFT_KERNEL_H */

