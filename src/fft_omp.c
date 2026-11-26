/*
 * Version 2 — OpenMP, single machine.
 *
 * Each FFT stage contains n/2 independent butterflies (disjoint index pairs),
 * so we flatten them into one index space and split it across threads with
 * one `parallel for` per stage. The bit-reversal reorder stays serial — an
 * O(N) term that the report discusses under Amdahl's law.
 *
 * Usage: fft_omp <log2N> [input.bin] [output.bin]      (threads: OMP_NUM_THREADS)
 * Prints: RESULT,omp,<log2N>,1,<threads>,<fft_seconds>
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "fft_kernel.h"

static void fft_omp(cplx *x, size_t n) {
    bit_reverse(x, n);                                 /* serial O(N) part */
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        size_t half = len >> 1;
        size_t nbf = n >> 1;                           /* butterflies this stage */
        #pragma omp parallel for schedule(static)
        for (long long b = 0; b < (long long)nbf; b++) {
            size_t blk = ((size_t)b / half) * len;     /* which block            */
            size_t k   = (size_t)b % half;             /* butterfly within block */
            double wr = cos(ang * (double)k), wi = sin(ang * (double)k);
            cplx a = x[blk + k], c = x[blk + k + half];
            double tr = c.re * wr - c.im * wi;
            double ti = c.re * wi + c.im * wr;
            x[blk + k].re        = a.re + tr; x[blk + k].im        = a.im + ti;
            x[blk + k + half].re = a.re - tr; x[blk + k + half].im = a.im - ti;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <log2N> [input.bin] [output.bin]\n", argv[0]); return 1; }
    int p = atoi(argv[1]);
    size_t n = (size_t)1 << p;
    cplx *x = (cplx *)malloc(n * sizeof *x);
    if (!x) { fprintf(stderr, "out of memory for N=2^%d\n", p); return 1; }

    if (argc >= 3) {
        FILE *f = fopen(argv[2], "rb");
        if (!f || fread(x, sizeof *x, n, f) != n) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        fclose(f);
    } else {
        for (size_t i = 0; i < n; i++) {
            double t = (double)i / 16000.0;
            x[i].re = sin(2 * M_PI * 440 * t) + 0.5 * sin(2 * M_PI * 1000 * t);
            x[i].im = 0.0;
        }
    }

    int threads = omp_get_max_threads();
    double t0 = now_sec();
    fft_omp(x, n);
    double t1 = now_sec();
    printf("N=2^%d  OpenMP FFT (%d threads): %.4f s\n", p, threads, t1 - t0);
    printf("RESULT,omp,%d,1,%d,%.6f\n", p, threads, t1 - t0);

    if (argc >= 4) { FILE *f = fopen(argv[3], "wb"); fwrite(x, sizeof *x, n, f); fclose(f); }
    free(x);
    return 0;
}
