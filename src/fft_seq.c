/*
 * Version 1 — sequential baseline.
 *
 * Usage: fft_seq <log2N> [input.bin] [output.bin]
 *   input.bin / output.bin hold N complex doubles (re,im interleaved).
 *   Without an input file a synthetic 16 kHz two-tone signal is generated.
 *
 * Prints a machine-readable line the benchmark scripts collect:
 *   RESULT,seq,<log2N>,<procs>,<threads>,<fft_seconds>
 */
#include <stdio.h>
#include <stdlib.h>
#include "fft_kernel.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <log2N> [input.bin] [output.bin]\n", argv[0]); return 1; }
    int p = atoi(argv[1]);
    size_t n = (size_t)1 << p;
    cplx *x = (cplx *)malloc(n * sizeof *x);
    if (!x) { fprintf(stderr, "out of memory for N=2^%d\n", p); return 1; }

    if (argc >= 3) {
        FILE *f = fopen(argv[2], "rb");
        if (!f || fread(x, sizeof *x, n, f) != n) { fprintf(stderr, "cannot read %zu samples from %s\n", n, argv[2]); return 1; }
        fclose(f);
    } else {
        for (size_t i = 0; i < n; i++) {              /* 440 Hz + 1 kHz tones at 16 kHz */
            double t = (double)i / 16000.0;
            x[i].re = sin(2 * M_PI * 440 * t) + 0.5 * sin(2 * M_PI * 1000 * t);
            x[i].im = 0.0;
        }
    }

    double t0 = now_sec();
    fft_radix2(x, n);
    double t1 = now_sec();
    printf("N=2^%d (%zu points)  sequential FFT: %.4f s\n", p, n, t1 - t0);
    printf("RESULT,seq,%d,1,1,%.6f\n", p, t1 - t0);

    if (argc >= 4) { FILE *f = fopen(argv[3], "wb"); fwrite(x, sizeof *x, n, f); fclose(f); }
    free(x);
    return 0;
}
