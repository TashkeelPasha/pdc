/*
 * Version 3 — hybrid MPI + OpenMP distributed FFT (Bailey's four-step algorithm).
 *
 * A 1-D FFT of N = N1*N2 points is computed as a 2-D decomposition:
 *   view input x as an N1 x N2 row-major matrix A (A[n1][n2] = x[n1*N2 + n2]), then
 *     1. global TRANSPOSE                     (MPI_Alltoall  — the communication step)
 *     2. N2 independent FFTs of length N1     (local, OpenMP across rows)
 *     3. twiddle multiply by W_N^(n2*k1)      (local, OpenMP)
 *     4. global TRANSPOSE back                (MPI_Alltoall)
 *     5. N1 independent FFTs of length N2     (local, OpenMP across rows)
 *   The result C[k1][k2] relates to the true spectrum by X[k1 + N1*k2] = C[k1][k2],
 *   i.e. X = C^T flattened — tools/check_fft.py applies that reordering when
 *   comparing against numpy.
 *
 * Derivation (used in the report):
 *   X[k1 + N1*k2] = sum_n2 W_N^(n2*k1) * ( sum_n1 A[n1][n2] W_N1^(n1*k1) ) * W_N2^(n2*k2)
 *
 * Rows are block-distributed: rank r holds rows [r*N1/P, (r+1)*N1/P). P must divide
 * both N1 and N2 (all powers of two here, so any power-of-two P works).
 *
 * Usage:  mpiexec -n <P> fft_hybrid <log2N> [input.bin] [output.bin]
 *         (OpenMP threads per process: OMP_NUM_THREADS)
 * Prints: RESULT,mpi,<log2N>,<P>,<threads>,<total_s>,<comm_s>
 */
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "fft_kernel.h"

/* Distributed transpose: `in` is my lrA rows x dimB cols slice of a (P*lrA) x dimB
 * matrix; `out` receives my lrB rows x dimA cols slice of its transpose
 * (dimB = P*lrB, dimA = P*lrA). sbuf/rbuf are scratch of the same size. */
static void transpose_exchange(const cplx *in, cplx *out, cplx *sbuf, cplx *rbuf,
                               size_t lrA, size_t lrB, int P, MPI_Comm comm) {
    size_t dimB = (size_t)P * lrB;
    size_t blk = lrA * lrB;                       /* elements per rank pair */
    /* pack: block destined for rank q = my rows x q's columns */
    #pragma omp parallel for schedule(static)
    for (long long q = 0; q < P; q++)
        for (size_t r = 0; r < lrA; r++)
            for (size_t c = 0; c < lrB; c++)
                sbuf[(size_t)q * blk + r * lrB + c] = in[r * dimB + (size_t)q * lrB + c];

    MPI_Alltoall(sbuf, (int)(blk * 2), MPI_DOUBLE, rbuf, (int)(blk * 2), MPI_DOUBLE, comm);

    /* unpack: block from rank s holds global rows [s*lrA ...) of my columns */
    size_t dimA = (size_t)P * lrA;
    #pragma omp parallel for schedule(static)
    for (long long s = 0; s < P; s++)
        for (size_t r = 0; r < lrA; r++)
            for (size_t c = 0; c < lrB; c++)
                out[c * dimA + (size_t)s * lrA + r] = rbuf[(size_t)s * blk + r * lrB + c];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) { if (!rank) fprintf(stderr, "usage: mpiexec -n <P> %s <log2N> [in.bin] [out.bin]\n", argv[0]); MPI_Finalize(); return 1; }
    int p = atoi(argv[1]);
    size_t N  = (size_t)1 << p;
    size_t N1 = (size_t)1 << ((p + 1) / 2);       /* N1 >= N2, both powers of two */
    size_t N2 = (size_t)1 << (p / 2);
    size_t lr1 = N1 / (size_t)P, lr2 = N2 / (size_t)P;
    if (lr1 == 0 || lr2 == 0 || lr1 * (size_t)P != N1 || lr2 * (size_t)P != N2) {
        if (!rank) fprintf(stderr, "P=%d must divide N1=%zu and N2=%zu\n", P, N1, N2);
        MPI_Finalize(); return 1;
    }

    size_t L = lr1 * N2;                           /* local elements = N/P */
    cplx *A    = (cplx *)malloc(L * sizeof *A);    /* data  */
    cplx *T    = (cplx *)malloc(L * sizeof *T);    /* transposed view */
    cplx *sbuf = (cplx *)malloc(L * sizeof *sbuf);
    cplx *rbuf = (cplx *)malloc(L * sizeof *rbuf);
    if (!A || !T || !sbuf || !rbuf) { fprintf(stderr, "rank %d: out of memory (N=2^%d, P=%d)\n", rank, p, P); MPI_Abort(MPI_COMM_WORLD, 1); }

    size_t r0 = (size_t)rank * lr1;                /* my first global row of A */
    if (argc >= 3) {                               /* read my slice of the input */
        MPI_File f;
        if (MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f) != MPI_SUCCESS) {
            if (!rank) fprintf(stderr, "cannot open %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_File_read_at(f, (MPI_Offset)(r0 * N2 * sizeof(cplx)), A, (int)(L * 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&f);
    } else {                                       /* same synthetic signal as fft_seq */
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)L; i++) {
            size_t g = r0 * N2 + (size_t)i;
            double t = (double)g / 16000.0;
            A[i].re = sin(2 * M_PI * 440 * t) + 0.5 * sin(2 * M_PI * 1000 * t);
            A[i].im = 0.0;
        }
    }

    int threads = omp_get_max_threads();
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime(), comm = 0.0, tc;

    /* 1. transpose A (N1 x N2) -> T (N2 x N1) */
    tc = MPI_Wtime();
    transpose_exchange(A, T, sbuf, rbuf, lr1, lr2, P, MPI_COMM_WORLD);
    comm += MPI_Wtime() - tc;

    /* 2. FFTs of length N1 along my lr2 rows of T */
    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < (long long)lr2; r++)
        fft_radix2(T + (size_t)r * N1, N1);

    /* 3. twiddle: T[n2][k1] *= W_N^(n2*k1) */
    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < (long long)lr2; r++) {
        size_t n2 = (size_t)rank * lr2 + (size_t)r;
        for (size_t k1 = 0; k1 < N1; k1++) {
            unsigned long long m = ((unsigned long long)n2 * k1) % (unsigned long long)N;
            double ang = -2.0 * M_PI * (double)m / (double)N;
            double wr = cos(ang), wi = sin(ang);
            cplx v = T[(size_t)r * N1 + k1];
            T[(size_t)r * N1 + k1].re = v.re * wr - v.im * wi;
            T[(size_t)r * N1 + k1].im = v.re * wi + v.im * wr;
        }
    }

    /* 4. transpose back T (N2 x N1) -> A (N1 x N2), rows now indexed by k1 */
    tc = MPI_Wtime();
    transpose_exchange(T, A, sbuf, rbuf, lr2, lr1, P, MPI_COMM_WORLD);
    comm += MPI_Wtime() - tc;

    /* 5. FFTs of length N2 along my lr1 rows */
    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < (long long)lr1; r++)
        fft_radix2(A + (size_t)r * N2, N2);

    MPI_Barrier(MPI_COMM_WORLD);
    double total = MPI_Wtime() - t0;
    double total_max, comm_max;
    MPI_Reduce(&total, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm,  &comm_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (argc >= 4) {                               /* write C row-major; X = C^T flat */
        MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f);
        MPI_File_write_at(f, (MPI_Offset)(r0 * N2 * sizeof(cplx)), A, (int)(L * 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_close(&f);
    }

    if (!rank) {
        printf("N=2^%d (N1=%zu x N2=%zu)  hybrid FFT, %d procs x %d threads: %.4f s (comm %.4f s = %.0f%%)\n",
               p, N1, N2, P, threads, total_max, comm_max, 100.0 * comm_max / total_max);
        printf("RESULT,mpi,%d,%d,%d,%.6f,%.6f\n", p, P, threads, total_max, comm_max);
    }

    free(A); free(T); free(sbuf); free(rbuf);
    MPI_Finalize();
    return 0;
}
