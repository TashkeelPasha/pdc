/*
 * Step 5 — batched spectrogram engine (hybrid MPI + OpenMP).
 *
 * The industrially realistic use of a speech FFT: not one giant transform, but
 * short-time FFTs over THOUSANDS of clips — the feature-extraction stage that
 * feeds ASR training pipelines. Parallel structure:
 *   - MPI  : master-worker over audio FILES (rank 0 dispatches; clips vary in
 *            length, so dynamic dispatch load-balances better than a static split)
 *   - OpenMP: across the FRAMES of each file (frames are independent FFTs)
 *
 * Framing follows Whisper's front end: 25 ms window (400 samples @16 kHz),
 * 10 ms hop (160), Hann window, zero-padded to a 512-point radix-2 FFT
 * (reusing src/fft_kernel.h). Log-power spectra are computed in memory and
 * checksummed (writing thousands of outputs would benchmark the disk, not us).
 *
 * Usage:
 *   python tools/make_filelist.py data/audio filelist.txt
 *   mpiexec -n 4 ./build/spectrogram_engine filelist.txt     (OMP_NUM_THREADS=4)
 *
 * Prints: RESULT,spec,<files>,<procs>,<threads>,<wall_s>,<audio_s>
 *         (throughput = audio_s / wall_s = "x realtime")
 */
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fft_kernel.h"

#define WIN 400                 /* 25 ms @ 16 kHz */
#define HOP 160                 /* 10 ms          */
#define NFFT 512                /* zero-padded radix-2 size */
#define MAX_FILES 1000000
#define TAG_ASK 1
#define TAG_JOB 2

/* Minimal RIFF/WAVE reader: PCM16 mono 16 kHz (the YODAS format). Returns
 * sample count and malloc'd samples, or 0 to skip the file. */
static size_t read_wav16k(const char *path, double **out) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint8_t h[12];
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) { fclose(f); return 0; }
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0, dlen = 0;
    long dpos = -1;
    uint8_t ck[8];
    while (fread(ck, 1, 8, f) == 8) {                  /* walk chunks */
        uint32_t len = ck[4] | ck[5] << 8 | ck[6] << 16 | (uint32_t)ck[7] << 24;
        if (!memcmp(ck, "fmt ", 4)) {
            uint8_t b[16];
            if (len < 16 || fread(b, 1, 16, f) != 16) { fclose(f); return 0; }
            fmt = b[0] | b[1] << 8; ch = b[2] | b[3] << 8;
            rate = b[4] | b[5] << 8 | b[6] << 16 | (uint32_t)b[7] << 24;
            bits = b[14] | b[15] << 8;
            fseek(f, (long)len - 16 + (len & 1), SEEK_CUR);
        } else if (!memcmp(ck, "data", 4)) {
            dlen = len; dpos = ftell(f);
            fseek(f, (long)len + (len & 1), SEEK_CUR);
        } else {
            fseek(f, (long)len + (len & 1), SEEK_CUR);
        }
    }
    if (fmt != 1 || ch != 1 || rate != 16000 || bits != 16 || dpos < 0) { fclose(f); return 0; }
    size_t n = dlen / 2;
    int16_t *raw = (int16_t *)malloc(dlen);
    fseek(f, dpos, SEEK_SET);
    if (fread(raw, 2, n, f) != n) { free(raw); fclose(f); return 0; }
    fclose(f);
    double *x = (double *)malloc(n * sizeof *x);
    for (size_t i = 0; i < n; i++) x[i] = raw[i] / 32768.0;
    free(raw);
    *out = x;
    return n;
}

/* Log-power spectrogram of one clip; returns samples processed. The checksum
 * accumulation stops the compiler from deleting the work. */
static size_t process_file(const char *path, const double *hann, double *checksum) {
    double *x;
    size_t n = read_wav16k(path, &x);
    if (n < WIN) { free(x); return 0; }
    long long nfr = (long long)((n - WIN) / HOP) + 1;
    double local = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:local)
    for (long long fr = 0; fr < nfr; fr++) {
        cplx buf[NFFT];                                /* per-thread, on stack */
        const double *seg = x + (size_t)fr * HOP;
        for (int i = 0; i < WIN; i++) { buf[i].re = seg[i] * hann[i]; buf[i].im = 0.0; }
        memset(buf + WIN, 0, (NFFT - WIN) * sizeof(cplx));
        fft_radix2(buf, NFFT);
        for (int k = 0; k <= NFFT / 2; k++) {          /* one-sided log-power */
            double p = buf[k].re * buf[k].re + buf[k].im * buf[k].im;
            local += log(p + 1e-10);
        }
    }
    *checksum += local;
    free(x);
    return n;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    if (argc < 2) { if (!rank) fprintf(stderr, "usage: mpiexec -n <P> %s <filelist.txt>\n", argv[0]); MPI_Finalize(); return 1; }

    /* every rank loads the file list */
    char (*names)[260] = malloc((size_t)MAX_FILES * 260);
    if (!names) { fprintf(stderr, "rank %d: cannot allocate file list\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    int nfiles = 0;
    FILE *fl = fopen(argv[1], "r");
    if (!fl) { if (!rank) fprintf(stderr, "cannot open %s\n", argv[1]); MPI_Finalize(); return 1; }
    while (nfiles < MAX_FILES && fgets(names[nfiles], 260, fl)) {
        names[nfiles][strcspn(names[nfiles], "\r\n")] = 0;
        if (names[nfiles][0]) nfiles++;
    }
    fclose(fl);

    double hann[WIN];
    for (int i = 0; i < WIN; i++) hann[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / (WIN - 1));

    int threads = omp_get_max_threads();
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    size_t my_samples = 0;
    double checksum = 0.0;
    int my_files = 0;

    if (P == 1) {                                      /* no master needed */
        for (int i = 0; i < nfiles; i++) { my_samples += process_file(names[i], hann, &checksum); my_files++; }
    } else if (rank == 0) {                            /* master: dynamic dispatch */
        int next = 0, done = 0, worker;
        MPI_Status st;
        while (done < P - 1) {
            MPI_Recv(&worker, 1, MPI_INT, MPI_ANY_SOURCE, TAG_ASK, MPI_COMM_WORLD, &st);
            int job = (next < nfiles) ? next++ : -1;
            if (job < 0) done++;
            MPI_Send(&job, 1, MPI_INT, st.MPI_SOURCE, TAG_JOB, MPI_COMM_WORLD);
        }
    } else {                                           /* worker: ask, process, repeat */
        for (;;) {
            int job;
            MPI_Send(&rank, 1, MPI_INT, 0, TAG_ASK, MPI_COMM_WORLD);
            MPI_Recv(&job, 1, MPI_INT, 0, TAG_JOB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (job < 0) break;
            my_samples += process_file(names[job], hann, &checksum);
            my_files++;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double wall = MPI_Wtime() - t0;
    long long samp = (long long)my_samples, samp_all = 0;
    int files_all = 0;
    double chk_all = 0.0;
    MPI_Reduce(&samp, &samp_all, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_files, &files_all, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&checksum, &chk_all, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!rank) {
        double audio_s = samp_all / 16000.0;
        printf("spectrograms: %d files, %.1f min of audio, %d procs x %d threads: %.2f s wall = %.1fx realtime (checksum %.3e)\n",
               files_all, audio_s / 60.0, P, threads, wall, audio_s / wall, chk_all);
        printf("RESULT,spec,%d,%d,%d,%.6f,%.1f\n", files_all, P, threads, wall, audio_s);
    }
    free(names);
    MPI_Finalize();
    return 0;
}
