# Hybrid MPI + OpenMP Parallel FFT for Large Speech Signals

**PDC Semester Project #11 — Final Report**

M Tashkeel Pasha (23k-2014) · M Kashan (23k-0043) · Hamadullah (23k-0723)

Repository: https://github.com/TashkeelPasha/pdc

---

## 1. Problem and motivation

The Fast Fourier Transform converts a signal into its frequency content in O(N log N)
operations and is the computational core of speech processing: every speech-recognition
front end (e.g. Whisper's log-mel spectrogram) and most audio codecs are built on it.
We implement the FFT from scratch in C and parallelize it two ways — OpenMP within a
machine and MPI across machines — on signals large enough that the sequential version
is noticeably slow: 2^24 to 2^26 complex points, i.e. **17.5 to 70 minutes of real
Urdu speech at 16 kHz**. At 2^24 points an optimized single-threaded library FFT
(NumPy) already takes ~1 s and our sequential C implementation 2–6 s depending on the
machine, so the problem size qualifies as large, and at 2^28 the working set of the
distributed algorithm (4 GiB of data plus equal-sized communication buffers) exceeds a
single 8 GB laptop — distribution becomes necessary, not just faster.

A deliberate secondary question: our data is phone-style speech, so **does parallelism
help a real-time voice pipeline?** Real pipelines transform one 512-point frame
(20–32 ms) at a time. We measure the crossover size below which parallelism hurts.

## 2. Data

Real Urdu speech, openly licensed:

| Source | License | Used for |
|---|---|---|
| YODAS `ur000` (espnet/yodas, Hugging Face) — 22.4 h mined from Creative-Commons YouTube videos | CC-BY-3.0 (clip IDs retained for attribution) | main signal |
| Mozilla Common Voice Urdu | CC0 | extension to 2^28 |

`tools/prepare_signal.py` decodes each clip to mono 16 kHz, concatenates, peak-normalizes,
and writes N complex doubles. The 2^24 signal used throughout was built from 132 YODAS
clips (256 MiB on disk). Audio is never committed to the repository.

## 3. Implementations

All three share one radix-2 iterative Cooley–Tukey kernel (`src/fft_kernel.h`):
bit-reversal reorder, then log2 N stages of N/2 butterflies
(a' = a + w·b, b' = a − w·b).

**V1 — sequential (`src/fft_seq.c`).** The kernel on the whole array. Baseline for all
speedups.

**V2 — OpenMP (`src/fft_omp.c`).** Within one stage, the N/2 butterflies touch disjoint
index pairs, so they are flattened into a single iteration space and split with one
`#pragma omp parallel for schedule(static)` per stage. The implicit barrier between
stages provides the required ordering; the bit-reversal stays serial (an O(N) Amdahl
term discussed in §5).

**V3 — hybrid MPI + OpenMP (`src/fft_hybrid_mpi.c`), Bailey's four-step algorithm.**
Writing n = n1·N2 + n2 and k = k1 + N1·k2 factors the DFT into:

> X[k1 + N1·k2] = Σ_n2 W_N^(n2·k1) · ( Σ_n1 A[n1][n2] · W_N1^(n1·k1) ) · W_N2^(n2·k2)

Viewed as an N1×N2 matrix (N1 = N2 = √N here), that is: **global transpose →
N2 row-FFTs of length N1 → twiddle multiply → global transpose → N1 row-FFTs of
length N2**. Rows are block-distributed over P ranks; each transpose is one
`MPI_Alltoall` (the only communication, timed separately on every run), and each rank's
row FFTs are OpenMP-parallel. Input/output use MPI-IO (`MPI_File_read_at`/`write_at`)
so no rank gathers the whole signal. The output C[k1][k2] is the spectrum with
X[k1 + N1·k2] = C[k1][k2]; the checker accounts for this reordering.

**Correctness.** `tools/check_fft.py` compares every version against `numpy.fft.fft`
on random complex input: max relative error ≈ 1e-15 (double-precision rounding) on
Windows (mingw gcc + MS-MPI) and Linux (gcc 13 + Open MPI), at P = 2 and P = 4.

## 4. Experimental setup

| Environment | Hardware | Role |
|---|---|---|
| Laptop | i7-13620H, 10 cores / 16 threads, 7.7 GB RAM, Windows 11 | OpenMP + single-node hybrid scaling |
| Google Colab (free) | 2 vCPUs, 12 GB RAM, Ubuntu 24.04 | Linux build, data prep, correctness, 2-vCPU control run |
| 2–3 laptops over LAN (MS-MPI) | group machines, Wi-Fi | multi-node hybrid (`bench/run_mpi_lan.md`) |

Method: best of 3 repetitions (5 for the crossover), timing only the transform
(I/O excluded); `RESULT` lines collected into `results/*.csv`;
`bench/plot_results.py` produces every graph from the raw CSVs.

## 5. Results

> Graphs: `results/time_24.png`, `results/speedup_24.png`, `results/efficiency_24.png`,
> `results/crossover.png`. Raw data: `results/*.csv`.

### 5.1 Sequential baseline

<!-- RESULTS:SEQ -->

### 5.2 OpenMP scaling (laptop, 2^24)

<!-- RESULTS:OMP -->

Two effects dominate. First, **parallelization overhead is real and measured**: with
one thread, V2 is slower than V1 (the flattened loop pays an integer div/mod per
butterfly and a parallel region per stage). Second, speedup saturates well below
core count because the FFT is **memory-bandwidth-bound**: each stage streams the whole
256 MiB array with only a handful of flops per element, so threads quickly saturate
DRAM; beyond the 10 physical cores the SMT siblings add almost nothing.

### 5.3 Hybrid four-step (single node, 2^24)

<!-- RESULTS:MPI -->

The hybrid's printed communication share isolates the transposes. Even on one machine
(ranks communicating through shared memory) the two transposes cost a significant
fraction of runtime — this is the memory traffic of pack → exchange → unpack. The
four-step with P = 1 is also our "row-parallel" OpenMP variant, contrasting with V2's
"butterfly-parallel" strategy.

### 5.4 Control run: Colab's 2 vCPUs

On the free Colab VM (2 vCPUs) the same binaries show **no speedup** (seq 4.95 s;
OpenMP-2T 6.67 s; hybrid n=2 5.20 s, n=4 5.26 s — best of 2). This is kept
deliberately: parallel speedup requires physical cores, and oversubscribed vCPUs
measure scheduling, not scaling.

### 5.5 Multi-machine run

<!-- RESULTS:LAN (fill after the group LAN session; see bench/run_mpi_lan.md) -->

Expectation set by the model in §6: over Wi-Fi the transpose ships ~2·N·16·(P−1)/P
bytes per FFT at ~10–30 MB/s, so communication dominates at 2^24 and the run may be
slower than one machine — the compute/communication ratio (N log N : N) improves with
larger N, which is the weak-scaling argument for 2^26/2^28.

### 5.6 Crossover: why real-time speech doesn't parallelize one frame

<!-- RESULTS:CROSSOVER -->

At 512 points — one 20–32 ms voice frame — 16 threads are drastically slower than one:
the FFT itself takes microseconds while every OpenMP region costs ~2 µs (and an MPI
message 1–30 µs). Parallelism only pays above ~10^5–10^6 points. Production speech
systems therefore parallelize **across** frames, calls and files (throughput), never
inside one small FFT (latency). Our batched-corpus framing (one rank per audio file,
threads across frames) is the industrially realistic use of this project's code.

## 6. Analysis

**Speedup / efficiency / Amdahl.** S(p) = T_seq/T_p, E(p) = S(p)/p, plotted in
`results/`. Fitting Amdahl's 1/((1−f)+f/p) to the laptop curve estimates the effective
serial fraction f (bit-reversal + per-stage overhead + bandwidth saturation acting as
serialization). <!-- fill fitted f after benchmarks -->

**Communication model.** Per transpose each rank sends P−1 blocks of N·16/P² bytes;
total bytes on the wire ≈ 2·N·16·(P−1)/P per FFT. On one node this is memcpy traffic;
across machines it divides by network bandwidth — measured directly as the printed
`comm` share.

**Memory.** The four-step needs 4 buffers of N/P per rank (data, transposed view,
send, recv) = 64·N bytes total: 1 GiB at 2^24, 4 GiB at 2^26, 16 GiB at 2^28 — the
last cannot fit one laptop, making the distributed version necessary, not optional.

## 7. Conclusions

1. From-scratch radix-2 + four-step FFT verified to numpy precision on two OSes.
2. Parallelism pays on large signals (hours of speech, corpus processing) and costs
   on voice-frame sizes — quantified by the crossover experiment.
3. The FFT is bandwidth-bound: thread speedup saturates near the memory wall, and the
   distributed version's ceiling is the transpose's network cost.
4. Honest negative results (Colab's 2 vCPUs, OpenMP-1T overhead, Wi-Fi transpose cost)
   are measurements, not failures — they map exactly where each parallel model is the
   right tool.

## 8. Work split

| Member | Owned area (all members can defend all parts) |
|---|---|
| M Tashkeel Pasha | four-step MPI algorithm, MPI-IO, repo/Colab infrastructure |
| M Kashan | OpenMP versions, correctness tooling, crossover experiment |
| Hamadullah | data pipeline (YODAS/Common Voice), benchmark runs, graphs |

## References

- J. W. Cooley, J. W. Tukey, "An Algorithm for the Machine Calculation of Complex
  Fourier Series," Math. Comp., 1965.
- D. H. Bailey, "FFTs in External or Hierarchical Memory," J. Supercomputing, 1990
  (the four-step algorithm).
- MPI 4.0 Standard — MPI_Alltoall, MPI-IO. OpenMP 5.2 Specification.
- YODAS: Li et al., "YODAS: Youtube-Oriented Dataset for Audio and Speech," ASRU 2023;
  dataset espnet/yodas (CC-BY-3.0). Mozilla Common Voice (CC0).
- FFTW documentation (threading thresholds), used for comparison discussion only; all
  project FFT code is our own.
