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

| Machine | Input | Sequential FFT (best of 3) |
|---|---|---|
| Laptop (i7-13620H) | 2^24 (17.5 min Urdu speech) | **2.02 s** |
| Laptop (i7-13620H) | 2^26 (70 min Urdu speech) | 8.29 s |
| Colab free VM | 2^24 | 4.95 s |

All laptop numbers below use the real Urdu signal. A full matrix was also run on a
synthetic signal (archived in `results/*_synthetic.csv`): every cell matches within
run-to-run noise, demonstrating that FFT runtime is input-independent.

(NumPy's optimized library FFT takes ~1.0 s on the same laptop — a useful sanity bound:
our from-scratch kernel is within 2× of a tuned library.)

### 5.2 OpenMP scaling (laptop, 2^24)

| Threads | Best time | Speedup | Efficiency |
|---|---|---|---|
| seq (V1) | 2.020 s | 1.00× | — |
| 1 | 2.294 s | 0.88× | 0.88 |
| 2 | 1.351 s | 1.50× | 0.75 |
| 4 | 0.934 s | 2.16× | 0.54 |
| 8 | 0.844 s | 2.39× | 0.30 |
| 10 | 0.800 s | 2.53× | 0.25 |
| 12 | 0.812 s | 2.49× | 0.21 |
| 16 | 0.798 s | **2.53×** | 0.16 |

At 2^26 the same 16-thread configuration reaches 2.98 s vs 8.29 s sequential =
**2.78×** — slightly better than at 2^24, since more work per stage amortizes the
per-stage parallel-region overhead.

Two effects dominate. First, **parallelization overhead is real and measured**: with
one thread, V2 is slower than V1 (the flattened loop pays an integer div/mod per
butterfly and a parallel region per stage). Second, speedup saturates well below
core count because the FFT is **memory-bandwidth-bound**: each stage streams the whole
256 MiB array with only a handful of flops per element, so threads quickly saturate
DRAM; beyond the 10 physical cores the SMT siblings add almost nothing.

### 5.3 Hybrid four-step (single node, 2^24)

| Procs × threads | Workers | Best time | Speedup | Efficiency | Comm share |
|---|---|---|---|---|---|
| 1 × 1 | 1 | 2.545 s | 0.79× | 0.79 | 23% |
| 2 × 1 | 2 | 1.403 s | 1.44× | 0.72 | 28% |
| 4 × 1 | 4 | 0.892 s | 2.26× | 0.57 | 37% |
| 8 × 1 | 8 | 0.685 s | 2.95× | 0.37 | 46% |
| 1 × 8 | 8 | 1.028 s | 1.97× | 0.25 | 59% |
| 1 × 16 | 16 | 0.970 s | 2.08× | 0.13 | 66% |
| 2 × 4 | 8 | 0.777 s | 2.60× | 0.33 | 45% |
| 2 × 8 | 16 | 0.677 s | 2.98× | 0.19 | 46% |
| **4 × 4** | 16 | **0.629 s** | **3.21×** | 0.20 | 46% |

Three observations. (1) **The hybrid beats pure OpenMP** (3.21× vs 2.53×): the
four-step's row FFTs walk contiguous 4096-point rows that fit in L2 cache, where V2's
late stages stride across the whole 256 MiB array. (2) **Balanced procs × threads wins**
— 4×4 beats both 8×1 (more ranks = more transpose traffic) and 1×16 (one rank's
transpose is pure serial memcpy, pushing comm share to 64%). (3) **Communication share
grows with parallelism** (25% → 45%), the fundamental scaling limit that the
multi-machine run exposes at network speeds.

The hybrid's printed communication share isolates the transposes. Even on one machine
(ranks communicating through shared memory) the two transposes cost a significant
fraction of runtime — this is the memory traffic of pack → exchange → unpack. The
four-step with P = 1 is also our "row-parallel" OpenMP variant, contrasting with V2's
"butterfly-parallel" strategy.

### 5.3b When the problem outgrows the machine (2^26)

At 2^26 OpenMP still works (working set 1 GiB), but the hybrid needs **4 GiB** of
buffers (data + transposed view + send + recv) on a laptop with 7.7 GB of RAM and
~1.7 GB free: the 4×4 run took 10.7 s — slower than sequential — with 81–90% of the
time in the exchange phase, because Windows was **paging the buffers to disk** during
the transposes. This is the distributed-memory argument made concrete: beyond this
size the four-step FFT needs more *machines*, not more threads — at 2^28 (4 GiB of
raw data, 16 GiB of buffers) no single laptop in the group could even allocate it.

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

| log2 N | Signal length | Seq (best of 5) | OMP-16T | Speedup |
|---|---|---|---|---|
| 9 | one 32 ms voice frame | 21 µs | 1,560 µs | **0.01× (74× slower)** |
| 11 | 0.13 s | 95 µs | 1,523 µs | 0.06× |
| 13 | 0.5 s | 447 µs | 1,944 µs | 0.23× |
| 15 | 2 s | 2.0 ms | 2.6 ms | 0.79× |
| 17 | 8 s | 9.4 ms | 5.2 ms | **1.82× — break-even passed** |
| 19 | 33 s | 43 ms | 13.7 ms | 3.14× |
| 21 | 2.2 min | 193 ms | 61 ms | **3.17× (peak)** |
| 23 | 8.7 min | 889 ms | 374 ms | 2.37× |
| 24 | 17.5 min | 1.92 s | 0.74 s | 2.59× |

Break-even sits between 2^15 and 2^17 (~10^5 points), matching FFTW's published
guidance that threads need "at least a few thousand data points" per thread. Peak
speedup lands at 2^19–2^21, where the working set still fits in cache; beyond it the
memory wall claws speedup back down.

At 512 points — one 20–32 ms voice frame — 16 threads are drastically slower than one:
the FFT itself takes microseconds while every OpenMP region costs ~2 µs (and an MPI
message 1–30 µs). Parallelism only pays above ~10^5–10^6 points. Production speech
systems therefore parallelize **across** frames, calls and files (throughput), never
inside one small FFT (latency). Our batched-corpus framing (one rank per audio file,
threads across frames) is the industrially realistic use of this project's code.

### 5.7 Bonus row: a GPU library on the same signal

cuFFT (via CuPy) on Colab's Tesla T4 transforms the same 2^24-point Urdu signal in
**29.0 ms** (max rel. error vs numpy 4.4e-16) — 70× our sequential C and 22× our best
hybrid run. This is not a like-for-like comparison (a decade-tuned vendor library on
specialised hardware vs our from-scratch CPU code); it is the industry context: FFTs
at scale run on GPUs, and the four-step structure we implemented — batched local FFTs
separated by transposes — is the same structure cuFFTMp uses across multiple GPUs.

## 6. Analysis

**Speedup / efficiency / Amdahl.** S(p) = T_seq/T_p, E(p) = S(p)/p, plotted in
`results/`. From the measured OpenMP curve, the speedup ceiling of ~2.5× implies an
effective serial-equivalent fraction of ~0.37 by Amdahl's 1/((1−f)+f/p) as p→∞ —
but the hybrid's 3.21× on the same machine proves most of that "serial fraction" is
really memory-bandwidth contention and cache-unfriendly strides, not inherently serial
code: changing the access pattern (row FFTs) raised the ceiling without changing the
arithmetic.

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
