# Viva preparation — every member must be able to answer ALL of these

The viva is 3/10 marks and the rule is "every group member must be able to explain and
defend the full project." Practice out loud, not just reading.

## The algorithm

**Q: What does an FFT compute?**
The Discrete Fourier Transform X[k] = Σ x[n]·e^(−2πi·nk/N) — how much of each frequency
is in the signal — in O(N log N) instead of the DFT's O(N²). For our 2^24-point signal
that's ~4×10^8 operations instead of ~2.8×10^14.

**Q: How does radix-2 Cooley–Tukey work (`src/fft_kernel.h`)?**
Reorder the input by bit-reversed index, then log2(N) stages; each stage does N/2
"butterflies": a' = a + w·b, b' = a − w·b, where w is a twiddle factor e^(−2πik/len).
Each stage doubles the transform length.

**Q: Derive the four-step decomposition (`src/fft_hybrid_mpi.c` header comment).**
Write n = n1·N2 + n2 and k = k1 + N1·k2. Substituting into the DFT splits it into:
(1) N2 FFTs of length N1 over n1, (2) a twiddle multiply by W_N^(n2·k1), (3) N1 FFTs
of length N2 over n2. Viewing x as an N1×N2 matrix, step (1) works on columns — so we
transpose first, FFT rows, twiddle, transpose back, FFT rows. Output satisfies
X[k1 + N1·k2] = C[k1][k2], i.e. the file we write is the transposed spectrum.

**Q: Why is the transpose the communication step?**
Rows are block-distributed across ranks. A transpose means every rank needs a piece of
every other rank's rows — exactly MPI_Alltoall: P·(P−1) messages of N·16/P² bytes.

## Parallelization choices

**Q: Where is the OpenMP parallelism in each version?**
- `fft_omp.c`: the n/2 butterflies of ONE stage are independent (disjoint index pairs),
  so one `#pragma omp parallel for` per stage over a flattened butterfly index.
- `fft_hybrid_mpi.c`: whole ROWS are independent FFTs — `parallel for` over rows, each
  thread runs the serial kernel on its rows. Also the pack/unpack loops and twiddle.

**Q: Why is there no race condition?**
Within a stage, butterfly b touches only indices (blk+k, blk+k+half); different b values
touch disjoint pairs. Across rows, each row is a separate memory range. Stages are
separated by the implicit barrier at the end of `parallel for`.

**Q: Why was OpenMP-1-thread SLOWER than sequential (7.8 s vs 5.0 s on Colab)?**
The flattened loop pays an integer divide + modulo per butterfly to recover (block, k),
and the parallel-for machinery adds per-stage overhead. That's measured parallelization
overhead — Amdahl's law in the small. (The laptop numbers quantify it too.)

**Q: What stays serial? (Amdahl)**
Bit-reversal reorder (O(N)), file I/O, and in the hybrid the pack/unpack memory copies.
Speedup = 1/((1−f) + f/p) — with the serial fraction measured from our timings.

**Q: Why hybrid instead of pure MPI?**
On one node, OpenMP threads share memory — no packing or message overhead. MPI is kept
for what it's needed for: crossing machine boundaries. P ranks × T threads also cuts the
Alltoall message count P² versus using (P·T) pure ranks.

## The measurements

**Q: Define speedup and efficiency.**
S(p) = T_seq / T_p ; E(p) = S(p)/p. We plot both against workers = procs × threads,
best-of-3 repetitions.

**Q: What limited OpenMP speedup on the laptop?**
Memory bandwidth: an FFT does little arithmetic per byte (log N passes over 256 MiB),
so 16 threads saturate DRAM long before 16× compute. Also 10 physical cores — the
extra 6 are SMT siblings.

**Q: What did the communication share show?**
~17–20% of hybrid runtime at 2^24 even on ONE machine (memory pack/unpack + local
exchange). Over Wi-Fi/LAN it grows to dominate — the compute/communication ratio
N log N : N means bigger N amortizes communication better (weak scaling argument).

**Q: Why does Colab show no speedup?**
Free Colab allocates 2 vCPUs. Honest finding, kept in the report: parallel speedup
needs real cores; Colab is our Linux build/correctness/data environment.

**Q: The crossover graph — what's the point?**
At 512 points (one 20–32 ms voice frame) 16 threads are far SLOWER than one thread
(overhead ≫ work: a 512-pt FFT is ~µs, an OpenMP region ~2 µs, an MPI message 1–30 µs).
Parallelism pays only above ~10^5–10^6 points. Real-time speech systems therefore
parallelize ACROSS calls/frames (throughput), not inside one small FFT (latency).

**Q: How do you know the answers are right?**
`tools/check_fft.py` compares every version against numpy.fft on random complex input:
max relative error ~1e-15 (double-precision rounding), on Windows and Linux, P=2 and 4.

## Data & engineering

**Q: What data, and is it legal?**
Real Urdu speech: YODAS `ur000` (22.4 h, CC-BY-3.0, built from Creative-Commons YouTube
videos — we keep the video IDs for attribution) and Mozilla Common Voice Urdu (CC0).
`tools/prepare_signal.py` decodes to mono 16 kHz, concatenates, normalizes, writes
complex doubles. 2^24 = 17.5 min, 2^26 = 70 min, 2^28 = 4.7 h.

**Q: Why N1 = N2 = √N?**
Balances the two FFT passes and makes both transposes move the same data; any split
works, powers of two keep the radix-2 kernel valid for rows.

**Q: Memory cost of the hybrid?**
4 buffers of N/P each per rank (data, transposed view, send, recv) = 4·16·N bytes total
across ranks: 1 GiB at 2^24, 4 GiB at 2^26 — why 2^28 needs multiple machines (the
"distributed" in distributed FFT is about memory as much as speed).

**Q: How are ranks reading one file?**
MPI-IO (`MPI_File_read_at`): each rank seeks to its row offset — no rank-0 scatter.

**Q: One thing you'd do next?**
Precompute twiddle tables (kills the cos/sin per butterfly), SIMD-vectorize the kernel,
overlap transpose communication with row FFTs (MPI_Ialltoall), compare against FFTW.

## Each member should own one deep area
- Member A: kernel + OpenMP version internals (stages, butterflies, race-freedom)
- Member B: four-step math + MPI transpose + MPI-IO
- Member C: benchmark methodology + graphs + data pipeline/licensing
...but ANYONE can be asked ANY question above.
