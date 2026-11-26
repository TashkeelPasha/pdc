# Hybrid MPI + OpenMP Parallel FFT for Large Speech Signals

PDC semester project (#11 — Parallel FFT on Large Signals), applied to real Urdu speech audio.

**Group:** M Tashkeel Pasha (23k-2014) · M Kashan (23k-0043) · Hamadullah (23k-0723)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TashkeelPasha/pdc/blob/main/notebooks/PDC_FFT.ipynb)

## What it does

Computes the FFT of 2^24–2^28-point signals (17.5 minutes to 4.7 hours of 16 kHz Urdu
speech) three ways, and measures where parallelism pays off:

| Version | File | Parallel model |
|---|---|---|
| Sequential baseline | `src/fft_seq.c` | none — iterative radix-2 Cooley–Tukey |
| OpenMP | `src/fft_omp.c` | n/2 butterflies per stage split across threads |
| Hybrid | `src/fft_hybrid_mpi.c` | Bailey four-step: MPI_Alltoall transposes between ranks, OpenMP across local row FFTs |

Data: [YODAS Urdu `ur000`](https://huggingface.co/datasets/espnet/yodas) (CC-BY-3.0) and
Mozilla Common Voice Urdu (CC0). Raw audio is **not** committed — `tools/prepare_signal.py`
builds the input signal from it.

## Build & run

**Colab / Linux** — open the notebook badge above, or:
```bash
sudo apt install -y openmpi-bin libopenmpi-dev
make                         # builds build/fft_seq, fft_omp, fft_hybrid
python3 tools/check_fft.py 16        # verify all three against numpy
mpiexec -n 4 --allow-run-as-root --oversubscribe ./build/fft_hybrid 24
```

**Windows (mingw gcc + MS-MPI)**
```powershell
.\build_all.ps1
python tools\check_fft.py 16
mpiexec -n 4 .\build\fft_hybrid.exe 24
```

With a real speech input: `.\build\fft_hybrid.exe 24 data\urdu_2e24.bin out.bin`
(make the .bin with `python tools/prepare_signal.py <audio_dir> 24 data/urdu_2e24.bin`).

## Measurement plan (honest hardware notes)

- **Colab (free, T4 runtime)** has only ~2 vCPUs → used for the Linux build, data prep,
  correctness, transpose-cost measurement, and a cuFFT GPU bonus comparison — not for
  scaling curves.
- **OpenMP scaling** runs on a 10-core/16-thread laptop: `bench\run_omp_scaling.ps1`.
- **Multi-machine MPI** runs on 2–3 group laptops over LAN with MS-MPI (every machine
  needs the same input .bin locally).
- All runs append `RESULT,...` lines to `results/*.csv`; `bench/plot_results.py` turns
  them into the time / speedup / efficiency graphs for the report.

## How the four-step FFT works (viva summary)

A length-N FFT with N = N1·N2 is viewed as an N1×N2 matrix:
transpose (MPI_Alltoall) → N2 FFTs of length N1 (OpenMP rows) → twiddle multiply by
W_N^(n2·k1) → transpose back → N1 FFTs of length N2. The output C satisfies
X[k1 + N1·k2] = C[k1][k2]; the checker compares `C.T.ravel()` against `numpy.fft.fft`.
The two transposes are the only communication — their share of runtime is printed on
every run, which is exactly the compute-vs-communication trade-off the report analyses.
