"""Build the big input signal for the FFT from real Urdu speech.

Walks a folder of audio files (wav/flac/mp3 — YODAS extracts, Common Voice clips),
decodes each to mono 16 kHz, concatenates until 2^log2N samples, and writes them
as complex doubles (imaginary part 0) ready for fft_seq / fft_omp / fft_hybrid.

Usage: python tools/prepare_signal.py <audio_dir> <log2N> <output.bin>
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    audio_dir, p, out_path = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    n_target = 1 << p

    files = sorted(f for f in audio_dir.rglob("*") if f.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no audio files under {audio_dir}")

    chunks, n_have, n_files = [], 0, 0
    for f in files:
        try:
            data, sr = sf.read(f, dtype="float64", always_2d=True)
        except Exception as e:  # skip unreadable clips, keep going
            print(f"  skip {f.name}: {e}")
            continue
        mono = data.mean(axis=1)
        if sr != TARGET_SR:
            mono = resample_poly(mono, TARGET_SR, sr)
        chunks.append(mono)
        n_have += len(mono)
        n_files += 1
        if n_have >= n_target:
            break

    if n_have < n_target:
        sys.exit(f"only {n_have:,} samples ({n_have / TARGET_SR / 3600:.2f} h) from "
                 f"{n_files} files — need {n_target:,} (2^{p}). Add more audio.")

    signal = np.concatenate(chunks)[:n_target]
    signal /= max(np.max(np.abs(signal)), 1e-9)          # normalise to [-1, 1]
    out = np.zeros(n_target, dtype=np.complex128)
    out.real = signal
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(out_path)
    print(f"wrote {out_path}: 2^{p} = {n_target:,} samples "
          f"({n_target / TARGET_SR / 60:.1f} min of speech from {n_files} files, "
          f"{out_path.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
