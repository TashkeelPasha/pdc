"""Multiply the corpus the way ASR training pipelines do: 3-way speed perturbation.

Each source clip is resampled to 0.9x and 1.1x speed (the Kaldi-standard triple,
with the 1.0x original kept in place), tripling the hours of distinct audio.

Usage: python tools/augment_corpus.py <src_dir> <dst_dir>
Writes <dst_dir>/sp09_<name>.wav and <dst_dir>/sp11_<name>.wav (16 kHz PCM16 mono).
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

n, secs = 0, 0.0
for f in sorted(src.rglob("*.wav")):
    try:
        x, sr = sf.read(f, dtype="float64")
    except Exception:
        continue
    if sr != 16000 or x.ndim != 1:
        continue
    # speed s = resample by 1/s then play at 16 kHz (shifts pitch too — standard SP)
    for tag, up, down in (("sp09", 10, 9), ("sp11", 10, 11)):
        y = resample_poly(x, up, down)
        sf.write(dst / f"{tag}_{f.name}", np.clip(y, -1, 1), 16000, subtype="PCM_16")
        secs += len(y) / 16000.0
        n += 1
print(f"wrote {n} augmented clips ({secs / 3600.0:.2f} h) -> {dst}")
