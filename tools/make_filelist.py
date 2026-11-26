"""Write a filelist of .wav paths for the spectrogram engine.

Usage: python tools/make_filelist.py <audio_dir> <filelist.txt>
"""
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(__doc__)
files = sorted(Path(sys.argv[1]).rglob("*.wav"))
Path(sys.argv[2]).write_text("\n".join(str(f) for f in files) + "\n", encoding="utf-8")
print(f"{len(files)} wav files -> {sys.argv[2]}")
