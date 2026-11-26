"""Correctness check: run each FFT version on random input and compare with numpy.

Usage: python tools/check_fft.py [log2N] [--mpi-procs P]
Expects the executables in build/ (fft_seq, fft_omp, fft_hybrid; .exe added on Windows).
The hybrid version writes C[k1][k2] row-major where X[k1 + N1*k2] = C[k1][k2],
so its output is reordered with a transpose before comparing.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
EXE = ".exe" if os.name == "nt" else ""
TOL = 1e-9


def run(cmd):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def compare(name, ours, ref):
    err = np.max(np.abs(ours - ref)) / np.max(np.abs(ref))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  {name}: max relative error vs numpy = {err:.2e} -> {status}")
    return status == "PASS"


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 16
    procs = int(sys.argv[sys.argv.index("--mpi-procs") + 1]) if "--mpi-procs" in sys.argv else 2
    n = 1 << p
    n1, n2 = 1 << ((p + 1) // 2), 1 << (p // 2)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
    BUILD.mkdir(exist_ok=True)
    inp = BUILD / "check_in.bin"
    x.tofile(inp)
    ref = np.fft.fft(x)
    ok = True

    for name in ("fft_seq", "fft_omp"):
        out = BUILD / f"check_{name}.bin"
        run([str(BUILD / (name + EXE)), str(p), str(inp), str(out)])
        ok &= compare(name, np.fromfile(out, dtype=np.complex128), ref)

    out = BUILD / "check_mpi.bin"
    mpi = ["mpiexec", "-n", str(procs)]
    if os.name != "nt":  # Colab runs as root; allow oversubscribing its 2 vCPUs
        mpi += ["--allow-run-as-root", "--oversubscribe"]
    run(mpi + [str(BUILD / ("fft_hybrid" + EXE)), str(p), str(inp), str(out)])
    c = np.fromfile(out, dtype=np.complex128).reshape(n1, n2)
    ok &= compare(f"fft_hybrid (P={procs})", c.T.ravel(), ref)

    print("ALL PASS" if ok else "FAILURES — fix before benchmarking")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
