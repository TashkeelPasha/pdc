"""Plot the crossover experiment: OpenMP-16T speedup over sequential vs problem size.

Reads results/crossover_laptop.csv (RESULT lines from run_crossover.ps1) and writes
results/crossover.png. The vertical marker at 2^9 is one 512-point voice frame
(20-32 ms of phone audio) — the sizes real-time speech pipelines use.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
csv = ROOT / "results" / "crossover_laptop.csv"

best = defaultdict(lambda: float("inf"))        # (ver, p) -> best seconds
for line in csv.read_text().splitlines():
    if line.startswith("RESULT,"):
        q = line.split(",")
        best[(q[1], int(q[2]))] = min(best[(q[1], int(q[2]))], float(q[5]))

ps = sorted({p for (_, p) in best})
speedup = [best[("seq", p)] / best[("omp", p)] for p in ps]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ps, speedup, marker="o", color="tab:blue", label="seq time / OpenMP-16T time")
ax.axhline(1.0, color="k", ls="--", alpha=0.5, label="break-even")
ax.axvline(9, color="tab:red", ls=":", alpha=0.8)
ax.annotate("one 512-pt voice frame\n(20-32 ms of a call)", xy=(9, 1.0),
            xytext=(10, max(speedup) * 0.5), color="tab:red", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:red", alpha=0.7))
ax.set_xlabel("log2(N)  — signal size")
ax.set_ylabel("speedup of 16 threads over sequential")
ax.set_title("Where parallelism starts to pay (10C/16T laptop)")
ax.grid(alpha=0.3)
ax.legend()
out = ROOT / "results" / "crossover.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
