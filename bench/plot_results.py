"""Turn RESULT csv lines from results/*.csv into the report graphs.

Line formats (the executables print these):
  RESULT,seq,<log2N>,1,1,<seconds>
  RESULT,omp,<log2N>,1,<threads>,<seconds>
  RESULT,mpi,<log2N>,<procs>,<threads>,<seconds>,<comm_seconds>

Produces results/time_<log2N>.png, speedup_<log2N>.png, efficiency_<log2N>.png
(one set per problem size found). Speedup is measured against the sequential
baseline at the same N; workers = procs x threads.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

rows = []
for f in RES.glob("*.csv"):
    for line in f.read_text().splitlines():
        if line.startswith("RESULT,"):
            q = line.split(",")
            rows.append({"ver": q[1], "p": int(q[2]), "procs": int(q[3]),
                         "threads": int(q[4]), "sec": float(q[5]),
                         "comm": float(q[6]) if len(q) > 6 else 0.0})
if not rows:
    raise SystemExit(f"no RESULT lines found under {RES}")

for p in sorted({r["p"] for r in rows}):
    sub = [r for r in rows if r["p"] == p]
    best = defaultdict(lambda: float("inf"))        # (ver, workers) -> best seconds
    for r in sub:
        key = (r["ver"], r["procs"] * r["threads"])
        best[key] = min(best[key], r["sec"])
    seq = best.get(("seq", 1))
    series = defaultdict(list)                       # ver -> [(workers, sec)]
    for (ver, w), sec in sorted(best.items()):
        if ver != "seq":
            series[ver].append((w, sec))

    for kind in ("time", "speedup", "efficiency"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for ver, pts in series.items():
            xs = [w for w, _ in pts]
            if kind == "time":
                ys = [s for _, s in pts]
            elif seq:
                ys = [seq / s for _, s in pts]
                if kind == "efficiency":
                    ys = [y / w for y, w in zip(ys, xs)]
            else:
                continue
            ax.plot(xs, ys, marker="o", label=ver)
        if kind == "speedup" and series:
            lim = max(w for pts in series.values() for w, _ in pts)
            ax.plot([1, lim], [1, lim], "k--", alpha=0.4, label="ideal")
        if kind == "time" and seq:
            ax.axhline(seq, color="k", ls=":", alpha=0.6, label="sequential")
        ax.set_xlabel("workers (processes x threads)")
        ax.set_ylabel({"time": "seconds", "speedup": "speedup", "efficiency": "efficiency"}[kind])
        ax.set_title(f"N = 2^{p} ({(1 << p) / 16000 / 60:.0f} min of 16 kHz audio)")
        ax.grid(alpha=0.3)
        ax.legend()
        out = RES / f"{kind}_{p}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)
