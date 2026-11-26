# Multi-machine MPI run (2–3 laptops over Wi-Fi/LAN) — group session runbook

This produces `results/mpi_lan.csv`, the real distributed-memory numbers for the report.
Takes ~30 minutes with everyone in the same room on the same network (hostel/uni Wi-Fi
or one phone hotspot).

## One-time setup on EVERY laptop
1. Install **MS-MPI runtime + SDK** (msmpisetup.exe and msmpisdk.msi from Microsoft's
   MS-MPI page) and mingw-w64 gcc.
2. `git clone https://github.com/TashkeelPasha/pdc && cd pdc && .\build_all.ps1`
   (if gcc can't find `libmsmpi.a`, copy the `mpi-setup` folder from Tashkeel's machine
   and pass `-MpiLib <path>`).
3. Copy the SAME `data\urdu_2e24.bin` to the SAME path on every machine
   (`pdc\data\urdu_2e24.bin`) — MPI-IO reads it locally on each rank.
4. Start the SMPD service **as the same local user with the same password** on each
   machine: `mpiexec -register` (stores credentials), then allow `mpiexec`/`smpd`
   through Windows Firewall (Private network) when prompted — or temporarily:
   `netsh advfirewall firewall add rule name=smpd dir=in action=allow program="C:\Program Files\Microsoft MPI\Bin\smpd.exe"`
5. Find each machine's IP: `ipconfig` (e.g. 192.168.1.10, 192.168.1.11, 192.168.1.12).

## Smoke test (from Tashkeel's laptop)
```powershell
mpiexec -hosts 2 192.168.1.10 1 192.168.1.11 1 hostname
```
Both hostnames printing = the cluster works.

## The measurement
```powershell
cd pdc
# 2 machines, 1 rank each, 8 OpenMP threads per rank:
$env:OMP_NUM_THREADS="8"
foreach ($i in 1..3) { mpiexec -hosts 2 192.168.1.10 1 192.168.1.11 1 `
    .\build\fft_hybrid.exe 24 data\urdu_2e24.bin | Tee-Object -Append results\mpi_lan_raw.txt }
# 3 machines if available:
foreach ($i in 1..3) { mpiexec -hosts 3 192.168.1.10 1 192.168.1.11 1 192.168.1.12 1 `
    .\build\fft_hybrid.exe 24 data\urdu_2e24.bin | Tee-Object -Append results\mpi_lan_raw.txt }
# also rerun the 1-machine baseline the same day for a fair comparison:
foreach ($i in 1..3) { mpiexec -n 1 .\build\fft_hybrid.exe 24 data\urdu_2e24.bin | Tee-Object -Append results\mpi_lan_raw.txt }
```
Then copy the `RESULT,...` lines into `results/mpi_lan.csv`, commit, and run
`python bench\plot_results.py`.

## What to expect (write this in the report, whatever the numbers say)
- The transpose's share of runtime will JUMP versus the single-machine run
  (Wi-Fi moves ~10–30 MB/s; each transpose ships ~N·16·(P-1)/P bytes twice).
- Wall-clock may well be SLOWER than one machine at 2^24 — that is a valid,
  publishable finding: communication-bound scaling. If 2^26 fits in RAM on every
  laptop, try it — the compute/communication ratio improves with N.
- Ethernet or a USB-C cable between two laptops gives a much better curve than Wi-Fi
  if available.

## Troubleshooting
- "Access denied" → `mpiexec -register` again on every machine, same user/password.
- Hangs at startup → firewall on one machine; allow smpd + mpiexec or disable
  Private-profile firewall for the session.
- "cannot open data\urdu_2e24.bin" → the .bin path must exist on EVERY machine.
