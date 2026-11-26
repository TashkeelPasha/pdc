# Crossover experiment: where does parallelism start paying?
# Runs sequential vs OpenMP-16T at sizes from one voice frame (2^9 = 512 points,
# a 20-32 ms phone-call window) up to 2^24 (17.5 min of speech).
# Appends to results/crossover_laptop.csv ; plot with bench\plot_crossover.py
param([int]$Reps = 5)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
New-Item -ItemType Directory -Force "$root\results" | Out-Null
$csv = "$root\results\crossover_laptop.csv"

foreach ($p in 9, 11, 13, 15, 17, 19, 21, 23, 24) {
    for ($r = 0; $r -lt $Reps; $r++) {
        (& .\build\fft_seq.exe $p) | Where-Object { $_ -like "RESULT*" } | Add-Content $csv
    }
    $env:OMP_NUM_THREADS = "16"
    for ($r = 0; $r -lt $Reps; $r++) {
        (& .\build\fft_omp.exe $p) | Where-Object { $_ -like "RESULT*" } | Add-Content $csv
    }
    Remove-Item Env:\OMP_NUM_THREADS
    Write-Output "done 2^$p"
}
Write-Output "appended -> $csv"
