# Full laptop benchmark matrix for the report (10C/16T machine).
# Produces results/omp_laptop.csv (seq + OpenMP) and results/mpi_laptop.csv (hybrid).
# Usage: .\bench\run_laptop_matrix.ps1 [-Log2N 24] [-Input data\urdu_2e24.bin] [-Reps 3]
param(
    [int]$Log2N = 24,
    [string]$Input = "data\urdu_2e24.bin",
    [int]$Reps = 3
)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
New-Item -ItemType Directory -Force "$root\results" | Out-Null
if (-not (Test-Path $Input)) { Write-Output "input $Input missing - using synthetic signal"; $Input = "" }

function Invoke-Run([string[]]$cmd, [string]$csv) {
    $out = & $cmd[0] $cmd[1..($cmd.Count - 1)] 2>$null
    $out | Where-Object { $_ -like "RESULT*" } | Add-Content $csv
    Write-Output ($out | Where-Object { $_ -notlike "RESULT*" } | Select-Object -First 1)
}

$ompCsv = "$root\results\omp_laptop.csv"
$mpiCsv = "$root\results\mpi_laptop.csv"

# --- sequential baseline ---
for ($r = 0; $r -lt $Reps; $r++) {
    $c = @(".\build\fft_seq.exe", "$Log2N"); if ($Input) { $c += $Input }
    Invoke-Run $c $ompCsv
}

# --- OpenMP thread sweep (butterfly-parallel) ---
foreach ($t in 1, 2, 4, 8, 10, 12, 16) {
    $env:OMP_NUM_THREADS = "$t"
    for ($r = 0; $r -lt $Reps; $r++) {
        $c = @(".\build\fft_omp.exe", "$Log2N"); if ($Input) { $c += $Input }
        Invoke-Run $c $ompCsv
    }
}

# --- hybrid: pure-MPI sweep, then MPI x OpenMP combos (four-step, row-parallel) ---
$combos = @(@(1, 1), @(2, 1), @(4, 1), @(8, 1), @(1, 8), @(1, 16), @(2, 4), @(2, 8), @(4, 4))
foreach ($pair in $combos) {
    $np = $pair[0]; $t = $pair[1]
    $env:OMP_NUM_THREADS = "$t"
    for ($r = 0; $r -lt $Reps; $r++) {
        $c = @("mpiexec", "-n", "$np", ".\build\fft_hybrid.exe", "$Log2N"); if ($Input) { $c += $Input }
        Invoke-Run $c $mpiCsv
    }
}
Remove-Item Env:\OMP_NUM_THREADS -ErrorAction SilentlyContinue
Write-Output "done -> $ompCsv, $mpiCsv ; graphs: python bench\plot_results.py"
