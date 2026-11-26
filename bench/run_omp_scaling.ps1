# OpenMP scaling run for the report graphs (run on the 10C/16T laptop).
# Usage: .\bench\run_omp_scaling.ps1 [-Log2N 24] [-Input data\urdu_2e24.bin] [-Reps 3]
param(
    [int]$Log2N = 24,
    [string]$Input = "",
    [int]$Reps = 3
)
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force "$root\results" | Out-Null
$csv = "$root\results\omp_laptop.csv"

# sequential baseline first
for ($r = 0; $r -lt $Reps; $r++) {
    $out = if ($Input) { & "$root\build\fft_seq.exe" $Log2N $Input } else { & "$root\build\fft_seq.exe" $Log2N }
    $out | Where-Object { $_ -like "RESULT*" } | Add-Content $csv
    Write-Output ($out | Select-Object -First 1)
}

foreach ($t in 1, 2, 4, 8, 10, 12, 16) {
    $env:OMP_NUM_THREADS = "$t"
    for ($r = 0; $r -lt $Reps; $r++) {
        $out = if ($Input) { & "$root\build\fft_omp.exe" $Log2N $Input } else { & "$root\build\fft_omp.exe" $Log2N }
        $out | Where-Object { $_ -like "RESULT*" } | Add-Content $csv
        Write-Output ($out | Select-Object -First 1)
    }
}
Remove-Item Env:\OMP_NUM_THREADS
Write-Output "appended to $csv - plot with: python bench\plot_results.py"
