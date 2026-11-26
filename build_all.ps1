# Windows build (mingw-w64 gcc + MS-MPI). Linux/Colab: use `make` instead.
# MS-MPI ships MSVC-format .lib files, so gcc links against a pre-generated
# import library (libmsmpi.a) — pass its folder via -MpiLib if not the default.
param(
    [string]$MpiInc = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include",
    [string]$MpiLib = "$PSScriptRoot\..\mpi-setup"
)
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = $machinePath + ";" + $userPath

New-Item -ItemType Directory -Force "$PSScriptRoot\build" | Out-Null
$ok = $true

gcc -O3 -std=c11 -Wall "$PSScriptRoot\src\fft_seq.c" -o "$PSScriptRoot\build\fft_seq.exe" -lm
if ($LASTEXITCODE -ne 0) { $ok = $false }

gcc -O3 -std=c11 -Wall -fopenmp "$PSScriptRoot\src\fft_omp.c" -o "$PSScriptRoot\build\fft_omp.exe" -lm
if ($LASTEXITCODE -ne 0) { $ok = $false }

gcc -O3 -std=c11 -Wall -fopenmp "$PSScriptRoot\src\fft_hybrid_mpi.c" -I "$MpiInc" -L "$MpiLib" -lmsmpi -o "$PSScriptRoot\build\fft_hybrid.exe" -lm
if ($LASTEXITCODE -ne 0) { $ok = $false }

if ($ok) {
    Write-Output "Built build\fft_seq.exe, build\fft_omp.exe, build\fft_hybrid.exe"
    Write-Output "Verify:  python tools\check_fft.py 16"
    Write-Output "Run:     mpiexec -n 4 .\build\fft_hybrid.exe 24"
} else {
    Write-Output "BUILD FAILED - see errors above."
}
