# Linux / Colab build. Windows: use build_all.ps1 instead.
CC      = gcc
MPICC   = mpicc
CFLAGS  = -O3 -std=c11 -Wall
LDLIBS  = -lm

all: build/fft_seq build/fft_omp build/fft_hybrid

build:
	mkdir -p build

build/fft_seq: src/fft_seq.c src/fft_kernel.h | build
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

build/fft_omp: src/fft_omp.c src/fft_kernel.h | build
	$(CC) $(CFLAGS) -fopenmp $< -o $@ $(LDLIBS)

build/fft_hybrid: src/fft_hybrid_mpi.c src/fft_kernel.h | build
	$(MPICC) $(CFLAGS) -fopenmp $< -o $@ $(LDLIBS)

check: all
	python3 tools/check_fft.py 16

clean:
	rm -rf build

.PHONY: all check clean
