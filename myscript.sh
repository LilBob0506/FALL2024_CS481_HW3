#!/bin/bash
cd FALL2024_CS481_HW3
module load intel
icx -O -o -fopenmp gameoflife_threaded1.c gameoflife1.c
./gameoflife <size> <max number of generations> <number of threads> /scratch/ualclsd0193
