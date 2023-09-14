#!/bin/bash -e

cd $(dirname $0)

nvcc -O2 -I /qulacs/include -L /qulacs/lib main.cu -lvqcsim_static -lcppsim_static -lcsim_static -lgpusim_static -D _USE_GPU -lcublas -Xcompiler -fopenmp -o main
