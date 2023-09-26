#!/bin/bash -e

cd $(dirname $0)

g++ -O2 -I /qulacs/include -L /qulacs/lib main.cpp -lcppsim_static -lcsim_static -fopenmp -o main