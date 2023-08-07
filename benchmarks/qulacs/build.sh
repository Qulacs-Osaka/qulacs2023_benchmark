set -eu

g++ -D _USE_GPU -O3 -I /qulacs/include -L /usr/local/cuda/lib64 -L /qulacs/lib main.cpp -o main -fopenmp -lcppsim_static -lcsim_static -lgpusim_static -lcudart -lcublas
