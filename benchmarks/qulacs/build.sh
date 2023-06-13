set -eu

cd qulacs/

./script/build_gcc_with_gpu.sh

cd -

g++ -O3 -I ./qulacs/include -L ./qulacs/lib main.cpp -o main -fopenmp -lcppsim_static -lcsim_static
