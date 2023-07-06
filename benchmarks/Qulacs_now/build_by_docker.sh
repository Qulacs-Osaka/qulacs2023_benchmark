cd $(dirname $0)

if [ ! -d qulacs ]; then
    git clone https://github.com/qulacs/qulacs.git --depth 1
    cd qulacs
    USE_GPU=Yes pip install .
    cd ../
    sh qulacs/script/build_gcc_with_gpu.sh
fi

nvcc -O2 -I ./qulacs/include -L ./qulacs/lib main.cu -lvqcsim_static -lcppsim_static -lcsim_static -lgpusim_static -D _USE_GPU -lcublas -Xcompiler -fopenmp -o home/main

