cd /

if [ ! -d ./qulacs ]; then
    git clone https://github.com/qulacs/qulacs.git --depth 1
    cd /qulacs
    USE_GPU=Yes pip install .
    sh /qulacs/script/build_gcc_with_gpu.sh
fi
