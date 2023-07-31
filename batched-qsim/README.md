# batched-qsim


## Building (CPU, Apple Silicon)

```bash
git clone git@github.com:keichi/batched-qsim.git
mkdir build
cd build
cmake -DKokkos_ENABLE_OPENMP=ON \
      -DCMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include -Xpreprocessor -fopenmp" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$(brew --prefix libomp)/lib -lomp" \
      -DCMAKE_MODULE_LINKER_FLAGS="-L$(brew --prefix)/lib -lomp" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
make
```

## Building (GPU)

```bash
git clone git@github.com:keichi/batched-qsim.git
mkdir build
cd build
cmake -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ..
make
```
