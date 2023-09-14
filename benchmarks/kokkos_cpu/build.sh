#!/bin/bash -e

cd $(dirname $0)

if [ ! -d /benchmarks/kokkos ]; then
    git clone --recursive https://github.com/kokkos/kokkos.git /benchmarks/kokkos --depth 1
    # kokkosのビルドとインストール
    mkdir -p /benchmarks/kokkos/build && cd /benchmarks/kokkos/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/benchmarks/install \
        -DKokkos_ENABLE_SERIAL=ON
    make
    make install
fi

# アプリケーションのビルド
mkdir -p /benchmarks/build && cd /benchmarks/build
cmake .. -DCMAKE_PREFIX_PATH=/benchmarks/install
make

cp myTarget /benchmarks/main

