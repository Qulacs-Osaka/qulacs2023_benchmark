if [ ! -d /app/kokkos ]; then
    git clone --recursive https://github.com/kokkos/kokkos.git /app/kokkos --depth 1
    # kokkosのビルドとインストール
    mkdir -p /app/kokkos/build && cd /app/kokkos/build
    cmake .. -DCMAKE_INSTALL_PREFIX=/app/install \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_CUDA=ON
    make
    make install
fi

# アプリケーションのビルド
mkdir -p /app/build && cd /app/build
cmake .. -DCMAKE_PREFIX_PATH=/app/install
make
cp myTarget ../main
