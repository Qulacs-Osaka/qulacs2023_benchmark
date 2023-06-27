if [ ! -d /app/kokkos ]; then
    git clone --recursive https://github.com/kokkos/kokkos.git /app/kokkos --depth 1
    git clone --recursive https://github.com/desul/desul.git /app/kokkos/core/src/desul --depth 1
fi

# kokkosのビルドとインストール
mkdir -p /app/kokkos/build && cd /app/kokkos/build
cmake .. -DCMAKE_INSTALL_PREFIX=/app/install
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_OPENMP=ON \
make
make install

# アプリケーションのビルド
mkdir -p /app/build && cd /app/build
cmake .. -DCMAKE_PREFIX_PATH=/app/install
make
cp myTarget ../main.out
