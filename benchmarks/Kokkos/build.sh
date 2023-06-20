#!/bin/bash -e

cd $(dirname $0)
docker build -t my_kokkos_image .
docker run --rm -u $(id -u):$(id -g) -ti --mount type=bind,source="$(pwd)",target=/app my_kokkos_image \
    bash -c "\
        if [ ! -d /app/include/kokkos ]; then \
            git clone --recursive https://github.com/kokkos/kokkos.git /app/include/kokkos; \
        fi && \
        mkdir -p /app/include/kokkos/build && \
        cd /app/include/kokkos/build && \
        cmake .. -DCMAKE_INSTALL_PREFIX=/app/install && \
        make install && \
        
        mkdir -p /app/build && \
        cd /app/build && \
        cmake .. -DCMAKE_PREFIX_PATH=/app/install && \
        make && \
        cp myTarget ../main.out"
