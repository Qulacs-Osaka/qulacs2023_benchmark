#!/bin/bash -e

cd $(dirname $0)
docker build -t my_kokkos_image .
docker run --rm -u 0 -ti --mount type=bind,source="$(pwd)",target=/app my_kokkos_image \
    bash -c "\
        if [ ! -d /app/include/kokkos ]; then \
            git clone --recursive https://github.com/kokkos/kokkos.git /app/include/kokkos; \
        fi && \
        mkdir -p /app/build && \
        cd /app/build && \
        cmake .. && \
        make && \
        cp myTarget ../main.out"
        