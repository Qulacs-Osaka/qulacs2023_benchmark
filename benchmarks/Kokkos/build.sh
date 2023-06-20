#!/bin/bash -e

docker pull quay.io/cbgeo/kokkos

cd $(dirname $0)
docker run --rm --mount type=bind,source="$(pwd)",target=/app quay.io/cbgeo/kokkos \
    bash -c "\
        if [ ! -d /app/include/kokkos ]; then \
            git clone --recursive https://github.com/kokkos/kokkos.git /app/include/kokkos; \
        fi && \
        mkdir -p /app/build && \
        cd /app/build && \
        cmake .. \
        -DKokkos_ROOT=/app/include/kokkos && \
        make && \
        cp myTarget ../main.out"
        
