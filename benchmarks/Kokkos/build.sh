#!/bin/bash -e

cd $(dirname $0)
docker build -t my_kokkos_image .
docker run --rm --gpus=all -u $(id -u):$(id -g) -ti --mount type=bind,source="$(pwd)",target=/app my_kokkos_image \
    bash -c "/app/build_by_Dockerfile.sh"
