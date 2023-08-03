#!/bin/bash -e

cd $(dirname $0)
docker run --rm --gpus=all --mount type=bind,source="$(pwd)",target=/app my_kokkos_image \
    /app/main 4 24 10
