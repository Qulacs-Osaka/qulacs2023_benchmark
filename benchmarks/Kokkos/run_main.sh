#!/bin/bash -e

cd $(dirname $0)
docker run --rm --mount type=bind,source="$(pwd)",target=/app my_kokkos_image /app/main.out
