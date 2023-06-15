#!/bin/bash -e

docker build -t kokkos-env .

cd $(dirname $0)
docker run --rm --mount type=bind,source="$(pwd)",target=/app kokkos/kokkos kokkos-compile.sh /app/main.cpp -o /app/main.out
cp run_main.sh main
