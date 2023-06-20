#!/bin/bash -e

cd $(dirname $0)
docker run --rm --mount type=bind,source="$(pwd)",target=/app quay.io/cbgeo/kokkos /app/main.out
