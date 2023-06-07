#!/bin/bash -e

docker pull intel/oneapi-hpckit

cd $(dirname $0)
docker run --rm --mount type=bind,source="$(pwd)",target=/app intel/oneapi-hpckit icpx -fsycl /app/main.cpp -o /app/main.out
cp run_main.sh main