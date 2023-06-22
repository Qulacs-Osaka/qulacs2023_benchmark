#!/bin/bash -e

cd $(dirname $0)
docker run --rm --mount type=bind,source="$(pwd)",target=/app nvcr.io/nvidia/nvhpc:23.5-devel-cuda_multi-ubuntu20.04 /app/main.out