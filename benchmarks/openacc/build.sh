#!/bin/bash -e

docker pull nvcr.io/nvidia/nvhpc:23.5-devel-cuda_multi-ubuntu20.04

cd $(dirname $0)
docker run --gpus all -it --rm --mount type=bind,source="$(pwd)",target=/app nvcr.io/nvidia/nvhpc:23.5-devel-cuda_multi-ubuntu20.04 pgc++ -acc -Minfo=accel main.cpp -o main.out
cp run_main.sh main