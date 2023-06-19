#!/bin/bash -e

# DockerでLAMMPSのイメージをpull
docker pull nvcr.io/hpc/lammps:patch_3Nov2022

cd $(dirname $0)
docker run --rm --gpus all --mount type=bind,source="$(pwd)",target=/app nvcr.io/hpc/lammps:patch_3Nov2022 \
    g++ /app/main.cpp -o /app/main.out
cp run_main.sh main
