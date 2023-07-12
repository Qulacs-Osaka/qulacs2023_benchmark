#!/bin/bash -e

cd $(dirname $0)
icpx -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O2 main.cpp -o main
