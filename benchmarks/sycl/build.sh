#!/bin/bash -e

source /opt/intel/oneapi/setvars.sh
cd $(dirname $0)
icpx -fsycl -O2 main.cpp -o main
