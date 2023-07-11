#!/bin/bash -e

cd $(dirname $0)
icpx -fsycl -O2 main.cpp -o main
