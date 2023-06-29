#!/bin/bash -e

cd $(dirname $0)
docker run --rm --mount type=bind,source="$(pwd)",target=/app intel/oneapi-hpckit /app/main.out