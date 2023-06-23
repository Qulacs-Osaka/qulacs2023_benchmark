FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    git \
    libboost-dev \
    libpython3-dev \
    python3 \
    python3-pip \
    wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/qulacs/qulacs /qulacs \
    && cd /qulacs \
    && mkdir -p build \
    && ./script/build_gcc_with_gpu.sh

WORKDIR /benchmarks
