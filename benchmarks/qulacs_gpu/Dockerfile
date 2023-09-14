FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

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

RUN git clone https://github.com/qulacs/qulacs --depth 1 /qulacs \
    && cd /qulacs \
    && USE_GPU=Yes pip install . \
    && sh /qulacs/script/build_gcc_with_gpu.sh

# Set up a user with the same UID/GID as the host user to create files for benchmark output.
# Without the following steps, the files created by the container is owned by root and it requires root permission to use the file in Python script.
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} qulacs_gpu \
    && useradd -m -u ${USER_ID} -g qulacs_gpu -s /bin/bash qulacs_gpu

USER qulacs_gpu

WORKDIR /benchmarks
