FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    libboost-dev \
    libpython3-dev \
    make \
    python3 \
    python3-pip \
    wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists*

RUN git clone https://github.com/qulacs/qulacs --depth 1 /qulacs \
    && cd /qulacs
COPY build_gcc.sh /qulacs/script/build_gcc.sh
RUN sh /qulacs/script/build_gcc.sh

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} vscode \
    && useradd -m -u ${USER_ID} -g vscode -s /bin/bash vscode

USER vscode

WORKDIR /benchmarks
