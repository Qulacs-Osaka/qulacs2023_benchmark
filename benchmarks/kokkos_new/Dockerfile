FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
       make git g++ gfortran ccache vim clang llvm wget sudo curl ninja-build clang-format zstd libhwloc-dev libomp-dev gnupg2 cmake && \
    apt-get purge --autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Set up a user with the same UID/GID as the host user to create files for benchmark output.
# Without the following steps, the files created by the container is owned by root and it requires root permission to use the file in Python script.
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} kokkos \
    && useradd -m -u ${USER_ID} -g kokkos -s /bin/bash kokkos

USER kokkos

ENV CCACHE_MAXSIZE=250M

WORKDIR /benchmarks
