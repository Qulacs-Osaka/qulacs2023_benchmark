ARG TAG=latest
FROM ubuntu:${TAG}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
       make git g++ gfortran ccache vim clang llvm wget sudo curl ninja-build clang-format zstd libhwloc-dev libomp-dev gnupg2 cmake && \
    apt-get purge --autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -G sudo -u 1001 kokkos
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER kokkos
ENV CCACHE_MAXSIZE=250M
WORKDIR /home/kokkos
