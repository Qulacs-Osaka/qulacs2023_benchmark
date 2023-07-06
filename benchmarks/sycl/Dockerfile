FROM nvidia-cuda:ubuntu22.04

RUN apt-get update -y && apt-get install wget gpg -y
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update -y \
    && apt-get install -y intel-basekit intel-hpckit

RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh"
ENV PATH $PATH:/opt/intel/oneapi/compiler/latest/linux/bin
