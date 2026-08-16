FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/kokkos.git /workspace/repo \
    && git -C /workspace/repo checkout 5836b7c2df57227d26777912dd1a5f4b82b18fbd \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && cmake -S . -B build -G Ninja -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cd /workspace/repo && cmake --build build --target Kokkos_CoreUnitTest_Serial1 Kokkos_CoreUnitTest_Serial2 --parallel
WORKDIR /workspace/repo
