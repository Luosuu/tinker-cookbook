FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/kokkos.git /workspace/repo \
    && git -C /workspace/repo checkout 0d40eccc4c79d463279045fca1624873a682795e \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && cmake -S . -B build -G Ninja -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF -DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cd /workspace/repo && cmake --build build --target Kokkos_CoreUnitTest_Serial2 --parallel
WORKDIR /workspace/repo
