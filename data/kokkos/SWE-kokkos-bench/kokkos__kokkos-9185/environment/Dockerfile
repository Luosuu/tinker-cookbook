FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/kokkos.git /workspace/repo \
    && git -C /workspace/repo checkout 9918f68e0ba8f551793b0f6b5ea694363649a530 \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && cmake -S . -B build -G Ninja -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_IMPL_MDSPAN=ON -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cd /workspace/repo && cmake --build build --target Kokkos_CoreUnitTest_Serial_ViewSupport --parallel
WORKDIR /workspace/repo
