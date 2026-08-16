FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/kokkos-kernels.git /workspace/repo \
    && git -C /workspace/repo checkout 90a08d70c5b35be697e9b0da703e8e92e52c5718 \
    && git -C /workspace/repo submodule update --init --recursive \
    && git clone --branch 5.2.0 --depth 1 https://github.com/kokkos/kokkos.git /workspace/kokkos \
    && cmake -S /workspace/kokkos -B /workspace/kokkos-build -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/kokkos -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /workspace/kokkos-build --parallel \
    && cmake --install /workspace/kokkos-build \
    && cd /workspace/repo && cmake -S . -B build -G Ninja -DKokkosKernels_ENABLE_TESTS=ON -DKokkos_ROOT=/opt/kokkos -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cd /workspace/repo && cmake --build build --target KokkosKernels_batched_dla_openmp --parallel
WORKDIR /workspace/repo
