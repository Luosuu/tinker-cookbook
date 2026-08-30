FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/kokkos-kernels.git /workspace/repo \
    && git -C /workspace/repo checkout 6380905f3f938bffa33777adef2f3435bf590b09 \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && git checkout -q --orphan __base && git -c user.email=bench@example.com -c user.name=bench commit -q -m 'base revision' && git for-each-ref --format='%(refname)' | grep -v '^refs/heads/__base$' | xargs -r -n1 git update-ref -d && git remote remove origin && git reflog expire --expire=now --all && git gc --prune=now -q \
    && git clone --branch 5.2.0 --depth 1 https://github.com/kokkos/kokkos.git /workspace/kokkos \
    && cmake -S /workspace/kokkos -B /workspace/kokkos-build -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/kokkos -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /workspace/kokkos-build --parallel \
    && cmake --install /workspace/kokkos-build \
    && cd /workspace/repo && cmake -S . -B build -G Ninja -DKokkosKernels_ENABLE_TESTS=ON -DKokkos_ROOT=/opt/kokkos -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cd /workspace/repo && cmake --build build --target KokkosKernels_batched_dla_serial --parallel
WORKDIR /workspace/repo
