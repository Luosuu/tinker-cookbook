FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git init /workspace/repo \
    && git -C /workspace/repo remote add origin https://github.com/kokkos/kokkos.git \
    && git -C /workspace/repo fetch --depth=1 origin a0b7eec785ee94633a518cba5ede57403c2cbf9d \
    && git -C /workspace/repo checkout --detach FETCH_HEAD \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && git checkout -q --detach a0b7eec785ee94633a518cba5ede57403c2cbf9d && printf '%s\n' a0b7eec785ee94633a518cba5ede57403c2cbf9d > .git/shallow && git for-each-ref --format='%(refname)' | xargs -r -n1 git update-ref -d && git remote remove origin && git reflog expire --expire=now --all && git gc --prune=now -q \
    && cd /workspace/repo && cmake -S . -B build -G Ninja -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=/workspace/repo/bin/nvcc_wrapper -DKokkos_ARCH_ADA89=ON \
    && cd /workspace/repo && cmake --build build --target Kokkos_CoreUnitTest_CudaInterOpGraph --parallel
WORKDIR /workspace/repo
