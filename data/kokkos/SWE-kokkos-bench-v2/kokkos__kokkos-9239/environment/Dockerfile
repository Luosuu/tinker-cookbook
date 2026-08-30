FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/kokkos.git /workspace/repo \
    && git -C /workspace/repo checkout 42712cb43401bdbfcaad4aee238a21a290d7ef91 \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && git checkout -q --orphan __base && git -c user.email=bench@example.com -c user.name=bench commit -q -m 'base revision' && git for-each-ref --format='%(refname)' | grep -v '^refs/heads/__base$' | xargs -r -n1 git update-ref -d && git remote remove origin && git reflog expire --expire=now --all && git gc --prune=now -q \
    && cd /workspace/repo && cmake -S . -B build -G Ninja -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cd /workspace/repo && cmake --build build --target Kokkos_CoreUnitTest_Serial1 --parallel
WORKDIR /workspace/repo
