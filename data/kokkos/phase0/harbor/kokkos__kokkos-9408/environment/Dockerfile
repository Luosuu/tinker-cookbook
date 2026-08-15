FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/kokkos.git /workspace/repo \
    && cd /workspace/repo \
    && git checkout dcc7879ea2ecadea6b6cdca0c5f815e0ecd0bd77
WORKDIR /workspace/repo
RUN cmake -S . -B build -G Ninja -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_TESTS=ON -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
RUN cmake --build build --target Kokkos_CoreTestCompileOnly Kokkos_CoreUnitTest_Serial_ViewSupport --parallel
