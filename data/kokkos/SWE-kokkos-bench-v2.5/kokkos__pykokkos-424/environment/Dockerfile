FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git init /workspace/repo \
    && git -C /workspace/repo remote add origin https://github.com/kokkos/pykokkos.git \
    && git -C /workspace/repo fetch --depth=1 origin 0e451c786c134ae4f3c5f785ffcc736b5f686a00 \
    && git -C /workspace/repo checkout --detach FETCH_HEAD \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && git checkout -q --detach 0e451c786c134ae4f3c5f785ffcc736b5f686a00 && printf '%s\n' 0e451c786c134ae4f3c5f785ffcc736b5f686a00 > .git/shallow && git for-each-ref --format='%(refname)' | xargs -r -n1 git update-ref -d && git remote remove origin && git reflog expire --expire=now --all && git gc --prune=now -q \
    && python -m pip install --break-system-packages numpy patchelf pybind11 pytest setuptools wheel \
    && cd /workspace/repo && PIP_BREAK_SYSTEM_PACKAGES=1 python install_base.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=4 -DENABLE_CUDA=OFF -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON \
    && cd /workspace/repo && python -m pip install --break-system-packages -e . --no-deps \
    && cd /workspace/repo && python -m compileall -q pykokkos/interface/parallel_dispatch.py
WORKDIR /workspace/repo
