FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git libboost-all-dev libhdf5-dev libopenmpi-dev \
    ninja-build openmpi-bin python-is-python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kokkos/pykokkos.git /workspace/repo \
    && git -C /workspace/repo checkout 933cbcb26b6b56bf1c027ce137afc1c54ba73dcd \
    && git -C /workspace/repo submodule update --init --recursive \
    && cd /workspace/repo && git checkout -q --orphan __base && git -c user.email=bench@example.com -c user.name=bench commit -q -m 'base revision' && git for-each-ref --format='%(refname)' | grep -v '^refs/heads/__base$' | xargs -r -n1 git update-ref -d && git remote remove origin && git reflog expire --expire=now --all && git gc --prune=now -q \
    && python -m pip install --break-system-packages numpy patchelf pybind11 pytest setuptools wheel \
    && cd /workspace/repo && PIP_BREAK_SYSTEM_PACKAGES=1 python install_base.py install -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=4 -DENABLE_CUDA=OFF -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON \
    && cd /workspace/repo && python -m pip install --break-system-packages -e . --no-deps \
    && cd /workspace/repo && python -m compileall -q pykokkos/interface/views.py tests/test_typeinference.py
WORKDIR /workspace/repo
