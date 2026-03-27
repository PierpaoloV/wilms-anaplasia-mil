# ============================================================================
# Unified Pathology Pipeline — CUDA 12.4 | Python 3.10 | PyTorch 2.4.1
# Includes: pathology-segmentation-pipeline + slide2vec + CellViT++
# Target:   pierpaolov93/pathology-pipeline:cellvit
#
# Source: https://github.com/PierpaoloV/pathology-segmentation-pipeline
# The pre-built image is available at:
#   docker pull pierpaolov93/pathology-pipeline:cellvit
# Building from source requires cloning the pathology-segmentation-pipeline
# repository alongside this one (pathology-common, pathology-fast-inference,
# and code/ directories are referenced by the COPY instructions below).
#
# Multi-stage build:
#   build   — devel image: compile flash-attn + libjpeg-turbo 3.x + all pip
#   runtime — lean runtime image: ASAP install + copy from build
#
# Version reconciliation across all three tools:
#   numpy         1.24.4  (<1.25 for numba, <2 for slide2vec)
#   pandas        1.5.3   (CellViT++ targets 1.4.3 API; slide2vec unversioned)
#   albumentations 1.3.1  (satisfies CellViT++ 1.3.0 and slide2vec 1.3.0;
#                          minor regression from our 1.4.x — avoid 2.x still holds)
#   tensorflow-cpu 2.12.0 (sidesteps CUDA 11.8 vs 12.4 incompatibility;
#                          CellViT++ is PyTorch-based, TF used for utils only)
# ============================================================================


# -----------------------------------------------------------------------
# Stage 1: build
# Requires CUDA devel headers to compile flash-attn and xformers.
# All Python packages are installed here and copied to the runtime stage.
# -----------------------------------------------------------------------
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS build

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-dev python-is-python3 \
        curl git cmake build-essential \
        zlib1g-dev libnuma1 libtiff-dev \
        libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# libjpeg-turbo 3.1.0
# Ubuntu 22.04 ships 2.x; slide2vec's PyTurboJPEG>=2 requires 3.x.
# Compiled from source and installed to /usr/local.
# -----------------------------------------------------------------------
ARG LIBJPEG_TURBO_VERSION=3.1.0
RUN curl -fsSL \
        https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz \
        | tar xz -C /tmp && \
    cd /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION} && \
    cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local . && \
    make -j"$(nproc)" && make install && ldconfig && \
    rm -rf /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}

RUN python3 -m pip install --upgrade pip setuptools

# -----------------------------------------------------------------------
# PyTorch — must be installed before flash-attn and xformers (they
# compile against the installed torch + CUDA devel headers).
# torchaudio included: CellViT++ check_environment.py validates it.
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1

# -----------------------------------------------------------------------
# flash-attn (slide2vec) — compiled against PyTorch + CUDA devel headers
# psutil required by flash-attn's build system
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir psutil && \
    python3 -m pip install --no-cache-dir \
        'flash-attn>=2.7.1,<=2.8.0' --no-build-isolation

# -----------------------------------------------------------------------
# Core numerical / data stack
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        numpy==1.24.4 \
        pandas==1.5.3 \
        scipy==1.11.4 \
        scikit-learn==1.3.2 \
        scikit-image==0.21.0 \
        matplotlib==3.7.3

# -----------------------------------------------------------------------
# Our pathology-segmentation-pipeline packages
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        shapely==2.0.6 \
        albumentations==1.3.1 \
        segmentation-models-pytorch==0.3.4 \
        rdp==0.8 \
        seaborn \
        wholeslidedata==0.0.15 \
        opencv-python-headless==4.8.1.78 \
        rich==13.7.1 \
        wandb==0.17.9 \
        huggingface_hub==0.24.6 \
        jupyterlab==4.2.5 \
        httpx==0.27.2 \
        openpyxl==3.1.5 \
        h5py==3.9.0

# -----------------------------------------------------------------------
# slide2vec packages
# xformers compiled in this stage against PyTorch + CUDA devel headers.
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        'hs2p>=2.5.1,<3' \
        'omegaconf>=2.3.0' \
        PyTurboJPEG \
        transformers \
        einops \
        einops-exts \
        sacremoses \
        timm==1.0.8 \
        xformers \
        slide2vec

# -----------------------------------------------------------------------
# CellViT++ — heavy / GPU packages
# tensorflow-cpu: avoids CUDA 11.8 vs 12.4 conflict; CellViT++ is
#   PyTorch-based and only uses TF for utility operations.
# cupy-cuda12x / cucim-cu12: pip-installable RAPIDS packages for CUDA 12.x.
#   CellViT++ check_environment.py hard-requires both with live GPU access.
# numba: requires numpy<1.25 — satisfied by numpy==1.24.4 above.
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        tensorflow-cpu==2.12.0 \
        keras==2.12.0 \
        numba==0.59.0 \
        cupy-cuda12x \
        cucim-cu12

# -----------------------------------------------------------------------
# CellViT++ — WSI / geospatial / analysis packages
# -----------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
        openslide-python==1.3.1 \
        pyvips==2.2.3 \
        rasterio==1.3.5.post1 \
        ray==2.9.3 \
        xgboost==2.1.1 \
        torchmetrics==0.11.4 \
        torchstain==1.3.0 \
        torchinfo==1.8.0 \
        pathopatch==1.0.2 \
        geojson==3.0.1 \
        evalutils==0.4.2 \
        natsort==8.4.0 \
        pyaml==24.7.0 \
        schema==0.7.5 \
        pydantic==1.10.4 \
        pydantic-compat==0.1.2 \
        pydantic-core==2.20.1 \
        simpleitk==2.2.1 \
        colour==0.1.5 \
        tabulate==0.9.0 \
        ujson==5.8.0 \
        wsidicom==0.20.4 \
        wsidicomizer==0.14.1 \
        pandarallel==1.6.5 \
        scikit-base==0.7.8 \
        cachetools==5.3.3 \
        colorama==0.4.6 \
        future==0.18.2 \
        termcolor==2.4.0 \
        opt-einsum==3.3.0 \
        flatbuffers==24.3.7 \
        pydicom==2.4.4 \
        python-snappy==0.7.3 \
        pyjwt==2.6.0 \
        tqdm==4.65.0


# -----------------------------------------------------------------------
# Stage 2: runtime
# Lean runtime image. ASAP is installed here (needs apt to resolve its
# own .deb dependencies). All Python packages are copied from build.
# -----------------------------------------------------------------------
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Amsterdam

# -----------------------------------------------------------------------
# System runtime libraries
# -----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev python-is-python3 \
        curl git openssh-server sudo pv vim screen zip unzip \
        libgl1 libgomp1 \
        libopenslide0 libtiff5 libjpeg-turbo8 libtiff-dev \
        libqt5concurrent5 libqt5core5a libqt5gui5 libqt5widgets5 \
        libboost-filesystem1.74.0 libboost-regex1.74.0 \
        libboost-thread1.74.0 libboost-iostreams1.74.0 \
        libglib2.0-0 libgsf-1-114 libexif12 librsvg2-2 libfftw3-3 \
        libopenjp2-7-dev libnuma1 \
        libvips-dev \
        libsnappy-dev \
    && mkdir /var/run/sshd \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# libjpeg-turbo 3.x shared libs (compiled in build stage)
# -----------------------------------------------------------------------
COPY --from=build /usr/local/lib/libjpeg* /usr/local/lib/
COPY --from=build /usr/local/lib/libturbojpeg* /usr/local/lib/
RUN ldconfig

# -----------------------------------------------------------------------
# ASAP 2.2 Nightly (Ubuntu 22.04, Python 3.10 bindings)
# Installed here so apt resolves its own runtime deps cleanly.
# -----------------------------------------------------------------------
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN apt-get update && \
    curl -L "${ASAP_URL}" -o /tmp/ASAP.deb && \
    apt-get install --assume-yes /tmp/ASAP.deb && \
    SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    rm /tmp/ASAP.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# Python packages + CLI entry points from build stage
# -----------------------------------------------------------------------
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

# -----------------------------------------------------------------------
# Register nvimgcodec so cucim can use GPU-accelerated JPEG decoding
# -----------------------------------------------------------------------
RUN SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    printf "${SITE_PACKAGES}/nvidia/nvimgcodec\n" > /etc/ld.so.conf.d/nvimgcodec.conf && \
    ldconfig

# -----------------------------------------------------------------------
# Non-root user
# -----------------------------------------------------------------------
RUN useradd -m -s /bin/bash user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R user:user /home/user/

# -----------------------------------------------------------------------
# Source code — our pipeline is baked in.
# CellViT++ and slide2vec source should be mounted at runtime:
#   -v /path/to/CellViT-Plus-Plus:/home/user/CellViT-Plus-Plus
#   -v /path/to/slide2vec:/home/user/slide2vec
# -----------------------------------------------------------------------
COPY --chown=user:user pathology-common         /home/user/source/pathology-common
COPY --chown=user:user pathology-fast-inference /home/user/source/pathology-fast-inference
COPY --chown=user:user code                     /home/user/source/code
COPY --chown=user:user download_models.py       /home/user/source/download_models.py
COPY --chown=user:user execute.sh               /home/user/execute.sh

RUN mkdir -p /home/user/source/models && \
    chown -R user:user /home/user/source/models

# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------
ENV PYTHONPATH="/home/user/source/pathology-common:/home/user/source/pathology-fast-inference" \
    MPLBACKEND="Agg"

STOPSIGNAL SIGINT
EXPOSE 22 6006 8888

USER user
WORKDIR /home/user

ENTRYPOINT ["/home/user/execute.sh"]
