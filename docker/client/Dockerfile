# Builder stage
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS triton-client

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    ninja-build \
    git \
    gcc \
    g++ \
    libboost-all-dev \
    libgl1-mesa-glx \
    # Add some dev tools
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.8 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Use build arguments to pass the current user's name and UID
ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create a non-root user with the provided username and UID
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# Enable colored output in the shell
ENV TERM=xterm-256color

# Copy requirements files
COPY requirements/*.txt /tmp/

# Install PyTorch first
RUN python3 -m pip install -r /tmp/torch-requirements.txt
# Install other dependencies
RUN python3 -m pip install -r /tmp/requirements.txt

# Clone and build OpenPCDet
WORKDIR /opt/dependencies
RUN git clone https://github.com/open-mmlab/OpenPCDet.git && \
    cd OpenPCDet && \
    rm -rf build/ && \
    rm -rf pcdet/ops/*.so && \
    rm -rf pcdet/ops/**/*.so && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" python3 setup.py build_ext --inplace && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" python3 setup.py develop

# # Verify CUDA modules are built and loadable
RUN cd /opt/dependencies/OpenPCDet && \
    python3 -c "import torch; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print('Successfully imported CUDA modules')" && \
    ls -l pcdet/ops/iou3d_nms/iou3d_nms_cuda.cpython-38-x86_64-linux-gnu.so

WORKDIR /workspace