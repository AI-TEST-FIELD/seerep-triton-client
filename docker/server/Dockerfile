# Builder stage
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS builder

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
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.8 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install PyTorch first
RUN python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
RUN python3 -m pip install numpy==1.23.5 \
    kornia==0.6.8 \
    sharedarray \
    tqdm \
    tensorboardx \
    scikit-image==0.19.3 \
    numba==0.56.4

# Clone and build OpenPCDet
WORKDIR /opt/dependencies
RUN git clone https://github.com/open-mmlab/OpenPCDet.git && \
    cd OpenPCDet && \
    rm -rf build/ && \
    rm -rf pcdet/ops/*.so && \
    rm -rf pcdet/ops/**/*.so && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" python3 setup.py build_ext --inplace && \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" python3 setup.py develop

# Verify CUDA modules are built and loadable
RUN cd /opt/dependencies/OpenPCDet && \
    python3 -c "import torch; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print('Successfully imported CUDA modules')" && \
    ls -l pcdet/ops/iou3d_nms/iou3d_nms_cuda.cpython-38-x86_64-linux-gnu.so

WORKDIR /opt/dependencies/OpenPCDet

# Second stage: Runtime
FROM nvcr.io/nvidia/tritonserver:22.12-py3

# Copy built OpenPCDet from builder
COPY --from=builder /opt/dependencies/OpenPCDet /opt/dependencies/OpenPCDet

# Copy and install requirements
COPY docker/server_3d/requirements.txt /tmp/requirements.txt

# Install dependencies with pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements.txt

RUN apt-get update && apt-get install -y libboost-all-dev libgl1-mesa-glx

# Set Python path
ENV PYTHONPATH=/opt/dependencies/OpenPCDet
