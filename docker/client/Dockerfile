# Builder stage
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS triton-client

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    ninja-build \
    software-properties-common \
    git \
    gcc \
    g++ \
    libboost-all-dev \
    libgl1-mesa-glx \
    # Add some dev tools
    curl \
    wget \
    nano \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# # Install Python 3.10
# RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" > /etc/apt/sources.list.d/deadsnakes.list && \
#     apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
#     apt-get update && \
#     apt-get install -y python3.10 python3.10-venv python3-pip && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Make Python 3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 
# curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# # Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Use build arguments to pass the current user's name and UID
ARG USERNAME
ARG USER_UID
ARG USER_GID

# # Create a non-root user with the provided username and UID
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}

# # Enable colored output in the shell
ENV TERM=xterm-256color

# Copy requirements files
COPY requirements/*.txt /tmp/

ENV VIRTUAL_ENV=/opt/python310-venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

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
    python -c "import torch; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print('Successfully imported CUDA modules')" && \
    ls -l pcdet/ops/iou3d_nms/iou3d_nms_cuda.cpython-310-x86_64-linux-gnu.so

RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> /etc/bash.bashrc
ENV PYTHONPATH=/opt/dependencies/OpenPCDet
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace