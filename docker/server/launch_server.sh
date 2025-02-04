#! /bin/bash

# Prompt user for model repository path
echo "Please enter the path to your model repository:"
read MODEL_REPO_PATH

# Build the Docker image
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/server/Dockerfile --platform linux/amd64 -t niqbal996/triton-server:22.12-py3-PCDet .

# Run the container with the user-provided model repository path
docker run -it --rm --runtime=nvidia --net=host --name \
triton-server-3D --ipc=host -e PYTHONPATH=/opt/dependencies/OpenPCDet \
-v"${MODEL_REPO_PATH}":/opt/model_repo \
niqbal996/triton-server:22.12-py3-PCDet tritonserver --model-repository=/opt/model_repo