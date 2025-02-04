# Triton server with Det3D

This Dockerfile builds a Triton server image with OpenPCDet and spconv to run 3D detectors with python backend.

## Usage

```bash
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/server/Dockerfile --platform linux/amd64 -t niqbal996/triton-server:22.12-py3-PCDet .

docker run -it --rm --runtime=nvidia --net=host --name \
triton-server-3D --shm-size=256m -e PYTHONPATH=/opt/dependencies/OpenPCDet \
-v/home/niqbal/model_repository/backup_repo:/opt/model_repo \
niqbal996/triton-server:22.12-py3-PCDet tritonserver --model-repository=/opt/model_repo
```

# Dependencies 

 Nvidia container runtime for docker:
 ```bash
 distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
 sudo apt-get update \
    && sudo apt-get install -y nvidia-container-toolkit
```

Setting default runtime for docker by editing ``/etc/docker/daemon.json`` and then ``sudo systemctl daemon-reload && sudo systemctl restart docker``:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```