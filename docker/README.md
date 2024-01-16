# Run container for 3D inference


```bash
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --platform linux/amd64 -t niqbal996/triton-client:22.04-py3-PCDet .
```

```bash
docker run -it --runtime=nvidia --net=host --name=triton-client  --ipc=host -e PYTHONPATH=/opt/depencies/OpenPCDet niqbal996/triton-client:22.04-py3-PCDet 
```

