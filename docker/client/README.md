# Run container for 3D inference

## Standart Installation
```bash
sudo -H DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile --platform linux/amd64 -t niqbal996/triton-client:22.04-py3-PCDet .
```

```bash
docker run -it --runtime=nvidia --net=host --name=triton-client  --ipc=host -e PYTHONPATH=/opt/depencies/OpenPCDet niqbal996/triton-client:22.04-py3-PCDet 
```

## Installation with X11 Forwarding

### Host Setup
For detailed information about X11 and Docker go to [this](https://www.baeldung.com/linux/docker-container-gui-applications) website.
1. Configure the ssh configuration file for X11 forwarding.
```bash
$ sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.orig
$ sudo vim /etc/ssh/sshd_config

...
X11Forwarding   yes
X11UseLocalhost no
...
```
2. Open the forwarded X server port

```bash
$ sudo xhost +local:docker
```

### Docker Image creation
(Nvidia Docker Runtime has to be installed, see [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

```bash
$ docker run -it --runtime=nvidia --ipc=host --net=host \
--name={container-name} -v /tmp/.X11-unix:/tmp/.X11-unix \
-e DISPLAY -v {seerep-client-path}:/opt/client \
niqbal996/triton-client:22.04-py3-PCDet
```
With: \
{container-name} = The Container Name
{seerep-client-path} = The full path to your local seerep triton client repositoty (e.g. /home/robot/git/seerep-triton-client)

### openPC Detection installation
Within the Docker Container execute the following commands:

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd openPCDet
python setup.py develop
```
