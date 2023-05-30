# seerep-triton-client
A client repo to communicate with SEEREP and perform inference with Triton Inference Server

## Installation
Tested on Ubuntu 20.04 LTS Python 3.7 
```bash
python3 -m pip install -r requirements.txt
```
## Usage
This repo assumes that the Triton server with AI models are served via Triton at a given URL e.g.  <triton-grpc-endpoint:8001> and SEEREP server is accessible at another given URL e.g. <seerep-server-grpc-endpoint:9090>. For an example model e.g. `YOLOv5nCOCO`, and a SEEREP project`aitf-labelled-data` you can run inference as following:

```bash
python3 main.py -cT <triton-grpc-endpoint>:8001 -cS <seerep-server-grpc-endpoint>:9090 -p aitf-labelled-data -m YOLOv5nCOCO 
```
After all the images satisfying your SEEREP query have been processed, Average precision metrics should be generated for your query using `PyCOCOtools`. An example output is shown below:

```bash
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.338
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.500
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.500
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.494
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.556
```