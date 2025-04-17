# seerep-triton-client
A client repo to fetch data from SEEREP server and send the dataset to Triton Server for predictions from a given model. Supports both Image and LiDAR pointcloud modalities. 

## Installation (Recommended)
Tested with DEV Container Ubuntu 20.04 LTS Python 3.10.12 
## Usage
This repo assumes that the Triton server with AI models are served via Triton at a given URL e.g.  <triton-grpc-endpoint:8001> and SEEREP server is accessible at another givedn URL e.g. <seerep-server-grpc-endpoint:9090>. An example query scenario based on sample UUIDs could be:
```python3
example_uuids = ['1957e21a-2f4f-40f5-b4de-b0a693fd7bc0',
               'f90f8e7c-1e08-4868-ad8d-a6f690ebca94']
model_name = 'yolov5m_coco'
seerep_endpoint = "localhost:9090"             
triton_endpoint = "10.249.6.30:8001"

triton_client  = TritonInference(
                                model_name=model_name,
                                seerep_endpoint_url=seerep_endpoint,
                                triton_endpoint_url=triton_endpoint,
                                log_level='info',
                                modality='image')
# 1. When we want to process data in terms of samples from the SEEREP server
triton_client.generate_annotations_by_sample_uuids(sample_uuids=example_uuids)
```
## List of available models 
You can currently pass following model strings for inference from triton-server as arguments to above command. All model names are tailed by the dataset they were trained on e.g. `coco`, `kitti`, and `iso` :
```bash
fcos_coco      
frcnn_800_coco    
pointpillar_kitti ---> Not working on RTX-3090 TI GPUs
retinanet_coco 
second_iou_kitti
yolov5m_coco   
yolov5m_iso  
```
## TODOs
- [ ] Pointpillar crashes on RTX-3090 TI GPUs. 
- [x] Triton class with data UUIDs as input. 
- [ ] Parallize and memory management on large datasets. 
- [ ] Send pointcloud predictions back as Datumaro dataset. 
- [ ] Visualize opencv inside dev container.  
## Acknowledgements
1. [Triton Inference Server](https://github.com/triton-inference-server/server)
2. [SEEREP](https://github.com/DFKI-NI/seerep/tree/main)
3. [Ultralytics](https://github.com/ultralytics/ultralytics)
4. [Detectron2](https://github.com/facebookresearch/detectron2)
5. [Detrex](https://github.com/IDEA-Research/detrex)
6. [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

