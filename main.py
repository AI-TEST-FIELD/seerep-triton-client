import logging
from triton import TritonInference
from logger import Client_logger, TqdmToLogger
logger = Client_logger(name='SEEREP-Client', level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)

example_project_name = 'EV41_Kleidung_Sonnenbrille_DunkleCap_225Deg_2024-10-10-18-58-13_0'
example_project_uuid = '52a469bd-4222-4753-9c3e-8b5bfef1d15a'
example_uuids = ['abe39aac-a91a-464d-a2b4-896a3a9075fc',
               '5542cecf-1dfb-4875-9ae5-c0d2a6d54bc7']

model_name = 'yolov5m_coco'
model_names = ['yolov5m_coco', 'retinanet_coco']


# example_project_name = 'map'
# example_project_uuid = 'c8a411ab-f3c9-4f41-b38f-9d1c73515cce'
# example_uuids = ['7ffc1491-b083-4929-b70a-b2e1c45e49ba',
#            'c0cb19f2-1a02-4d90-85e0-9a900b8a0f46']
# model_names = ['second_iou_kitti']


seerep_endpoint = "agrigaia-ur.ni.dfki:9090" #    URL tested with images
# seerep_endpoint = "localhost:9090"             #    URL tested with SEEREP server v0.3.5 
triton_endpoint = "10.249.6.4:8001"

visualize = False

triton_client  = TritonInference(
                                model_name=model_names,
                                seerep_endpoint_url=seerep_endpoint,
                                triton_endpoint_url=triton_endpoint,
                                log_level='info',
                                visualize=visualize,
                                modality='image')
# Fetch data only once before generating annotations for each model
data = triton_client.seerep_endpoint.fetch_data_by_sample_uuid(example_uuids, model_name=model_names)
for model_name in model_names:
    logger.info("Generating annotations for model: %s", model_name)
    data = triton_client.generate_datumaro_predictions(data, model_key=model_name)
    print('data')
    data = triton_client.seerep_endpoint.send_dataset(
                                                    uuids=example_uuids,
                                                    data=data,
                                                    category='groundtruth')
    
    
# 1. When we want to process data in terms of samples from the SEEREP server
# triton_client.generate_annotations_by_sample_uuids(sample_uuids=example_uuids)

# 2. When we want to process data in terms of projects uuids from the SEEREP server
# triton_client.generate_annotations_by_project_uuids(project_uuids=[example_project_uuid])

# 3. When we want to process data in terms of projects names from the SEEREP server
# project_uuid = triton_client.get_project_uuid(project_name=example_project_name)
# triton_client.generate_annotations_by_project_uuids(project_uuids=[project_uuid])