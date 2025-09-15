import logging
from triton import TritonInference
from logger import Client_logger, TqdmToLogger
logger = Client_logger(name='SEEREP-Client', level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)

# example_project_name = 'EV41_Kleidung_Sonnenbrille_DunkleCap_225Deg_2024-10-10-18-58-13_0'
example_project_uuid = ['2918b2dd-0bc9-40d5-bf94-0a60ff0f1b3a']
# example_uuids = ['70c5a873-4fac-45ee-ac3a-2383c348bfad', # processed AND with ground truth
#                'da1889e1-ad5c-474a-8fd8-a30ca65f51ef']
# example_uuids = ['3491e170-dea1-42fa-b856-086ea7d63481']
# example_uuids = ['2c05a782-b273-48bf-af55-48f5f49d08d5',    # unprocessed AND no ground truth
#                  '8d54e1d5-8750-4d28-aaad-cc455c400a49',
#                  'bcd06e43-c970-46e3-8d56-6c7e08496541',
#                  'f5931119-142e-4c20-8509-f702d8e1320a']
# example_uuids = ["e9297202-ad5f-42ca-802f-10744be88bca"]

# example_uuids = ['e64377ff-898a-4ab0-bf42-89f667a2f2b9',    # unprocessed AND with ground truth
#                  'fbaef724-5c59-4a4f-ad04-dc1845b0381b']
# model_names = ['retinanet_coco']
# model_names = ['yolov5m_coco']
# model_names = ['retinanet_coco',]


# example_project_name = 'map'
# example_project_uuid = 'c8a411ab-f3c9-4f41-b38f-9d1c73515cce'
# example_uuids = ['7ffc1491-b083-4929-b70a-b2e1c45e49ba',
#            'c0cb19f2-1a02-4d90-85e0-9a900b8a0f46']
model_names = ['second_iou_kitti', 'pointpillar_kitti']
# model_names = ['second_iou_kitti']


seerep_endpoint = "agrigaia-ur.ni.dfki:9090" #    URL tested with images
# seerep_endpoint = "localhost:9090"             #    URL tested with SEEREP server v0.3.5
triton_endpoint = "10.249.6.4:8001"
# triton_endpoint = "localhost:8001"

visualize = False

triton_client  = TritonInference(
                                model_name=model_names,
                                seerep_endpoint_url=seerep_endpoint,
                                triton_endpoint_url=triton_endpoint,
                                log_level='info',  # 'error' or 'info'
                                visualize=visualize,
                                modality='pointcloud')  # 'image', 'pointcloud'
# Fetch data only once before generating annotations for each model
# data = triton_client.seerep_endpoint.fetch_data_by_sample_uuid(example_uuids, model_name=model_names)
# project_uuid = triton_client.seerep_endpoint.get_project_uuid(project_name=example_project_name)
data = triton_client.seerep_endpoint.fetch_data_by_project(
                                                        example_project_uuid,
                                                        model_name=model_names,
                                                        num_samples=1,
                                                        modality='pointcloud'
                                                        )
data_uuids = [sample['uuid'] for sample in data]
batch_size = 100
for batch in range(0, len(data_uuids), batch_size):
    for model_name in model_names:
        logger.info("Generating annotations for model: %s", model_name)
        data = triton_client.generate_datumaro_predictions(data[batch:batch + batch_size], model_key=model_name)
        # triton_client.seerep_endpoint.send_dataset(
        #                                     uuids=data_uuids[batch:batch + batch_size],
        #                                     data=data,
        #                                     category=model_name,     # 'groundtruth' or model_name
        #                                     ignore_ground_truth=False)
        data = []
# If True, predictions will be sent as ground truth. coupled with the category parameter as 'groundtruth'


# 1. When we want to process data in terms of samples from the SEEREP server
# triton_client.generate_annotations_by_sample_uuids(sample_uuids=example_uuids)

# 2. When we want to process data in terms of projects uuids from the SEEREP server
# triton_client.generate_annotations_by_project_uuids(project_uuids=[example_project_uuid])

# 3. When we want to process data in terms of projects names from the SEEREP server
# project_uuid = triton_client.get_project_uuid(project_name=example_project_name)
# triton_client.generate_annotations_by_project_uuids(project_uuids=[project_uuid])