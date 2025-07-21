import sys
import numpy as np
import struct
import logging
import yaml
import grpc
import open3d as o3d
import flatbuffers
import tritonclient.grpc.model_config_pb2 as mc
import itertools
from typing import List, Tuple, Set
# import uuid
import cv2
import json
# import os

from tqdm.auto import tqdm
from copy import copy
from scipy.spatial.transform import Rotation as R
from grpc import Channel

from seerep.fb import (
    Boundingbox,
    Empty,
    FrameQuery,
    Header,
    Image,
    Point,
    ProjectInfos,
    Query,
    StringVector,
    TimeInterval,
    Timestamp,
    TRANSMISSION_STATE,
    LabelCategory
)

from seerep.fb import (
    PointCloud2 as pc2,
    image_service_grpc_fb as imageService,
    point_cloud_service_grpc_fb as pointCloudService,
    meta_operations_grpc_fb as metaOperations,
    tf_service_grpc_fb as tfService,
    TransformStamped
)

from seerep.util import fb_helper as util_fb
from seerep.util.common import get_gRPC_channel
from seerep.fb.ServerResponse import ServerResponse
from seerep.util.fb_helper import (
    create_dataset_uuid_label,
    createHeader,
    create_label,
    create_label_category,
    createTimeStamp,
    createTransformStampedQuery,
    createTimeInterval,
    createQuery,
)
from tools.pointcloud import (
    pcd_o3d_to_numpy,
    pcd_ros_to_o3d,
)
from visual_utils import Visualizer
# from visual_utils import open3d_vis_utils as visualizer
from utils import cxcy2xyxy
from logger import Client_logger, TqdmToLogger

logger = Client_logger(name='SEEREP-Client', level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)
O3D_DEVICE = o3d.core.Device("CPU:0")  # can also be set to GPU

class APIError(Exception):
    pass

class DatumaroError(Exception):
    pass

Point_Field_Datatype =  {
    0: 'unset',
    1: np.dtype(np.int8),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int16),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.uint32),
    7: np.dtype(np.float32),
    8: np.dtype(np.float64),
}

class SeerepEndpoint:
    """
    A SeerepEndpoint establishes a connection between the triton client and SEEREP.
    """
    def __init__(self,
                 endpoint_url='agrigaia-ur.ni.dfki:9090',
                 modality='image',
                 visualize=False,
                 log_level='ERROR'):
        self._grpc_stub = None
        self._grpc_stubmeta = None
        self._builder = None
        self.endpoint_url = endpoint_url
        self.visualize = visualize
        # register and initialise the stub
        self.modality = modality
        self.intialize_gRPC_stubs()
        
        # self.ann_dict = self.annotation_dict(format=format)
        if self.visualize and self.modality == 'image':
            self.source_window = 'SEEREP source image'
            cv2.namedWindow(self.source_window)
        elif self.visualize and self.modality == 'pointcloud':
            self.source_window = 'SEEREP source pointcloud'
            self.visualizer = Visualizer(origin=True)
        logger.info("SEEREP Channel initialized successfully via endpoint : {} ".format(self.endpoint_url))
        # Set logger level based on log_level argument
        log_level_map = {
            'ERROR': logging.ERROR,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
        }
        logger.setLevel(log_level_map.get(log_level.upper(), logging.ERROR))

    def register_grpc_channel(self):
        """
        Returns:
            A gRPC stub for the SEEREP server
        """
        channel = grpc.insecure_channel(self.endpoint_url,
                                        options=[('grpc.max_message_length', -1),
                                         ('grpc.max_send_message_length', -1),
                                         ('grpc.max_receive_message_length', -1)])
        return channel

    def intialize_gRPC_stubs(self):
        """
         register SEEREP gRPC channel
         endpoint_url: String or IP address and PORT of the SEEREP server
         e.g. agrigaia-ur.ni.dfki:9090
        """
        self.grpc_channel = self.register_grpc_channel()
        if self.modality == 'image':
            self._grpc_stub  = imageService.ImageServiceStub(self.grpc_channel)
        elif self.modality == 'pointcloud':
            self._grpc_stub  = pointCloudService.PointCloudServiceStub(self.grpc_channel)
        else:
            logger.error("Modality not supported. Please use image or pointcloud")
            sys.exit(0)
        self._grpc_stubmeta = metaOperations.MetaOperationsStub(self.grpc_channel)
        self._builder = self.init_builder()

    def secondary_channel(self):
        """
         Establish another channel for sending data to SEEREP server
         register gRPC SEEREP channel
         socket: String, Port and IP address of seerep server
         seerep.robot.10.249.3.13.nip.io:32141
        """
        grpc_channel = self.register_grpc_channel()
        if self.modality == 'image':
            grpc_stub  = imageService.ImageServiceStub(grpc_channel)
        elif self.modality == 'pointcloud':
            grpc_stub  = pointCloudService.PointCloudServiceStub(grpc_channel)
        else:
            logger.error(
                "Modality not supported. Please use image or pointcloud. \n"+ 
                "Cannot create secondary channel for target modality {}".format(self.modality))
            sys.exit(0)
        grpc_stubmeta = metaOperations.MetaOperationsStub(grpc_channel)
        builder = self.init_builder()

        return (grpc_stub, grpc_stubmeta, builder)

    def fetch_channel(self):
        """
        return grpc stub
        """
        return self._grpc_stub
    
    def tf_channel(self):
        """
         Establish a channel for querying TFs
        """
        grpc_channel = self.register_grpc_channel()
        tf_stub  = tfService.TfServiceStub(grpc_channel)
        builder = self.init_builder()

        return (tf_stub, builder)

    def init_builder(self):
        """
        Initialize flatbuffer builder
        """
        builder = flatbuffers.Builder(1024)

        return builder
    
    def deserialize_bytes_float(self, encoded_tensor):
        strs = list()
        offset = 0
        val_buf = encoded_tensor
        datatype = "f"
        l = struct.calcsize(datatype)
        while offset < len(val_buf):
            sb = struct.unpack_from(datatype, val_buf, offset)[0]
            offset += l
            strs.append(sb)
        return (np.array(strs, dtype=np.object_))

    def deserialize_bytes_int(self, encoded_tensor):
        strs = list()
        offset = 0
        val_buf = encoded_tensor
        datatype = "l"
        l = struct.calcsize(datatype)
        while offset < len(val_buf):
            sb = struct.unpack_from(datatype, val_buf, offset)[0]
            offset += l
            strs.append(sb)
        return (np.array(strs, dtype=np.object_))

    def get_project_uuid(self, project_name: list[str], log=False):
        '''
        Returns UUID of the project given by project_name.
        #TODO return a list of projects or single project uuids? 
        '''
        Empty.Start(self._builder)
        emptyMsg = Empty.End(self._builder)
        self._builder.Finish(emptyMsg)
        buf = self._builder.Output()
        projects = {}
        responseBuf = self._grpc_stubmeta.GetProjects(bytes(buf))
        response = ProjectInfos.ProjectInfos.GetRootAs(responseBuf)
        curr_proj = None
        duplicate = False
        logger.info("List of available projects on the SEEREP Server")
        for i in range(response.ProjectsLength()):
            try:
                tmp = response.Projects(i).Name().decode("utf-8")
                if tmp in projects:
                    logger.info(tmp+'_2' + " " + response.Projects(i).Uuid().decode("utf-8"))
                    logger.warning('Found multiple projects with same project name but with different UUIDs! Please check SEEREP Server!')
                    projects[tmp+'_2'] = response.Projects(i).Uuid().decode("utf-8")
                    duplicate = True
                else:
                    logger.info(tmp + " " + response.Projects(i).Uuid().decode("utf-8"))
                    projects[tmp] = response.Projects(i).Uuid().decode("utf-8")
            except Exception as e:
                logger.error(e)
        if project_name in projects:
            curr_proj = project_name
            projectuuid = projects[curr_proj]
            logger.info("Found project {} with UUID: {}".format(curr_proj, projectuuid))
            return projectuuid
        else:
            logger.error("The requested project \n {} is not available on the SEEREP Server! Note that project names are case-sensitive! Please select a project from the list displayed above!".format(project_name))
            sys.exit(0)
            
    def fetch_data_by_sample_uuid(self, data_uuids: list[str], model_name: str)->dict:
        '''
        Fetches data from SEEREP server based on the data sample UUIDs and looks through all projects. 
        It also checks if the predictions for the model_name have already been generated for the data samples. 
        If yes, then sets 'processed' flag to True.
        Returns a list of dictionaries containing the data samples with the following
        keys: 'uuid', 'image', 'timestamp', 'processed', 'no_grountruth', 'annotations'
        '''
        queryMsg = util_fb.createQuery(
            self._builder,
            # projectUuids=projectUuids,
            dataUuids=data_uuids,
            # withoutData=False,
            sortByTime=True,  # from version 0.2.5 onwards
        )
        self._builder.Finish(queryMsg)
        buffer = self._builder.Output()
        if self.modality == 'image':
            return self.process_images(buffer, model_name=model_name, num_samples=len(data_uuids))
        elif self.modality == 'pointcloud':
            return self.process_pointclouds(buffer, model_name=model_name, num_samples=len(data_uuids))
        else:
            logger.error("Modality not supported. Please use image or pointcloud")
            sys.exit(0)
    
    def process_images(self, buffer, model_name, num_samples:int=None)->dict:
        '''
        buffer: flatbuffer buffer containing the query message 
        generated using the SEEREP createQuery function
        model_name: name of the model for which the predictions are to be generated.
        num_samples: [DEBUG ONLY] number of samples to fetch from the SEEREP server.
        Returns a list of dictionaries containing the data samples with the following
        keys: 'uuid', 'image', 'timestamp', 'processed', 'no_grountruth', 'annotations'
        '''
        data = []
        data_generator = self._grpc_stub.GetImage(bytes(buffer))
        # Limit the generator if num_samples is provided
        if num_samples is not None:
            data_generator = itertools.islice(data_generator, 1600, 1600+num_samples)
        
        # Set tqdm total only if num_samples is provided
        tqdm_kwargs = {
            "desc": "Fetching images from the SEEREP server",
            "colour": "blue",
            "unit": " image(s)"
        }
        if num_samples is not None:
            tqdm_kwargs["total"] = num_samples
        for responseBuf in tqdm(data_generator, **tqdm_kwargs):
            sample = {}
            response = Image.Image.GetRootAs(responseBuf)
            msguuid = response.Header().UuidMsgs().decode("utf-8")
            sample['uuid'] = msguuid
            sample['image'] = np.reshape(response.DataAsNumpy(), (response.Height(), response.Width(), -1))[:, :, 0:3] # When more than 3 channels
            sample['image'] = np.ascontiguousarray(sample['image'], dtype=np.uint8).astype(np.uint8)
            sample['timestamp'] = [response.Header().Stamp().Seconds(), response.Header().Stamp().Nanos()]  # seconds nanos
            sample['project_uuid'] = response.Header().UuidProject().decode("utf-8")
            sample['processed'] = []
            sample['no_grountruth'] = False
            sample['annotations'] = {
                "info": {},
                "categories": {
                    "label": {
                        "labels": [],
                        "attributes": [],
                    },
                    "points": {"items": []},
                },
                "items": [],
            }
            labels: Set[Tuple[str, int]] = set()
            for label_idx in range(response.LabelsLength()):    # Here LabelsLength correspond to number of categories
                category_with_labels = response.Labels(label_idx)
                if(not (category_with_labels.Category().decode() == 'labelGeneral')):
                    item = json.loads(category_with_labels.DatumaroJson().decode())
                    sample['annotations']['categories']["label"]["labels"].append(category_with_labels.Category().decode())
                    sample['annotations']["items"].append(item)
                    for j in range(category_with_labels.LabelsLength()):    # Here LabelsLength correspond to number of labels per category
                        labels.add(
                            (
                                category_with_labels.Labels(j).Label().decode(),
                                category_with_labels.Labels(j).LabelIdDatumaro(),
                            )
                        )
            # This condition makes sure we do not predict the labels twice and send them back AGAIN!!! to SEEREP
            # Check which entries in model_name are present in the labels and add them to 'processed'
            if isinstance(model_name, list):
                sample['processed'] = [name for name in model_name if name in sample['annotations']['categories']['label']['labels']]
            else:
                if model_name in sample['annotations']['categories']['label']['labels']:
                    sample['processed'].append(model_name)
            # DEBUG
            if len(sample['annotations']['items']) == 0:
                sample['no_grountruth'] = True
            data.append(sample.copy())
            sample = {}  # flush the sample data for new incoming samples
        logger.info('Fetched {} images from the current SEEREP project'.format(len(data)))
        return data
    
    def process_pointclouds(self, buffer, model_name, num_samples:int=10)->dict:
        '''
        buffer: flatbuffer buffer containing the query message 
        generated using the SEEREP createQuery function
        model_name: name of the model for which the predictions are to be generated.
        Returns a list of dictionaries containing the data samples with the following keys:
        - 'uuid': Unique identifier for the point cloud sample.
        - 'sensor_name': Name of the sensor that captured the point cloud.
        - 'timestamp': Timestamp of the point cloud sample as a tuple (seconds, nanoseconds).
        - 'point_cloud': Dictionary containing the fields of the point cloud (e.g., x, y, z, intensity, etc.).
        - 'lidar_feature': The feature field used for the point cloud (e.g., 'reflectivity' or 'intensity').
        '''
        data = []
        sample = {}
        data_generator = self._grpc_stub.GetPointCloud2(bytes(buffer))
        for responseBuf, curr_sample in tqdm(zip(data_generator, range(num_samples)),
                        total=num_samples,
                        colour='GREEN',
                        desc='Receiving pointclouds from the SEEREP server',
                        unit=" samples"):
            response = pc2.PointCloud2.GetRootAs(responseBuf)
            msguuid = response.Header().UuidMsgs().decode("utf-8")
            projuuid= response.Header().UuidProject().decode("utf-8")
            timestamp = response.Header().Stamp().Seconds(), response.Header().Stamp().Nanos()
            height = response.Width()
            width = response.Height()
            sample['uuid'] = msguuid
            sample['project_uuid'] = projuuid
            sample['sensor_name'] = response.Header().FrameId().decode("utf-8")
            sample['timestamp'] = timestamp
            sample['processed'] = []
            sample['no_grountruth'] = False
            sample['annotations'] = {
                "info": {},
                "categories": {
                    "label": {
                        "labels": [],
                        "attributes": [],
                    },
                    "points": {"items": []},
                },
                "items": [],
            }
            raw_data = response.DataAsNumpy()
            fields = {}
            dtype = None
            for j in range(response.FieldsLength()):
                fields[response.Fields(j).Name().decode('utf-8')] = {}
                fields[response.Fields(j).Name().decode('utf-8')]['offset'] = response.Fields(j).Offset()
                fields[response.Fields(j).Name().decode('utf-8')]['dtype'] = response.Fields(j).Datatype()
                dtype = Point_Field_Datatype[response.Fields(j).Datatype()]
                try:
                    # Use the lookup function to get the struct format character
                    c = self.get_data_type_character(np.dtype(dtype).type)
                except ValueError as e:
                    logger.error(e)
                    continue
                fields[response.Fields(j).Name().decode('utf-8')]['data_string'] = c
                fields[response.Fields(j).Name().decode('utf-8')]['size'] = struct.calcsize(c) 
            for field in fields:
                strs = list()
                for i in range(fields[field]['offset'], raw_data.shape[0], response.PointStep()):      # Each chunk size must have one entry for each field i.e. x,y,z,intensity, t, reflectivity, ring, ambient, range
                    sb = struct.unpack(fields[field]['data_string'], raw_data[i : i + fields[field]['size']])
                    strs.append(sb)
                fields[field]['data'] = (np.array(strs, dtype=np.object_))
                strs = []
            sample['pointcloud'] = copy(fields) 
            if 'reflectivity' in fields:
                sample['lidar_feature'] = 'reflectivity'
            else:
                sample['lidar_feature'] = 'intensity'
            if self.visualize:
                pc = np.zeros((height*width, 3), dtype=np.float64)
                pc[:, 0] = fields['x']['data'][:, 0]
                pc[:, 1] = fields['y']['data'][:, 0]
                pc[:, 2] = fields['z']['data'][:, 0]
                self.visualizer.draw_scenes(points=pc)
                
            # TODO Fetch the labels for pointclouds from SEEREP server
            # labels: Set[Tuple[str, int]] = set()
            # for label_idx in range(response.LabelsLength()):    # Here LabelsLength correspond to number of categories
            #     category_with_labels = response.Labels(label_idx)
            #     if(not (category_with_labels.Category().decode() == 'labelGeneral')):
            #         item = json.loads(category_with_labels.DatumaroJson().decode())
            #         sample['annotations']['categories']["label"]["labels"].append(category_with_labels.Category().decode())
            #         sample['annotations']["items"].append(item)
            #         for j in range(category_with_labels.LabelsLength()):    # Here LabelsLength correspond to number of labels per category
            #             labels.add(
            #                 (
            #                     category_with_labels.Labels(j).Label().decode(),
            #                     category_with_labels.Labels(j).LabelIdDatumaro(),
            #                 )
            #             )
            # # This condition makes sure we do not predict the labels twice and send them back AGAIN!!! to SEEREP
            # # Check which entries in model_name are present in the labels and add them to 'processed'
            if isinstance(model_name, list):
                sample['processed'] = [name for name in model_name if name in sample['annotations']['categories']['label']['labels']]
            else:
                if model_name in sample['annotations']['categories']['label']['labels']:
                    sample['processed'].append(model_name)
            # DEBUG
            if len(sample['annotations']['items']) == 0:
                sample['no_grountruth'] = True
            data.append(sample.copy())
            # flush the sample data for new incoming samples
            sample={}    
        logger.info('Fetched {} pointclouds from the current SEEREP project'.format(len(data)))
        # TODO Does the parent frame change a lot to be dynamic? 
        data = self.run_query_tf(data, parent_frame='base_link')
        data = self.preprocess_pc(data, model_name=model_name, num_samples=num_samples)
        return data
    
    @staticmethod
    def get_data_type_character(dtype):
        """
        # https://docs.python.org/2/library/struct.html          
        Returns the struct format character for a given numpy data type using a lookup table.
        Args:
            dtype: Numpy data type (e.g., np.int16, np.uint16, etc.)
        Returns:
            A string representing the struct format character (e.g., 'h', 'H', etc.)
        """
        # Lookup table for data type to struct format character
        dtype_lookup = {
            np.int16: 'h',     # 16-bit short
            np.uint16: 'H',    # 16-bit unsigned short
            np.int32: 'i',     # 32-bit int
            np.uint32: 'I',    # 32-bit unsigned int
            np.float32: 'f',   # 32-bit float
            np.float64: 'd',   # 64-bit double
        }

        # Get the struct format character from the lookup table
        c = dtype_lookup.get(dtype, None)

        if c is None:
            raise ValueError(f"Invalid data type: {dtype}")

        return c
    
    def fetch_uuids_by_project_uuid(self, project_uuid: list[str])->list[str]:
        if isinstance(project_uuid, str):
            projectUuids = [project_uuid]
        elif isinstance(project_uuid, list) and all(isinstance(item, str) for item in project_uuid):
            projectUuids = project_uuid
        else:
            logger.error('Please provide a valid project UUID')
            sys.exit(0)
        queryMsg = util_fb.createQuery(
            self._builder,
            projectUuids=projectUuids,
            # dataUuids=data_uuids,
            # withoutData=False,
            sortByTime=True,  # from version 0.2.5 onwards
        )
        self._builder.Finish(queryMsg)
        buffer = self._builder.Output()
        return self.process_uuids(buffer)
    
    def fetch_data_by_project(self, project_uuids: list[str], model_name: str, num_samples:int=None)->dict:
        '''
        Fetches data from SEEREP server based on the project_uuids. It also checks if the predictions for the model_name
        have already been generated for the data samples. If yes, then sets 'processed' flag to True.
        Args:
        project_uuids: list of project UUIDs to fetch data from.
        model_name: name of the model for which the predictions are to be generated.
        num_samples: [DEBUG ONLY] number of samples to fetch from the SEEREP server
        Returns a list of dictionaries containing the data samples with the following
        keys: 'uuid', 'image', 'timestamp', 'processed', 'no_grountruth', 'annotations'
        '''
        queryMsg = util_fb.createQuery(
            self._builder,
            # boundingBox=boundingboxStamped,
            # timeInterval=timeInterval,
            # labels=['RetinaNet'],
            # mustHaveAllLabels=False,
            projectUuids=project_uuids,
            # instanceUuids=instanceUuids,
            # dataUuids=dataUuids,
            # withoutData=False,
            sortByTime=True,  # from version 0.2.5 onwards
        )
        self._builder.Finish(queryMsg)
        buffer = self._builder.Output()
        return self.process_images(buffer, model_name=model_name, num_samples=num_samples)

    # TODO Can this be related to AGROVOC?
    def annotation_dict(self, format='aitf'):
        anns_dict = {}
        class_names= []
        if format == 'coco':
            filepath = 'config/coco.names'
        elif format == 'kitti':
            filepath = 'config/kitti.names'
        elif format == 'aitf':
            filepath = 'config/aitf.names'
        elif format == 'crop':
            filepath = 'config/crop.names'
        with open(filepath, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        for id,idx in zip(class_names, range(len(class_names))):
            anns_dict[idx+1] = id

        return anns_dict
        
    def process_uuids(self, buffer)->list[str]:
        '''
        buffer: flatbuffer buffer containing the query message 
        generated using the SEEREP createQuery function
        model_name: name of the model for which the predictions are to be generated.
        Returns a list of string containing the UUIDs of the data samples
        '''
        data = []
        if self.modality == 'image':
            data_generator = self._grpc_stub.GetImage(bytes(buffer))
            response_generator =  Image.Image.GetRootAs
        elif self.modality == 'pointcloud':
            data_generator = self._grpc_stub.GetPointCloud2(bytes(buffer))
            response_generator =  pc2.PointCloud2.GetRootAs
        else:
            logger.error("Modality not supported. Please use image or pointcloud")
            sys.exit(0)
        for responseBuf in tqdm(data_generator, desc="Fetching UUIDs", unit=" uuid(s)", colour="blue"):
            response = response_generator(responseBuf)
            sample_uuid = response.Header().UuidMsgs().decode("utf-8")
            data.append(sample_uuid)
        logger.info('Fetched {} UUIDs from the current SEEREP project'.format(len(data)))
        return data
    
    def preprocess_pc(self, pointcloud_data: dict, model_name: list, num_samples:int=10) -> dict:
        """
        Processing Steps:
            1. Transforms Point Cloud into the Robot base_frame, based on homegenous transform from the calibration procedure.
            2. Translate Point Cloud into the Dataset specific detector training dataset frame. Adjusts the Point Cloud to mimic the relative Lidar position from the detectors training dataset.
            3. Normalize feature field [0, 1], by maximal possible feature value (reflectance/intensity = 255).


        Args:
            sample : Dictionary containing the point cloud and sensor specific information.
            sensor_name : Sensor specific name tag associated with the point cloud.
            dataset_name : The name of the dataset used for training of the object detector.

        Return:
            preprocessed_np_pcd : Preprocessed point cloud as numpy array [[x, y, z, feature], ...]

        """

        # dictionary for sensor and dataset transformations
        # TODO how to change this to be compatible with a list of models?
        if 'kitti' in model_name[0]:
            dataset_translation = [0.0, 0.0, -1.026558971]
        # TODO add nuscenes translation
        elif 'nuscenes' in model_name[0]: 
            dataset_translation = [0.0, 0.0, -1.026558971]  
        else:
            logger.error(f"Dataset {model_name[0]} not supported. Please use kitti or nuscenes.")
            return None
        for sample in tqdm(pointcloud_data,
                            desc='Transforming pointclouds',
                            unit="samples",
                            colour='YELLOW',
                            total=len(pointcloud_data)):
            # check if the sample has a tf
            if 'transform_matrix' in sample:
                # get the transform matrix for the current pc sample
                transform_matrix = np.array(sample['transform_matrix']).reshape(4, 4)
            else:
                logger.error(f"Sample {sample['uuid']} does not have a tf. Skipping...")
                continue
            
            MAX_FEATURE_VALUE = 255
            # sensor and data specific params
            sensor_to_robot_base_transform = o3d.core.Tensor(
                transform_matrix, device=O3D_DEVICE
            )
            robot_base_to_train_dataset_translation = o3d.core.Tensor(
                dataset_translation, device=O3D_DEVICE
            )
            feature_field = sample['lidar_feature']

            # preprocessing steps
            raw_o3d_pcd = pcd_ros_to_o3d(ros_pcd=sample['pointcloud'], feature_field=feature_field)
            preprocessed_o3d_pcd = raw_o3d_pcd.transform(sensor_to_robot_base_transform)
            preprocessed_o3d_pcd = preprocessed_o3d_pcd.translate(
                robot_base_to_train_dataset_translation
            )
            preprocessed_np_pcd = pcd_o3d_to_numpy(
                o3d_pcd=preprocessed_o3d_pcd, feature_field=feature_field
            )
            preprocessed_np_pcd[:, 3] /= MAX_FEATURE_VALUE
            pointcloud_data[pointcloud_data.index(sample)]['pointcloud_processed'] = preprocessed_np_pcd
            
        return pointcloud_data
    
    def run_query_tf_frames(self, target_proj_uuid: str = None, grpc_channel: Channel = get_gRPC_channel()
                            )-> dict:
        """
        Flat buffers based gRPC query for frames
        Args:
            target_proj_uuid: UUID of the project to query frames from
            grpc_channel: gRPC channel to use for the query
        Returns:
            frame_dict: dictionary containing the frames
        https://github.com/DFKI-NI/seerep/blob/main/examples/python/gRPC/tf/gRPC_pb_queryFrames.py
        """
        stub = tfService.TfServiceStub(grpc_channel)
        builder = self.init_builder()
        projectUuid = builder.CreateString(target_proj_uuid)
        FrameQuery.Start(builder)
        FrameQuery.AddProjectuuid(builder, projectUuid)
        frameQuery = FrameQuery.End(builder)
        builder.Finish(frameQuery)
        buf = builder.Output()
        
        responseBuf = stub.GetFrames(bytes(buf))
        response = StringVector.StringVector.GetRootAs(responseBuf)
        for idx in range(response.StringVectorLength()):
            frame_str = response.StringVector(idx).decode("utf-8")
            try:
                frame_dict = yaml.safe_load(frame_str)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing frame string as YAML: {e}")
        return frame_dict
    
    def run_query_tf(self, 
                     data: list[dict],
                     parent_frame: str='base_link',) -> list[dict]:
        """
        Query the TF for each pointcloud
        Args:
            data: list of dictionaries containing pointcloud data
        Returns:
            data: list of dictionaries containing pointcloud data with TFs
        """
        frames = self.run_query_tf_frames(data[0]['project_uuid'], self.grpc_channel)
        tf_stub, builder = self.tf_channel()
        for sample, index in tqdm(zip(data, range(len(data))),
                            total=len(data),
                            colour='BLUE',
                            desc='Query TF for each pointcloud',
                            unit=" samples"):
            
            timestamp = createTimeStamp(builder, sample['timestamp'][0], sample['timestamp'][1])    # [0] is seconds, [1] is nanoseconds
            header = createHeader(
                builder=builder,
                timeStamp=timestamp,
                frame=parent_frame, # Parent frame ID should be the base link
                msgUuid=sample['uuid'],
                projectUuid=sample['project_uuid']
            )
            if parent_frame in frames:
                tf_query = createTransformStampedQuery(
                    builder=builder,
                    header=header,  
                    childFrameId=sample['sensor_name'],  # Child frame ID should be the sensor name sample['sensor_name']
                )
            else:
                logger.error(f"Parent frame {parent_frame} not found in the following list of frames:")
                logger.error(", \n".join(frames.keys()))
            builder.Finish(tf_query)
            tf_buf: bytearray = tf_stub.GetTransformStamped(bytes(builder.Output()))
            try:
                tf = TransformStamped.TransformStamped.GetRootAs(tf_buf)
                x = tf.Transform().Translation().X()
                y = tf.Transform().Translation().Y()
                z = tf.Transform().Translation().Z()
                qx = tf.Transform().Rotation().X()
                qy = tf.Transform().Rotation().Y()
                qz = tf.Transform().Rotation().Z()
                qw = tf.Transform().Rotation().W()
                quaternion = [qw, qx, qy, qz] # w, x, y, z
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
                # Combine translation and rotation into a 4x4 transformation matrix
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = [x, y, z]
                data[index]['transform_matrix'] = transformation_matrix.tolist()
            except Exception as e:
                logger.error(f"Error querying TF for pointcloud {sample['uuid']}: {e}")
                data[index]['tf'] = None
        return data
    
    # TODO run query will be deprecated. Out of date.
    def run_query_images(self, model_name='None'):
        projectUuids = [self._projectid]
        # timeMin = createTimeStamp(self._builder, 1687445582, 0)
        # timeMax = createTimeStamp(self._builder, 1687445586, 0)
        # timeInterval = createTimeInterval(self._builder, timeMin, timeMax)
        queryMsg = util_fb.createQuery(
            self._builder,
            # boundingBox=boundingboxStamped,
            # timeInterval=timeInterval,
            # labels=['RetinaNet'],
            # mustHaveAllLabels=False,
            projectUuids=projectUuids,
            # instanceUuids=instanceUuids,
            # dataUuids=dataUuids,
            # withoutData=False,
            sortByTime=True,  # from version 0.2.5 onwards
        )
        self._builder.Finish(queryMsg)
        buf = self._builder.Output()
        data = []
        for responseBuf in self._grpc_stub.GetImage(bytes(buf)):
            sample = {}
            logger.info('Receiving messages from the SEEREP server')
            response = Image.Image.GetRootAs(responseBuf)
            self._msguuid = response.Header().UuidMsgs().decode("utf-8")
            sample['uuid'] = self._msguuid
            sample['image'] = np.reshape(response.DataAsNumpy(), (response.Height(), response.Width(), -1))[:, :, 0:3] # When more than 3 channels
            sample['image'] = np.ascontiguousarray(sample['image'], dtype=np.uint8).astype(np.uint8)
            sample['timestamp'] = [response.Header().Stamp().Seconds(), response.Header().Stamp().Nanos()]  # seconds nanos
            sample['processed'] = False
            sample['no_grountruth'] = False
            sample['annotations'] = {
                "info": {},
                "categories": {
                    "label": {
                        "labels": [],
                        "attributes": [],
                    },
                    "points": {"items": []},
                },
                "items": [],
            }
            labels: Set[Tuple[str, int]] = set()
            for label_idx in range(response.LabelsLength()):    # Here LabelsLength correspond to number of categories
                category_with_labels = response.Labels(label_idx)
                if(not (category_with_labels.Category().decode() == 'labelGeneral')):
                    item = json.loads(category_with_labels.DatumaroJson().decode())
                    sample['annotations']['categories']["label"]["labels"].append(category_with_labels.Category().decode())
                    sample['annotations']["items"].append(item)
                    for j in range(category_with_labels.LabelsLength()):    # Here LabelsLength correspond to number of labels per category
                        labels.add(
                            (
                                category_with_labels.Labels(j).Label().decode(),
                                category_with_labels.Labels(j).LabelIdDatumaro(),
                            )
                        )
            if model_name in sample['annotations']['categories']['label']['labels']:
                sample['processed']  = True
            if len(sample['annotations']['items'][0]['annotations']) == 0:
                sample['no_grountruth'] = True
            data.append(sample.copy())
        logger.info('Fetched {} images from the current SEEREP project'.format(len(data)))
        if self.vis:
            cv2.destroyWindow(self.source_window)
        return data
    
    # TODO Send dataset should also be on project or data UUID basis. Out of date.
    def send_dataset(self, 
                     data: list[dict], 
                     uuids: list[str], 
                     category: str='yolov5m_coco', 
                     ignore_ground_truth: bool=False):
        """
            Send the previously fetched SEEREP dataset augmented with Datumaro annotations
            using model from triton server under the name category. If ignore_ground_truth is set to True,
            then the predictions will be sent as ground truth annotations. If set to False, then the
            predictions will be sent only if there are ground truth annotations present in the data sample.
            
            Args:
                data (list[dict]): List of dictionaries containing the data samples with ['annotations']
                uuids (list[str]): List of UUIDs of the data samples
                category (str): The name of the model used for predictions
                ignore_ground_truth (bool): If True, detections will be sent even if there are no ground truth annotations.
                                            This can lead to scenario where false positives will be evaluated against ground truth.
                                            If False, detections will be sent only if there are ground truth annotations present.
            Returns:
                str: The UUID of the created SEEREP project.
            """
        data_stub, _, builder = self.secondary_channel()
        query = util_fb.createQuery(
                            builder,
                            dataUuids=uuids,
                            # timeInterval=timeInterval,
                            withoutData=True,
                        )
        builder.Finish(query)
        buffer = builder.Output()
        if self.modality == 'image':
            response_ls: List = list(data_stub.GetImage(bytes(buffer)))
        elif self.modality == 'pointcloud':
            response_ls: List = list(data_stub.GetPointCloud2(bytes(buffer)))
        else:
            logger.error("Cannot create a response buffer for target modality: {}".format(self.modality))
            sys.exit(0)
        if not response_ls:
            logger.error("""
                No samples found. Check if the provided UUIDs in the createQuery are correct. 
            """)
            sys.exit()
        msgToSend = []
        label_list: List[Tuple[str, bytearray]] = []
        for responseBuf in tqdm(response_ls,
                                total=len(data),
                                colour="GREEN",
                                desc="Sending Predictions to SEEREP Server:",
                                unit="predictions"
                                ):
            response = Image.Image.GetRootAs(responseBuf)
            # Fetch the image UUID from the response that we have already previously fetched for inference
            img_uuid = response.Header().UuidMsgs().decode("utf-8")
            labels = []
            # Match the image UUID with the data sample which were inferenced from previous fetch. 
            anns = [sample for sample in data if sample['uuid']==img_uuid][0]
            # This ignore_ground_truth flag is only to be used to send dummy predictions to SEEREP server as ground truth annotations. 
            # DEBUG_ONLY
            if ignore_ground_truth:
                logger.warning("Model predictions will be sent as Ground truth since ignore_ground_truth is set to True")
                if len(anns['annotations']['items']) > 0:
                    for prediction in anns['annotations']['items'][-1]['annotations']:  #last added item is new prediction. TODO double check!
                        labels.append(create_label(builder=builder,
                                                    label='person',
                                                    label_id=int(prediction['label_id']),
                                                    instance_uuid=str(img_uuid),        # TODO The instance uuid and id are optional. keeping it to dummy values to not break things
                                                    instance_id=int(prediction['id'])
                                                    ))
                    labelsCategory = []
                    labelsCategory.append(create_label_category(
                                                builder=builder,
                                                labels=labels,
                                                datumaro_json=json.dumps(anns['annotations']['items'][-1]), # must be json encoded string not a regular string
                                                category=category)) 
                    dataset_uuid_label = create_dataset_uuid_label(builder=builder,
                                                                    projectUuid=anns['project_uuid'],
                                                                    datasetUuid=img_uuid,
                                                                    labels=labelsCategory)
                    builder.Finish(dataset_uuid_label)
                    buf = builder.Output()
                    label_list.append((img_uuid,buf))
                    msgToSend.append(bytes(buf))
                # Ground truth found but no predictions were generated by the model aka 'category'
                else:
                    logger.info("Skipping image with UUID: {} since no predictions were generated by the current model {}".format(img_uuid, category))
            else:
                # Predicted and sent to SEEREP already from a previous run --> DONT SEND TO SEEREP
                if category in anns['processed']:
                    logger.info("Skipping image with UUID: {}. Already processed by Model: {} from previous requests".format(img_uuid, category))
                    pass
                # Not predicted and not sent to SEEREP --> SEND DATA TO SEEREP
                else:
                    if len(anns['annotations']['items'][-1]['annotations']) > 0:
                        for prediction in anns['annotations']['items'][-1]['annotations']:  #last added item is new prediction. TODO double check!
                            labels.append(create_label(builder=builder,
                                                        label='person',
                                                        label_id=int(prediction['label_id']),
                                                        instance_uuid=str(img_uuid),        # TODO The instance uuid and id are optional. keeping it to dummy values to not break things
                                                        instance_id=int(prediction['id'])
                                                        ))
                        labelsCategory = []
                        labelsCategory.append(create_label_category(
                                                    builder=builder,
                                                    labels=labels,
                                                    datumaro_json=json.dumps(anns['annotations']['items'][-1]), # must be json encoded string not a regular string
                                                    category=category)) 
                        dataset_uuid_label = create_dataset_uuid_label(builder=builder,
                                                                        projectUuid=anns['project_uuid'],
                                                                        datasetUuid=img_uuid,
                                                                        labels=labelsCategory)
                        builder.Finish(dataset_uuid_label)
                        buf = builder.Output()
                        label_list.append((img_uuid,buf))
                        msgToSend.append(bytes(buf))
                    else:
                        logger.info("Generating dummy predictions since nothing detected by the current model {}".format(img_uuid, category))
                        labels.append(create_label(builder=builder,
                                                        label='person',
                                                        label_id=int(1000),
                                                        instance_uuid=str(img_uuid),        # TODO The instance uuid and id are optional. keeping it to dummy values to not break things
                                                        instance_id=int(1000),
                                                        ))
                        labelsCategory = []
                        labelsCategory.append(create_label_category(
                                                    builder=builder,
                                                    labels=labels,
                                                    datumaro_json=json.dumps(anns['annotations']['items'][-1]), # must be json encoded string not a regular string
                                                    category=category)) 
                        dataset_uuid_label = create_dataset_uuid_label(builder=builder,
                                                                        projectUuid=anns['project_uuid'],
                                                                        datasetUuid=img_uuid,
                                                                        labels=labelsCategory)
                        builder.Finish(dataset_uuid_label)
                        buf = builder.Output()
                        label_list.append((img_uuid,buf))
                        msgToSend.append(bytes(buf))
        try:
            if len(msgToSend) != 0:
                data_stub.AddLabels(iter(msgToSend))
            else:
                logger.warning("No predictions to send to SEEREP server. Skipping...")
        except grpc.RpcError as e:
            logger.error(f"Failed to send labels to SEEREP server: {e}")
            return False
        return True


def main():
    model_name = 'retina_big'
    project_name = 'EV41_Kleidung_Sonnenbrille_DunkleCap_225Deg_2024-10-10-18-58-13_0'
    seerep_channel = SeerepEndpoint(
            endpoint_url='agrigaia-ur.ni.dfki:9090',
            modality='image',
            visualize=True,
        )
    project_uuid = seerep_channel.get_project_uuid(project_name)
    data = seerep_channel.fetch_data_by_project([project_uuid], 
                                                model_name=model_name)
    uuids = seerep_channel.fetch_uuids_by_project_uuid([project_uuid])
    data = seerep_channel.fetch_data_by_sample(uuids, 
                                                model_name=model_name)
    print('uuids')
        
if __name__ == "__main__":
    main()
