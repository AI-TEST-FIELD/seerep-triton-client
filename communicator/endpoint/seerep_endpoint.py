import sys
import numpy as np
import struct
import logging
import yaml
import grpc
import open3d as o3d
import flatbuffers
import tritonclient.grpc.model_config_pb2 as mc
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
        if self.visualize:
            self.source_window = 'SEEREP source image'
            cv2.namedWindow(self.source_window)
        logger.info("SEEREP Channel initialized successfully via endpoint : {} ".format(self.endpoint_url))

    def register_grpc_channel(self):
        """
        Returns:
            A gRPC stub for the SEEREP server
        """
        channel = grpc.insecure_channel(self.endpoint_url) 
        return channel

    def intialize_gRPC_stubs(self):
        """
         register SEEREP gRPC channel
         endpoint_url: String or IP address and PORT of the SEEREP server
         e.g. agrigaia-ur.ni.dfki:9090
        """
        grpc_channel = self.register_grpc_channel()
        if self.modality == 'image':
            self._grpc_stub  = imageService.ImageServiceStub(grpc_channel)
        elif self.modality == 'pointcloud':
            self._grpc_stub  = pointCloudService.PointCloudServiceStub(grpc_channel)
        else:
            logger.error("Modality not supported. Please use image or pointcloud")
            sys.exit(0)
        self._grpc_stubmeta = metaOperations.MetaOperationsStub(grpc_channel)
        self._builder = self.init_builder()

    def secondary_channel(self):
        """
         Establish another channel for sending data to SEEREP server
         register gRPC SEEREP channel
         socket: String, Port and IP address of seerep server
         seerep.robot.10.249.3.13.nip.io:32141
        """
        grpc_channel = self.register_grpc_channel()
        grpc_stub  = imageService.ImageServiceStub(grpc_channel)
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
        tf_stub  = tfService.TfServiceStub(self.channel)
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
        return self.process_images(buffer, model_name=model_name, )
    
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
    
    def fetch_data_by_project(self, project_uuids: list[str], model_name: str)->dict:
        '''
        Fetches data from SEEREP server based on the project_uuids. It also checks if the predictions for the model_name
        have already been generated for the data samples. If yes, then sets 'processed' flag to True.
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
        return self.process_images(buffer, model_name=model_name)

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

    def process_images(self, buffer, model_name)->dict:
        '''
        buffer: flatbuffer buffer containing the query message 
        generated using the SEEREP createQuery function
        model_name: name of the model for which the predictions are to be generated.
        Returns a list of dictionaries containing the data samples with the following
        keys: 'uuid', 'image', 'timestamp', 'processed', 'no_grountruth', 'annotations'
        '''
        data = []
        if self.modality == 'image':
            data_generator = self._grpc_stub.GetImage(bytes(buffer))
        elif self.modality == 'pointcloud':
            data_generator = self._grpc_stub.GetPointCloud2(bytes(buffer))
        for responseBuf in tqdm(data_generator, desc="Fetching images", unit=" image(s)", colour="blue"):
            sample = {}
            response = Image.Image.GetRootAs(responseBuf)
            msguuid = response.Header().UuidMsgs().decode("utf-8")
            sample['uuid'] = msguuid
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
            # This condition makes sure we do not predict the labels twice and send them back again to SEEREP
            if model_name in sample['annotations']['categories']['label']['labels']:
                sample['processed']  = True
            if len(sample['annotations']['items'][0]['annotations']) == 0:
                sample['no_grountruth'] = True
            data.append(sample.copy())
        logger.info('Fetched {} images from the current SEEREP project'.format(len(data)))
        return data
        
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
        elif self.modality == 'pointcloud':
            data_generator = self._grpc_stub.GetPointCloud2(bytes(buffer))
        else:
            logger.error("Modality not supported. Please use image or pointcloud")
            sys.exit(0)
        for responseBuf in tqdm(data_generator, desc="Fetching UUIDs", unit=" uuid(s)", colour="blue"):
            logger.info('Receiving messages from the SEEREP server')
            response = Image.Image.GetRootAs(responseBuf)
            sample_uuid = response.Header().UuidMsgs().decode("utf-8")
            data.append(sample_uuid)
        logger.info('Fetched {} UUIDs from the current SEEREP project'.format(len(data)))
        return data
    
    def preprocess_pc(self, pointcloud_data: dict) -> dict:
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
        if 'kitti' in self.model_name:
            dataset_translation = [0.0, 0.0, -1.026558971]
        # TODO add nuscenes translation
        elif 'nuscenes' in self.model_name: 
            dataset_translation = [0.0, 0.0, -1.026558971]  
        else:
            logger.error(f"Dataset {self.model_name} not supported. Please use kitti or nuscenes.")
            return None
        for sample in tqdm(pointcloud_data,
                                    desc='Transforming pointclouds',
                                    unit="samples",
                                    colour='YELLOW',
                                    total=len(pointcloud_data)):
            # check if the sample has a tf
            if sample['transform_matrix'] is not None:
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
            raw_o3d_pcd = pcd_ros_to_o3d(ros_pcd=sample['point_cloud'], feature_field=feature_field)
            preprocessed_o3d_pcd = raw_o3d_pcd.transform(sensor_to_robot_base_transform)
            preprocessed_o3d_pcd = preprocessed_o3d_pcd.translate(
                robot_base_to_train_dataset_translation
            )
            preprocessed_np_pcd = pcd_o3d_to_numpy(
                o3d_pcd=preprocessed_o3d_pcd, feature_field=feature_field
            )
            preprocessed_np_pcd[:, 3] /= MAX_FEATURE_VALUE
            pointcloud_data[pointcloud_data.index(sample)]['point_cloud_processed'] = preprocessed_np_pcd
            
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
        frames = self.run_query_tf_frames(self._projectid, self.channel)
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
                # frame=sample['sensor_name'],
                frame=parent_frame,
                projectUuid=self._projectid
            )
            if parent_frame in frames:
                tf_query = createTransformStampedQuery(
                    builder=builder,
                    header=header,
                    # childFrameId=parent_frame,  
                    childFrameId=sample['sensor_name'],  # base_link
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
            # tmp = sample['image']
            # if sample['image'].shape[2] == 4:
            #     tmp = cv2.cvtColor(sample['image'], cv2.COLOR_A2BGR)
            #     # tmp = tmp[:, :, 0:3]    # ignore last channel for visualization
            # elif sample['image'].shape[2] == 3:
            #     tmp = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR)
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
                # For DEBUG
            #     if self.vis and len(sample['annotations']['items'][0]['annotations']) != 0:
            #         for ann_idx, ann in enumerate(sample['annotations']['items'][0]['annotations']):
            #             bbox = sample['annotations']['items'][0]['annotations'][ann_idx]['bbox']
            #             bbox = cxcy2xyxy(bbox)
            #             label = int(sample['annotations']["items"][0]['annotations'][ann_idx]['id'])
            #             cv2.rectangle(tmp,
            #                             (bbox[0], bbox[1]),
            #                             (bbox[2], bbox[3]),
            #                             (255, 0, 0), 2)
            #             (tw, th), _ = cv2.getTextSize(self.ann_dict[label], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            #             cv2.rectangle(tmp,
            #                             (bbox[0], bbox[1] - 25),
            #                             (bbox[0] + tw, bbox[1]),
            #                             (255, 0, 0), -1)
            #             cv2.putText(tmp,
            #                         self.ann_dict[label],
            #                         (bbox[0], bbox[1] - 5),
            #                         cv2.FONT_HERSHEY_SIMPLEX,
            #                         0.9, (255,255,255), 2)
            # if self.vis:
            #     cv2.imshow(self.source_window, tmp)
            #     cv2.waitKey(0)
            #     tmp = None
            # This condition makes sure we do not predict the labels twice and send them back again to SEEREP
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
    def send_dataset(self, data, category):
        """
            Send a Datumaro dataset to SEEREP.

            Args:
                project_name (str): Name for the SEEREP project to create from the dataset.
                dataset_path (str, optional): Path to the Datumaro dataset base directory \
                    (one level up from the 'images' directory). Defaults to the current directory.

            Returns:
                str: The UUID of the created SEEREP project.

            Raises:
                FileNotFoundError: If the Datumaro dataset base directory does not exist.
                APIError: If there is an error sending the dataset.

            """
        image_stub, grpc_stubmeta, builder, projectid = self.secondary_channel()
        query = util_fb.createQuery(
                            builder,
                            projectUuids=[projectid],
                            # timeInterval=timeInterval,
                            withoutData=True,
                        )
        builder.Finish(query)
        buffer = builder.Output()
        response_ls: List = list(image_stub.GetImage(bytes(buffer)))
        if not response_ls:
            print("""
                No images found. Please create a project with labeled images
                using gRPC_pb_sendLabeledImage.py first.
            """)
            sys.exit()
        msgToSend = []
        label_list: List[Tuple[str, bytearray]] = []
        for responseBuf in tqdm(response_ls,
                                total=len(data),
                                colour="GREEN",
                                desc="Sending Predictions to SEEREP Server:",
                                unit="predictions",
                                ascii=True):

            response = Image.Image.GetRootAs(responseBuf)
            img_uuid = response.Header().UuidMsgs().decode("utf-8")
            labels = []
            anns = [sample for sample in data if sample['uuid']==img_uuid][0]
            category_groundtruth_index = anns['annotations']['categories']['label']['labels'].index('groundtruth')
            # No objects exist in the current image according to ground truth
            if len(anns['annotations']['items'][category_groundtruth_index]['annotations']) == 0:
                pass    # TODO what if no ground truth but the box was detected?
            # There are gt annotations in the image
            else:
                # Ground truth exists AND predicted as well already from a previous run
                if anns['processed'] == True:
                    pass
                # Ground truth exists AND predicted in the current run and will be sent back to SEEREP
                else:
                    if len(anns['annotations']['items']) > 1:
                        # category_model_index = anns['annotations']['categories']['label']['labels'].index(category)
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
                                                    category=category))  # TODO Fetch this dynamically with model_name
                        dataset_uuid_label = create_dataset_uuid_label(builder=builder,
                                                                        projectUuid=projectid,
                                                                        datasetUuid=img_uuid,
                                                                        labels=labelsCategory)
                        builder.Finish(dataset_uuid_label)
                        buf = builder.Output()
                        label_list.append((img_uuid,buf))
                        msgToSend.append(bytes(buf))
                    # Ground truth found but no predictions were generated by the model aka 'category'
                    else:
                        pass
        image_stub.AddLabels(iter(msgToSend))
        return label_list


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
