from communicator.channel.base_channel import BaseChannel
from logger import Client_logger, TqdmToLogger
import logging
#from base_channel import BaseChannel

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

# import os
import sys
# import cv2
import numpy as np
import struct
# import uuid
import yaml
from tqdm.auto import tqdm
# from typing import List
from copy import copy
from scipy.spatial.transform import Rotation as R
import flatbuffers
import grpc
import open3d as o3d
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
    Timestamp
)
 
from seerep.fb import (
    PointCloud2 as pc2,
    image_service_grpc_fb as imageService,
    point_cloud_service_grpc_fb as pointCloudService,
    meta_operations_grpc_fb as metaOperations,
    tf_service_grpc_fb as tfService,
    TransformStamped
)

from seerep.util.fb_helper import (
    createHeader,
    createTimeStamp,
    createTransformStampedQuery,
    createQuery
    )
from seerep.util.common import get_gRPC_channel
from visual_utils import open3d_vis_utils as visualizer
from tools.pointcloud import (
    pcd_o3d_to_numpy,
    pcd_ros_to_o3d,
)

logger = Client_logger(name='SEEREP-Client', level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)
O3D_DEVICE = o3d.core.Device("CPU:0")  # can also be set to GPU

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

class SEEREPChannel():
    """
    A SEEREPChannel is establishes a connection between the triton client and SEEREP.
    """
    def __init__(self, 
                 project_name='testproject', 
                 endpoint='agrigaia-ur.ni.dfki:9090', 
                 modality='image',
                 format='coco',
                 visualize=False):
        
        self._meta_data = {}
        self._grpc_stub = None
        self._grpc_stubmeta = None
        self._builder = None
        self._projectid = None
        self._msguuid = None
        self.endpoint = endpoint
        self.projname = project_name
        self.normalized_coors = False
        self.visualize = visualize

        # register and initialise the stub
        self.channel = self.make_channel()
        self.vis = visualize
        self.modality = modality
        self.register_channel()
        self.ann_dict = self.annotation_dict(format=format)

    def make_channel(self):
        '''
        Make a channel to the SEEREP server
        at the given socket address
        Returns:
            channel: grpc channel to the SEEREP server
        '''
        channel = grpc.insecure_channel(self.endpoint) # use with local deployment
        return channel
    
    def register_channel(self):
        """
         register all the gRPC stubs for data modalities, meta operations and tf service
         and fetch the project id from the SEEREP server
        """
        if self.modality == 'image':
            self._grpc_stub  = imageService.ImageServiceStub(self.channel)
        elif self.modality == 'pointcloud':
            self._grpc_stub  = pointCloudService.PointCloudServiceStub(self.channel)
        self._tf_stub = tfService.TfServiceStub(self.channel)
        self._grpc_stubmeta = metaOperations.MetaOperationsStub(self.channel) 
        self._builder = self.init_builder()
        self._msgUuid = None
        self._projectid = self.retrieve_project(self.projname, log=True)

    def secondary_channel(self):
        """
         Establish another channel for sending
         register grpc triton channel
         socket: String, Port and IP address of seerep server
         seerep.robot.10.249.3.13.nip.io:32141
        """
        grpc_stub  = imageService.ImageServiceStub(self.channel)
        grpc_stubmeta = metaOperations.MetaOperationsStub(self.channel)  
        builder = self.init_builder()
        projectid = self._projectid

        return (grpc_stub, grpc_stubmeta, builder, projectid)
    
    def tf_channel(self):
        """
         Establish a channel for querying TFs
        """
        tf_stub  = tfService.TfServiceStub(self.channel)
        builder = self.init_builder()

        return (tf_stub, builder)

    def fetch_channel(self):
        """
        return grpc stub
        """
        return self._grpc_stub

    def _grpc_metadata(self):
        """
        TODO Figure out if this is needed for SEEREP
        Initiate all meta data required for models
        """
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        self._meta_data["metadata_request"] = service_pb2.ModelMetadataRequest(
            name=self.FLAGS.model_name, version=self.FLAGS.model_version)
        self._meta_data["metadata_response"] = self._grpc_stub.ModelMetadata(self._meta_data["metadata_request"])

        self._meta_data["config_request"] = service_pb2.ModelConfigRequest(name=self.FLAGS.model_name,
                                                                           version=self.FLAGS.model_version)
        self._meta_data["config_response"] = self._grpc_stub.ModelConfig(self._meta_data["config_request"])

        # set
        self._set_grpc_members()

    def get_metadata(self):
        """
        return meta_data dictionary form
        @rtype: dictionary
        """
        return self._meta_data

    def _set_grpc_members(self):
        """
        set essential grpc members
        """
        self.input = service_pb2.ModelInferRequest().InferInputTensor()
        self.request = service_pb2.ModelInferRequest()
        self.request.model_name = self.FLAGS.model_name
        self.request.model_version = self.FLAGS.model_version
        self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()

    def perform_inference(self):
        """
        inference based on grpc_stud
        @return: inference of grpc
        """
        return self._grpc_stub.ModelInfer(self.request)

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

    def retrieve_project(self, projname, log=False):
        '''
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
        # TODO here we should already receive a list of project UUIDs directly from the mastermind MM. This is redundant atm. 
        for i in range(response.ProjectsLength()):
            if log==True:
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
        if projname in projects:
            curr_proj = projname
            projectuuid = projects[curr_proj]
            logger.info("Found project {} with UUID: {}".format(curr_proj, projectuuid))
            return projectuuid
        else:
            logger.error("The requested project \n {} is not available on the SEEREP Server! Note that project names are case-sensitive! Please select a project from the list displayed above!".format(projname, ))
            sys.exit(0)

    def init_builder(self):
        '''
        Initialize a flatbuffers builder
        Returns:
            fb_builder: flatbuffers builder
        '''
        fb_builder = flatbuffers.Builder(1024)
        
        return fb_builder
    
    def annotation_dict(self, format='coco'):
        '''
        Initialize a dictionary of class names and their corresponding indices
        Returns:
            anns_dict: dictionary of class names and their corresponding indices
        '''
        anns_dict = {}
        class_names= []
        if format == 'coco':
            filepath = 'config/coco.names'
        elif format == 'kitti':
            filepath = 'config/kitti.names'
        elif format == 'crop':
            filepath = 'config/crop.names'
        with open(filepath, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        for id,idx in zip(class_names, range(len(class_names))):
            anns_dict[id] = idx+1

        return anns_dict

    def run_query_images(self, *args):
        pass
    
    def run_query_pointclouds(self, model_name:str) -> list[dict]:
        """
        Query the pointclouds from the SEEREP server
        Returns:
            data: list of dictionaries containing pointcloud data
        """
        self.model_name = model_name
        projectUuids = [self._projectid]
        queryMsg = createQuery(
            self._builder,
            # boundingBox=boundingboxStamped,
            # timeInterval=timeInterval,
            # labels=labelCategory,
            # mustHaveAllLabels=False,
            projectUuids=projectUuids,
            # instanceUuids=instanceUuids,
            # dataUuids=dataUuids,
            withoutData=False,
            sortByTime=True,  # from version 0.2.5 onwards
        )
        self._builder.Finish(queryMsg)
        buf = self._builder.Output()
        # Collect list of all data samples returned from SEEREP into data
        data = []
        # Collect the UUIDs and data from each sample sent by SEEREP project. 
        sample = {}
        # TODO the num_samples should be replaced by the total number of samples inside the seerep project
        num_samples = 10
        for responseBuf, curr_sample in tqdm(zip(self._grpc_stub.GetPointCloud2(bytes(buf)),
                                    range(num_samples)),
                                    total=num_samples,
                                    colour='GREEN',
                                    desc='Receiving pointclouds from the SEEREP server',
                                    unit=" samples"):
            response = pc2.PointCloud2.GetRootAs(responseBuf)
            self._msguuid = response.Header().UuidMsgs().decode("utf-8")
            timestamp = response.Header().Stamp().Seconds(), response.Header().Stamp().Nanos()
            height = response.Width()
            width = response.Height()

            sample['uuid'] = self._msguuid
            sample['sensor_name'] = response.Header().FrameId().decode("utf-8")
            sample['timestamp'] = timestamp
            raw_data = response.DataAsNumpy()
            fields = {}
            dtype = None
            for j in range(response.FieldsLength()):
                fields[response.Fields(j).Name().decode('utf-8')] = {}
                fields[response.Fields(j).Name().decode('utf-8')]['offset'] = response.Fields(j).Offset()
                fields[response.Fields(j).Name().decode('utf-8')]['dtype'] = response.Fields(j).Datatype()
                dtype = Point_Field_Datatype[response.Fields(j).Datatype()]
                # https://docs.python.org/2/library/struct.html          
                if dtype == np.int16:    # 16 bit short
                    c = 'h'
                elif dtype == np.uint16:    # 16 bit unsigned-short
                    c = 'H'
                elif dtype == np.int32:    # 32 bit int 
                    c = 'i'
                elif dtype == np.uint32:    # 32 bit unsigned int 
                    c = 'I'
                elif dtype == np.float32:   # 32 bit float
                    c = 'f'
                elif dtype == np.float64:   # 64 bit double 
                    c = 'd'
                else: 
                    print('Invalid data type')
                fields[response.Fields(j).Name().decode('utf-8')]['data_string'] = c
                fields[response.Fields(j).Name().decode('utf-8')]['size'] = struct.calcsize(c) 
            for field in fields:
                strs = list()
                for i in range(fields[field]['offset'], raw_data.shape[0], response.PointStep()):      # Each chunk size must have one entry for each field i.e. x,y,z,intensity, t, reflectivity, ring, ambient, range
                    sb = struct.unpack(fields[field]['data_string'], raw_data[i : i + fields[field]['size']])
                    strs.append(sb)
                fields[field]['data'] = (np.array(strs, dtype=np.object_))
                strs = []
            sample['point_cloud'] = copy(fields) 
            if 'reflectivity' in fields:
                sample['lidar_feature'] = 'reflectivity'
            else:
                sample['lidar_feature'] = 'intensity'
            if self.visualize:
                pc = np.zeros((height*width, 3), dtype=np.float64)
                pc[:, 0] = fields['x']['data'][:, 0]
                pc[:, 1] = fields['y']['data'][:, 0]
                pc[:, 2] = fields['z']['data'][:, 0]
                # ry = R.from_euler('y', 30, degrees=True).as_matrix()
                # rz = R.from_euler('z', 90, degrees=True).as_matrix()
                # pc = np.matmul(ry, pc.T).T
                # pc = np.matmul(rz, pc.T).T
                # pc += [0., 0., -1.026558971]
                visualizer.draw_scenes(pc)
            # Store the sample into data collection
            data.append(sample)
            # flush the sample data for new incoming samples
            sample={}
            curr_sample+=1
            if curr_sample==0:
                break
        logger.info('Fetched {} pointclouds from the current SEEREP project'.format(len(data)))
        data = self.run_query_tf(data)
        data = self.preprocess_pc(data)
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
                frame=sample['sensor_name'],
                projectUuid=self._projectid
            )
            if parent_frame in frames:
                tf_query = createTransformStampedQuery(
                    builder=builder,
                    header=header,
                    childFrameId=parent_frame,  # map_odom odom_base_link
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

def main():
    schan = SEEREPChannel()
    ts = schan.gen_timestamp(1610549273, 1938549273)
    schan.run_query(ti=ts)

if __name__ == "__main__":
    main()

