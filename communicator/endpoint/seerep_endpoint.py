from logger import Client_logger, TqdmToLogger
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

import os
import sys
import cv2
import json
import numpy as np
import struct
import flatbuffers
import grpc
import uuid
import logging
from tqdm import tqdm
from copy import copy
from typing import Set, Tuple, List
from scipy.spatial.transform import Rotation as R

from seerep.fb import (
    Boundingbox,
    Empty,
    Header,
    Image,
    Point,
    ProjectInfos,
    Query,
    TimeInterval,
    Timestamp,
    TRANSMISSION_STATE,
    LabelCategory
)
from seerep.fb import PointCloud2 as pc2
from seerep.fb import image_service_grpc_fb as imageService
from seerep.fb import point_cloud_service_grpc_fb as pointCloudService
from seerep.fb import meta_operations_grpc_fb as metaOperations
from seerep.util import fb_helper as util_fb
from seerep.fb.ServerResponse import ServerResponse
from seerep.util.fb_helper import (
    create_dataset_uuid_label,
    create_label,
    create_label_category,
    createEmpty,
    createTimeStamp,
    createTimeInterval,
    createQuery,
)
# from visual_utils import open3d_vis_utils as visualizer
from utils import cxcy2xyxy

logger = Client_logger(name='SEEREP-Client', level=logging.ERROR).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)

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
                 visualize=False):
        self._grpc_stub = None
        self._grpc_stubmeta = None
        self._builder = None
        self.endpoint_url = endpoint_url
        self.visualize = visualize

        # register and initialise the stub
        self.vis = visualize
        self.modality = modality
        self.intialize_gRPC_stubs()
        # self.ann_dict = self.annotation_dict(format=format)
        if self.visualize:
            self.source_window = 'SEEREP source image'
            cv2.namedWindow(self.source_window)

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
        if self.modality == 'images':
            self._grpc_stub  = imageService.ImageServiceStub(grpc_channel)
        elif self.modality == 'pointclouds':
            self._grpc_stub  = pointCloudService.PointCloudServiceStub(grpc_channel)
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
        return self.process_images(buffer, model_name=model_name)
    
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
        for responseBuf in tqdm(self._grpc_stub.GetImage(bytes(buffer)), desc="Fetching images", unit=" image(s)", colour="blue"):
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
        for responseBuf in tqdm(self._grpc_stub.GetImage(bytes(buffer)), desc="Fetching UUIDs", unit=" uuid(s)", colour="blue"):
            logger.info('Receiving messages from the SEEREP server')
            response = Image.Image.GetRootAs(responseBuf)
            sample_uuid = response.Header().UuidMsgs().decode("utf-8")
            data.append(sample_uuid)
        logger.info('Fetched {} UUIDs from the current SEEREP project'.format(len(data)))
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
