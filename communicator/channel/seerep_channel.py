
from communicator.channel.base_channel import BaseChannel
from logger import Client_logger, TqdmToLogger
#from base_channel import BaseChannel
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

logger = Client_logger(name='SEEREP-Client', level=logging.INFO).get_logger()
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

class SEEREPChannel():
    """
    A SEEREPChannel is establishes a connection between the triton client and SEEREP.
    """
    def __init__(self,
                 project_name='testproject',
                 socket='agrigaia-ur.ni.dfki:9090',
                 modality='images',
                 format='coco',
                 visualize=False):

        self._meta_data = {}
        self._grpc_stub = None
        self._grpc_stubmeta = None
        self._builder = None
        self._projectid = None
        self._msguuid = None
        self.socket = socket
        self.projname = project_name
        self.normalized_coors = False
        self.visualize = visualize

        # register and initialise the stub
        self.channel = self.make_channel(secure=False)
        self.vis = visualize
        self.modality = modality
        self.register_channel()
        self.ann_dict = self.annotation_dict(format=format)
        if self.visualize:
            self.source_window = 'SEEREP source image'
            cv2.namedWindow(self.source_window)

    def make_channel (self, secure=False):
        # server with certs
        if secure:
            __location__ = os.path.realpath(
                os.path.join(os.getcwd(), os.path.dirname(__file__)))
            with open(os.path.join(__location__, 'tls.pem'), 'rb') as f:
                root_cert = f.read()
            creds = grpc.ssl_channel_credentials(root_cert)

            channel = grpc.secure_channel(self.socket, creds) # use with non-local deployment

        else:
            channel = grpc.insecure_channel(self.socket) # use with local deployment

        return channel

    def register_channel(self):
        """
         register grpc triton channel
         socket: String, Port and IP address of seerep server
         seerep.robot.10.249.3.13.nip.io:32141
        """
        if self.modality == 'images':
            self._grpc_stub  = imageService.ImageServiceStub(self.channel)
        elif self.modality == 'pointclouds':
            self._grpc_stub  = pointCloudService.PointCloudServiceStub(self.channel)
        # self._grpc_stubmeta = metaOperations.MetaOperationsStub(self.channel)
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
                    if response.Projects(i).Name().decode("utf-8") == projname:
                        curr_proj = tmp
                        projectuuid = response.Projects(i).Uuid().decode("utf-8")
                except Exception as e:
                    logger.error(e)
            else:
                try:
                    projectuuid = response.Projects(i).Uuid().decode("utf-8")
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

    def string_to_fbmsg (self, projectuuid):
        projectuuidString = self._builder.CreateString(projectuuid)
        '''
        Query.StartProjectuuidVector(self._builder, 1)
        self._builder.PrependUOffsetTRelative(projectuuidString)
        projectuuidMsg = self._builder.EndVector()

        return projectuuidMsg
        '''
        return projectuuidString

    def init_builder(self):
        builder = flatbuffers.Builder(1024)

        return builder

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

    def unpack_point_fields(self, point_cloud: pc2.PointCloud2) -> dict:
        """Extract the point fields from a Flatbuffer pcl message"""
        return {
            "name": [point_cloud.Fields(i).Name().decode("utf-8") for i in range(point_cloud.FieldsLength())],
            "datatype": [point_cloud.Fields(i).Datatype() for i in range(point_cloud.FieldsLength())],
            "offset": [point_cloud.Fields(i).Offset() for i in range(point_cloud.FieldsLength())],
            "count": [point_cloud.Fields(i).Count() for i in range(point_cloud.FieldsLength())],
        }

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
            # projectUuids=projectUuids,
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

    def run_query_pointclouds(self, *args):
        projectUuids = [self._projectid]
        # projectuuidString = self._builder.CreateString(self._projectid)
        # Query.StartProjectuuidVector(self._builder, 1)
        # self._builder.PrependUOffsetTRelative(projectuuidString)
        # projectuuidMsg = self._builder.EndVector()
        # projectUuids = [projectuuidString]
        queryMsg = util_fb.createQuery(
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
        num_samples = 1E10
        curr_sample = 0
        for responseBuf in self._grpc_stub.GetPointCloud2(bytes(buf)):
            # logger.info('Receiving pointclouds from the SEEREP server')
            response = pc2.PointCloud2.GetRootAs(responseBuf)
            self._msguuid = response.Header().UuidMsgs().decode("utf-8")
            seconds = response.Header().Stamp().Seconds()
            nanos = response.Header().Stamp().Nanos()
            height = response.Width()
            width = response.Height()
            # TODO change the decoding to numpy for cleaner implementation
            # point_fields = self.unpack_point_fields(response)
            # dtypes = np.dtype(
            #     {
            #         "names": point_fields["name"],
            #         "formats": [Point_Field_Datatype[datatype] for datatype in point_fields['datatype']],
            #         "offsets": point_fields["offset"],
            #         "itemsize": response.PointStep(),
            #     }
            # )
            # decoded_payload = np.frombuffer(response.DataAsNumpy(), dtype=dtypes)
            # reshaped_data = np.reshape(decoded_payload, (response.Height(), response.Width()))

            sample['uuid'] = self._msguuid
            sample['time'] = {'seconds' : seconds, 'nanos': nanos}
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
            if False:
                from math import sin, cos
                angle=15
                pc = np.zeros((height*width, 3), dtype=np.float64)
                pc[:, 0] = fields['x']['data'][:, 0]
                pc[:, 1] = fields['y']['data'][:, 0]
                pc[:, 2] = fields['z']['data'][:, 0]
                ry = R.from_euler('y', 30, degrees=True).as_matrix()
                rz = R.from_euler('z', 90, degrees=True).as_matrix()
                rotation_matrix = np.array([[cos(angle), 0, sin(angle)],
                                [0, 1, 0],
                                [-sin(angle), 0, cos(angle)]])
                # rotation_matrix = np.array([[0.82638931, -0.02497454,  0.56254509],
                #                             [0.01212522,  0.99957356,  0.02656451],
                #                             [-0.56296864, -0.01513165, 0.82633973]])
                pc = np.matmul(ry, pc.T).T
                pc = np.matmul(rz, pc.T).T
                pc += [0., 0., -1.026558971]
                # pc = r.apply(pc)
                visualizer.draw_scenes(pc)
            # Store the sample into data collection
            data.append(sample)
            # flush the sample data for new incoming samples
            sample={}
            curr_sample+=1
            if curr_sample>=num_samples:
                break
        logger.info('Fetched {} pointclouds from the current SEEREP project'.format(len(data)))
        return data

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
        # timeMin = createTimeStamp(builder, 1687445582, 0)
        # timeMax = createTimeStamp(builder, 1687445586, 0)
        # timeInterval = createTimeInterval(builder, timeMin, timeMax)
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
    schan = SEEREPChannel()

if __name__ == "__main__":
    main()
