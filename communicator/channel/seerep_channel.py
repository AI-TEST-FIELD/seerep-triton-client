
from communicator.channel.base_channel import BaseChannel
from logger import Client_logger, TqdmToLogger
import logging
#from base_channel import BaseChannel

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

import os
import sys
import cv2
import numpy as np
import struct
import uuid
from tqdm import tqdm
from copy import copy

import flatbuffers
import grpc
from seerep.fb import BoundingBoxes2DLabeledStamped, Boundingbox, Empty, Header, Image, Point, ProjectInfos, Query, TimeInterval, Timestamp
from seerep.fb import PointCloud2 as pc2

from seerep.fb import image_service_grpc_fb as imageService
from seerep.fb import point_cloud_service_grpc_fb as pointCloudService
from seerep.fb import meta_operations_grpc_fb as metaOperations
# from seerep.util.fb_helper import rosToNumpyDtype

import communicator.channel.util_fb as util_fb
from visual_utils import open3d_vis_utils as visualizer

logger = Client_logger(name='SEEREP-Client', level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)

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
        self.register_channel()
        self.ann_dict = self.annotation_dict(format=format)
        # if ros_test:
        #     rospy.init_node('ros_infer')
        #     self.ros_message = PointCloud2()
        #     self.publisher = rospy.Publisher('/pointcloud2/test_pc', PointCloud2, queue_size=1)

        #self._grpc_metadata() #

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
        # self._grpc_stub  = imageService.ImageServiceStub(self.channel)
        # self._grpc_stubmeta = metaOperations.MetaOperationsStub(self.channel)
        self._grpc_stub = pointCloudService.PointCloudServiceStub(self.channel) 
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

        responseBuf = self._grpc_stubmeta.GetProjects(bytes(buf))
        response = ProjectInfos.ProjectInfos.GetRootAs(responseBuf)
        curr_proj = None
        logger.info("List of available projects on the SEEREP Server")
        for i in range(response.ProjectsLength()):
            if log==True:
                try:
                    tmp = response.Projects(i).Name().decode("utf-8")
                    logger.info(tmp + " " + response.Projects(i).Uuid().decode("utf-8"))
                    if response.Projects(i).Name().decode("utf-8") == projname:
                        projectuuid = response.Projects(i).Uuid().decode("utf-8")
                        curr_proj = tmp
                    else:
                        logger.error("The requested project \n {} is not available on the SEEREP Server! Note that project names are case-sensitive! Please select a project from the list displayed above!".format(projname, ))
                except Exception as e:
                    logger.error(e)
            else:
                try:
                    projectuuid = response.Projects(i).Uuid().decode("utf-8")
                except Exception as e:
                    logger.error(e)
        if curr_proj == None:
            logger.error("No SEEREP projects found on the server. Please check the SEEREP server host!")
            sys.exit(0)
        else:
            logger.info("Found project {} with UUID: {}".format(curr_proj, projectuuid))
            return projectuuid

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

    def gen_boundingbox(self, start_coord, end_coord):
        '''
        Add a bounding box to the query builder
        Args:
            start_coord : An interable of the X, Y and Z start co ordinates of the point, in this order.
            end_coord : An interable of the X, Y and Z end co ordinates of the point, in this order.
        '''
        
        Point.Start(self._builder)
        Point.AddX(self._builder, start_coord[0])
        Point.AddY(self._builder, start_coord[1])
        #Point.AddZ(self._builder, start_coord[2])
        pointMin = Point.End(self._builder)

        Point.Start(self._builder)
        Point.AddX(self._builder, end_coord[0])
        Point.AddY(self._builder, end_coord[1])
        #Point.AddZ(self._builder, end_coord[2])
        pointMax = Point.End(self._builder)

        frameId = self._builder.CreateString("map")
        Header.Start(self._builder)
        Header.AddFrameId(self._builder, frameId)
        header = Header.End(self._builder)

        Boundingbox.Start(self._builder)
        Boundingbox.AddPointMin(self._builder, pointMin)
        Boundingbox.AddPointMax(self._builder, pointMax)
        #Boundingbox.AddHeader(self._builder, header)
        boundingbox = Boundingbox.End(self._builder)

        #Query.AddBoundingbox(self._builder, boundingbox)
        return boundingbox

    def gen_boundingbox2dlabeledstamped (self, boundingBoxes):

        projuuid_str = self.string_to_fbmsg(self._projectid)
        msguuid_str = self.string_to_fbmsg(self._msguuid)

        # build header
        Header.Start(self._builder)
        Header.AddUuidProject(self._builder, projuuid_str)
        Header.AddUuidMsgs(self._builder, msguuid_str)
        header = Header.End(self._builder)

        # a labels_bb array which will hold all the bbs
        label_bbs = []

        # create bounding box(es)
        for bb in boundingBoxes:
            x = self.gen_boundingbox(bb[0], bb[1])
            label_bbs.append(x)

        BoundingBoxes2DLabeledStamped.StartLabelsBbVector(self._builder, len(label_bbs))
        for bb in reversed(label_bbs):
            self._builder.PrependUOffsetTRelative(bb)
        self._builder.EndVector()

        # Start a new bounding box 2d labeled stamped
        BoundingBoxes2DLabeledStamped.Start(self._builder)

        BoundingBoxes2DLabeledStamped.AddHeader(self._builder, header)

        for label_bb in label_bbs:
            BoundingBoxes2DLabeledStamped.AddLabelsBb(self._builder, label_bb)

        return BoundingBoxes2DLabeledStamped.End(self._builder)

    def gen_timestamp(self, starttime, endtime):
        '''
        Add a time range to the query builder
        Args:
            starttime : Start time as an int
            endtime : End time as an int
        '''

        Timestamp.Start(self._builder)
        Timestamp.AddSeconds(self._builder, starttime)
        Timestamp.AddNanos(self._builder, 0)
        timeMin = Timestamp.End(self._builder)

        Timestamp.Start(self._builder)
        Timestamp.AddSeconds(self._builder, endtime)
        Timestamp.AddNanos(self._builder, 0)
        timeMax = Timestamp.End(self._builder)

        TimeInterval.Start(self._builder)
        TimeInterval.AddTimeMin(self._builder, timeMin)
        TimeInterval.AddTimeMax(self._builder, timeMax)
        timeInterval = TimeInterval.End(self._builder)

        #Query.AddTimeinterval(self._builder, timeInterval)
        return timeInterval

    def gen_label(self, label):
        label = builder.CreateString("1")
        Query.StartLabelVector(builder, 1)
        builder.PrependUOffsetTRelative(label)
        labelMsg = builder.EndVector()

        #Query.AddLabel(builder, labelMsg)
        return labelMsg

    def run_query(self, **kwargs):
        projectuuidString = self._builder.CreateString(self._projectid)
        Query.StartProjectuuidVector(self._builder, 1)
        self._builder.PrependUOffsetTRelative(projectuuidString)
        projectuuidMsg = self._builder.EndVector()

        Query.Start(self._builder)
        Query.AddProjectuuid(self._builder, projectuuidMsg)
        # Query.AddWithoutdata
        for key, value in kwargs.items():
            if key == "bb": Query.AddBoundingbox(self._builder, value)
            if key == "ti": Query.AddTimeinterval(self._builder, value)

        queryMsg = Query.End(self._builder)

        self._builder.Finish(queryMsg)
        buf = self._builder.Output()
        
        data = []
        sample = {}

        for responseBuf in self._grpc_stub.GetImage(bytes(buf)):
            print('[INFO] Receiving images . . .')
            response = Image.Image.GetRootAs(responseBuf)

            # this should not be inside the loop
            self._msguuid = response.Header().UuidMsgs().decode("utf-8")
            sample['uuid'] = self._msguuid
            sample['image'] = np.reshape(response.DataAsNumpy(), (response.Height(), response.Width(), 3))

            # Uncomment to visualize images
            # import matplotlib.pyplot as plt
            # plt.imshow(np.reshape(response.DataAsNumpy(), (response.Height(), response.Width(), 3)))
            # plt.show()
            # TODO why are the bounding boxes empty?????
            nbbs = response.LabelsBbLength()
            sample['boxes'] = nbbs
            for category in range(response.LabelsBbLength()):
                logger.info("Category name: {}".format(response.LabelsBb(category).Category().decode("utf-8")))
                for x in range(response.LabelsBb(0).BoundingBox2dLabeledLength()):
                    logger.info(f"uuidmsg: {response.Header().UuidMsgs().decode('utf-8')}")
                    logger.info("first label: " + response.LabelsBb(0).BoundingBox2dLabeled(x).LabelWithInstance().Label().Label().decode("utf-8") 
                        + " ; confidence: " 
                        + str(response.LabelsBb(0).BoundingBox2dLabeled(x).LabelWithInstance().Label().Confidence())
                        )
                    logger.info(
                        "bounding box number (Xcenter,Ycenter,Xextent,Yextent):"
                        + str(response.LabelsBb(0).BoundingBox2dLabeled(x).BoundingBox().CenterPoint().X())
                        + " "
                        + str(response.LabelsBb(0).BoundingBox2dLabeled(x).BoundingBox().CenterPoint().Y())
                        + " "
                        + str(response.LabelsBb(0).BoundingBox2dLabeled(x).BoundingBox().SpatialExtent().X())
                        + " "
                        + str(response.LabelsBb(0).BoundingBox2dLabeled(x).BoundingBox().SpatialExtent().Y())
                        + "\n"
                    )
            data.append(sample)
            sample={}
        return data
    
    def annotation_dict(self, format='coco'):
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

    def unpack_point_fields(self, point_cloud: pc2.PointCloud2) -> dict:
        """Extract the point fields from a Flatbuffer pcl message"""
        return {
            "name": [point_cloud.Fields(i).Name().decode("utf-8") for i in range(point_cloud.FieldsLength())],
            "datatype": [point_cloud.Fields(i).Datatype() for i in range(point_cloud.FieldsLength())],
            "offset": [point_cloud.Fields(i).Offset() for i in range(point_cloud.FieldsLength())],
            "count": [point_cloud.Fields(i).Count() for i in range(point_cloud.FieldsLength())],
        }
    
    def publish_pc(self, response, fields):
        import std_msgs.msg
        import sensor_msgs.point_cloud2 as pcl2

        self.ros_message.height = response.Height()
        self.ros_message.width = response.Width()
        self.ros_message.is_bigendian = response.IsBigendian()
        self.ros_message.is_dense = response.IsDense()
        self.ros_message.row_step = response.RowStep()
        for f in fields:
            if fields[f]['dtype'] == 7:
                dtype = PointField.FLOAT32
            elif fields[f]['dtype'] == 4:
                dtype = PointField.UINT16
            elif fields[f]['dtype'] == 6:
                dtype = PointField.UINT32
            self.ros_message.fields.append(PointField(
                name=f,
                offset=fields[f]['offset'],
                datatype=dtype, 
                count=1
            ))
        self.ros_message.data = response.DataAsNumpy()
        # Initialize empty header
        header = std_msgs.msg.Header()
        header.seq = response.Header().Seq()
        header.stamp = rospy.Time.now()
        header.frame_id = response.Header().FrameId().decode("utf-8")
        self.ros_message.header = header
        for i in range(10000):
            self.publisher.publish(self.ros_message)

    def run_query_images(self, *args):
        projectuuidString = self._builder.CreateString(self._projectid)
        Query.StartProjectuuidVector(self._builder, 1)
        self._builder.PrependUOffsetTRelative(projectuuidString)
        projectuuidMsg = self._builder.EndVector()
        projectUuids = [projectuuidString]
        categories = ['ground_truth']
        # labels = [[util_fb.createLabelWithConfidence(self._builder, "person"), 
        #         #    util_fb.createLabelWithConfidence(self._builder, "weather_general_sun"),
        #         #    util_fb.createLabelWithConfidence(self._builder, "weatherGeneral_cloudy"),
        #         #    util_fb.createLabelWithConfidence(self._builder, "weatherGeneral_rain"),
        #          ]]
        labels = [[util_fb.createLabelWithConfidence(self._builder, semantic) for semantic in args[0]]]
        labelCategory = util_fb.createLabelWithCategory(self._builder, categories, labels)
        queryMsg = util_fb.createQuery(
            self._builder,
            # boundingBox=boundingboxStamped,
            # timeInterval=timeInterval,
            # labels=labelCategory,
            # mustHaveAllLabels=False,
            projectUuids=projectUuids,
            # instanceUuids=instanceUuids,
            # dataUuids=dataUuids,
            withoutData=True,
        )
        self._builder.Finish(queryMsg)
        buf = self._builder.Output()
        data = []
        sample = {}
        category = 1 # ground_truth
        for responseBuf in self._grpc_stub.GetImage(bytes(buf)):
            logger.info('Receiving messages from the SEEREP server')
            response = Image.Image.GetRootAs(responseBuf)
            self._msguuid = response.Header().UuidMsgs().decode("utf-8")
            sample['uuid'] = self._msguuid
            sample['image'] = np.reshape(response.DataAsNumpy(), (response.Height(), response.Width(), 3))
            tmp = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR)
            sample['boxes'] = []
            # for category in range(response.LabelsBbLength()):
            for j in range(response.LabelsBb(0).BoundingBox2dLabeledLength()):
                label = response.LabelsBb(0).BoundingBox2dLabeled(j).LabelWithInstance().Label().Label().decode("utf-8")
                confidence = np.float16(response.LabelsBb(0).BoundingBox2dLabeled(j).LabelWithInstance().Label().Confidence())
                x, y = response.LabelsBb(0).BoundingBox2dLabeled(j).BoundingBox().CenterPoint().X(), response.LabelsBb(0).BoundingBox2dLabeled(j).BoundingBox().CenterPoint().Y()
                w, h = response.LabelsBb(0).BoundingBox2dLabeled(j).BoundingBox().SpatialExtent().X(), response.LabelsBb(0).BoundingBox2dLabeled(j).BoundingBox().SpatialExtent().Y()
                x_tl, y_tl = x - (w/2), y - (h/2)
                if x<=1 and y<=1:
                    self.normalized_coors = True 
                    sample['normalized'] = True
                else:
                    sample['normalized'] = False
                    self.normalized_coors = False 
                if sample['normalized'] == False:
                    sample['boxes'].append([x_tl, y_tl, w, h, self.ann_dict[label], confidence])
                else:
                    scale_x, scale_y = sample['image'].shape[1], sample['image'].shape[0]
                    sample['boxes'].append([x_tl * scale_x, y_tl * scale_y, w * scale_x, h * scale_y, self.ann_dict[label], confidence])
                # For DEBUG
                # if self.vis:
            #         cv2.rectangle(tmp, 
            #                       (int(sample['boxes'][j][0]), int(sample['boxes'][j][1])), 
            #                       (int(sample['boxes'][j][0]+sample['boxes'][j][2]), int(sample['boxes'][j][1]+sample['boxes'][j][3])), 
            #                       (255, 0, 0), 2)
            #         (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            #         cv2.rectangle(tmp, 
            #                       (int(sample['boxes'][j][0]), (int(sample['boxes'][j][1]) - 25)), 
            #                       (int(sample['boxes'][j][0] + tw), int(sample['boxes'][j][1])), 
            #                       (255, 0, 0), -1)
            #         cv2.putText(tmp, label, (int(sample['boxes'][j][0]), int(sample['boxes'][j][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            # if self.vis:
            #     winname = 'image'
            #     cv2.namedWindow(winname)
            #     cv2.imshow(winname, tmp)
            #     cv2.moveWindow(winname, 3000,200)
            #     cv2.waitKey(0)
            #     cv2.destroyWindow(winname)    
            #     tmp = None
            data.append(sample)
            sample={}
        logger.info('Fetched {} images from the current SEEREP project'.format(len(data)))
        return data
    
    def run_query_pointclouds(self, *args):
        projectuuidString = self._builder.CreateString(self._projectid)
        Query.StartProjectuuidVector(self._builder, 1)
        self._builder.PrependUOffsetTRelative(projectuuidString)
        projectuuidMsg = self._builder.EndVector()
        projectUuids = [projectuuidString]
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
        )
        self._builder.Finish(queryMsg)
        buf = self._builder.Output()
        # Collect list of all data samples returned from SEEREP into data
        data = []
        # Collect the UUIDs and data from each sample sent by SEEREP project. 
        sample = {}
        # TODO the num_samples should be replaced by the total number of samples inside the seerep project
        num_samples = 2
        for responseBuf, curr_sample in tqdm(zip(self._grpc_stub.GetPointCloud2(bytes(buf)),
                                    range(num_samples)),
                                    total=num_samples,
                                    colour='GREEN',
                                    file=tqdm_out,
                                    desc='Receiving pointclouds from the SEEREP server',
                                    unit="samples"):
            # logger.info('Receiving pointclouds from the SEEREP server')
            response = pc2.PointCloud2.GetRootAs(responseBuf)
            self._msguuid = response.Header().UuidMsgs().decode("utf-8")
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
            if self.visualize:    # TODO visualize flag with open3d
                pc = np.zeros((height*width, 3), dtype=np.float64)
                pc[:, 0] = fields['x']['data'][:, 0]
                pc[:, 1] = fields['y']['data'][:, 0]
                pc[:, 2] = fields['z']['data'][:, 0]
                visualizer.draw_scenes(pc)
            # Store the sample into data collection
            data.append(sample)
            # flush the sample data for new incoming samples
            sample={}
            curr_sample+=1
            if curr_sample==0:
                break
        logger.info('Fetched {} pointclouds from the current SEEREP project'.format(len(data)))
        return data
    
    def sendboundingbox(self, sample, bbs, labels, confidences, model_name):
        header = util_fb.createHeader(
            self._builder,
            projectUuid = self._projectid,
            msgUuid= sample['uuid']
        )

        boundingBoxes = util_fb.createBoundingBoxes2d(
            self._builder,
            [util_fb.createPoint2d(self._builder, bb[0][0], bb[0][1]) for bb in bbs],
            [util_fb.createPoint2d(self._builder, bb[1][0], bb[1][1]) for bb in bbs],
        )
        if model_name == "ground_truth":
            labelWithInstances = util_fb.createLabelsWithInstance(
            self._builder,
            [label for label in labels],
            [1.0 for conf in confidences],
            [str(uuid.uuid4()) for _ in range(len(bbs))],
            )
        else:   
            labelWithInstances = util_fb.createLabelsWithInstance(
            self._builder,
            [label for label in labels],
            [conf for conf in confidences],
            [str(uuid.uuid4()) for _ in range(len(bbs))],
            )
        labelsBb = util_fb.createBoundingBoxes2dLabeled(self._builder, labelWithInstances, boundingBoxes)

        boundingBox2DLabeledWithCategory = util_fb.createBoundingBox2DLabeledWithCategory(
            self._builder, self._builder.CreateString(model_name), labelsBb
        )

        labelsBbVector = util_fb.createBoundingBox2dLabeledStamped(self._builder, header, [boundingBox2DLabeledWithCategory])
        self._builder.Finish(labelsBbVector)
        buf = self._builder.Output()

        msg = [bytes(buf)]

        send_channel, _, _, _ = self.secondary_channel()
        try:
            ret = send_channel.AddBoundingBoxes2dLabeled( iter(msg) )
        except Exception as e:
            logger.error(e)   


    '''
    def sendboundingbox(self, bb):
        send_channel, _, _, _ = self.secondary_channel()

        self._builder.Finish(bb)
        buf = self._builder.Output()
        bufBytes = [ bytes(buf) ]

        ret = send_channel.AddBoundingBoxes2dLabeled( iter(bufBytes) )

        print("[bb service call]" + str(ret.decode()))
    '''

def main():
    schan = SEEREPChannel()
    ts = schan.gen_timestamp(1610549273, 1938549273)
    schan.run_query(ti=ts)

if __name__ == "__main__":
    main()

