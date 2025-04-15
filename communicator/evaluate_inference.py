# import time
import cv2
import imutils
import time
import numpy as np
import sys
import logging
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import open3d as o3d

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

from tools.pointcloud import (
    pcd_o3d_to_numpy,
    pcd_ros_to_o3d,
)
from .channel import grpc_channel
from .base_inference import BaseInference
from communicator.channel import seerep_channel
from tools.seerep2coco import COCO_SEEREP
from logger import Client_logger, TqdmToLogger
from visual_utils import open3d_vis_utils as visualizer

logger = Client_logger(name="Triton-Client", level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger, level=logging.INFO)
O3D_DEVICE = o3d.core.Device("CPU:0")  # can also be set to GPU


class EvaluateInference(BaseInference):

    """
    A RosInference to support ROS input and provide input to channel for inference.
    """

    def __init__(self, args, channel, client, format="coco"):
        """
        channel: channel of type communicator.channel
        client: client of type clients

        """

        super().__init__(channel, client)

        self.image = None
        self.args = args
        self._register_inference()  # register inference based on type of client
        self.client_postprocess = client.get_postprocess()  # get postprocess of client
        self.client_preprocess = client.get_preprocess()
        self.model_name = client.model_name
        self.format = format
        if "COCO" in self.model_name or "coco" in self.model_name.lower():
            self.class_names = self.client_postprocess.load_class_names(dataset="COCO")
            self.modality = "image"
            logger.info(f"Using COCO image modality for inference")
        elif "CROP" in self.model_name or "crop" in self.model_name.lower():
            self.class_names = self.client_postprocess.load_class_names(dataset="CROP")
            self.modality = "image"
            logger.info(f"Using CROP image modality for inference")
        elif "KITTI" in self.model_name or "kitti" in self.model_name.lower():
            self.class_names = self.client_postprocess.load_class_names(dataset="KITTI")
            self.modality = "pointcloud"
            logger.info(f"Using KITTI pointcloud modality for inference")
        else:
            # TODO shutdown?
            self.class_names = None
        self.input_datatypes = {
            "UINT8": np.dtype(np.uint8),
            "INT16": np.dtype(np.int16),
            "INT32": np.dtype(np.int32),
            "FP16": np.dtype(np.float16),
            "FP32": np.dtype(np.float32),
            "FP64": np.dtype(np.float64),
        }
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "debug": logging.DEBUG,
            "critical": logging.CRITICAL,
        }
        logging.basicConfig(level=log_level[args.log_level])
        self.viz = args.visualize
        self.count = 0
        self.id_list_preds = []
        self.id_list_gts = []
        self.all_predictions = []
        self.all_groundtruths = []
        self.bag_processed = False
        self.gt_processed = False
        self.img_processed = False

    def _register_inference(self):
        """
        register inference
        """
        # for GRPC channel
        if type(self.channel) == grpc_channel.GRPCChannel:
            self._set_grpc_channel_members()
        else:
            pass

    def _set_grpc_channel_members(self):
        """
        Set properties for grpc channel, queried from the server.
        """
        # collect meta data of model and configuration
        meta_data = self.channel.get_metadata()

        # parse the model requirements from client
        # self.channel.input.name, output_name, c, h, w, format, self.channel.input.datatype = self.client.parse_model(
        #     meta_data["metadata_response"], meta_data["config_response"].config)

        input_metadata, output_metadata = self.client.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config
        )
        self.channel.input = [input["name"] for input in input_metadata]
        self.channel.output = [output["name"] for output in output_metadata]

        self.inputs = {}
        for input, i in zip(input_metadata, range(len(input_metadata))):
            self.inputs[
                "input_{}".format(i)
            ] = service_pb2.ModelInferRequest().InferInputTensor()
            self.inputs["input_{}".format(i)].name = input["name"]
            self.inputs["input_{}".format(i)].datatype = input["dtype"]
            if -1 in input["shape"]:
                input["shape"][0] = 10000  # tmp
            self.inputs["input_{}".format(i)].shape.extend(input["shape"])
            # assign the gathered model inputs to the grpc channel
            self.channel.request.inputs.extend([self.inputs["input_{}".format(i)]])

        self.outputs = {}
        for output, i in zip(output_metadata, range(len(output_metadata))):
            self.outputs[
                "output_{}".format(i)
            ] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs["output_{}".format(i)].name = output["name"]
            # assign the gathered model outputs to the grpc channel
            self.channel.request.outputs.extend([self.outputs["output_{}".format(i)]])

    def seerep_infer_pc(self, sample: np.array, 
                        confidence_threshold=0.4, 
                        class_idx=2, 
                        visualize=False):
        """
        Perform inference on the point cloud data.
        :param sample: Point cloud data
        :param confidence_threshold: Confidence threshold for filtering predictions
        :param class_idx: Class index for filtering predictions
        :return: Filtered predictions
        """
        self.pc = self.client_preprocess.filter_pc(sample)
        num_voxels = self.pc["voxels"].shape[0]
        self.channel.request.ClearField(
            "raw_input_contents"
        )  # Flush the previous sample content
        for key, idx in zip(self.inputs, range(len(self.inputs))):
            tmp_shape = self.inputs[key].shape
            self.inputs[key].ClearField("shape")
            tmp_shape[0] = num_voxels
            self.channel.request.inputs[idx].ClearField("shape")
            self.channel.request.inputs[idx].shape.extend(tmp_shape)
            self.inputs[key].shape.extend(tmp_shape)
        # Insert batch dimensions into the voxel coordinates------>change from N x 3 to N x 3+1. Assume batch size 1
        tmp_data = np.zeros(
            (self.pc["voxel_coords"].shape[0], self.pc["voxel_coords"].shape[1] + 1),
            dtype=self.pc["voxel_coords"].dtype,
        )
        tmp_data[:, 1:] = self.pc["voxel_coords"].copy()
        self.pc["voxel_coords"] = tmp_data.copy()
        del tmp_data
        # Make sure the data types and shapes are correct for each input before sending them as bytes, this causes wrong array values on the server
        assert (
            self.pc["voxels"].dtype
            == self.input_datatypes[self.inputs["input_0"].datatype]
        )
        assert (
            self.pc["voxel_coords"].dtype
            == self.input_datatypes[self.inputs["input_1"].datatype]
        )
        assert (
            self.pc["voxel_num_points"].dtype
            == self.input_datatypes[self.inputs["input_2"].datatype]
        )
        self.channel.request.raw_input_contents.extend(
            [
                self.pc["voxels"].tobytes(),
                self.pc["voxel_coords"].tobytes(),
                self.pc["voxel_num_points"].tobytes(),
            ]
        )
        self.channel.response = (
            self.channel.do_inference()
        )  # perform the channel Inference
        box_array, scores, labels = self.client_postprocess.extract_boxes(
            self.channel.response
        )
        
        # Show only persons above given confidence threshold
        indices = np.where((labels == class_idx) & (scores > confidence_threshold))[0].tolist()
        # indices = [i for i in range(len(labels))]

        if visualize:
            visualizer.draw_scenes(
                points=self.pc["points"],
                ref_boxes=box_array[indices, :],
                ref_scores=scores[indices],
                ref_labels=labels[indices],
            )

    def process_pc(self, data, seerep_channel: seerep_channel.SEEREPChannel):
        # traverse through the samples
        infer_array = np.zeros(len(data), dtype=np.float16)
        for sample, idx in tqdm(
            zip(data, range(len(data))),
            total=len(data),
            colour="GREEN",
            desc="Sending inference request to Triton",
            unit="request(s)",
        ):
            # perform an inference on each image, iteratively
            t3 = time.time()
            # pc = np.zeros_like(sample["point_cloud_processed"])
            # pc[:, 0] = sample["point_cloud"]["x"]["data"][:, 0]
            # pc[:, 1] = sample["point_cloud"]["y"]["data"][:, 0]
            # pc[:, 2] = sample["point_cloud"]["z"]["data"][:, 0]
            # pc[:, 3] = sample["point_cloud"]["reflectivity"]["data"][:, 0]/255.0
            pred = self.seerep_infer_pc(sample["point_cloud_processed"])
            # pred = self.seerep_infer_pc(pc)
            t4 = time.time()
            infer_array[idx] = t4 - t3
            # logger.info('Inference time: {}'.format(t4 - t3))
            # logger.info('Sent boxes for image under category name {}'.format(self.model_name))
        # t6 = time.time()
        
    def start_inference(self, model_name, format="coco"):
        schan = seerep_channel.SEEREPChannel(
            project_name=self.args.seerep_project,
            endpoint=self.args.channel_seerep,
            modality=self.modality,
            format=self.format,  # TODO make it dynamic with Source_Kitti
            visualize=self.viz,
        )
        if self.modality == "image":
            data = schan.run_query_images(self.args.semantics)
            self.process_images(data, schan)
        elif self.modality == "pointcloud":
            data = schan.run_query_pointclouds(model_name)
            self.process_pc(data, schan)
        else:
            logger.error("Invalid modality: {}".format(self.modality))
            sys.exit(0)