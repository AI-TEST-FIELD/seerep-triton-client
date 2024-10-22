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
import torchvision
import torch
# import open3d as o3d

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

# from tools.pointcloud import (
#     pcd_o3d_to_numpy,
#     pcd_ros_to_o3d,
# )
from .channel import grpc_channel
from .base_inference import BaseInference
from communicator.channel import seerep_channel
from tools.seerep2coco import COCO_SEEREP
from logger import Client_logger, TqdmToLogger
from utils import cxcy2xyxy, xyxy2cxcy
# from visual_utils import open3d_vis_utils as visualizer

logger = Client_logger(name="Triton-Client", level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger, level=logging.INFO)
# O3D_DEVICE = o3d.core.Device("CPU:0")  # can also be set to GPU

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
        # self.class_names = self.load_class_names()
        self.args = args
        self._register_inference()  # register inference based on type of client
        self.client_postprocess = client.get_postprocess()  # get postprocess of client
        self.client_preprocess = client.get_preprocess()
        self.model_name = client.model_name
        self.format = format
        if "COCO" or "coco" in self.model_name:
            self.class_names = self.client_postprocess.load_class_names(dataset="COCO")
        elif "CROP" in self.model_name:
            self.class_names = self.client_postprocess.load_class_names(dataset="CROP")
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
        if self.viz:
            self.winname = "Prediction {}".format(self.model_name)
            cv2.namedWindow(self.winname)
        self.count = 0
        self.id_list_preds = []
        self.id_list_gts = []
        self.all_predictions = []
        self.all_groundtruths = []
        self.datumaro_item = {
            "id": "example_img",
            "annotations": [
                {
                    "id": 0,
                    "type": "bbox",
                    "attributes":{
                        "occluded": "false",
                        "rotation": 0.0,
                    },
                    "group": 0,
                    "label_id": 0, 
                    "z_order": 0,
                    "bbox":[
                        0, 
                        1,
                        2,
                        3
                    ]
                },
            ],
            "attr": {
                "frame": 0
            },
            "point_cloud": {
                "path": ""
            },
            "media": {
                "path": ""
            }
        }
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

        self.input_metadata, self.output_metadata = self.client.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config
        )
        self.channel.input = [input["name"] for input in self.input_metadata]
        self.channel.output = [output["name"] for output in self.output_metadata]

        self.inputs = {}
        for input, i in zip(self.input_metadata, range(len(self.input_metadata))):
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
        for output, i in zip(self.output_metadata, range(len(self.output_metadata))):
            self.outputs[
                "output_{}".format(i)
            ] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs["output_{}".format(i)].name = output["name"]
            # assign the gathered model outputs to the grpc channel
            self.channel.request.outputs.extend([self.outputs["output_{}".format(i)]])

        # self.input_size = [h, w]
        # if format == mc.ModelInput.FORMAT_NHWC:
        #     self.channel.input.shape.extend([h, w, c])
        # else:
        #     self.channel.input.shape.extend([c, h, w])

        # if len(output_name) > 1:  # Models with multiple outputs Boxes, Classes and scores
        #     self.output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # boxes
        #     self.output0.name = output_name[0]
        #     self.output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # class_IDs
        #     self.output1.name = output_name[1]
        #     self.output2 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # scores
        #     self.output2.name = output_name[2]
        #     self.output3 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # image dims
        #     self.output3.name = output_name[3]

        #     self.channel.request.outputs.extend([self.output0,
        #                                          self.output1,
        #                                          self.output2,
        #                                          self.output3])
        # else:
        #     self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        #     self.output.name = output_name[0]
        #     self.channel.request.outputs.extend([self.output])
        # self.channel.output.name = output_name[0]
        # self.channel.request.outputs.extend([self.channel.output])

    def resize(self, image):
        # cv_image = cv2.resize(cv_image, (self.channel.input.shape[2], self.channel.input.shape[1]))
        tmp = image.copy()
        s_h, s_w = image.shape[0], image.shape[1]
        n_h, n_w = self.input_metadata[0]['shape'][1], self.input_metadata[0]['shape'][2]
        if s_w > s_h:
            # same aspect ratio
            padded_image = np.zeros((n_h, n_w, 3), dtype=np.uint8)
            cv_image = imutils.resize(tmp, width=n_w)
            # different aspect ratio
            if cv_image.shape[0] > n_h:
                cv_image = imutils.resize(tmp, height=n_h)
            if cv_image.shape[1] > n_w:
                cv_image = imutils.resize(tmp, width=n_w)
        else:
            padded_image = np.zeros((n_h, n_w, 3), dtype=np.uint8)
            cv_image = imutils.resize(tmp, height=n_h)
        # padded image
        padded_image[0 : cv_image.shape[0], 0 : cv_image.shape[1]] = cv_image
        # named_window = 'resized'
        # cv2.imshow(named_window, padded_image)
        # cv2.waitKey()
        # cv2.destroyWindow(named_window)
        return padded_image, cv_image.shape[0], cv_image.shape[1]

    def seerep_infer_image(self, image):
        # convert numpy array to cv2
        cv_image = image
        self.orig_size = cv_image.shape[0:2]
        self.orig_image = cv_image.copy()
        s_h, s_w = cv_image.shape[0], cv_image.shape[1]
        # n_h, n_w = self.channel.input.shape[1], self.channel.input.shape[2]
        cv_image, r_h, r_w = self.resize(cv_image)
        # named_window = 'Resized source image'
        # cv2.imshow(named_window, cv_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow(named_window)
        tmp = cv_image.copy()
        self.image = self.client_preprocess.image_adjust(cv_image)
        # convert to input data type the model expects
        self.image = self.image.astype(
            self.input_datatypes[self.input_metadata[0]['dtype']]
        )
        if self.image is not None:
            self.channel.request.ClearField("inputs")
            self.channel.request.ClearField("raw_input_contents")  # Flush the previous image contents
            self.channel.request.inputs.extend([self.inputs['input_0']])
            self.channel.request.raw_input_contents.extend([self.image.tobytes()])
            self.channel.response = self.channel.do_inference()  # Inference
            self.prediction = self.client_postprocess.extract_boxes(
                self.channel.response, conf_thres=0.3,
            )
            if len(self.prediction[1]) > 0:
                # if self.viz:
                #     # tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR).astype(np.uint8)
                #     for box in self.prediction[0]:
                #         cv2.rectangle(
                #             tmp,
                #             (int(box[0]), int(box[1])),
                #             (int(box[2]), int(box[3])),
                #             (255, 0, 0),
                #             2,
                #         )
                #     named_window = "Resized source image with prediction"
                #     cv2.imshow(named_window, tmp)
                #     cv2.waitKey()
                #     cv2.destroyWindow(named_window)
                self.prediction[0] = self._scale_box_array(
                    self.prediction[0], source_dim=(r_h, r_w), padded=True
                )
                if self.format == "kitti":
                    persons = np.where(self.prediction[1] == 0)  # filter Pedestrians
                    return (
                        self.prediction[0][persons],
                        self.prediction[1][persons],
                        self.prediction[2][persons],
                    )
                elif self.format == "coco":
                    persons = np.where(self.prediction[1] == 0)  # filter persons
                    return (
                        self.prediction[0][persons],
                        self.prediction[1][persons],
                        self.prediction[2][persons],
                    )
                elif self.format == "aitf":
                    persons = np.where(self.prediction[1] == 0)  # filter persons
                    return (
                        self.prediction[0][persons],
                        self.prediction[1][persons],
                        self.prediction[2][persons],
                    )
                else:
                    return self.prediction
            else:
                return self.prediction

    # def preprocess_pc(self, ros_pcd, sensor_name: str, dataset_name: str) -> np.ndarray:
    #     """
    #     Processing Steps:
    #         1. Transforms Point Cloud into the Robot base_frame, based on homegenous transform from the calibration procedure.
    #         2. Translate Point Cloud into the Dataset specific detector training dataset frame. Adjusts the Point Cloud to mimic the relative Lidar position from the detectors training dataset.
    #         3. Normalize feature field [0, 1], by maximal possible feature value (reflectance/intensity = 255).


    #     Args:
    #         ros_pcd : Point cloud as Datastructure generated from ros message.
    #         sensor_name : Sensor specific name tag associated with the point cloud.
    #         dataset_name : The name of the dataset used for training of the object detector.

    #     Return:
    #         preprocessed_np_pcd : Preprocessed point cloud as numpy array [[x, y, z, feature], ...]

    #     """

    #     # dictionary for sensor and dataset transformations
    #     TRANSFORM_DICT = {
    #         "base_transformation": {
    #             "ouster": [
    #                 [0.82638931, -0.02497454, 0.56254509, 0.191287],
    #                 [0.01212522, 0.99957356, 0.02656451, -0.35169424],
    #                 [-0.56296864, -0.01513165, 0.82633973, 1.90064396],
    #                 [0.0, 0.0, 0.0, 1.0],
    #             ],
    #             "velodyne": [],
    #             "robosense": [],
    #         },
    #         "dataset_translation": {
    #             "kitti": [0.0, 0.0, -1.026558971],
    #             "coco": [],
    #         },
    #     }

    #     # lidar sensors which use the reflectivity field instead of intensity
    #     REFLECTIVITY_SENSORS = ["ouster"]

    #     MAX_FEATURE_VALUE = 255

    #     # sensor and data specific params
    #     sensor_to_robot_base_transform = o3d.core.Tensor(
    #         TRANSFORM_DICT["base_transformation"][sensor_name], device=O3D_DEVICE
    #     )
    #     robot_base_to_train_dataset_translation = o3d.core.Tensor(
    #         TRANSFORM_DICT["dataset_translation"][dataset_name], device=O3D_DEVICE
    #     )
    #     feature_field = (
    #         "reflectivity" if sensor_name in REFLECTIVITY_SENSORS else "intensity"
    #     )

    #     # preprocessing steps
    #     # raw_o3d_pcd = pcd_ros_to_o3d(ros_pcd=ros_pcd, feature_field=feature_field)
    #     preprocessed_o3d_pcd = raw_o3d_pcd.transform(sensor_to_robot_base_transform)
    #     preprocessed_o3d_pcd = preprocessed_o3d_pcd.translate(
    #         robot_base_to_train_dataset_translation
    #     )
    #     preprocessed_np_pcd = pcd_o3d_to_numpy(
    #         o3d_pcd=preprocessed_o3d_pcd, feature_field=feature_field
    #     )
    #     preprocessed_np_pcd[:, 3] /= MAX_FEATURE_VALUE

    #     self.pc = preprocessed_np_pcd
    #     return preprocessed_np_pcd

    def seerep_infer_pc(self, pointclouds: np.array):
        self.preprocess_pc(
            ros_pcd=pointclouds, sensor_name="velodyne", dataset_name="kitti"
        )

        self.pc = self.client_preprocess.filter_pc(self.pc)
        num_voxels = self.pc["voxels"].shape[0]
        self.channel.request.ClearField("raw_input_contents")  # Flush the previous sample content
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
        indices = np.where((labels == 2) & (scores > 0.4))[0].tolist()
        # indices = [i for i in range(len(labels))]

        if True:
            visualizer.draw_scenes(
                points=self.pc["points"],
                ref_boxes=box_array[indices, :],
                ref_scores=scores[indices],
                ref_labels=labels[indices],
            )

    def process_images(self, data):
        t2 = time.time()
        if len(data) == 0:
            logger.critical(
                "No data samples found in the SEEREP database matching your query"
            )
        else:
            # logger.info('Fetching time: {} s'.format(np.round(t2 - t1, 3)))
            color1 = (0, 0, 255)  # red
            color2 = (255, 255, 255)  # white
            text_color = (255, 255, 255)
            color3 = (255, 0, 0)
            # traverse through the images
            # logger.info('Sending inference request to Triton for each sample')
            infer_array = np.zeros(len(data), dtype=np.float16)
            for sample, seerep_sample_idx in tqdm(
                zip(data, range(len(data))),
                total=len(data),
                colour="GREEN",
                desc="Sending inference request to Triton",
                unit="requests",
                ascii=True,
            ):
                if self.model_name in sample['annotations']['categories']['label']['labels']:
                    pass
                else:
                    predictions = {
                    'annotations':[],
                    'dm_format_version':1,
                    }
                    # perform an inference on each image, iteratively
                    t3 = time.time()
                    pred = self.seerep_infer_image(sample["image"])
                    t4 = time.time()
                    infer_array[seerep_sample_idx] = t4 - t3
                    # logger.info('Inference time: {}'.format(t4 - t3))
                    tmp = {
                        'bbox':[],
                        'id':'1',
                        'label_id':'',
                        'score':1,
                    }
                    
                    # traverse the predictions for the current image
                    if len(pred[1]) == 0:
                        pass
                    else:
                        for obj in range(len(pred[1])):
                            start_cord, end_cord = (pred[0][obj, 0], pred[0][obj, 1]), \
                                                (pred[0][obj, 2], pred[0][obj, 3])
                            x, y, w, h = (
                                np.round((start_cord[0] + end_cord[0]) / 2, 2),
                                np.round((start_cord[1] + end_cord[1]) / 2, 2),
                                np.round(end_cord[0] - start_cord[0], 2),
                                np.round(end_cord[1] - start_cord[1], 2),
                            )
                            assert x > 0 and y > 0 and w > 0 and h > 0
                            tmp['bbox'] = [x, y, w, h]
                            tmp['score'] = np.round(pred[2][obj], 2)
                            tmp['id'] = str(int(pred[1][obj]))
                            tmp['label_id'] = str(seerep_sample_idx)
                            predictions['annotations'].append(tmp)
                            tmp = {
                                'bbox':[],
                                'id':'1',
                                'label_id':'',
                                'score':1,
                            }
                            # Visualize the predictions generated by triton inference
                            if self.viz:
                                label = self.class_names[int(pred[1][obj])]
                                (tw, th), _ = cv2.getTextSize(
                                    # "{} {} %".format(label, round(pred[2][obj] * 100, 2)),
                                    "{}".format(label),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    2,
                                )
                                # Plot prediction box
                                cv2.rectangle(
                                    sample["image"],
                                    (int(pred[0][obj, 0]), int(pred[0][obj, 1])),
                                    (int(pred[0][obj, 2]), int(pred[0][obj, 3])),
                                    color1,
                                    2,
                                )
                                # Plot prediction label background box
                                cv2.rectangle(
                                    sample["image"],
                                    (int(start_cord[0]), int(start_cord[1] - 25)),
                                    (int(start_cord[0] + tw), int(start_cord[1])),
                                    color1,
                                    -1,
                                )
                                # Put class label and confidence value
                                cv2.putText(
                                    sample["image"],
                                    # "{} {} %".format(label, round(pred[2][obj], 2) * 100),
                                    "{}".format(label),
                                    (int(pred[0][obj, 0]), int(pred[0][obj, 1]) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    text_color,
                                    2,
                                )
                            
                        data[seerep_sample_idx]['annotations']['items'].append(predictions) 
                # Visualize the groundtruth annotations on the same image as predictions
                if self.viz:
                    for ann_idx, ann in enumerate(sample['annotations']['items'][0]['annotations']):
                        bbox = ann['bbox']
                        bbox = cxcy2xyxy(bbox)
                        label = int(ann['id'])
                        cv2.rectangle(sample['image'], 
                                        (bbox[0], bbox[1]), 
                                        (bbox[2], bbox[3]), 
                                        color3, 2)
                        # (tw, th), _ = cv2.getTextSize(self.ann_dict[label], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        # cv2.rectangle(sample['image'], 
                        #                 (bbox[0], bbox[1] - 25), 
                        #                 (bbox[0] + tw, bbox[1]), 
                        #                 color3, -1)
                        cv2.putText(sample['image'], 
                                    str(label), 
                                    (bbox[0], bbox[1] - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.9, (255,255,255), 2)
                    cv2.imshow(self.winname, sample['image'])
                    cv2.waitKey() 
            if self.viz:
                cv2.destroyWindow(self.winname) 
            # child = False
            # male = False
            # adults = []
            # with open('evaluation_list.txt', 'w') as f:
            #     for sample in data: 
            #         # Find at least one adult male dummy 
            #         adults = [i[4] for i in sample['boxes']]
            #         if 1 in adults:
            #             for det in sample['boxes']:
            #                 # Adult == 1 Child == 2
            #                 if det[4] == 1 and det[6] > 0.2: # check if adult and IoU greater than 40 percent
            #                     male = True
            #                 # elif det[4] == 2 and det[6] > 0.3:    # TODO need to add more intelligent and thorough checks for child class
            #                 #     child = True
            #             f.writelines('{} {} {}\n'.format(sample['timestamp'][0], sample['timestamp'][1], int(male)))  
            #         else:                 # No male dummy exists in the ground truth
            #             f.writelines('{} {} {}\n'.format(sample['timestamp'][0], sample['timestamp'][1], -1)) 
            #         # child = False
            #         male = False
            # cv2.imwrite('./rainy/image_{}.png'.format(idx), cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            logger.info('Processed all inference requests in current data subset! ')
        return data

    def process_pc(self, data, seerep_channel: seerep_channel.SEEREPChannel):
        # traverse through the samples
        infer_array = np.zeros(len(data), dtype=np.float16)
        for sample, idx in tqdm(
            zip(data, range(len(data))),
            total=len(data),
            colour="GREEN",
            # file=tqdm_out,
            desc="Sending inference request to Triton",
            unit="request",
        ):
            # perform an inference on each image, iteratively
            t3 = time.time()
            pred = self.seerep_infer_pc(sample["point_cloud"])
            t4 = time.time()
            infer_array[idx] = t4 - t3
            # logger.info('Inference time: {}'.format(t4 - t3))
            sample["predictions"] = []
            bbs = []
            labels = []
            confidences = []
            # traverse the predictions for the current pointclouds
            # for obj in range(len(pred[1])):
            #     pass
            # if self.viz:
            #     pass
            # cv2.imwrite('./rainy/image_{}.png'.format(idx), cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            # TODO run evaluation without inference call
            # schan.sendboundingbox(sample, bbs, labels, confidences, self.model_name+'2')
            # logger.info('Sent boxes for image under category name {}'.format(self.model_name))
        # Convert groundtruth and predictions to PyCOCO format for evaluation
        # logger.info('Average Inference time / image: {} s'.format(np.round(np.sum(infer_array)/len(infer_array), 3)))
        # t5 = time.time()
        # coco_data = COCO_SEEREP(seerep_data=data, format=self.format)
        # cocoEval = COCOeval(coco_data.ground_truth, coco_data.predictions, 'bbox')
        # cocoEval.evaluate()
        # cocoEval.accumulate()
        # cocoEval.summarize()
        # t6 = time.time()
        # logger.info('Evaluation time: {} s'.format(np.round(t6 - t5, 3)))

    def start_inference(self, model_name, format="coco", modality="images"):
        schan = seerep_channel.SEEREPChannel(
            project_name=self.args.seerep_project,
            socket=self.args.channel_seerep,
            modality=modality,
            format=self.format,  # TODO make it dynamic with Source_Kitti
            visualize=self.viz,
        )
        
        # TODO! make a decision based on Images or PointCloud or both for selecting service stubs
        sample_type = "image"
        if sample_type == "image":
            data = schan.run_query_images(self.args.semantics)
            data = self.process_images(data)
            # Send predictions back to SEEREP for future use
            schan.send_dataset(data, category=self.model_name)
        elif sample_type == "point_clouds":
            data = schan.run_query_pointclouds(self.args.semantics)
            self.process_pc(data, schan)
    
    def _scale_boxes(self, box, normalized=False):
        """
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        """
        if normalized:
            # TODO make it dynamic with mc.Modelshape according to CHW or HWC
            xtl, xbr = box[0] * self.orig_size[1], box[2] * self.orig_size[1]
            ytl, ybr = box[1] * self.orig_size[0], box[3] * self.orig_size[0]
        else:
            xtl, xbr = box[0] * (self.orig_size[1] / self.input_size[0]), box[2] * (
                self.orig_size[1] / self.input_size[0]
            )
            ytl, ybr = (
                box[1] * self.orig_size[0] / self.input_size[1],
                box[3] * self.orig_size[0] / self.input_size[1],
            )

        return [xtl, ytl, xbr, ybr]

    def _scale_box_array(self, box, source_dim=(512, 512), padded=False):
        """
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        """
        # if normalized:
        #     # TODO make it dynamic with mc.Modelshape according to CHW or HWC
        #     xtl, xbr = box[0] * self.orig_size[1], box[2] * self.orig_size[1]
        #     ytl, ybr = box[1] * self.orig_size[0], box[3] * self.orig_size[0]
        if padded:
            xtl, xbr = box[:, 0] * (self.orig_size[1] / source_dim[1]), box[:, 2] * (
                self.orig_size[1] / source_dim[1]
            )
            ytl, ybr = (
                box[:, 1] * self.orig_size[0] / source_dim[0],
                box[:, 3] * self.orig_size[0] / source_dim[0],
            )
        else:
            xtl, xbr = box[:, 0] * (self.orig_size[1] / self.input_size[1]), box[
                :, 2
            ] * (self.orig_size[1] / self.input_size[1])
            ytl, ybr = (
                box[:, 1] * self.orig_size[0] / self.input_size[0],
                box[:, 3] * self.orig_size[0] / self.input_size[0],
            )
        xtl = np.reshape(xtl, (len(xtl), 1))
        xbr = np.reshape(xbr, (len(xbr), 1))

        ytl = np.reshape(ytl, (len(ytl), 1))
        ybr = np.reshape(ybr, (len(ybr), 1))
        return np.concatenate((xtl, ytl, xbr, ybr, box[:, 4:6]), axis=1)
