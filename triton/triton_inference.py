# import time
import cv2
import time
import numpy as np
import sys
import logging
from tqdm import tqdm
from tritonclient.grpc import service_pb2
from communicator.endpoint import SeerepEndpoint, TritonEndpoint
from logger import Client_logger, TqdmToLogger
from visual_utils import Visualizer
from utils import (
    resize,
    scale_box_array,
    process_model_output,
    visualize,
)

logger = Client_logger(name="Triton-Inference", level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger, level=logging.INFO)

class TritonInference:
    def __init__(self, 
                 model_name='yolov5m_coco', 
                 seerep_endpoint_url='agrigaia-ur.ni.dfki:9090', 
                 triton_endpoint_url='10.249.6.23:8001', 
                 log_level='error',
                 visualize=False,
                 modality='image'):
        self.model_name = model_name
        self.seerep_endpoint = SeerepEndpoint(
            seerep_endpoint_url,
            modality=modality,
            visualize=False, 
            log_level=log_level
            )
        self.triton_endpoint = TritonEndpoint(
            model_name=model_name,
            endpoint_url=triton_endpoint_url,
            log_level=log_level
            )
        
        self.modality = modality
        self.log_level = log_level
        self.visualize = visualize
        self._initialize_models()
        self._initialize_misc()
        logging.basicConfig(level=self.log_level[log_level])
        
    def _initialize_models(self):
        """
        Initialize the models for the given modality. 
        """
        if self.modality == 'image':
            try:
                from models import Ultralytics, Detectron2_Det, Detrex_Det
                self.models_database = {
                    'yolov5m_coco': Ultralytics,
                    'yolov5m_iso': Ultralytics,
                    'fcos_coco':Detectron2_Det,
                    'frcnn_800_coco':Detectron2_Det,
                    'retinanet_coco':Detectron2_Det,
                }
            except ImportError as e:
                logger.error(f"Error importing models for image modality: {e}")
                sys.exit(1)
        elif self.modality == 'pointcloud':
            try:
                from models import OpenPCDet
                self.models_database = {
                    'second_iou_kitti':OpenPCDet,
                    'pointpillar_kitti':OpenPCDet,
                    # more clients can be added
                }
            except ImportError as e:
                logger.error(f"Error importing models for pointcloud modality: {e}")
                sys.exit(1)
        else:
            logger.error(f"Unsupported modality: {self.modality} \n Supported modalities are case-sensitive: image, pointcloud")
            sys.exit(1)
        self.model = self.models_database.get(self.model_name)(model_name=self.model_name)
        self.model_preprocess = self.model.get_preprocess()
        self.model_postprocess = self.model.get_postprocess()
        self._initialize_model_inputs()
        
    def _initialize_model_inputs(self):
        """
        Register the GRPC endpoint for Triton server and fetch model metadata and configuration.
        HTTP endpoint is not supported yet. 
        """
        # for GRPC channel
        if type(self.triton_endpoint) == TritonEndpoint:
            self._init_model_io()
        else:
            logger.error("Triton endpoint is not initialized correctly.")
            sys.exit(1)
    
    def _init_model_io(self):
        """
        Fetch the model metadata and initialize the model input and output tensors
        on the client side from the gRPC endpoint of Triton server.
        Rest API is not supported. 
        """
        # collect meta data of model and configuration
        meta_data = self.triton_endpoint.get_metadata()

        # parse the model requirements from client
        self.input_metadata, self.output_metadata = self.model.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config
        )
        self.triton_endpoint.input = [input["name"] for input in self.input_metadata]
        self.triton_endpoint.output = [output["name"] for output in self.output_metadata]

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
            self.triton_endpoint.request.inputs.extend([self.inputs["input_{}".format(i)]])

        self.outputs = {}
        for output, i in zip(self.output_metadata, range(len(self.output_metadata))):
            self.outputs[
                "output_{}".format(i)
            ] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs["output_{}".format(i)].name = output["name"]
            # assign the gathered model outputs to the grpc channel
            self.triton_endpoint.request.outputs.extend([self.outputs["output_{}".format(i)]])

    def _initialize_misc(self):
        """
        Initialize miscellaneous configurations for the triton inference
        e.g. datatypes, logging levels, datumaro labels, class names.
        """
        self.input_datatypes = {
            "UINT8": np.dtype(np.uint8),
            "INT16": np.dtype(np.int16),
            "INT32": np.dtype(np.int32),
            "FP16": np.dtype(np.float16),
            "FP32": np.dtype(np.float32),
            "FP64": np.dtype(np.float64),
        }
        self.log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "debug": logging.DEBUG,
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
        }
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
        if "COCO" or "coco" in self.model_name:
            self.class_names = self.model_postprocess.load_class_names(dataset="COCO")
            self.format = "coco"
        elif "CROP" in self.model_name:
            self.class_names = self.model_postprocess.load_class_names(dataset="CROP")
            self.format = "crop"
        elif "KITTI" in self.model_name or "kitti" in self.model_name.lower():
            self.class_names = self.client_postprocess.load_class_names(dataset="KITTI")
            self.format = "kitti"
        else:
            logger.error("Class names not found for the model. Make sure coco or crop is in the model name")    
            self.class_names = None
            
    def _configure_model_params(self):
        """
        Fetch and set the model metadata and configuration 
        on the client side from the gRPC endpoint of Triton server.
        Rest API is not supported. 
        """
        # collect meta data of model and configuration
        meta_data = self.triton_endpoint.get_metadata()

        # parse the model requirements from client
        self.input_metadata, self.output_metadata = self.model.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config
        )
        self.triton_endpoint.input = [input["name"] for input in self.input_metadata]
        self.triton_endpoint.output = [output["name"] for output in self.output_metadata]

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
            self.triton_endpoint.request.inputs.extend([self.inputs["input_{}".format(i)]])

        self.outputs = {}
        for output, i in zip(self.output_metadata, range(len(self.output_metadata))):
            self.outputs[
                "output_{}".format(i)
            ] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs["output_{}".format(i)].name = output["name"]
            # assign the gathered model outputs to the grpc channel
            self.triton_endpoint.request.outputs.extend([self.outputs["output_{}".format(i)]])

    # TODO add dynamic confidence value and 
    def triton_infer_image(self, cv_image, filter_class_idx=0)->list[np.ndarray]:
        """
        Perform inference on a single image via the Triton server gRPC endpoint.
        By default, the image is resized to the model input size.
        The model input size is determined by the model metadata. 
        Class ID=0 is filtered out by default. since persons are interesting only for the model 
        based on COCO dataset.
        Args:
            cv_image (numpy.ndarray): Input image in BGR format.
        Returns:
            A list of numpy arrays containing the bounding boxes (tlbr xyxy), class IDs, and scores.
        """
        self.orig_image = cv_image.copy()
        original_h, original_w = cv_image.shape[0], cv_image.shape[1]
        cv_image, model_input_h, model_input_w = resize(cv_image, self.input_metadata)
        # named_window = 'Resized source image'
        # cv2.imshow(named_window, cv_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow(named_window)
        # if self.visualize:
        #     tmp = cv_image.copy()
        self.image = self.model_preprocess.image_adjust(cv_image)
        # convert to input data type the model expects
        self.image = self.image.astype(
            self.input_datatypes[self.input_metadata[0]['dtype']]
        )
        if self.image is not None:
            self.triton_endpoint.request.ClearField("inputs")
            self.triton_endpoint.request.ClearField("raw_input_contents")  # Flush the previous image contents
            self.triton_endpoint.request.inputs.extend([self.inputs['input_0']])
            self.triton_endpoint.request.raw_input_contents.extend([self.image.tobytes()])
            self.triton_endpoint.response = self.triton_endpoint.do_inference()  # Inference
            self.prediction = self.model_postprocess.extract_boxes(
                self.triton_endpoint.response,
            )
            if len(self.prediction[1]) > 0:
                self.prediction[0] = scale_box_array(
                    self.prediction[0], 
                    model_input_dim=(model_input_h, model_input_w), 
                    image_dim=(original_h, original_w), 
                    padded=True
                )
                # if self.visualize:
                #     self.visualize_img(self.orig_image, self.prediction[0], mode='BGR')
                if self.format == "kitti" or self.format == "coco" or self.format == "aitf":
                    persons = np.where(self.prediction[1] == 0)  # filter Pedestrians
                    return (
                        self.prediction[0][persons],
                        self.prediction[1][persons],
                        self.prediction[2][persons],
                    )
                else:
                    return self.prediction
            else:
                return self.prediction
    
    def triton_infer_pointcloud(self, pointcloud:np.ndarray, filter_class_idx=2, confidence=0.4)->list[np.ndarray]:
        """
        Perform inference on the point cloud data.
        :param sample: Point cloud data
        :param confidence_threshold: Confidence threshold for filtering predictions
        :param class_idx: Class index for filtering predictions
        :return: Filtered predictions
        """
        self.pc = self.client_preprocess.filter_pc(pointcloud)
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
        indices = np.where((labels == filter_class_idx) & (scores > confidence))[0].tolist()

        if self.visualize:
            Visualizer.draw_scenes(
                points=self.pc["points"],
                ref_boxes=box_array[indices, :],
                ref_scores=scores[indices],
                ref_labels=labels[indices],
            )
    
    def generate_datumaro_predictions(self, data: list[dict])->list[dict]:
        """
        Iterates through the data samples and performs inference on each image sample.
        Adds the predictions to the individual sample as data[sample_idx]['annotations'] in datumaro format.
        Args:
            data (list): List of data samples from SEEREP.
        
        """
        if self.modality == 'image':
            data_key = 'image'
            infer_function = self.triton_infer_image  
        elif self.modality == 'pointcloud':
            data_key = 'pointcloud'
            infer_function = self.triton_infer_pointcloud
        else:
            logger.error("Unsupported modality: {}".format(self.modality))
            sys.exit(1)
        t2 = time.time()
        if len(data) == 0:
            logger.critical(
                "No data samples found in the SEEREP database matching your query"
            )
        else:
            # traverse through the images
            infer_array = np.zeros(len(data), dtype=np.float16)
            for sample, seerep_sample_idx in tqdm(
                zip(data, range(len(data))),
                total=len(data),
                colour="GREEN",
                desc="Sending inference request to Triton",
                unit="request(s)"
            ):
                if False:
                    self.processed_counter += 1
                # elif sample['no_grountruth']:
                #     self.no_gt_counter += 1
                else:
                    predictions = {
                    'annotations':[],
                    'dm_format_version':1,
                    }
                    # perform inference on each image, iteratively
                    t3 = time.time()
                    pred = infer_function(sample[data_key])
                    t4 = time.time()
                    infer_array[seerep_sample_idx] = t4 - t3
                    # traverse the predictions for the current image
                    predictions['annotations'] = process_model_output(
                                                                    sample=sample,
                                                                    model_output=pred,
                                                                    sample_idx=seerep_sample_idx,
                                                                    visualize=self.visualize,
                                                                    class_names=self.class_names)
                            
                    data[seerep_sample_idx]['annotations']['items'].append(predictions) 
                # Visualize the groundtruth annotations on the same image as predictions
                if self.visualize:
                    visualize(sample, 
                            'self.winname', 
                            self.class_names,
                            new_model_key=False,
                            model_name=None
                            )
            if self.visualize:
                cv2.destroyWindow(self.winname) 
            logger.info('Processed all inference requests in current data subset!')
            # logger.info('{} were skipped since predictions were already stored from previous runs'.format(self.processed_counter))
            # logger.info('{} image had no ground truth associated with them.'.format(self.no_gt_counter))
        return data
             
    def generate_annotations_by_sample_uuids(self, sample_uuids: list)->list[dict]:
        """
        Generate annotations for the given sample UUIDs using the Triton inference model.
        Args:
            sample_uuids (list): List of sample UUIDs to generate annotations for.
        Returns:
        """
        data = self.seerep_endpoint.fetch_data_by_sample_uuid(sample_uuids, model_name=self.model_name)
        data = self.generate_datumaro_predictions(data)
        return data
    
    def get_project_uuid(self, project_name: str)->str:
        """
        Get the project UUID for the given project name.
        Args:
            project_name (str): Name of the project.
        Returns:
            str: Project UUID.
        """
        return self.seerep_endpoint.get_project_uuid(project_name)
    
    def generate_annotations_by_project_uuids(self, project_uuids: list)->list[dict]:
        """
        Generate annotations for the given sample UUIDs using the Triton inference model.
        Args:
            sample_uuids (list): List of sample UUIDs to generate annotations for.
        Returns:
        """
        # project_uuid = self.seerep_endpoint.get_project_uuid('EV41_Kleidung_Sonnenbrille_DunkleCap_225Deg_2024-10-10-18-58-13_0')
        # data = self.seerep_endpoint.fetch_data_by_project([project_uuid], 
        #                                             model_name=self.model_name)
        # # NOTE! This is a temporary fix to fetch data by sample uuids. UUIDs will be fetched directly inside the Triton class.
        sample_uuids = self.seerep_endpoint.fetch_uuids_by_project_uuid(project_uuids)
        data = self.seerep_endpoint.fetch_data_by_sample_uuid(sample_uuids, model_name=self.model_name)
        data = self.generate_datumaro_predictions(data)

        return data
    