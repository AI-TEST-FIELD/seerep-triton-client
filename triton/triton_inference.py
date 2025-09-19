# import time
import cv2
import time
import numpy as np
import sys
import logging
from tqdm import tqdm
from tritonclient.grpc import service_pb2
from communicator.endpoint import SeerepEndpoint, TritonEndpoint
from models.base_model import Model
# from models.postprocess import Postprocess
from logger import Client_logger, TqdmToLogger
from models.mmdet import MMDet
from visual_utils import Visualizer
from utils import (
    resize,
    scale_box_array,
    process_model_output,
    DatumaroAnnotation,
    visualize,
)

logger = Client_logger(name="Triton-Inference", level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger, level=logging.INFO)

class ModelData:
    def __init__(self, model, model_preprocess, model_postprocess):
        self.model = model
        self.model_preprocess = model_preprocess
        self.model_postprocess = model_postprocess
        self.model_name = model.model_name
        self.dynamic = False
        self.batching_supported = False

    def init_endpoint(self, endpoint_url: str, log_level: str="info"):
        """
        Initialize the Triton endpoint for the model.
        Args:
            endpoint_url (str): URL of the Triton server.
            log_level (str): Logging level for the Triton client.
        """
        self.endpoint = TritonEndpoint(
            model_name=self.model_name,
            endpoint_url=endpoint_url,
            log_level=log_level,
        )
        self.class_names = None
        if "COCO" in self.model_name or "coco" in self.model_name:
            self.class_names = self.model_postprocess.load_class_names(dataset="COCO")
            self.format = "coco"
        elif "CROP" in self.model_name:
            self.class_names = self.model_postprocess.load_class_names(dataset="CROP")
            self.format = "crop"
        elif "KITTI" in self.model_name or "kitti" in self.model_name.lower():
            self.class_names = self.model_postprocess.load_class_names(dataset="KITTI")
            self.format = "kitti"
        else:
            logger.error("Class names not found for the model. Make sure coco or crop is in the model name")
            self.class_names = None
            self.format = None
        self._init_model_io()

    def _init_model_io(self):
        """
        Fetch the model metadata and initialize the model input and output tensors
        on the client side from the gRPC endpoint of Triton server.
        """
        self.meta_data = self.endpoint.get_metadata()
        self.input_metadata, self.output_metadata = self.model.parse_model(
            self.meta_data["metadata_response"], self.meta_data["config_response"].config
        )
        self.endpoint.input = [input["name"] for input in self.input_metadata]
        self.endpoint.output = [output["name"] for output in self.output_metadata]

        # Check if model supports batching
        if self.format == "kitti":
            # Point cloud models in OpenPCDet do not support batching
            self.batching_supported = False
            self.max_batch_size = 1
        else:
            max_batch_size = self.meta_data["config_response"].config.max_batch_size
            self.batching_supported = max_batch_size > 0
            self.max_batch_size = max_batch_size if self.batching_supported else 1

        self.inputs = {}
        for input, i in zip(self.input_metadata, range(len(self.input_metadata))):
            self.inputs[f"input_{i}"] = service_pb2.ModelInferRequest().InferInputTensor()
            self.inputs[f"input_{i}"].name = input["name"]
            self.inputs[f"input_{i}"].datatype = input["dtype"]
            self.endpoint.request.inputs.extend([self.inputs[f"input_{i}"]])
            # DON'T set the shape here for batching models!
            # The input shapes are set dynamically in the respective inference function e.g.
            # triton_infer_pointcloud() and triton_infer_image()
            # self.inputs[f"input_{i}"].shape.extend(shape_to_set)

        # Check for dynamic shapes
        if any(-1 in input["shape"] for input in self.input_metadata):
            self.dynamic = True

        self.outputs = {}
        for output, i in zip(self.output_metadata, range(len(self.output_metadata))):
            self.outputs[f"output_{i}"] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs[f"output_{i}"].name = output["name"]
            self.endpoint.request.outputs.extend([self.outputs[f"output_{i}"]])

class TritonInference:
    def __init__(self,
                 model_name: list[str]=['yolov5m_coco'],
                 seerep_endpoint_url='agrigaia-ur.ni.dfki:9090',
                 triton_endpoint_url='10.249.6.23:8001',
                 log_level='error',
                 visualize=False,
                 modality='image'):
        self.model_names = model_name
        self.seerep_endpoint = SeerepEndpoint(
            seerep_endpoint_url,
            modality=modality,
            visualize=visualize,
            log_level=log_level
            )
        # TODO Create multiple endpoints for multiple models passed as string list
        self.models = {}
        self.modality = modality
        self.log_level = log_level
        self.visualize = visualize
        if self.visualize and self.modality == 'image':
            self.source_window = 'Inferred image'
            cv2.namedWindow(self.source_window)
        elif self.visualize and self.modality == 'pointcloud':
            self.source_window = 'Inferred pointcloud'
            self.visualizer = Visualizer(origin=True)
        else:
            if not self.visualize:
                logger.info("Visualization is disabled for {} modality.".format(self.modality))
            else:
                logger.error("Visualization is only supported for image and pointcloud modalities. case-senstive.")
        self.triton_endpoint_url = triton_endpoint_url
        self._initialize_models()
        self._initialize_misc()
        log_level_map = {
            'ERROR': logging.ERROR,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
        }
        logger.setLevel(log_level_map.get(log_level.upper(), logging.ERROR))

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
                    'rtmdet_coco':MMDet,
                    'conditional_detr_r50_coco': MMDet,
                    'crowdhuman_r_50_coco': MMDet,
                    'grounding_dino_b_swin_coco': MMDet,
                    'faster_rcnn_r_101_coco': MMDet,

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
                    'parta2_free_kitti':OpenPCDet,
                    'pv_rcnn_kitti':OpenPCDet,
                    'pointrcnn_iou_kitti':OpenPCDet,
                    # more clients can be added
                }
            except ImportError as e:
                logger.error(f"Error importing models for pointcloud modality: {e}")
                sys.exit(1)
        else:
            logger.error(f"Unsupported modality: {self.modality} \n Supported modalities are case-sensitive: image, pointcloud")
            sys.exit(1)
        for model in self.model_names:
            if model not in self.models_database.keys():
                logger.error(
                    f"Model {self.model_names} not found in the models database for {self.modality} modality.\n"
                    f"Supported models are: {list(self.models_database.keys())}"
                )
                sys.exit(1)
            curr_model = self.models_database.get(model)(model_name=model)
            model_instance = ModelData(
                model=curr_model,
                model_preprocess=curr_model.get_preprocess(),
                model_postprocess=curr_model.get_postprocess()
            )
            model_instance.init_endpoint(endpoint_url=self.triton_endpoint_url, log_level=self.log_level)
            self.models[model] = model_instance

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

    # TODO add dynamic confidence value and
    def triton_infer_image(self, cv_image, model_key: str, filter_class_idx: int=0)->list[np.ndarray]:
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
        if self.models[model_key].dynamic:
            model_input_h, model_input_w = original_h, original_w
        else:
            cv_image, model_input_h, model_input_w = resize(cv_image,
                                                            self.models[model_key].input_metadata)
        # named_window = 'Resized source image'
        # cv2.imshow(named_window, cv_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow(named_window)
        # if self.visualize:
        #     tmp = cv_image.copy()
        self.image = self.models[model_key].model_preprocess.image_adjust(cv_image)
        # convert to input data type the model expects
        self.image = self.image.astype(
            self.input_datatypes[self.models[model_key].input_metadata[0]['dtype']]
        )
        if self.image is not None:
            # Clear previous request data
            self.models[model_key].endpoint.request.ClearField("inputs")
            self.models[model_key].endpoint.request.ClearField("raw_input_contents")

            # Set the input tensor with the correct shape for this specific image
            input_tensor = self.models[model_key].inputs['input_0']
            input_tensor.ClearField("shape")

            # Set the actual shape: [1, 3, height, width] for batch_size=1
            actual_shape = list(self.image.shape)  # Add batch dimension
            input_tensor.shape.extend(actual_shape)

            # Add to request
            self.models[model_key].endpoint.request.inputs.extend([input_tensor])
            self.models[model_key].endpoint.request.raw_input_contents.extend([self.image.tobytes()])

            # Perform inference
            self.models[model_key].endpoint.response = self.models[model_key].endpoint.do_inference()
            self.prediction = self.models[model_key].model_postprocess.extract_boxes(
                self.models[model_key].endpoint.response,
            )
            if len(self.prediction[1]) > 0:
                if not self.models[model_key].dynamic:
                    self.prediction[0] = scale_box_array(
                        self.prediction[0],
                        model_input_dim=(model_input_h, model_input_w),
                        image_dim=(original_h, original_w),
                        padded=True
                    )
                # if self.visualize:
                #     visualize(self.orig_image, self.prediction[0], mode='BGR')
                if self.models[model_key].format == "coco" or self.models[model_key].format == "aitf":
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

    def triton_infer_pointcloud(self,
                                pointcloud:np.ndarray,
                                model_key: str,
                                confidence_threshold: float=0.2,
                                class_idx: int=2)->list[np.ndarray]:
        """
        Perform inference on the point cloud data.
        :param sample: Point cloud data
        :param confidence_threshold: Confidence threshold for filtering predictions
        :param class_idx: Class index for filtering predictions
        :return: Filtered predictions
        """
        self.pc = self.models[model_key].model_preprocess.filter_pc(pointcloud)
        num_voxels = self.pc["voxels"].shape[0]
        self.models[model_key].endpoint.request.ClearField(
            "raw_input_contents"
        )  # Flush the previous sample content
        for key, idx in zip(self.models[model_key].inputs, range(len(self.models[model_key].inputs))):
            tmp_shape = list(self.models[model_key].input_metadata[idx]['shape'])
            tmp_shape[tmp_shape.index(-1)] = num_voxels
            self.models[model_key].inputs[key].ClearField("shape")
            self.models[model_key].endpoint.request.inputs[idx].ClearField("shape")
            self.models[model_key].endpoint.request.inputs[idx].shape.extend(tmp_shape)
            self.models[model_key].inputs[key].shape.extend(tmp_shape)
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
            == self.input_datatypes[self.models[model_key].inputs["input_0"].datatype]
        )
        assert (
            self.pc["voxel_coords"].dtype
            == self.input_datatypes[self.models[model_key].inputs["input_1"].datatype]
        )
        assert (
            self.pc["voxel_num_points"].dtype
            == self.input_datatypes[self.models[model_key].inputs["input_2"].datatype]
        )
        self.models[model_key].endpoint.request.raw_input_contents.extend(
            [
                self.pc["voxels"].tobytes(),
                self.pc["voxel_coords"].tobytes(),
                self.pc["voxel_num_points"].tobytes(),
            ]
        )
        self.models[model_key].endpoint.response = (
            self.models[model_key].endpoint.do_inference()
        )  # perform the channel Inference
        self.prediction = self.models[model_key].model_postprocess.extract_boxes(
            self.models[model_key].endpoint.response
        )



        # Show only persons above given confidence threshold
        # indices = np.where((labels == class_idx) & (scores > confidence_threshold))[0].tolist()
        # box_array, scores, labels =box_array[indices, :], scores[indices], labels[indices]

        if self.visualize:
            self.visualizer.draw_scenes(
                points=self.pc["points"],
                ref_boxes=self.prediction[0],
                ref_scores=self.prediction[1],
                ref_labels=self.prediction[2],
            )

        return self.prediction

    def generate_datumaro_predictions(self,
                                      data: list[dict],
                                      model_key: str)->list[dict]:
        """
        Iterates through the data samples and performs inference on each image sample.
        Adds the predictions to the individual sample as data[sample_idx]['annotations'] in datumaro format.
        Args:
            data (list): List of data samples from SEEREP.

        """
        if self.modality == 'image':
            data_key = 'image'
            infer_function = self.triton_infer_image
            datumaro_processor = DatumaroAnnotation(format=self.models[model_key].format)
            datumaro_converter = datumaro_processor.toCOCO
        elif self.modality == 'pointcloud':
            data_key = 'pointcloud_processed'
            infer_function = self.triton_infer_pointcloud
            datumaro_processor = DatumaroAnnotation(format=self.models[model_key].format)
            datumaro_converter = datumaro_processor.toKITTI

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
                # if model_key in sample['processed']:
                #     # self.processed_counter += 1
                #     logger.info('Skipping sample {} since it was already processed by model {} from previous requests'.format(
                #         sample['uuid'], model_key))
                # else:
                predictions = {
                'annotations':[],
                'dm_format_version':1,
                }
                # perform inference on each image, iteratively
                t3 = time.time()
                pred = infer_function(sample[data_key], model_key=model_key)
                t4 = time.time()
                infer_array[seerep_sample_idx] = t4 - t3
                # traverse the predictions for the current image
                predictions['annotations'] = datumaro_converter(sample=sample,
                                                                model_output=pred,
                                                                sample_idx=seerep_sample_idx,
                                                                visualize=self.visualize,
                                                                class_names=self.models[model_key].class_names,
                                                                model_name=self.models[model_key].model_name)
                data[seerep_sample_idx]['annotations']['items'].append(predictions)
                # Visualize the groundtruth annotations on the same image as predictions
                # if True:
            #     if self.visualize:
            #         visualize(sample, 
            #                 self.models[model_key].model_name, 
            #                 self.models[model_key].class_names,
            #                 # new_model_key=False,
            #                 # model_name=self.models[model_key].model_name,
            #                 save=True
            #                 )
            # if self.visualize:
            #     cv2.destroyWindow(self.winname) 
            logger.info('Processed all inference requests in current data subset!')
        return data

    def generate_annotations_by_sample_uuids(self, sample_uuids: list)->list[dict]:
        """
        Generate annotations for the given sample UUIDs using the Triton inference model.
        Args:
            sample_uuids (list): List of sample UUIDs to generate annotations for.
        Returns:
        """
        data = self.seerep_endpoint.fetch_data_by_sample_uuid(sample_uuids, model_name=self.model_names)
        data = self.generate_datumaro_predictions(data)
        data = self.seerep_endpoint.send_dataset(uuids=sample_uuids,
                                                data=data,
                                                category=self.model_names)
        return data

    def fetch_data_by_sample_uuids(self, sample_uuids: list[str])->list[dict]:
        """
        Fetch data from the SEEREP server for the given sample UUIDs.
        Args:
            sample_uuids (list): List of sample UUIDs to fetch data for.
        Returns:
            list: List of data samples.
        """
        return self.seerep_endpoint.fetch_data_by_sample_uuid(sample_uuids, model_name=self.model_names)

    def request_triton_inference(self, data: list[dict])->list[dict]:
        """
        Perform inference on the given data using the Triton inference model.
        Args:
            data (list): List of data samples to perform inference on.
        Returns:
            list: List of data samples with predictions.
        """
        return self.generate_datumaro_predictions(data)

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
        data = self.seerep_endpoint.fetch_data_by_project(project_uuids,
                                                    model_name=self.model_names)
        # NOTE! This is a temporary fix to fetch data by sample uuids. UUIDs will be fetched directly inside the Triton class.
        sample_uuids = self.seerep_endpoint.fetch_uuids_by_project_uuid(project_uuids)
        data = self.seerep_endpoint.fetch_data_by_sample_uuid(sample_uuids, model_name=self.model_names)
        data = self.generate_datumaro_predictions(data)

        return data
