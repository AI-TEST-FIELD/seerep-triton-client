# import time
import cv2
import time
import numpy as np
import sys
import logging

from tqdm import tqdm
# TODO move this to triton end point
from tritonclient.grpc import service_pb2
from .endpoint import triton_endpoint
from communicator.endpoint.seerep_endpoint import SeerepEndpoint
from logger import Client_logger, TqdmToLogger
from utils import (
    resize,
    scale_box_array,
    process_model_output,
    visualize,
)

logger = Client_logger(name="Triton-Client", level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger, level=logging.INFO)

class EvaluateInference:
    """
    This class automatically fetches data from given SEEREP args.seerep_project
    at args.channel_seerep and performs inference on the images using the Triton server
    at args.channel_triton. It uses the gRPC channel.
    """

    def __init__(self, 
                 triton_stub, 
                 model, 
                 log_level="error", 
                 visualize=False):
        """
        channel: channel of type communicator.channel
        client: client of type clients
        """
        self.model = model
        self.model_name = model.model_name
        self.model_postprocess = model.get_postprocess()
        self.model_preprocess = model.get_preprocess()
        self.triton_channel = triton_stub
        self.seerep_endpoint = 'agrigaia-ur.ni.dfki:9090'
        self._initialize_model_inputs()
        self._initialize_misc()
        self.format = format
        # Initialize miscellaneous parameters

        logging.basicConfig(level=self.log_level[log_level])
        self.visualize = visualize
        if self.visualize:
            self.winname = "Prediction {}".format(self.model_name)
            cv2.namedWindow(self.winname)
        self.processed_counter = 0
        self.no_gt_counter = 0

    def _initialize_model_inputs(self):
        """
        Register the GRPC endpoint for Triton server and fetch model metadata and configuration.
        HTTP endpoint is not supported yet. 
        """
        # for GRPC channel
        if type(self.triton_channel) == triton_endpoint.TritonEndpoint:
            self._configure_model_params()
        else:
            sys.exit(1)

    def _initialize_misc(self):
        """
        Initialize miscellaneous parameters for the triton inference
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
        elif "CROP" in self.model_name:
            self.class_names = self.model_postprocess.load_class_names(dataset="CROP")
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
        meta_data = self.triton_channel.get_metadata()

        # parse the model requirements from client
        self.input_metadata, self.output_metadata = self.model.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config
        )
        self.triton_channel.input = [input["name"] for input in self.input_metadata]
        self.triton_channel.output = [output["name"] for output in self.output_metadata]

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
            self.triton_channel.request.inputs.extend([self.inputs["input_{}".format(i)]])

        self.outputs = {}
        for output, i in zip(self.output_metadata, range(len(self.output_metadata))):
            self.outputs[
                "output_{}".format(i)
            ] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs["output_{}".format(i)].name = output["name"]
            # assign the gathered model outputs to the grpc channel
            self.triton_channel.request.outputs.extend([self.outputs["output_{}".format(i)]])

    def triton_infer_image(self, cv_image):
        """
        Perform inference on the images using Triton server
        """
        self.orig_image = cv_image.copy()
        original_h, original_w = cv_image.shape[0], cv_image.shape[1]
        cv_image, model_input_h, model_input_w = resize(cv_image, self.input_metadata)
        # named_window = 'Resized source image'
        # cv2.imshow(named_window, cv_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow(named_window)
        if self.visualize:
            tmp = cv_image.copy()
        self.image = self.model_preprocess.image_adjust(cv_image)
        # convert to input data type the model expects
        self.image = self.image.astype(
            self.input_datatypes[self.input_metadata[0]['dtype']]
        )
        if self.image is not None:
            self.triton_channel.request.ClearField("inputs")
            self.triton_channel.request.ClearField("raw_input_contents")  # Flush the previous image contents
            self.triton_channel.request.inputs.extend([self.inputs['input_0']])
            self.triton_channel.request.raw_input_contents.extend([self.image.tobytes()])
            self.triton_channel.response = self.triton_channel.do_inference()  # Inference
            self.prediction = self.model_postprocess.extract_boxes(
                self.triton_channel.response,
            )
            if len(self.prediction[1]) > 0:
                self.prediction[0] = scale_box_array(
                    self.prediction[0], 
                    model_input_dim=(model_input_h, model_input_w), 
                    image_dim=(original_h, original_w), 
                    padded=True
                )
                # if self.visualize:
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

    # TODO This function needs to be optimized for large number of samples.
    def postprocess_seerep_data(self, data):
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
                unit="requests",
                ascii=True,
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
                    pred = self.triton_infer_image(sample["image"])
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
            logger.info('{} were skipped since predictions were already stored from previous runs'.format(self.processed_counter))
            logger.info('{} image had no ground truth associated with them.'.format(self.no_gt_counter))
        return data

    def start_inference(self, model_name, modality="images"):
        seerep_channel = SeerepEndpoint(
            endpoint_url=self.seerep_endpoint,
            modality=modality,
            # visualize=self.visualize,
        )
        if True:
            project_uuid = seerep_channel.get_project_uuid('EV41_Kleidung_Sonnenbrille_DunkleCap_225Deg_2024-10-10-18-58-13_0')
            # data = seerep_channel.fetch_data_by_project([project_uuid], 
            #                                             model_name=self.model_name)
            # NOTE! This is a temporary fix to fetch data by sample uuids. UUIDs will be fetched directly inside the Triton class.
            uuids = seerep_channel.fetch_uuids_by_project_uuid([project_uuid])
            data = seerep_channel.fetch_data_by_sample_uuid(uuids, model_name=self.model_name)
            data = self.postprocess_seerep_data(data)
            # Send predictions back to SEEREP for future use
            # seerep_channel.send_dataset(data, category=self.model_name)
        elif False:
            buffer = seerep_channel.fetch_data_by_sample(['41e13b76-a890-41e2-acf0-cb415b9cd546',
                                                        '06ced1d2-e0cc-4254-b56c-aa27532e66fe'], 
                                                        model_name=self.model_name)
            data = seerep_channel.process_images(buffer, model_name)
            # Send predictions back to SEEREP for future use
            # seerep_channel.send_dataset(data, category=self.model_name)
