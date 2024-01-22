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

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc
from .channel import grpc_channel
from .base_inference import BaseInference
from communicator.channel import seerep_channel
from tools.seerep2coco import COCO_SEEREP
from logger import Client_logger, TqdmToLogger
from visual_utils import open3d_vis_utils as visualizer

logger = Client_logger(name='Triton-Client', level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)

class EvaluateInference(BaseInference):

    """
    A RosInference to support ROS input and provide input to channel for inference.
    """

    def __init__(self, args, channel, client, format='coco'):
        '''
            channel: channel of type communicator.channel
            client: client of type clients

        '''

        super().__init__(channel, client)

        self.image = None
        # self.class_names = self.load_class_names()
        self.args = args
        self._register_inference() # register inference based on type of client
        self.client_postprocess = client.get_postprocess() # get postprocess of client
        self.client_preprocess = client.get_preprocess()
        self.model_name = client.model_name
        self.format=format
        if 'COCO' or 'coco' in self.model_name:
            self.class_names = self.client_postprocess.load_class_names(dataset='COCO')
        elif 'CROP' in self.model_name:
            self.class_names = self.client_postprocess.load_class_names(dataset='CROP')
        else:
            # TODO shutdown? 
            self.class_names=None
        self.input_datatypes = {
            'UINT8': np.dtype(np.uint8),
            'INT16': np.dtype(np.int16),
            'INT32': np.dtype(np.int32),
            'FP16':np.dtype(np.float16),
            'FP32': np.dtype(np.float32),
            'FP64': np.dtype(np.float64),
        }
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'debug':logging.DEBUG,
            'critical':logging.CRITICAL,
        }
        logging.basicConfig(level=log_level[args.log_level])
        self.viz = args.visualize
        self.count  = 0
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
            meta_data["metadata_response"], meta_data["config_response"].config)
        self.channel.input = [input['name'] for input in input_metadata]
        self.channel.output = [output['name'] for output in output_metadata]

        self.inputs = {}
        for input, i in zip(input_metadata, range(len(input_metadata))):
            self.inputs['input_{}'.format(i)] = service_pb2.ModelInferRequest().InferInputTensor()
            self.inputs['input_{}'.format(i)].name = input['name']
            self.inputs['input_{}'.format(i)].datatype = input['dtype']
            if -1 in input['shape']:
                input['shape'][0] = 10000           # tmp
            self.inputs['input_{}'.format(i)].shape.extend(input['shape'])
            # assign the gathered model inputs to the grpc channel
            self.channel.request.inputs.extend([self.inputs['input_{}'.format(i)]])

        self.outputs = {}
        for output,i  in zip(output_metadata, range(len(output_metadata))):
            self.outputs['output_{}'.format(i)] = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.outputs['output_{}'.format(i)].name = output['name']
            # assign the gathered model outputs to the grpc channel
            self.channel.request.outputs.extend([self.outputs['output_{}'.format(i)]])

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
        n_h, n_w = self.channel.input.shape[1], self.channel.input.shape[2]
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
        padded_image[0:cv_image.shape[0], 0:cv_image.shape[1]] = cv_image
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
        n_h, n_w = self.channel.input.shape[1], self.channel.input.shape[2]
        cv_image, r_h, r_w = self.resize(cv_image)
        # named_window = 'Resized source image'
        # cv2.imshow(named_window, cv_image)
        # cv2.waitKey()
        # cv2.destroyWindow(named_window)
        tmp = cv_image.copy()
        self.image = self.client_preprocess.image_adjust(cv_image)
        # convert to input data type the model expects
        self.image = self.image.astype(self.input_datatypes[self.channel.input.datatype]) 
        if self.image is not None:
            self.channel.request.ClearField("inputs")
            self.channel.request.ClearField("raw_input_contents")   # Flush the previous image contents
            self.channel.request.inputs.extend([self.channel.input])
            self.channel.request.raw_input_contents.extend([self.image.tobytes()])
            self.channel.response = self.channel.do_inference()  # Inference
            self.prediction = self.client_postprocess.extract_boxes(self.channel.response)
            if len(self.prediction[1]) > 0:
                if self.viz:
                    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR).astype(np.uint8)
                    for box in self.prediction[0]:
                        cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                    named_window = 'Resized source image with prediction'
                    cv2.imshow(named_window, tmp)
                    cv2.waitKey()
                    cv2.destroyWindow(named_window)
                self.prediction[0] = self._scale_box_array(self.prediction[0],  
                                                           source_dim=(r_h, r_w),
                                                           padded=True)
                if self.format=='kitti':
                    persons = np.where(self.prediction[1] == 0) # filter Pedestrians
                    return self.prediction[0][persons], self.prediction[1][persons], self.prediction[2][persons]
                elif self.format=='coco':
                    persons = np.where(self.prediction[1] == 0) # filter persons
                    return self.prediction[0][persons], self.prediction[1][persons], self.prediction[2][persons]
                else:
                    return self.prediction
            else:
                return self.prediction
    
    def rotate_pc(self, pointclouds: np.array)-> np.array:
        tx_pc = pointclouds.copy()
        pc = tx_pc[:, 0:3]
        from scipy.spatial.transform import Rotation as R
        from math import cos, sin
        ry = R.from_euler('y', 30, degrees=True).as_matrix()
        rz = R.from_euler('z', 90, degrees=True).as_matrix()
        # rotation_matrix = np.array([[cos(angle), 0, sin(angle)], 
        #                             [0, 1, 0], 
        #                             [-sin(angle), 0, cos(angle)]])
        # rotation_matrix = np.array([[0.82638931, -0.02497454,  0.56254509], 
        #                             [0.01212522,  0.99957356,  0.02656451], 
        #                             [-0.56296864, -0.01513165, 0.82633973]])
        pc = np.matmul(ry, pc.T).T
        pc = np.matmul(rz, pc.T).T
        pc += [0., 0., -1.026558971]
        tx_pc[:, 0:3] = pc 
        return tx_pc

    def seerep_infer_pc(self, pointclouds: np.array):
        num_points = pointclouds['x']['data'].shape[0]
        # Keep only x,y,z i.e. 3 dims 
        self.pc = np.zeros((num_points, 4), dtype=np.float32)
        self.pc[:, 0] = pointclouds['x']['data'][:, 0]
        self.pc[:, 1] = pointclouds['y']['data'][:, 0]
        self.pc[:, 2] = pointclouds['z']['data'][:, 0]
        self.pc[:, 3] = pointclouds['intensity']['data'][:, 0] / np.max(pointclouds['intensity']['data'][:, 0])
        # self.pc[:, 3] = pointclouds['intensity']['data'][:, 0]
        # self.pc = self.rotate_pc(self.pc)
        # DEBUG only
        # with open('000076.bin', 'rb') as f:
        #     self.pc = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        self.pc = self.client_preprocess.filter_pc(self.pc)
        num_voxels = self.pc['voxels'].shape[0]
        self.channel.request.ClearField("raw_input_contents")  # Flush the previous sample content
        for key, idx in zip(self.inputs, range(len(self.inputs))):
            tmp_shape = self.inputs[key].shape
            self.inputs[key].ClearField("shape")
            tmp_shape[0] = num_voxels
            self.channel.request.inputs[idx].ClearField("shape")
            self.channel.request.inputs[idx].shape.extend(tmp_shape)
            self.inputs[key].shape.extend(tmp_shape)
        # Insert batch dimensions into the voxel coordinates------>change from N x 3 to N x 3+1. Assume batch size 1
        tmp_data = np.zeros((self.pc['voxel_coords'].shape[0],
                             self.pc['voxel_coords'].shape[1]+1),
                             dtype=self.pc['voxel_coords'].dtype)
        tmp_data[:, 1:] = self.pc['voxel_coords'].copy()
        self.pc['voxel_coords'] = tmp_data.copy()
        del tmp_data
        # Make sure the data types and shapes are correct for each input before sending them as bytes, this causes wrong array values on the server 
        assert self.pc['voxels'].dtype == self.input_datatypes[self.inputs['input_0'].datatype]
        assert self.pc['voxel_coords'].dtype == self.input_datatypes[self.inputs['input_1'].datatype]
        assert self.pc['voxel_num_points'].dtype == self.input_datatypes[self.inputs['input_2'].datatype]
        self.channel.request.raw_input_contents.extend([self.pc['voxels'].tobytes(),
                                                        self.pc['voxel_coords'].tobytes(),
                                                        self.pc['voxel_num_points'].tobytes(),
                                                        ])
        self.channel.response = self.channel.do_inference() # perform the channel Inference
        box_array, scores, labels = self.client_postprocess.extract_boxes(self.channel.response)
        # Show only persons above given confidence threshold
        indices = np.where((labels == 2) & (scores > 0.4))[0].tolist()
        # indices = [i for i in range(len(labels))]
        if True:
            visualizer.draw_scenes(points=self.pc['points'],
                                   ref_boxes=box_array[indices, :],
                                   ref_scores=scores[indices],
                                   ref_labels=labels[indices])                      

    def process_images(self, data, seerep_channel: seerep_channel.SEEREPChannel):
        t2 = time.time()
        if len(data) == 0:
            logger.critical('No data samples found in the SEEREP database matching your query')
        else:
            # logger.info('Fetching time: {} s'.format(np.round(t2 - t1, 3)))
            color1 = (255, 0, 0)    #red
            color2 = (255, 255, 255)    #green
            text_color = (255, 255, 255)
            # traverse through the images
            # logger.info('Sending inference request to Triton for each sample')
            infer_array = np.zeros(len(data), dtype=np.float16)
            for sample, idx in tqdm(zip(data, range(len(data))), 
                                    total=len(data),
                                    colour='GREEN',
                                    desc='Sending inference request to Triton',
                                    unit='requests',
                                    ascii=True):
                # perform an inference on each image, iteratively
                t3 = time.time()
                pred = self.seerep_infer_image(sample['images'])
                t4 = time.time()
                infer_array[idx] = t4 - t3
                # logger.info('Inference time: {}'.format(t4 - t3))
                sample['predictions'] = []
                bbs = []
                labels = []
                confidences = []
                # traverse the predictions for the current image
                for obj in range(len(pred[1])):
                    start_cord, end_cord = (pred[0][obj, 0], pred[0][obj, 1]), (pred[0][obj, 2], pred[0][obj, 3])
                    x, y, w, h = (start_cord[0] + end_cord[0]) / 2, (start_cord[1] + end_cord[1])/2, end_cord[0] - start_cord[0], end_cord[1] - start_cord[1]
                    assert x>0 and y>0 and w>0 and h>0
                    label = self.class_names[int(pred[1][obj])]
                    confidences.append(pred[2][obj])
                    bbs.append(((x,y), (w,h)))     # SEEREP expects center x,y and width, height
                    labels.append(label)
                    data[idx]['predictions'].append([start_cord[0], start_cord[1], w, h, pred[1][obj], pred[2][obj]])
                if self.viz:
                    (tw, th), _ = cv2.getTextSize('{} {} %'.format(label, round(pred[2][obj]*100, 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(sample['image'], (int(pred[0][obj, 0]), int(pred[0][obj, 1])), (int(pred[0][obj, 2]), int(pred[0][obj, 3])), color1, 2)
                    cv2.rectangle(sample['image'], (int(start_cord[0]), int(start_cord[1] - 25)), (int(start_cord[0] + tw), int(start_cord[1])), color1, -1)
                    cv2.putText(sample['image'], '{} {} %'.format(label, round(pred[2][obj], 2)*100), (int(pred[0][obj, 0]), int(pred[0][obj, 1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                    # sample['predictions'].append([x, y, w, h, pred[1][obj], pred[2][obj]])
                    # for obj in range(       
                    #     cv2.putText(sample['image'], '{} {} %'.format(label, round(pred[2][obj], 2)*100), (int(pred[0][obj, 0]), int(pred[0][obj, 1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                    winname = 'Predicted image number {}'.format(idx+1)
                    cv2.namedWindow(winname)  
                    cv2.imshow(winname, cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
                    cv2.moveWindow(winname, 0,0)
                    cv2.waitKey()
                    cv2.destroyWindow(winname)
                # cv2.imwrite('./rainy/image_{}.png'.format(idx), cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
                # TODO run evaluation without inference call
                # schan.sendboundingbox(sample, bbs, labels, confidences, self.model_name+'2')
                # logger.info('Sent boxes for image under category name {}'.format(self.model_name))

    def process_pc(self, data, seerep_channel: seerep_channel.SEEREPChannel):
        # traverse through the samples
        infer_array = np.zeros(len(data), dtype=np.float16)
        for sample, idx in tqdm(zip(data, range(len(data))), 
                                total=len(data),
                                colour='GREEN',
                                # file=tqdm_out,
                                desc='Sending inference request to Triton',
                                unit='request'):
            # perform an inference on each image, iteratively
            t3 = time.time()
            pred = self.seerep_infer_pc(sample['point_cloud'])
            t4 = time.time()
            infer_array[idx] = t4 - t3
            # logger.info('Inference time: {}'.format(t4 - t3))
            sample['predictions'] = []
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

    def start_inference(self, model_name, format='coco'):
        schan = seerep_channel.SEEREPChannel(project_name=self.args.seerep_project,
                                            socket=self.args.channel_seerep,
                                            format=self.format,     #TODO make it dynamic with Source_Kitti
                                            visualize=self.viz)
                                            # socket='localhost:9090')

        # TODO! make a decision based on Images or PointCloud or both for selecting service stubs
        sample_type = 'point_clouds'
        if sample_type == 'image':
            data = schan.run_query_images(self.args.semantics)
            self.process_images(data, schan)
        elif sample_type == 'point_clouds':
            data = schan.run_query_pointclouds(self.args.semantics)
            self.process_pc(data, schan)

    def _scale_boxes(self, box, normalized=False):
        '''
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        '''
        if normalized:
            # TODO make it dynamic with mc.Modelshape according to CHW or HWC
            xtl, xbr = box[0] * self.orig_size[1], box[2] * self.orig_size[1]
            ytl, ybr = box[1] * self.orig_size[0], box[3] * self.orig_size[0]
        else:
            xtl, xbr = box[0] * (self.orig_size[1] / self.input_size[0]), \
                       box[2] * (self.orig_size[1] / self.input_size[0])
            ytl, ybr = box[1] * self.orig_size[0] / self.input_size[1], \
                       box[3] * self.orig_size[0] / self.input_size[1]

        return [xtl, ytl, xbr, ybr]

    def _scale_box_array(self, box, source_dim=(512,512), padded=False):
        '''
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        '''
        # if normalized:
        #     # TODO make it dynamic with mc.Modelshape according to CHW or HWC
        #     xtl, xbr = box[0] * self.orig_size[1], box[2] * self.orig_size[1]
        #     ytl, ybr = box[1] * self.orig_size[0], box[3] * self.orig_size[0]
        if padded:
            xtl, xbr = box[:, 0] * (self.orig_size[1] / source_dim[1]), \
                       box[:, 2] * (self.orig_size[1] / source_dim[1])
            ytl, ybr = box[:, 1] * self.orig_size[0] / source_dim[0], \
                       box[:, 3] * self.orig_size[0] / source_dim[0]
        else:
            xtl, xbr = box[:, 0] * (self.orig_size[1] / self.input_size[1]), \
                       box[:, 2] * (self.orig_size[1] / self.input_size[1])
            ytl, ybr = box[:, 1] * self.orig_size[0] / self.input_size[0], \
                       box[:, 3] * self.orig_size[0] / self.input_size[0]
        xtl = np.reshape(xtl, (len(xtl), 1))
        xbr = np.reshape(xbr, (len(xbr), 1))

        ytl = np.reshape(ytl, (len(ytl), 1))
        ybr = np.reshape(ybr, (len(ybr), 1))
        return np.concatenate((xtl, ytl, xbr, ybr, box[:, 4:6]), axis=1)