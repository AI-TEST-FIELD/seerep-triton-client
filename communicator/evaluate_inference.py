# import time
import cv2
import imutils
import time
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import logging
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc
from .channel import grpc_channel
from .base_inference import BaseInference
from communicator.channel import seerep_channel
from tools.seerep2coco import COCO_SEEREP
# from utils import image_util
logger = logging.getLogger("Triton-Client")
class EvaluateInference(BaseInference):

    """
    A RosInference to support ROS input and provide input to channel for inference.
    """

    def __init__(self, args, channel, client):
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
        if 'COCO' or 'coco' in self.model_name:
            self.class_names = self.client_postprocess.load_class_names(dataset='COCO')
        elif 'CROP' in self.model_name:
            self.class_names = self.client_postprocess.load_class_names(dataset='CROP')
        else:
            # TODO shutdown? 
            self.class_names=None
        self.input_datatypes = {
            'UINT8': np.uint8,
            'FP32': np.float32,
            'FP16':np.float16,
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
        self.channel.input.name, output_name, c, h, w, format, self.channel.input.datatype = self.client.parse_model(
            meta_data["metadata_response"], meta_data["config_response"].config)

        self.input_size = [h, w]
        if format == mc.ModelInput.FORMAT_NHWC:
            self.channel.input.shape.extend([h, w, c])
        else:
            self.channel.input.shape.extend([c, h, w])

        if len(output_name) > 1:  # Models with multiple outputs Boxes, Classes and scores
            self.output0 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # boxes
            self.output0.name = output_name[0]
            self.output1 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # class_IDs
            self.output1.name = output_name[1]
            self.output2 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # scores
            self.output2.name = output_name[2]
            self.output3 = service_pb2.ModelInferRequest().InferRequestedOutputTensor() # image dims
            self.output3.name = output_name[3]

            self.channel.request.outputs.extend([self.output0,
                                                 self.output1,
                                                 self.output2,
                                                 self.output3])
        else:
            self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            self.output.name = output_name[0]
            self.channel.request.outputs.extend([self.output])
        # self.channel.output.name = output_name[0]
        # self.channel.request.outputs.extend([self.channel.output])

    def resize(self, image):
        # cv_image = cv2.resize(cv_image, (self.channel.input.shape[2], self.channel.input.shape[1]))
        tmp = image.copy()
        s_h, s_w = image.shape[0], image.shape[1]
        n_h, n_w = self.channel.input.shape[1], self.channel.input.shape[2]
        padded_image = np.zeros((n_h, n_w, 3), dtype=np.uint8)
        cv_image = imutils.resize(tmp, width=n_w)
        # padded image
        padded_image[0:cv_image.shape[0], 0:cv_image.shape[1]] = cv_image
        # named_window = 'resized'
        # cv2.imshow(named_window, padded_image)
        # cv2.waitKey()
        # cv2.destroyWindow(named_window)
        return padded_image

    def seerep_infer(self, image):
        # convert numpy array to cv2
        cv_image = image
        self.orig_size = cv_image.shape[0:2]
        self.orig_image = cv_image.copy()
        s_h, s_w = cv_image.shape[0], cv_image.shape[1]
        n_h, n_w = self.channel.input.shape[1], self.channel.input.shape[2]
        cv_image = self.resize(cv_image)
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
                # if self.viz:
                #     tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR).astype(np.uint8)
                #     for box in self.prediction[0]:
                #         cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                #     named_window = 'Resized source image with prediction'
                #     cv2.imshow(named_window, tmp)
                #     cv2.waitKey()
                #     cv2.destroyWindow(named_window)
                self.prediction[0] = self._scale_box_array(self.prediction[0], normalized=False)
                return self.prediction
            else:
                return self.prediction
            
    def start_inference(self, model_name):
        schan = seerep_channel.SEEREPChannel(project_name=self.args.seerep_project,
                                            socket=self.args.channel_seerep,
                                            format='coco',
                                            visualize=self.viz)
                                            # socket='localhost:9090')

        # data = schan.run_query()
        t1 = time.time()
        data = schan.run_query_aitf(self.args.semantics)
        t2 = time.time()
        logger.info('Fetching time: {} s'.format(np.round(t2 - t1, 3)))
        color1 = (255, 0, 0)    #red
        color2 = (255, 255, 255)    #green
        text_color = (255, 255, 255)
        # traverse through the images
        logger.info('Sending inference request to Triton for each image sample')
        infer_array = np.zeros(len(data), dtype=np.float16)
        for sample, idx in zip(data, range(len(data))):
            # perform an inference on each image, iteratively
            t3 = time.time()
            pred = self.seerep_infer(sample['image'])
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
                (tw, th), _ = cv2.getTextSize('{} {} %'.format(label, round(pred[2][obj]*100, 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(sample['image'], (int(pred[0][obj, 0]), int(pred[0][obj, 1])), (int(pred[0][obj, 2]), int(pred[0][obj, 3])), color1, 2)
                cv2.rectangle(sample['image'], (int(start_cord[0]), int(start_cord[1] - 25)), (int(start_cord[0] + tw), int(start_cord[1])), color1, -1)
                cv2.putText(sample['image'], '{} {} %'.format(label, round(pred[2][obj], 2)*100), (int(pred[0][obj, 0]), int(pred[0][obj, 1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                sample['predictions'].append([x, y, w, h, pred[1][obj], pred[2][obj]])
                # for obj in range(       
                    # cv2.putText(sample['image'], '{} {} %'.format(label, round(pred[2][obj], 2)*100), (int(pred[0][obj, 0]), int(pred[0][obj, 1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
            if self.viz:
                winname = 'Predicted image number {}'.format(idx+1)
                cv2.namedWindow(winname)  
                cv2.imshow(winname, cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
                cv2.moveWindow(winname, 0,0)
                cv2.waitKey()
                cv2.destroyWindow(winname)
            # cv2.imwrite('./rainy/image_{}.png'.format(idx), cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            # TODO run evaluation without inference call
            # schan.sendboundingbox(sample, bbs, labels, confidences, self.model_name+'2')
            logger.info('Sent boxes for image under category name {}'.format(self.model_name))
        # Convert groundtruth and predictions to PyCOCO format for evaluation
        logger.info('Average Inference time / image: {} s'.format(np.round(np.sum(infer_array)/len(infer_array), 3)))
        t5 = time.time()
        coco_data = COCO_SEEREP(seerep_data=data)
        cocoEval = COCOeval(coco_data.ground_truth, coco_data.predictions, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        t6 = time.time()
        logger.info('Evaluation time: {} s'.format(np.round(t6 - t5, 3)))
        # self.calculate_metrics()

    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def ap_per_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()  # number of labels
            n_p = i.sum()  # number of predictions

            if n_p == 0 or n_l == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_l + 1e-16)  # recall curve
                r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0:
                        py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)
        # if plot:
        #     plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        #     plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        #     plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        #     plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

        i = f1.mean(0).argmax()  # max F1 index
        return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

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

    def _scale_box_array(self, box, normalized=False):
        '''
        box: Bounding box generated for the image size (e.g. 512 x 512) expected by the model at triton server
        return: Scaled bounding box according to the input image from the ros topic.
        '''
        if normalized:
            # TODO make it dynamic with mc.Modelshape according to CHW or HWC
            xtl, xbr = box[0] * self.orig_size[1], box[2] * self.orig_size[1]
            ytl, ybr = box[1] * self.orig_size[0], box[3] * self.orig_size[0]
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