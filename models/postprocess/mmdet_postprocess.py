import sys
import numpy as np
from .base_postprocess import Postprocess

import os

class MMDet_postprocess(Postprocess):
    def __init__(self):
        pass

    def postprocess(self):
        pass

    def load_class_names(self, dataset='COCO'):
        dirname = os.path.dirname(__file__)
        if dataset=='COCO':
            namesfile = os.path.join(dirname, '../../config/coco.names')
        elif dataset=='CROP':
            namesfile = os.path.join(dirname, '../../config/crop.names')
        else:
            print('[ERROR] No valid dataset was provided. Exiting!')
            sys.exit(0)
        class_names = []
        with open(namesfile, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)
        return class_names

    def extract_boxes(self, prediction, confidence=0.4):
        """Converts raw output to detections with confidence (default=0.4) thresholding. 
            Returns:
                 list of detections, on (n,4) tensor per image [xyxy, conf, cls]
        """
        boxes = self.deserialize_bytes_float(prediction.raw_output_contents[0])
        boxes = np.reshape(boxes, prediction.outputs[0].shape)
        labels = self.deserialize_bytes_int(prediction.raw_output_contents[2])
        labels = np.reshape(labels, prediction.outputs[2].shape)
        scores = self.deserialize_bytes_float(prediction.raw_output_contents[1])
        scores = np.reshape(scores, prediction.outputs[1].shape)
        # conf_inds = np.where(scores > confidence)
        return [boxes, labels, scores]