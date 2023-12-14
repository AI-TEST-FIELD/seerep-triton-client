from .base_postprocess import Postprocess
import numpy as np
# import struct
# import os
# import math
# import time
# import torch
# import torchvision

class PCDetPostprocess(Postprocess):
    def __init__(self):
        pass

    def postprocess(self):
        pass

    def load_class_names(self, namesfile='/opt/client/config/kitti.names', dataset='KITTI'):
            """
            Load class names from a file.

            Args:
                namesfile (str): Path to the file containing class names.
                dataset (str): Name of the dataset.

            Returns:
                list: List of class names.
            """
            class_names = []
            with open(namesfile, 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                class_names.append(line)
            return class_names

    def extract_boxes(self, prediction):
        """
        Extracts boxes, scores, and classes from the prediction.

        Args:
            prediction (Prediction): The prediction object containing raw output contents and output metadata.
            
        Returns:
            tuple: A tuple containing the extracted boxes, scores, and classes.
        """
        outputs = []
        for output_idx in range(len(prediction.raw_output_contents)):
            outputs.append(self.deserialize_bytes(prediction.raw_output_contents[output_idx], 
                                                  prediction.outputs[output_idx].datatype))
            outputs[output_idx] = np.reshape(outputs[output_idx], 
                                             prediction.outputs[output_idx].shape)
        # TODO make this dynamic? 
        return outputs[0], outputs[1], outputs[2]   # boxes, scores, classes