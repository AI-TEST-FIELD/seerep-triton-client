# from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import struct
import os
import math
import time
import torch
import torchvision

numpy_datatypes = {
    'b': np.dtype(np.int8),    # char
    'B': np.dtype(np.uint8),   # unsigned char
    'h': np.dtype(np.int16),   # short
    'H': np.dtype(np.uint16),  # unsigned short
    'l': np.dtype(np.int32),   # long
    'L': np.dtype(np.uint32),  # unsigned long
    'q': np.dtype(np.int64),   # long long
    'Q': np.dtype(np.uint64),  # unsigned long long
    'e': np.dtype(np.float16),            # half
    'f': np.dtype(np.float32),            # float 
    'd': np.dtype(np.float64),            # double
    's': np.dtype((np.str_, 1)),          # char[]
    '?': np.dtype(np.bool_),              # bool
    }

datatypes_dict = {
    'INT8':'b',     # char
    'UINT8':'B',   # unsigned char
    'INT16':'h',   # short
    'UINT16':'H',  # unsigned short
    'INT32':'l',   # long
    'UINT32':'L',  # unsigned long
    'INT64':'q',   # long long
    'UINT64':'Q',  # unsigned long long
    'FP16':'e',            # half
    'FP32':'f',            # float 
    'FP64':'d',            # double
    'STR':'s',          # char[]
    'BOOL':'?',              # bool
    }


class Postprocess(ABC):
    """

    """
    def deserialize_bytes_float(self, encoded_tensor):
            """
            Deserialize a byte-encoded tensor into a numpy array of floats.

            Args:
                encoded_tensor (bytes): The byte-encoded tensor.

            Returns:
                numpy.ndarray: The deserialized numpy array of floats.
            """
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
            """
            Deserialize a byte-encoded tensor into an array of integers.

            Args:
                encoded_tensor (bytes): The byte-encoded tensor.

            Returns:
                np.ndarray: An array of integers.
            """
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

    def deserialize_bytes(self, encoded_tensor, data_format="f"):
        """
        Deserialize a byte-encoded tensor into a numpy array.

        Args:
            encoded_tensor (bytes): The byte-encoded tensor.
            data_format (str, optional): The format of the data. Defaults to "f".

        Returns:
            numpy.ndarray: The deserialized numpy array.
        """
        strs = list()
        offset = 0
        val_buf = encoded_tensor
        datatype = datatypes_dict[data_format]
        byte_size = struct.calcsize(datatype)
        while offset < len(val_buf):
            sb = struct.unpack_from(datatype, val_buf, offset)[0]
            offset += byte_size
            strs.append(sb)
        return np.array(strs, dtype=numpy_datatypes[datatype])
    
    def xywh2xyxy(self,x):
            """
            Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            
            Args:
                x (torch.Tensor or numpy.ndarray): Input boxes in [x, y, w, h] format
                
            Returns:
                torch.Tensor or numpy.ndarray: Converted boxes in [x1, y1, x2, y2] format
            """
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
            y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
            y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
            return y

    def box_iou(self, box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

    def nms_cpu(self, boxes, confs, nms_thresh=0.5, min_mode=False):
        # print(boxes.shape)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]

        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]

        return np.array(keep)

    @abstractmethod
    def postprocess(self):
        """
        """
