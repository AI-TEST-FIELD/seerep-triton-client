from .base_preprocess import Preprocess
import numpy as np
import cv2

class FCOSpreprocess(Preprocess):

    def __init__(self):
        pass

    def preprocess(self):
        pass

    def image_adjust(self, cv_image):
        '''
        cv_image: input image in RGB order
        return: Normalized image in BCHW dimensions.
        '''
        # pad = np.zeros((16, 1280, 3), dtype=np.uint8)
        # cv_image = np.concatenate((cv_image, pad), axis=0)
        # orig = cv_image.copy()
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv_image = np.transpose(cv_image, (2, 1, 0))
        cv_image = np.expand_dims(cv_image, axis=0)
        # cv_image /= 255.0

        return cv_image