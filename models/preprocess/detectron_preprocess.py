import numpy as np
import cv2
from .base_preprocess import Preprocess

class Detectron2_preprocess(Preprocess):

    def __init__(self):
        pass

    def preprocess(self):
        pass

    def image_adjust(self, cv_image):
        '''
        cv_image: input image in RGB order
        return: Normalized image in BCHW dimensions.
        '''
        orig = cv_image.copy()
        orig = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        orig = np.transpose(cv_image, (2, 0, 1))    
        # orig = np.expand_dims(cv_image, axis=0)

        return orig