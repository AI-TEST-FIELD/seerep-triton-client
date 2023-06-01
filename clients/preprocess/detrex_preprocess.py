from .base_preprocess import Preprocess
import numpy as np
import cv2

class Detrexpreprocess(Preprocess):

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
        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        # channels first
        orig = np.transpose(orig, (2, 0, 1))
        orig = np.expand_dims(orig, axis=0)
        # orig = orig / 255

        return orig