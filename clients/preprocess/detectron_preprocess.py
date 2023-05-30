from .base_preprocess import Preprocess
import numpy as np

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
        cv_image = np.transpose(cv_image, (2, 0, 1)).astype(np.float32)
        cv_image = np.expand_dims(cv_image, axis=0)
        # cv_image /= 255.0

        return cv_image.astype(np.float32)