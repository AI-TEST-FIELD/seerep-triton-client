import numpy as np
from .base_preprocess import Preprocess

class MMDet_preprocess(Preprocess):

    def __init__(self):
        pass

    def preprocess(self):
        pass

    def image_adjust(self, cv_image):
        '''
        cv_image: input image in RGB order
        return: RGB Image in NCHW dimensions.
        '''
        orig = cv_image.copy()
        # orig = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        orig = np.transpose(orig, (2, 0, 1))  # Channel first
        orig = np.expand_dims(orig, axis=0)  # Added batch dimension
        return orig