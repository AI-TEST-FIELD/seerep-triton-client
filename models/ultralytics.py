from .base_model import Model
from .preprocess import Ultralytics_preprocess
from .postprocess import Ultralytics_postprocess

class Ultralytics(Model):
    """
    """

    def __init__(self, model_name='None'):
        super().__init__()
        self.model_name = model_name

    def register_client(self, clienttype, client):
        """
        Implement the method to register the client for
        """
        self._clients[clienttype] = client

    def get_preprocess(self):
        return Ultralytics_preprocess()

    def get_postprocess(self):
        return Ultralytics_postprocess()
