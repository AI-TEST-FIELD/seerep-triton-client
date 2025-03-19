from .base_model import Model
from .preprocess import Detectron2_preprocess
from .postprocess import Detectron2_Postprocess

class Detectron2_Det(Model):
    """

    """
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
    def register_client(self, clienttype, client):
        """
        Implement the method to register the client for
        """
        self._clients[clienttype] = client

    def get_preprocess(self):
        return Detectron2_preprocess()

    def get_postprocess(self):
        return Detectron2_Postprocess()