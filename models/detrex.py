from .base_model import Model
from .preprocess import Detrex_Preprocess
from .postprocess import Detrex_Postprocess

class Detrex_Det(Model):
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
        return Detrex_Preprocess()

    def get_postprocess(self):
        return Detrex_Postprocess()