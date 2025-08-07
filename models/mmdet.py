from .base_model import Model
from .preprocess import MMDet_preprocess
from .postprocess import MMDet_postprocess

class MMDet(Model):
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
        return MMDet_preprocess()

    def get_postprocess(self):
        return MMDet_postprocess()