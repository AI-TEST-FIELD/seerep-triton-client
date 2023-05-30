from .base_client import Client
from .preprocess import FCOSpreprocess
from .postprocess import FCOSpostprocess
class FCOS_client(Client):
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
        return FCOSpreprocess()

    def get_postprocess(self):
        return FCOSpostprocess()