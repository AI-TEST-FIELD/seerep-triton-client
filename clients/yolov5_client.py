from .base_client import Client
from .preprocess import yolov5_preprocess
from .postprocess import yolov5_postprocess
# import tritonclient.grpc.model_config_pb2 as mc

class Yolov5client(Client):
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
        return yolov5_preprocess.Yolov5preprocess()

    def get_postprocess(self):
        return yolov5_postprocess.Yolov5postprocess()
