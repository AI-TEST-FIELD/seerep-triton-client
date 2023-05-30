from .base_client import Client
from .preprocess import PointpillarPreprocess
from .postprocess import PointPillarPostprocess

class Pointpillars_client(Client):
    """

    """
    def __init__(self, ):
        super().__init__()

    def register_client(self, clienttype, client):
        """
        Implement the method to register the client for
        """
        self._clients[clienttype] = client

    def get_preprocess(self):
        return PointpillarPreprocess()

    def get_postprocess(self):
        return PointPillarPostprocess()