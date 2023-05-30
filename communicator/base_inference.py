import grpc
from abc import ABC, abstractmethod
from tritonclient.grpc import service_pb2, service_pb2_grpc
from communicator.channel import base_channel

class BaseInference(ABC):

    def __init__(self, channel:base_channel.BaseChannel, client):
        self.channel = channel
        self.client = client
