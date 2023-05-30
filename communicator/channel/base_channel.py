from abc import ABC, abstractmethod

class BaseChannel(ABC):
    """
    Basic representation of channels for triton client
    """
    def __init__(self,FLAGS):
        self.channel_type = {}
        self.FLAGS = FLAGS

    @abstractmethod
    def register_channel(self):
        """
        register channel dynamically
        """

    @abstractmethod
    def fetch_channel(self):
        """
        fetch instance of channel
        """

    @abstractmethod
    def get_metadata(self):
        """
        fetch metadata for grpc channel and model specified
        @return:
        """
    @abstractmethod
    def do_inference(self):
        """
        perform inference based on channel implementation
        """
