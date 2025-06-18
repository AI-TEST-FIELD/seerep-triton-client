import grpc
import logging
import tritonclient.grpc.model_config_pb2 as mc
from tritonclient.grpc import service_pb2, service_pb2_grpc
from logger import Client_logger, TqdmToLogger
logger = Client_logger(name='Triton-Client', level=logging.INFO).get_logger()
tqdm_out = TqdmToLogger(logger,level=logging.INFO)

class TritonEndpoint:
    """
    A TritonEndpoint is responsible for establishing connection between client and triton server using gRPC only.
    """

    def __init__(self,
                 model_name='yolov5m_coco',
                 model_version='1',
                 endpoint_url='10.249.6.4:8001',
                 batch_size=1,
                 grpc_packet_size=17671546,
                 log_level='error'):
        self._meta_data = {}
        self._grpc_stub = None
        self.endpoint_url = endpoint_url
        self.batch_size = batch_size 
        self.packet_size = grpc_packet_size
        self.model_name = model_name
        self.model_version = model_version
        # self.log_level = log_level

        self.register_grpc_channel() # register and initialise the stub
        self._fetch_model_metadata() #
        if self._meta_data is not None:
            logger.info("Triton Channel initialized successfully for model {} at endpoint: {}".format(self.model_name, self.endpoint_url))
        else:
            logger.error("Triton Channel initialization failed for model {} at endpoint: {}".format(self.model_name, self.endpoint_url))
            raise Exception("Triton Channel initialization failed for model {} at endpoint: {}".format(self.model_name, self.endpoint_url))

    def register_grpc_channel(self):
        """
        Register the connection via the gRPC endpoint of the Triton server
        """
        grpc_channel =  grpc.insecure_channel(self.endpoint_url ,options=[
                                   ('grpc.max_send_message_length', self.batch_size*self.packet_size),
                                   ('grpc.max_receive_message_length', self.batch_size*self.packet_size),
                                    ])
        self._grpc_stub  = service_pb2_grpc.GRPCInferenceServiceStub(grpc_channel)

    def fetch_channel(self):
        """
        return grpc stub
        """
        return self._grpc_stub

    def _fetch_model_metadata(self):
        """
        Initiate all meta data required for models
        """
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        self._meta_data["metadata_request"] = service_pb2.ModelMetadataRequest(
            name=self.model_name, version=self.model_version)
        self._meta_data["metadata_response"] = self._grpc_stub.ModelMetadata(self._meta_data["metadata_request"])

        self._meta_data["config_request"] = service_pb2.ModelConfigRequest(name=self.model_name,
                                                                           version=self.model_version)
        self._meta_data["config_response"] = self._grpc_stub.ModelConfig(self._meta_data["config_request"])

        # set essential grpc members
        self._init_model_io()

    def get_metadata(self):
        """
        return meta_data dictionary form
        @rtype: dictionary
        """
        return self._meta_data

    def _init_model_io(self):
        """
        set essential grpc members
        """
        self.input = service_pb2.ModelInferRequest().InferInputTensor()
        self.request = service_pb2.ModelInferRequest()
        self.request.model_name = self.model_name
        self.request.model_version = self.model_version
        self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()

    def do_inference(self):
        """
        inference based on grpc_stud
        @return: inference of grpc
        """
        return self._grpc_stub.ModelInfer(self.request)