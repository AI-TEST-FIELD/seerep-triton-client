import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc


class TritonEndpoint:
    """
    A TritonEndpoint is responsible for establishing connection between client and triton server using gRPC only.
    """

    def __init__(self,FLAGS):
        self._meta_data = {}
        self._grpc_stub = None
        self.FLAGS = FLAGS

        self.register_grpc_channel() # register and initialise the stub
        self._fetch_model_metadata() #

    def register_grpc_channel(self):
        """
        Register the connection via the gRPC endpoint of the Triton server
        """
        grpc_channel =  grpc.insecure_channel(self.FLAGS.channel_triton ,options=[
                                   ('grpc.max_send_message_length', self.FLAGS.batch_size*17671546),
                                   ('grpc.max_receive_message_length', self.FLAGS.batch_size*17671546),
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
            name=self.FLAGS.model_name, version=self.FLAGS.model_version)
        self._meta_data["metadata_response"] = self._grpc_stub.ModelMetadata(self._meta_data["metadata_request"])

        self._meta_data["config_request"] = service_pb2.ModelConfigRequest(name=self.FLAGS.model_name,
                                                                           version=self.FLAGS.model_version)
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
        self.request.model_name = self.FLAGS.model_name
        self.request.model_version = self.FLAGS.model_version
        self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()

    def do_inference(self):
        """
        inference based on grpc_stud
        @return: inference of grpc
        """
        return self._grpc_stub.ModelInfer(self.request)