from communicator.channel.base_channel import BaseChannel

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc


class GRPCChannel(BaseChannel):
    """
    A GRPCChannel is responsible for establishing connection between client and server using gRPC.
    """

    def __init__(self,FLAGS):
        super().__init__(FLAGS)
        self._meta_data = {}
        self._grpc_stub = None
        self.args = FLAGS

        self.register_channel() # register and initialise the stub
        self._grpc_metadata() #


    def register_channel(self):
        """
         register grpc triton channel
        """
        grpc_channel =  grpc.insecure_channel(self.args['channel_triton'] ,options=[
                                   ('grpc.max_send_message_length', self.FLAGS['batch_size']*8568044),
                                   ('grpc.max_receive_message_length', self.FLAGS['batch_size']*8568044),
                                    ])
        self._grpc_stub  = service_pb2_grpc.GRPCInferenceServiceStub(grpc_channel)

    def fetch_channel(self):
        """
        return grpc stub
        """
        return self._grpc_stub

    def _grpc_metadata(self):
        """
        Initiate all meta data required for models
        """
        # Make sure the model matches our requirements, and get some
        # properties of the model that we need for preprocessing
        self._meta_data["metadata_request"] = service_pb2.ModelMetadataRequest(
            name=self.FLAGS['model_name'], version=self.FLAGS['model_version'])
        self._meta_data["metadata_response"] = self._grpc_stub.ModelMetadata(self._meta_data["metadata_request"])

        self._meta_data["config_request"] = service_pb2.ModelConfigRequest(name=self.FLAGS['model_name'],
                                                                           version=self.FLAGS['model_version'])
        self._meta_data["config_response"] = self._grpc_stub.ModelConfig(self._meta_data["config_request"])

        # set
        self._set_grpc_members()

    def get_metadata(self):
        """
        return meta_data dictionary form
        @rtype: dictionary
        """
        return self._meta_data

    def _set_grpc_members(self):
        """
        set essential grpc members
        """
        self.input = service_pb2.ModelInferRequest().InferInputTensor()
        self.request = service_pb2.ModelInferRequest()
        self.request.model_name = self.FLAGS['model_name']
        self.request.model_version = self.FLAGS['model_version']
        self.output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()

    def do_inference(self):
        """
        inference based on grpc_stud
        @return: inference of grpc
        """
        return self._grpc_stub.ModelInfer(self.request)






