from abc import ABC, abstractmethod
import tritonclient.grpc.model_config_pb2 as mc

class Model(ABC):
    """
        Declares the functionality for several model types for triton server
    """

    def __init__(self, model_name='None'):
        self._clients = {}
        self.model_name = model_name

    @abstractmethod
    def register_client(self,clienttype,client):
        """
        Implement the method to register the client for
        """
        
    @abstractmethod
    def get_postprocess(self):
        """
        """
    @abstractmethod
    def get_preprocess(self):
        """"""
    
    def parse_model(self, model_metadata, model_config):
        """
            Check the configuration of a model to make sure it meets the
            requirements for an image yolov4 network (as expected by
            this client)
        """
        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = [output for output in model_metadata.outputs]
        input_batch_dim = (model_config.max_batch_size > 0)
        input_batch_dim = False
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                    format(expected_input_dims, model_metadata.name,
                           len(input_metadata.shape)))

        if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
                (input_config.format != mc.ModelInput.FORMAT_NHWC)):
            raise Exception("unexpected input format " +
                            mc.ModelInput.Format.Name(input_config.format) +
                            ", expecting " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                            " or " +
                            mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

        if input_config.format == mc.ModelInput.FORMAT_NHWC:
            h = input_metadata.shape[1 if input_batch_dim else 0]
            w = input_metadata.shape[2 if input_batch_dim else 1]
            c = input_metadata.shape[3 if input_batch_dim else 2]
        else:
            c = input_metadata.shape[1 if input_batch_dim else 0]
            h = input_metadata.shape[2 if input_batch_dim else 1]
            w = input_metadata.shape[3 if input_batch_dim else 2]

        input_metadata = [{'name': input.name, 
                           'shape': input.shape,
                           'dtype': input.datatype} for input in model_metadata.inputs]
        output_metadata = [{'name': output.name, 
                           'shape': output.shape,
                           'dtype': output.datatype} for output in model_metadata.outputs]
        return (input_metadata, output_metadata)

