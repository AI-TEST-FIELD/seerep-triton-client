from .base_client import Client
from .preprocess import PCDetPreprocess
from .postprocess import PCDetPostprocess

class PCDet_client(Client):
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
        return PCDetPreprocess()

    def get_postprocess(self):
        return PCDetPostprocess()

    # Override function from the base class.
    def parse_model(self, model_metadata, model_config):
        if len(model_metadata.inputs) != 3:     # voxels, coords, numpoints     
            raise Exception("expecting 3 input, got {}".format(
                len(model_metadata.inputs)))
        if len(model_metadata.outputs) != 3:    # bbox_preds, dir_scores, scores
            raise Exception("expecting 3 output, got {}".format(
                len(model_metadata.outputs)))

        input_metadata = [{'name': input.name, 
                           'shape': input.shape,
                           'dtype': input.datatype} for input in model_metadata.inputs]
        output_metadata = [{'name': output.name, 
                           'shape': output.shape,
                           'dtype': output.datatype} for output in model_metadata.outputs]
        input_batch_dim = (model_config.max_batch_size > 0)
        input_batch_dim = False
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        
        return (input_metadata, output_metadata)    
