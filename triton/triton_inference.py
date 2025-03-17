from communicator.endpoint import SeerepEndpoint, TritonEndpoint
from communicator.base_inference import BaseInference

class TritonInference(BaseInference):
    def __init__(self, 
                 model_name='yolov5m_coco', 
                 seerep_endpoint_url='agrigaia-ur.ni.dfki:9090', 
                 triton_endpoint_url='10.249.6.23:8001', 
                 modality='image'):
        self.model_name = model_name
        self.seerep_endpoint = SeerepEndpoint(seerep_endpoint_url,
                                              modality=modality,
                                              visualize=False)
        self.triton_endpoint = TritonEndpoint(triton_endpoint_url,
                                              modality=modality,
                                              visualize=False)
        self.modality = modality

    def generate_annotations(self, sample_uuids: list, type: str):
        pass

    def get_image(self, sample_uuids):
        pass
    
    