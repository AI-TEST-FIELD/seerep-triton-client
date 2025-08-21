import torch
import json
import numpy as np
from pathlib import Path
import warnings
import ssl
# MMDetection imports
from mmdet.apis import DetInferencer
import triton_python_backend_utils as pb_utils
warnings.filterwarnings("ignore", category=FutureWarning, module="mmengine")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mmengine")
warnings.filterwarnings("ignore", category=UserWarning, module="mmengine")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
'''
    RTM DeTR model via MMDet stack
    RTMDet: An Empirical Study of Designing Real-Time Object Detectors
    https://arxiv.org/abs/2212.07784
'''
ssl._create_default_https_context = ssl._create_unverified_context
class TritonPythonModel:
    """
    Triton Python model for MMDetection inference.
    Based on the minimal_inference.py implementation.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        """Initialize the detection model."""
        # Load config
        self.folder_path = Path(__file__).parent
        self.mmseg_config = 'rtmdet_l_swin_b_p6_4xb16-100e_coco'
        # self.model_path = osp.join(self.folder_path, 'rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth')
        self.model_config = model_config = json.loads(args['model_config'])

        # Get output configurations
        bboxes_config = pb_utils.get_output_config_by_name(model_config, "bboxes")
        scores_config = pb_utils.get_output_config_by_name(model_config, "scores")
        labels_config = pb_utils.get_output_config_by_name(model_config, "labels")
        # num_detections_config = pb_utils.get_output_config_by_name(model_config, "num_detections")

        # Convert Triton types to numpy types
        self.bboxes_dtype = pb_utils.triton_string_to_numpy(bboxes_config['data_type'])
        self.scores_dtype = pb_utils.triton_string_to_numpy(scores_config['data_type'])
        self.labels_dtype = pb_utils.triton_string_to_numpy(labels_config['data_type'])
        # self.num_detections_dtype = pb_utils.triton_string_to_numpy(num_detections_config['data_type'])

        # Get model repository path and instance device ID
        self.model_repository = args['model_repository']
        self.model_version = args['model_version']
        device_id = args.get('model_instance_device_id', '0')
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

        # Initialize the MMDetection model
        self.init_mmdet_model()

    def init_mmdet_model(self):
        """Initialize the MMDetection model similar to MinimalDetector."""
        # Model paths - adjust these according to your setup
        self.score_threshold = 0.5  # Default threshold, can be made configurable
        self.model = DetInferencer(self.mmseg_config, device=self.device, show_progress=False)

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model.
        
        This function receives preprocessed tensors from the client and
        returns post-processed detection results as tensors.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Process each request
        for request in requests:
            # TODO : Assume batch size 1, make it dynamic with >1 batch size
            inputs = pb_utils.get_input_tensor_by_name(request, "inputs").as_numpy()[0]
            inputs = inputs.transpose(1, 2, 0)  # Convert from CHW to HWC
            # Forward pass

            results = self.model(inputs)

            # Batch processing - combine results
            bboxes = []
            scores = []
            labels = []

            result = results['predictions'][0]
            indices = np.where(np.array(result['scores']) > self.score_threshold)[0]
            bboxes = np.array(result['bboxes'])[indices]
            scores = np.array(result['scores'])[indices]
            labels = np.array(result['labels'])[indices]
            num_detections = len(scores)
            
            # Create output tensors
            bboxes_tensor = pb_utils.Tensor(
                "bboxes", 
                bboxes.astype(self.bboxes_dtype)
            )
            scores_tensor = pb_utils.Tensor(
                "scores", 
                scores.astype(self.scores_dtype)
            )
            labels_tensor = pb_utils.Tensor(
                "labels", 
                labels.astype(self.labels_dtype)
            )
        
            # Create response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    bboxes_tensor, 
                    scores_tensor, 
                    labels_tensor, 
                    # num_detections_tensor
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up MMDetection model...')
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None