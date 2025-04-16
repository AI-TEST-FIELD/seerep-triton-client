from triton import TritonInference

image_uuids = ['1957e21a-2f4f-40f5-b4de-b0a693fd7bc0',
               'f90f8e7c-1e08-4868-ad8d-a6f690ebca94']

seerep_endpoint = "agrigaia-ur.ni.dfki:9090"
triton_endpoint = "10.249.6.30:8001"

triton_client  = TritonInference(
                                model_name='yolov5m_coco',
                                seerep_endpoint_url=seerep_endpoint,
                                triton_endpoint_url=triton_endpoint,
                                log_level='info',
                                modality='image')
triton_client.generate_annotations(sample_uuids=image_uuids)
