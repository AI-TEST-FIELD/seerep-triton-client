#! /bin/bash
python3 main3d.py \
-cT 10.249.6.30:8001 \
-cS 10.249.6.30:9090 \
--seerep-project aitf_image_test \
--model-name second_iou \
--log-level info \
--semantics Source_Kitti