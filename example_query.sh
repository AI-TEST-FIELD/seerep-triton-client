#! /bin/bash
export PROJECT_ROOT=${PWD}
python3 main3d.py \
-cT 10.249.6.30:8001 \
-cS 10.249.3.19:9090 \
--seerep-project map \
--model-name pointpillar_kitti \
--log-level info \
--semantics Source_Kitti