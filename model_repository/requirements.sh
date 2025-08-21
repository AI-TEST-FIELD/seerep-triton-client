#! /bin/bash
python3 -m pip install -U openmim
python3 -m mim install mmengine
python3 -m mim install "mmcv==2.2.0"
python3 -m mim install mmdet
python3 -m pip install fairscale transformers