#!/usr/bin/env python

import argparse
import yaml
import os

from communicator import EvaluateInference
from communicator.channel import grpc_channel, seerep_channel
from clients import Yolov5client, FCOS_client, Detrex_client

clients = {
    'YOLOv5nCROP': Yolov5client,
    'YOLOv5nCOCO': Yolov5client,
    'FCOS_detectron':FCOS_client,
    'dino_coco_600_squared':Detrex_client,
    # 'second_iou':Pointpillars_client,
    # more clients can be added
}


FLAGS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-P',
                        '--file-path',
                        required=True,
                        type=str,
                        help='Path of config files directory')
    args=parser.parse_args()
    
    file_list=[]

    for files in os.listdir(args.file_path):
        if files.endswith('.yaml'):
            file_path=os.path.join(args.file_path,files)
            file_list.append(file_path)
    
    return file_list


if __name__ == '__main__':
    FLAGS = parse_args()

    for file in FLAGS:
        with open(file) as f:
            data=yaml.load(f,Loader=yaml.loader.SafeLoader)
            print('Starting with config file {}'.format(os.path.basename(file)))
            # select client operations based on the model
            client = clients[data['model_name']](model_name=data['model_name'])

            #define channel
            channel = grpc_channel.GRPCChannel(data)

            #define inference
            evaluation = EvaluateInference(data, channel, client)
            evaluation.start_inference(data['model_name'])

            print('Config file {} ended'.format(os.path.basename(file)))
            print('###################################################################################################################')
