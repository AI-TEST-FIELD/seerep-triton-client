#!/usr/bin/env python

import argparse
import yaml

from communicator import EvaluateInference
from communicator.channel import grpc_channel, seerep_channel
from clients import Yolov5client, FCOS_client, Detrex_client

clients = {
    'YOLOv5nCROP': Yolov5client,
    'YOLOv5nCOCO': Yolov5client,
    'yolov5m_coco': Yolov5client,
    'FCOS_detectron':FCOS_client,
    'frcnn_800':FCOS_client,
    'dino_coco_600_squared':Detrex_client,
    'dino_coco_800':Detrex_client,
    'retinanet_coco':FCOS_client,
    'retina_big':FCOS_client,
    # 'second_iou':Pointpillars_client,
    # more clients can be added
}


FLAGS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cT',
                        '--channel-triton',
                        type=str,
                        required=True,
                        default="localhost:8001",
                        help='gRPC endpoint of the Triton inference server')
    parser.add_argument('-cS',
                        '--channel-seerep',
                        type=str,
                        required=True,
                        default="agrigaia-ur.ni.dfki:9090",
                        help='gRPC endpoint of the SEEREP server')
    parser.add_argument('-p',
                        '--seerep-project',
                        type=str,
                        required=True,
                        default="agrigaia-ur.ni.dfki:9090",
                        help='Name of the SEEREP Project where the data is store e.g. "aitf-triton-data"')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=False,
                        default="aitf-triton-data",
                        help='Name of the model. This has to match exactly (also case sensitive) with name string on Triton server')
    parser.add_argument('-x',
                        '--model-version',
                        type=str,
                        required=False,
                        default="",
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('-l',
                        '--log-level',
                        type=str,
                        required=False,
                        default='error',
                        choices=['info', 'warning', 'debug', "critical", "error"],
                        help='Set logging level')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-v',
                        '--visualize',
                        action='store_true',
                        required=False,
                        help='Visualize images'),
    parser.add_argument('-d',
                        '--mode',
                        type=str,
                        required=False,
                        default='images',
                        choices=['images', 'pointclouds'],
                        help='Data modality on which we want to perform inference. Default is images.')
    parser.add_argument('-s',
                        '--semantics',
                        nargs='+',
                        required=False,
                        help='Provide SEEREP semantics as a list e.g. "-s person weather_general_cloudy "')
    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = parse_args()
    # select client operations based on the model
    queries = [query.lower() for query in FLAGS.semantics]
    if len([v for v in queries if 'kitti' in v]) != 0:
        format='kitti'
    else:
        format='coco'
    format='aitf'
    client = clients[FLAGS.model_name](model_name=FLAGS.model_name)

    #define channel
    channel = grpc_channel.GRPCChannel(FLAGS)

    #define inference
    evaluation = EvaluateInference(args=FLAGS, channel=channel, client=client, format=format)
    evaluation.start_inference(model_name=FLAGS.model_name, modality=FLAGS.mode)
