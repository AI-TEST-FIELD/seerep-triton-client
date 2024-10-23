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
    'yolov5m_iso_trt': Yolov5client,
    'yolov5m_iso_onnx': Yolov5client,
    'FCOS_detectron':FCOS_client,
    'frcnn_800':FCOS_client,
    'dino_coco_600_squared':Detrex_client,
    'dino_coco_800':Detrex_client,
    'dino_coco_600':Detrex_client,
    'retinanet_coco':FCOS_client,
    'retina_big':FCOS_client,
    # 'second_iou':Pointpillars_client,
    # more clients can be added
}

project_list = {
    # 'EV35_VTE_Setup_4m_180Deg__2024-10-11-23-27-42_0',
    # 'EV35_VTE_Setup_8m_0Deg__2024-10-11-23-42-38_0',
    # 'EV41_Kleidung_blank_225Deg_2024-10-11-22-16-05_0',
    # 'EV41_Kleidung_GrueneJacke_225Deg_2024-10-10-18-27-56_0',
    # 'EV41_Kleidung_Jagdkleidung_225Deg_2024-10-10-18-42-54_0',
    # 'EV41_Kleidung_SchwarzeBaumwolljacke_225Deg_2024-10-10-18-12-30_0',
    # 'EV41_Kleidung_SchwarzeHose_225Deg_2024-10-11-16-19-23_0',
    # 'EV41_Kleidung_SchwarzeJacke_225Deg_2024-10-10-17-58-23_0',
    # 'EV41_Kleidung_Sonnenbrille_BlondeHaare_225Deg_2024-10-11-16-47-57_0',
    # 'EV41_Kleidung_Sonnenbrille_BrauneHaare_225Deg_2024-10-11-16-33-47_0',
    # 'EV41_Kleidung_Sonnenbrille_DunkleCap_225Deg_2024-10-10-18-58-13_0',
    # 'EV41_Kleidung_Sonnenbrille_LangeHaare_225Deg_2024-10-10-19-13-33_0',
    # 'Maize_2023_06_01_2024-07-12-12-57-48_0',
    # 'Maize_2023_06_02_2024-07-12-13-08-25_0',
    # 'Maize_2023_06_06_2024-08-26-13-55-11_0',
    # 'Maize_2023_06_16_2024-07-12-13-33-33_0',
    # 'Maize_2023_06_22_2024-08-26-13-58-04_0',
    'Rye_20240318_2024-10-09-11-36-12_0',
    # 'Rye_20240327_2024-10-09-11-40-23_0',
    # 'Rye_20240411_2024-10-09-11-44-22_0',
    # 'Rye_20240424_2024-10-09-11-48-47_0',
    'Rye_20240508_2024-10-09-11-54-00_0',
    # 'Test_fast_2024-10-09-15-28-54_0',
    }

model_list = [
    "frcnn_800",
    "retina_big",
    "yolov5m_iso_onnx",
    # "yolov5m_coco",
    ]

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

    for model in model_list:
        FLAGS.model_name = model
        client = clients[FLAGS.model_name](model_name=FLAGS.model_name)
        for project in project_list:
            FLAGS.seerep_project = project
            #define channel
            channel = grpc_channel.GRPCChannel(FLAGS)

            #define inference
            evaluation = EvaluateInference(args=FLAGS, channel=channel, client=client, format=format)
            evaluation.start_inference(model_name=FLAGS.model_name, modality=FLAGS.mode)
