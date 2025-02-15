import sys
import time
import rospy
from copy import copy
from pyquaternion import Quaternion
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PoseWithCovariance
import numpy as np
from visual_utils import open3d_vis_utils as visualizer
# from utils import image_util


class RosInference3D():

    """
    A RosInference to support ROS input and provide input to channel for inference.
    """

    def __init__(self, topic=None):
        '''
            channel: channel of type communicator.channel
            client: client of type clients

        '''
        self.image = None
        self.topic = topic


    def start_inference(self):
        rospy.init_node('pc_listener')
        rospy.Subscriber(self.topic, PointCloud2, self._pc_callback, queue_size=50)
        # if self.jsk:
        #     self.publisher = rospy.Publisher(self.channel.params['pub_topic'], BoundingBoxArray, queue_size=1)
        # else:
        #     self.publisher = rospy.Publisher(self.channel.params['pub_topic'], Detection3DArray, queue_size=1)
        rospy.spin()

    def yaw2quaternion(self, yaw: float) -> Quaternion:
        return Quaternion(axis=[0,0,1], radians=yaw)

    def _pc_callback(self, msg):
        # TODO what is the 4th attribute of the point clouds from KITTI and what is their data range
        self.x = np.array(list(point_cloud2.read_points(msg, field_names = ("x"), skip_nans=True)))
        self.y = np.array(list(point_cloud2.read_points(msg, field_names = ("y"), skip_nans=True)))
        self.z = np.array(list(point_cloud2.read_points(msg, field_names = ("z"), skip_nans=True)))
        self.pc = np.zeros((self.x.shape[0], 3), dtype=np.float64)
        self.pc[:, 0] = self.x[:, 0]
        self.pc[:, 1] = self.y[:, 0]
        self.pc[:, 2] = self.z[:, 0]
        visualizer.draw_scenes(self.pc)
        self.i = np.array(list(point_cloud2.read_points(msg, field_names = ("intensity"), skip_nans=True)))
        self.t = np.array(list(point_cloud2.read_points(msg, field_names = ("t"), skip_nans=True)))
        self.reflec = np.array(list(point_cloud2.read_points(msg, field_names = ("reflectivity"), skip_nans=True)))
        self.ring = np.array(list(point_cloud2.read_points(msg, field_names = ("ring"), skip_nans=True)))
        self.ambient = np.array(list(point_cloud2.read_points(msg, field_names = ("ambient"), skip_nans=True)))
        self.range = np.array(list(point_cloud2.read_points(msg, field_names = ("range"), skip_nans=True)))
        # self.pc = np.array(list(point_cloud2.read_points(msg)))
        self.pc = self.client_preprocess.filter_pc(self.pc)
        # the number of voxels changes every sample



if __name__ =="__main__":
    RosInfer = RosInference3D(topic='/ai_test_field/edge/hsos/sensors/ouster/points')
    RosInfer.start_inference()