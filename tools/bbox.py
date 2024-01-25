import open3d as o3d
import numpy as np
from dataclasses import InitVar
from typing import List, Union, Tuple
from attr import dataclass
import math


def euler_matrix(ai, aj, ak, axes="sxyz"):
    # from tf.transformations import euler_matrix
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    # map axes strings to/from tuples of inner axis, parity, repetition, frame
    _AXES2TUPLE = {
        "sxyz": (0, 0, 0, 0),
        "sxyx": (0, 0, 1, 0),
        "sxzy": (0, 1, 0, 0),
        "sxzx": (0, 1, 1, 0),
        "syzx": (1, 0, 0, 0),
        "syzy": (1, 0, 1, 0),
        "syxz": (1, 1, 0, 0),
        "syxy": (1, 1, 1, 0),
        "szxy": (2, 0, 0, 0),
        "szxz": (2, 0, 1, 0),
        "szyx": (2, 1, 0, 0),
        "szyz": (2, 1, 1, 0),
        "rzyx": (0, 0, 0, 1),
        "rxyx": (0, 0, 1, 1),
        "ryzx": (0, 1, 0, 1),
        "rxzx": (0, 1, 1, 1),
        "rxzy": (1, 0, 0, 1),
        "ryzy": (1, 0, 1, 1),
        "rzxy": (1, 1, 0, 1),
        "ryxy": (1, 1, 1, 1),
        "ryxz": (2, 0, 0, 1),
        "rzxz": (2, 0, 1, 1),
        "rxyz": (2, 1, 0, 1),
        "rzyz": (2, 1, 1, 1),
    }

    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
    # axis sequences for Euler angles
    _NEXT_AXIS = [1, 2, 0, 1]

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


@dataclass
class BoundingBoxCentroid:
    """
    Defines a axially-oriented Bounding Box in 3D Space (Right Handed Coordinate System with pos. x-Axis pointing forward, pos. y-Asix left and pos. z-Axis upward (ROS convention)).

    x, y, z - Centroid of the Box in ROS Coordinate System
    z_rotaion - CCW rotation around the Z-Axis to point the front into the front direction of the object (other rotations are not considered)
    height - z-Axis Size
    width - y-Axis Size
    depth - x-Axis Size
    confidence - confidence of the object
    label - the detection class
    """

    x: float
    y: float
    z: float
    z_rotation: InitVar[Union[float, None]]
    height: float
    width: float
    depth: float
    confidence: float
    label: int

    def __post_init__(self, z_rotation):
        if z_rotation is None:
            self.has_rotation = False
            self.z_rotation = 0.0
        elif isinstance(z_rotation, float):
            self.has_rotation = True
            self.z_rotation = z_rotation
        else:
            breakpoint()
            raise NotImplementedError

    def convert_to_corners(self, camera_frame=False, rotation=True) -> np.ndarray:
        """

          7-----6
         /     /|
        4-----5 |
        |     | |
        | 3---|-2
        |/    |/
        0-----1

        """

        # Create a bounding box outline
        h, w, d = self.height, self.width, self.depth

        if not camera_frame:
            """
            ROS convention: x-forward, y-left, z-up
                    Z
                 X  |
                  \ |
                   \|
              Y-----o
            """

            bounding_box = np.array(
                [
                    [-d / 2, -d / 2, d / 2, d / 2, -d / 2, -d / 2, d / 2, d / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
                ]
            )
        else:
            """
            Camera Convention: x-right, y-down, z-forward
                 Z
                /
               /
              O-----X
              |
              |
              y
            """

            bounding_box = np.array(
                [
                    [-w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2],
                    [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
                    [-d / 2, -d / 2, d / 2, d / 2, -d / 2, -d / 2, d / 2, d / 2],
                ]
            )

        # homogenous transformation matrix
        if rotation:
            z_rot_sin = np.sin(self.z_rotation)
            z_rot_cos = np.cos(self.z_rotation)
            transform = np.array(
                [
                    [z_rot_cos, -z_rot_sin, 0.0, self.x],
                    [z_rot_sin, z_rot_cos, 0.0, self.y],
                    [0.0, 0.0, 1.0, self.z],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        else:
            transform = np.array(
                [
                    [1.0, 0.0, 0.0, self.x],
                    [0.0, 1.0, 0.0, self.y],
                    [0.0, 0.0, 1.0, self.z],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        # Transform the bounding box
        bounding_box = np.vstack((bounding_box, np.ones((1, bounding_box.shape[-1]))))
        corner_box = transform.dot(bounding_box).T[:, :-1]

        return corner_box

    def convert_to_2d_corners(self, camera_frame=False):
        h, w = self.height, self.width

        if not camera_frame:
            """
            ROS convention: x-forward, y-left, z-up
                    Z
                 X  |
                  \ |
                   \|
              Y-----o
            """

            bounding_box = np.array(
                [
                    [0, 0, 0, 0],
                    [w / 2, -w / 2, -w / 2, w / 2],
                    [-h / 2, -h / 2, h / 2, h / 2],
                ]
            )
        else:
            """
            Camera Convention: x-right, y-down, z-forward
                 Z
                /
               /
              O-----X
              |
              |
              y
            """
            raise NotImplementedError
            bounding_box = np.array(
                [
                    [-w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2],
                    [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2],
                    [-d / 2, -d / 2, d / 2, d / 2, -d / 2, -d / 2, d / 2, d / 2],
                ]
            )

        # homogenous transformation matrix
        z_rot_sin = np.sin(self.z_rotation)
        z_rot_cos = np.cos(self.z_rotation)
        transform = np.array(
            [
                [z_rot_cos, -z_rot_sin, 0.0, self.x],
                [z_rot_sin, z_rot_cos, 0.0, self.y],
                [0.0, 0.0, 1.0, self.z],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Transform the bounding box
        bounding_box = np.vstack((bounding_box, np.ones((1, bounding_box.shape[-1]))))
        corner_box = transform.dot(bounding_box).T[:, :-1]

        return corner_box

    def get_top_left_front_position(self):
        position = np.asarray(
            [
                self.x - (self.depth / 2),
                self.y + (self.width / 2),
                self.z + (self.height / 2),
            ]
        )

        return position

    def convert_to_line_box(self, color: Tuple[int, int, int] = (1, 0, 0)):
        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [4, 5],
            [5, 6],
            [6, 7],
            [4, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        colors = [color for _ in range(len(lines))]

        line_box = o3d.geometry.LineSet()
        line_box.points = o3d.utility.Vector3dVector(self.convert_to_corners())
        line_box.lines = o3d.utility.Vector2iVector(lines)
        line_box.colors = o3d.utility.Vector3dVector(colors)

        return line_box

    def convert_to_obb(self, device: o3d.core.Device):
        rotation = euler_matrix(0, 0, self.z_rotation)[:3, :3].astype(np.float64)

        try:
            oriented_bbox = o3d.t.geometry.OrientedBoundingBox(
                center=[self.x, self.y, self.z],
                rotation=rotation,  # cw-negative, ccw-positive
                extent=[self.width, self.depth, self.height],  # x, y, z
            ).to(device)

        except BaseException as e:
            breakpoint()
            print(e)

        return oriented_bbox
    
    @classmethod
    def from_triton_bbox(cls, bbox):
        return cls(
            x=-bbox[2],
            y=-bbox[0],
            z=bbox[1],
            height=bbox[5],
            width=bbox[4],
            depth=bbox[3],
            z_rotation=-bbox[6],
            confidence=-1,
            label=-1
        )


def janosch_bbox_magic(orig_bboxes):
    lineset = []
    ros_bbox = []

    for orig_bbox in orig_bboxes:
        tmp_bbox = BoundingBoxCentroid.from_triton_bbox(bbox=orig_bbox)
        print(f'BBox Stats: x-{tmp_bbox.x}, y-{tmp_bbox.y}, z-{tmp_bbox.z}, height-{tmp_bbox.height}, width-{tmp_bbox.width}, depth-{tmp_bbox.depth}, z-rotation-{tmp_bbox.z_rotation}')
        lineset.append(tmp_bbox.convert_to_line_box())
        ros_bbox.append(tmp_bbox)

    return lineset, ros_bbox