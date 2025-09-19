"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
# import torch
import open3d
import matplotlib
import numpy as np
import time

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]
class Visualizer:
    def __init__(self, 
         origin=True,
         origin_size: float = 3.0,
         point_colors=None,
         stream=True) -> None:
        self.point_colors = point_colors
        self.stream = stream
        self._origin_added = origin
        self.origin_size = float(origin_size)
        
        open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)
        if self.stream==False:
            self.vis = open3d.visualization.Visualizer()
        else:
            self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        # key-driven stepping
        self._advance = False
        self.vis.register_key_callback(256, self._on_next)   # ESC
        self.vis.register_key_callback(ord(' '), self._on_next)  # Space (optional)

        # Draw origin
        if origin==True:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=self.origin_size, origin=[0, 0, 0])
            self.vis.add_geometry(axis_pcd)
        
        self.vis.get_render_option().point_size = 1.5
        self.vis.get_render_option().background_color = np.zeros(3)
        
        # Set initial camera once
        self._set_ego_camera_view()

    def _on_next(self, vis):
        # Callback to continue to next sample
        self._advance = True
        return False

    def _wait_for_key(self):
        # Block until user presses ESC/Space; allow camera interaction
        self._advance = False
        while not self._advance:
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

    def _set_ego_camera_view(self, eye=None, lookat=None, up=None):
        ctr = self.vis.get_view_control()
        eye = np.array([-5.0, 0.0, 5.0], dtype=np.float64) if eye is None else np.asarray(eye, dtype=np.float64)
        lookat = np.array([0.0, 0.0, 0.0], dtype=np.float64) if lookat is None else np.asarray(lookat, dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64)

        forward = lookat - eye
        forward /= (np.linalg.norm(forward) + 1e-12)
        right = np.cross(forward, up)
        right /= (np.linalg.norm(right) + 1e-12)
        true_up = np.cross(right, forward)

        R = np.stack([right, true_up, -forward], axis=0)
        t = -R @ eye

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    def get_coor_colors(self, obj_labels):
        """
        Args:
            obj_labels: 1 is ground, labels > 1 indicates different instance cluster

        Returns:
            rgb: [N, 3]. color for each point.
        """
        colors = matplotlib.colors.XKCD_COLORS.values()
        max_color_num = obj_labels.max()

        color_list = list(colors)[:max_color_num+1]
        colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
        label_rgba = np.array(colors_rgba)[obj_labels]
        label_rgba = label_rgba.squeeze()[:, :3]

        return label_rgba
    
    def draw_box(self, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
        for i in range(gt_boxes.shape[0]):
            line_set, box3d = self.translate_boxes_to_open3d_instance(gt_boxes[i])
            if ref_labels is None:
                line_set.paint_uniform_color(color)
            else:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]])

            self.vis.add_geometry(line_set)
    
    def translate_boxes_to_open3d_instance(self, gt_boxes):
        """
                4-------- 6
            /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1
            |/         |/
            2 -------- 0
        """
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = open3d.utility.Vector2iVector(lines)

        return line_set, box3d
    
    def draw_scenes(self,
                    points: np.array, 
                    gt_boxes=None, 
                    ref_boxes=None, 
                    ref_labels=None, 
                    ref_scores=None, 
                    point_colors=None):
        # Clear previous geometries
        self.vis.clear_geometries()
        if hasattr(self, '_origin_added') and self._origin_added:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=self.origin_size, origin=[0, 0, 0])
            self.vis.add_geometry(axis_pcd)
        
        # Points
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        self.vis.add_geometry(pts)
        if point_colors is None:
            pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        else:
            pts.colors = open3d.utility.Vector3dVector(point_colors)

        # Boxes
        if gt_boxes is not None:
            self.draw_box(gt_boxes, (0, 0, 1))
        if ref_boxes is not None:
            self.draw_box(ref_boxes, (0, 1, 0), ref_labels, ref_scores)

        # Do NOT reset camera here; let the user control it
        # Block until ESC/Space pressed
        self._wait_for_key()

    def Initialize_scene(self,
                points: np.array, 
                gt_boxes=None, 
                ref_boxes=None, 
                ref_labels=None, 
                ref_scores=None, 
                point_colors=None):

        # Draw points
        self.pts = open3d.geometry.PointCloud()
        self.pts.points = open3d.utility.Vector3dVector(points[:, :3])
        self.vis.add_geometry(self.pts, reset_bounding_box=False)

        # Colorize points
        if point_colors is None:
            self.pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        else:
            self.pts.colors = open3d.utility.Vector3dVector(point_colors)

        # Draw ground truth boxes
        if gt_boxes is not None:
            self.gt_boxes = gt_boxes
            self.draw_box(gt_boxes, (0, 0, 1))

        # Draw reference boxes
        if ref_boxes is not None:
            self.ref_boxes = ref_boxes
            self.draw_box(ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    
        # Set camera view to ego perspective
        self._set_ego_camera_view()
        
        # Display window
        self.vis.run()

    def destroy(self):
        self.vis.destroy_window()