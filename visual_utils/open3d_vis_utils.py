"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
# import torch
import open3d
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def draw_scenes(points: np.array, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    # if isinstance(points, torch.Tensor):
    #     points = points.cpu().numpy()
    # if isinstance(gt_boxes, torch.Tensor):
    #     gt_boxes = gt_boxes.cpu().numpy()
    # if isinstance(ref_boxes, torch.Tensor):
    #     ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)
    vis.get_render_option().point_size = 1.5
    vis.get_render_option().background_color = np.zeros(3)
    
    #This line will obtain the default camera parameters .
    camera_params = ctr.convert_to_pinhole_camera_parameters() 
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
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


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
    
class Visualizer():
    def __init__(self, 
                 origin=True,
                 point_colors=None,
                 stream=True) -> None:
        self.point_colors = point_colors
        self.stream = stream
        open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Debug)
        if self.stream==False:
            self.vis = open3d.visualization.Visualizer()
        else:
            self.vis = open3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        ctr = self.vis.get_view_control()
        # #This line will obtain the default camera parameters .
        camera_params = ctr.convert_to_pinhole_camera_parameters() 
        # Define the desired camera extrinsic parameters.
        # TODO this does not do anything. WHY?
        camera_params.extrinsic = np.array([[ 0.033585759937874757, -0.13889353862189011, 0.98973763273833582, 0.0],
                                            [-0.99703926833339696, 0.063882646483045008, 0.042798421460690204, 0.0],
                                            [-0.069171483507295253, -0.98824470269635745, -0.13633676489483001, 0.0],
                                            [1.2936070652362615, 7.8928817882678945, 21.5193268164304, 1.0]])
            # [
            #     0.033585759937874757,
            #     -0.13889353862189011,
            #     0.98973763273833582,
            #     0.0,
            #     -0.99703926833339696,
            #     0.063882646483045008,
            #     0.042798421460690204,
            #     0.0,
            #     -0.069171483507295253,
            #     -0.98824470269635745,
            #     -0.13633676489483001,
            #     0.0,
            #     1.2936070652362615,
            #     7.8928817882678945,
            #     21.5193268164304,
            #     1.0
            # ],
        camera_params.extrinsic = np.eye(4)
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        # draw origin
        if origin==True:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            self.vis.add_geometry(axis_pcd)
        self.vis.get_render_option().point_size = 1.5
        self.vis.get_render_option().background_color = np.zeros(3)

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
            # if score is not None:
            #     corners = box3d.get_box_points()
            #     self.vis.add_3d_label(corners[5], '%.2f' % score[i])
        return  self.vis
    
    def update_box(self, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
        for i in range(gt_boxes.shape[0]):
            line_set, box3d = self.translate_boxes_to_open3d_instance(gt_boxes[i])
            if ref_labels is None:
                line_set.paint_uniform_color(color)
            else:
                line_set.paint_uniform_color(box_colormap[ref_labels[i]])

            self.vis.update_geometry(line_set)
            # if score is not None:
            #     corners = box3d.get_box_points()
            #     self.vis.add_3d_label(corners[5], '%.2f' % score[i])
        return  self.vis
    
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

        self.vis.create_window()
        # Draw points
        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        self.vis.add_geometry(pts)

        # Colorize points
        if point_colors is None:
            pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        else:
            pts.colors = open3d.utility.Vector3dVector(point_colors)

        # Draw ground truth boxes
        if gt_boxes is not None:
            self.vis = self.draw_box(gt_boxes, (0, 0, 1))

        # Draw reference boxes
        if ref_boxes is not None:
            self.vis = self.draw_box(ref_boxes, (0, 1, 0), ref_labels, ref_scores)

        # Display window and then destroy it. 
        self.vis.run()
        # self.vis.destroy_window()

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
        self.vis.add_geometry(self.pts)

        # Colorize points
        if point_colors is None:
            self.pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        else:
            self.pts.colors = open3d.utility.Vector3dVector(point_colors)

        # Draw ground truth boxes
        if gt_boxes is not None:
            self.gt_boxes = gt_boxes
            self.vis = self.draw_box(gt_boxes, (0, 0, 1))

        # Draw reference boxes
        if ref_boxes is not None:
            self.ref_boxes = ref_boxes
            self.vis = self.draw_box(ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        # self.vis.register_key_callback(ord("P"), self.visualizer_callback(self))
        # Display window and then destroy it. 
        self.vis.run()
        # self.vis.destroy_window()
    
    def destroy(self):
        self.vis.destroy_window()

    @staticmethod
    def visualizer_callback(self):
        self.vis.update_geometry(self.pts)
        for i in range(self.ref_boxes.shape[0]):
            center = self.ref_boxes[i, 0:3]
            lwh = self.ref_boxes[i, 3:6]
            axis_angles = np.array([0, 0, self.ref_boxes[i, 6] + 1e-10])
            rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
            box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

            line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
            lines = np.asarray(line_set.lines)
            lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

            line_set.lines = open3d.utility.Vector2iVector(lines)
            # if self.ref_labels is None:
            #     line_set.paint_uniform_color(color)
            # else:
            line_set.paint_uniform_color(box_colormap[self.ref_labels[i]])
            self.vis.update_geometry(line_set)
        # self.key_to_callback = {}
        # self.key_to_callback[ord("K")] = self.visualizer_callback(self)
        # self.key_to_callback[ord("R")] = self.visualizer_callback(self)
        # self.key_to_callback[ord(",")] = self.visualizer_callback(self)
        # self.key_to_callback[ord(".")] = self.visualizer_callback(self)
        self.vis.poll_events()
        self.vis.update_renderer()
        # self.vis.run()

    def update_scene(self, 
                points: np.array,
                gt_boxes=None, 
                ref_boxes=None, 
                ref_labels=None, 
                ref_scores=None, 
                point_colors=None):
        self.pts.points = open3d.utility.Vector3dVector(points[:, :3])
        # Colorize pointsp
        if point_colors is None:
            self.pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        else:
            self.pts.colors = open3d.utility.Vector3dVector(point_colors)

        self.ref_boxes = ref_boxes
        self.gt_boxes = gt_boxes
        self.ref_labels = ref_labels
        self.ref_scores = ref_scores
        update_geo = self.vis.register_key_action_callback(32, self.visualizer_callback(self))  #space
        # self.vis.run()
        

        


        
