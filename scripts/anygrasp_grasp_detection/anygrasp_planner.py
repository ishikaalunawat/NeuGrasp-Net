import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
import time

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import vgn
from vgn.utils.transform import Rotation, Transform

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
# parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
# parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
# parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
# parser.add_argument('--debug', action='store_true', help='Enable visualization')
# cfgs = parser.parse_args()
# cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# For use with GIGA setup:
class AnyGraspPlanner(object):
    def __init__(self, checkpoint_path, max_gripper_width=0.08, gripper_height=0.045, 
                    force_detection=False, qual_th=0.3, debug=False, **kwargs):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Only works with cuda for now
        # self.device = "cpu"
        self.checkpoint_path = checkpoint_path
        self.qual_th = qual_th
        self.force_detection = force_detection
        self.debug = debug # visualization
    
        self.max_gripper_width = max_gripper_width # Franka is 0.08
        self.gripper_height = gripper_height # TODO: Check this value

        # set workspace
        # xmin, xmax = -0.19, 0.12
        xmin, xmax = -0.45, 0.45
        # ymin, ymax = 0.02, 0.15
        ymin, ymax = -0.45, 0.45
        zmin, zmax = 0.0, 1.0
        self.lims = [xmin, xmax, ymin, ymax, zmin, zmax]
        top_down_grasp = False # TODO: Make it configurable

        # add all to cfgs namespace
        self.cfgs = argparse.Namespace(checkpoint_path=checkpoint_path, max_gripper_width=max_gripper_width, 
                                        gripper_height=gripper_height, force_detection=force_detection, 
                                        qual_th=qual_th, debug=debug, top_down_grasp=top_down_grasp)
        
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()

    def __call__(self, state, scene_mesh=None, sim=None, o3d_vis=None, first_call=False, **kwargs):
        """ assumes pcl in camera frame (X right, Y down, Z forward) """
        tic = time.time()
        # points = state.pcl
        ## Make points yourself using depth image and camera intrinsic
        depths = state.depths[0] # assuming single camera view
        # color/RGB is optional
        # colors = state.colors
        colors = np.zeros((depths.shape[0], depths.shape[1], 3), dtype=np.float32)
        intrinsic = state.intrinsic.K

        points, colors = self.depth_to_pcl(depths, intrinsic, colors)

        # get prediction
        # relax all lims by 0.2m
        grasp_result = self.anygrasp.get_grasp(points, colors, self.lims)
        if len(grasp_result) > 2:
            # no grasps found
            gg = []
        else:
            gg, cloud = grasp_result
        # output is grasps in camera frame

        if len(gg) == 0:
            print('Warning, no grasps detected!')
        else:
            gg = gg.nms().sort_by_score()
            if len(gg) > 20:
                gg = gg[0:20]
            # print(gg.scores)
            # print('best grasp score:', gg[0].score)


        # TODO: Convert grasps to GIGA-style grasps
        extrinsic = state.extrinsics[0] # assuming single camera view
        grasps = []
        scores = []
        for grasp in gg:
            # Add rotation by Y axis 90 degrees to convert to GIGA-style grasps
            grasp_R = Rotation.from_matrix(grasp.rotation_matrix) * Rotation.from_euler('Y', np.pi/2)
            grasp_tf = Transform(grasp_R, grasp.translation)
            # transform to world frame
            grasp_tf_world = np.linalg.inv(extrinsic.as_matrix()) @ grasp_tf.as_matrix()
            # filter grasps outside workspace (sim.size)
            if grasp_tf_world[0,3] > sim.size or grasp_tf_world[0,3] < 0 or \
                    grasp_tf_world[1,3] > sim.size or grasp_tf_world[1,3] < 0:
                # gg.remove(grasp)
                continue
            # DEBUG: add an in-frame translation offset to the grasp since we want to move closer (Franka gripper and not Robotiq)
            offset = np.array([0,0,-0.02])
            offset_tf = Transform(Rotation.from_matrix(np.eye(3)), offset)
            grasp_tf_world = grasp_tf_world @ offset_tf.as_matrix()

            curr_grasp = vgn.grasp.Grasp(Transform.from_matrix(grasp_tf_world), grasp.width)#self.cfg.max_gripper_width)
            grasps.append(curr_grasp)
            scores.append(grasp.score)

        if len(grasps) > 0:
            # debug visualization
            if self.cfgs.debug:
                grippers = gg.to_open3d_geometry_list()
                viewer = o3d.visualization.Visualizer()
                viewer.create_window()
                # create open3d geom from variable 'points'
                pcl = o3d.geometry.PointCloud()
                pcl.points = o3d.utility.Vector3dVector(points)
                viewer.add_geometry(pcl)
                # viewer.add_geometry(grippers[0])
                for gripper in grippers:
                    viewer.add_geometry(gripper)
                opt = viewer.get_render_option()
                opt.show_coordinate_frame = True
                opt.background_color = np.asarray([0.5, 0.5, 0.5])
                # Add a sphere at 0,0,0
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                sphere.paint_uniform_color([0.5, 0.0, 0.5])
                viewer.add_geometry(sphere)
                viewer.run()
                viewer.destroy_window()
                # o3d.visualization.draw_geometries([*grippers, cloud], show_cooridates_frame=True)
                # o3d.visualization.draw_geometries([grippers[0], cloud])
        else:
            print('Warning, no grasps after filtering!')
        toc = time.time() - tic
                
        return grasps, scores, np.zeros(len(grasps)), toc # last two returns are dummy

    def depth_to_pcl(self, depths, intrinsic, colors=None):
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        scale = 1.0 # GIGA gives them scaled
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        points = np.stack([points_x, points_y, points_z], axis=-1)
        # remove outliers and points outside lims
        xmin, xmax, ymin, ymax, zmin, zmax = self.lims
        mask = (points_z > zmin) & (points_z < zmax) & \
                (points_x > xmin) & (points_x < xmax) & \
                (points_y > ymin) & (points_y < ymax)
        points = points[mask].astype(np.float32)
        # remove points outside lims
        if colors is not None:
            colors = colors[mask].astype(np.float32)
        # print('PCL min, max: ', points.min(axis=0), points.max(axis=0))
        return points, colors