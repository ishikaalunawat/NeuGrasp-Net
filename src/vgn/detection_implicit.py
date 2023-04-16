from pathlib import Path
import time

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy import ndimage
import torch

#from vgn import vis
from vgn.grasp import *
from vgn.io import read_json
from vgn.perception import CameraIntrinsic
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network
from vgn.utils import visual
from vgn.utils.implicit import as_mesh
from vgn.grasp_sampler import GpgGraspSamplerPcl
from vgn.utils.misc import apply_noise

LOW_TH = 0.5
axes_cond = lambda x,z: np.isclose(np.abs(np.dot(x, z)), 1.0, 1e-4)


class VGNImplicit(object):
    def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, **kwargs):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.net = load_network(model_path, self.device, model_type=model_type)
        self.model_type = model_type
        self.qual_th = qual_th
        self.best = best
        self.force_detection = force_detection
        self.out_th = out_th
        self.visualize = visualize
        
        self.resolution = resolution
        self.finger_depth = 0.05
        #x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / self.resolution, steps=self.resolution))
        # 1, self.resolution, self.resolution, self.resolution, 3
        #pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0).to(self.device)
        #self.pos = pos.view(1, self.resolution * self.resolution * self.resolution, 3)
        
    @staticmethod
    def sample_grasp_points(pcd, finger_depth=0.05, eps=0.1):
        # Use masks instead of while loop
        # points Shape (N, 3)
        # normals Shape (N, 3)
        points = np.asarray(pcd.points)        
        normals = np.asarray(pcd.normals)
        mask  = normals[:,-1] > -0.1
        grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
        points = points[mask] + normals[mask] * grasp_depth
        return points, normals[mask]
    
    def get_grasp_queries(self, points, normals):
        # Initializing axes (PyB cam)
        z_axes = -normals
        x_axes = []
        y_axes = []
        num_rotations = int(self.resolution**3//len(points)) + 1

        # Possible x axis
        x1 = np.r_[1.0, 0.0, 0.0]
        x2 = np.r_[0.0, 1.0, 0.0]

        # Defining x_axis and y_axis for each point based on dot product condition (overlapping x and z)
        y_axes = [np.cross(z, x2) if axes_cond(x1,z) else np.cross(z, x1) for z in z_axes]
        x_axes = [np.cross(y, z) for y, z in zip(y_axes, z_axes)]
        
        # Defining rotation matrix with fixed (roll, pitch)
        R = [Rotation.from_matrix(np.vstack((x, y, z)).T) for x,y,z in zip(x_axes, y_axes, z_axes)]

        # Varying yaws from 0, PI with evenly spaced steps
        yaws = np.linspace(0, np.pi, num_rotations)
        pos_queries = []
        rot_queries = []

        # Loop for saving queries
        for i, p in enumerate(points):
            for yaw in yaws:
                ori = R[i] * Rotation.from_euler("z", yaw)
                pos_queries.append(p)
                rot_queries.append(ori.as_quat())

        return pos_queries, rot_queries
            
    
    def __call__(self, state, scene_mesh=None, sim=None, debug_data=None, seed=None, aff_kwargs={}):
        if hasattr(state, 'tsdf_process'):
            tsdf_process = state.tsdf_process
        else:
            tsdf_process = state.tsdf

        if isinstance(state.tsdf, np.ndarray):
            tsdf_vol = state.tsdf
            voxel_size = 0.3 / self.resolution
            size = 0.3
            pc_extended = state.pc_extended # Using extended PC for grasp sampling

        else:
            tsdf_vol = state.tsdf.get_grid()
            voxel_size = tsdf_process.voxel_size
            tsdf_process = tsdf_process.get_grid()
            size = state.tsdf.size
            pc_extended = state.pc_extended # Using extended PC for grasp sampling

        tic = time.time()

        if debug_data is not None:
            # debug validation
            # get grasps from dataset
            grasps, pos_queries, rot_queries, debug_labels = [], [], [], []
            # Optional: Use only successful grasps for debug
            # debug_data = debug_data[debug_data["label"] == 1]
            for i in debug_data.index:
                rota = debug_data.loc[i, "qx":"qw"].to_numpy(np.single)
                posi = debug_data.loc[i, "x":"z"].to_numpy(np.single)
                grasps.append(Grasp(Transform(Rotation.from_quat(rota), posi), sim.gripper.max_opening_width))
                pos_queries.append(posi)
                rot_queries.append(rota)
                debug_labels.append(debug_data.loc[i, "label"])
                # if len(grasps) >= 16:
                #     break
            best_only = False # Show all grasps and not just the best one
        else:
            # Use GPG grasp sampling:
            # Optional: Get GT point cloud from scene mesh:
            # Get scene point cloud and normals using ground truth meshes
            # o3d_scene_mesh = scene_mesh.as_open3d
            # o3d_scene_mesh.compute_vertex_normals()
            # pc_extended = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)
            # Optional: Downsample point cloud if too large
            pc_extended_down = pc_extended.voxel_down_sample(voxel_size=0.005)

            sampler = GpgGraspSamplerPcl(0.05-0.0075) # Franka finger depth is actually a little less than 0.05
            safety_dist_above_table = 0.05 # table is spawned at finger_depth
            grasps, pos_queries, rot_queries = sampler.sample_grasps(pc_extended_down, num_grasps=40, max_num_samples=180,
                                                safety_dis_above_table=safety_dist_above_table, show_final_grasps=False, verbose=False)
            self.qual_th = 0.5
            best_only = False # Show all grasps and not just the best one

            # Use standard GIGA grasp sampling:
            # points, normals = self.sample_grasp_points(pc_extended, self.finger_depth)
            # pos_queries, rot_queries = self.get_grasp_queries(points, normals) # Grasp queries :: (pos ;xyz, rot ;as quat)
            # best_only = True

            if (len(pos_queries) < 2):
                print("[Warning]: No grasps found by GPG")
                # import pdb; pdb.set_trace()
                if self.visualize:
                    return [], [], 0.0, scene_mesh
                else:
                    return [], [], 0.0

        # Convert to torch tensor
        pos_queries = torch.Tensor(pos_queries).view(1, -1, 3)
        rot_queries = torch.Tensor(rot_queries).view(1, -1, 4)
        # Choose queries upto resolution^3 (TODO: Check if needed)
        # chosen_indices = np.random.choice(len(pos_queries),size=(self.resolution**3))
        # pos_queries = pos_queries[chosen_indices].view(1, -1, 3)
        # rot_queries = rot_queries[chosen_indices].view(1, -1, 4)
        # Normalize 3D pos queries
        pos_queries = pos_queries/size - 0.5

        # Variable rot_vol replaced with ==> rot
        assert tsdf_vol.shape == (1, self.resolution, self.resolution, self.resolution)

        # Query network
        if 'neu' in self.model_type:
            # Also generate grasp point clouds
            render_settings = read_json(Path("data/pile/data_pile_train_random_raw_4M_radomized_views/grasp_cloud_setup.json"))
            # remove table because we dont want to render it
            sim.world.remove_body(sim.world.bodies[0]) # normally table is the first body
            grasps_pc_local = torch.zeros((len(grasps),render_settings['max_points'],3))
            grasps_pc = grasps_pc_local.clone()
            bad_indices = []
            for ind, grasp in enumerate(grasps):
                result, grasp_pc_local, grasp_pc = generate_grasp_cloud(sim, render_settings, grasp, scene_mesh, debug=False)
                # import pdb; pdb.set_trace()
                grasp_pc_local = grasp_pc_local / size # - 0.5 DONT SUBTRACT HERE!
                grasp_pc = grasp_pc / size - 0.5
                if result:
                    grasps_pc_local[ind, :grasp_pc_local.shape[0],:] = torch.tensor(grasp_pc_local)
                    grasps_pc[ind, :grasp_pc.shape[0], :] = torch.tensor(grasp_pc)
                else:
                    # no surface points found for these grasps
                    bad_indices.append(ind)
            # Add table back
            sim.place_table(height=sim.gripper.finger_depth)

            # Make separate queries for each grasp
            pos_queries = pos_queries.transpose(0,1)
            rot_queries = rot_queries.transpose(0,1)
            tsdf_vol = np.repeat(tsdf_vol,len(grasps), axis=0)

            qual_vol, width_vol = predict(tsdf_vol, (pos_queries, rot_queries, grasps_pc_local, grasps_pc), self.net, self.device, seed=seed)
            # import pdb; pdb.set_trace()
            qual_vol[bad_indices] = 0.0 # set bad grasp scores to zero
            
            # put back into original shape so that we can use the same code as before
            pos_queries = pos_queries.transpose(0,1)
            rot_queries = rot_queries.transpose(0,1)
        else:
            qual_vol, width_vol = predict(tsdf_vol, (pos_queries, rot_queries), self.net, self.device, seed=seed)
        
        # reject voxels with predicted widths that are too small or too large
        # qual_vol, width_vol = process(tsdf_process, qual_vol, rot, width_vol, out_th=self.out_th)
        min_width=0.033
        max_width=0.233
        # qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

        # qual_vol = bound(qual_vol, voxel_size) # TODO: Check if needed

        # Reshape to 3D grid. TODO: Check if needed. Update: Not needed
        # qual_vol = qual_vol.reshape((self.resolution, self.resolution, self.resolution))
        # rot_queries = rot_queries.reshape((self.resolution, self.resolution, self.resolution, 4))
        # width_vol = width_vol.reshape((self.resolution, self.resolution, self.resolution))

        if self.visualize:
            ### TODO - check affordance - visual (WEIRD)
            colored_scene_mesh = scene_mesh
            # colored_scene_mesh = visual.affordance_visual(qual_vol, rot, scene_mesh, size, self.resolution, **aff_kwargs)
        grasps, scores, bad_grasps, bad_scores = select(qual_vol.copy(), pos_queries[0].squeeze(), rot_queries[0].squeeze(), width_vol, best_only=best_only, threshold=self.qual_th, force_detection=self.force_detection, max_filter_size=8 if self.visualize else 4)
        toc = time.time() - tic

        bad_grasps, bad_scores = np.asarray(bad_grasps), np.asarray(bad_scores)
        grasps, scores = np.asarray(grasps), np.asarray(scores)

        new_bad_grasps = []
        new_grasps = []
        if len(grasps) > 0:
            if self.best:
                p = np.arange(len(grasps))
                p_bad = np.arange(len(bad_grasps))
            else:
                p = np.random.permutation(len(grasps))
                p_bad = np.random.permutation(len(bad_grasps))
            for bad_g in bad_grasps[p_bad]:
                pose = bad_g.pose
                # Un-normalize
                pose.translation = (pose.translation + 0.5) * size
                width = bad_g.width * size
                new_bad_grasps.append(Grasp(pose, width))
            for g in grasps[p]:
                pose = g.pose
                # Un-normalize
                pose.translation = (pose.translation + 0.5) * size
                width = g.width * size
                new_grasps.append(Grasp(pose, width))
            bad_scores = bad_scores[p_bad]
            scores = scores[p]
        bad_grasps = new_bad_grasps
        grasps = new_grasps

        if self.visualize:
            composed_scene = colored_scene_mesh
            # # Need coloured mesh from affordance_visual()
            # grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            # composed_scene = trimesh.Scene(colored_scene_mesh)
            # for i, g_mesh in enumerate(grasp_mesh_list):
            #     composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            # bad_grasp_mesh_list = [visual.grasp2mesh(g, s, color='red') for g, s in zip(bad_grasps, bad_scores)]
            # for i, g_mesh in enumerate(bad_grasp_mesh_list):
            #     composed_scene.add_geometry(g_mesh, node_name=f'bad_grasp_{i}')
            # # Optional: Show grasps (for debugging)
            # composed_scene.show()
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc

def generate_grasp_cloud(sim, render_settings, grasp, scene_mesh=None, debug=False):
    if debug:
        # DEBUG: Viz scene point cloud and normals using ground truth meshes
        o3d_scene_mesh = scene_mesh.as_open3d
        o3d_scene_mesh.compute_vertex_normals()
        pc = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)
        pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pc.points).shape[0], 1)))
        o3d.visualization.draw_geometries([pc])

    # Create our own camera(s)
    width, height = render_settings['camera_image_res'], render_settings['camera_image_res'] # relatively low resolution (128 by default)
    width_fov  = np.deg2rad(render_settings['camera_fov']) # angular FOV (120 by default)
    height_fov = np.deg2rad(render_settings['camera_fov']) # angular FOV (120 by default)
    f_x = width  / (np.tan(width_fov / 2.0))
    f_y = height / (np.tan(height_fov / 2.0))
    intrinsic = CameraIntrinsic(width, height, f_x, f_y, width/2, height/2)
    # To capture 5cms on both sides of the gripper, using a 120 deg FOV, we need to be atleast 0.05/tan(60) = 2.8 cms away
    height_max_dist = sim.gripper.max_opening_width/2.5
    width_max_dist  = sim.gripper.max_opening_width/2.0 + 0.005 # 0.5 cm extra
    # width_max_dist += 0.02 # 2 cms extra?
    dist_from_gripper = width_max_dist/np.tan(width_fov/2.0)
    min_measured_dist = 0.001
    max_measured_dist = dist_from_gripper + sim.gripper.finger_depth + 0.005 # 0.5 cm extra
    camera = sim.world.add_camera(intrinsic, min_measured_dist, max_measured_dist+0.05) # adding 5cm extra for now but will filter it below
    if render_settings['three_cameras']:
        # Use one camera for wrist and two cameras for the fingers
        finger_height_max_dist = sim.gripper.max_opening_width/2.5
        finger_width_max_dist = sim.gripper.finger_depth/2.0 + 0.005 # 0.5 cm extra
        dist_from_finger = finger_width_max_dist/np.tan(width_fov/2.0)
        finger_max_measured_dist = dist_from_finger + 0.95*sim.gripper.max_opening_width
        finger_camera  = sim.world.add_camera(intrinsic, min_measured_dist, finger_max_measured_dist+0.05) # adding 5cm extra for now but will filter it below
    
    # Load the grasp
    pos = grasp.pose.translation
    rotation = grasp.pose.rotation
    grasp = Grasp(Transform(rotation, pos), sim.gripper.max_opening_width)
    if debug:
        # DEBUG: Viz grasp
        grasps_scene = trimesh.Scene()
        from vgn.utils import visual
        grasp_mesh_list = [visual.grasp2mesh(grasp)]
        for i, g_mesh in enumerate(grasp_mesh_list):
            grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
        # grasps_scene.show()
        composed_scene = trimesh.Scene([scene_mesh, grasps_scene])
        composed_scene.show()

    ## Move camera to grasp offset frame
    grasp_center = grasp.pose.translation
    # Unfortunately VGN/GIGA grasps are not in the grasp frame we want (frame similar to PointNetGPD), so we need to transform them
    grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', np.pi/2) * Rotation.from_euler('Z', np.pi)
    grasp_tf = Transform(grasp_frame_rot, grasp_center).as_matrix()
    offset_pos =  (grasp_tf @ np.array([[-dist_from_gripper],[0],[0],[1.]]))[:3].squeeze() # Move to offset frame
    # Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
    grasp_up_axis = grasp_tf.T[2,:3] # np.array([0.0, 0.0, 1.0]) # grasp_tf z-axis
    extrinsic_bullet = Transform.look_at(eye=offset_pos, center=grasp_center, up=grasp_up_axis)
    ## render image
    depth_img = camera.render(extrinsic_bullet)[1]
    # Optional: Add some dex noise like GIGA
    if render_settings['add_noise']:
        depth_img = apply_noise(depth_img, noise_type='dex')
    if debug:
        # DEBUG: Viz
        plt.imshow(depth_img)
        plt.show()
    
    ## Do the same for the other cameras
    if render_settings['three_cameras']:
        ## Move camera to finger offset frame
        fingers_center =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[0],[0],[1.]]))[:3].squeeze()
        left_finger_offset_pos  =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[ (dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
        right_finger_offset_pos =  (grasp_tf @ np.array([[sim.gripper.finger_depth/2.0],[-(dist_from_finger + sim.gripper.max_opening_width/2.0)],[0],[1.]]))[:3].squeeze()
        
        # Unfortunately the bullet renderer uses the OpenGL format so we have to use yet another extrinsic
        left_finger_extrinsic_bullet  = Transform.look_at(eye=left_finger_offset_pos,  center=fingers_center, up=grasp_up_axis)
        right_finger_extrinsic_bullet = Transform.look_at(eye=right_finger_offset_pos, center=fingers_center, up=grasp_up_axis)

        ## render image
        left_finger_depth_img  = finger_camera.render(left_finger_extrinsic_bullet )[1]
        right_finger_depth_img = finger_camera.render(right_finger_extrinsic_bullet)[1]
        # Optional: Add some dex noise like GIGA
        if render_settings['add_noise']:
            left_finger_depth_img = apply_noise(left_finger_depth_img, noise_type='dex')
            right_finger_depth_img = apply_noise(right_finger_depth_img, noise_type='dex')
    
    ## Convert to point cloud
    pixel_grid = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.dstack((pixel_grid[0],pixel_grid[1])).reshape(-1, 2)

    # depth_eps = 0.0001
    depth_array = depth_img.reshape(-1)
    relevant_mask = depth_array < (max_measured_dist) #- depth_eps) # only depth values in range
    filt_pixels = np.array(pixels[relevant_mask]) # only consider pixels with depth values in range
    filt_pixels = np.hstack((filt_pixels, np.ones((filt_pixels.shape[0], 2)))) # Homogenous co-ordinates
    # Project pixels into camera space
    filt_pixels[:,:3] *= depth_array[relevant_mask].reshape(-1, 1) # Multiply by depth
    intrinsic_hom = np.eye(4)
    intrinsic_hom[:3,:3] = intrinsic.K
    p_local = np.linalg.inv(intrinsic_hom) @ filt_pixels.T
    # Also filter out points that are more than max dist height
    p_local = p_local[:, p_local[1,:] <  height_max_dist]
    p_local = p_local[:, p_local[1,:] > -height_max_dist]
    p_world = np.linalg.inv(extrinsic_bullet.as_matrix()) @ p_local
    surface_pc = o3d.geometry.PointCloud()
    surface_pc.points = o3d.utility.Vector3dVector(p_world[:3,:].T)

    if debug:
        ## DEBUG: Viz point cloud and grasp
        grasp_cam_world_depth_pc = o3d.geometry.PointCloud()
        grasp_cam_world_depth_pc.points = o3d.utility.Vector3dVector(p_world[:3,:].T)
        grasp_cam_world_depth_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (np.asarray(grasp_cam_world_depth_pc.points).shape[0], 1)))
        # viz original pc and gripper
        o3d_gripper_mesh = as_mesh(grasps_scene).as_open3d
        gripper_pc = o3d_gripper_mesh.sample_points_uniformly(number_of_points=3000)
        gripper_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 1, 0]), (np.asarray(gripper_pc.points).shape[0], 1)))
        o3d.visualization.draw_geometries([gripper_pc, grasp_cam_world_depth_pc, pc])

    if render_settings['three_cameras']:
        left_finger_depth_array = left_finger_depth_img.reshape(-1)
        left_relevant_mask = left_finger_depth_array < (finger_max_measured_dist)# - depth_eps) # only depth values in range
        left_filt_pixels = np.array(pixels[left_relevant_mask]) # only consider pixels with depth values in range
        
        left_filt_pixels = np.hstack((left_filt_pixels, np.ones((left_filt_pixels.shape[0], 2)))) # Homogenous co-ordinates
        # Project pixels into camera space
        left_filt_pixels[:,:3] *= left_finger_depth_array[left_relevant_mask].reshape(-1, 1) # Multiply by depth
        left_p_local = np.linalg.inv(intrinsic_hom) @ left_filt_pixels.T
        # Also filter out points that are more than max dist height and width
        left_p_local = left_p_local[:, left_p_local[0,:] <  finger_width_max_dist]
        left_p_local = left_p_local[:, left_p_local[0,:] > -finger_width_max_dist]
        left_p_local = left_p_local[:, left_p_local[1,:] <  finger_height_max_dist]
        left_p_local = left_p_local[:, left_p_local[1,:] > -finger_height_max_dist]
        left_p_world = np.linalg.inv(left_finger_extrinsic_bullet.as_matrix()) @ left_p_local

        right_finger_depth_array = right_finger_depth_img.reshape(-1)
        right_relevant_mask = right_finger_depth_array < (finger_max_measured_dist)# - depth_eps) # only depth values in range
        right_filt_pixels = np.array(pixels[right_relevant_mask]) # only consider pixels with depth values in range
        
        right_filt_pixels = np.hstack((right_filt_pixels, np.ones((right_filt_pixels.shape[0], 2)))) # Homogenous co-ordinates
        # Project pixels into camera space
        right_filt_pixels[:,:3] *= right_finger_depth_array[right_relevant_mask].reshape(-1, 1) # Multiply by depth
        right_p_local = np.linalg.inv(intrinsic_hom) @ right_filt_pixels.T
        # Also filter out points that are more than max dist height and width
        right_p_local = right_p_local[:, right_p_local[0,:] <  finger_width_max_dist]
        right_p_local = right_p_local[:, right_p_local[0,:] > -finger_width_max_dist]
        right_p_local = right_p_local[:, right_p_local[1,:] <  finger_height_max_dist]
        right_p_local = right_p_local[:, right_p_local[1,:] > -finger_height_max_dist]
        right_p_world = np.linalg.inv(right_finger_extrinsic_bullet.as_matrix()) @ right_p_local    

        if debug:
            # Viz
            left_cam_world_depth_pc = o3d.geometry.PointCloud()
            left_cam_world_depth_pc.points = o3d.utility.Vector3dVector(left_p_world[:3,:].T)
            right_cam_world_depth_pc = o3d.geometry.PointCloud()
            right_cam_world_depth_pc.points = o3d.utility.Vector3dVector(right_p_world[:3,:].T)

            left_cam_world_depth_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (np.asarray(left_cam_world_depth_pc.points).shape[0], 1)))
            right_cam_world_depth_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 1, 0]), (np.asarray(right_cam_world_depth_pc.points).shape[0], 1)))
            o3d.visualization.draw_geometries([left_cam_world_depth_pc, right_cam_world_depth_pc, gripper_pc, grasp_cam_world_depth_pc, pc])

        # Combine surface point cloud
        combined_world_points = np.hstack((p_world, left_p_world, right_p_world))
        surface_pc.points = o3d.utility.Vector3dVector(combined_world_points[:3,:].T)

    down_surface_pc = surface_pc.voxel_down_sample(voxel_size=render_settings['voxel_downsample_size'])
    # If more than max points, uniformly sample
    if len(down_surface_pc.points) > render_settings['max_points']:
        indices = np.random.choice(np.arange(len(down_surface_pc.points)), render_settings['max_points'], replace=False)
        down_surface_pc = down_surface_pc.select_by_index(indices)
    # If less than min points, skip this grasp
    if len(down_surface_pc.points) < render_settings['min_points']:
        # Points are too few! skip this grasp
        print("[Warning]: Points are too few! Skipping this grasp...")
        # import pdb; pdb.set_trace()
        return False, 0, 0
    if debug:
        # viz original pc and gripper
        down_surface_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.45, 0.]), (np.asarray(down_surface_pc.points).shape[0], 1)))
        o3d.visualization.draw_geometries([down_surface_pc, gripper_pc, pc])
    
    grasp_pc = np.asarray(down_surface_pc.points)
    T_inv = Transform(rotation, pos).inverse()
    grasp_pc_local = T_inv.transform_point(grasp_pc)

    # pc_trimesh = trimesh.points.PointCloud(down_surface_pc.points)
    # pc_colors = np.array([[255, 255, 0] for i in down_surface_pc.points])
    # # increase sphere size of trimesh points
    # pc_trimesh.colors = np.array([255, 125, 0])
    # box = trimesh.creation.box(extents=[0.5, 0.5, 0.1])
    # box.visual.face_colors = [0.9, 0.9, 0.9, 1.0]
    # translation = [0.15, 0.15, -0.05+0.05]
    # box.apply_translation(translation)
    # trimesh.Scene([composed_scene, pc_trimesh, box]).show(line_settings= {'point_size': 20})
    # import pdb; pdb.set_trace()

    if debug:
        # viz local and global and original pc and gripper
        gripper_pc_local = T_inv.transform_point(np.asarray(gripper_pc.points))
        gripper_pc_local_o3d = o3d.geometry.PointCloud()
        gripper_pc_local_o3d.points = o3d.utility.Vector3dVector(gripper_pc_local)
        gripper_pc_local_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 1, 0.]), (np.asarray(gripper_pc_local_o3d.points).shape[0], 1)))
        grasp_pc_local_o3d = o3d.geometry.PointCloud()
        grasp_pc_local_o3d.points = o3d.utility.Vector3dVector(grasp_pc_local)
        grasp_pc_local_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.45, 0.]), (np.asarray(grasp_pc_local_o3d.points).shape[0], 1)))
        o3d.visualization.draw_geometries([grasp_pc_local_o3d, gripper_pc_local_o3d, down_surface_pc, gripper_pc, pc])

    return True, grasp_pc_local, grasp_pc

def bound(qual_vol, voxel_size, limit=[0.02, 0.02, 0.055]):
    # avoid grasp out of bound [0.02  0.02  0.055]
    x_lim = int(limit[0] / voxel_size)
    y_lim = int(limit[1] / voxel_size)
    z_lim = int(limit[2] / voxel_size)
    qual_vol[:x_lim] = 0.0
    qual_vol[-x_lim:] = 0.0
    qual_vol[:, :y_lim] = 0.0
    qual_vol[:, -y_lim:] = 0.0
    qual_vol[:, :, :z_lim] = 0.0
    return qual_vol

def predict(tsdf_vol, pos, net, device, seed=0):
    #_, rot = pos
    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).to(device)

    # forward pass
    # if seed != 100:
    net.eval()
    with torch.no_grad():
        qual_vol, width_vol = net(tsdf_vol, pos)

    # move output back to the CPU
    qual_vol = qual_vol.cpu().squeeze().numpy()
    #rot = rot.cpu().squeeze().numpy()
    width_vol = width_vol.cpu().squeeze().numpy()
    return qual_vol, width_vol

def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1,
    min_width=0.033,
    max_width=0.233,
    out_th=0.5
):
    tsdf_vol = tsdf_vol.squeeze()

    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=int(gaussian_filter_sigma), mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > out_th
    inside_voxels = np.logical_and(1e-3 < tsdf_vol, tsdf_vol < out_th)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0

    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    #return qual_vol, rot_vol, width_vol
    return qual_vol, width_vol


def select(qual_vol, center_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4, force_detection=False, best_only=False):
    
    bad_mask = np.where(qual_vol<LOW_TH, 1.0, 0.0)

    qual_vol[qual_vol < LOW_TH] = 0.0
    if force_detection and (qual_vol >= threshold).sum() == 0:
        pass
        # best_only = True
    else:
        # threshold on grasp quality
        qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression. TODO: Check if needed
    # max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    # qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    bad_grasps, bad_scores = [], []
    for index in np.argwhere(bad_mask):
        index = index.squeeze()
        ori = Rotation.from_quat(rot_vol[index])
        pos = center_vol[index].numpy()
        bad_grasp, bad_score = Grasp(Transform(ori, pos), width_vol[index]), qual_vol[index]
        # bad_grasp, bad_score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        bad_grasps.append(bad_grasp)
        bad_scores.append(bad_score)
    
    grasps, scores = [], []
    for index in np.argwhere(mask):
        index = index.squeeze()
        ori = Rotation.from_quat(rot_vol[index])
        pos = center_vol[index].numpy()
        grasp, score = Grasp(Transform(ori, pos), width_vol[index]), qual_vol[index]
        # grasp, score = select_index(qual_vol, center_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    
    sorted_bad_grasps = [bad_grasps[i] for i in np.argsort(bad_scores)]
    sorted_bad_scores = [bad_scores[i] for i in np.argsort(bad_scores)]
    sorted_grasps = [grasps[i] for i in reversed(np.argsort(scores))]
    sorted_scores = [scores[i] for i in reversed(np.argsort(scores))]

    if best_only:
        # Take only best grasp(s)
        if len(sorted_grasps) >= 100:
            max_grasps = 10
        else:
            max_grasps = 1

        sorted_grasps = [sorted_grasps[i] for i in range(max_grasps)]
        sorted_scores = [sorted_scores[i] for i in range(max_grasps)]

        sorted_bad_grasps = [sorted_bad_grasps[i] for i in range(max_grasps)]
        sorted_bad_scores = [sorted_bad_scores[i] for i in range(max_grasps)]

        
    return sorted_grasps, sorted_scores, sorted_bad_grasps, sorted_bad_scores


def select_index(qual_vol, center_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[i, j, k])
    #pos = np.array([i, j, k], dtype=np.float64)
    pos = center_vol[i, j, k].numpy()
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score