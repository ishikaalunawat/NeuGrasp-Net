import time

import numpy as np
import trimesh
from scipy import ndimage
import torch

#from vgn import vis
from vgn.grasp import *
from vgn.utils.transform import Transform, Rotation
from vgn.networks import load_network
from vgn.utils import visual
from vgn.utils.implicit import as_mesh
from vgn.grasp_sampler import GpgGraspSamplerPcl

LOW_TH = 0.5
axes_cond = lambda x,z: np.isclose(np.abs(np.dot(x, z)), 1.0, 1e-4)


class VGNImplicit(object):
    def __init__(self, model_path, model_type, best=False, force_detection=False, qual_th=0.9, out_th=0.5, visualize=False, resolution=40, **kwargs):
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.net = load_network(model_path, self.device, model_type=model_type)
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
            
    
    def __call__(self, state, scene_mesh=None, aff_kwargs={}):
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

        # Use GPG grasp sampling:
        # Optional: Get GT point cloud from scene mesh:
        # Get scene point cloud and normals using ground truth meshes
        o3d_scene_mesh = scene_mesh.as_open3d
        o3d_scene_mesh.compute_vertex_normals()
        pc_extended = o3d_scene_mesh.sample_points_uniformly(number_of_points=1000)

        sampler = GpgGraspSamplerPcl(0.05-0.0075) # Franka finger depth is actually a little less than 0.05
        safety_dist_above_table = 0.05 # table is spawned at finger_depth
        grasps, pos_queries, rot_queries = sampler.sample_grasps(pc_extended, num_grasps=40, max_num_samples=180,
                                            safety_dis_above_table=safety_dist_above_table, show_final_grasps=False, verbose=False)
        self.qual_th = 0.5        
        best_only = False # Show all grasps and not just the best one
        # Use standard GIGA grasp sampling:
        # points, normals = self.sample_grasp_points(pc_extended, self.finger_depth)
        # pos_queries, rot_queries = self.get_grasp_queries(points, normals) # Grasp queries :: (pos ;xyz, rot ;as quat)
        # best_only = True
        
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
        qual_vol, width_vol = predict(tsdf_vol, (pos_queries, rot_queries), self.net, self.device)
        
        # reject voxels with predicted widths that are too small or too large
        # qual_vol, width_vol = process(tsdf_process, qual_vol, rot, width_vol, out_th=self.out_th)
        min_width=0.033
        max_width=0.233
        qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

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
            # Need coloured mesh from affordance_visual()
            grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
            composed_scene = trimesh.Scene(colored_scene_mesh)
            for i, g_mesh in enumerate(grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
            bad_grasp_mesh_list = [visual.grasp2mesh(g, s, color='red') for g, s in zip(bad_grasps, bad_scores)]
            for i, g_mesh in enumerate(bad_grasp_mesh_list):
                composed_scene.add_geometry(g_mesh, node_name=f'bad_grasp_{i}')
            return grasps, scores, toc, composed_scene
        else:
            return grasps, scores, toc
    

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

def predict(tsdf_vol, pos, net, device):
    #_, rot = pos
    # move input to the GPU
    tsdf_vol = torch.from_numpy(tsdf_vol).to(device)

    # forward pass
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