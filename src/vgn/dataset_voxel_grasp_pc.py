import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list

class DatasetVoxelGraspPC(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, augment=False):
        self.root = root# Why don't you just read the pandas df here?????
        self.augment = augment
        self.raw_root = raw_root
        self.num_th = 32 # Unused?
        self.df = read_df_with_surface_clouds(raw_root)
        self.size, _, _, _ = read_setup(raw_root)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        rotation = self.df.loc[i, "qx":"qw"].to_numpy(np.single).reshape(1, 4) # <- Changed to predict only grasp quality
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single).reshape(1, 3)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.long)
        voxel_grid = read_voxel_grid(self.root, scene_id) # TODO: Can I load the whole thing into memory?
        if self.augment:
            voxel_grid, ori, pos = apply_aug_transform(voxel_grid, ori, pos)
        grasp_pc = self.read_grasp_pc(self.df.loc[i, "grasp_id"]) # Load grasp point cloud
        
        # Normalize
        pos = pos / self.size - 0.5
        width = width / self.size
        grasp_pc = grasp_pc / self.size - 0.5

        # Transform grasp point cloud to local frame and return this too
        grasp_pc_local = transform_to_frame(grasp_pc, Rotation.from_quat(rotation[0]), pos[0])
        
        tsdf, y, rot = voxel_grid[0], (label, width), rotation # Predict grasp quality, width

        grasp_query = (pos, rot, grasp_pc_local, grasp_pc)
        
        return tsdf, y, grasp_query

    def read_grasp_pc(self, grasp_id):
        grasp_pc_path = self.raw_root / 'grasp_point_clouds' / (str(grasp_id) + '.npz')
        grasp_pc_data = np.load(grasp_pc_path)
        grasp_pc = grasp_pc_data['pc']
        return grasp_pc

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene


class DatasetVoxelGraspPCOcc(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, num_point_occ=8000, augment=False):
        self.root = root# Why don't you just read the pandas df here?????
        self.augment = augment
        self.num_point = num_point
        self.num_point_occ = num_point_occ
        self.raw_root = raw_root
        self.num_th = 32 # Unused?
        self.df = read_df_with_surface_clouds(raw_root)
        self.size, _, _, _ = read_setup(raw_root)
        self.max_points_grasp_pc = read_json(raw_root / "grasp_cloud_setup.json")["max_points"]

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        rotation = self.df.loc[i, "qx":"qw"].to_numpy(np.single).reshape(1, 4) # <- Changed to predict only grasp quality
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single).reshape(1, 3)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"].astype(np.long)
        voxel_grid = read_voxel_grid(self.root, scene_id) # TODO: Can I load the whole thing into memory?
        if self.augment:
            voxel_grid, ori, pos = apply_aug_transform(voxel_grid, ori, pos)
        grasp_pc = self.read_grasp_pc(self.df.loc[i, "grasp_id"]) # Load grasp point cloud
        
        # Normalize
        pos = pos / self.size - 0.5
        width = width / self.size
        grasp_pc = grasp_pc / self.size - 0.5

        # Transform grasp point cloud to local frame and return this too
        grasp_pc_local = transform_to_frame(grasp_pc, Rotation.from_quat(rotation[0]), pos[0])
        # If the number of points is less than the maximum number of points, pad with zeros
        if grasp_pc.shape[0] < self.max_points_grasp_pc:
            grasp_pc = np.vstack((grasp_pc, np.zeros((self.max_points_grasp_pc - grasp_pc.shape[0], 3))))
            grasp_pc_local = np.vstack((grasp_pc_local, np.zeros((self.max_points_grasp_pc - grasp_pc_local.shape[0], 3))))
        # assert label==None, f"{i}"
        tsdf, y, rot = voxel_grid[0], (label, width), rotation # <- Changed to predict only grasp quality
        occ_points, occ = self.read_occ(scene_id, self.num_point_occ) # TODO: Can I load the whole thing into memory?
        occ_points = occ_points / self.size - 0.5

        grasp_query = (pos, rot, grasp_pc_local, grasp_pc)
        
        return tsdf, y, grasp_query, occ_points, occ

    def read_grasp_pc(self, grasp_id):
        grasp_pc_path = self.raw_root / 'grasp_point_clouds_world_reduced' / (str(grasp_id) + '.npz')
        grasp_pc_data = np.load(grasp_pc_path)
        grasp_pc = grasp_pc_data['pc']
        return grasp_pc

    def read_occ(self, scene_id, num_point):
        occ_paths = list((self.raw_root / 'occ' / scene_id).glob('*.npz'))
        path_idx = torch.randint(high=len(occ_paths), size=(1,), dtype=int).item()
        occ_path = occ_paths[path_idx]
        occ_data = np.load(occ_path)
        points = occ_data['points']
        occ = occ_data['occ']
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene

def transform_to_frame(points, orientation, position):

    # transform point set to T frame
    T = Transform(orientation, position)
    T_inv = T.inverse()
    
    points_transformed = T_inv.transform_point(points)

    return points_transformed

def apply_aug_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    z_offset = np.random.uniform(6, 34) - position[2]

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position

def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]