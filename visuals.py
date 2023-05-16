import os
import numpy as np
from vgn.networks import load_network
import torch
import plotly.graph_objects as go
from vgn.dataset_voxel_grasp_pc import DatasetVoxelGraspPCOcc
from pathlib import Path
import random
import argparse
import mcubes

from scripts.rendering.voxel_graph import VoxelData

# from vgn.simulation import ClutterRemovalSim
# from vgn.utils.transform import Rotation, Transform
# from vgn.utils.implicit import get_scene_from_mesh_pose_list, as_mesh
# from vgn.perception import *

np.random.seed(0)
torch.manual_seed(0)


unsq = lambda x: torch.as_tensor(x).unsqueeze(0).float()

def load_data(data):
    # pc, y, grasp_query, occ_points, occ =  data[index]
    pc, (label, width), (pos, rotations, grasps_pc_local, grasps_pc), pos_occ, occ_value = data
    pc, label, width, pos, rotations, grasps_pc_local, grasps_pc, pos_occ, occ_value = unsq(pc), unsq(label), unsq(width), unsq(pos), unsq(rotations), unsq(grasps_pc_local), unsq(grasps_pc), unsq(pos_occ), unsq(occ_value)
    return pc, (label, width, occ_value), (pos, rotations, grasps_pc_local, grasps_pc), pos_occ

def make_occ_grid(resolution):
    x, y, z = torch.meshgrid(torch.linspace(start=-0.5, end=0.5 - 1.0 / resolution, steps= resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / resolution, steps=resolution), torch.linspace(start=-0.5, end=0.5 - 1.0 / resolution, steps=resolution))
    # 1, self.resolution, self.resolution, self.resolution, 3
    pos = torch.stack((x, y, z), dim=-1).float().unsqueeze(0)
    pos = pos.view(1, resolution*resolution*resolution, 3)
    return pos

def create_plot(geometry, camera_eye):
    fig = go.Figure(geometry)
    camera = dict(eye=dict(x=camera_eye[0], y=camera_eye[1], z=camera_eye[2]))
    fig.update_layout(scene_camera=camera)
    fig.update_xaxes(tickmode='linear')
    fig.update_layout(autosize=False,
                      width  = 1600,
                      height = 1600)
    return fig

def make_save_fig(geometry, camera_eye, file_name, type="mesh"):
    camera_eye[2, 3] += 0.015
    camera_eye = np.linalg.inv(camera_eye.squeeze())[:3, 3]

    if type=="voxel":
        Voxels = VoxelData(geometry.numpy().squeeze())
        voxels = go.Mesh3d(
                x=Voxels.vertices[0],
                y=Voxels.vertices[1],
                z=Voxels.vertices[2],
                i=Voxels.triangles[0],
                j=Voxels.triangles[1],
                k=Voxels.triangles[2],
                color='cornflowerblue',
                opacity=1)
        create_plot(voxels, camera_eye).write_image(file = file_name)

    else:
        vertices, triangles = geometry
        x, y, z = vertices.T
        i, j, k = triangles.T
        mesh = go.Mesh3d(x=x, y=y, z=z,
                         i=i, j=j, k=k,
                         color='rgb(194, 30, 86)')
        create_plot(mesh, camera_eye).write_image(file = file_name)


def main(args):
    device = "cpu" if not args.cuda else "cuda"

    cam_eye = np.load(args.cam_eye_path)
    net = load_network(args.model_path, device=device, model_type=args.net)
    net = net.eval()

    # NOTE: Renamed "grasps_with_clouds_gt".csv to "grasps_with_clouds.csv"
    data = DatasetVoxelGraspPCOcc(args.root, args.raw_root, use_grasp_occ=False, num_point_occ=8000)
    scenes = ["58c827f23ec34543a72a4e7bc6fe362d", "d7e7d6e296ec4abfaad79acd252ac9b3", "3a8723a4d0f34067a925835bd39ec9d5"] if not args.random else random.sample(range(data.df["scene_id"]), args.num_scenes)
    
    for scene in scenes:
        print("Processing scene: ", scene)
        index = random.choice(data.df[data.df.scene_id==scene].index)
        tsdf, _, _, _= load_data(data[index])
        pos = make_occ_grid(args.resolution)
        c = net.encode_inputs(tsdf)

        # Reconstruction
        occupancies = net.decoder_tsdf(pos, c,)
        vertices, triangles = mcubes.marching_cubes(occupancies.view(64, 64, 64).detach().numpy(), 0.5)

        
        # Plotting
        (args.save_path / f"scene_{scene}").mkdir(parents=True, exist_ok=True)
        make_save_fig((vertices, triangles), cam_eye, file_name= args.save_path / f"scene_{scene}/reconstruction_{scene}.png")
        make_save_fig(tsdf, cam_eye, file_name= args.save_path / f"scene_{scene}/tsdf_{scene}.png", type="voxel")


        # Ground Truth Mesh
        # mesh_list_file = os.path.join(args.raw_root, 'mesh_pose_list', scene + '.npz')
        # mesh_pose_list = np.load(mesh_list_file, allow_pickle=True)['pc']
        # sim = ClutterRemovalSim('pile', 'pile/train', gui=False, data_root=args.data_root) # parameters scene and object_set are not used
        # sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list, table=args.see_table, data_root=args.data_root) # Setting table to False because we don't want to render it
        # scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, data_root=args.data_root)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default="data/pile/data_pile_train_constructed_4M_HighRes_radomized_views_GPG_only")
    parser.add_argument("--raw_root", type=Path, default="data/pile/data_pile_train_random_raw_4M_radomized_views")
    parser.add_argument("--save_path", type=Path, default="figures/")
    parser.add_argument("--net", default="neu_grasp_pn_deeper")
    parser.add_argument("--model_path", type=Path, default='best_models/23-05-01-08-11-39_dataset=data_pile_train_constructed_4M_HighRes_radomized_views,augment=False,net=6d_neu_grasp_pn_deeper,batch_size=32,lr=5e-05,PN_deeper_DIMS_CONT/best_neural_grasp_neu_grasp_pn_deeper_val_acc=0.9097.pt')
    parser.add_argument("--net_with_grasp_occ", type=bool, default='', help="Also use grasp pc occupancy values")
    parser.add_argument("--see_table", type=bool, default=True)
    parser.add_argument("--data_root", type=Path, default="/home/hypatia/6D-DAAD/GIGA")
    parser.add_argument("--size", type=float, default=0.3)
    parser.add_argument("--num_scenes", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--random", type=bool, default=False)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--cam_eye_path", type=Path, default="viewpoint_f614e39ed9df4e1094d569cddc20979b.npy")
    args, _ = parser.parse_known_args()
    # print(args)
    main(args)
