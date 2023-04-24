import os
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt

from vgn.io import *
from vgn.perception import *
from vgn.grasp import Grasp
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list, as_mesh
from vgn.utils.misc import apply_noise

from joblib import Parallel, delayed

def transform_to_frame(points, T):
    return T.inverse().transform_point(points)

def generate_from_existing_grasps(grasp_data_entry, args):
    # Get mesh pose list
    scene_id = grasp_data_entry['scene_id']
    file_name = scene_id + ".npz"
    mesh_pose_list = np.load(args.raw_root / "mesh_pose_list" / file_name, allow_pickle=True)['pc']

    # Re-create the saved simulation
    sim = ClutterRemovalSim('pile', 'pile/train', gui=args.sim_gui) # parameters 'pile' and 'pile/train' are not used
    sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list, table=args.render_table)

    # Obtain point cloud and normals using ground truth meshes
    scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
    o3d_scene_mesh = scene_mesh.as_open3d
    o3d_scene_mesh.compute_vertex_normals()
    pc = o3d_scene_mesh.sample_points_uniformly(number_of_points=args.sample_points)
    pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (np.asarray(pc.points).shape[0], 1)))
    # o3d.visualization.draw_geometries([pc])

    # Load the grasp
    pos = grasp_data_entry["x":"z"].to_numpy(np.single)
    rotation = Rotation.from_quat(grasp_data_entry["qx":"qw"].to_numpy(np.single))
    grasp = Grasp(Transform(rotation, pos), sim.gripper.max_opening_width)
    if args.debug:
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
    grasp_tf = Transform(grasp_frame_rot, grasp_center)
    p_local = transform_to_frame(pc.points, grasp_tf)

    # Find local surface point cloud around grasp
    # X limit (depth)
    p_local = p_local[p_local[:,0] <  sim.gripper.finger_depth]
    p_local = p_local[p_local[:,0] >  0]

    # Y limit (max opening width)
    p_local = p_local[p_local[:,1] <  sim.gripper.max_opening_width/2]
    p_local = p_local[p_local[:,1] > -sim.gripper.max_opening_width/2]

    # Z limit (height)
    p_local = p_local[p_local[:,2] <  sim.gripper.height/2]
    p_local = p_local[p_local[:,2] > -sim.gripper.height/2]

    # Get 50 to 1000 points
    # If more than max points, uniformly sample
    if len(p_local) > args.max_points:
        indices = np.random.choice(np.arange(len(p_local)), args.max_points, replace=False)
        p_local = p_local[indices]
    # If less than min points, skip this grasp
    if len(p_local) < args.min_points:
        # Points are too few! skip this grasp
        print("[Warning]: Points are too few! Skipping this grasp...")
        return False

    # Convert local pointcloud to world frame    
    p_world = transform_to_frame(p_local, grasp_tf.inverse())
    
    # if args.debug:
    #     # viz original pc and gripper
    #     down_surface_pc.colors = o3d.utility.Vector3dVector(np.tile(np.array([1.0, 0.45, 0.]), (np.asarray(down_surface_pc.points).shape[0], 1)))
    #     o3d.visualization.draw_geometries([down_surface_pc, gripper_pc, pc])
    

    if not args.debug:
        ## Save grasp & surface point cloud
        grasp_id = grasp_data_entry['grasp_id']
        # save grasp to another grasps_with_clouds csv
        grasp_data_entry["qx"]
        append_csv(args.csv_path,
                grasp_id, scene_id, grasp_data_entry["qx"], grasp_data_entry["qy"], grasp_data_entry["qz"], grasp_data_entry["qw"],
                grasp_data_entry['x'], grasp_data_entry['y'], grasp_data_entry['z'], grasp_data_entry['width'], grasp_data_entry['label'])
        # save surface point cloud
        surface_pc_path = args.raw_root / "grasp_point_clouds_world" / f"{grasp_id}.npz"
        # Saving grasp-local point clouds in world frame
        np.savez_compressed(surface_pc_path, pc=np.asarray(p_world))

    if args.sim_gui:
        sim.world.p.disconnect()

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate surface point clouds for each grasp")
    parser.add_argument("--raw_root", type=Path)
    parser.add_argument("--sample_points", type=int, default=10000)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object_set", type=str, default="pile/train")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--render_table", type=bool, default=False, help="Also render table in depth images")
    parser.add_argument("--max_points", type=int, default=1000)
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--sim_gui", type=bool, default=False)
    args, _ = parser.parse_known_args()

    if args.raw_root is None:
        raise ValueError("Root directory is not specified")

    # Write grasp cloud parameters
    write_json(dict({"scene": args.scene,
                     "object_set": args.object_set,
                     "render_table": args.render_table,
                     "max_points": args.max_points,
                     "min_points": args.min_points,
                     }), args.raw_root / "grasp_cloud_setup.json")

    # Read all grasp data
    df = read_df(args.raw_root)
    print('Num grasps in raw dataset: %d' % len(df))
    if 'grasp_id' not in df.columns:
        # Add a column for grasp id. Use index values
        df.insert(0,'grasp_id',df.index)
    if not args.debug:
        # Create a directory for storing grasp point clouds
        os.makedirs(args.raw_root / "grasp_point_clouds_world", exist_ok=True)
        # Crate another csv file for storing grasps that have point clouds
        args.csv_path = args.raw_root / "grasps_with_clouds_world.csv"
        if args.csv_path.exists():
            print("[Error]: CSV file with same name already exists. Exiting...")
            exit()
        create_csv(
            args.csv_path,
            ["grasp_id", "scene_id", "qx", "qy", "qz", "qw", "x", "y", "z", "width", "label"],
        )

    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time
    g_num_total_jobs = len(df)
    g_completed_jobs = []
    g_starting_time = time.time()
    print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))

    if args.debug:
        assert args.num_proc == 1, "Debug mode only allowed with num_proc=1"
        # Test only one scene
        indices = [np.random.randint(len(df))]
        results = Parallel(n_jobs=args.num_proc)(delayed(generate_from_existing_grasps)(df.loc[index], args) for index in indices)
    else:
        results = Parallel(n_jobs=args.num_proc)(delayed(generate_from_existing_grasps)(df.loc[index], args) for index in range(len(df)))
    
    for result in results:
        g_completed_jobs.append(result)
        elapsed_time = time.time() - g_starting_time
        if len(g_completed_jobs) % 1000 == 0:
            msg = "%05d/%05d finished! " % (len(g_completed_jobs), g_num_total_jobs)
            msg = msg + 'Elapsed time: ' + \
                    time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
            print(msg)
