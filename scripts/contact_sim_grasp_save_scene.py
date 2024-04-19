import json
import argparse
import numpy as np
from pathlib import Path
from vgn.utils.transform import Transform, Rotation
from vgn.grasp import Grasp
from vgn.experiments import contact_clutter_removal_single
from vgn.utils.implicit import get_scene_from_mesh_pose_list

T_try = np.array(
    [[0, -1, 0, 0],
     [0, 0, 1, 0],
     [-1, 0, 0, 0],
     [0, 0, 0, 1]]
)
gripper_width = 0.08

def load_data(scene_id, result_path, pc_path, data_root, raw_root):
    pc_full = np.load(pc_path + scene_id + '.npz', allow_pickle=True)['pc'] # OURS
    results = np.load(result_path + f'predictions_{scene_id}.npz', allow_pickle=True)
    pred_grasps_cam, scores = results['pred_grasps_cam'].item()[-1], results['scores'].item()[-1]
    mesh_pose_list = np.load(raw_root + scene_id + '.npz', allow_pickle=True)['pc']
    # scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, data_root=data_root)
    return (pred_grasps_cam, scores), pc_full, mesh_pose_list

def pre_process_grasp(grasps_info):
    grasps_info = grasps_info[0]

    grasps = []
    for i in range(len(grasps_info)):
        sample_grasp = grasps_info[i]
        sample_grasp = np.dot(sample_grasp, T_try)

        grasp_rot, grasp_pos = sample_grasp[:3, :3], sample_grasp[:3, 3]
        grasp_rot = Rotation.from_matrix(grasp_rot)
        grasp = Grasp(Transform(grasp_rot, grasp_pos), gripper_width) # make grasp with width 0.08 (max gripper width)?

        grasp_frame_rot =  grasp.pose.rotation * Rotation.from_euler('Y', -0.5*np.pi) * Rotation.from_euler('Z', np.pi/2)
        grasp = Grasp(Transform(grasp_frame_rot, grasp_pos), gripper_width) # make grasp
        # post multiply with a translation in z axis
        new_matrix = grasp.pose.as_matrix() @ np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 1, 0.05],
                                                        [0, 0, 0, 1]])
        grasp = Grasp(Transform.from_matrix(new_matrix), gripper_width)
        grasps.append(grasp)

    return grasps

def main(args):
    #wandb.init(config=args, project="6dgrasp", entity="irosa-ias")
    grasps_info, pc, mesh_pose_list = load_data(args.scene_id, args.result_path, args.pc_path, args.data_root, args.raw_root)
    grasps_info = pre_process_grasp(grasps_info)

    results = {}
    for n in range(args.num_rounds):
        args.seed = np.random.randint(3000)
        save_dir = args.save_dir / f'round_{n:03d}'
        results[n], visual_mesh = contact_clutter_removal_single.run(grasps_info=grasps_info,
                                                                    pc=pc,
                                                                    mesh_pose_list=mesh_pose_list,
                                                                    save_dir=save_dir,
                                                                    data_root = args.data_root,
                                                                    scene=args.scene,
                                                                    object_set=args.object_set,
                                                                    num_objects=args.num_objects,
                                                                    n=args.num_view,
                                                                    seed=args.seed,
                                                                    sim_gui=args.sim_gui,
                                                                    add_noise=args.add_noise,
                                                                    sideview=args.sideview)
        print(f'Round {n} finished, result: {results[n]}')
        if args.viz:
            visual_mesh.export(args.save_dir / 'scene_'.format(n), 'obj')
        #wandb.log({'Grasps' : wandb.Object3D(open(args.save_dir + '/scene_{}.obj' % n))})

    with open(args.save_dir / 'results_contact_grasp_net.json', 'w') as f:
        json.dump(results, f, indent=2)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", type=str, required=True)
    parser.add_argument("--result-path", type=str, default='/home/hypatia/6D-DAAD/contact_graspnet/results/')
    parser.add_argument("--pc-path", type=str, 
                        default='/home/hypatia/6D-DAAD/GIGA/data/pile/data_pile_train_constructed_FULL/point_clouds/')
    parser.add_argument("--data-root", type=str, 
                        default="/home/hypatia/6D-DAAD/GIGA/")
    parser.add_argument("--raw-root", type=str, 
                        default='/home/hypatia/6D-DAAD/GIGA/data/pile/data_pile_train_random_raw_FULL/mesh_pose_list/')
    parser.add_argument("--save-dir", type=Path, default=Path('results_contact_grasp_net'))
    parser.add_argument("--scene",type=str,choices=["pile", "packed"],default="pile")
    parser.add_argument("--object-set", type=str, default="pile")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num-view", type=int, default=1)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-gui", action="store_true")
    # parser.add_argument("--grad-refine", action="store_true")
    parser.add_argument("--qual-th", type=float, default=0.9)
    parser.add_argument("--best",action="store_true",help="Whether to use best valid grasp (or random valid grasp)")
    parser.add_argument("--force",action="store_true",help="When all grasps are under threshold, force the detector to select the best grasp")
    parser.add_argument("--add-noise",type=str,default='',help="Whether add noise to depth observation, trans | dex | norm | ''")
    parser.add_argument("--sideview",action="store_true",help="Whether to look from one side")
    parser.add_argument("--simple-constrain",action="store_true",help="Whether to contrain grasp from backward")
    parser.add_argument("--res", type=int, default=40)
    parser.add_argument("--out-th", type=float, default=0.5)
    parser.add_argument("--silence",action="store_true",help="Whether to disable tqdm bar")
    parser.add_argument("--select-top",action="store_true",help="Use top heuristic")
    parser.add_argument("--viz",action="store_true",help="visualize and save affordance")

    args = parser.parse_args()
    main(args)
