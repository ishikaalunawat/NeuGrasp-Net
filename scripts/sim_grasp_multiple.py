import argparse
import numpy as np
import json
from pathlib import Path

from vgn.detection import VGN
from vgn.detection_implicit import VGNImplicit
from vgn.experiments import clutter_removal
from vgn.utils.misc import set_random_seed


def main(args):

    if args.type in ['giga', 'giga_hr', 'giga_hr_deeper', 'giga_hr_affnet', 'giga_aff', 'neu_grasp_pn', 'neu_grasp_pn_deeper', 'neu_grasp_pn_deeper4', 'pointnetgpd',
                     'neu_grasp_pn_no_local_cloud', 'neu_grasp_dgcnn', 'neu_grasp_dgcnn_deeper', 'neu_grasp_vn_pn_pn_deeper',
                     'neu_grasp_pn_affnet', 'neu_grasp_pn_affnet_sem']:
        grasp_planner = VGNImplicit(args.model,
                                    args.type,
                                    best=args.best,
                                    qual_th=args.qual_th,
                                    aff_thresh=args.aff_thresh,
                                    force_detection=args.force,
                                    seen_pc_only=args.seen_pc_only,
                                    out_th=0.1,
                                    select_top=False,
                                    resolution=args.resolution,
                                    visualize=args.vis)
    elif args.type == 'vgn':
        grasp_planner = VGN(args.model,
                            args.type,
                            best=args.best,
                            qual_th=args.qual_th,
                            force_detection=args.force,
                            out_th=0.1,
                            visualize=args.vis)
    else:
        raise NotImplementedError(f'model type {args.type} not implemented!')

    gsr = []
    dr = []
    aff_accuracy = []
    aff_precision = []
    aff_recall = []
    aff_mean_precision = []
    # seen_gsr = []
    # unseen_gsr = []
    # seen_cnts = []
    # unseen_cnts = []
    for seed in args.zeeds:
        set_random_seed(seed)
        success_rate, declutter_rate, affordance_accuracy, affordance_precision, affordance_recall, affordance_mean_precision = clutter_removal.run(
            grasp_plan_fn=grasp_planner,
            logdir=args.logdir,
            resolution=args.resolution,
            description=args.description,
            scene=args.scene,
            object_set=args.object_set,
            num_objects=args.num_objects,
            n=args.num_view,
            num_rounds=args.num_rounds,
            seed=seed,
            aff_thresh=args.aff_thresh,
            sim_gui=args.sim_gui,
            result_path=args.result_path,
            add_noise=args.add_noise,
            randomize_view=args.randomize_view,
            tight_view=args.tight_view,
            see_table=args.see_table,
            sideview=args.sideview,
            silence=args.silence,
            visualize=args.vis,
            save_dir=args.save_dir,
            use_nvisii=args.use_nvisii)
        gsr.append(success_rate)
        # seen_gsr.append(seen_success_rate)
        # unseen_gsr.append(unseen_success_rate)
        # seen_cnts.append(seen_cnt)
        # unseen_cnts.append(unseen_cnt)
        dr.append(declutter_rate)
        aff_accuracy.append(affordance_accuracy.tolist())
        aff_precision.append(affordance_precision.tolist())
        aff_recall.append(affordance_recall.tolist())
        aff_mean_precision.append(affordance_mean_precision)
    results = {
        'gsr': {
            'mean': np.mean(gsr),
            'std': np.std(gsr),
            'val': gsr
        },
        'dr': {
            'mean': np.mean(dr),
            'std': np.std(dr),
            'val': dr
        },
        'aff_accuracy': {
            'mean': np.mean(aff_accuracy, axis=0).tolist(),
            'std': np.std(aff_accuracy, axis=0).tolist(),
            'val': aff_accuracy
        },
        'aff_precision': {
            'mean': np.mean(aff_precision, axis=0).tolist(),
            'std': np.std(aff_precision, axis=0).tolist(),
            'val': aff_precision
        },
        'aff_recall': {
            'mean': np.mean(aff_recall, axis=0).tolist(),
            'std': np.std(aff_recall, axis=0).tolist(),
            'val': aff_recall
        },
        'aff_mean_precision': {
            'mean': np.mean(aff_mean_precision),
            'std': np.std(aff_mean_precision),
            'val': aff_mean_precision
        },
    }
    print('Average results:')
    print(f'Grasp sucess rate: {np.mean(gsr):.2f} ± {np.std(gsr):.2f} %')
    # print(f'Seen grasp sucess rate: {np.mean(seen_gsr):.2f} ± {np.std(seen_gsr):.2f} %')
    # print(f'Unseen grasp sucess rate: {np.mean(unseen_gsr):.2f} ± {np.std(unseen_gsr):.2f} %')
    # print(f'Seen grasp count: {np.mean(seen_cnts):.2f} ± {np.std(seen_cnts):.2f}')
    # print(f'Unseen grasp count: {np.mean(unseen_cnts):.2f} ± {np.std(unseen_cnts):.2f}')
    print(f'Declutter rate: {np.mean(dr):.2f} ± {np.std(dr):.2f} %')
    print(f'Affordance accuracy: {np.mean(aff_accuracy, axis=0).tolist()} ± {np.std(aff_accuracy, axis=0).tolist()}')
    print(f'Affordance precision: {np.mean(aff_precision, axis=0).tolist()} ± {np.std(aff_precision, axis=0).tolist()}')
    print(f'Affordance recall: {np.mean(aff_recall, axis=0).tolist()} ± {np.std(aff_recall, axis=0).tolist()}')
    print(f'Affordance mean precision: {np.mean(aff_mean_precision):.2f} ± {np.std(aff_mean_precision):.2f}')
    with open(args.result_path+'.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--logdir", type=Path, default="data/experiments")
    parser.add_argument("--save-dir", type=Path, default=None, required=False)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--scene",
                        type=str,
                        choices=["pile", "packed", "egad", "affnet"],
                        default="pile")
    parser.add_argument("--object_set", type=str, default="pile/test")
    parser.add_argument("--num-objects", type=int, default=5)
    parser.add_argument("--num_view", type=int, default=1) # No need to change
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--zeeds", type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument("--sim-gui", action="store_true")
    # parser.add_argument("--grad-refine", action="store_true")
    parser.add_argument("--qual_th", type=float, default=0.5)
    parser.add_argument("--aff_thresh", type=float, default=0.5)
    parser.add_argument("--eval_geo",
                        action="store_true",
                        help='whether evaluate geometry prediction')
    parser.add_argument(
        "--best",
        action="store_true",
        help="UNUSED. Whether to use best valid grasp (or random valid grasp). UNUSED")
    parser.add_argument(
        "--randomize_view",
        type=bool, default='',
        help="Whether to use a random view input tsdf/point cloud")
    parser.add_argument(
        "--tight_view",
        type=bool, default='',
        help="Whether to use a TIGHT view input tsdf/point cloud. Very partial view")
    parser.add_argument(
        "--see_table",
        type=bool, default='',
        help="Whether the network sees the table in the input tsdf/point cloud")
    parser.add_argument(
        "--seen_pc_only",
        type=bool, default='',
        help="Whether to use the 'seen' point cloud or the reconstructed point cloud for grasp candidate generation")
    parser.add_argument("--result_path", type=str)
    parser.add_argument(
        "--force",
        action="store_true",
        help=
        "When all grasps are under threshold, force the detector to select the best grasp"
    )
    parser.add_argument(
        "--add_noise",
        type=str,
        default='',
        help="Whether add noise to depth observation, trans | dex | norm | ''")
    parser.add_argument("--sideview",
                        action="store_true",
                        help="Whether to look from one side")
    parser.add_argument("--silence",
                        action="store_true",
                        help="Whether to disable tqdm bar")
    parser.add_argument("--vis",
                        action="store_true",
                        help="visualize and save affordance")
    parser.add_argument("--use_nvisii",
                        action="store_true",
                        help="visualize in nvisii renderer")
    parser.add_argument("--save-fails",
                        action="store_true",
                        help="Save grasp failure visualizations")
    
    args, _ = parser.parse_known_args()
    main(args)
