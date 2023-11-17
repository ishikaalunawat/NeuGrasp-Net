import numpy as np
import os
import argparse
# from contact_graspnet.inference import inference
import config_utils as config_utils
import os
import sys
import argparse
import numpy as np
import glob
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.reset_default_graph()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

from data import regularize_pc_point_count, depth2pc, load_available_input_data
from contact_grasp_estimator import GraspEstimator
# from visualization_utils import visualize_grasps, show_image

def mask(arr2, arr0, arr1):
    arr0 = np.ma.compressed(np.ma.masked_where(arr2>0.8,arr0))
    arr1 = np.ma.compressed(np.ma.masked_where(arr2>0.8,arr1))
    arr2 = np.ma.compressed(np.ma.masked_where(arr2>0.8,arr2))

    return arr0, arr2, arr1

def inference(global_config, checkpoint_dir, input_paths, K=None, local_regions=False, skip_border_objects=False, filter_grasps=False, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    # Process example test scenes
    print(input_paths)
    for p in glob.glob(input_paths):
        tf.reset_default_graph()

        pc_segments = {}
        segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        
        if segmap is None and (local_regions or filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=z_range)

        #Flipping Z-axis of PC:
        # pc_full[:, 2] *= -1
        pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full, pc_segments=pc_segments, 
                                                                                          local_regions=local_regions, filter_grasps=filter_grasps, forward_passes=forward_passes)  

        # Taking best 20
        pred_grasps_cam, scores, contact_pts = pred_grasps_cam[-1], scores[-1], contact_pts[-1]
        scores_sort = scores.argsort()[::-1]
        pred_grasps_cam = pred_grasps_cam[scores_sort, :, :]
        contact_pts = contact_pts[scores_sort, :]
        scores = scores[scores_sort]
        pred_grasps_cam, scores, contact_pts = {-1 : pred_grasps_cam[:20, :, :]}, {-1 : scores[:20]}, {-1 : contact_pts[:20, :]}

        # Save results
        np.savez('results/predictions_{}'.format(os.path.basename(p.replace('png','npz').replace('npy','npz'))), 
                  pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        # Visualize results          
        print(f"Processed scene {p}")
        # show_image(rgb, segmap)
        # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        
    if not glob.glob(input_paths):
        print('No files found: ', input_paths)

def run_experiments(args):
    if not args.use_depth_imgs:
        pc_root = args.root + 'point_clouds/'
    
        if args.random and args.num_scenes:
            scene_ids = np.random.choice(os.listdir(pc_root), args.num_scenes)
        elif args.scene_ids:
            scene_ids = [scene_id + '.npz' for scene_id in args.scene_ids]
            # scene_ids = eval(str(args.scene_ids))

        scenes = [pc_root + scene_id for scene_id in scene_ids]
        global_config = config_utils.load_config(args.ckpt_dir, batch_size=args.forward_passes, arg_configs=args.arg_configs)

    else:
        depth_root = args.root + '/scenes/'
        if args.random and args.num_scenes:
            scene_ids = np.random.choice(os.listdir(depth_root), args.num_scenes)
        elif args.scene_ids:
            scene_ids = [scene_id + '.npz' for scene_id in args.scene_ids]
            # scene_ids = eval(str(args.scene_ids))

        scenes = [depth_root + scene_id for scene_id in scene_ids]
        global_config = config_utils.load_config(args.ckpt_dir, batch_size=args.forward_passes, arg_configs=args.arg_configs)
    
    for scene in scenes:
        print(f"\n\nProcessing scene: {scene.strip()}\n\n")
        inference(global_config, args.ckpt_dir, scene, z_range=eval(str(args.z_range)),forward_passes=args.forward_passes)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/contact_graspnet_train_and_test', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--root', default='/home/hypatia/6D-DAAD/GIGA/data/packed/data_packed_train_random_raw_4M_GPG_60_packked', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--random', action="store_true", help='If random, enter num_scenes, else enter scene_ids')
    parser.add_argument('--num_scenes', type=int, default=1, help='Number of scenes to test only if random is provdied')
    # parser.add_argument('--scene_ids', default=[], help='Scene ids to test only if random is not provided')
    parser.add_argument('--scene_ids', default=[], nargs='+')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--forward_passes', type=int, default=5,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    parser.add_argument('--use_depth_imgs', action="store_true", help='use depth images')
    
    args = parser.parse_args()
    run_experiments(args)
