import os
import glob
import time
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from pathlib import Path

from joblib import Parallel, delayed

from vgn.utils.implicit import get_scene_from_mesh_pose_list, sample_iou_points

def sample_occ(mesh_pose_list, num_point, uniform, data_root=''):
    scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True, data_root=data_root)
    points, occ = sample_iou_points(mesh_list, scene.bounds, num_point, uniform=uniform)
    return points, occ

def sample_occ_and_sem_class(aff_dataset, mesh_pose_list, num_point, uniform, data_root=''):
    scene, mesh_list = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=True, data_root=data_root)
    points, occ, sem = sample_iou_points(mesh_list, scene.bounds, num_point, uniform=uniform, aff_dataset=aff_dataset, mesh_pose_list=mesh_pose_list)
    return points, occ, sem

def save_occ(mesh_pose_list_path, args):
    # points, occ, sem = get_occ_and_sem_class_specific_points(args.aff_dataset, mesh_pose_list, mesh_list, grasp_pc)
    if args.save_occ_semantics:
        # load affnet dataset
        affnet_root = Path(args.data_root) / 'data/3DGraspAff/'
        aff_path = os.path.join(affnet_root, 'filt_scaled_anntd_remapped_full_shape_'+args.object_set+'_data.pkl')
        with open(aff_path, 'rb') as f:
            aff_dataset = pkl.load(f)
        
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        if len(mesh_pose_list) == 0:
            print('No meshes in %s' % mesh_pose_list_path) 
            return
        points, occ, sem = sample_occ_and_sem_class(aff_dataset, mesh_pose_list, args.num_point_per_file * args.num_file, args.uniform, data_root=args.data_root)
    else:
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        if len(mesh_pose_list) == 0:
            print('No meshes in %s' % mesh_pose_list_path) 
            return
        points, occ = sample_occ(mesh_pose_list, args.num_point_per_file * args.num_file, args.uniform, data_root=args.data_root)
    points = points.astype(np.float16).reshape(args.num_file, args.num_point_per_file, 3)
    occ = occ.reshape(args.num_file, args.num_point_per_file)
    name = os.path.basename(mesh_pose_list_path)[:-4]
    save_root = os.path.join(args.raw, 'occ', name)
    os.makedirs(save_root)
    if args.save_occ_semantics:
        sem = sem.reshape(args.num_file, args.num_point_per_file)
        for i in range(args.num_file):
            np.savez(os.path.join(save_root, '%04d.npz' % i), points=points[i], occ=occ[i], sem=sem[i])
    else:
        for i in range(args.num_file):
            np.savez(os.path.join(save_root, '%04d.npz' % i), points=points[i], occ=occ[i])

def log_result(result):
    g_completed_jobs.append(result)
    elapsed_time = time.time() - g_starting_time

    if len(g_completed_jobs) % 1000 == 0:
        msg = "%05d/%05d %s finished! " % (len(g_completed_jobs), g_num_total_jobs, result)
        msg = msg + 'Elapsed time: ' + \
                time.strftime("%H:%M:%S", time.gmtime(elapsed_time)) + '. '
        print(msg)

def main(args):
    mesh_list_files = glob.glob(os.path.join(args.raw, 'mesh_pose_list', '*.npz'))

    global g_completed_jobs
    global g_num_total_jobs
    global g_starting_time

    g_num_total_jobs = len(mesh_list_files)
    g_completed_jobs = []

    g_starting_time = time.time()

    if args.num_proc > 1:
        print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        
        results = Parallel(n_jobs=args.num_proc)(delayed(save_occ)(f, args) for f in tqdm(mesh_list_files))
        
        for result in results:
            log_result(result)
        # pool = mp.Pool(processes=args.num_proc) 
        # print('Total jobs: %d, CPU num: %d' % (g_num_total_jobs, args.num_proc))
        # for f in mesh_list_files:
        #     pool.apply_async(func=save_occ, args=(f,args), callback=log_result)
        # pool.close()
        # pool.join()
    else:
        for f in mesh_list_files:
            save_occ(f, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("raw", type=str)
    parser.add_argument("--data_root", type=Path, default="", help="Root directory for the dataset obj files")
    parser.add_argument("--object_set", type=str, default="train")
    parser.add_argument("num_point_per_file", type=int)
    parser.add_argument("num_file", type=int)
    parser.add_argument("--uniform", action='store_true', help='sample uniformly in the bbox, else sample in the tight bbox')
    parser.add_argument("--save_occ_semantics", type=bool, default='', help="Also save the occupancy semantics")
    args = parser.parse_args()
    main(args)