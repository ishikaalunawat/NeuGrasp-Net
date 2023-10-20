import collections
import argparse
from datetime import datetime
import os


import numpy as np
import trimesh
from vgn.utils import visual

from vgn import io#, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list

MAX_CONSECUTIVE_FAILURES = 2


State = collections.namedtuple("State", ["tsdf", "pc"])

def visualize_grasps(scene_mesh, grasps, pc_full):
    # Debug: Viz grasps
    seen_grasps_scene = trimesh.Scene()
    # scene_pc = trimesh.points.PointCloud(pc_full, colors=np.tile(np.array([0, 0, 0, 1]), (len(pc_full), 1)))
    seen_grasp_mesh_list = [visual.grasp2mesh(g) for g in grasps]
    for i, g_mesh in enumerate(seen_grasp_mesh_list):
        seen_grasps_scene.add_geometry(g_mesh, node_name=f'grasp_{i}')
    composed_scene = trimesh.Scene([scene_mesh, seen_grasps_scene])#, unseen_grasps_scene])
    # composed_scene.add_geometry(trimesh.creation.axis())
    return composed_scene

def run( grasps_info, pc, mesh_pose_list, save_dir, data_root, scene, object_set, num_objects=5, n=6, N=None, seed=1, sim_gui=False, add_noise=False,
        sideview=False, resolution=40, silence=False, save_freq=8, visualize = True):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    os.makedirs(save_dir, exist_ok=True)
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview, save_dir=save_dir)
    cnt = 0
    success = 0
    left_objs = 0
    total_objs = 0
    cons_fail = 0
    no_grasp = 0

    sim.reset(0)
    sim.setup_sim_scene_from_mesh_pose_list(mesh_pose_list, table=True)
    sim.save_state()

    total_objs += sim.num_objects
    consecutive_failures = 1
    last_label = None
    trial_id = -1

    while consecutive_failures < MAX_CONSECUTIVE_FAILURES:
        trial_id += 1
        timings = {}
        grasps = grasps_info
        # scan the scene
        # tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N, resolution=40)
        # state = argparse.Namespace(tsdf=tsdf, pc=pc)
        # if resolution != 40:
        #     extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
        #     state.tsdf_process = extra_tsdf

        if len(pc)==0:
            break  # empty point cloud, abort this round TODO this should not happen

        ## TODO: Figure out visual_mesh
        if visualize:
            scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list, data_root=data_root)
            visual_mesh = visualize_grasps(scene_mesh, grasps, pc)
            # grasps, scores, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh)
        else:
            visual_mesh = None
            # grasps, scores, timings["planning"] = grasp_plan_fn(state)

        if len(grasps) == 0:
            no_grasp += 1
            break  # no detections found, abort this round

        # execute grasp
        # grasp, score = grasps[0], scores[0]
        grasp = grasps[0] # Same as GIGA (best)
        for grasp in grasps:
            label, _ = sim.execute_grasp(grasp, allow_contact=True, remove=False)
            sim.restore_state()
            cnt += 1
            if label != Label.FAILURE:
                success += 1

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                cons_fail += 1
            last_label = label
    # left_objs += sim.num_objects

    return (success, cnt, total_objs), visual_mesh