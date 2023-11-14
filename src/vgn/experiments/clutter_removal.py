import collections
import os
import argparse
from datetime import datetime
import uuid

import numpy as np
import pandas as pd
import tqdm

from vgn import io#, vis
from vgn.grasp import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
from vgn.assign_grasp_affordance import affrdnce_label_dict, aff_labels_to_names, viz_color_legend, get_affordance


MAX_CONSECUTIVE_FAILURES = 2
MAX_SKIPS = 3

State = collections.namedtuple("State", ["tsdf", "pc"])


def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    aff_thresh=0.5,
    sim_gui=False,
    result_path=None,
    add_noise=False,
    randomize_view=False,
    tight_view=False,
    sideview=False,
    resolution=64,
    silence=False,
    visualize=False
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    #sideview=False
    #n = 6
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, add_noise=add_noise, sideview=sideview)
    logger = Logger(logdir, description)
    cnt = 0
    success = 0
    aff_tp = np.zeros(len(affrdnce_label_dict.keys())) # true positive
    aff_fp = aff_tp.copy() # false positive
    aff_tn = aff_tp.copy() # true negative
    aff_fn = aff_tp.copy() # false negative
    left_objs = 0
    total_objs = 0
    cons_fail = 0
    no_grasp = 0
    planning_times = []
    total_times = []

    for _ in tqdm.tqdm(range(num_rounds), disable=silence):
        sim.reset(num_objects)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)
        total_objs += sim.num_objects
        consecutive_failures = 1
        skip_time = 0 # number of times we skip a round because of no grasp or no good quality grasp
        last_label = None
        trial_id = -1

        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES and skip_time<MAX_SKIPS:
            trial_id += 1
            timings = {}

            # scan the scene
            # see_table = False
            # if see_table == False:
            #     sim.world.remove_body(sim.world.bodies[0]) # remove table because we dont want to render it # normally table is the first body
            tsdf, pc, timings["integration"] = sim.acquire_tsdf(n=n, N=N, resolution=resolution, randomize_view=randomize_view, tight_view=tight_view)
            state = argparse.Namespace(tsdf=tsdf, pc=pc)#, pc_extended=pc_extended)
            # if resolution != 40:
            #     extra_tsdf, _, _ = sim.acquire_tsdf(n=n, N=N, resolution=resolution)
            #     state.tsdf_process = extra_tsdf
            # if see_table == False:
            #     sim.place_table(height=sim.gripper.finger_depth) # Add table back

            if pc.is_empty():
                print("[WARNING] Empty point cloud, aborting this round")
                total_objs -= sim.num_objects
                break  # empty point cloud, abort this round TODO this should not happen

            # plan grasps
            if visualize:
                mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
                scene_mesh = get_scene_from_mesh_pose_list(mesh_pose_list)
                grasps, scores, affs, timings["planning"], visual_mesh = grasp_plan_fn(state, scene_mesh)
                logger.log_mesh(scene_mesh, visual_mesh, f'round_{round_id:03d}_trial_{trial_id:03d}')
            else:
                grasps, scores, affs, timings["planning"] = grasp_plan_fn(state)
            planning_times.append(timings["planning"])
            total_times.append(timings["planning"] + timings["integration"])

            if len(grasps) == 0:
                # not enough candidate grasps were sampled OR no grasps above quality threshold were found
                no_grasp += 1
                skip_time += 1
                print("No good grasps, skipping...")
                continue  # no good grasps found, skip

            # Check affordances:
            # TODO: Get metrics for the affordances
            # get assignment of grasps to affordances
            affrdnce_labels = []
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, object_set)
            for grasp in grasps:
                affrdnce_labels.append(get_affordance(sim, sim.aff_dataset, grasp, mesh_pose_list)) # affdnces are 0/1 labels for each aff class
            affrdnce_labels = np.array(affrdnce_labels)
            # get true positive, false positive, true negative, false negative
            aff_tp += np.sum(affrdnce_labels * (affs > aff_thresh), axis=0)
            aff_fp += np.sum(affrdnce_labels * (affs <= aff_thresh), axis=0)
            aff_tn += np.sum((1 - affrdnce_labels) * (affs <= aff_thresh), axis=0)
            aff_fn += np.sum((1 - affrdnce_labels) * (affs > aff_thresh), axis=0)
            # Print the affordances of the best grasp
            aff_vector = affs[0]
            aff_values = np.where(aff_vector > aff_thresh)[0] # get indices where aff_vector > thresh
            best_grasp_affs_text = aff_labels_to_names(aff_values)
            print("[Grasp Affordances: ", best_grasp_affs_text)

            # execute grasp
            grasp, score = grasps[0], scores[0]
            print("[BEST Score: %.2f]" % score)
            label, _ = sim.execute_grasp(grasp, allow_contact=True)
            print("[RESULT: %s]" % label)
            cnt += 1
            if label != Label.FAILURE:
                success += 1

            # log the grasp
            logger.log_grasp(round_id, state, timings, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                cons_fail += 1
            last_label = label
        left_objs += sim.num_objects
    success_rate = 100.0 * success / cnt
    affordance_accuracy = 100.0 * (aff_tp + aff_tn) / (aff_tp + aff_fp + aff_tn + aff_fn)
    affordance_precision = 100.0 * aff_tp / (aff_tp + aff_fp)
    affordance_recall = 100.0 * aff_tp / (aff_tp + aff_fn)
    affordance_mean_precision = np.mean(affordance_precision)
    declutter_rate = 100.0 * success / total_objs
    print('Grasp success rate: %.2f %%, Declutter rate: %.2f %%' % (success_rate, declutter_rate))
    print('Affordance accuracy: ', affordance_accuracy)
    print('Affordance precision: ', affordance_precision)
    print('Affordance recall: ', affordance_recall)
    print('Affordance mean precision: ', affordance_mean_precision)
    print(f'Average planning time: {np.mean(planning_times)}, total time: {np.mean(total_times)}')
    print('Consecutive failures and no detections: %d, %d' % (cons_fail, no_grasp))
    if result_path is not None:
        with open(result_path, 'w') as f:
            f.write('%.2f%%, %.2f%%; %d, %d\n' % (success_rate, declutter_rate, cons_fail, no_grasp))
    return success_rate, declutter_rate, affordance_accuracy, affordance_precision, affordance_recall, affordance_mean_precision
    


class Logger(object):
    def __init__(self, root, description):
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        description = "{}_{}".format(time_stamp, description).strip("_")

        self.logdir = root / description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.mesh_dir = self.logdir / "meshes"
        self.mesh_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_mesh(self, scene_mesh, aff_mesh, name):
        scene_mesh.export(self.mesh_dir / (name + "_scene.obj"))
        aff_mesh.export(self.mesh_dir / (name + "_aff.obj"))

    def log_grasp(self, round_id, state, timings, grasp, score, label):
        # log scene
        tsdf, points = state.tsdf, np.asarray(state.pc.points)
        scene_id = uuid.uuid4().hex
        scene_path = self.scenes_dir / (scene_id + ".npz")
        np.savez_compressed(scene_path, grid=tsdf.get_grid(), points=points)

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
