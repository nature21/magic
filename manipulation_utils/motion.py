from typing import Optional, List

import numpy as np
from mplib import pymp
from sapien.core import Pose

from env.tableScene import TableScene


def gen_qpos_to_qpos_trajectory(
        table_scene: TableScene,
        start_qpos,
        end_qpos: np.ndarray,
        exclude_ids: Optional[List[int]] = None,
        planning_time: float = 1.0,
        pcd_resolution: float = 1e-3,
        ignore_env: bool = False,
        verbose: bool = False
):
    if ignore_env:
        table_scene.planner.remove_point_cloud()
    else:
        table_scene.update_env_pcd(exclude_ids=exclude_ids, pcd_resolution=pcd_resolution, verbose=verbose)

    result = table_scene.planner.plan_qpos(
        [end_qpos],
        start_qpos,
        time_step=1 / table_scene.fps,
        planning_time=planning_time,
        verbose=verbose,
    )
    return result


def gen_qpos_to_pose_trajectory(
        table_scene: TableScene,
        start_qpos,
        end_pose: pymp.Pose,
        exclude_ids: Optional[List[int]] = None,
        planning_time: float = 1.0,
        pcd_resolution: float = 1e-3,
        ignore_env: bool = False,
        verbose: bool = False,
        use_screw: bool = False
):
    if ignore_env:
        table_scene.planner.remove_point_cloud()
    else:
        table_scene.update_env_pcd(exclude_ids=exclude_ids, pcd_resolution=pcd_resolution)
    if not use_screw:
        result = table_scene.planner.plan_pose(
            end_pose,
            start_qpos,
            mask=np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
            time_step=1 / table_scene.fps,
            planning_time=planning_time,
            verbose=verbose
        )
    else:
        result = table_scene.planner.plan_screw(
            end_pose,
            start_qpos,
            time_step=1 / table_scene.fps,
            verbose=verbose
        )
    return result


def trajectory_warp(
        reference_pose_list, reference_init_pose,
        target_init_pose, target_align_init_pose,
        reference_grasp_center=None, reference_pixel_to_3d_fn=None,
        target_grasp_center=None, target_pixel_to_3d_fn=None,
):
    target_pose_list = [reference_pose * reference_init_pose.inv() * target_align_init_pose for reference_pose
                        in reference_pose_list]

    if target_grasp_center is None:
        assert reference_grasp_center is not None, "Need to provide reference_grasp_center"
        assert reference_pixel_to_3d_fn is not None, "Need to provide reference_pixel_to_3d_fn"
        reference_init_grasp_center_world = reference_pixel_to_3d_fn(reference_grasp_center)
        target_init_grasp_center_world = (
                target_init_pose * target_align_init_pose.inv() * Pose(reference_init_grasp_center_world)).p

    else:
        assert target_pixel_to_3d_fn is not None, "Need to provide target_pixel_to_3d_fn"
        target_init_grasp_center_world = target_pixel_to_3d_fn(target_grasp_center)

    target_p_array = np.array([pose.p for pose in target_pose_list])
    target_q_array = np.array([pose.q for pose in target_pose_list])

    return {
        'target_p_array': target_p_array,
        'target_q_array': target_q_array,
        'init_grasp_center_world': target_init_grasp_center_world
    }
