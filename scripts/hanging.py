import argparse
import os
import time
import numpy as np
from PIL import Image
from mplib import pymp
from sapien.core import Pose, CameraEntity

from magic.match import match
from magic.utils_2d import find_nearest_point_with_mask
from manipulation_utils.contact_samplers import sample_grasp
from manipulation_utils.motion import gen_qpos_to_qpos_trajectory, gen_qpos_to_pose_trajectory
from utils.mesh_utils import get_normal_of_nearest_point
from env.envs.hookScene import HookScene
from env.envs.mugScene import MugScene, SKIP_MUGS_AND_REASONS
from utils.camera_utils import uvz2world, imgs2mp4
from utils.rotation_utils import normalize_vector, rotate_vector, wxyz2xyzw, cross, get_quaternion_from_matrix, \
    quat_mul, xyzw2wxyz, quat_conjugate, rpy2wxyz
from utils.ui_utils import get_click_coordinates_from_array


def get_reference(manual_center=False) -> tuple[Image.Image, np.ndarray, CameraEntity, tuple[int, int], HookScene]:
    """Get the reference image and the center of the reference image."""
    reference_scene = HookScene(
        'hanger',
        'none',
        target_half_size=0.05,
        target_x_y=np.array([0.25, 0]),
        add_robot=False
    )
    reference_scene.hide_env_visual()
    reference_img, reference_depth_img, reference_camera = reference_scene.get_picture(
        direction='+z',
        additional_translation=np.array([0.6, -0.6, 0.2 - 0.375 * 1.414]),
        debug_viewer=False,
        get_depth=True
    )
    if manual_center:
        reference_center = get_click_coordinates_from_array(np.asarray(reference_img))
        print('manually specified reference center:', reference_center)
    else:
        reference_center = (359, 161)
        print('default reference center:', reference_center)

    # we must return the reference_scene so that the scene is not garbage collected,
    # otherwise the camera will not be able to take picture or use to compute uvz2world
    return reference_img, reference_depth_img, reference_camera, reference_center, reference_scene


def transfer(
        alignment_result,
        reference_depth_img,
        reference_camera,
        target_depth_img,
        target_camera,
        target_current_pose,
        final_rotation_contact_2_wxyz,
        final_pos_contact_2,
        only_dino=False,
        target_init_pose=None,
        target_pcd=None,
) -> tuple[Pose, float, float]:
    """Transfer the alignment result to the final pose of the target object."""
    center_of_curvature_1 = alignment_result['center_of_curvature_1']
    center_of_curvature_2 = alignment_result['center_of_curvature_2']
    center_of_contact_1 = alignment_result['center_of_contact_1']
    center_of_contact_2 = alignment_result['center_of_contact_2'].astype(int)

    center_of_contact_1 = find_nearest_point_with_mask((reference_depth_img > 1e-6), center_of_contact_1, 5)
    center_of_contact_2 = find_nearest_point_with_mask((target_depth_img > 1e-6), center_of_contact_2, 5)

    depth_1 = reference_depth_img[center_of_contact_1[1] - 1, center_of_contact_1[0]]
    depth_2 = target_depth_img[center_of_contact_2[1] - 1, center_of_contact_2[0]]

    center_of_contact_1_world = uvz2world(reference_camera, np.array([*center_of_contact_1, depth_1]))
    center_of_contact_2_world = uvz2world(target_camera, np.array([*center_of_contact_2, depth_2]))

    z_axis_2 = rotate_vector([-1, 0, 0], wxyz2xyzw(target_camera.get_pose().q))
    if not only_dino:
        center_of_curvature_1_world = uvz2world(reference_camera, np.array([*center_of_curvature_1, depth_1]))
        center_of_curvature_2_world = uvz2world(target_camera, np.array([*center_of_curvature_2, depth_2]))

        radius_of_curvature_1_sign = 1 if alignment_result['radius_of_curvature_1'] > 0 else -1
        radius_of_curvature_2_sign = 1 if alignment_result['radius_of_curvature_2'] > 0 else -1
        radius_of_curvature_1_world = np.linalg.norm(
            center_of_curvature_1_world - center_of_contact_1_world) * radius_of_curvature_1_sign
        radius_of_curvature_2_world = np.linalg.norm(
            center_of_curvature_2_world - center_of_contact_2_world) * radius_of_curvature_2_sign
        print(radius_of_curvature_1_world, radius_of_curvature_2_world)
        x_axis_2 = normalize_vector(center_of_contact_2_world - center_of_curvature_2_world)
    else:
        radius_of_curvature_1_world, radius_of_curvature_2_world = None, None
        center_of_contact_2_object = (target_init_pose.inv() * Pose(center_of_contact_2_world)).p
        normal_2, center_of_contact_2_object = get_normal_of_nearest_point(target_pcd, center_of_contact_2_object,
                                                                           return_point=True)
        # o3d.visualization.draw_geometries([target_pcd], point_show_normal=True)
        normal_2 = -normal_2
        center_of_contact_2_world = (target_init_pose * Pose(center_of_contact_2_object)).p
        normal_2_world = rotate_vector(normal_2, wxyz2xyzw(target_init_pose.q))
        x_axis_2 = normalize_vector(normal_2_world - np.dot(normal_2_world, z_axis_2) * z_axis_2)

    y_axis_2 = cross(z_axis_2, x_axis_2)

    current_rotation_contact_2 = np.array([x_axis_2, y_axis_2, z_axis_2]).T

    pos_current_2 = target_current_pose.p
    q_current_2 = wxyz2xyzw(target_current_pose.q)

    q_current_rotation_contact_2 = get_quaternion_from_matrix(current_rotation_contact_2)
    q_final_rotation_contact_2 = wxyz2xyzw(final_rotation_contact_2_wxyz)
    q_relative_2 = quat_mul(q_final_rotation_contact_2, quat_conjugate(q_current_rotation_contact_2))
    q_final_2 = xyzw2wxyz(quat_mul(q_relative_2, q_current_2))

    pos_final_2 = final_pos_contact_2 + rotate_vector(pos_current_2 - center_of_contact_2_world, q_relative_2)

    return Pose(pos_final_2, q_final_2), radius_of_curvature_1_world, radius_of_curvature_2_world


def analogy(
        mug_id, top_k=3, read_from_result=False, only_compute_match=False, desc='exp', use_reflection=False
):
    """
    Perform analogy on the mug with the specified mug_id.
    """
    reference_img, reference_depth_img, reference_camera, reference_center, reference_scene = get_reference(
        manual_center=False)

    target_scene = MugScene(
        mug_id,
        add_robot=True,
        fps=480
    )

    target_scene.hide_env_visual()
    target_img, target_depth_img, target_camera = target_scene.get_picture(direction='+x', debug_viewer=False,
                                                                           get_depth=True)
    target_scene.unhide_env_visual()

    analogy_results = []

    if read_from_result:
        alignment_results = np.load(f'alignment_results/hanging_{desc}/{mug_id}/alignment_results.npy',
                                    allow_pickle=True)
    else:
        pca = True
        patch_size = 13
        alignment_results = match(reference_img, target_img, reference_center,
                                  parameter_save_dir=f'alignment_results/hanging_{desc}/{mug_id}',
                                  save_dir=f'results/hanging_{desc}/{mug_id}', top_k=top_k,
                                  pca=pca, patch_size=patch_size, use_reflection=use_reflection)

    if only_compute_match:
        return

    for alignment_result in alignment_results[:top_k]:
        init_pose = target_scene.mug.get_pose()
        final_rotation_contact_2_wxyz = rpy2wxyz([0, -np.pi / 2, 0])
        offset = 0.01
        final_pos_contact_2 = np.array([0.62, -0.2, 0.375 + offset])
        try:
            final_pose, _, _ = transfer(
                alignment_result,
                reference_depth_img, reference_camera,
                target_depth_img, target_camera,
                init_pose, final_rotation_contact_2_wxyz, final_pos_contact_2
            )
        except Exception as e:
            print(e)
            continue
        is_final_pose_feasible = target_scene.check_final_pose_feasibility(final_pose)
        if not is_final_pose_feasible:
            continue
        analogy_results.append(final_pose)

    return {
        'target_scene': target_scene,
        'analogy_results': analogy_results
    }


def plan(
        target_scene, final_pose, intermediate_pose=None,
        video_save_path=None, human_viewer=False, timeout=60,
        return_traj=False, surface_pointing_tol=0.9,
        use_pre_grasp_point=False
):
    target_scene.reset()
    if intermediate_pose is None:
        intermediate_pose = Pose(p=final_pose.p + np.array([-0.04, 0, 0]), q=final_pose.q)
    target_scene.open_gripper()
    start_time = time.time()
    grasp_generator = sample_grasp(
        target_scene,
        object_id=target_scene.mug.get_id(),
        verbose=False, start_time=start_time,
        timeout=timeout,
        surface_pointing_tol=surface_pointing_tol,
    )
    checkpoint = target_scene.scene.pack()
    result_1_to_2 = None
    result_2_to_3 = None
    result_3_to_4 = None
    final_result = None
    while True:
        current_time = time.time()
        if current_time - start_time > timeout:
            print('Time out, failed to find a feasible solution')
            if return_traj:
                return False, None
            else:
                return False
        try:
            grasp_parameter = next(grasp_generator)
        except StopIteration:
            print('Failed to find a feasible grasp')
            if return_traj:
                return False, None
            else:
                return False
        if use_pre_grasp_point:
            checkpoint_1 = target_scene.scene.pack()
            target_scene.robot.set_qpos(grasp_parameter.qpos)
            ee_pose_2 = target_scene.end_effector.get_pose()
            target_scene.scene.unpack(checkpoint_1)

            ee_pose_1_to_1_5 = Pose([0, 0, 0.1]) * ee_pose_2

            result_1_to_1_5 = gen_qpos_to_pose_trajectory(
                target_scene,
                target_scene.robot.get_qpos(),
                ee_pose_1_to_1_5,
                pcd_resolution=5e-3,
                # pcd_resolution=1e-6,
                # ignore_env=True,
            )

            if result_1_to_1_5['status'] != 'Success':
                print('Failed to generate pre-grasp trajectory')
                target_scene.scene.unpack(checkpoint)
                continue

            target_scene.follow_path(result_1_to_1_5)

            result_1_5_to_2 = gen_qpos_to_pose_trajectory(
                target_scene,
                target_scene.robot.get_qpos(),
                ee_pose_2,
                pcd_resolution=5e-3,
            )

            if result_1_5_to_2['status'] != 'Success':
                print('Failed to generate grasp trajectory')
                target_scene.scene.unpack(checkpoint)
                continue

            target_scene.follow_path(result_1_5_to_2)

            result_1_to_2 = {'position': np.concatenate([result_1_to_1_5['position'], result_1_5_to_2['position']]),
                             'velocity': np.concatenate([result_1_to_1_5['velocity'], result_1_5_to_2['velocity']])}
        else:
            result_1_to_2 = gen_qpos_to_qpos_trajectory(
                target_scene,
                target_scene.robot.get_qpos(),
                grasp_parameter.qpos,
                # pcd_resolution=1e-2,
                pcd_resolution=5e-3,
            )

            if result_1_to_2['status'] != 'Success':
                print('Failed to generate grasp trajectory')
                target_scene.scene.unpack(checkpoint)
                continue
            target_scene.follow_path(result_1_to_2)

        target_scene.attach_object(target_scene.mug)
        target_scene.close_gripper()
        target_scene.planner_attach_obj(target_scene.mug)

        # lift ee a little
        ee_pose_2 = target_scene.end_effector.get_pose()
        ee_pose_3 = Pose(ee_pose_2.p + np.array([0, 0, 0.1]), ee_pose_2.q)

        result_2_to_3 = gen_qpos_to_pose_trajectory(
            target_scene,
            grasp_parameter.qpos,
            pymp.Pose(ee_pose_3.p, ee_pose_3.q),
            exclude_ids=[target_scene.mug.get_id()],
            planning_time=5,
            pcd_resolution=1e-3,
            verbose=True
        )

        if result_2_to_3['status'] != 'Success':
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            print('Failed to generate lift trajectory')
            continue

        target_scene.follow_path(result_2_to_3)
        qpos_3 = target_scene.robot.get_qpos()
        # get intermediate ee pose
        mug_pose_3 = target_scene.mug.get_pose()
        ee_pose_4 = intermediate_pose * mug_pose_3.inv() * ee_pose_3

        # generate the trajectory from current qpos to target qpos with motion planning
        result_3_to_4 = gen_qpos_to_pose_trajectory(
            target_scene,
            qpos_3,
            ee_pose_4,
            exclude_ids=[target_scene.mug.get_id()],
            planning_time=5,
            pcd_resolution=5e-3
        )

        if result_3_to_4['status'] != 'Success':
            print('Failed to generate intermediate motion planning trajectory')
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            continue

        target_scene.follow_path(result_3_to_4)
        qpos_4 = target_scene.robot.get_qpos()
        final_ee_pose = final_pose * mug_pose_3.inv() * ee_pose_3

        final_result = gen_qpos_to_pose_trajectory(
            target_scene,
            qpos_4,
            pymp.Pose(final_ee_pose.p, final_ee_pose.q),
            exclude_ids=[target_scene.mug.get_id()],
            planning_time=5,
            pcd_resolution=1e-5
        )

        if final_result['status'] != 'Success':
            print('Failed to generate final motion planning trajectory')
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            continue

        target_scene.set_up_planner()
        target_scene.detach_object()
        target_scene.scene.unpack(checkpoint)
        break

    print('finished planning, now executing the plan and generating a video, this might take a while')

    if video_save_path is not None:
        viewer = None
        camera = target_scene.add_camera(direction='+x+z')
        camera_pose = camera.get_pose()
        camera.set_pose(Pose(p=camera_pose.p + np.array([0.5, 0, 0.5]), q=camera_pose.q))

        target_scene.open_gripper()
        _, imgs_1_to_2 = target_scene.follow_path(result_1_to_2, camera=camera, camera_interval=16)
        target_scene.attach_object(target_scene.mug)
        target_scene.close_gripper(0.005)
        _, imgs_2_to_3 = target_scene.follow_path(result_2_to_3, camera=camera, camera_interval=16)
        _, imgs_3_to_4 = target_scene.follow_path(result_3_to_4, camera=camera, camera_interval=16)
        _, imgs_5 = target_scene.follow_path(final_result, camera=camera, camera_interval=16)

        imgs = imgs_1_to_2 + imgs_2_to_3 + imgs_3_to_4 + imgs_5
        imgs2mp4(imgs, video_save_path, fps=120)
    else:
        viewer = target_scene.create_viewer() if human_viewer else None
        target_scene.open_gripper()
        target_scene.follow_path(result_1_to_2)
        target_scene.attach_object(target_scene.mug)
        target_scene.close_gripper()
        target_scene.follow_path(result_2_to_3)
        target_scene.follow_path(result_3_to_4)
        target_scene.follow_path(final_result)

    success = target_scene.is_success()
    if viewer is not None:
        viewer.close()
    target_scene.detach_object()
    target_scene.set_up_planner()
    target_scene.reset()

    if return_traj:
        return success, {
            'result_1_to_2': result_1_to_2,
            'result_2_to_3': result_2_to_3,
            'result_3_to_4': result_3_to_4,
            'final_result': final_result
        }
    return success


def plan_without_robot(target_scene, final_pose, human_viewer=False):
    target_scene.reset()
    viewer = target_scene.create_viewer() if human_viewer else None
    target_scene.mug.set_pose(final_pose)
    success = target_scene.is_success()
    if viewer is not None:
        viewer.close()
    return success


def main(mug_id, desc, with_robot=False, timeout=60, np_seed=0, save_video=False, use_e2=False):
    np.random.seed(np_seed)
    read_from_result = True if os.path.exists(f'alignment_results/hanging_{desc}/{mug_id}') else False
    if mug_id in SKIP_MUGS_AND_REASONS:
        print(f'Mug {mug_id} is not suitable for hanging: {SKIP_MUGS_AND_REASONS[mug_id]}')
        return
    print(f'Processing mug {mug_id}')
    analogy_results = analogy(mug_id, top_k=3, read_from_result=read_from_result, only_compute_match=False, desc=desc, use_reflection=use_e2)
    target_scene = analogy_results['target_scene']
    analogy_results = analogy_results['analogy_results']
    success = False
    top_i = 0
    for final_pose in analogy_results:
        success = plan_without_robot(target_scene, final_pose, human_viewer=False)
        if with_robot:
            if success:
                success = plan(target_scene, final_pose, human_viewer=False, timeout=timeout,
                               video_save_path=f'video/mug_{desc}/{mug_id}.mp4' if save_video else None,
                               surface_pointing_tol=0.8)
        top_i += 1
        if success:
            break
    print('Success:', success)
    print(f'------------------------- Mug {mug_id} success: {success} ------------------------')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mug_id', type=int, default=1, help='choose from 0 to 200')
    args.add_argument('--with_robot', type=int, default=1)
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--timeout', type=int, default=60)
    args.add_argument('--desc', type=str, default='exp')
    args.add_argument('--save_video', type=int, default=0)
    args.add_argument('--use_e2', type=int, default=0)
    args = args.parse_args()

    mug_id = args.mug_id
    assert 0 <= mug_id <= 200, 'mug id must be from 0 to 200'
    desc = args.desc
    with_robot = bool(args.with_robot)
    timeout = args.timeout# this is only used for the spoon task
    np_seed = args.seed
    save_video = bool(args.save_video)
    use_e2 = bool(args.use_e2)

    main(mug_id=mug_id, desc=desc, with_robot=with_robot, timeout=timeout, np_seed=np_seed, save_video=save_video, use_e2=use_e2)
