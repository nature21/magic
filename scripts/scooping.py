import argparse
import os
import time
import numpy as np
from mplib import pymp
from sapien.core import Pose
from tqdm import tqdm

from magic.match import double_match
from manipulation_utils.contact_samplers import sample_grasp
from manipulation_utils.motion import gen_qpos_to_qpos_trajectory, gen_qpos_to_pose_trajectory, trajectory_warp
from manipulation_utils.controllers import TrajectoryPositionController
from env.envs.spoonScene import SpoonScene, get_reference_traj
from utils.camera_utils import imgs2mp4
from utils.mesh_utils import get_normal_of_nearest_point
from utils.rotation_utils import normalize_vector, rotate_vector, cross, wxyz2xyzw, quat_mul, \
    get_quaternion_from_matrix, quat_conjugate, xyzw2wxyz
from utils.sapien_utils import get_actor_pcd
from utils.ui_utils import get_click_coordinates_from_array


def get_reference(manual_center=False, plane_origin=np.array([0, 0, 0]), plane_normal=np.array([0, -1, 0])):
    """Get the reference image and the center of the reference image."""
    reference_scene = SpoonScene(0)
    reference_img, reference_pixel_to_3d_fn = reference_scene.get_slice(plane_origin, plane_normal)
    if manual_center:
        reference_support_center = get_click_coordinates_from_array(np.asarray(reference_img))
        reference_collide_center = get_click_coordinates_from_array(np.asarray(reference_img))
        reference_grasp_center = get_click_coordinates_from_array(np.asarray(reference_img))
        print(f'using manual support center and collide center: {reference_support_center}, {reference_collide_center}')
    else:
        reference_support_center = (179, 409)
        reference_collide_center = (99, 365)
        reference_grasp_center = (626, 387)
        print(
            f'using default support center and collide center: {reference_support_center}, {reference_collide_center}')

    reference_init_pose = reference_scene.spoon.get_pose()

    reference_init_pcd = get_actor_pcd(reference_scene.spoon, num_points=5000, to_world_frame=False, return_o3d=True)

    return reference_img, reference_support_center, reference_collide_center, reference_grasp_center, reference_pixel_to_3d_fn, reference_init_pose, reference_init_pcd


def transfer(
        alignment_result,
        reference_collide_center,
        reference_plane_normal,
        reference_pixel_to_3d_fn,
        target_plane_normal,
        target_pixel_to_3d_fn,
        target_init_pose,
        only_dino=False,
        reference_init_pose=None,
        reference_pcd=None,
        target_pcd=None
):
    """Transfer the alignment result to the final pose of the target object."""
    center_of_curvature_1 = alignment_result['center_of_curvature_1']
    center_of_curvature_2 = alignment_result['center_of_curvature_2']

    center_of_contact_1 = alignment_result['center_of_contact_1']
    center_of_contact_2 = alignment_result['center_of_contact_2']

    center_of_curvature_1_world = reference_pixel_to_3d_fn(center_of_curvature_1)
    center_of_curvature_2_world = target_pixel_to_3d_fn(center_of_curvature_2)

    center_of_contact_1_world = reference_pixel_to_3d_fn(center_of_contact_1)
    center_of_contact_2_world = target_pixel_to_3d_fn(center_of_contact_2)

    if not only_dino:
        x_axis_1 = normalize_vector(center_of_curvature_1_world - center_of_contact_1_world)
        x_axis_2 = normalize_vector(center_of_curvature_2_world - center_of_contact_2_world)
    else:
        center_of_contact_1_object = (reference_init_pose.inv() * Pose(center_of_contact_1_world)).p
        center_of_contact_2_object = (target_init_pose.inv() * Pose(center_of_contact_2_world)).p
        normal_1 = get_normal_of_nearest_point(reference_pcd, center_of_contact_1_object)
        normal_2 = get_normal_of_nearest_point(target_pcd, center_of_contact_2_object)
        normal_1_world = rotate_vector(normal_1, wxyz2xyzw(reference_init_pose.q))
        normal_2_world = rotate_vector(normal_2, wxyz2xyzw(target_init_pose.q))
        x_axis_1 = normalize_vector(
            normal_1_world - np.dot(normal_1_world, reference_plane_normal) * reference_plane_normal)
        x_axis_2 = normalize_vector(normal_2_world - np.dot(normal_2_world, target_plane_normal) * target_plane_normal)

    y_axis_1 = reference_plane_normal
    z_axis_1 = cross(x_axis_1, y_axis_1)
    y_axis_2 = target_plane_normal
    z_axis_2 = cross(x_axis_2, y_axis_2)

    goal_rotation_contact_2 = np.array([x_axis_1, y_axis_1, z_axis_1]).T
    init_rotation_contact_2 = np.array([x_axis_2, y_axis_2, z_axis_2]).T

    pos_init_2 = target_init_pose.p
    q_init_2 = wxyz2xyzw(target_init_pose.q)

    q_goal_rotation_collide_2 = get_quaternion_from_matrix(goal_rotation_contact_2)
    q_init_rotation_collide_2 = get_quaternion_from_matrix(init_rotation_contact_2)
    q_relative_2 = quat_mul(q_goal_rotation_collide_2, quat_conjugate(q_init_rotation_collide_2))
    q_goal_2 = xyzw2wxyz(quat_mul(q_relative_2, q_init_2))

    center_of_collide_1_world = reference_pixel_to_3d_fn(reference_collide_center)
    if 'center_of_collide_2' in alignment_result:
        target_collide_center = alignment_result['center_of_collide_2']
        center_of_collide_2_world = target_pixel_to_3d_fn(target_collide_center)
        goal_pos_collide_2 = center_of_collide_1_world
        pos_goal_2 = goal_pos_collide_2 + rotate_vector(pos_init_2 - center_of_collide_2_world, q_relative_2)
    else:
        # in this case, align contact points
        goal_pos_collide_2 = center_of_contact_1_world
        pos_goal_2 = goal_pos_collide_2 + rotate_vector(pos_init_2 - center_of_contact_2_world, q_relative_2)

    return Pose(pos_goal_2, q_goal_2), center_of_curvature_1_world, center_of_contact_1_world, center_of_collide_1_world


def analogy(spoon_id, desc, top_k=3, read_from_result=False, only_compute_match=False, radius=0.035, use_reflection=False):
    plane_origin = np.array([0, -0.35, 0])
    plane_normal = np.array([0, -1, 0])
    (reference_img, reference_support_center, reference_collide_center, reference_grasp_center,
     reference_pixel_to_3d_fn, reference_init_pose, reference_pcd_o3d) = get_reference(
        manual_center=False, plane_origin=plane_origin, plane_normal=plane_normal
    )

    reference_p_array, reference_q_array = get_reference_traj()

    reference_pose_list = [Pose(p, q) for p, q in zip(reference_p_array, reference_q_array)]

    target_scene = SpoonScene(
        spoon_id,
        fps=480,
        add_robot=True,
        radius=radius
    )

    target_img, target_pixel_to_3d_fn = target_scene.get_slice(plane_origin, plane_normal)

    analogy_results = []

    if read_from_result:
        alignment_results = np.load(f'alignment_results/scooping_{desc}/{spoon_id}/alignment_results.npy',
                                    allow_pickle=True)
    else:
        pca = True
        patch_size = 13
        alignment_results = double_match(reference_img, target_img, reference_support_center, reference_collide_center,
                                         grasp_center=reference_grasp_center,
                                         parameter_save_dir=f'alignment_results/scooping_{desc}/{spoon_id}',
                                         save_dir=f'results/scooping_{desc}/{spoon_id}',
                                         top_k=top_k,
                                         pca=pca,
                                         patch_size=patch_size,
                                         use_reflection=use_reflection)

    if only_compute_match:
        return

    for alignment_result in alignment_results[:top_k]:
        target_scene.reset()
        init_pose = target_scene.spoon.get_pose()
        target_align_init_pose, final_center_of_curvature_world, final_center_of_contact_world, final_center_of_collide_world = transfer(
            alignment_result,
            reference_collide_center,
            plane_normal,
            reference_pixel_to_3d_fn,
            plane_normal,
            target_pixel_to_3d_fn,
            init_pose
        )

        analogy_results.append(
            trajectory_warp(
                reference_pose_list, reference_init_pose,
                init_pose, target_align_init_pose,
                target_grasp_center=alignment_result['center_of_grasp_2'],
                target_pixel_to_3d_fn=target_pixel_to_3d_fn,
            )
        )

    return {
        'target_scene': target_scene,
        'analogy_results': analogy_results
    }


def plan(target_scene, target_p_array, target_q_array, init_grasp_center_world, video_save_path=None,
         human_viewer=False,
         timeout=60, grasp_guidance_radius=0.01):
    target_scene.reset()
    init_pose = target_scene.spoon.get_pose()
    target_scene.open_gripper()
    pose_1 = target_scene.spoon.get_pose()
    p_grasp_center_rel_init = init_grasp_center_world - init_pose.p
    grasp_center_1 = pose_1.p + rotate_vector(p_grasp_center_rel_init, wxyz2xyzw((pose_1 * init_pose.inv()).q))

    start_time = time.time()
    grasp_generator = sample_grasp(
        target_scene, object_id=target_scene.spoon.get_id(),
        guidance_center=grasp_center_1,
        guidance_radius=grasp_guidance_radius,
        start_time=start_time,
        timeout=timeout
    )

    checkpoint = target_scene.scene.pack()

    result_1_to_2 = None
    result_2_to_3 = None
    result_3_to_4 = None
    result_5 = None

    while True:
        if time.time() - start_time > timeout:
            print('Timeout, failed to find a feasible solution')
            return False
        try:
            grasp_parameter = next(grasp_generator)
        except StopIteration:
            print('Failed to find grasp solution')
            return False

        result_1_to_2 = gen_qpos_to_qpos_trajectory(
            target_scene,
            target_scene.robot.get_qpos(),
            grasp_parameter.qpos,
            pcd_resolution=1e-2,
        )

        if result_1_to_2['status'] != 'Success':
            target_scene.set_up_planner()
            target_scene.scene.unpack(checkpoint)
            print('Failed to find grasp trajectory solution')
            continue

        collision = target_scene.follow_path(result_1_to_2, check_collision=True,
                                             collision_obj_1=target_scene.spoon.get_id(),
                                             collision_obj_2=target_scene.end_effector.get_id())
        if collision:
            target_scene.set_up_planner()
            target_scene.scene.unpack(checkpoint)
            print('Grasp trajectory collides with the object')
            continue

        target_scene.attach_object(target_scene.spoon)
        target_scene.close_gripper(0.03)
        target_scene.planner_attach_obj(target_scene.spoon)

        ee_pose_2 = target_scene.end_effector.get_pose()
        ee_pose_3 = Pose(ee_pose_2.p + np.array([0, 0, 0.1]), ee_pose_2.q)
        result_2_to_3 = gen_qpos_to_pose_trajectory(
            target_scene,
            grasp_parameter.qpos,
            pymp.Pose(ee_pose_3.p, ee_pose_3.q),
            pcd_resolution=1e-2,
            exclude_ids=[target_scene.spoon.get_id()]
        )

        if result_2_to_3['status'] != 'Success':
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            print('Failed to find post-grasp trajectory solution')
            continue

        target_scene.follow_path(result_2_to_3)
        qpos_3 = target_scene.robot.get_qpos()

        spoon_pose_3 = target_scene.spoon.get_pose()
        spoon_pose_4 = Pose(target_p_array[0], target_q_array[0])
        ee_pose_4 = spoon_pose_4 * spoon_pose_3.inv() * ee_pose_3

        result_3_to_4 = gen_qpos_to_pose_trajectory(
            target_scene,
            qpos_3,
            pymp.Pose(ee_pose_4.p, ee_pose_4.q),
            # pcd_resolution=5e-3,
            pcd_resolution=5e-4,
            exclude_ids=[target_scene.spoon.get_id()],
            planning_time=5,
            verbose=True
        )

        if result_3_to_4['status'] != 'Success':
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            print('Failed to find pre-scoop trajectory solution')
            continue

        collision = target_scene.follow_path(result_3_to_4, check_collision=True,
                                             collision_obj_1=target_scene.spoon.get_id(),
                                             collision_obj_2=target_scene.plane.get_id())
        if collision:
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            print('Failed to find pre-scoop trajectory solution due to collision')
            continue

        qpos_4 = target_scene.robot.get_qpos()
        target_scene.planner_detach_obj()

        ee_pose_list = [Pose(target_p_array[i], target_q_array[i]) * spoon_pose_3.inv() * ee_pose_3 for i in
                        range(len(target_p_array))]

        qpos_list = []
        success = True
        for i in range(len(ee_pose_list)):
            current_qpos = qpos_4 if len(qpos_list) == 0 else qpos_list[-1]
            qpos = target_scene.ee_ik_without_collision_check(ee_pose_list[i], current_qpos, return_closest=True,
                                                              verbose=True)
            if qpos is None or np.linalg.norm(qpos - current_qpos) > 1:
                # if qpos is None:
                success = False
                break
            qpos_list.append(qpos)
        if not success:
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            print('Failed to find scoop trajectory solution')
            continue

        result_5 = {}
        # qpos_array = np.repeat(np.array(qpos_list), 10, axis=0)
        # don't repeat but do interpolation
        qpos_array = np.array(qpos_list)
        interpolated_qpos_array = np.zeros(((len(qpos_array) - 1) * 5, qpos_array.shape[1]))
        n = len(qpos_array)
        for i in range(n - 1):
            for j in range(5):
                interpolated_qpos_array[i * 5 + j] = qpos_array[i] + (qpos_array[i + 1] - qpos_array[i]) * j / 5
        qpos_array = interpolated_qpos_array

        vel_qpos_array = np.zeros_like(qpos_array)
        result_5['position'] = qpos_array
        result_5['velocity'] = vel_qpos_array

        target_scene.set_up_planner()
        target_scene.detach_object()
        target_scene.reset()
        break

    print('finished planning, now executing the plan and generating a video, this might take a while')

    if video_save_path is not None:
        camera = target_scene.add_camera(direction='+x+z')
        target_scene.open_gripper()
        _, imgs_1_to_2 = target_scene.follow_path(result_1_to_2, camera=camera, camera_interval=16)
        target_scene.attach_object(target_scene.spoon)
        target_scene.close_gripper(0.002)
        _, imgs_2_to_3 = target_scene.follow_path(result_2_to_3, camera=camera, camera_interval=16)
        _, imgs_3_to_4 = target_scene.follow_path(result_3_to_4, camera=camera, camera_interval=16)
        _, imgs_4_to_5 = target_scene.follow_path(result_5, camera=camera, camera_interval=16)
        imgs = imgs_1_to_2 + imgs_2_to_3 + imgs_3_to_4 + imgs_4_to_5
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        imgs2mp4(imgs, video_save_path, 120)

    else:
        viewer = target_scene.create_viewer() if human_viewer else None
        target_scene.open_gripper()
        target_scene.follow_path(result_1_to_2)
        target_scene.attach_object(target_scene.spoon)
        target_scene.close_gripper(0.002)
        target_scene.follow_path(result_2_to_3)
        target_scene.follow_path(result_3_to_4)
        target_scene.follow_path(result_5)
        if human_viewer:
            viewer.close()

    success = target_scene.is_success()
    target_scene.detach_object()
    target_scene.set_up_planner()
    target_scene.reset()

    return success


def plan_without_robot(target_scene, target_p_array, target_q_array, human_viewer=False):
    target_scene.reset()
    trajectory_controller = TrajectoryPositionController(
        target_scene.spoon,
        p_array=target_p_array,
        q_array=target_q_array,
        gain=1 * target_scene.fps,
        coef=1e-2
    )

    viewer = target_scene.create_viewer() if human_viewer else None

    target_scene.spoon.set_pose(Pose(target_p_array[0], target_q_array[0]))
    success = False
    for _ in range(len(target_p_array) * 2):
        trajectory_controller.set_velocities()
        target_scene.step()
        target_scene.update_render()
        if viewer is not None:
            viewer.render()
        if target_scene.is_success():
            success = True
            break

    if viewer is not None:
        viewer.close()

    return success


def main(spoon_id, target_half_size, with_robot=False, seed=0, timeout=60, desc='exp', save_video=False, use_e2=False):
    np.random.seed(seed)
    grasp_guidance_radius = 0.01
    read_from_result = False
    analogy_results = analogy(spoon_id, desc, top_k=3, read_from_result=read_from_result, only_compute_match=False,
                              radius=target_half_size, use_reflection=use_e2)
    scene = analogy_results['target_scene']
    analogy_results = analogy_results['analogy_results']
    for j, analogy_result in enumerate(analogy_results):
        target_p_array = analogy_result['target_p_array']
        target_q_array = analogy_result['target_q_array']
        success = plan_without_robot(scene, target_p_array, target_q_array, human_viewer=False)
        if with_robot:
            if success:
                success = plan(scene, target_p_array, target_q_array,
                               analogy_result['init_grasp_center_world'],
                               video_save_path=f'videos/scooping_{desc}/scooping_{spoon_id}.mp4' if save_video else None,
                               human_viewer=True,
                               grasp_guidance_radius=grasp_guidance_radius,
                               timeout=timeout)
        print(f'spoon {spoon_id}, top {j}, radius {target_half_size}, success: {success}')
        if success:
            break



if __name__ == '__main__':
    spoon_ids = [1, 7, 8, 9, 11, 12]
    target_half_sizes = [0.015, 0.035, 0.045]
    args = argparse.ArgumentParser()
    args.add_argument('--spoon_id', type=int, default=1, help=f'choose from 1 to 6')
    args.add_argument('--target', type=float, default=0.035, help=f'choose from f{target_half_sizes}')
    args.add_argument('--with_robot', type=int, default=1)
    args.add_argument('--num_epoch', type=int, default=10)
    args.add_argument('--timeout', type=int, default=60)
    args.add_argument('--desc', type=str, default='exp')
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--save_video', type=int, default=0)
    args.add_argument('--use_e2', type=int, default=0)
    args = args.parse_args()

    spoon_id = args.spoon_id
    target = args.target
    assert 1 <= spoon_id <= 6, 'spoon id must be from 1 to 6'
    assert target in target_half_sizes, f'target should be in {target_half_sizes}'
    with_robot = bool(args.with_robot)
    num_epoch = args.num_epoch
    timeout = args.timeout
    desc = args.desc
    seed = args.seed
    save_video = bool(args.save_video)
    use_e2 = bool(args.use_e2)

    main(spoon_id=spoon_id, target_half_size=target, with_robot=with_robot, timeout=timeout, seed=seed, desc=desc, save_video=save_video, use_e2=use_e2)

