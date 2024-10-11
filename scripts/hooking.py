import argparse
import os
from time import time
import numpy as np
from mplib import pymp
from sapien.core import Pose

from magic.match import match
from magic.utils_2d import find_nearest_point_with_mask
from manipulation_utils.contact_samplers import load_indirect_push_parameter, sample_grasp
from manipulation_utils.motion import gen_qpos_to_pose_trajectory
from manipulation_utils.controllers import LinePositionController
from env.envs.hookScene import HookScene
from utils.camera_utils import uvz2world, imgs2mp4
from utils.mesh_utils import get_actor_mesh
from utils.rotation_utils import rpy2wxyz, quat_mul_wxyz, wxyz2xyzw, rotate_vector, normalize_vector, cross, \
    get_quaternion_from_matrix, quat_conjugate, quat_mul, xyzw2wxyz
from utils.ui_utils import get_click_coordinates_from_array


def get_reference(manual_center=False, reference_push_parameter_path='reference/hook/push_parameter.json'):
    """Get the reference image and the center of the reference image."""
    reference_scene = HookScene(
        'hook',
        'ball',
        target_half_size=0.025,
        target_x_y=np.array([0.9, 0]),
        add_robot=True,
    )

    push_parameter = load_indirect_push_parameter(reference_push_parameter_path)
    tool_quat_wxyz = quat_mul_wxyz(rpy2wxyz([0, 0, np.pi * 10 / 180]), push_parameter.tool_quat_wxyz)
    reference_pose = Pose(push_parameter.tool_pos + np.array([0.65, 0, -0.025]), tool_quat_wxyz)
    reference_scene.tool.set_pose(reference_pose)

    reference_init_pose = reference_scene.tool.get_pose()

    reference_scene.hide_env_visual()
    reference_img, reference_depth_img, reference_camera = reference_scene.get_picture(direction='+z',
                                                                                       additional_translation=np.array(
                                                                                           [0.8, 0, -0.375 * 1.414]),
                                                                                       debug_viewer=False,
                                                                                       get_depth=True)
    if manual_center:
        reference_contact_center = get_click_coordinates_from_array(np.asarray(reference_img))
        reference_grasp_center = get_click_coordinates_from_array(np.asarray(reference_img))
        print('using manual contact and grasp center', reference_contact_center, reference_grasp_center)
    else:
        reference_contact_center = (379, 200)
        reference_grasp_center = (439, 532)
        print('using default contact and grasp center', reference_contact_center, reference_grasp_center)
    return reference_img, reference_depth_img, reference_camera, reference_contact_center, reference_grasp_center, reference_scene, reference_init_pose, reference_pose


def transfer(
        alignment_result,
        reference_depth_img,
        target_depth_img,
        reference_camera,
        target_camera,
        final_direction_contact,
        final_pos_contact,
        tool_init_pose,
):
    """Transfer the alignment result to the final pose of the target object."""
    center_of_curvature_1 = alignment_result['center_of_curvature_1']
    center_of_curvature_2 = alignment_result['center_of_curvature_2']
    center_of_contact_1 = alignment_result['center_of_contact_1']
    center_of_contact_2 = alignment_result['center_of_contact_2'].astype(int)

    center_of_contact_1 = find_nearest_point_with_mask((reference_depth_img > 1e-5), center_of_contact_1, 20)
    center_of_contact_2 = find_nearest_point_with_mask((target_depth_img > 1e-5), center_of_contact_2, 20)

    depth_1 = reference_depth_img[center_of_contact_1[1], center_of_contact_1[0]]
    depth_2 = target_depth_img[center_of_contact_2[1], center_of_contact_2[0]]

    center_of_contact_1_world = uvz2world(reference_camera, np.array([*center_of_contact_1, depth_1]))
    center_of_contact_2_world = uvz2world(target_camera, np.array([*center_of_contact_2, depth_2]))

    z_axis_2 = rotate_vector([-1, 0, 0], wxyz2xyzw(target_camera.get_pose().q))
    center_of_curvature_1_world = uvz2world(reference_camera, np.array([*center_of_curvature_1, depth_1]))
    center_of_curvature_2_world = uvz2world(target_camera, np.array([*center_of_curvature_2, depth_2]))
    sign_1 = 1 if alignment_result['radius_of_curvature_1'] > 0 else -1
    sign_2 = 1 if alignment_result['radius_of_curvature_2'] > 0 else -1
    radius_of_curvature_1_world = np.linalg.norm(center_of_curvature_1_world - center_of_contact_1_world) * sign_1
    radius_of_curvature_2_world = np.linalg.norm(center_of_curvature_2_world - center_of_contact_2_world) * sign_2
    x_axis_2 = normalize_vector(center_of_contact_2_world - center_of_curvature_2_world)
    y_axis_2 = cross(z_axis_2, x_axis_2)
    current_rotation_contact_2 = np.array([x_axis_2, y_axis_2, z_axis_2]).T

    x_axis_final = normalize_vector(final_direction_contact)
    z_axis_final = np.array([0, 0, 1])
    y_axis_final = cross(z_axis_final, x_axis_final)
    final_rotation_contact_2 = np.array([x_axis_final, y_axis_final, z_axis_final]).T

    pos_current_2 = tool_init_pose.p
    q_current_2 = wxyz2xyzw(tool_init_pose.q)

    q_current_rotation_contact_2 = get_quaternion_from_matrix(current_rotation_contact_2)
    q_final_rotation_contact_2 = get_quaternion_from_matrix(final_rotation_contact_2)
    q_relative_2 = quat_mul(q_final_rotation_contact_2, quat_conjugate(q_current_rotation_contact_2))
    q_final_2 = xyzw2wxyz(quat_mul(q_relative_2, q_current_2))

    pos_final_2 = final_pos_contact + rotate_vector(pos_current_2 - center_of_contact_2_world, q_relative_2)

    center_of_grasp_2 = alignment_result['center_of_grasp_2'] if 'center_of_grasp_2' in alignment_result else None
    if center_of_grasp_2 is not None:
        try:
            center_of_grasp_2 = find_nearest_point_with_mask((target_depth_img > 1e-5), center_of_grasp_2.astype(int),
                                                             50)
            center_of_grasp_2_world = uvz2world(target_camera, np.array([*center_of_grasp_2, depth_2]))
        except Exception as e:
            print(e)
            center_of_grasp_2_world = None
    else:
        center_of_grasp_2_world = None

    return Pose(pos_final_2,
                q_final_2), center_of_grasp_2_world, radius_of_curvature_1_world, radius_of_curvature_2_world


def analogy(
        tool_name, target_name='ball', top_k=3, read_from_result=False, only_compute_match=False,
        target_half_size=0.025, desc='exp', use_reflection=False,
):
    (reference_img, reference_depth_img, reference_camera,
     reference_contact_center, reference_grasp_center, reference_scene, reference_init_pose,
     reference_pose) = get_reference(manual_center=False)

    target_scene = HookScene(
        tool_name,
        target_name,
        target_half_size=target_half_size,
        add_robot=True
    )

    target_scene.hide_env_visual()
    additional_translation = np.array([0.6, -0.6, 0.2 - 0.375 * 1.414])
    if tool_name == 'watch':
        additional_translation += np.array([0, 0.2, 0])
    target_img, target_depth_img, target_camera = target_scene.get_picture(
        direction='+z',
        additional_translation=additional_translation,
        debug_viewer=False,
        get_depth=True
    )
    target_scene.unhide_env_visual()

    analogy_results = []

    if read_from_result:
        alignment_results = np.load(f'alignment_results/hooking_{desc}/{tool_name}/alignment_results.npy',
                                    allow_pickle=True)
    else:
        pca = True
        patch_size = 13
        alignment_results = match(reference_img, target_img, reference_contact_center, reference_grasp_center,
                                  parameter_save_dir=f'alignment_results/hooking_{desc}/{tool_name}',
                                  save_dir=f'results/hooking_{desc}/{tool_name}', top_k=top_k,
                                  pca=pca, patch_size=patch_size, use_reflection=use_reflection)

    if only_compute_match:
        return

    for alignment_result in alignment_results[:top_k]:
        target_scene.reset()
        target_x, target_y = target_scene.target_x_y
        transfer_tool_pose, grasp_center, radius_1, radius_2 = transfer(
            alignment_result,
            reference_depth_img,
            target_depth_img,
            reference_camera,
            target_camera,
            final_direction_contact=np.array([1, 0, 0]),
            final_pos_contact=np.array([target_x + target_half_size * 1.5, target_y, 0.025]),
            tool_init_pose=target_scene.tool.get_pose()
        )

        analogy_results.append({
            'transfer_tool_pose': transfer_tool_pose,
            'init_grasp_center_world': grasp_center,
            'reference_radius_of_curvature': radius_1,
            'target_radius_of_curvature': radius_2
        })

    return {
        'target_scene': target_scene,
        'analogy_results': analogy_results
    }


def plan(
        target_scene: HookScene,
        transfer_tool_pose,
        init_grasp_center_world,
        video_save_path=None,
        human_viewer=False,
        timeout=60,
        return_traj=False,
        surface_pointing_tol=0.6,
        push_distance=0.3,
):
    target_scene.reset()
    target_scene.tool.set_pose(transfer_tool_pose)
    target_mesh = get_actor_mesh(target_scene.tool)
    # get min z and max z
    min_z = np.min(target_mesh.vertices[:, 2])
    max_z = np.max(target_mesh.vertices[:, 2])
    target_half_size = target_scene.target_half_size

    if target_half_size + min_z / 2 - max_z / 2 < 0:
        delta_z = -min_z / 2
    else:
        delta_z = target_half_size - min_z / 2 - max_z / 2

    offset = target_scene.target_half_size / 2
    if target_scene.tool.get_name() == 'watch':
        offset += 0.02  # there is something unusual with the mesh of watch, so we need to add an additional offset
    transfer_tool_pose = Pose(transfer_tool_pose.p + np.array([0, 0, delta_z + offset]), transfer_tool_pose.q)

    target_scene.reset()

    tool_init_pose = target_scene.tool.get_pose()
    target_scene.open_gripper()
    tool_pose_1 = target_scene.tool.pose
    grasp_center_1 = tool_pose_1.p + rotate_vector(init_grasp_center_world - tool_init_pose.p,
                                                   wxyz2xyzw((tool_pose_1 * tool_init_pose.inv()).q))

    start_time = time()
    grasp_generator = sample_grasp(
        target_scene,
        target_scene.tool.get_id(),
        guidance_center=grasp_center_1,
        guidance_direction=np.array([0, 0, 1]),
        surface_pointing_tol=surface_pointing_tol,
        guidance_radius=0.05,
        max_test_points=1000000,
        start_time=start_time,
        timeout=timeout,
        verbose=False
    )

    checkpoint = target_scene.scene.pack()
    result_1_to_2 = None
    result_2_to_3 = None
    result_3_to_4 = None
    result_4_to_5 = None
    result_5_to_6 = None

    while True:
        if time() - start_time > timeout:
            print('Timeout, failed to find a feasible solution')
            if return_traj:
                return False, None
            else:
                return False
        try:
            grasp_parameter = next(grasp_generator)
        except StopIteration:
            print('Failed to find grasp solution')
            if return_traj:
                return False, None
            else:
                return False


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
            ignore_env=True
        )

        if result_1_5_to_2['status'] != 'Success':
            print('Failed to generate grasp trajectory')
            target_scene.scene.unpack(checkpoint)
            continue

        target_scene.follow_path(result_1_5_to_2)

        result_1_to_2 = {'position': np.concatenate([result_1_to_1_5['position'], result_1_5_to_2['position']]),
                         'velocity': np.concatenate([result_1_to_1_5['velocity'], result_1_5_to_2['velocity']])}

        target_scene.attach_object(target_scene.tool)
        target_scene.close_gripper(0.03)
        target_scene.planner_attach_obj(target_scene.tool)

        # lift ee a little
        ee_pose_2 = target_scene.end_effector.get_pose()
        ee_pose_3 = Pose(ee_pose_2.p + np.array([0, 0, 0.1]), ee_pose_2.q)

        result_2_to_3 = gen_qpos_to_pose_trajectory(
            target_scene,
            grasp_parameter.qpos,
            pymp.Pose(ee_pose_3.p, ee_pose_3.q),
            exclude_ids=[target_scene.tool.get_id()],
            planning_time=5,
            pcd_resolution=1e-3
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
        tool_pose_3 = target_scene.tool.get_pose()
        ee_pose_5 = transfer_tool_pose * tool_pose_3.inv() * ee_pose_3
        ee_pose_4 = Pose(ee_pose_5.p + np.array([0, 0, 0.1]), ee_pose_5.q)

        result_3_to_4 = gen_qpos_to_pose_trajectory(
            target_scene,
            qpos_3,
            ee_pose_4,
            exclude_ids=[target_scene.tool.get_id()],
            planning_time=5,
            pcd_resolution=1e-3,
            verbose=True
        )

        if result_3_to_4['status'] != 'Success':
            print('Failed to generate prepush motion planning trajectory')
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            continue

        target_scene.follow_path(result_3_to_4)
        qpos_4 = target_scene.robot.get_qpos()

        result_4_to_5 = gen_qpos_to_pose_trajectory(
            target_scene,
            qpos_4,
            ee_pose_5,
            exclude_ids=[target_scene.tool.get_id()],
            planning_time=5,
            pcd_resolution=1e-5,
            verbose=True,
            ignore_env=True,
        )

        if result_4_to_5['status'] != 'Success':
            print('Failed to generate push motion planning trajectory')
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            continue

        target_scene.follow_path(result_4_to_5)
        qpos_5 = target_scene.robot.get_qpos()

        target_scene.planner_detach_obj()

        ee_pose_list = [Pose(
            ee_pose_5.p + np.array([-push_distance * (i + 1) / 480, 0, 0]), ee_pose_5.q
        ) for i in range(480)]

        success = True
        qpos_list = []
        for i in range(len(ee_pose_list)):
            current_qpos = qpos_5 if i == 0 else qpos_list[-1]
            qpos = target_scene.ee_ik_without_collision_check(
                ee_pose_list[i], current_qpos, return_closest=True, verbose=True
            )
            if qpos is None or np.linalg.norm(qpos - current_qpos) > 1:
                # if qpos is None:
                success = False
                print(f'ik step {i} failed')
                break
            qpos_list.append(qpos)
        if not success:
            target_scene.set_up_planner()
            target_scene.detach_object()
            target_scene.scene.unpack(checkpoint)
            print('Failed to generate postpush motion planning trajectory')
            continue

        result_5_to_6 = {}
        # interpolation
        qpos_array = np.array(qpos_list)
        interpolated_qpos_array = np.zeros(((len(qpos_array) - 1) * 5, qpos_array.shape[1]))
        n = len(qpos_array)
        for i in range(n - 1):
            for j in range(5):
                interpolated_qpos_array[i * 5 + j] = qpos_array[i] + (qpos_array[i + 1] - qpos_array[i]) * j / 5
        qpos_array = interpolated_qpos_array

        vel_qpos_array = np.zeros_like(qpos_array)
        result_5_to_6['position'] = qpos_array
        result_5_to_6['velocity'] = vel_qpos_array

        target_scene.set_up_planner()
        target_scene.detach_object()
        target_scene.reset()
        break

    print('finished planning, now executing the plan and generating a video, this might take a while')
    total_traj_len = (len(result_1_to_2['position']) + len(result_2_to_3['position']) + len(result_3_to_4['position'])
                      + len(result_4_to_5['position']) + len(result_5_to_6['position']))
    print('total traj length = ', total_traj_len)

    if video_save_path is not None:
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        camera = target_scene.add_camera(direction='+x+z')
        camera_pose = camera.get_pose()
        camera_interval = total_traj_len // 4000
        camera.set_pose(Pose(camera_pose.p + np.array([1, 0, 0.5]), camera_pose.q))
        target_scene.open_gripper()
        _, imgs_1_to_2 = target_scene.follow_path(result_1_to_2, camera=camera, camera_interval=camera_interval)
        target_scene.attach_object(target_scene.tool)
        target_scene.close_gripper(0.015)
        # target_scene.close_gripper(0.02) # this is for watch
        _, imgs_2_to_3 = target_scene.follow_path(result_2_to_3, camera=camera, camera_interval=camera_interval)
        _, imgs_3_to_4 = target_scene.follow_path(result_3_to_4, camera=camera, camera_interval=camera_interval)
        _, imgs_4_to_5 = target_scene.follow_path(result_4_to_5, camera=camera, camera_interval=camera_interval)
        _, imgs_5_to_6 = target_scene.follow_path(result_5_to_6, camera=camera, camera_interval=camera_interval)

        imgs = imgs_1_to_2 + imgs_2_to_3 + imgs_3_to_4 + imgs_4_to_5 + imgs_5_to_6
        imgs2mp4(imgs, video_save_path, fps=120)
    else:
        viewer = target_scene.create_viewer() if human_viewer else None
        target_scene.open_gripper()
        target_scene.follow_path(result_1_to_2)
        target_scene.attach_object(target_scene.tool)
        target_scene.close_gripper(0.015)
        target_scene.follow_path(result_2_to_3)
        target_scene.follow_path(result_3_to_4)
        target_scene.follow_path(result_4_to_5)
        target_scene.follow_path(result_5_to_6)
        if viewer is not None:
            viewer.close()

    success = target_scene.is_success()
    target_scene.detach_object()
    target_scene.set_up_planner()
    target_scene.reset()

    if return_traj:
        return success, {
            'result_1_to_2': result_1_to_2,
            'result_2_to_3': result_2_to_3,
            'result_3_to_4': result_3_to_4,
            'result_4_to_5': result_4_to_5,
            'result_5_to_6': result_5_to_6,
        }
    return success


def plan_without_robot(target_scene, transfer_tool_pose, human_viewer=False):
    target_scene.reset()
    target_scene.tool.set_pose(transfer_tool_pose)

    target_mesh = get_actor_mesh(target_scene.tool)
    # get min z and max z
    min_z = np.min(target_mesh.vertices[:, 2])
    max_z = np.max(target_mesh.vertices[:, 2])
    target_half_size = target_scene.target_half_size

    if target_half_size + min_z / 2 - max_z / 2 < 0:
        delta_z = -min_z / 2
    else:
        delta_z = target_half_size - min_z / 2 - max_z / 2

    offset = target_scene.target_half_size / 2
    if target_scene.tool.get_name() == 'watch':
        offset += 0.02 # there is something unusual with the mesh of watch, so we need to add an additional offset
    target_scene.tool.set_pose(Pose(transfer_tool_pose.p + np.array([0, 0, delta_z + offset]), transfer_tool_pose.q))

    total_push_distance = 0.3
    push_dir = np.array([-1, 0, 0])
    init_pos = target_scene.tool.get_pose().p
    target_pose = init_pos + total_push_distance * push_dir
    num_steps = int(target_scene.fps * 10)
    # in the feature, write a compliant controller to automatically adjust gain_coef
    gain_coef = 0.5 if target_scene.tool.get_name() != 'watch' else 1
    controller = LinePositionController(target_scene.tool, init_pos, target_pose, num_steps, gain_coef * target_scene.fps)
    success = False
    viewer = target_scene.create_viewer() if human_viewer else None
    for _ in range(num_steps * 2):
        controller.set_velocities()
        target_scene.step()
        target_scene.update_render()
        if viewer is not None:
            viewer.render()
        if target_scene.is_terminated():
            success = target_scene.is_success()
            break
    if viewer is not None:
        viewer.close()
    return success


def main(tool, target_half_size, np_seed=0, with_robot=False, timeout=60, desc='exp', save_video=False, use_e2=False):
    np.random.seed(np_seed)
    print('analogy tool:', tool, 'target_half_size:', target_half_size)
    read_from_results = False
    analogy_results = analogy(tool, target_name='cylinder', top_k=3, read_from_result=read_from_results,
                              only_compute_match=False, target_half_size=target_half_size, desc=desc,
                              use_reflection=use_e2)
    target_scene = analogy_results['target_scene']
    analogy_results = analogy_results['analogy_results']
    for top_i, analogy_result in enumerate(analogy_results):
        transfer_tool_pose = analogy_result['transfer_tool_pose']
        reference_radius_of_curvature = analogy_result['reference_radius_of_curvature']
        target_radius_of_curvature = analogy_result['target_radius_of_curvature']
        print('radii_of_curvature:', reference_radius_of_curvature, target_radius_of_curvature)
        success = plan_without_robot(target_scene, transfer_tool_pose, human_viewer=False)
        if with_robot:
            if success:  # no need to plan with robot if plan without robot fails
                init_grasp_center_world = analogy_result['init_grasp_center_world']
                success = plan(target_scene, transfer_tool_pose, init_grasp_center_world, human_viewer=False,
                               timeout=timeout,
                               video_save_path=f'videos/hooking_{desc}/{tool}.mp4' if save_video else None)

        print(f'---------{tool}, target half size: {target_half_size}, top {top_i}, success: {success}---------')
        if success:
            return


if __name__ == '__main__':
    tools = ['caliper', 'hanger', 'scissors', 'watch']
    target_half_sizes = [0.015, 0.020, 0.025, 0.030, 0.035]
    args = argparse.ArgumentParser()
    args.add_argument('--tool', type=str, default='caliper', help=f'choose from {tools}')
    args.add_argument('--target', type=float, default=0.025, help=f'choose from {target_half_sizes}')
    args.add_argument('--with_robot', type=int, default=1)
    args.add_argument('--timeout', type=int, default=60)
    args.add_argument('--desc', type=str, default='exp')
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--save_video', type=int, default=0)
    args.add_argument('--use_e2', type=int, default=0)
    args = args.parse_args()

    tool = args.tool
    target = args.target
    assert tool in tools, f'tool must be in {tools}'
    assert target in target_half_sizes, f'target must be in {target_half_sizes}'
    with_robot = bool(args.with_robot)
    timeout = args.timeout
    desc = args.desc
    seed = args.seed
    save_video = bool(args.save_video)
    use_e2 = bool(args.use_e2)

    main(tool=tool, target_half_size=target, np_seed=seed, with_robot=with_robot, timeout=timeout, desc=desc, save_video=save_video, use_e2=use_e2)
