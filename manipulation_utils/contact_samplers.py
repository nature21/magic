import itertools
import json
import time
from dataclasses import dataclass
from time import time
from typing import Optional, Iterator, Sequence, Callable, cast, List, Union

import numpy as np
import open3d as o3d
import tabulate
from mani_skill2.utils.trimesh_utils import get_actor_mesh
from open3d import geometry
from sapien.core.pysapien import Scene, Actor, Pose

from env.tableScene import TableScene
from utils.mesh_utils import trimesh_to_open3d_mesh
from utils.rotation_utils import enumerate_quaternion_from_vectors, quat_mul, rotate_vector, wxyz2xyzw, xyzw2wxyz, \
    cross, find_orthogonal_vector, get_quaternion_from_axes, normalize_vector
from utils.sapien_utils import get_contacts_by_id, get_contacts_with_articulation  # , ActorSaver


@dataclass
class GraspParameter(object):
    point1: np.ndarray
    normal1: np.ndarray
    point2: np.ndarray
    normal2: np.ndarray
    ee_pos: np.ndarray
    ee_quat_wxyz: np.ndarray
    qpos: np.ndarray


def sample_grasp(
        table_scene: TableScene,
        object_id: Union[int, List[int]],
        gripper_distance: float = 0.08,
        max_intersection_distance: float = 10,
        verbose: bool = False,
        max_test_points_before_first: int = 250,
        max_test_points: int = 100000000,
        batch_size: int = 100,
        surface_pointing_tol: float = 0.9,
        min_point_distance: float = 0.0001,
        np_random: Optional[np.random.RandomState] = None,
        guidance_center: Optional[np.ndarray] = None,
        guidance_direction: Optional[np.ndarray] = None,
        guidance_radius: float = 0.01,
        start_time=None,
        timeout=60,
        exclude_ids: Optional[List[int]] = None,
        fix_grasp_center_z: Optional[float] = None,
        use_farthest_pair: bool = False,
        guidance_direction_hard: bool = False
) -> Iterator[GraspParameter]:
    """
    Given the name of the object, sample a 6D grasp pose.
    Before calling this function, make sure that the gripper is open.
    """
    scene = table_scene.scene
    robot = table_scene.robot

    if np_random is None:
        np_random = np.random

    if isinstance(object_id, int):
        mesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(object_id)))
    elif isinstance(object_id, list):
        meshes = [trimesh_to_open3d_mesh(get_actor_mesh(scene.find_articulation_link_by_link_id(obj_id))) for obj_id in object_id]
        mesh = o3d.geometry.TriangleMesh()
        for m in meshes:
            mesh += m
    else:
        raise ValueError('object_id should be either an int or a list of int.')

    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    found = False
    nr_test_points_before_first = 0

    for _ in range(int(max_test_points / batch_size)):
        # TODO: accelerate the computation.
        if guidance_center is None:
            pcd = mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        else:
            pcd: o3d.geometry.PointCloud = mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
            # get the points that are within the guidance radius
            feasible_point_indices = np.where(np.linalg.norm(np.asarray(pcd.points) - guidance_center, axis=1) < guidance_radius)[0][:batch_size]
            pcd = pcd.select_by_index(feasible_point_indices)

        indices = list(range(len(pcd.points)))
        np_random.shuffle(indices)
        for i in indices:
            if start_time is not None:
                if time() - start_time > timeout:
                    return
            if not found:
                nr_test_points_before_first += 1

            point = np.asarray(pcd.points[i])
            normal = np.asarray(pcd.normals[i])

            if guidance_direction is not None:
                if np.abs(np.dot(normal, guidance_direction)) > 1 - surface_pointing_tol:
                    continue

            if verbose:
                print('sample_grasp_v2_gen', 'point', point, 'normal', normal)

            point2 = point - normal * max_intersection_distance
            other_intersection = mesh_line_intersect(t_mesh, point2, normal, use_farthest_pair=use_farthest_pair)

            if verbose:
                print('  other_intersection', other_intersection)

            # if no intersection, try the next point.
            if other_intersection is None:
                if verbose:
                    print('  skip: no intersection')
                continue

            other_point, other_normal = other_intersection

            # if two intersection points are too close, try the next point.
            if np.linalg.norm(other_point - point) < min_point_distance:
                if verbose:
                    print('  skip: too close')
                continue

            # if the surface normals are too different, try the next point.
            if np.abs(np.dot(normal, other_normal)) < surface_pointing_tol:
                if verbose:
                    print('  skip: normal too different')
                continue

            grasp_center = (point + other_point) / 2

            if fix_grasp_center_z is not None:
                grasp_center[2] = fix_grasp_center_z

            grasp_distance = np.linalg.norm(point - other_point)
            grasp_normal = normal

            if grasp_distance > gripper_distance:
                if verbose:
                    print('  skip: too far')
                continue

            if guidance_direction_hard:
                ee_d = normalize_vector(grasp_normal - np.dot(grasp_normal, guidance_direction) * guidance_direction)
                ee_u = guidance_direction
                ee_v = cross(ee_u, ee_d)
            else:
                ee_d = grasp_normal
                # ee_u and ee_v are two vectors that are perpendicular to ee_d
                ee_u = find_orthogonal_vector(ee_d)
                ee_v = cross(ee_u, ee_d)

            if verbose:
                print(f"ee_d: {ee_d}, ee_u: {ee_u}, ee_v: {ee_v}")

            # enumerate four possible grasp orientations
            for ee_norm1 in [ee_u, ee_v, -ee_u, -ee_v]:
                ee_norm2 = cross(ee_d, ee_norm1)
                ee_quat = get_quaternion_from_axes(ee_norm2, ee_d, ee_norm1)
                ee_quat_wxyz = xyzw2wxyz(ee_quat)

                qpos, hand_pos= table_scene.grasp_center_ik(
                    grasp_center=grasp_center,
                    ee_quat_wxyz=ee_quat_wxyz,
                    start_qpos=robot.get_qpos(),
                    mask=[0, 0, 0, 0, 0, 0, 0, 1, 1], # don't change the qpos of the gripper fingers
                    threshold=1e-4,
                    exclude_ids=exclude_ids,
                    verbose=verbose
                )

                if qpos is None:
                    if verbose:
                        print('  skip: ik fail')
                    continue

                rv = collision_free_qpos(table_scene, object_id, qpos, verbose=verbose)
                if rv:
                    found = True
                    yield GraspParameter(
                        point1=point, normal1=normal,
                        point2=other_point, normal2=other_normal,
                        ee_pos=hand_pos, ee_quat_wxyz=ee_quat_wxyz,
                        qpos=qpos
                    )
                elif verbose:
                    print('    gripper pos', grasp_center)
                    print('    gripper quat', ee_quat)
                    print('    skip: collision')

        if not found and nr_test_points_before_first > max_test_points_before_first:
            if verbose:
                print(f'Failed to find a grasp after {nr_test_points_before_first} points tested.')
            return


@dataclass
class PushParameter(object):
    push_pos: np.ndarray
    push_dir: np.ndarray
    distance: float


def sample_push_with_support(
        scene: Scene,
        object_id: int,
        support_id: int,
        max_attempts: int = 1000,
        batch_size: int = 100,
        push_distance_fn: Optional[Callable] = None,
        np_random: Optional[np.random.RandomState] = None,
        verbose: bool = False
) -> Iterator[PushParameter]:
    if push_distance_fn is None:
        push_distance_fn = lambda: 0.1

    if np_random is None:
        np_random = np.random

    object_mesh: geometry.TriangleMesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(object_id)))

    nr_batches = int(max_attempts / batch_size)
    feasible_point_indices = list()
    for _ in range(nr_batches):
        pcd = object_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)

        # get the contact points between the object and the support object
        contact_normal = get_single_contact_normal(scene, object_id, support_id)

        # filter out the points that are not on the contact plane
        feasible_point_cond = np.abs(np.asarray(pcd.normals).dot(contact_normal)) < 0.1
        feasible_point_indices = np.where(feasible_point_cond)[0]

        if len(feasible_point_indices) == 0:
            continue

        np_random.shuffle(feasible_point_indices)
        rows = list()
        for index in feasible_point_indices:
            rows.append((index, pcd.points[index], -pcd.normals[index]))

        if verbose:
            print(tabulate.tabulate(rows, headers=['index', 'point', 'normal']))

        # create a new point cloud
        for index in feasible_point_indices:
            if verbose:
                print('sample_push_with_support', 'point', pcd.points[index], 'normal', -pcd.normals[index])
            yield PushParameter(np.asarray(pcd.points[index]), -np.asarray(pcd.normals[index]), push_distance_fn())

    if len(feasible_point_indices) == 0:
        raise ValueError(
            f'No feasible points for {object_id} on {support_id} after {nr_batches * batch_size} attempts.')


@dataclass
class IndirectPushParameter(object):
    object_push_pos: np.ndarray
    object_push_dir: np.ndarray

    tool_pos: np.ndarray
    tool_quat_wxyz: np.ndarray

    tool_point_pos: np.ndarray
    tool_point_normal: np.ndarray

    prepush_distance: float = 0.05
    push_distance: float = 0.1

    @property
    def total_push_distance(self):
        return self.prepush_distance + self.push_distance


def load_indirect_push_parameter(file_path: str) -> IndirectPushParameter:
    with open(file_path, 'r') as f:
        data = json.load(f)
        for key in data.keys():
            if type(data[key]) is list:
                data[key] = np.array(data[key])
        return IndirectPushParameter(**data)


def sample_indirect_push_with_support(
        scene: Scene,
        tool_id: int,
        object_id: int,
        support_id: int,
        prepush_distance: float = 0.05,
        max_attempts: int = 10000000,
        batch_size: int = 1000,
        filter_push_dir: Optional[np.ndarray] = None,
        push_distance_distribution: Sequence[float] = (0.1, 0.15),
        push_distance_sample: bool = False,
        contact_normal_tol: float = 0.01,
        np_random: Optional[np.random.RandomState] = None,
        verbose: bool = False,
        check_reachability: Callable[[np.ndarray], bool] = None,
        tool_contact_point_filter: Callable[[np.ndarray], np.ndarray] = None
) -> Iterator[IndirectPushParameter]:
    if np_random is None:
        np_random = np.random

    tool_mesh: geometry.TriangleMesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(tool_id)))
    object_mesh: geometry.TriangleMesh = trimesh_to_open3d_mesh(get_actor_mesh(scene.find_actor_by_id(object_id)))

    current_tool_pose = scene.find_actor_by_id(tool_id).get_pose()
    current_tool_pos, current_tool_quat_wxyz = current_tool_pose.p, current_tool_pose.q
    current_tool_quat_xyzw = wxyz2xyzw(current_tool_quat_wxyz)

    nr_batches = int(max_attempts / batch_size)
    contact_normal = get_single_contact_normal(scene, object_id, support_id)

    for _ in range(nr_batches):
        tool_pcd = tool_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        object_pcd = object_mesh.sample_points_uniformly(batch_size, use_triangle_normal=True)
        feasible_object_point_cond = np.abs(np.asarray(object_pcd.normals).dot(contact_normal)) < contact_normal_tol
        if filter_push_dir is not None:
            feasible_object_point_cond = np.logical_and(
                feasible_object_point_cond,
                np.asarray(object_pcd.normals, dtype=np.float32).dot(-filter_push_dir) > 0.8
            )

        feasible_object_point_indices = np.where(feasible_object_point_cond)[0]

        if tool_contact_point_filter is not None:
            # transform the pcd to world frame
            world2tool = np.linalg.inv(scene.find_actor_by_id(tool_id).pose.to_transformation_matrix())
            tool_pcd_np = np.asarray(tool_pcd.points)
            tool_pcd_homogeneous = np.concatenate([tool_pcd_np, np.ones([*tool_pcd_np.shape[:-1], 1])], axis=-1)
            tool_pcd_tool = (tool_pcd_homogeneous @ world2tool.T)[:,:-1]
            tool_contact_point_cond = tool_contact_point_filter(tool_pcd_tool)
            filtered_tool_contact_point_indices = np.where(tool_contact_point_cond)[0]
        else:
            filtered_tool_contact_point_indices = range(batch_size)

        all_index_pairs = list(itertools.product(feasible_object_point_indices, filtered_tool_contact_point_indices))
        np_random.shuffle(all_index_pairs)
        for object_index, tool_index in all_index_pairs:
            object_point_pos = np.asarray(object_pcd.points[object_index])
            object_point_normal = -np.asarray(object_pcd.normals[object_index])  # point inside

            tool_point_pos = np.asarray(tool_pcd.points[tool_index])
            tool_point_normal = np.asarray(tool_pcd.normals[tool_index])  # point outside (towards the tool)            

            # Solve for a quaternion that aligns the tool normal with the object normal
            for rotation_quat_xyzw in enumerate_quaternion_from_vectors(tool_point_normal, object_point_normal, 4):
                # This is the world coordinate for the tool point after rotation.
                new_tool_point_pos = current_tool_pos + rotate_vector(tool_point_pos - current_tool_pos,
                                                                      rotation_quat_xyzw)
                # Now compute the displacement for the tool object
                final_tool_pos = object_point_pos - new_tool_point_pos + current_tool_pos
                final_tool_pos -= object_point_normal * prepush_distance
                final_tool_quat_xyzw = quat_mul(rotation_quat_xyzw, current_tool_quat_xyzw)

                success = True
                # check collision
                init_state = scene.pack()  # backup state
                cast(Actor, scene.find_actor_by_id(tool_id)).set_pose(
                    Pose(final_tool_pos, xyzw2wxyz(final_tool_quat_xyzw)))

                scene.step()
                contacts = get_contacts_by_id(scene, tool_id)
                if len(contacts) > 0:
                    success = False
                if check_reachability is not None and success:
                    if not check_reachability(np.asarray(trimesh_to_open3d_mesh(
                            get_actor_mesh(scene.find_actor_by_id(tool_id))).sample_points_uniformly(1000,
                                                                                                     use_triangle_normal=True).points)):
                        success = False

                scene.unpack(init_state)  # reset state

                if success:
                    if push_distance_sample:
                        distances = [np_random.choice(push_distance_distribution)]
                    else:
                        distances = push_distance_distribution
                    kwargs = dict(
                        object_push_pos=object_point_pos,
                        object_push_dir=object_point_normal,
                        tool_pos=final_tool_pos,
                        tool_quat_wxyz=xyzw2wxyz(final_tool_quat_xyzw),
                        tool_point_pos=rotate_vector(tool_point_pos - current_tool_pos,
                                                     rotation_quat_xyzw) + final_tool_pos,
                        tool_point_normal=rotate_vector(tool_point_normal, rotation_quat_xyzw),
                        prepush_distance=prepush_distance
                    )
                    for distance in distances:
                        yield IndirectPushParameter(**kwargs, push_distance=distance)


def collision_free_qpos(table_scene: TableScene, object_id: int, qpos: np.ndarray, exclude: Optional[List[int]] = None, verbose: bool = False) -> bool:
    """Check whether the given qpos is collision free. The function also accepts a list of object ids to exclude (e.g., the object in hand).

    Args:
        robot: the robot.
        qpos: the qpos to check.
        exclude: the object ids to exclude.
        verbose: whether to print the collision information.

    Returns:
        True if the qpos is collision free.
    """
    # check self collision
    collisions = table_scene.planner.check_for_self_collision(state=qpos)
    if len(collisions) > 0:
        if verbose:
            print(f'  collision_free_qpos: self collision')
        return False

    # check collision with the environment
    scene = table_scene.scene
    robot = table_scene.robot
    init_state = scene.pack()  # backup state
    robot.set_qpos(qpos)

    scene.step()

    contacts = get_contacts_with_articulation(scene, robot, distance_threshold=0.0001)
    if exclude is not None:
        for c in contacts:
            if c.actor0.get_id() not in exclude and c.actor1.get_id() not in exclude:
                if verbose:
                    print(f'  collision_free_qpos: collide between {c.actor0.get_name()} and {c.actor1.get_name()}')
                scene.unpack(init_state)
                return False
    else:
        for c in contacts:
            if verbose:
                print(f'  collision_free_qpos: collide between {c.actor0.get_name()} and {c.actor1.get_name()}')
            scene.unpack(init_state)
            return False
    scene.unpack(init_state)
    return True


def mesh_line_intersect(t_mesh: o3d.t.geometry.TriangleMesh, ray_origin: np.ndarray, ray_direction: np.ndarray,
                        use_farthest_pair: bool = False) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Intersects a ray with a mesh.

    Args:
        t_mesh: the mesh to intersect with.
        ray_origin: the origin of the ray.
        ray_direction: the direction of the ray.

    Returns:
        A tuple of (point, normal) if an intersection is found, None otherwise.
    """

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    ray = o3d.core.Tensor.from_numpy(np.array(
        [[ray_origin[0], ray_origin[1], ray_origin[2], ray_direction[0], ray_direction[1], ray_direction[2]]],
        dtype=np.float32
    ))
    result = scene.cast_rays(ray)

    # no intersection.
    if result['geometry_ids'][0] == scene.INVALID_ID:
        return None

    if use_farthest_pair:
        inter_point = np.asarray(ray_origin) + np.asarray(ray_direction) * result['t_hit'][-1].item()
        inter_normal = result['primitive_normals'][-1].numpy()
    else:
        inter_point = np.asarray(ray_origin) + np.asarray(ray_direction) * result['t_hit'][0].item()
        inter_normal = result['primitive_normals'][0].numpy()
    return inter_point, inter_normal


def get_single_contact_normal(scene: Scene, object_id: int, support_id: int, deviation_tol: float = 0.05) -> np.ndarray:
    contacts = get_contacts_by_id(scene, object_id, support_id)

    if len(contacts) == 0:
        raise ValueError(
            f'No contact between {scene.find_actor_by_id(object_id).get_name()} and {scene.find_actor_by_id(support_id).get_name()}')

    contact_normals = np.array([point.normal for contact in contacts for point in contact.points])
    contact_normal_avg = np.mean(contact_normals, axis=0)
    contact_normal_avg /= np.linalg.norm(contact_normal_avg)

    deviations = np.abs(1 - contact_normals.dot(contact_normal_avg) / np.linalg.norm(contact_normals, axis=1))
    if np.max(deviations) > deviation_tol:
        raise ValueError(
            f'Contact normals of {scene.find_actor_by_id(object_id).get_name()} and {scene.find_actor_by_id(support_id).get_name()} are not consistent. This is likely due to multiple contact points.\n'
            f'  Contact normals: {contact_normals}\n  Deviations: {deviations}.'
        )

    return contact_normal_avg
