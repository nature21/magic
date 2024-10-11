import json
import sys
from os import path as osp
from typing import List, Union
from typing import Optional, cast

import numpy as np
import open3d as o3d
import pandas as pd
import sapien.core as sapien
from sapien.core import Scene, Contact, Pose, Actor, ActorBase, PhysicalMaterial, RenderMaterial, SapienRenderer, \
    ArticulationBase

from utils.mesh_utils import trimesh_to_open3d_mesh, get_actor_mesh, get_articulation_meshes, merge_meshes
from utils.rotation_utils import rpy, quat_mul, xyzw2wxyz


def create_box(
        scene: sapien.Scene,
        pose: sapien.Pose,
        half_size,
        color=None,
        density=1000,
        mu=0.6,
        e=0.1,
        name='',
        only_visual=False
) -> sapien.Actor:
    """Create a box actor in the scene."""
    builder = scene.create_actor_builder()
    if not only_visual:
        material = scene.create_physical_material(mu, mu, e)
        builder.add_box_collision(half_size=half_size, density=density, material=material)
    builder.add_box_visual(half_size=half_size, color=color)
    if only_visual:
        actor = builder.build_static(name=name)
    else:
        actor = builder.build(name=name)
    actor.set_pose(pose)
    return actor


def create_cylinder(
        scene: sapien.Scene,
        renderer: SapienRenderer,
        pose: sapien.Pose,
        radius: float,
        half_height: float = None,
        color=None,
        density=1000,
        mu=0.6,
        e=0.1,
        name: str = '',
        only_visual=False
) -> sapien.Actor:
    """Create a cylinder actor in the scene."""
    builder = scene.create_actor_builder()
    if not only_visual:
        material = scene.create_physical_material(mu, mu, e)
        builder.add_multiple_collisions_from_file(
            'assets/custom_obj/cylinder/cylinder_vhacd.obj',
            scale=np.array([radius] * 2 + [half_height]),
            density=density,
            material=material
        )
    visual_material = renderer.create_material()
    visual_material.set_base_color(color)
    builder.add_visual_from_file(
        'assets/custom_obj/cylinder/cylinder_vhacd.obj',
        scale=np.array([radius] * 2 + [half_height]),
        material=visual_material
    )
    if only_visual:
        obj = builder.build_static(name=name)
    else:
        obj = builder.build(name=name)
    obj.set_pose(pose)
    return obj


def create_sphere(
        scene: sapien.Scene,
        pose: sapien.Pose,
        radius,
        color=None,
        density=1000,
        mu=0.6,
        e=0.1,
        name='',
        only_visual=False
) -> sapien.Actor:
    """Create a sphere actor in the scene."""
    builder = scene.create_actor_builder()
    if not only_visual:
        material = scene.create_physical_material(mu, mu, e)
        builder.add_sphere_collision(radius=radius, density=density, material=material)
    builder.add_sphere_visual(radius=radius, color=color)
    if only_visual:
        actor = builder.build_static(name=name)
    else:
        actor = builder.build(name=name)
    actor.set_pose(pose)

    return actor


def load_obj_from_file(
        scene: Scene,
        collision_file: str,
        pose: Pose,
        name: str,
        scale: float,
        only_visual: bool = False,
        visual_file: str = None,
        is_kinematic: bool = False,
        is_static: bool = False,
        density: float = 1000,
        physical_material: PhysicalMaterial = None,
        render_material: RenderMaterial = None
) -> ActorBase:
    assert not (is_static and is_kinematic), 'an obj cannot be static and kinematic the same time'

    builder = scene.create_actor_builder()
    # check whether scale is a scalar
    if isinstance(scale, (int, float)):
        scale = np.array([scale] * 3)

    # add collision
    if not only_visual:
        builder.add_multiple_collisions_from_file(
            collision_file,
            scale=scale,
            density=density,
            material=physical_material,
        )

    # add visual
    if visual_file is None:
        visual_file = collision_file
    builder.add_visual_from_file(
        visual_file,
        scale=scale,
        material=render_material
    )

    if is_static:
        obj = builder.build_static(name=name)
    elif is_kinematic:
        obj = builder.build_kinematic(name=name)
    else:
        obj = builder.build(name=name)

    obj.set_pose(pose)

    return obj


def get_custom_object_dir() -> str:
    return osp.join(osp.dirname(__file__), '../assets/custom_obj')


def get_shapenet_mug_dir() -> str:
    return osp.join(osp.dirname(__file__), '../assets/shapenet_mug')


def get_spoon_dir() -> str:
    return osp.join(osp.dirname(__file__), '../assets/spoons')


def get_custom_metadata() -> pd.DataFrame:
    return pd.read_csv(osp.join(get_custom_object_dir(), 'custom_obj_data.csv'))


def get_spoon_metadata() -> dict:
    with open(osp.join(get_spoon_dir(), 'metadata_auto.json'), 'r') as f:
        return json.load(f)


def get_spoon_data_custom() -> pd.DataFrame:
    return pd.read_csv(osp.join(get_spoon_dir(), 'spoon_data.csv'))


def parse_tuple(s):
    if s.startswith('('):
        s = s[1:]
    if s.endswith(')'):
        s = s[:-1]
    return np.array(tuple(float(x) for x in s.split(',')))


def load_custom_obj(
        scene: Scene,
        renderer: SapienRenderer,
        obj_name: str,
        x: float, y: float,
        additional_scale: float = 1.0,
        additional_rotation_xyzw: Optional[np.ndarray] = None,
        additional_height: float = 0.,
        actor_name: Optional[str] = None,
        color: np.ndarray = None,
        density: float = 1000,
        physical_material: PhysicalMaterial = None,
        is_kinematic: bool = False
) -> tuple[ActorBase, dict]:
    object_data_custom = get_custom_metadata()
    obj_file = osp.join(get_custom_object_dir(), obj_name, f'{obj_name}_vhacd.obj')
    obj_visual_file = osp.join(get_custom_object_dir(), obj_name, f'{obj_name}.obj')

    this_object_data = object_data_custom[object_data_custom['name'] == obj_name]
    if len(this_object_data) > 0:
        this_object_data = this_object_data.iloc[0]
        print('Found object data:', this_object_data, file=sys.stderr)
        if not np.isnan(this_object_data['scale']):
            additional_scale = additional_scale * this_object_data['scale']
            print('Using additional scale:', additional_scale, file=sys.stderr)
        if isinstance(this_object_data['rotation'], str):
            additional_rotation2 = rpy(*parse_tuple(this_object_data['rotation']))
            if additional_rotation_xyzw is None:
                additional_rotation_xyzw = additional_rotation2
            else:
                additional_rotation_xyzw = quat_mul(additional_rotation2, additional_rotation_xyzw)
            print('Using additional rotation:', additional_rotation_xyzw, file=sys.stderr)
        if not np.isnan(this_object_data['height']):
            additional_height = additional_height + this_object_data['height']
            print('Using additional height:', additional_height, file=sys.stderr)
    else:
        this_object_data = None

    pos = np.array([x, y, additional_height])

    rotation = additional_rotation_xyzw if additional_rotation_xyzw is not None else np.array([0, 0, 0, 1])

    if color is None:
        color = [1, 0.8, 0, 1]
    render_material = renderer.create_material()
    render_material.set_base_color(color)

    if actor_name is None:
        actor_name = obj_name
    body = load_obj_from_file(
        scene=scene,
        collision_file=obj_file,
        pose=Pose(pos, xyzw2wxyz(rotation)),
        name=actor_name,
        scale=additional_scale,
        render_material=render_material,
        density=density,
        visual_file=obj_visual_file,
        is_kinematic=is_kinematic,
        physical_material=physical_material
    )

    pcd = get_actor_pcd(body, 1000)
    pcd_mean = pcd.mean(axis=0)
    pcd_min = pcd.min(axis=0)  # z-axis min

    print('PCD mean:', pcd_mean, file=sys.stderr)
    print('PCD min:', pcd_min, file=sys.stderr)

    pos = np.array((x, y, additional_height + (pcd_mean[2] - pcd_min[2])))
    new_pose = 2 * pos - pcd.mean(axis=0)

    cast(Actor, body).set_pose(Pose(new_pose, xyzw2wxyz(rotation)))

    return body, this_object_data


def load_spoon(
        scene: Scene,
        renderer: SapienRenderer,
        spoon_id: int,
        x: float, y: float,
        additional_scale: float = 1.0,
        additional_rotation_xyzw: Optional[np.ndarray] = None,
        additional_height: float = 0.,
        actor_name: Optional[str] = 'spoon',
        color: np.ndarray = None,
        density: float = 1000,
        is_kinematic: bool = False,
        physical_material: PhysicalMaterial = None
) -> ActorBase:
    object_data_auto = get_spoon_metadata()
    object_data_custom = get_spoon_data_custom()

    obj_file = osp.join(get_spoon_dir(), f'{spoon_id:02d}', 'model_vhacd.obj')
    visual_file = osp.join(get_spoon_dir(), f'{spoon_id:02d}', 'model.obj')

    this_metadata = object_data_auto[f'{spoon_id:02d}'] if f'{spoon_id:02d}' in object_data_auto else {}
    this_object_data = object_data_custom[object_data_custom['mug_id'] == spoon_id]

    if len(this_object_data) > 0:
        this_object_data = this_object_data.iloc[0]
        print('Found object data:', this_object_data, file=sys.stderr)
        if not np.isnan(this_object_data['scale']):
            additional_scale = additional_scale * this_object_data['scale']
            print('Using additional scale:', additional_scale, file=sys.stderr)
        if isinstance(this_object_data['rotation'], str):
            additional_rotation2 = rpy(*parse_tuple(this_object_data['rotation']))
            if additional_rotation_xyzw is None:
                additional_rotation_xyzw = additional_rotation2
            else:
                additional_rotation_xyzw = quat_mul(additional_rotation2, additional_rotation_xyzw)
            print('Using additional rotation:', additional_rotation_xyzw, file=sys.stderr)
        # if not np.isnan(this_object_data['height']):
        #     additional_height = additional_height + this_object_data['height']
        #     print('Using additional height:', additional_height, file=sys.stderr)

    # pos = np.array([x, y, this_metadata['height'] * additional_scale + additional_height])
    pos = np.array([x, y, additional_height])

    rotation = this_metadata['rotation']
    print('Original rotation:', rotation, file=sys.stderr)
    if additional_rotation_xyzw is not None:
        rotation = quat_mul(additional_rotation_xyzw, rotation)
    print('New rotation:', rotation, file=sys.stderr)

    if color is None:
        color = [1, 0.8, 0, 1]
    render_material = renderer.create_material()
    render_material.set_base_color(color)

    body = load_obj_from_file(
        scene=scene,
        collision_file=obj_file,
        visual_file=visual_file,
        pose=Pose(pos, xyzw2wxyz(rotation)),
        name=actor_name,
        scale=this_metadata['scale'] * additional_scale,
        render_material=render_material,
        density=density,
        is_kinematic=is_kinematic,
        physical_material=physical_material
    )

    pcd = get_actor_pcd(body, 1000)
    pcd_mean = pcd.mean(axis=0)
    pcd_min = pcd.min(axis=0)  # z-axis min

    print('PCD mean:', pcd_mean, file=sys.stderr)
    print('PCD min:', pcd_min, file=sys.stderr)

    pos = np.array((x, y, additional_height + (pcd_mean[2] - pcd_min[2])))
    new_pose = 2 * pos - pcd.mean(axis=0)

    cast(Actor, body).set_pose(Pose(new_pose, xyzw2wxyz(rotation)))
    return body


def load_mug(
        scene: Scene,
        renderer: SapienRenderer,
        mug_id: int,
        x: float, y: float,
        additional_scale: float = 1.0,
        additional_rotation_xyzw: Optional[np.ndarray] = None,
        additional_height: float = 0.,
        actor_name: Optional[str] = None,
        density=1000,
        color: np.ndarray = None,
        is_kinematic: bool = False
) -> ActorBase:
    pos = np.array([x, y, additional_height])

    rotation = additional_rotation_xyzw if additional_rotation_xyzw is not None else np.array([0, 0, 0, 1])

    if color is None:
        color = [1, 0.8, 0, 1]
    render_material = renderer.create_material()
    render_material.set_base_color(color)

    if actor_name is None:
        actor_name = f'mug_{mug_id}'

    obj_file = osp.join(get_shapenet_mug_dir(), f'{mug_id:03d}', f'model_normalized_vhacd.obj')
    obj_visual_file = osp.join(get_shapenet_mug_dir(), f'{mug_id:03d}', f'model_normalized.obj')
    body = load_obj_from_file(
        scene=scene,
        collision_file=obj_file,
        pose=Pose(pos, xyzw2wxyz(rotation)),
        name=actor_name,
        scale=additional_scale,
        render_material=render_material,
        visual_file=obj_visual_file,
        is_kinematic=is_kinematic,
        density=density
    )

    pcd = get_actor_pcd(body, 1000)
    pcd_mean = pcd.mean(axis=0)
    pcd_min = pcd.min(axis=0)  # z-axis min

    print('PCD mean:', pcd_mean, file=sys.stderr)
    print('PCD min:', pcd_min, file=sys.stderr)

    pos = np.array((x, y, additional_height + (pcd_mean[2] - pcd_min[2])))
    new_pose = 2 * pos - pcd.mean(axis=0)

    cast(Actor, body).set_pose(Pose(new_pose, xyzw2wxyz(rotation)))

    return body


def get_contacts_by_id(scene: Scene, object1_id: int, object2_id: Optional[int] = None,
                       distance_threshold: float = 0.002) -> List[Contact]:
    """Remember to call scene.step() before calling this function to update the contacts."""
    all_contacts = scene.get_contacts()
    contacts = []
    if object2_id is not None:
        contact12 = False
        contact21 = False
    for c in all_contacts:
        if c.actor0.get_id() == c.actor1.get_id():
            raise ValueError(f"object {c.actor0.get_id()} has self contact")
        else:
            # reject the contacts that aren't close enough
            is_valid = False
            for point in c.points:
                if point.separation < distance_threshold:
                    is_valid = True

            if not is_valid:
                continue

        if object2_id is not None:
            if c.actor0.get_id() == object1_id and c.actor1.get_id() == object2_id:
                contact12 = True
                if contact21:
                    raise ValueError(
                        f"more than one contact relation is found for object {object1_id} and object {object2_id}")
                contacts.append(c)
            elif c.actor0.get_id() == object2_id and c.actor1.get_id() == object1_id:
                contact21 = True
                # TODO(Yuyao Liu @ 2024/03/26): Not sure how to reverse them
                if contact12:
                    raise ValueError(
                        f"more than one contact relation is found for object {object1_id} and object {object2_id}")
                contacts.append(c)
            else:
                pass
        else:
            if c.actor0.get_id() == object1_id or c.actor1.get_id() == object1_id:
                contacts.append(c)

    return contacts


def get_contacts_with_articulation(scene: Scene, articulation: sapien.Articulation,
                                   distance_threshold: float = 0.002) -> List[Contact]:
    """Remember to call scene.step() before calling this function to update the contacts."""
    all_contacts = scene.get_contacts()
    contacts = []
    for c in all_contacts:
        if c.actor0.get_id() == c.actor1.get_id():
            raise ValueError(f"object {c.actor0.get_id()} has self contact")
        else:
            # reject the contacts that aren't close enough
            is_valid = False
            for point in c.points:
                if point.separation < distance_threshold:
                    is_valid = True

            if not is_valid:
                continue

        articulation_id_set = set([link.get_id() for link in articulation.get_links()])
        identifier0 = 1 if c.actor0.get_id() in articulation_id_set else 0
        identifier1 = 1 if c.actor1.get_id() in articulation_id_set else 0
        if identifier0 + identifier1 == 1:
            if ((c.actor0.get_name == 'panda_link0' and c.actor1.get_name == 'ground') or
                    (c.actor1.get_name == 'panda_link0' and c.actor0.get_name == 'ground')):
                continue
            contacts.append(c)
        else:
            pass

    return contacts


def get_actor_pcd(actor: ActorBase, num_points: int, to_world_frame: bool = True, return_o3d: bool = False,
                  visual=False) -> Union[np.ndarray, o3d.geometry.PointCloud]:
    o3d_pcd = trimesh_to_open3d_mesh(get_actor_mesh(actor, to_world_frame, visual)).sample_points_uniformly(num_points,
                                                                                                            use_triangle_normal=True)
    if return_o3d:
        return o3d_pcd
    else:
        return np.asarray(o3d_pcd.points)


def get_articulation_pcd(articulation: ArticulationBase, num_points: int, return_o3d: bool = False, visual=False) -> \
Union[np.ndarray, o3d.geometry.PointCloud]:
    o3d_pcd = trimesh_to_open3d_mesh(
        merge_meshes(get_articulation_meshes(articulation, visual=visual))).sample_points_uniformly(num_points,
                                                                                                    use_triangle_normal=True)
    if return_o3d:
        return o3d_pcd
    else:
        return np.asarray(o3d_pcd.points)

