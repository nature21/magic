import os.path as osp
import sys
from typing import Optional, List, cast

import jacinle
import jacinle.io as io
import numpy as np
import pandas as pd
from sapien.core import PhysicalMaterial
from sapien.core.pysapien import Scene, Pose, Actor, ActorBase, SapienRenderer

from utils.rotation_utils import rpy, quat_mul, xyzw2wxyz
from utils.sapien_utils import load_obj_from_file, get_actor_pcd


def get_igibson_dir() -> str:
    return osp.join(osp.dirname(__file__), '../assets/ig/processed_dataset')


def get_igibson_metafile(filename: str) -> str:
    return osp.join(get_igibson_dir(), filename)


@jacinle.cached_result
def get_igibson_categories() -> tuple[str, ...]:
    # names = (
    #     'caliper', 'chopstick', 'calculator', 'comb', 'dipper', 'document', 'hammer', 'lollipop', 'lipstick', 'honing_steel',
    #     'pen', 'pencil', 'scissors', 'spatula', 'soup_ladle', 'spoon', 'tablefork', 'toothbrush', 'toasting_fork', 'watch',
    #     'tape', 'highlighter', 'plate', 'medicine', 'platter', 'razor', 'screwdriver', 'scrub_brush', 'bell', 'hairbrush',
    #     'hanger', 'scraper', 'walnut'
    # )
    names = (
        'caliper', 'chopstick', 'comb', 'dipper', 'hammer', 'lollipop', 'lipstick', 'honing_steel',
        'pen', 'pencil', 'scissors', 'spatula', 'soup_ladle', 'spoon', 'tablefork', 'toothbrush', 'toasting_fork',
        'watch',
        'tape', 'highlighter', 'plate', 'medicine', 'platter', 'razor', 'screwdriver', 'scrub_brush', 'bell',
        'hairbrush',
        'hanger', 'scraper'
    )
    return tuple(sorted(names))


@jacinle.cached_result
def get_igibson_object_data_auto() -> dict:
    """This file returns an automatically generated metadata file for each object in iGibson, including its canonical scale and rotation."""
    return io.load(get_igibson_metafile('igibson_object_data_auto.json'))


@jacinle.cached_result
def get_igibson_object_data_custom() -> pd.DataFrame:
    """Return a pandas DataFrame containing the customized object data. For some object files, this metadata contains additional information about
    its scale, rotation, and height. This data should be jointly used with the metadata returned by :func:`get_igibson_metadata()`."""
    return pd.read_csv(get_igibson_metafile('igibson_object_data.csv'))


def get_igibson_hook_categories(split: Optional[str] = None) -> tuple[str, ...]:
    names = ['caliper', 'hairbrush', 'hammer', 'lollipop', 'soup_ladle']

    if split is not None:
        if split == 'train':
            names = ['caliper', 'hairbrush', 'hammer', 'lollipop']
        elif split == 'test':
            names = ['soup_ladle']
        else:
            raise ValueError(f'Unknown split: {split}.')
    return tuple(sorted(names))


def find_igibson_object_by_category(category: str) -> List[str]:
    path = osp.join(get_igibson_dir(), category)
    all_files = io.lsdir(path, '*/*vhacd.obj')
    assert len(all_files) > 0, f'No obj files found in {path}'
    return all_files


def load_igibson_category(
        scene: Scene,
        renderer: SapienRenderer,
        category: str,
        x: float, y: float,
        additional_scale: float = 1.0,
        additional_rotation_xyzw: Optional[np.ndarray] = None,
        additional_height: float = 0.,
        name: Optional[str] = None,
        color: np.ndarray = None,
        density: float = 1000,
        physical_material: Optional[PhysicalMaterial] = None,
        is_kinematic: bool = False
) -> tuple[ActorBase, dict, dict]:
    object_data_auto = get_igibson_object_data_auto()
    object_data_custom = get_igibson_object_data_custom()

    obj_files = find_igibson_object_by_category(category)
    obj_file = obj_files[0]
    visual_file = obj_file.replace('_vhacd', '')

    print('Loading object from', obj_file, file=sys.stderr)

    instance_name = obj_file.split('/')[-2]
    # this_metadata = object_data_auto[f'objects/{category}/{instance_name}/{instance_name}.urdf']
    this_metadata_key = f'objects/{category}/{instance_name}/{instance_name}.urdf'
    this_metadata = object_data_auto[this_metadata_key] if this_metadata_key in object_data_auto else {}
    # Load the custom metadata.
    this_object_data = object_data_custom[
        (object_data_custom['name'] == category) | (
                object_data_custom['filename'] == f'objects/{category}/{instance_name}/{instance_name}.urdf')
        ]
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

    pos = np.array([x, y, this_metadata['height'] * additional_scale + additional_height])

    rotation = this_metadata['transform_quat']
    print('Original rotation:', rotation, file=sys.stderr)
    if additional_rotation_xyzw is not None:
        rotation = quat_mul(additional_rotation_xyzw, rotation)
    print('New rotation:', rotation, file=sys.stderr)

    if color is None:
        color = [1, 0.8, 0, 1]
    render_material = renderer.create_material()
    render_material.set_base_color(color)

    if category == 'caliper':
        scale = np.array([this_metadata['scale'] * additional_scale,
                          this_metadata['scale'] * additional_scale,
                          this_metadata['scale'] * additional_scale * 3])
    else:
        scale = this_metadata['scale'] * additional_scale
    body = load_obj_from_file(
        scene=scene,
        collision_file=obj_file,
        visual_file=visual_file,
        pose=Pose(pos, xyzw2wxyz(rotation)),
        name=category,
        scale=scale,
        render_material=render_material,
        density=density,
        physical_material=physical_material,
        is_kinematic=is_kinematic
    )

    pcd = get_actor_pcd(body, 1000)
    pcd_mean = pcd.mean(axis=0)
    pcd_min = pcd.min(axis=0)  # z-axis min

    print('PCD mean:', pcd_mean, file=sys.stderr)
    print('PCD min:', pcd_min, file=sys.stderr)

    pos = np.array((x, y, additional_height + (pcd_mean[2] - pcd_min[2])))
    new_pose = 2 * pos - pcd.mean(axis=0)

    cast(Actor, body).set_pose(Pose(new_pose, xyzw2wxyz(rotation)))

    rv_metadata = {
        'name': category,
        'obj_file': obj_file,
        'translation': pcd.mean(axis=0) - pos,
        'rotation': rotation,
        'scale': float(this_metadata['scale']),
        'pos': pos
    }

    return body, rv_metadata, this_object_data


def parse_tuple(s):
    if s.startswith('('):
        s = s[1:]
    if s.endswith(')'):
        s = s[:-1]
    return np.array(tuple(float(x) for x in s.split(',')))


if __name__ == '__main__':
    import sapien.core as sapien
    from sapien.utils import Viewer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default=None)
    args = parser.parse_args()

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    fps = 2400
    scene = engine.create_scene()
    scene.set_timestep(1 / fps)

    material = scene.create_physical_material(0.6, 0.6, 0.1)  # Create a physical material
    ground = scene.add_ground(altitude=0, render_half_size=[10, 10], material=material)  # Add a ground

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer, resolutions=(768, 768))  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=1.2, y=0.25, z=0.4)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot: sapien.Articulation = loader.load("SAPIEN-tutorial/assets/panda/panda.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    # Set initial joint positions
    init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
    robot.set_qpos(init_qpos)

    if args.category is not None:
        categories = [args.category]
    else:
        categories = get_igibson_categories()

    categories = (category for category in categories)

    step = 0
    ig_obj = None
    while not viewer.closed:  # Press key q to quit
        qf = robot.compute_passive_force(
            gravity=True,
            coriolis_and_centrifugal=True,
            external=False
        )
        robot.set_qf(qf)

        scene.step()
        scene.update_render()  # Update the world to the renderer
        viewer.render()

        step += 1
        category = None
        if step % 480 == 1:
            if ig_obj is not None:
                scene.remove_actor(ig_obj)
            try:
                category = next(categories)
            except StopIteration:
                break
            ig_obj, _, _ = load_igibson_category(scene, renderer, category, 0.5, 0)
        if step % 480 == 0:
            picture = viewer.window.get_float_texture(viewer.target_name)
            from PIL import Image
            import os

            os.makedirs('image_scene/ig', exist_ok=True)
            Image.fromarray((picture.clip(0, 1) * 255).astype(np.uint8)).save(f'image_scene/ig/{category}.png')
