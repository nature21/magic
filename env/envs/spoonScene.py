import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.transforms import Transform
from mplib import pymp
from sapien.core import Pose, CameraEntity, Actor

from manipulation_utils.contact_samplers import sample_grasp
from manipulation_utils.motion import gen_qpos_to_qpos_trajectory, gen_qpos_to_pose_trajectory
from manipulation_utils.controllers import TrajectoryPositionController
from env.tableScene import TableScene
from utils.mesh_utils import get_actor_mesh
from utils.rotation_utils import quaternion_from_vectors, quat2matrix, rpy2xyzw, quat_mul_wxyz, rpy2wxyz, rotate_vector, \
    wxyz2xyzw
from utils.sapien_utils import load_spoon, load_custom_obj, create_sphere, create_box, get_contacts_by_id


def pixel_to_3d(pixel, img_size_y, plt_transform, to_3D, debug=False):
    # pixel_array = np.asarray(pixel)
    ## there is a bug, we have to convert the pixel to int, I really don't know what is going on
    pixel_array = np.around(np.asarray(pixel)).astype(int)
    ndim = pixel_array.ndim
    pixel_array = np.atleast_2d(pixel_array)
    pixel_array[:, 1] = img_size_y - pixel_array[:, 1] - 1
    pixel_array = plt_transform.inverted().transform(pixel_array)
    pixel_array = np.hstack([
        pixel_array,
        np.zeros((pixel_array.shape[0], 1)),
        np.ones((pixel_array.shape[0], 1))
    ])
    point_3d = np.dot(to_3D, pixel_array.T).T[:, :3]
    if ndim == 1:
        point_3d = point_3d[0]
    return point_3d


class SpoonScene(TableScene):
    def __init__(
            self,
            spoon_id: int,
            fps: float = 240.0,
            add_robot: bool = False,
            radius = 0.035
    ):
        super().__init__(fps, add_robot)
        if add_robot:
            current_qpos = self.robot.get_qpos()
            current_ee_pose = self.end_effector.get_pose()
            new_ee_pose = Pose(current_ee_pose.p + np.array([0, 0, 0.1]), current_ee_pose.q)
            new_qpos = self.ee_ik(new_ee_pose, current_qpos, return_closest=True)
            self.robot.set_qpos(new_qpos)
            self.set_drive_target(new_qpos)

        self.table = create_box(self.scene, Pose([0.57, -0.4, 0.1]), [0.15, 0.1, 0.1], color=np.array([0.8, 0.8, 0.8, 1]), density=1e4, name='table')

        x, y = 0.5, -0.35
        z = 0.2
        additional_scale = 2.5
        spoon_physical_material = self.scene.create_physical_material(0.5, 0.5, 0.1)
        self.spoon: Actor = load_spoon(
            self.scene,
            self.renderer,
            spoon_id,
            x=x,
            y=y,
            additional_scale=additional_scale,
            additional_height=z,
            physical_material=spoon_physical_material
        )

        bowl_physical_material = self.scene.create_physical_material(0.1, 0.1, 0.1)

        self.arc_slope, _ = load_custom_obj(
            self.scene,
            self.renderer,
            'arc_slope',
            0.5,
            0.0,
            density=5e2,
            color=np.array([0.8, 0.8, 0.8, 1]),
            physical_material=bowl_physical_material,
        )
        self.arc_slope.set_pose(Pose([0.5, 0.05, 0]))

        self.ball = create_sphere(
            self.scene,
            # Pose(p=[0.5,0,0.045]),
            Pose(p=[0.5,0,0.035]),
            radius=radius,
            color=[1, 0, 0],
            density=1,
            mu=0.1,
            e=0.0,
            name='ball'
        )

        self.init_state = self.scene.pack()

    def reset_ball(self):
        # we need this function because the ball will have some observable velocity due to noisy simulation
        self.ball.set_pose(Pose(p=[0.5,0,0.045]))
        self.ball.set_velocity(np.zeros(6))
        self.ball.set_angular_velocity(np.zeros(3))

    def reset(self):
        self.scene.unpack(self.init_state)

    def is_success(self) -> bool:
        if (self.ball.get_pose().p[2] > 0.2 and
                len(get_contacts_by_id(self.scene, self.spoon.get_id(), self.ball.get_id())) > 0 and
                np.linalg.norm(self.ball.get_velocity()) < 0.1 and
                np.linalg.norm(self.spoon.get_velocity()) < 0.1
        ):
            return True
        else:
            return False

    def add_camera(
            self,
            direction: str = '+x',
            fovy: float = None,
            width: int = 768,
            height: int = 768,
    ) -> CameraEntity:
        additional_translation = np.array([0.45, 0, 0])
        camera = super().add_camera(direction, fovy, width, height)
        if direction == '+y':
            camera = super().add_camera(direction, fovy, width, height)
            # camera.near = 0.25
            # camera.far = 20
            pos = camera.get_pose().p
            pos = pos + additional_translation
            pos[1] = 0.3
            pos[2] = 0.02
            camera.set_pose(Pose(pos, camera.get_pose().q))
        elif direction == '+z':
            additional_translation += np.array([0, 0, -0.5])
            camera.set_pose(Pose(camera.get_pose().p + additional_translation, camera.get_pose().q))
        else:
            camera.set_pose(Pose(camera.get_pose().p + additional_translation, camera.get_pose().q))
        return camera

    def get_slice(self, plane_origin=np.array([0, -0.35, 0]), plane_normal=np.array([0, -1, 0])) \
            -> tuple[Image.Image, callable]:
        spoon_mesh = get_actor_mesh(self.spoon)
        slice_3d = spoon_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        to_2D = quat2matrix(quaternion_from_vectors(plane_normal, [0, 0, 1]), homogeneous=True)
        slice_2d, to_3d = slice_3d.to_planar(to_2D=to_2D)

        x_min = np.min(slice_2d.vertices[:, 0])
        x_max = np.max(slice_2d.vertices[:, 0])
        y_min = np.min(slice_2d.vertices[:, 1])
        y_max = np.max(slice_2d.vertices[:, 1])

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(7.68, 7.68))
        img_size_y = 768
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.invert_xaxis()
        ax.axis('off')

        for polygon in slice_2d.polygons_full:
            x, y = polygon.exterior.xy
            ax.fill(x, y, color=(1, 0.8, 0, 1), linewidth=1)

        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        plt_transform: Transform = ax.transData
        plt.close()
        plt.style.use('default')

        pixel_to_3d_fn = lambda pixel: pixel_to_3d(pixel, img_size_y, plt_transform, to_3d)

        return img, pixel_to_3d_fn


def get_reference_traj():
    demo_scene = SpoonScene(
        spoon_id=0,
        fps=240,
    )
    init_spoon_q = demo_scene.spoon.get_pose().q
    stage_1_spoon_q = quat_mul_wxyz(rpy2wxyz(np.array([-np.pi / 2, np.pi / 2, 0])), init_spoon_q)
    delta_y = 0.01132
    delta_z = 0.02357
    z_offset = 0.008
    start_spoon_p = np.array([0.51, delta_y-0.1, delta_z+z_offset])
    start_spoon_pose = Pose(start_spoon_p, stage_1_spoon_q)
    demo_scene.spoon.set_pose(start_spoon_pose)

    # stage 1
    num_steps_stage_1 = 120
    intermediate_spoon_p = start_spoon_p+np.array([0, 0.15, 0])
    stage_1_q_array = np.tile(stage_1_spoon_q, (num_steps_stage_1 + 1, 1))
    stage_1_p_array = np.array([
        start_spoon_p + (intermediate_spoon_p - start_spoon_p) * i / num_steps_stage_1
        for i in range(num_steps_stage_1 + 1)
    ])

    # stage 2
    num_steps_stage_2 = 480
    omega = np.array([np.pi / 2, 0, 0]) / num_steps_stage_2
    stage_2_q_array = np.array([
        quat_mul_wxyz(rpy2wxyz(omega * i), stage_1_spoon_q) for i in range(num_steps_stage_2 + 1)
    ])

    contact_omega = np.arccos(0.04/0.2) / num_steps_stage_2
    contact_point_p_array = np.array([
        [
            0.5,
            0.05+(0.2-z_offset)*np.sin(contact_omega*i),
            0.2-(0.2-z_offset)*np.cos(contact_omega*i)
        ]
        for i in range(num_steps_stage_2 + 1)
    ])

    stage_2_p_array = np.array([
        contact_point_p_array[i] + rotate_vector(intermediate_spoon_p-contact_point_p_array[0], rpy2xyzw(omega * i))
        for i in range(num_steps_stage_2 + 1)
    ])


    stage_3_num_steps = 240
    stage_3_p_array = np.array([
        stage_2_p_array[-1] + np.array([0, 0, 0.05]) * i / stage_3_num_steps for i in range(stage_3_num_steps+1)
    ])

    stage_3_q_array = np.tile(stage_2_q_array[-1], (stage_3_num_steps+1, 1))

    total_p_array = np.concatenate([stage_1_p_array, stage_2_p_array[1:], stage_3_p_array[1:]], axis=0)
    total_q_array = np.concatenate([stage_1_q_array, stage_2_q_array[1:], stage_3_q_array[1:]], axis=0)

    return total_p_array, total_q_array

