import numpy as np
import sapien.core as sapien
from sapien.core import Pose, CameraEntity

from env.tableScene import TableScene
from utils.igibson_utils import load_igibson_category
from utils.rotation_utils import rpy2wxyz, rpy2xyzw
from utils.sapien_utils import load_custom_obj, load_mug, get_contacts_by_id

SKIP_MUGS_AND_REASONS = {
    4: 'Mug is solid',
    6: 'Handle is too small or no handle',
    7: 'Handle is too small and not closed',
    8: 'Mug is solid',
    11: 'Mug is solid',
    12: 'Mug is solid',
    14: 'Mug is solid',
    18: 'Mug is solid',
    20: 'Mug is solid',
    23: 'Mug is solid',
    29: 'Mug is solid',
    30: 'Mug is solid',
    31: 'Mug is solid',
    34: 'Mug is solid',
    43: 'Mug is solid',
    48: 'Mug is solid',
    49: 'Mug is solid',
    51: 'Mug is solid',
    58: 'Mug is solid',
    61: 'Mug is solid',
    62: 'Mug is solid',
    67: 'Mug is solid',
    68: 'Mug is solid',
    69: 'Mug is solid',
    74: 'Mug is solid',
    80: 'Mug is solid',
    83: 'Mug is solid',
    97: 'Mug is solid',
    98: 'Handle is too small or no handle',
    99: 'Mug is solid',
    103: 'Handle is too small or no handle',
    104: 'Mug is solid',
    105: 'Mug is solid',
    107: 'Mug is solid',
    109: 'Mug is solid',
    110: 'Mug is solid',
    112: 'Mug is solid',
    113: 'Mug is solid',
    114: 'Mug is solid',
    117: 'Mug is solid',
    119: 'Mug is solid',
    128: 'Handle is too small or no handle',
    131: 'Mug is solid',
    136: 'Mug is solid',
    139: 'Mug is solid',
    144: 'Mug is solid',
    147: 'Mug is solid',
    148: 'Handle is too small or no handle',
    154: 'Mug is solid',
    155: 'Mug is solid',
    156: 'Mug is solid',
    160: 'Mug is solid',
    163: 'Mug is solid',
    169: 'Mug is solid',
    170: 'Mug is solid',
    171: 'Mug is solid',
    173: 'Handle is too small or no handle',
    175: 'Mug is solid',
    181: 'Mug is solid',
    183: 'Mug is solid',
    184: 'Mug is solid',
    185: 'Mug is solid',
    188: 'Mug is solid',
    190: 'Mug is solid',
    191: 'Mug is solid',
    193: 'Mug is solid',
    195: 'Mug is solid'
}


class MugScene(TableScene):
    def __init__(
            self,
            mug_id: int,
            fps: float = 240.0,
            add_robot: bool = False,
            hanger_demo=False,
            mug_demo=False
    ):
        sapien.SceneConfig().default_restitution = 0.1
        sapien.SceneConfig().default_static_friction = 1
        sapien.SceneConfig().default_dynamic_friction = 1

        super().__init__(fps, add_robot)

        x, y = 0.6, 0.0
        z = 0
        additional_scale = 0.2
        additional_rotation_xyzw = rpy2xyzw([np.pi / 2, 0, 0])

        self.hanger_demo_pose = Pose([0.846, -0.060, -2.020 + 0.01], [0.786, 0.001, -0.049, 0.617])
        self.mug_demo_pose = Pose([0.614005, -0.22702, 0.322003], [-8.04311e-10, 2.40448e-10, -0.99567, 0.0929617])
        if not hanger_demo and not mug_demo:
            self.mug = load_mug(
                self.scene,
                self.renderer,
                mug_id,
                x=x,
                y=y,
                additional_scale=additional_scale,
                additional_rotation_xyzw=additional_rotation_xyzw,
                additional_height=z,
                density=500
            )

        elif mug_demo:
            self.mug = load_mug(
                self.scene,
                self.renderer,
                3,
                x=x,
                y=y,
                additional_scale=additional_scale,
                additional_rotation_xyzw=additional_rotation_xyzw,
                additional_height=z,
                density=500
            )
            self.mug.set_pose(self.mug_demo_pose)

        else:
            self.mug, _, _ = load_igibson_category(
                self.scene,
                self.renderer,
                category='hanger',
                x=x,
                y=y,
                additional_scale=0.5,
                additional_height=z,
                name='hanger',
            )
            self.mug.set_pose(self.hanger_demo_pose)

        mugtree_physical_material = self.scene.create_physical_material(3, 3, 0)
        if not hanger_demo:
            self.mugtree, _ = load_custom_obj(
                self.scene,
                self.renderer,
                'mugtree',
                x=0.7,
                y=-0.2,
                additional_height=0.3,
                additional_scale=0.5,
                color=np.array([0.8, 0.5, 0.2, 1]),
                density=10000,
                is_kinematic=False,
                physical_material=mugtree_physical_material
            )
        else:
            self.mugtree, _ = load_custom_obj(
                self.scene,
                self.renderer,
                'rack',
                x=0.7,
                y=-0.2,
                additional_height=0.3,
                additional_scale=1,
                color=np.array([0.8, 0.5, 0.2, 1]),
                density=10000,
                is_kinematic=False,
                physical_material=mugtree_physical_material
            )
            self.mug.set_pose(Pose(self.hanger_demo_pose.p + [0, 0, 0.4], self.hanger_demo_pose.q))

        self.mugtree.set_pose(Pose([0.7, -0.2, 0], rpy2wxyz([0, 0, np.pi / 2])))

        self.init_state = self.scene.pack()

    def reset(self):
        self.scene.unpack(self.init_state)

    def hide_env_visual(self):
        self.mugtree.hide_visual()
        self.ground.hide_visual()
        self.plane.hide_visual()
        if self.robot is not None:
            self.hide_robot_visual()
        self.is_hiding_env_visual = True

    def unhide_env_visual(self):
        self.mugtree.unhide_visual()
        self.ground.unhide_visual()
        self.plane.unhide_visual()
        if self.robot is not None:
            self.unhide_robot_visual()
        self.is_hiding_env_visual = False

    def check_final_pose_feasibility(self, final_pose: Pose) -> bool:
        checkpoint = self.scene.pack()
        self.mug.set_pose(final_pose)
        self.step()
        contact_between_mug_and_mugtree = get_contacts_by_id(
            self.scene,
            self.mug.get_id(),
            self.mugtree.get_id(),
            distance_threshold=1e-5
        )
        self.scene.unpack(checkpoint)
        if len(contact_between_mug_and_mugtree) > 0:
            print('Final pose is infeasible')
            return False
        else:
            return True

    def is_success(self) -> bool:
        if self.attach_drive is not None:
            self.detach_object()
            attached = True
        else:
            attached = False
        checkpoint = self.scene.pack()
        if self.robot is not None:
            init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0,
                         0]
            self.robot.set_qpos(init_qpos)
            self.robot.set_drive_target(init_qpos)
        for _ in range(540):
            if self.robot is not None:
                qf = self.robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                    external=False
                )
                self.robot.set_qf(qf)
            self.scene.step()
            self.scene.update_render()
            if self.viewer is not None:
                self.viewer.render()

        if self.mug.get_pose().p[2] > 0.2 and np.linalg.norm(self.mug.get_velocity()) < 0.5:
            success = True
        else:
            success = False

        self.scene.unpack(checkpoint)

        if self.robot is not None:
            if attached:
                self.attach_object(self.mug)
            self.robot.set_drive_target(self.robot.get_qpos())

        return success

    def add_camera(
            self,
            direction: str = '+x',
            fovy: float = None,
            width: int = 768,
            height: int = 768
    ) -> CameraEntity:
        if direction != '+x':
            return super().add_camera(direction, fovy, width, height)
        else:
            camera = super().add_camera(direction, fovy, width, height)
            additional_translation = np.array([0.6 - 0.75 * 1.414 * 3 / 4, 0, 0.1])
            camera.set_pose(Pose(camera.get_pose().p + additional_translation, camera.get_pose().q))
            return camera
