import numpy as np
from sapien.core import Pose

from env.tableScene import TableScene
from utils.igibson_utils import load_igibson_category
from utils.rotation_utils import wxyz2xyzw, rpy2wxyz
from utils.sapien_utils import create_box, create_sphere, load_custom_obj, create_cylinder


class HookScene(TableScene):
    def __init__(
            self,
            tool: str,
            target: str,
            tool_additional_pose: Pose = Pose([0.6, -0.6, 0.2], rpy2wxyz([0, 0, 0])),
            tool_additional_scale: float = 0.5,
            tool_density: float = 1000,
            target_x_y: np.ndarray = np.array([0.9, 0]),
            is_igibson_tool: bool = None,
            target_half_size: float = 0.05,
            target_e: float = 1,
            target_mu: float = 0.05,
            target_density: float = 10,
            fps: float = 240.0,
            add_robot: bool = False,
            use_disturbance: bool = True
    ):
        super().__init__(fps, add_robot)

        self.target = None
        if tool == 'watch':
            target_x_y = target_x_y - np.array([0.1, 0])
        self.target_x_y = target_x_y
        self.target_half_size = target_half_size
        if target == 'ball':
            self.target = create_sphere(
                self.scene,
                Pose(p=np.concatenate([target_x_y, [target_half_size]], axis=0)),
                radius=target_half_size,
                color=[1, 0, 0],
                name='ball',
                mu=target_mu,
                e=target_e,
                density=target_density
            )
        elif target == 'box':
            self.target = create_box(
                self.scene,
                Pose(p=np.concatenate([target_x_y, [target_half_size]], axis=0)),
                half_size=[target_half_size, target_half_size, target_half_size],
                color=[1, 0, 0],
                name='box',
                mu=target_mu,
                e=target_e,
                density=target_density
            )
        elif target == 'cylinder':
            self.target = create_cylinder(
                self.scene,
                self.renderer,
                Pose(p=np.concatenate([target_x_y, [target_half_size]], axis=0)),
                radius=target_half_size,
                half_height=target_half_size,
                color=[1, 0, 0, 1],
                name='cylinder',
                mu=target_mu,
                e=target_e,
                density=target_density
            )
        elif target == 'none':
            self.target = None
        else:
            raise ValueError(f'no such target object type {target}')

        self.table = create_box(self.scene, Pose([0.6, -0.7, 0.1]), [0.2, 0.2, 0.1], color=np.array([0.8, 0.8, 0.8, 1]),
                                density=1e4, name='table')
        if tool == 'watch':
            self.table.set_pose(Pose([0, 0.15, 0]) * self.table.get_pose())

        self.tool = None
        x, y, z = tool_additional_pose.p
        tool_physical_material = self.scene.create_physical_material(0.1, 0.1, 1)
        if tool == 'scissors':
            y = y - 0.1
        elif tool == 'watch':
            y = y + 0.2
        additional_rotation_xyzw = wxyz2xyzw(tool_additional_pose.q)
        if is_igibson_tool is None:
            if tool in ['hook', 'mug']:
                is_igibson_tool = False
            else:
                is_igibson_tool = True
        if is_igibson_tool:
            self.tool, _, _ = load_igibson_category(
                self.scene,
                self.renderer,
                category=tool,
                x=x,
                y=y,
                additional_scale=tool_additional_scale,
                additional_rotation_xyzw=additional_rotation_xyzw,
                additional_height=z,
                name=tool,
                physical_material=tool_physical_material,
                density=tool_density
            )
        else:
            self.tool, _ = load_custom_obj(
                self.scene,
                self.renderer,
                obj_name=tool,
                x=x,
                y=y,
                additional_scale=tool_additional_scale,
                additional_rotation_xyzw=additional_rotation_xyzw,
                additional_height=z,
                density=tool_density,
                physical_material=tool_physical_material,
                actor_name=tool,
            )

        self.init_state = self.scene.pack()
        self.use_disturbance = use_disturbance

    def reset(self):
        self.scene.unpack(self.init_state)

    def hide_env_visual(self):
        if self.target is not None:
            self.target.hide_visual()
        self.ground.hide_visual()
        self.plane.hide_visual()
        self.table.hide_visual()
        if self.robot is not None:
            self.hide_robot_visual()
        self.is_hiding_env_visual = True

    def unhide_env_visual(self):
        if self.target is not None:
            self.target.unhide_visual()
        self.ground.unhide_visual()
        self.plane.unhide_visual()
        self.table.unhide_visual()
        if self.robot is not None:
            self.unhide_robot_visual()
        self.is_hiding_env_visual = False

    def step(self):
        if self.use_disturbance:
            target_velocity = self.target.get_velocity()
            disturbance = np.array([0, np.random.normal(0, self.target_half_size), 0])
            self.target.set_velocity(target_velocity + disturbance)
        self.scene.step()

    def is_terminated(self) -> bool:
        if self.target.get_pose().p[0] <= self.target_x_y[0] - 0.2:
            return True
        else:
            return False

    def is_success(self) -> bool:
        target_pos = self.target.get_pose().p
        thres = 0.2 if self.tool.get_name() != 'watch' else 0.1
        if target_pos[0] <= self.target_x_y[0] - thres and np.abs(
                target_pos[1]) < 0.5 * self.target_half_size and np.linalg.norm(self.target.get_velocity()) < 0.5:
            return True
        return False
