import os
from typing import Optional, List, Union

import mplib
import numpy as np
import open3d as o3d
from PIL import Image
from mplib import pymp
from mplib.pymp.collision_detection import fcl
from sapien.core import Actor
from sapien.core import Engine, SapienRenderer, CameraEntity, Pose, URDFLoader, Articulation
from sapien.core import PhysicalMaterial, Drive
from sapien.utils import Viewer

from utils.camera_utils import get_depth_img, get_rgba_img
from utils.common_utils import generate_random_string
from utils.mesh_utils import get_actor_mesh, get_articulation_meshes, merge_meshes
from utils.rotation_utils import quat_mul_wxyz, rpy2wxyz, rotate_vector, wxyz2xyzw
from utils.sapien_utils import get_actor_pcd, create_box, get_contacts_by_id

DIRECTION2POSE = {
    '+y+z': Pose(p=[0.0, 0.75, 0.75], q=quat_mul_wxyz(rpy2wxyz([0, 0, -np.pi / 2]), rpy2wxyz([0, np.arctan2(1, 1), 0]))),
    '+y-z': Pose(p=[0.0, 0.75, -0.75], q=quat_mul_wxyz(rpy2wxyz([0, 0, -np.pi / 2]), rpy2wxyz([0, np.arctan2(-1, 1), 0]))),
    '+x+z': Pose(p=[0.75, 0.0, 0.75], q=quat_mul_wxyz(rpy2wxyz([0, np.arctan2(1, -1), 0]), rpy2wxyz([np.pi, 0, 0]))),
    '+x-z': Pose(p=[0.75, 0.0, -0.75], q=quat_mul_wxyz(rpy2wxyz([0, np.arctan2(-1, -1), 0]), rpy2wxyz([np.pi, 0, 0]))),
    '-x+z': Pose(p=[-0.75, 0.0, 0.75], q=rpy2wxyz([0, np.arctan2(1, 1), 0])),
    '-x-z': Pose(p=[-0.75, 0.0, -0.75], q=rpy2wxyz([0, np.arctan2(-1, 1), 0])),
    '+x+y': Pose(p=[0.75, 0.75, 0.0], q=quat_mul_wxyz(rpy2wxyz([0, 0, -np.pi / 2]), rpy2wxyz([0, 0, np.arctan2(-1, 1)]))),
    '-x+y': Pose(p=[-0.75, 0.75, 0.0], q=quat_mul_wxyz(rpy2wxyz([0, 0, -np.pi / 2]), rpy2wxyz([0, 0, np.arctan2(1, 1)]))),
    '-y+z': Pose(p=[0.0, -0.75, 0.75], q=quat_mul_wxyz(rpy2wxyz([0, 0, np.pi / 2]), rpy2wxyz([0, np.arctan2(1, 1), 0]))),
    '-y-z': Pose(p=[0.0, -0.75, -0.75], q=quat_mul_wxyz(rpy2wxyz([0, 0, np.pi / 2]), rpy2wxyz([0, np.arctan2(-1, 1), 0]))),
    '+x-y': Pose(p=[0.75, -0.75, 0.0], q=quat_mul_wxyz(rpy2wxyz([0, 0, np.pi / 2]), rpy2wxyz([0, 0, np.arctan2(1, 1)]))),
    '-x-y': Pose(p=[-0.75, -0.75, 0.0], q=quat_mul_wxyz(rpy2wxyz([0, 0, np.pi / 2]), rpy2wxyz([0, 0, np.arctan2(-1, 1)]))),
    '+z': Pose(p=[0.0, 0.0 ,0.75*1.414], q=rpy2wxyz([0, np.pi/2, 0])),
    '+x': Pose(p=[0.75*1.414, 0.0, 0.0], q=rpy2wxyz([0, 0, np.pi])),
    '+y': Pose(p=[0.0, 0.75*1.414, 0.0], q=rpy2wxyz([0, 0, -np.pi/2]))
}


def print_collisions(collisions):
    """Helper function to abstract away the printing of collisions"""
    if len(collisions) == 0:
        print("No collision")
        return
    for collision in collisions:
        print(
            f"{collision.link_name1} of entity {collision.object_name1} collides"
            f" with {collision.link_name2} of entity {collision.object_name2}"
        )


class TableScene:
    def __init__(
            self,
            fps: float = 240.0,
            add_robot: bool = False
    ):
        self.engine = Engine()
        self.renderer = SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.fps = fps
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(1 / fps)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.viewer = None

        material = self.scene.create_physical_material(0.5, 0.5, 0.1)  # Create a physical material
        self.ground = self.scene.add_ground(altitude=-0.1, render_half_size=[10, 10, 0.1], material=material)  # Add a ground
        self.plane = create_box(self.scene, Pose([1, 0, -0.05]), [0.8, 1, 0.05], color=[0.9,0.9,0.9,1])

        self.robot = None
        self.planner: Optional[mplib.Planner] = None
        self.attach_drive: Optional[Drive] = None

        if add_robot:
            loader: URDFLoader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            physics_material: PhysicalMaterial = self.scene.create_physical_material(5, 5, 0)
            # default physics material: static friction = 0.3, dynamic friction = 0.3, restitution = 0.1
            config = {}
            config['material'] = physics_material
            self.robot: Articulation = loader.load("assets/panda/panda.urdf", config)
            self.end_effector = self.robot.get_links()[-3]
            self.robot.set_root_pose(Pose([0, 0, 0], [1, 0, 0, 0]))

            # Set initial joint positions
            init_qpos = [0, 0.19634954084936207, 0.0, -2.617993877991494, 0.0, 2.941592653589793, 0.7853981633974483, 0, 0]
            self.robot.set_qpos(init_qpos)

            self.active_joints = self.robot.get_active_joints()
            for i, joint in enumerate(self.active_joints):
                joint.set_drive_property(stiffness=1000, damping=200)
                # joint.set_drive_property(stiffness=750, damping=150)
                joint.set_drive_target(init_qpos[i])

            self.set_up_planner()

        self.is_hiding_env_visual = False

    def set_drive_target(self, qpos, vel_qpos=None):
        for i in range(7):
            self.active_joints[i].set_drive_target(qpos[i])
            if vel_qpos is not None:
                self.active_joints[i].set_drive_velocity_target(vel_qpos[i])

    def set_up_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        floor = fcl.Box([2, 2, 0.2])  # create a 2 x 2 x 0.1m box
        # create a collision object for the floor, with a 10cm offset in the z direction
        floor_fcl_collision_object = fcl.CollisionObject(floor, pymp.Pose())
        # a very small offset of 0.0001 is used to prevent the collision between link0 and the floor
        floor_fcl_object = fcl.FCLObject('floor', pymp.Pose([0, 0, -0.1001], [1, 0, 0, 0]), [floor_fcl_collision_object], [pymp.Pose()])
        self.planner = mplib.Planner(
            urdf="assets/panda/panda.urdf",
            srdf="assets/panda/panda.srdf",
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand",
            # joint_vel_limits=np.ones(7) * 0.5,
            joint_vel_limits=np.ones(7) * 0.1,
            joint_acc_limits=np.ones(7) * 0.1,
            objects=[floor_fcl_object]
        )

    def update_env_pcd(self, exclude_ids: list[int] = None, pcd_resolution = 1e-3, verbose=False):
        """Update the point cloud of the environment for planner collision avoidance"""
        all_actor_ids = [actor.get_id() for actor in self.scene.get_all_actors() if len(actor.get_collision_shapes()) > 0]
        all_links = self.scene.get_all_articulations()
        for articulation in all_links:
            all_actor_ids += [link.get_id() for link in articulation.get_links()]
        robot_actor_ids = [link.get_id() for link in self.robot.get_links()]
        ground_actor_id = self.ground.get_id()  # ground collision avoidance is handled in the planner
        excluded_actor_ids = robot_actor_ids + [ground_actor_id]
        if exclude_ids is not None:
            excluded_actor_ids += exclude_ids
        all_object_ids = list(set(all_actor_ids) - set(excluded_actor_ids))
        pcds = []
        for object_id in all_object_ids:
            if object_id != self.plane.get_id():
                current_actor = self.scene.find_actor_by_id(object_id)
                if current_actor is None:
                    current_actor = self.scene.find_articulation_link_by_link_id(object_id)
                if current_actor is not None:
                    pcds.append(get_actor_pcd(current_actor, 500000))
            else:
                x = np.linspace(0.2, 0.8, 1000)
                y = np.linspace(-0.4, 0.4, 1000)
                x, y = np.meshgrid(x, y)
                x = x.flatten()
                y = y.flatten()
                z = np.zeros_like(x)
                pcds.append(np.stack([x, y, z], axis=-1))

        if len(pcds) == 0:
            pcd = np.array([[1e6, 1e6, 1e6]])  # empty point cloud
        else:
            pcd = np.concatenate(pcds, axis=0)

        if verbose:
            # visualize the pcd
            print(pcd.shape)
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
            o3d.visualization.draw_geometries([o3d_pcd])

        self.planner.update_point_cloud(pcd, resolution=pcd_resolution)
        return pcd # return the point cloud for debugging

    def grasp_center_ik(self, grasp_center: np.ndarray, ee_quat_wxyz: np.ndarray,
                        start_qpos: np.ndarray, mask: list = None,
                        threshold: float = 1e-3, exclude_ids: Optional[List[int]] = None,
                        verbose: bool = False
    ) -> tuple[Union[np.ndarray, None], np.ndarray]:
        if self.robot is None:
            raise ValueError("No robot in the scene")
        pos_delta = np.array([0, 0, 0.1])
        ee_quat_xyzw = wxyz2xyzw(ee_quat_wxyz)
        hand_pos = grasp_center - rotate_vector(pos_delta, ee_quat_xyzw)
        return self.ee_ik(Pose(p=hand_pos, q=ee_quat_wxyz), start_qpos, mask, threshold, exclude_ids, verbose=verbose), hand_pos

    def ee_ik(
            self,
            ee_pose: Pose,
            start_qpos: np.ndarray,
            mask: list = None,
            threshold: float = 1e-3,
            exclude_ids: Optional[List[int]] = None,
            return_closest: bool = False,
            verbose: bool = False,
            pcd_resolution = 1e-3
    ) -> Union[np.ndarray, None]:
        if self.robot is None:
            raise ValueError("No robot in the scene")
        self.update_env_pcd(exclude_ids, pcd_resolution=pcd_resolution)
        if mask is None:
            mask = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        status, qpos = self.planner.IK(
            goal_pose=pymp.Pose(ee_pose.p, ee_pose.q),
            start_qpos=start_qpos,
            mask=mask,
            threshold=threshold,
            return_closest=return_closest,
            verbose=verbose
        )
        if status == "Success":
            if return_closest:
                return qpos
            else:
                return qpos[0]
        else:
            return None

    def ee_ik_without_collision_check(
            self,
            ee_pose: Pose,
            start_qpos: np.ndarray,
            mask: list = None,
            threshold: float = 1e-3,
            return_closest: bool = False,
            verbose: bool = False,
    ) -> Union[np.ndarray, None]:
        self.planner.remove_point_cloud()
        if mask is None:
            mask = [0, 0, 0, 0, 0, 0, 0, 1, 1]
        status, qpos = self.planner.IK(
            goal_pose=pymp.Pose(ee_pose.p, ee_pose.q),
            start_qpos=start_qpos,
            mask=mask,
            threshold=threshold,
            return_closest=return_closest,
            verbose=verbose
        )
        if status == "Success":
            if return_closest:
                return qpos
            else:
                return qpos[0]
        else:
            return None

    def follow_path(self, result, check_collision=False, collision_obj_1=None, collision_obj_2=None, threshold=1e-3, camera=None, camera_interval=4):
        n_step = result['position'].shape[0]
        collision = False
        images = []
        for i in range(n_step):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
                external=False)
            self.robot.set_qf(qf)
            # for j in range(7):
            #     self.active_joints[j].set_drive_target(result['position'][i][j])
            #     self.active_joints[j].set_drive_velocity_target(result['velocity'][i][j])
            self.set_drive_target(result['position'][i], result['velocity'][i])
            self.scene.step()
            if check_collision:
                collisions = get_contacts_by_id(self.scene, collision_obj_1, collision_obj_2, threshold)
                if len(collisions) > 0:
                    collision = True
                    break
            if i % 4 == 0:
                self.scene.update_render()
                if self.viewer is not None:
                    self.viewer.render()
                if camera is not None and i % camera_interval == 0:
                    image = get_rgba_img(camera=camera)
                    images.append(image)

        if camera is not None:
            return collision, images
        else:
            return collision

    def open_gripper(self, gripper_target=0.04, camera=None, camera_interval=4):
        images = []
        qpos = self.robot.get_qpos()
        for i, joint in enumerate(self.active_joints):
            if i < 7:
                joint.set_drive_target(qpos[i])
            else:
                joint.set_drive_target(gripper_target)

        for i in range(int(self.fps)):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
                external=False)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                if self.viewer is not None:
                    self.viewer.render()
                if camera is not None and i % camera_interval == 0:
                    image = get_rgba_img(camera=camera)
                    images.append(image)

        if camera is not None:
            return images
        else:
            return None


    def close_gripper(self, gripper_target=0.01, camera=None, camera_interval=4):
        images = []
        qpos = self.robot.get_qpos()
        for i, joint in enumerate(self.active_joints):
            if i < 7:
                joint.set_drive_target(qpos[i])
            else:
                joint.set_drive_target(gripper_target)
        for i in range(int(self.fps)):
            qf = self.robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
                external=False)
            self.robot.set_qf(qf)
            self.scene.step()
            if i % 4 == 0:
                self.scene.update_render()
                if self.viewer is not None:
                    self.viewer.render()
                if camera is not None and i % camera_interval == 0:
                    image = get_rgba_img(camera=camera)
                    images.append(image)

        if camera is not None:
            return images
        else:
            return None

    def attach_object(self, object: Actor):
        if self.attach_drive is not None:
            raise ValueError("An object is already attached")
        self.attach_drive = self.scene.create_drive(
            self.end_effector,
            Pose(),
            object,
            object.get_pose().inv() * self.end_effector.get_pose()
        )
        self.attach_drive.lock_motion(True, True, True, True, True, True)

    def detach_object(self):
        if self.attach_drive is not None:
            self.scene.remove_drive(self.attach_drive)
            self.attach_drive = None
        else:
            print("No object attached")

    def planner_attach_obj(self, obj: Union[Actor, Articulation]):
        if isinstance(obj, Actor):
            object_mesh = get_actor_mesh(obj, to_world_frame=False)
        elif isinstance(obj, Articulation):
            object_meshes = get_articulation_meshes(obj)
            object_mesh = merge_meshes(object_meshes)
        else:
            raise ValueError("Unsupported object type, must be Actor or Articulation")
        os.makedirs('mesh_cache', exist_ok=True)
        random_path = f'mesh_cache/{generate_random_string()}.obj'
        object_mesh.export(random_path)
        object_pose = obj.get_pose()
        ee_pose = self.end_effector.get_pose()
        if isinstance(obj, Actor):
            object_pose_rel_ee = ee_pose.inv() * object_pose
        else:
            object_pose_rel_ee = ee_pose.inv()
        self.planner.update_attached_mesh(random_path, pose=pymp.Pose(object_pose_rel_ee.p, object_pose_rel_ee.q))
        os.remove(random_path)

    def planner_detach_obj(self):
        self.planner.detach_object('panda_9_mesh', also_remove=True)

    def step(self):
        self.scene.step()

    def update_render(self):
        self.scene.update_render()

    def hide_robot_visual(self):
        if self.robot is None:
            raise ValueError("No robot in the scene, cannot hide robot visual")
        for link in self.robot.get_links():
            link.hide_visual()

    def unhide_robot_visual(self):
        if self.robot is None:
            raise ValueError("No robot in the scene, cannot unhide robot visual")
        for link in self.robot.get_links():
            link.unhide_visual()

    def create_viewer(
            self,
            resolutions: tuple[int, int] = (1440, 1440),
            camera_xyz: tuple[float, float, float] = (1.2, 0.25, 0.4),
            camera_rpy: tuple[float, float, float] = (0.0, -0.4, 2.7),
            near: float = 0.05,
            far: float = 100,
            fovy: float = 1,
    ) -> Viewer:
        self.viewer = Viewer(self.renderer, resolutions=resolutions)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(*camera_xyz)
        self.viewer.set_camera_rpy(*camera_rpy)
        self.viewer.window.set_camera_parameters(near, far, fovy)

        return self.viewer

    def add_camera(
            self,
            direction: str = '+x',
            fovy: float = None,
            width: int = 768,
            height: int = 768,
    ) -> CameraEntity:
        if fovy is None:
            fovy = np.deg2rad(60)
        camera = self.scene.add_camera(
            name=direction,
            fovy=fovy,
            width=width,
            height=height,
            near=0.05,
            far=100
        )
        camera.set_pose(DIRECTION2POSE[direction])
        return camera

    def get_picture(
            self,
            direction: str = '+x',
            additional_translation: np.ndarray = None,
            additional_rotation: np.ndarray = None,
            get_depth: bool = False,
            debug_viewer: bool = False
    ) -> tuple[Image.Image, CameraEntity] or tuple[Image.Image, np.ndarray, CameraEntity]:
        camera = self.add_camera(direction)
        if additional_translation is not None:
            pose = camera.get_pose()
            p, q = pose.p, pose.q
            p += additional_translation
            if additional_rotation is not None:
                q = quat_mul_wxyz(additional_rotation, q)
            camera.set_pose(Pose(p=p, q=q))
        if debug_viewer:
            viewer = self.create_viewer()
            while not viewer.closed:
                # self.step()
                self.update_render()
                viewer.render()

        self.update_render()
        image = get_rgba_img(camera=camera)
        image = Image.fromarray(image).convert('RGB')
        if debug_viewer:
            image.show()

        if get_depth:
            depth_img = get_depth_img(camera)
            return image, depth_img, camera
        return image, camera

    def get_8_pictures(self, debug_viewer: bool = False) -> Image:
        cameras = []
        for direction in ['+y+z', '+y-z', '+x+y', '-x+y', '-y+z', '-y-z', '+x-y', '-x-y']:
            cameras.append(self.add_camera(direction))
        if debug_viewer:
            viewer = self.create_viewer()
            while not viewer.closed:
                self.step()
                self.update_render()
                viewer.render()

        self.step()
        self.update_render()

        imgs = []
        for i in range(8):
            camera = cameras[i]
            image = get_rgba_img(camera=camera)
            image = Image.fromarray(image)
            if debug_viewer:
                image.show()
            imgs.append(image)
        return imgs