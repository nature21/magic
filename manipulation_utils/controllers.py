import numpy as np
from sapien.core import Actor

from utils.rotation_utils import wxyz2xyzw, quat_diff_in_axis_angle, cross


class LinePositionController:
    def __init__(
            self,
            actor: Actor,
            initial_position: np.ndarray,
            target_position: np.ndarray,
            num_steps: int,
            gain: float,
    ):
        self.actor = actor
        self.initial_position = initial_position
        self.target_position = target_position
        self.num_steps = num_steps
        self.gain = gain
        self.velocity_by_step = (self.target_position - self.initial_position) / self.num_steps
        self.target_pose_xyzw = wxyz2xyzw(actor.get_pose().q)

    def get_projection_t(self, position: np.ndarray) -> float:
        return np.dot((position - self.initial_position), self.velocity_by_step) / np.sum(self.velocity_by_step**2)

    def get_velocity(self) -> np.ndarray:
        position = self.actor.get_pose().p
        t = self.get_projection_t(position)
        target_t = np.maximum(0, np.minimum(t + 1, self.num_steps))
        current_target_position = self.initial_position + target_t * self.velocity_by_step
        velocity = self.gain * (current_target_position - position)
        return velocity

    def get_angular_velocity(self) -> np.ndarray:
        current_pose_xyzw = wxyz2xyzw(self.actor.get_pose().q)
        return self.gain * quat_diff_in_axis_angle(self.target_pose_xyzw, current_pose_xyzw)

    def set_velocities(self):
        self.actor.set_velocity(self.get_velocity())
        self.actor.set_angular_velocity(self.get_angular_velocity())

    def get_current_t(self) -> float:
        return self.get_projection_t(self.actor.get_pose().p)


class TrajectoryPositionController:
    def __init__(
            self,
            actor: Actor,
            p_array: np.ndarray,
            q_array: np.ndarray,
            gain: float,
            coef: float = 1
    ):
        self.actor = actor
        self.p_array = p_array
        self.q_array = q_array
        self.target_pose_xyzw = wxyz2xyzw(actor.get_pose().q)
        self.gain = gain
        self.coef = coef
        self.num_steps = len(p_array) - 1
        self.current_t = 0

    def get_projection_t(self, current_p, current_q) -> int:
        current_p = np.array(current_p)
        current_q = np.array(current_q)
        start = max(self.current_t - 5, 0)
        end = min(self.current_t + 6, self.num_steps)
        p_dist = np.sum((self.p_array[start:end] - current_p)**2, axis=1)
        q_dist = np.sum((self.q_array[start:end] - current_q)**2, axis=1)
        idx = np.argmin(p_dist + self.coef * q_dist)
        t = idx + start
        return t

    def get_velocities(self, t=None) -> tuple[np.ndarray, np.ndarray]:
        current_p = self.actor.get_pose().p
        current_q = self.actor.get_pose().q
        if t is None:
            t = self.get_projection_t(current_p, current_q)
        self.current_t = t
        target_t = np.maximum(0, np.minimum(t + 1, self.num_steps))
        current_target_position = self.p_array[target_t]
        current_target_orientation = self.q_array[target_t]
        velocity = self.gain * (current_target_position - current_p)
        angular_velocity = self.gain * quat_diff_in_axis_angle(wxyz2xyzw(current_target_orientation), wxyz2xyzw(current_q))
        return velocity, angular_velocity

    def set_velocities(self, contact=None):
        velocity, angular_velocity = self.get_velocities()
        if contact is not None:
            contact_point = contact['point']
            contact_normal = contact['normal']
            actor_center = self.actor.get_pose().p
            velocity, angular_velocity = self.get_velocities(t=self.current_t+1)
            vel_contact_point = velocity + cross(angular_velocity, contact_point - actor_center)
            vel_contact_normal = np.dot(vel_contact_point, contact_normal) * contact_normal
            print(velocity, vel_contact_point, vel_contact_normal)
            velocity = velocity - vel_contact_normal
            print(velocity)
        self.actor.set_velocity(velocity)
        self.actor.set_angular_velocity(angular_velocity)

    def get_current_t(self) -> int:
        return self.get_projection_t(self.actor.get_pose().p, self.actor.get_pose().q)

