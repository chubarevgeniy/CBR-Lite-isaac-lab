# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.math import sample_uniform


from .chebirlite_env_cfg import ChebirliteEnvCfg


class ChebirliteEnv(DirectRLEnv):
    cfg: ChebirliteEnvCfg

    def __init__(self, cfg: ChebirliteEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.base_base_rotor_dof_name_idx, _ = self.robot.find_joints(self.cfg.base_base_rotor_dof_name)
        self.base_rotor_rotor_bar_dof_name_idx,_ = self.robot.find_joints(self.cfg.base_rotor_rotor_bar_dof_name)
        self.body_body_right_hip_dof_name_idx,_ = self.robot.find_joints(self.cfg.body_body_right_hip_dof_name)
        self.hip_right_right_hip_shin_dof_name_idx,_ = self.robot.find_joints(self.cfg.hip_right_right_hip_shin_dof_name)
        self.body_body_left_hip_dof_name_idx,_ = self.robot.find_joints(self.cfg.body_body_left_hip_dof_name)
        self.hip_left_left_hip_shin_dof_name_idx,_ = self.robot.find_joints(self.cfg.hip_left_left_hip_shin_dof_name)
        self.bar_bar_body_dof_name_idx,_ = self.robot.find_joints(self.cfg.bar_bar_body_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        # Initialize the robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add robot to the scene
        self.scene.articulations["robot"] = self.robot

        # cant be used with groundPlane (only with rigid bodies not static) see https://github.com/isaac-sim/IsaacLab/issues/1995
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # self.scene.sensors["contact_forces_body"] = ContactSensor(self.cfg.contact_forces_body_cfg)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions):
        self.actions = self._clip_and_scale_actions(actions.clone())
        #self._visualize_markers()

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions, joint_ids=[
            self.body_body_right_hip_dof_name_idx[0],
            self.body_body_left_hip_dof_name_idx[0],
            self.hip_right_right_hip_shin_dof_name_idx[0],
            self.hip_left_left_hip_shin_dof_name_idx[0],
        ])

    def _get_observations(self):
        # Example observation: joint positions and velocities
        return {
            "policy": torch.cat([
                self.joint_pos[:,:self.base_base_rotor_dof_name_idx[0]], 
                self.joint_pos[:,self.base_base_rotor_dof_name_idx[0]+1:],
                self.joint_vel
            ], dim=-1)
        }
    
    def _get_rewards(self):
        return compute_rewards(
            body_vel=self.joint_vel[:, self.base_base_rotor_dof_name_idx],
            body_height=self.joint_pos[:, self.base_rotor_rotor_bar_dof_name_idx],
            reset_terminated=self.reset_terminated,
        )

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.joint_pos[:,self.base_rotor_rotor_bar_dof_name_idx[0]] > self.cfg.termination_rod_angle
        return died, time_out

    def _reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self.bar_bar_body_dof_name_idx] += sample_uniform(
            -self.cfg.initial_tilt_angle_variation,
            self.cfg.initial_tilt_angle_variation,
            joint_pos[:, self.bar_bar_body_dof_name_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _clip_and_scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # Clip and scale actions if necessary
        actions =  ((actions.clamp(-1, 1) + 1) * self.cfg.action_scale)
        actions[:,0] *= -1
        return actions
    
@torch.jit.script
def compute_rewards(
    body_vel: torch.Tensor,
    body_height: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    termination_reward = reset_terminated.float() * -20.0  # Adjusted termination reward
    alive_reward = (1.0 - reset_terminated.float()) * 0.1
    height_reward = body_height.sum(dim=-1) * -1  # Adjusted height reward
    velocity_reward = body_vel.sum(dim=-1) * 5  # Adjusted velocity reward
    total_reward = alive_reward + termination_reward + velocity_reward + height_reward
    return total_reward