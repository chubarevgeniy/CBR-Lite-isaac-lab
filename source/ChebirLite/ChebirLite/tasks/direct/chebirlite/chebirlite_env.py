# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils  # <-- add import for math utils


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
        self.body_idx,_ = self.robot.find_bodies('body')
        self.left_hip_idx,_ = self.robot.find_bodies('hip_left')
        self.right_hip_idx,_ = self.robot.find_bodies('hip_right')

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Initialize command handling
        self.learning_step = 0
        self.command = torch.zeros(self.cfg.scene.num_envs, device=self.device)
        # Setup visualization for commands. Assume marker index 1 ("command") is the red arrow.
        self.visualization_markers = define_markers()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.marker_offset[:,-1] = 0.5  # Offset for visualization
        self._sample_commands()
        self.previous_actions = torch.zeros((self.cfg.scene.num_envs, self.cfg.action_space), device=self.device)

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

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])


    def _sample_commands(self) -> None:
        self.command = sample_uniform(
            -1.1,
            1.1,
            (self.cfg.scene.num_envs,),
            device=self.device,
        )

    def _pre_physics_step(self, actions):
        self.learning_step += 1
        if self.learning_step % 200 == 0:
            self._sample_commands()
        self.actions = self._clip_and_scale_actions(actions.clone())
        self._visualize_markers()

    def _get_left_knee_location(self) -> torch.Tensor:
        left_hip_loc = self.robot.data.body_state_w[:, self.left_hip_idx[0], :3]
        left_hip_rots = self.robot.data.body_state_w[:, self.left_hip_idx[0], 3:7]
        leg_offset = torch.zeros_like(left_hip_loc, device=left_hip_loc.device) + torch.tensor([0.0, self.cfg.thigh_length, 0.0], device=left_hip_loc.device)
        left_knee_loc = left_hip_loc + math_utils.quat_apply(left_hip_rots, leg_offset)
        return left_knee_loc

    def _get_right_knee_location(self) -> torch.Tensor:
        right_hip_loc = self.robot.data.body_state_w[:, self.right_hip_idx[0], :3]
        right_hip_rots = self.robot.data.body_state_w[:, self.right_hip_idx[0], 3:7]
        leg_offset = torch.zeros_like(right_hip_loc, device=right_hip_loc.device) + torch.tensor([0.0, self.cfg.thigh_length, 0.0], device=right_hip_loc.device)
        right_knee_loc = right_hip_loc - math_utils.quat_apply(right_hip_rots, leg_offset)
        return right_knee_loc

    def _get_top_torso_location(self) -> torch.Tensor:
        torso_loc = self.robot.data.body_state_w[:, self.body_idx[0], :3]
        torso_rots = self.robot.data.body_state_w[:, self.body_idx[0], 3:7]
        offset = torch.tensor([self.cfg.torso_length, 0.0, 0.0], device=torso_loc.device).expand_as(torso_loc)
        top_torso_loc = torso_loc + math_utils.quat_apply(torso_rots, offset)
        return top_torso_loc

    def _visualize_markers(self):
        # Arrow locations for command and speed visualization (not true torso top/bottom)
        torso_base_loc = self.robot.data.body_state_w[:, self.body_idx[0], :3]
        arrow_loc = torch.vstack((torso_base_loc + self.marker_offset*1.1, torso_base_loc + self.marker_offset))

        # Rotation for arrows
        ang_speed = self.joint_vel[:, self.base_base_rotor_dof_name_idx[0]]
        base_angle = self.joint_pos[:, self.base_base_rotor_dof_name_idx[0]]
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=arrow_loc.device)
        rots_actual = math_utils.quat_from_angle_axis(base_angle + torch.sign(ang_speed)*torch.pi/2, up_vec)
        rots_command = math_utils.quat_from_angle_axis(base_angle + torch.sign(self.command)*torch.pi/2, up_vec)
        arrow_rots = torch.vstack((rots_actual, rots_command))

        # Scaling for arrows
        base_scale = torch.tensor([0.25, 0.25, 0.5], device=arrow_loc.device)
        command_scale = (1+torch.abs(self.command)).unsqueeze(1) * base_scale
        actual_scale = (1+torch.abs(ang_speed)).unsqueeze(1) * base_scale
        arrow_scales = torch.vstack((actual_scale, command_scale))

        # Knees
        left_knee_loc = self._get_left_knee_location()
        right_knee_loc = self._get_right_knee_location()
        scales_knee = torch.ones_like(arrow_scales, device=arrow_loc.device) * 0.5  # Smaller scale for knees
        left_hip_rots = self.robot.data.body_state_w[:, self.left_hip_idx[0], 3:7]
        right_hip_rots = self.robot.data.body_state_w[:, self.right_hip_idx[0], 3:7]

        # Top torso marker (same marker as knees)
        top_torso_loc = self._get_top_torso_location()
        top_torso_rots = self.robot.data.body_state_w[:, self.body_idx[0], 3:7]
        top_torso_scales = torch.ones_like(arrow_scales, device=arrow_loc.device) * 0.5

        # Stack all marker locations, rotations, and scales
        loc = torch.vstack((arrow_loc, left_knee_loc, right_knee_loc, top_torso_loc))
        scales = torch.vstack((arrow_scales, scales_knee, scales_knee, top_torso_scales))
        rots = torch.vstack((arrow_rots, left_hip_rots, right_hip_rots, top_torso_rots))

        # Marker indices: 0=speed, 1=command, 2=knee, 2=knee, 2=top torso (same as knee)
        num_envs = self.cfg.scene.num_envs
        marker_indices = torch.hstack((
            torch.zeros(num_envs),                # speed arrow
            torch.ones(num_envs),                 # command arrow
            2*torch.ones(num_envs),               # left knee
            2*torch.ones(num_envs),               # right knee
            2*torch.ones(num_envs),               # top torso
        ))
        self.visualization_markers.visualize(loc, rots, marker_indices=marker_indices, scales=scales)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions, joint_ids=[
            self.body_body_right_hip_dof_name_idx[0],
            self.body_body_left_hip_dof_name_idx[0],
            self.hip_right_right_hip_shin_dof_name_idx[0],
            self.hip_left_left_hip_shin_dof_name_idx[0],
        ])

    def _get_observations(self):
        # Example observation: joint positions and velocities
        self.previous_actions = self.actions.clone()
        # print("!!!!!!!!!",self.joint_vel.shape, self.previous_actions.shape)
        return {
            "policy": torch.cat([
                self.joint_pos[:,:self.base_base_rotor_dof_name_idx[0]], 
                self.joint_pos[:,self.base_base_rotor_dof_name_idx[0]+1:],
                self.joint_vel,
                self.command.unsqueeze(1),
                self.actions,
            ], dim=-1)
        }
    
    def _get_rewards(self):
        return compute_rewards(
            body_vel=self.joint_vel[:, self.base_base_rotor_dof_name_idx],
            body_height=self.joint_pos[:, self.base_rotor_rotor_bar_dof_name_idx],
            reset_terminated=self.reset_terminated,
            command=self.command.unsqueeze(1),
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
    
    def _get_left_hip_location(self) -> torch.Tensor:
        """Returns left hip location for all envs."""
        return self.robot.data.body_state_w[:, self.left_hip_idx[0], :3]

    def _get_right_hip_location(self) -> torch.Tensor:
        """Returns right hip location for all envs."""
        return self.robot.data.body_state_w[:, self.right_hip_idx[0], :3]
    
@torch.jit.script
def compute_rewards(
    body_vel: torch.Tensor,
    body_height: torch.Tensor,
    reset_terminated: torch.Tensor,
    command: torch.Tensor,
):
    termination_reward = reset_terminated.float() * -20.0  # Adjusted termination reward
    alive_reward = (1.0 - reset_terminated.float()) * 0.1
    height_reward = body_height.sum(dim=-1) * -0.2  # Adjusted height reward
    velocity_reward = (body_vel - command).abs().sum(dim=-1) * -1  # Adjusted velocity reward
    total_reward = alive_reward + termination_reward + velocity_reward + height_reward
    return total_reward

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "speed": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "knee": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)