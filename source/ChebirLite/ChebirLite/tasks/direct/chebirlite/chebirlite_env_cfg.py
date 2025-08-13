# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
import math
from ChebirLite.robots.ChebirLite import CHEBIR_LITE_CONFIG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaaclab.terrains as terrain_gen

FLAT_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)

joint_names = [
    "body_body_left_hip",
    "hip_left_left_hip_shin",
    "body_body_right_hip",
    "hip_right_right_hip_shin",
]

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

@configclass
class ChebirliteEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 5.0
    # - spaces definition
    action_space = 4
    observation_space = 14
    state_space = 0

    # domain randomization config
    events: EventCfg = EventCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CHEBIR_LITE_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # cant be used with groundPlane (only with rigid bodies not static) see https://github.com/isaac-sim/IsaacLab/issues/1995
    # contact_forces_body_cfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/body",
    #     update_period=0.0,
    #     history_length=6,
    #     debug_vis=True,
    #     filter_prim_paths_expr=["/World/ground"],
    # )

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=FLAT_TERRAINS_CFG,
    #     max_init_terrain_level=0,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #         project_uvw=True,
    #         texture_scale=(0.25, 0.25),
    #     ),
    #     debug_vis=False,
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    base_base_rotor_dof_name = "base_base_rotor"
    base_rotor_rotor_bar_dof_name = "base_rotor_rotor_bar"
    bar_bar_body_dof_name = "bar_bar_body"
    body_body_left_hip_dof_name = "body_body_left_hip"
    hip_left_left_hip_shin_dof_name = "hip_left_left_hip_shin"
    body_body_right_hip_dof_name = "body_body_right_hip"
    hip_right_right_hip_shin_dof_name = "hip_right_right_hip_shin"
    # initial tilt angle variation
    initial_tilt_angle_variation = 20/180 * math.pi  # 20 degrees in radians
    thigh_length = 0.18
    torso_length = 0.16

    # - action scale
    action_scale = 70 / 180 * math.pi  #
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -20.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    termination_rod_angle = 0.05
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]