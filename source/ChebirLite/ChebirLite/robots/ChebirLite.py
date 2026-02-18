import os

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

usd_path = os.path.join(os.path.dirname(__file__), "ChebirLite.usda")

CHEBIR_LITE_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "base_base_rotor": 0.0,
            "base_rotor_rotor_bar": -0.2,
            "bar_bar_body": 0.0,
            "body_body_left_hip": 90.0 * np.pi / 180.0,  # Adjusted to radians
            "hip_left_left_hip_shin": 0.0,
            "body_body_right_hip": -90.0 * np.pi / 180.0,
            "hip_right_right_hip_shin": 0.0,
        },
        joint_vel={
            "base_base_rotor": 0.0,
            "base_rotor_rotor_bar": 0.0,
            "bar_bar_body": 0.0,
            "body_body_left_hip": 0.0,
            "hip_left_left_hip_shin": 0.0,
            "body_body_right_hip": 0.0,
            "hip_right_right_hip_shin": 0.0,
        },
    ),
    actuators={
        "base_base_rotor_actuator": ImplicitActuatorCfg(
            joint_names_expr=["base_base_rotor"],
            effort_limit=100.0,  # Adjusted effort limit
            velocity_limit=100.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "base_rotor_rotor_bar_actuator": ImplicitActuatorCfg(
            joint_names_expr=["base_rotor_rotor_bar"],
            effort_limit=100.0,  # Adjusted effort limit
            velocity_limit=100.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "bar_bar_body_actuator": ImplicitActuatorCfg(
            joint_names_expr=["bar_bar_body"],
            effort_limit=100.0,  # Adjusted effort limit
            velocity_limit=100.0,  # Adjusted velocity limit
            stiffness=0.0,
            damping=0.0,
        ),
        "body_body_left_hip_actuator": ImplicitActuatorCfg(
            joint_names_expr=["body_body_left_hip"],
            effort_limit=4.5,
            velocity_limit=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
        "hip_left_left_hip_shin_actuator": ImplicitActuatorCfg(
            joint_names_expr=["hip_left_left_hip_shin"],
            effort_limit=4.5,
            velocity_limit=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
        "body_body_right_hip_actuator": ImplicitActuatorCfg(
            joint_names_expr=["body_body_right_hip"],
            effort_limit=4.5,
            velocity_limit=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
        "hip_right_right_hip_shin_actuator": ImplicitActuatorCfg(
            joint_names_expr=["hip_right_right_hip_shin"],
            effort_limit=4.5,
            velocity_limit=572957.0,
            stiffness=27.9,
            damping=2.15,
        ),
    },
)