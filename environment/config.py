from dataclasses import dataclass, field, fields
from typing import Any


def _filtered_dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    valid_names = {item.name for item in fields(cls)}
    return {key: value for key, value in payload.items() if key in valid_names}


@dataclass
class ArenaConfig:
    arena_width: float = 2.24
    arena_height: float = 1.02
    agent_radius: float = 0.05
    agent_z: float = 0.029975  # nominal: WHEEL_BODY_Z_OFFSET(0.0099) + WHEEL_RADIUS(0.020) * wheel_radius_scale_nominal(1.003749132156372)
    tag_distance_factor: float = 1.2
    n_rays: int = 128
    action_frequency: int = 20
    episode_max_length: int = 40
    chaser_freeze_seconds: int = 2
    agent_max_linear_velocity: float = 1.35
    agent_max_angular_velocity: float = 31.0
    minimum_starting_separation: float = 0.12
    wall_margin_factor: float = 1.05
    action_scale: float = 1.0


@dataclass
class RewardConfig:
    win_reward: float = 1.0
    chaser_time_reward: float = -0.01
    evader_time_reward: float = 0.01
    collision_penalty: float = 10.0
    distance_shaping_scale: float = 0.0
    distance_shaping_gamma: float = 0.99
    forward_motion_reward_scale: float = 0.0
    action_smoothing_scale: float = 0.005


@dataclass
class LayoutRandomizationConfig:
    max_obstacles: int = 8
    obstacle_width: float = 0.20
    obstacle_candidate_pool: int = 64
    spawn_candidate_pool: int = 64
    obstacle_separation_factor: float = 1.0


@dataclass
class DynamicsRandomizationConfig:
    enabled: bool = True
    # Chassis mass — no sysid estimate, centered at 1.0
    chassis_mass_scale_min: float = 0.9
    chassis_mass_scale_max: float = 1.1
    # CoM offset — x/z centered from sysid, y kept centered at 0
    com_offset_x_nominal: float = 0.014703182503581047
    com_offset_x_delta: float = 0.008
    com_offset_z_nominal: float = -0.00038700297591276467
    com_offset_z_delta: float = 0.006
    com_offset_y_max: float = 0.003
    # Geometry scales centered from the latest evader-only sysid fit while
    # preserving the current randomization span wherever it remains valid.
    track_width_scale_nominal: float = 0.9207258820533752
    track_width_scale_min: float = 0.86
    track_width_scale_max: float = 1.00
    wheel_radius_scale_nominal: float = 1.0047537088394165
    wheel_radius_scale_min: float = 0.96
    wheel_radius_scale_max: float = 1.05
    # Simplified wheel-ground model: one longitudinal traction scale and one
    # turn scrub scale. Legacy wheel friction fields remain below for now but
    # are no longer the primary training/sysid knobs.
    traction_scale_nominal: float = 0.8315610289573669
    traction_scale_min: float = 0.60
    traction_scale_max: float = 1.40
    turn_scrub_scale_nominal: float = 0.9665983319282532
    turn_scrub_scale_min: float = 0.50
    turn_scrub_scale_max: float = 1.50
    # Legacy wheel friction scales kept for compatibility.
    wheel_slide_friction_scale_nominal: float = 0.8315610289573669
    wheel_slide_friction_scale_min: float = 0.60
    wheel_slide_friction_scale_max: float = 1.40
    wheel_torsional_friction_scale_nominal: float = 0.9665983319282532
    wheel_torsional_friction_scale_min: float = 0.50
    wheel_torsional_friction_scale_max: float = 1.50
    wheel_rolling_friction_scale_nominal: float = 0.9665983319282532
    wheel_rolling_friction_scale_min: float = 0.50
    wheel_rolling_friction_scale_max: float = 1.50
    # Shared chassis/bumper floor-contact slide friction scale
    body_contact_friction_scale_nominal: float = 0.6012741923332214
    body_contact_friction_scale_min: float = 0.30
    body_contact_friction_scale_max: float = 1.20
    # Caster geometry/contact
    caster_radius_scale_nominal: float = 0.849507212638855
    caster_radius_scale_min: float = 0.78
    caster_radius_scale_max: float = 0.92
    # Caster offset
    caster_offset_x_nominal: float = -0.0013577864738181233
    caster_offset_x_delta: float = 0.008
    caster_slide_friction_scale_nominal: float = 2.8258633613586426
    caster_slide_friction_scale_min: float = 1.60
    caster_slide_friction_scale_max: float = 3.60
    caster_torsional_friction_scale_nominal: float = 2.3276596069335938
    caster_torsional_friction_scale_min: float = 1.40
    caster_torsional_friction_scale_max: float = 3.10
    # Wheel joint and motor terms
    wheel_joint_damping_scale_nominal: float = 2.0157265663146973
    wheel_joint_damping_scale_min: float = 1.00
    wheel_joint_damping_scale_max: float = 3.00
    # Keep randomized, but in the small low-friction regime sysid consistently prefers
    wheel_joint_frictionloss_scale_nominal: float = 0.02858835831284523
    wheel_joint_frictionloss_scale_min: float = 0.01
    wheel_joint_frictionloss_scale_max: float = 0.06
    wheel_armature_scale_nominal: float = 2.4349827766418457
    wheel_armature_scale_min: float = 1.20
    wheel_armature_scale_max: float = 3.50
    motor_strength_scale_nominal: float = 0.9941380620002747
    motor_strength_scale_min: float = 0.70
    motor_strength_scale_max: float = 1.30
    back_emf_scale_nominal: float = 0.7814494371414185
    back_emf_scale_min: float = 0.55
    back_emf_scale_max: float = 1.05
    # Motor balance remains robot-specific, but still centered on the fitted nominal
    motor_balance_nominal: float = -0.008113667368888855
    motor_balance_delta_max: float = 0.14
    # Motor time constant
    motor_time_constant_seconds_nominal: float = 0.015150192193686962
    motor_time_constant_seconds_min: float = 0.008
    motor_time_constant_seconds_max: float = 0.035
    # Smooth high-command actuator compression and launch-slip attenuation.
    motor_curve_sharpness_nominal: float = 0.0
    motor_curve_sharpness_min: float = 0.0
    motor_curve_sharpness_max: float = 3.0
    traction_slip_threshold_nominal: float = 0.12944123148918152
    traction_slip_threshold_min: float = 0.08
    traction_slip_threshold_max: float = 0.35
    traction_loss_strength_nominal: float = 0.0
    traction_loss_strength_min: float = 0.0
    traction_loss_strength_max: float = 3.0
    traction_min_scale_nominal: float = 0.55
    traction_min_scale_min: float = 0.20
    traction_min_scale_max: float = 1.0


@dataclass
class PipelineRandomizationConfig:
    enabled: bool = True
    nominal_action_delay_substeps: int = 0
    action_delay_substeps_delta: int = 6
    nominal_observation_delay_substeps: int = 11
    observation_delay_substeps_delta: int = 4
    max_stale_observation_steps: int = 1
    action_drop_probability_max: float = 0.002
    frame_drop_probability_max: float = 0.02
    stale_observation_probability_max: float = 0.01
    position_noise_std_max: float = 0.001
    yaw_noise_std_max: float = 0.02
    hold_last_action_on_drop: bool = True


@dataclass
class CurriculumConfig:
    enabled: bool = True
    layout_start: float = 0.05
    pipeline_start: float = 0.10
    dynamics_start: float = 0.10
    full_progress: float = 0.60
    exponent: float = 1.0


@dataclass
class EnvironmentConfig:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    layout_randomization: LayoutRandomizationConfig = field(
        default_factory=LayoutRandomizationConfig
    )
    dynamics_randomization: DynamicsRandomizationConfig = field(
        default_factory=DynamicsRandomizationConfig
    )
    pipeline_randomization: PipelineRandomizationConfig = field(
        default_factory=PipelineRandomizationConfig
    )
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvironmentConfig":
        return cls(
            arena=ArenaConfig(
                **_filtered_dataclass_kwargs(ArenaConfig, payload.get("arena", {}))
            ),
            rewards=RewardConfig(
                **_filtered_dataclass_kwargs(RewardConfig, payload.get("rewards", {}))
            ),
            layout_randomization=LayoutRandomizationConfig(
                **_filtered_dataclass_kwargs(
                    LayoutRandomizationConfig,
                    payload.get("layout_randomization", {}),
                )
            ),
            dynamics_randomization=DynamicsRandomizationConfig(
                **_filtered_dataclass_kwargs(
                    DynamicsRandomizationConfig,
                    payload.get("dynamics_randomization", {}),
                )
            ),
            pipeline_randomization=PipelineRandomizationConfig(
                **_filtered_dataclass_kwargs(
                    PipelineRandomizationConfig,
                    payload.get("pipeline_randomization", {}),
                )
            ),
            curriculum=CurriculumConfig(
                **_filtered_dataclass_kwargs(
                    CurriculumConfig, payload.get("curriculum", {})
                )
            ),
        )

    @property
    def arena_width(self) -> float:
        return self.arena.arena_width

    @property
    def arena_height(self) -> float:
        return self.arena.arena_height

    @property
    def agent_radius(self) -> float:
        return self.arena.agent_radius

    @property
    def agent_z(self) -> float:
        return self.arena.agent_z

    @property
    def tag_distance_factor(self) -> float:
        return self.arena.tag_distance_factor

    @property
    def n_rays(self) -> int:
        return self.arena.n_rays

    @property
    def action_frequency(self) -> int:
        return self.arena.action_frequency

    @property
    def episode_max_length(self) -> int:
        return self.arena.episode_max_length

    @property
    def chaser_freeze_seconds(self) -> int:
        return self.arena.chaser_freeze_seconds

    @property
    def agent_max_linear_velocity(self) -> float:
        return self.arena.agent_max_linear_velocity

    @property
    def agent_max_angular_velocity(self) -> float:
        return self.arena.agent_max_angular_velocity

    @property
    def minimum_starting_separation(self) -> float:
        return self.arena.minimum_starting_separation

    @property
    def wall_margin_factor(self) -> float:
        return self.arena.wall_margin_factor

    @property
    def win_reward(self) -> float:
        return self.rewards.win_reward

    @property
    def chaser_time_reward(self) -> float:
        return self.rewards.chaser_time_reward

    @property
    def evader_time_reward(self) -> float:
        return self.rewards.evader_time_reward

    @property
    def collision_penalty(self) -> float:
        return self.rewards.collision_penalty

    @property
    def distance_shaping_scale(self) -> float:
        return self.rewards.distance_shaping_scale

    @property
    def distance_shaping_gamma(self) -> float:
        return self.rewards.distance_shaping_gamma

    @property
    def forward_motion_reward_scale(self) -> float:
        return self.rewards.forward_motion_reward_scale

    @property
    def action_smoothing_scale(self) -> float:
        return self.rewards.action_smoothing_scale

    @property
    def max_obstacles(self) -> int:
        return self.layout_randomization.max_obstacles

    @property
    def obstacle_width(self) -> float:
        return self.layout_randomization.obstacle_width

    def validate(self) -> None:
        if self.arena_width <= 0 or self.arena_height <= 0:
            raise ValueError("arena dimensions must be positive")
        if self.agent_radius <= 0 or self.agent_z <= 0:
            raise ValueError("agent dimensions must be positive")
        if self.tag_distance_factor <= 0:
            raise ValueError("tag_distance_factor must be positive")
        if self.n_rays <= 0:
            raise ValueError("n_rays must be positive")
        if self.action_frequency <= 0:
            raise ValueError("action_frequency must be positive")
        if self.episode_max_length <= 0:
            raise ValueError("episode_max_length must be positive")
        if self.chaser_freeze_seconds < 0:
            raise ValueError("chaser_freeze_seconds cannot be negative")
        if self.agent_max_linear_velocity <= 0:
            raise ValueError("agent_max_linear_velocity must be positive")
        if self.agent_max_angular_velocity <= 0:
            raise ValueError("agent_max_angular_velocity must be positive")
        if self.minimum_starting_separation <= 0:
            raise ValueError("minimum_starting_separation must be positive")
        if self.wall_margin_factor <= 0:
            raise ValueError("wall_margin_factor must be positive")
        if self.max_obstacles < 0:
            raise ValueError("max_obstacles cannot be negative")
        if self.max_obstacles > 0 and self.obstacle_width <= 0:
            raise ValueError(
                "obstacle_width must be positive when obstacles are enabled"
            )
        if self.layout_randomization.obstacle_candidate_pool <= 0:
            raise ValueError("obstacle_candidate_pool must be positive")
        if self.layout_randomization.spawn_candidate_pool <= 0:
            raise ValueError("spawn_candidate_pool must be positive")
        if self.layout_randomization.obstacle_separation_factor <= 0:
            raise ValueError("obstacle_separation_factor must be positive")

        dyn = self.dynamics_randomization
        if dyn.chassis_mass_scale_min <= 0 or dyn.chassis_mass_scale_max <= 0:
            raise ValueError("mass scales must be positive")
        if dyn.track_width_scale_min <= 0 or dyn.track_width_scale_max <= 0:
            raise ValueError("track width scales must be positive")
        if dyn.wheel_radius_scale_min <= 0 or dyn.wheel_radius_scale_max <= 0:
            raise ValueError("wheel radius scales must be positive")
        if dyn.traction_scale_min <= 0 or dyn.traction_scale_max <= 0:
            raise ValueError("traction scales must be positive")
        if dyn.turn_scrub_scale_min <= 0 or dyn.turn_scrub_scale_max <= 0:
            raise ValueError("turn scrub scales must be positive")
        if (
            dyn.wheel_slide_friction_scale_min <= 0
            or dyn.wheel_slide_friction_scale_max <= 0
        ):
            raise ValueError("wheel slide friction scales must be positive")
        if (
            dyn.wheel_torsional_friction_scale_min <= 0
            or dyn.wheel_torsional_friction_scale_max <= 0
        ):
            raise ValueError("wheel torsional friction scales must be positive")
        if (
            dyn.wheel_rolling_friction_scale_min <= 0
            or dyn.wheel_rolling_friction_scale_max <= 0
        ):
            raise ValueError("wheel rolling friction scales must be positive")
        if (
            dyn.body_contact_friction_scale_min <= 0
            or dyn.body_contact_friction_scale_max <= 0
        ):
            raise ValueError("body contact friction scales must be positive")
        if dyn.caster_radius_scale_min <= 0 or dyn.caster_radius_scale_max <= 0:
            raise ValueError("caster radius scales must be positive")
        if dyn.caster_offset_x_delta < 0:
            raise ValueError("caster offset delta must be nonnegative")
        if (
            dyn.caster_slide_friction_scale_min <= 0
            or dyn.caster_slide_friction_scale_max <= 0
        ):
            raise ValueError("caster slide friction scales must be positive")
        if (
            dyn.caster_torsional_friction_scale_min <= 0
            or dyn.caster_torsional_friction_scale_max <= 0
        ):
            raise ValueError("caster torsional friction scales must be positive")
        if (
            dyn.wheel_joint_damping_scale_min <= 0
            or dyn.wheel_joint_damping_scale_max <= 0
        ):
            raise ValueError("wheel joint damping scales must be positive")
        if (
            dyn.wheel_joint_frictionloss_scale_min <= 0
            or dyn.wheel_joint_frictionloss_scale_max <= 0
        ):
            raise ValueError("wheel joint frictionloss scales must be positive")
        if dyn.wheel_armature_scale_min <= 0 or dyn.wheel_armature_scale_max <= 0:
            raise ValueError("wheel armature scales must be positive")
        if dyn.motor_strength_scale_min <= 0 or dyn.motor_strength_scale_max <= 0:
            raise ValueError("motor strength scales must be positive")
        if dyn.back_emf_scale_min <= 0 or dyn.back_emf_scale_max <= 0:
            raise ValueError("back-EMF scales must be positive")
        if dyn.motor_balance_delta_max < 0 or dyn.motor_balance_delta_max >= 1.0:
            raise ValueError("motor_balance_delta_max must be in [0, 1)")
        if (
            dyn.motor_time_constant_seconds_min < 0
            or dyn.motor_time_constant_seconds_max < 0
        ):
            raise ValueError("motor time constants must be nonnegative")
        if dyn.motor_curve_sharpness_min < 0 or dyn.motor_curve_sharpness_max < 0:
            raise ValueError("motor curve sharpness must be nonnegative")
        if dyn.traction_slip_threshold_min < 0 or dyn.traction_slip_threshold_max < 0:
            raise ValueError("traction slip thresholds must be nonnegative")
        if dyn.traction_loss_strength_min < 0 or dyn.traction_loss_strength_max < 0:
            raise ValueError("traction loss strengths must be nonnegative")
        if not (
            0 < dyn.traction_min_scale_min <= 1.0
            and 0 < dyn.traction_min_scale_max <= 1.0
        ):
            raise ValueError("traction min scales must be in (0, 1]")
        if (
            dyn.com_offset_x_delta < 0
            or dyn.com_offset_z_delta < 0
            or dyn.com_offset_y_max < 0
        ):
            raise ValueError("CoM offset limits must be nonnegative")

        pipe = self.pipeline_randomization
        if (
            pipe.nominal_action_delay_substeps < 0
            or pipe.action_delay_substeps_delta < 0
            or pipe.nominal_observation_delay_substeps < 0
            or pipe.observation_delay_substeps_delta < 0
        ):
            raise ValueError("delay substeps must be nonnegative")
        if pipe.max_stale_observation_steps < 0:
            raise ValueError("max_stale_observation_steps must be nonnegative")
        for name, value in (
            ("action_drop_probability_max", pipe.action_drop_probability_max),
            ("frame_drop_probability_max", pipe.frame_drop_probability_max),
            (
                "stale_observation_probability_max",
                pipe.stale_observation_probability_max,
            ),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if pipe.position_noise_std_max < 0 or pipe.yaw_noise_std_max < 0:
            raise ValueError("noise std must be nonnegative")

        cur = self.curriculum
        if cur.layout_start < 0 or cur.pipeline_start < 0 or cur.dynamics_start < 0:
            raise ValueError("curriculum start points must be nonnegative")
        if cur.full_progress <= 0:
            raise ValueError("curriculum full_progress must be positive")
        if cur.exponent <= 0:
            raise ValueError("curriculum exponent must be positive")
