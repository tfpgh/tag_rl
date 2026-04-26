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
    agent_z: float = 0.030121827030181884  # nominal: WHEEL_BODY_Z_OFFSET(0.0099) + WHEEL_RADIUS(0.020) * wheel_radius_scale_nominal(1.0110913515090942)
    tag_distance_factor: float = 1.05
    n_rays: int = 128
    action_frequency: int = 20
    episode_max_length: int = 40
    chaser_freeze_seconds: int = 2
    agent_max_linear_velocity: float = 1.35
    agent_max_angular_velocity: float = 31.0
    minimum_starting_separation: float = 0.12
    wall_margin_factor: float = 1.05
    action_scale: float = 0.75


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
    max_obstacles: int = 3
    obstacle_width: float = 0.20
    obstacle_candidate_pool: int = 64
    spawn_candidate_pool: int = 64
    obstacle_separation_factor: float = 1.0


@dataclass
class DynamicsRandomizationConfig:
    enabled: bool = True
    # Geometry and dynamics centered on the latest sysid fit, with wide DR ranges.
    track_width_scale_nominal: float = 1.0422232151031494
    track_width_scale_min: float = 0.97
    track_width_scale_max: float = 1.12
    wheel_radius_scale_nominal: float = 1.0110913515090942
    wheel_radius_scale_min: float = 0.97
    wheel_radius_scale_max: float = 1.05
    # Wheel-ground friction: longitudinal traction and turn-scrub.
    traction_scale_nominal: float = 2.174974203109741
    traction_scale_min: float = 1.55
    traction_scale_max: float = 2.85
    turn_scrub_scale_nominal: float = 1.5484225749969482
    turn_scrub_scale_min: float = 1.10
    turn_scrub_scale_max: float = 1.95
    # Motor terms.
    motor_strength_scale_nominal: float = 1.786872148513794
    motor_strength_scale_min: float = 1.40
    motor_strength_scale_max: float = 2.20
    back_emf_scale_nominal: float = 1.7278169393539429
    back_emf_scale_min: float = 1.35
    back_emf_scale_max: float = 2.15
    motor_balance_nominal: float = 0.0
    motor_balance_delta_max: float = 0.12
    motor_time_constant_seconds_nominal: float = 0.023041656240820885
    motor_time_constant_seconds_min: float = 0.015
    motor_time_constant_seconds_max: float = 0.035


@dataclass
class PipelineRandomizationConfig:
    enabled: bool = True
    nominal_action_delay_substeps: int = 10
    action_delay_substeps_delta: int = 2
    nominal_observation_delay_substeps: int = 0
    observation_delay_substeps_delta: int = 3
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
        if dyn.track_width_scale_min <= 0 or dyn.track_width_scale_max <= 0:
            raise ValueError("track width scales must be positive")
        if dyn.wheel_radius_scale_min <= 0 or dyn.wheel_radius_scale_max <= 0:
            raise ValueError("wheel radius scales must be positive")
        if dyn.traction_scale_min <= 0 or dyn.traction_scale_max <= 0:
            raise ValueError("traction scales must be positive")
        if dyn.turn_scrub_scale_min <= 0 or dyn.turn_scrub_scale_max <= 0:
            raise ValueError("turn scrub scales must be positive")
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
