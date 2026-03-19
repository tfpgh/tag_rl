from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArenaConfig:
    arena_width: float = 2.24
    arena_height: float = 1.02
    agent_radius: float = 0.05
    agent_z: float = 0.0299
    tag_distance_factor: float = 1.03
    n_rays: int = 128
    action_frequency: int = 20
    episode_max_length: int = 20
    chaser_freeze_seconds: int = 2
    agent_max_linear_velocity: float = 1.35
    agent_max_angular_velocity: float = 31.0
    minimum_starting_separation: float = 0.13
    wall_margin_factor: float = 1.5


@dataclass
class RewardConfig:
    win_reward: float = 1.0
    chaser_time_reward: float = -0.004
    evader_time_reward: float = 0.004
    collision_penalty: float = 5.0
    distance_shaping_scale: float = 0.0
    distance_shaping_gamma: float = 0.99


@dataclass
class LayoutRandomizationConfig:
    max_obstacles: int = 4
    obstacle_width: float = 0.20
    obstacle_candidate_pool: int = 64
    spawn_candidate_pool: int = 64
    obstacle_separation_factor: float = 2.2


@dataclass
class DynamicsRandomizationConfig:
    enabled: bool = True
    chassis_mass_scale_min: float = 0.9
    chassis_mass_scale_max: float = 1.1
    com_offset_x_max: float = 0.003
    com_offset_y_max: float = 0.003
    inertia_scale_min: float = 0.93
    inertia_scale_max: float = 1.08
    floor_friction_scale_min: float = 0.92
    floor_friction_scale_max: float = 1.08
    wheel_friction_scale_min: float = 0.9
    wheel_friction_scale_max: float = 1.12
    caster_friction_scale_min: float = 0.9
    caster_friction_scale_max: float = 1.12
    wheel_damping_scale_min: float = 0.92
    wheel_damping_scale_max: float = 1.1
    wheel_frictionloss_scale_min: float = 0.92
    wheel_frictionloss_scale_max: float = 1.1
    motor_strength_scale_min: float = 0.88
    motor_strength_scale_max: float = 1.12
    motor_balance_delta_max: float = 0.06


@dataclass
class PipelineRandomizationConfig:
    enabled: bool = True
    max_action_delay_substeps: int = 20
    max_observation_delay_substeps: int = 20
    max_stale_observation_steps: int = 1
    action_drop_probability_max: float = 0.01
    frame_drop_probability_max: float = 0.02
    stale_observation_probability_max: float = 0.03
    position_noise_std_max: float = 0.004
    yaw_noise_std_max: float = 0.02
    hold_last_action_on_drop: bool = True


@dataclass
class CurriculumConfig:
    enabled: bool = True
    layout_start: float = 0.05
    pipeline_start: float = 0.10
    dynamics_start: float = 0.15
    full_progress: float = 0.60
    exponent: float = 1.3


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
            arena=ArenaConfig(**payload.get("arena", {})),
            rewards=RewardConfig(**payload.get("rewards", {})),
            layout_randomization=LayoutRandomizationConfig(
                **payload.get("layout_randomization", {})
            ),
            dynamics_randomization=DynamicsRandomizationConfig(
                **payload.get("dynamics_randomization", {})
            ),
            pipeline_randomization=PipelineRandomizationConfig(
                **payload.get("pipeline_randomization", {})
            ),
            curriculum=CurriculumConfig(**payload.get("curriculum", {})),
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
        if dyn.inertia_scale_min <= 0 or dyn.inertia_scale_max <= 0:
            raise ValueError("inertia scales must be positive")
        if dyn.floor_friction_scale_min <= 0 or dyn.floor_friction_scale_max <= 0:
            raise ValueError("floor friction scales must be positive")
        if dyn.wheel_friction_scale_min <= 0 or dyn.wheel_friction_scale_max <= 0:
            raise ValueError("wheel friction scales must be positive")
        if dyn.caster_friction_scale_min <= 0 or dyn.caster_friction_scale_max <= 0:
            raise ValueError("caster friction scales must be positive")
        if dyn.wheel_damping_scale_min <= 0 or dyn.wheel_damping_scale_max <= 0:
            raise ValueError("wheel damping scales must be positive")
        if (
            dyn.wheel_frictionloss_scale_min <= 0
            or dyn.wheel_frictionloss_scale_max <= 0
        ):
            raise ValueError("wheel frictionloss scales must be positive")
        if dyn.motor_strength_scale_min <= 0 or dyn.motor_strength_scale_max <= 0:
            raise ValueError("motor strength scales must be positive")
        if dyn.motor_balance_delta_max < 0 or dyn.motor_balance_delta_max >= 1.0:
            raise ValueError("motor_balance_delta_max must be in [0, 1)")
        if dyn.com_offset_x_max < 0 or dyn.com_offset_y_max < 0:
            raise ValueError("CoM offset limits must be nonnegative")

        pipe = self.pipeline_randomization
        if (
            pipe.max_action_delay_substeps < 0
            or pipe.max_observation_delay_substeps < 0
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
