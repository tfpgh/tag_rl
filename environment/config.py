from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    # Arena wall-to-wall inner dimensions
    arena_width: float = 2.24
    arena_height: float = 1.02

    agent_radius: float = 0.05
    agent_z: float = 0.0299  # Chassis center height

    # Agent radius multiplier for tag distance
    tag_distance_factor: float = 1.02

    # Perception rays
    n_rays: int = 256

    # Action frequency, hz
    action_frequency: int = 20

    # Episode timing, seconds
    episode_max_length: int = 20
    chaser_freeze_seconds: int = 4

    # Rewards
    win_reward: float = 1.0  # timeout for evader, tag for chaser
    time_reward: float = 0.006  # + for evader, - for chaser
    distance_shaping_scale: float = 0.00
    distance_shaping_gamma: float = 0.99
    collision_penalty: float = 2.0  # Per-step penalty for wall/obstacle contact

    # Normalization, not exact
    agent_max_linear_velocity: float = 1.35  # m/s
    agent_max_angular_velocity: float = 31.0  # rad/s

    minimum_starting_separation: float = 0.13  # m, center to center
    wall_margin_factor: float = 1.5  # Minimum distance from wall (* agent_radius)

    collision_epsilon_factor: float = 1.04

    # Obstacles
    max_obstacles: int = 10  # always allocated in MJCF
    obstacle_width: float = 0.1  # square box side length (meters)
