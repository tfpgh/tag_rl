from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    # Arena wall-to-wall inner dimensions
    arena_width: float = 2.24
    arena_height: float = 1.02

    agent_radius: float = 0.05
    agent_z: float = 0.0299  # Chassis center height

    # Agent radius multiplier for tag distance
    tag_distance_factor: float = 1.03

    # Perception rays
    n_rays: int = 256

    # Action frequency, hz
    action_frequency: int = 20

    # Episode timing, seconds
    episode_max_length: int = 20
    chaser_freeze_seconds: int = 2

    # Rewards
    win_reward: float = 1.0  # timeout for evader, tag for chaser
    chaser_time_reward: float = -0.004
    evader_time_reward: float = 0.004
    collision_penalty: float = 5.0  # Terminal penalty for wall/obstacle contact

    distance_shaping_scale: float = 0.00
    distance_shaping_gamma: float = 0.99

    # Normalization, not exact
    agent_max_linear_velocity: float = 1.35  # m/s
    agent_max_angular_velocity: float = 31.0  # rad/s

    minimum_starting_separation: float = 0.13  # m, center to center
    wall_margin_factor: float = 1.5  # Minimum distance from wall (* agent_radius)

    # Obstacles
    max_obstacles: int = 8  # always allocated in MJCF
    obstacle_width: float = 0.10  # square box side length (meters)
