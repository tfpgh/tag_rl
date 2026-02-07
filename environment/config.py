from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    # Arena wall-to-wall inner dimensions
    arena_width: float = 3.0
    arena_height: float = 2.0

    agent_radius: float = 0.05
    agent_z: float = 0.0299  # Chassis center height

    # Agent radius multiplier for tag distance
    tag_distance_factor: float = 1.1

    # Perception rays
    n_rays: int = 64

    # MuJoCo
    mujoco_timestep: float = 0.002

    # Action frequency, hz
    action_frequency: int = 50

    # Episode timing, seconds
    episode_max_length: int = 30
    chaser_freeze_seconds: int = 2

    # Rewards
    win_reward: float = 1.0  # timeout for evader, tag for chaser
    time_reward = 0.000001  # + for evader, - for chaser
    distance_shaping_scale = 0.001

    # Normalization, not exact
    agent_max_linear_velocity = 1.35  # m/s
    agent_max_angular_velocity = 31.0  # rad/s
