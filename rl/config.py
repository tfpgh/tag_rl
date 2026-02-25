from dataclasses import dataclass


@dataclass
class RLConfig:
    num_envs: int = 1024
    num_steps: int = 128

    total_timesteps: int = int(1e9)

    hidden_size: int = 128

    lr: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.04
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    video_interval: int = 50
    seed: int = 0
