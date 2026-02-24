from dataclasses import dataclass


@dataclass
class RLConfig:
    num_envs: int = 2048
    num_steps: int = 128

    total_timesteps: int = int(5e7)

    lr: float = 3e-4
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    anneal_lr: bool = False
    debug: bool = True
    hidden_size: int = 128
    seed: int = 0
