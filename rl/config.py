from dataclasses import dataclass


@dataclass
class RLConfig:
    num_envs: int = 65_536
    num_steps: int = 128

    total_timesteps: int = int(3e9)

    hidden_size: int = 256

    lr: float = 3e-4
    anneal_lr: bool = True
    lr_decay_start_fraction: float = 0.7
    final_lr_scale: float = 0.001
    update_epochs: int = 4
    num_minibatches: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    checkpoint_interval: int = 50
    seed: int = 0

    @property
    def timesteps_per_update(self) -> int:
        return self.num_steps * self.num_envs

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.timesteps_per_update

    def validate(self, actual_num_devices: int | None = None) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if actual_num_devices is not None and actual_num_devices <= 0:
            raise ValueError("actual_num_devices must be positive")
        if actual_num_devices is not None and self.num_envs % actual_num_devices != 0:
            raise ValueError(
                "num_envs must be divisible by the active device count for sharded execution"
            )
        if self.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive")
        if self.num_envs % self.num_minibatches != 0:
            raise ValueError(
                "num_envs must be divisible by num_minibatches so each PPO minibatch has "
                "the same number of environments"
            )
        if self.timesteps_per_update <= 0:
            raise ValueError("timesteps_per_update must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.lr_decay_start_fraction < 0 or self.lr_decay_start_fraction > 1:
            raise ValueError("lr_decay_start_fraction must be in [0, 1]")
        if self.final_lr_scale < 0 or self.final_lr_scale > 1:
            raise ValueError("final_lr_scale must be in [0, 1]")
