from dataclasses import dataclass


@dataclass
class RLConfig:
    num_devices: int = 4
    num_envs: int = 16384
    num_steps: int = 128

    total_timesteps: int = int(5e9)

    hidden_size: int = 256

    lr: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    checkpoint_interval: int = 50
    seed: int = 0
    model_type: str = "cnn_rnn"

    @property
    def local_num_envs(self) -> int:
        return self.num_envs // self.num_devices

    @property
    def timesteps_per_update(self) -> int:
        return self.num_steps * self.num_envs

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.timesteps_per_update

    def validate(self, actual_num_devices: int | None = None) -> None:
        if self.num_devices <= 0:
            raise ValueError("num_devices must be positive")
        if actual_num_devices is not None and actual_num_devices != self.num_devices:
            raise ValueError(
                f"Expected {self.num_devices} local devices, found {actual_num_devices}"
            )
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.num_envs % self.num_devices != 0:
            raise ValueError(
                "num_envs must be divisible by num_devices for sharded execution"
            )
        if self.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive")
        if self.local_num_envs % self.num_minibatches != 0:
            raise ValueError(
                "local_num_envs must be divisible by num_minibatches so each device can "
                "build equal PPO minibatches"
            )
        if self.timesteps_per_update <= 0:
            raise ValueError("timesteps_per_update must be positive")
