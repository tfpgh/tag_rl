import os
import pickle

import jax

from rl.rollout import RunnerState


def save_checkpoint(
    runner_state: RunnerState, global_step: int, checkpoint_dir: str = "checkpoints"
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{global_step}.pkl")
    checkpoint = {
        "chaser_params": jax.device_get(runner_state[0].params),
        "evader_params": jax.device_get(runner_state[1].params),
        "global_step": global_step,
    }
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint at step {global_step}")
