from __future__ import annotations

import os
from pathlib import Path


def configure_jax_cache() -> Path:
    cache_dir = Path(os.environ.get("TAG_RL_JAX_CACHE_DIR", ".cache/jax"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir.resolve()))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.25")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    return cache_dir


CACHE_DIR = configure_jax_cache()
