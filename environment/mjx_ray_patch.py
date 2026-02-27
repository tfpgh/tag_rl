"""Patch mjx.ray to support cylinder geoms (curved side only).

MJX's JAX backend ray function is missing GeomType.CYLINDER from its dispatch
table (_RAY_FUNC). This adds the missing intersection for the curved side of
the cylinder, which is sufficient for horizontal rays.
"""

import jax
import jax.numpy as jp
from mujoco.mjx._src import ray as mjx_ray
from mujoco.mjx._src.types import GeomType


def _ray_cylinder(size: jax.Array, pnt: jax.Array, vec: jax.Array) -> jax.Array:
    a = vec[0:2] @ vec[0:2]
    b = vec[0:2] @ pnt[0:2]
    c = pnt[0:2] @ pnt[0:2] - size[0] * size[0]
    x0, x1 = mjx_ray._ray_quad(a, b, c)

    # Accept only if the hit is between the flat caps
    x0 = jp.where(jp.abs(pnt[2] + x0 * vec[2]) <= size[1], x0, jp.inf)
    x1 = jp.where(jp.abs(pnt[2] + x1 * vec[2]) <= size[1], x1, jp.inf)
    return jp.minimum(x0, x1)


def patch_mjx_cylinder_rays() -> None:
    """Register cylinder ray intersection with MJX."""
    mjx_ray._RAY_FUNC[GeomType.CYLINDER] = _ray_cylinder
