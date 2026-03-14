import jax
import jax.numpy as jnp

from environment.config import EnvironmentConfig

HIT_TYPE_ENVIRONMENT = 0.0
HIT_TYPE_AGENT = 1.0


def rotate_into_box_frame(
    points: jax.Array, center: jax.Array, yaw: jax.Array
) -> jax.Array:
    translated = points - center
    c = jnp.cos(yaw)
    s = jnp.sin(yaw)
    x = translated[..., 0] * c + translated[..., 1] * s
    y = -translated[..., 0] * s + translated[..., 1] * c
    return jnp.stack([x, y], axis=-1)


def circle_intersects_box(
    circle_xy: jax.Array,
    radius: float,
    box_center_xy: jax.Array,
    box_yaw: jax.Array,
    half_extent: float,
) -> jax.Array:
    local = rotate_into_box_frame(circle_xy, box_center_xy, box_yaw)
    closest = jnp.clip(local, -half_extent, half_extent)
    delta = local - closest
    return jnp.dot(delta, delta) <= radius * radius


def circle_hits_any_obstacle(
    circle_xy: jax.Array,
    radius: float,
    obstacle_positions_xy: jax.Array,
    obstacle_yaws: jax.Array,
    obstacle_active: jax.Array,
    half_extent: float,
) -> jax.Array:
    hits = jax.vmap(circle_intersects_box, in_axes=(None, None, 0, 0, None))(
        circle_xy,
        radius,
        obstacle_positions_xy,
        obstacle_yaws,
        half_extent,
    )
    return jnp.any(hits & obstacle_active)


def point_out_of_bounds(
    position_xy: jax.Array, config: EnvironmentConfig, radius: float
) -> jax.Array:
    half_width = config.arena_width / 2 - radius
    half_height = config.arena_height / 2 - radius
    return (
        (position_xy[0] <= -half_width)
        | (position_xy[0] >= half_width)
        | (position_xy[1] <= -half_height)
        | (position_xy[1] >= half_height)
    )


def ray_circle_distance(
    origin: jax.Array, direction: jax.Array, center: jax.Array, radius: float
) -> jax.Array:
    offset = origin - center
    b = jnp.dot(direction, offset)
    c = jnp.dot(offset, offset) - radius * radius
    discriminant = b * b - c
    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    t0 = -b - sqrt_disc
    t1 = -b + sqrt_disc
    t = jnp.where(t0 > 0.0, t0, t1)
    valid = (discriminant >= 0.0) & (t > 0.0)
    return jnp.where(valid, t, jnp.inf)


def ray_box_distance(
    origin: jax.Array,
    direction: jax.Array,
    center: jax.Array,
    yaw: jax.Array,
    half_extent: float,
) -> jax.Array:
    local_origin = rotate_into_box_frame(origin, center, yaw)
    c = jnp.cos(yaw)
    s = jnp.sin(yaw)
    local_direction = jnp.array(
        [direction[0] * c + direction[1] * s, -direction[0] * s + direction[1] * c]
    )
    inv_direction = jnp.where(
        jnp.abs(local_direction) > 1e-6, 1.0 / local_direction, jnp.inf
    )
    t0 = (-half_extent - local_origin) * inv_direction
    t1 = (half_extent - local_origin) * inv_direction
    tmin = jnp.minimum(t0, t1)
    tmax = jnp.maximum(t0, t1)
    entry = jnp.max(tmin)
    exit = jnp.min(tmax)
    candidate = jnp.where(entry > 0.0, entry, exit)
    valid = (exit >= 0.0) & (entry <= exit) & (candidate > 0.0)
    return jnp.where(valid, candidate, jnp.inf)


def ray_arena_distance(
    origin: jax.Array, direction: jax.Array, config: EnvironmentConfig
) -> jax.Array:
    half_width = config.arena_width / 2
    half_height = config.arena_height / 2

    tx_min = (-half_width - origin[0]) / direction[0]
    tx_max = (half_width - origin[0]) / direction[0]
    y_at_tx_min = origin[1] + tx_min * direction[1]
    y_at_tx_max = origin[1] + tx_max * direction[1]
    valid_tx_min = (tx_min > 0.0) & (jnp.abs(y_at_tx_min) <= half_height)
    valid_tx_max = (tx_max > 0.0) & (jnp.abs(y_at_tx_max) <= half_height)

    ty_min = (-half_height - origin[1]) / direction[1]
    ty_max = (half_height - origin[1]) / direction[1]
    x_at_ty_min = origin[0] + ty_min * direction[0]
    x_at_ty_max = origin[0] + ty_max * direction[0]
    valid_ty_min = (ty_min > 0.0) & (jnp.abs(x_at_ty_min) <= half_width)
    valid_ty_max = (ty_max > 0.0) & (jnp.abs(x_at_ty_max) <= half_width)

    candidates = jnp.array(
        [
            jnp.where(valid_tx_min, tx_min, jnp.inf),
            jnp.where(valid_tx_max, tx_max, jnp.inf),
            jnp.where(valid_ty_min, ty_min, jnp.inf),
            jnp.where(valid_ty_max, ty_max, jnp.inf),
        ]
    )
    return jnp.min(candidates)


def raycast_scene(
    origin_xy: jax.Array,
    ray_angles: jax.Array,
    agent_yaw: jax.Array,
    other_agent_xy: jax.Array,
    other_agent_radius: float,
    obstacle_positions_xy: jax.Array,
    obstacle_yaws: jax.Array,
    obstacle_active: jax.Array,
    config: EnvironmentConfig,
) -> tuple[jax.Array, jax.Array]:
    obstacle_half_extent = config.obstacle_width / 2
    arena_diagonal = jnp.sqrt(config.arena_width**2 + config.arena_height**2)

    def cast_single(ray_angle: jax.Array) -> tuple[jax.Array, jax.Array]:
        angle = ray_angle + agent_yaw
        direction = jnp.array([jnp.cos(angle), jnp.sin(angle)])

        wall_distance = ray_arena_distance(origin_xy, direction, config)
        obstacle_distances = jax.vmap(
            ray_box_distance, in_axes=(None, None, 0, 0, None)
        )(
            origin_xy,
            direction,
            obstacle_positions_xy,
            obstacle_yaws,
            obstacle_half_extent,
        )
        obstacle_distance = jnp.min(
            jnp.where(obstacle_active, obstacle_distances, jnp.inf)
        )
        environment_distance = jnp.minimum(wall_distance, obstacle_distance)

        agent_distance = ray_circle_distance(
            origin_xy, direction, other_agent_xy, other_agent_radius
        )
        hit_type = jnp.where(
            agent_distance < environment_distance,
            HIT_TYPE_AGENT,
            HIT_TYPE_ENVIRONMENT,
        )
        distance = jnp.minimum(agent_distance, environment_distance)
        normalized_distance = jnp.clip(distance / arena_diagonal, 0.0, 1.0)
        return normalized_distance, hit_type

    return jax.vmap(cast_single)(ray_angles)
