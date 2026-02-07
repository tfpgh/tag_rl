from environment.config import EnvironmentConfig

WALL_GEOM_GROUP = 0
AGENT_GEOM_GROUP = 1
FLOOR_GEOM_GROUP = 2

WALL_THICKNESS = 0.02
WALL_HEIGHT = 0.1

CHASER_COLOR = "1 0.545 0.545 1"
EVADER_COLOR = "0.435 0.702 0.722 1"


def _agent_mjcf(
    name: str, lid_color: str, starting_pos: str = "0 0 0.0299"
) -> tuple[str, str, str]:
    """
    MJCF XML for an agent

    Returns body_xml, actuator_xml, sensor_xml
    """

    body_xml = f"""
    <body name="{name}" pos="{starting_pos}">
	    <freejoint name="{name}_root" />
	    <inertial
            pos="-0.0148 0 -0.009"
            diaginertia="0.000112 0.000112 0.000188"
            mass="0.15"
        />
		<geom
            name="{name}_chassis_base"
            type="cylinder"
            size="0.05 0.019"
            group="{AGENT_GEOM_GROUP}"
            rgba="0.1 0.1 0.1 1"
        />
		<geom
            name="{name}_chassis_top"
            type="cylinder"
            pos="0 0 0.0205"
            size="0.05 0.0015"
            group="{AGENT_GEOM_GROUP}"
            rgba="{lid_color}"
        />
        <body name="{name}_left_wheel" pos="0 0.037123 -0.0099">
            <joint
                name="{name}_left_wheel_joint"
                type="hinge"
                axis="0 1 0"
                damping="0.0023"
                frictionloss="0.019"
            />
            <geom
                name="{name}_left_wheel_geom"
                type="cylinder"
                size="0.020 0.0015"
                euler="90 0 0"
                mass="0.00425"
                group="{AGENT_GEOM_GROUP}"
                rgba="0.6 0.6 0.6 1"
                friction="1.1 0.005 0.002"
            />
        </body>
        <body name="{name}_right_wheel" pos="0 -0.037123 -0.0099">
            <joint
                name="{name}_right_wheel_joint"
                type="hinge"
                axis="0 1 0"
                damping="0.0023"
                frictionloss="0.019"
            />
            <geom
                name="{name}_right_wheel_geom"
                type="cylinder"
                size="0.020 0.0015"
                euler="90 0 0"
                mass="0.00425"
                group="{AGENT_GEOM_GROUP}"
                rgba="0.6 0.6 0.6 1"
                friction="1.1 0.005 0.002"
            />
        </body>
        <body name="{name}_caster" pos="-0.037 0 -0.0251">
            <geom
                name="{name}_caster_housing"
                type="box"
                size="0.006 0.006 0.004"
                pos="0 0 0.004"
                mass="0.004"
                group="{AGENT_GEOM_GROUP}"
                rgba="0.1 0.1 0.1 1"
            />
            <body name="{name}_caster_ball" pos="0 0 0">
                <joint
                    name="{name}_caster_ball_joint"
                    type="ball"
                    damping="0.000001"
                />
                <geom
                    name="{name}_caster_ball_geom"
                    type="sphere"
                    size="0.0048"
                    mass="0.002"
                    group="{AGENT_GEOM_GROUP}"
                    rgba="0.6 0.6 0.6 1"
                    friction="0.7 0.005 0.0001"
                />
            </body>
        </body>
	</body>
    """

    actuator_xml = f"""
    <motor
        name="{name}_left_motor"
        joint="{name}_left_wheel_joint"
        gear="0.19"
        ctrlrange="-1 1"
        ctrllimited="true"
    />
    <motor
        name="{name}_right_motor"
        joint="{name}_right_wheel_joint"
        gear="0.19"
        ctrlrange="-1 1"
        ctrllimited="true"
    />
    """

    sensor_xml = f"""
    <framepos name="{name}_pos" objtype="body" objname="{name}" />
    <framequat name="{name}_quat" objtype="body" objname="{name}" />
    <framelinvel name="{name}_vel" objtype="body" objname="{name}" />
    <frameangvel name="{name}_angvel" objtype="body" objname="{name}" />
    """

    return body_xml, actuator_xml, sensor_xml


def _arena_xml(config: EnvironmentConfig) -> str:
    """
    MJCF XML for the arena

    Returns just body_xml
    """

    arena_half_width = config.arena_width / 2
    arena_half_height = config.arena_height / 2
    half_wall_thickness = WALL_THICKNESS / 2
    half_wall_height = WALL_HEIGHT / 2

    return f"""
    <geom
        name="floor"
        type="plane"
        pos="0 0 -0.02"
        size="{arena_half_width + WALL_THICKNESS} {arena_half_height + WALL_THICKNESS} 0.04"
        group="{FLOOR_GEOM_GROUP}"
        rgba="0.2 0.2 0.25 1"
        friction="1.1 0.005 0.0001"
    />
    <geom
        name="wall_north"
        type="box"
        pos="0 {arena_half_height + half_wall_thickness} {half_wall_height}"
        size="{arena_half_width + WALL_THICKNESS} {half_wall_thickness} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
    />
    <geom
        name="wall_south"
        type="box"
        pos="0 {-arena_half_height - half_wall_thickness} {half_wall_height}"
        size="{arena_half_width + WALL_THICKNESS} {half_wall_thickness} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
    />
    <geom
        name="wall_east"
        type="box"
        pos="{arena_half_width + half_wall_thickness} 0 {half_wall_height}"
        size="{half_wall_thickness} {arena_half_height} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
    />
    <geom
        name="wall_west"
        type="box"
        pos="{-arena_half_width - half_wall_thickness} 0 {half_wall_height}"
        size="{half_wall_thickness} {arena_half_height} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
    />
    """


def generate_mjcf(config: EnvironmentConfig) -> str:
    """Build the complete MJCF XML for the environment"""

    chaser_body_xml, chaser_actuator_xml, chaser_sensor_xml = _agent_mjcf(
        "chaser", CHASER_COLOR, "0.3 0 0.0299"
    )
    evader_body_xml, evader_actuator_xml, evader_sensor_xml = _agent_mjcf(
        "evader", EVADER_COLOR, "-0.3 0 0.0299"
    )

    return f"""
    <mujoco model="tag">
        <option integrator="implicitfast"/>
        <visual>
            <headlight ambient="0.5 0.5 0.5" />
        </visual>
        <worldbody>
            {_arena_xml(config)}
            {chaser_body_xml}
            {evader_body_xml}
        </worldbody>
        <actuator>
            {chaser_actuator_xml}
            {evader_actuator_xml}
        </actuator>
        <sensor>
            {chaser_sensor_xml}
            {evader_sensor_xml}
        </sensor>
    </mujoco>
    """
