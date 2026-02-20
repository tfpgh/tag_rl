import io

import trimesh

from environment.config import EnvironmentConfig

WALL_GEOM_GROUP = 0
AGENT_GEOM_GROUP = 1
FLOOR_GEOM_GROUP = 2

# Contact bitmasks
CON_BM_VISUAL_ONLY = 0b00000
CON_BM_FLOOR = 0b00001
CON_BM_WHEEL = 0b00010
CON_BM_CASTER_BALL = 0b00100
CON_BM_CHASSIS = 0b01000
CON_BM_WALL = 0b10000

# contype and conaffinity
CONTACTS = {
    "floor": (CON_BM_FLOOR, CON_BM_WHEEL | CON_BM_CASTER_BALL | CON_BM_CHASSIS),
    "wheel": (CON_BM_WHEEL, CON_BM_FLOOR),
    "caster_ball": (CON_BM_CASTER_BALL, CON_BM_FLOOR),
    "chassis": (CON_BM_CHASSIS, CON_BM_FLOOR | CON_BM_WALL | CON_BM_CHASSIS),
    "wall": (CON_BM_WALL, CON_BM_CHASSIS),
    "visual_only": (CON_BM_VISUAL_ONLY, CON_BM_VISUAL_ONLY),
}

WALL_THICKNESS = 0.02
WALL_HEIGHT = 0.1

CHASSIS_RADIUS = 0.05
CHASSIS_BASE_HEIGHT = 0.038
CHASSIS_TOP_HEIGHT = 0.003

CYLINDER_MESH_SECTIONS = 16

CHASER_COLOR = "1 0.545 0.545 1"
EVADER_COLOR = "0.435 0.702 0.722 1"


def _generate_chassis_mesh(radius: float, height: float) -> bytes:
    mesh = trimesh.creation.cylinder(
        radius=radius, height=height, sections=CYLINDER_MESH_SECTIONS
    )
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()


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
            type="mesh"
            mesh="chassis_base_mesh"
            group="{AGENT_GEOM_GROUP}"
            rgba="0.1 0.1 0.1 1"
            contype="{CONTACTS["chassis"][0]}"
            conaffinity="{CONTACTS["chassis"][1]}"
        />
		<geom
            name="{name}_chassis_top"
            type="mesh"
            mesh="chassis_top_mesh"
            pos="0 0 0.0205"
            group="{AGENT_GEOM_GROUP}"
            rgba="{lid_color}"
            contype="{CONTACTS["chassis"][0]}"
            conaffinity="{CONTACTS["chassis"][1]}"
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
                contype="{CONTACTS["wheel"][0]}"
                conaffinity="{CONTACTS["wheel"][1]}"
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
                contype="{CONTACTS["wheel"][0]}"
                conaffinity="{CONTACTS["wheel"][1]}"
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
                contype="{CONTACTS["visual_only"][0]}"
                conaffinity="{CONTACTS["visual_only"][1]}"
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
                    contype="{CONTACTS["caster_ball"][0]}"
                    conaffinity="{CONTACTS["caster_ball"][1]}"
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
    <framepos name="{name}_position" objtype="body" objname="{name}" />
    <framequat name="{name}_quaternion" objtype="body" objname="{name}" />
    <framelinvel name="{name}_velocity" objtype="body" objname="{name}" />
    <frameangvel name="{name}_angular_velocity" objtype="body" objname="{name}" />
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
        contype="{CONTACTS["floor"][0]}"
        conaffinity="{CONTACTS["floor"][1]}"
    />
    <geom
        name="wall_north"
        type="box"
        pos="0 {arena_half_height + half_wall_thickness} {half_wall_height}"
        size="{arena_half_width + WALL_THICKNESS} {half_wall_thickness} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["wall"][0]}"
        conaffinity="{CONTACTS["wall"][1]}"
    />
    <geom
        name="wall_south"
        type="box"
        pos="0 {-arena_half_height - half_wall_thickness} {half_wall_height}"
        size="{arena_half_width + WALL_THICKNESS} {half_wall_thickness} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["wall"][0]}"
        conaffinity="{CONTACTS["wall"][1]}"
    />
    <geom
        name="wall_east"
        type="box"
        pos="{arena_half_width + half_wall_thickness} 0 {half_wall_height}"
        size="{half_wall_thickness} {arena_half_height} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["wall"][0]}"
        conaffinity="{CONTACTS["wall"][1]}"
    />
    <geom
        name="wall_west"
        type="box"
        pos="{-arena_half_width - half_wall_thickness} 0 {half_wall_height}"
        size="{half_wall_thickness} {arena_half_height} {half_wall_height}"
        group="{WALL_GEOM_GROUP}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["wall"][0]}"
        conaffinity="{CONTACTS["wall"][1]}"
    />
    """


def generate_mjcf(config: EnvironmentConfig) -> tuple[str, dict[str, bytes]]:
    """Build the complete MJCF XML for the environment"""

    assets = {
        "chassis_base.stl": _generate_chassis_mesh(CHASSIS_RADIUS, CHASSIS_BASE_HEIGHT),
        "chassis_top.stl": _generate_chassis_mesh(CHASSIS_RADIUS, CHASSIS_TOP_HEIGHT),
    }

    chaser_body_xml, chaser_actuator_xml, chaser_sensor_xml = _agent_mjcf(
        "chaser", CHASER_COLOR, "0.3 0 0.0299"
    )
    evader_body_xml, evader_actuator_xml, evader_sensor_xml = _agent_mjcf(
        "evader", EVADER_COLOR, "-0.3 0 0.0299"
    )

    mjcf_xml = f"""
    <mujoco model="tag">
        <option integrator="implicitfast"/>
        <asset>
            <mesh name="chassis_base_mesh" file="chassis_base.stl" />
            <mesh name="chassis_top_mesh" file="chassis_top.stl" />
        </asset>
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

    return mjcf_xml, assets
