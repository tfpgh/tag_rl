from typing import Literal

from environment.config import EnvironmentConfig

SceneMode = Literal["training", "render"]

GEOM_GROUP_ENVIRONMENT = 0
GEOM_GROUP_AGENT = 1
GEOM_GROUP_FLOOR = 2

CON_BM_VISUAL_ONLY = 0b00000
CON_BM_FLOOR = 0b00001
CON_BM_WHEEL = 0b00010
CON_BM_CASTER_BALL = 0b00100
CON_BM_CHASSIS = 0b01000

CONTACTS = {
    "floor": (CON_BM_FLOOR, CON_BM_WHEEL | CON_BM_CASTER_BALL | CON_BM_CHASSIS),
    "wheel": (CON_BM_WHEEL, CON_BM_FLOOR),
    "caster_ball": (CON_BM_CASTER_BALL, CON_BM_FLOOR),
    "chassis": (CON_BM_CHASSIS, CON_BM_FLOOR),
    "visual_only": (CON_BM_VISUAL_ONLY, CON_BM_VISUAL_ONLY),
}

WALL_THICKNESS = 0.02
WALL_HEIGHT = 0.1

CHASSIS_BASE_HEIGHT = 0.038
CHASSIS_TOP_HEIGHT = 0.003

OBSTACLE_COLOR = "0.6 0.6 0.65 1"

CHASER_COLOR = "1 0.545 0.545 1"
EVADER_COLOR = "0.435 0.702 0.722 1"


def _agent_mjcf(
    name: str, lid_color: str, starting_pos: str = "0 0 0.0299"
) -> tuple[str, str]:
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
            group="{GEOM_GROUP_AGENT}"
            size="0.05 0.019"
            rgba="0.1 0.1 0.1 1"
            contype="{CONTACTS["chassis"][0]}"
            conaffinity="{CONTACTS["chassis"][1]}"
        />
        <geom
            name="{name}_chassis_top"
            type="cylinder"
            pos="0 0 0.0205"
            size="0.05 0.0015"
            group="{GEOM_GROUP_AGENT}"
            rgba="{lid_color}"
            contype="{CONTACTS["chassis"][0]}"
            conaffinity="{CONTACTS["chassis"][1]}"
        />
        <geom
            name="{name}_bumper"
            type="capsule"
            pos="0.0415 0 -0.021"
            euler="90 0 0"
            size="0.00635 0.010"
            group="{GEOM_GROUP_AGENT}"
            rgba="0.1 0.1 0.1 1"
            contype="{CONTACTS["chassis"][0]}"
            conaffinity="{CONTACTS["chassis"][1]}"
        />
        <body name="{name}_left_wheel" pos="0 0.037123 -0.0099">
            <joint
                name="{name}_left_wheel_joint"
                type="hinge"
                axis="0 1 0"
                damping="0.0001"
                frictionloss="0.025"
            />
            <geom
                name="{name}_left_wheel_geom"
                type="cylinder"
                size="0.020 0.0020"
                euler="90 0 0"
                mass="0.00425"
                group="{GEOM_GROUP_AGENT}"
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
                damping="0.0001"
                frictionloss="0.025"
            />
            <geom
                name="{name}_right_wheel_geom"
                type="cylinder"
                size="0.020 0.0020"
                euler="90 0 0"
                mass="0.00425"
                group="{GEOM_GROUP_AGENT}"
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
                group="{GEOM_GROUP_AGENT}"
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
                    group="{GEOM_GROUP_AGENT}"
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
    <general
        name="{name}_left_motor"
        joint="{name}_left_wheel_joint"
        gainprm="0.126"
        biasprm="0 0 -0.0015"
        gaintype="fixed"
        biastype="affine"
        dyntype="none"
        ctrlrange="-1 1"
        ctrllimited="true"
    />
    <general
        name="{name}_right_motor"
        joint="{name}_right_wheel_joint"
        gainprm="0.126"
        biasprm="0 0 -0.0015"
        gaintype="fixed"
        biastype="affine"
        dyntype="none"
        ctrlrange="-1 1"
        ctrllimited="true"
    />
    """

    return body_xml, actuator_xml


def _floor_xml(config: EnvironmentConfig) -> str:
    arena_half_width = config.arena_width / 2
    arena_half_height = config.arena_height / 2
    return f"""
    <geom
        name="floor"
        type="plane"
        pos="0 0 0"
        size="{arena_half_width + WALL_THICKNESS} {arena_half_height + WALL_THICKNESS} 0.04"
        group="{GEOM_GROUP_FLOOR}"
        rgba="0.2 0.2 0.25 1"
        friction="0.1 0.005 0.0001"
        contype="{CONTACTS["floor"][0]}"
        conaffinity="{CONTACTS["floor"][1]}"
    />
    """


def _wall_visual_xml(config: EnvironmentConfig) -> str:
    arena_half_width = config.arena_width / 2
    arena_half_height = config.arena_height / 2
    half_wall_thickness = WALL_THICKNESS / 2
    half_wall_height = WALL_HEIGHT / 2
    return f"""
    <geom
        name="wall_north"
        type="box"
        pos="0 {arena_half_height + half_wall_thickness} {half_wall_height}"
        size="{arena_half_width + WALL_THICKNESS} {half_wall_thickness} {half_wall_height}"
        group="{GEOM_GROUP_ENVIRONMENT}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["visual_only"][0]}"
        conaffinity="{CONTACTS["visual_only"][1]}"
    />
    <geom
        name="wall_south"
        type="box"
        pos="0 {-arena_half_height - half_wall_thickness} {half_wall_height}"
        size="{arena_half_width + WALL_THICKNESS} {half_wall_thickness} {half_wall_height}"
        group="{GEOM_GROUP_ENVIRONMENT}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["visual_only"][0]}"
        conaffinity="{CONTACTS["visual_only"][1]}"
    />
    <geom
        name="wall_east"
        type="box"
        pos="{arena_half_width + half_wall_thickness} 0 {half_wall_height}"
        size="{half_wall_thickness} {arena_half_height} {half_wall_height}"
        group="{GEOM_GROUP_ENVIRONMENT}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["visual_only"][0]}"
        conaffinity="{CONTACTS["visual_only"][1]}"
    />
    <geom
        name="wall_west"
        type="box"
        pos="{-arena_half_width - half_wall_thickness} 0 {half_wall_height}"
        size="{half_wall_thickness} {arena_half_height} {half_wall_height}"
        group="{GEOM_GROUP_ENVIRONMENT}"
        rgba="0.8 0.8 0.8 1"
        contype="{CONTACTS["visual_only"][0]}"
        conaffinity="{CONTACTS["visual_only"][1]}"
    />
    """


def _obstacle_visual_xml(config: EnvironmentConfig, index: int) -> str:
    half_w = config.obstacle_width / 2
    half_h = WALL_HEIGHT / 2
    name = f"obstacle_{index}"
    return f"""
    <body name="{name}" mocap="true" pos="0 0 -10">
        <geom
            name="{name}_geom"
            type="box"
            size="{half_w} {half_w} {half_h}"
            group="{GEOM_GROUP_ENVIRONMENT}"
            rgba="{OBSTACLE_COLOR}"
            contype="{CONTACTS["visual_only"][0]}"
            conaffinity="{CONTACTS["visual_only"][1]}"
        />
    </body>
    """


def generate_mjcf(config: EnvironmentConfig, mode: SceneMode = "training") -> str:
    chaser_body_xml, chaser_actuator_xml = _agent_mjcf(
        "chaser", CHASER_COLOR, "0.3 0 0.0299"
    )
    evader_body_xml, evader_actuator_xml = _agent_mjcf(
        "evader", EVADER_COLOR, "-0.3 0 0.0299"
    )

    extras = ""
    if mode == "render":
        obstacle_bodies = "\n".join(
            _obstacle_visual_xml(config, i) for i in range(config.max_obstacles)
        )
        extras = f"{_wall_visual_xml(config)}\n{obstacle_bodies}"

    return f"""
    <mujoco model="tag_{mode}">
        <option timestep="0.005" integrator="implicitfast" solver="Newton" iterations="8" ls_iterations="8" ccd_iterations="100"/>
        <asset>
            <material name="default" rgba="1 1 1 1" />
        </asset>
        <visual>
            <global offwidth="1920" offheight="1080"/>
            <headlight ambient="0.5 0.5 0.5" />
        </visual>
        <statistic extent="{config.arena_width * 0.85}" center="0 0 0"/>
        <worldbody>
            {_floor_xml(config)}
            {extras}
            {chaser_body_xml}
            {evader_body_xml}
        </worldbody>
        <actuator>
            {chaser_actuator_xml}
            {evader_actuator_xml}
        </actuator>
    </mujoco>
    """
