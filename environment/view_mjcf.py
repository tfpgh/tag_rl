import mujoco
import mujoco.viewer

from environment.config import EnvironmentConfig
from environment.mjcf import generate_mjcf

config = EnvironmentConfig()
xml = generate_mjcf(config)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

mujoco.viewer.launch(model, data)
