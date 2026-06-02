# -*- coding: utf-8 -*-
"""
Habitat-Sim: capture RGB in MP3D glb scenes at agent poses (decoupled from KB, reusable).
"""

from pathlib import Path
from typing import Union

import numpy as np


def make_mp3d_sim(
    scene_glb: Union[str, Path],
    width: int,
    height: int,
    hfov: float,
    *,
    near: float = 0.01,
    far: float = 1000.0,
):
    """
    Load glb and create a Simulator with an RGB camera.
    Call ``sim.close()`` when done, or use try/finally.
    """
    import habitat_sim
    from habitat_sim import SimulatorConfiguration
    from habitat_sim.agent import AgentConfiguration
    from habitat_sim.sensor import CameraSensorSpec, SensorType

    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = str(Path(scene_glb).resolve())
    rgb_spec = CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.hfov = hfov
    rgb_spec.near = near
    rgb_spec.far = far
    rgb_spec.position = [0.0, 0.0, 0.0]
    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def render_rgb_at_pose(
    sim,
    position: np.ndarray,
    quat_xyzw: np.ndarray,
) -> np.ndarray:
    """
    Place agent at position + quaternion (x,y,z,w); return RGB uint8 with shape (H,W,3).
    """
    import habitat_sim
    from habitat_sim.utils.common import quat_from_coeffs

    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32).reshape(3)
    state.rotation = quat_from_coeffs(np.asarray(quat_xyzw, dtype=np.float64).tolist())
    sim.get_agent(0).set_state(state)
    rgb = sim.get_sensor_observations()["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb
