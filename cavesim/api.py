"""What an agent sees and what it may command. This file is the whole student contract.

Frames
------
* odom frame: where the drone *thinks* it is. It starts equal to the world frame
  (base at the origin, yaw 0) and drifts away from it as the drone flies.
* body frame: x forward, y left, z up, rotated by yaw only (the LiDAR is gimbal-levelled).

Ground truth positions are never given to agents.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    type: str                 # one of config.ARTIFACT_TYPES
    rel_pos: np.ndarray       # (3,) body frame, metres, noisy


@dataclass
class Observation:
    t: float                  # mission time, s
    drone_id: int
    pos: np.ndarray           # (3,) odom-frame position estimate
    yaw: float                # odom-frame yaw estimate, rad
    vel: np.ndarray           # (3,) velocity in body frame (from the flow sensor)
    battery: float            # 1.0 full, 0.0 empty (motors cut)
    lidar: np.ndarray         # (N, 3) hit points in body frame; misses are dropped
    lidar_dirs: np.ndarray    # (M, 3) unit beam directions (body); fixed, handy for free-space carving
    lidar_ranges: np.ndarray  # (M,) range per beam, np.inf for no return
    detections: list          # list[Detection]
    inbox: list               # list[(sender_id, payload)] delivered this tick
    neighbors: list           # drone ids you can talk to right now
    base_link: bool           # True if the base station is in direct range
    report_results: list      # list[(report, accepted: bool, reason: str)] from last tick
    landed: bool


@dataclass
class Command:
    vel: Any = (0.0, 0.0, 0.0)   # desired velocity in *odom* frame, m/s (clipped to limits)
    yaw_rate: float = 0.0        # rad/s, positive = counter-clockwise
    land: bool = False           # descend and cut motors on touchdown; stays landed
    send: list = field(default_factory=list)     # list[(dest, payload)], dest = drone id or "all"
    reports: list = field(default_factory=list)  # list[(type, pos)], pos in odom frame; needs base_link


class Agent:
    """Subclass this. One instance is created per drone."""

    def __init__(self, drone_id: int, n_drones: int, info: dict):
        """info: limits and sensor specs (see Simulator.agent_info)."""
        self.id = drone_id
        self.n = n_drones
        self.info = info

    def step(self, obs: Observation) -> Command:
        return Command()

    def debug(self) -> dict:
        """Optional extras for the viewer, e.g. {"path": [[x,y,z],...], "goal": [x,y,z],
        "points": [[x,y,z],...]} in odom frame. Called at viewer rate, keep it cheap."""
        return {}
