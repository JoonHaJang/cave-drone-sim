"""What an agent sees and what it may command. This file is the whole student contract.

Frames
------
* odom frame: where the drone *thinks* it is. It starts equal to the world frame
  (base at the origin, yaw 0) and drifts away from it as the drone flies.
* body frame: x forward, y left, z up, fixed to the airframe. It rolls and pitches
  when the drone accelerates, and every sensor is mounted in it. Use
  body_to_odom() to put LiDAR points into the odom frame.

Ground truth positions are never given to agents. Sensor specs: cavesim/config.py.
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
    # --- LiDAR-inertial odometry output (drifts; pos and vel are None with --odom off)
    pos: Any                  # (3,) odom-frame position estimate
    vel: Any                  # (3,) odom-frame velocity estimate
    yaw: float                # odom-frame heading estimate, rad (drifts like the gyro)
    # --- attitude from the flight controller's IMU filter (small noise)
    roll: float
    pitch: float
    # --- raw sensors
    imu: np.ndarray           # (K, 7) samples since last tick: t, ax, ay, az [m/s^2], gx, gy, gz [rad/s];
                              #   accelerometer reads specific force (about +9.81 on z when hovering)
    lidar: np.ndarray         # (N, 3) hit points in body frame; misses are dropped
    lidar_dirs: np.ndarray    # (M, 3) unit beam directions (body), M = channels * azimuth_steps
    lidar_ranges: np.ndarray  # (M,) range per beam, np.inf for no return.
                              #   reshape(channels, azimuth_steps) gives the range image
    range_down: float         # downward rangefinder (body -z), m, np.inf if out of range
    baro_alt: float           # barometric altitude relative to take-off, m
    battery: float            # 1.0 full, 0.0 empty (motors cut)
    # --- mission
    detections: list          # list[Detection]
    inbox: list               # list[(sender_id, payload)] delivered this tick
    neighbors: list           # drone ids you can talk to right now
    base_link: bool           # True if the base station is in direct range
    report_results: list      # list[(report, accepted: bool, reason: str)] from last tick
    landed: bool


def body_rotation(roll, pitch, yaw):
    """Body -> odom rotation matrix (ZYX Euler)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def body_to_odom(obs, pts):
    """Body-frame points (N, 3) -> odom frame, using the estimated pose."""
    return np.asarray(pts) @ body_rotation(obs.roll, obs.pitch, obs.yaw).T + obs.pos


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
