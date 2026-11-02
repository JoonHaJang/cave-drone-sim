"""Every frame convention crossing the sim -> agent boundary, against the simulator's truth.

Perfect odometry and an agile flight (tilts past 20 deg, yaw at the rate limit): the observed
attitude must rebuild the airframe's rotation, LiDAR returns placed with it must land on the
mesh, deskewing must help, gyro rates are body-frame, and commands are odom-frame."""
import dataclasses
from pathlib import Path

import numpy as np

import cavesim.config as C
from agents.planner import deskew
from cavesim.api import Agent, Command, body_rotation
from cavesim.flyable import Course
from cavesim.sim import Simulator

MAP = Path(__file__).resolve().parents[1] / "maps" / "rescue_easy"


def angle(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


class Wobble(Agent):
    def step(self, obs):
        t = obs.t
        return Command(vel=(2.5 * np.cos(0.8 * t), 2.5 * np.sin(1.6 * t), 0.4 * np.sin(0.5 * t)),
                       yaw_rate=0.8 * np.sign(np.sin(0.3 * t)))


def fly(distortion, seconds=12.0):
    cfg = C.SimConfig()
    cfg.lidar = dataclasses.replace(C.LIDAR_PRESETS["lite"], dust_density=0.0, dropout=0.0,
                                    precision=((1e9, 0.0),), motion_distortion=distortion)
    sim = Simulator(MAP, 1, Wobble, cfg=cfg, odom="perfect", seed=1)
    d, rows = sim.drones[0], []
    while sim.t < seconds:
        v0 = d.true_vel()
        sim.step()
        _, R, p = d.last_scan
        rows.append((d.last_obs, R, p, v0))
    sim.close()
    return rows


def test_observed_attitude_and_lidar_frames():
    rows = fly(distortion=False)
    tilt = max(np.degrees(np.arccos(R[2, 2])) for _, R, *_ in rows)
    assert tilt > 15
    # ZYX roll / pitch / yaw rebuild the airframe's rotation (up to 0.29 deg roll/pitch noise)
    err = [angle(body_rotation(o.roll, o.pitch, o.yaw), R) for o, R, *_ in rows]
    assert max(err) < 1.5
    # odom-frame velocity, at the observation's own time
    assert np.median([np.linalg.norm(o.vel - v0) for o, _, _, v0 in rows]) < 0.1
    # returns placed with the observed pose land on the rock
    course = Course(MAP)
    pts = np.vstack([o.lidar @ body_rotation(o.roll, o.pitch, o.yaw).T + o.pos for o, *_ in rows[::6]])
    assert np.percentile(course.distance(pts), 95) < 0.15


def test_gyro_is_body_frame_and_yaw_rate_sign():
    rows = fly(distortion=False, seconds=6.0)
    for (_, R0, *_), (o, R, *_) in zip(rows[1:], rows[2:]):   # the IMU samples of a tick lead up to its observation
        if len(o.imu) < 2:
            continue
        Rk, h = R0.copy(), np.diff(o.imu[:, 0]).mean()
        for w in o.imu[:, 4:7]:
            th = np.linalg.norm(w) * h
            if th > 0:
                k = w / np.linalg.norm(w)
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                Rk = Rk @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)
        assert angle(Rk, R) < 0.5
    turns = [np.sign(np.angle(np.exp(1j * (b[0].yaw - a[0].yaw)))) for a, b in zip(rows, rows[1:])]
    assert np.mean(np.array(turns) == np.sign(np.sin(0.3 * np.array([a[0].t for a in rows[:-1]])))) > 0.9


def test_deskew_moves_returns_onto_the_rock():
    rows = fly(distortion=True)
    course = Course(MAP)
    raw, fixed = [], []
    for o, *_ in rows[::6]:
        R = body_rotation(o.roll, o.pitch, o.yaw)
        raw.append(course.distance(o.lidar @ R.T + o.pos))
        fixed.append(course.distance(deskew(o) @ R.T + o.pos))
    assert np.percentile(np.concatenate(fixed), 95) < 0.5 * np.percentile(np.concatenate(raw), 95)
