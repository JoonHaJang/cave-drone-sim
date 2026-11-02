"""Each fidelity effect added on top of the datasheet models does what it claims."""
from pathlib import Path

import numpy as np

from cavesim.api import Agent, Command
from cavesim.config import IMU_PRESETS, LIDAR_PRESETS, SimConfig
from cavesim.sensors import Imu, Lidar, degeneracy, rot_z
from cavesim.sim import Simulator

MAP = Path(__file__).resolve().parent.parent / "maps" / "small_2"
EXACT = LIDAR_PRESETS["ideal"]   # no noise, no dropouts: ranges are pure geometry


def world():
    sim = Simulator(MAP, 1, Agent, cfg=SimConfig())
    return sim, sim.drones[0].true_pos()


def test_motion_distortion_fires_each_column_from_its_own_pose():
    sim, pos = world()
    lidar = Lidar(EXACT.__class__(**{**EXACT.__dict__, "motion_distortion": True}), 0.22)
    rng = np.random.default_rng(0)
    a, b = (pos, np.eye(3)), (pos, rot_z(0.3))
    smeared, _ = lidar.scan(sim.pb, [a] * 12 + [b] * 12, rng)
    only_a, _ = lidar.scan(sim.pb, [a], rng)
    only_b, _ = lidar.scan(sim.pb, [b], rng)
    first = lidar.fire < lidar.p.azimuth_steps // 2
    az = lidar.azimuth[np.arange(len(smeared)) % lidar.p.azimuth_steps]
    assert az[lidar.fire == 0][0] == -np.pi and az[lidar.fire == 1][0] > 3.0   # clockwise from behind
    assert np.array_equal(smeared[first], only_a[first])
    assert np.array_equal(smeared[~first], only_b[~first])
    t = lidar.ray_times(1.0)
    assert t.min() > 0.9 and t.max() == 1.0
    sim.close()


def test_dust_follows_beer_lambert_and_only_near_the_floor():
    sim, pos = world()
    lp = EXACT.__class__(**{**EXACT.__dict__, "dust_density": 0.3, "dust_radius": 2.0, "dust_height": 1.0})
    lidar = Lidar(lp, 0.22)
    rng = np.random.default_rng(1)
    clean, _ = lidar.scan(sim.pb, [(pos, np.eye(3))], rng, floor_gap=5.0)
    dusty, _ = lidar.scan(sim.pb, [(pos, np.eye(3))], rng, floor_gap=0.0)
    far = clean > lp.dust_radius                       # beams the dust could intercept fully
    frac = np.mean(dusty[far] < clean[far])
    expect = 1 - np.exp(-lp.dust_density * (lp.dust_radius - lidar.start))
    assert abs(frac - expect) < 0.03
    assert np.all(dusty[dusty < clean] < lp.dust_radius)
    sim.close()


def test_degeneracy_flags_the_tunnel_axis():
    ang = np.random.default_rng(2).uniform(0, 2 * np.pi, 500)
    tube = np.stack([np.zeros_like(ang), np.cos(ang), np.sin(ang)], 1)   # walls of a tunnel along x
    V, w = degeneracy(tube, 0.05)
    assert abs(V[:, np.argmax(w)] @ [1, 0, 0]) > 0.99 and w.max() > 0.99
    assert np.sort(w)[:2].max() == 0.0
    _, w_room = degeneracy(np.vstack([tube, np.tile([[1, 0, 0], [-1, 0, 0]], (100, 1))]), 0.05)
    assert w_room.max() == 0.0


def test_odometry_drifts_down_a_degenerate_tunnel_while_hovering():
    class Hover(Agent):
        def step(self, obs):
            return Command()

    sim = Simulator(MAP, 1, Hover, cfg=SimConfig(mission_time_s=20), aero="off")
    d = sim.drones[0]
    ang = np.linspace(0, 2 * np.pi, 400)
    tube = np.stack([np.zeros_like(ang), np.cos(ang), np.sin(ang)], 1)
    scan = sim.lidar.scan
    sim.lidar.scan = lambda *a, **k: (scan(*a, **k)[0], tube)   # every sweep: smooth tunnel along x
    while sim.step():
        pass
    err = d.odom_pos - d.true_pos()
    assert abs(err[0]) > 5 * max(abs(err[1]), abs(err[2]))
    sim.close()


def test_imu_quantizes_saturates_and_vibrates():
    rng = np.random.default_rng(3)
    ip = IMU_PRESETS["flight"]
    imu = Imu(ip, rng)
    hover = np.array([imu.sample(0, np.array([0, 0, 9.81]), np.zeros(3)) for _ in range(2400)])
    acc = hover[:, 1:4]
    assert np.allclose(acc / ip.accel_lsb, np.round(acc / ip.accel_lsb), atol=1e-6)
    quiet = Imu(IMU_PRESETS["datasheet"], np.random.default_rng(3))
    calm = np.array([quiet.sample(0, np.array([0, 0, 9.81]), np.zeros(3)) for _ in range(2400)])
    assert acc.std(0).min() > 3 * calm[:, 1:4].std(0).max()
    big = imu.sample(0, np.array([1e4, 0, 0]), np.array([0, 0, 1e3]))
    assert abs(big[1]) <= ip.accel_range + ip.accel_lsb and abs(big[6]) <= ip.gyro_range + ip.gyro_lsb
