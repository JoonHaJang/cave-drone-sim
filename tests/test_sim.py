import json
from pathlib import Path

import numpy as np
import pytest

from cavesim.api import Agent, Command
from cavesim.cavegen import PRESETS, grow_graph
from cavesim.config import AeroParams, SimConfig
from cavesim.sim import Simulator

MAP = Path(__file__).resolve().parent.parent / "maps" / "small_2"


def make(agent_cls, n=2, **kw):
    cfg = SimConfig(mission_time_s=kw.pop("t", 30))
    if "aero_params" in kw:
        cfg.aero = kw.pop("aero_params")
    return Simulator(MAP, n, agent_cls, cfg=cfg, **kw)


@pytest.mark.parametrize("size", list(PRESETS))
def test_graph_reaches_target_size(size):
    p = PRESETS[size]
    for seed in range(5):
        _, _, segs, _ = grow_graph(np.random.default_rng(seed), p["n_segments"], p["bounds"])
        assert len(segs) == p["n_segments"]


def test_hover_is_stable_and_sensed():
    seen = {}

    class Hover(Agent):
        def step(self, obs):
            seen[self.id] = obs
            return Command()

    sim = make(Hover, aero="off")
    start = [d.true_pos() for d in sim.drones]
    for _ in range(50):
        sim.step()
    for d, p0 in zip(sim.drones, start):
        assert d.status == "flying"
        assert np.linalg.norm(d.true_pos() - p0) < 0.1
    assert len(seen[0].lidar) > 1000
    assert seen[0].base_link and seen[0].neighbors == [1]
    sim.close()


def test_fast_wall_impact_crashes():
    class Ram(Agent):
        def step(self, obs):
            return Command(vel=(3.0, 3.0, 0.0))

    sim = make(Ram, n=1)
    while sim.step():
        pass
    assert sim.drones[0].status == "crashed"
    sim.close()


def test_perfect_odometry_matches_truth():
    class Fly(Agent):
        def step(self, obs):
            return Command(vel=(0.5, 0.0, 0.0), yaw_rate=0.3)

    sim = make(Fly, n=1, odom="perfect", t=5)
    while sim.step():
        pass
    d = sim.drones[0]
    assert np.linalg.norm(d.odom_pos - d.true_pos()) < 1e-6
    sim.close()


def test_report_rules():
    results = []

    class Reporter(Agent):
        def step(self, obs):
            results.extend(obs.report_results)
            if obs.t < 0.05:
                truth = sim.artifacts[0]
                return Command(reports=[(truth["type"], truth["pos"]), ("rope", [999, 0, 0])])
            return Command()

    sim = make(Reporter, n=1, t=1)
    while sim.step():
        pass
    assert [ok for _, ok, _ in results] == [True, False]
    assert sim.score()["artifacts_found"] == 1
    sim.close()


def test_imu_hover_matches_spec():
    """At hover the accelerometer reads ~+g on z with the configured white noise."""
    samples = []

    class Hover(Agent):
        def step(self, obs):
            if obs.t > 0.5:
                samples.append(obs.imu)
            return Command()

    sim = make(Hover, n=1, t=5, aero="off")
    while sim.step():
        pass
    imu = np.concatenate(samples)
    ip = sim.cfg.imu
    assert abs(imu[:, 3].mean() - 9.81) < 0.5
    assert imu[:, 1:4].std(0) == pytest.approx(ip.accel_noise_density * np.sqrt(ip.rate_hz), rel=0.25)
    assert imu[:, 4:7].std(0) == pytest.approx(ip.gyro_noise_density * np.sqrt(ip.rate_hz), rel=0.25)
    sim.close()


def test_sensors_tilt_and_shapes():
    seen = []

    class Dash(Agent):
        def step(self, obs):
            seen.append(obs)
            return Command(vel=(2.0, 0.0, 0.0) if obs.t < 1 else (0.0, 0.0, 0.0))

    sim = make(Dash, n=1, t=2, lidar="lite")
    while sim.step():
        pass
    lp = sim.cfg.lidar
    assert seen[0].lidar_ranges.shape == (lp.channels * lp.azimuth_steps,)
    assert max(abs(o.pitch) for o in seen) > np.radians(10)      # airframe tilts to accelerate
    assert all(o.range_down > 0.1 for o in seen)
    assert abs(seen[-1].baro_alt) < 1.5
    sim.close()


def test_odom_off_hides_position():
    seen = []

    class Look(Agent):
        def step(self, obs):
            seen.append(obs)
            return Command()

    sim = make(Look, n=1, t=1, odom="off")
    sim.step()
    assert seen[0].pos is None and seen[0].vel is None and np.isfinite(seen[0].yaw)
    sim.close()


def test_gusts_push_but_autopilot_holds():
    class Hover(Agent):
        def step(self, obs):
            return Command()

    sim = make(Hover, n=1, t=10)
    p0 = sim.drones[0].true_pos()
    moved = 0.0
    while sim.step():
        moved = max(moved, np.linalg.norm(sim.drones[0].true_pos() - p0))
    assert sim.drones[0].status == "flying"
    assert 0.01 < moved < 0.6
    sim.close()


def test_ceiling_effect_pulls_up():
    sim = make(Agent, n=1, aero_params=AeroParams(draft_std=0.0, wall_turb_std=0.0))
    d = sim.drones[0]
    hit = sim.pb.rayTest([0, 0, 0], [0, 0, 20])[0]
    top = 20 * hit[2]
    sim.pb.resetBasePositionAndOrientation(d.body, [0, 0, top - sim.cfg.drone.radius - 0.1],
                                           [0, 0, 0, 1])
    sim.aero_disturbance(sim.cfg.dt)
    ap = sim.cfg.aero
    assert d.clearance == pytest.approx(0.1, abs=0.05)
    assert d.aero_acc[2] > 0.5 * ap.ceiling_gain * 9.81
    sim.close()


def test_speleothems_are_solid():
    meta = json.loads((MAP / "map.json").read_text())
    spikes = [s for s in meta["speleothems"] if s["kind"] == "stalactite"]
    assert len(spikes) > 10
    sim = make(Agent, n=1)
    s = max(spikes, key=lambda s: s["length"])
    x, y, z = s["base"]
    z -= 0.3 * s["length"]          # a third of the way down the spike
    hit = sim.pb.rayTest([x - 1.5, y, z], [x + 1.5, y, z])[0]
    assert hit[0] == sim.cave and hit[2] < 0.5
    sim.close()
