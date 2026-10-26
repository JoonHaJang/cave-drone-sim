from pathlib import Path

import numpy as np
import pytest

from cavesim.api import Agent, Command
from cavesim.cavegen import PRESETS, grow_graph
from cavesim.config import SimConfig
from cavesim.sim import Simulator

MAP = Path(__file__).resolve().parent.parent / "maps" / "small_2"


def make(agent_cls, n=2, **kw):
    return Simulator(MAP, n, agent_cls, cfg=SimConfig(mission_time_s=kw.pop("t", 30)), **kw)


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

    sim = make(Hover)
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
