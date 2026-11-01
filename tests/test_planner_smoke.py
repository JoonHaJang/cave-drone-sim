"""The planner flies the rescue course for a while without raising (errors inside an agent are
caught by the simulator and leave the drone hovering, so a broken agent otherwise looks idle)."""
from pathlib import Path

import numpy as np

from agents.planner import PlannerAgent
from cavesim.sim import Simulator

MAPS = Path(__file__).resolve().parent.parent / "maps"


def test_planner_enters_the_cave_without_errors():
    sim = Simulator(MAPS / "rescue_easy", 1, PlannerAgent, lidar="lite", seed=0)
    d = sim.drones[0]
    start = d.true_pos().copy()
    while sim.step() and sim.t < 25.0:
        assert d.error is None, d.error
    assert np.linalg.norm(d.true_pos() - start) > 10.0


def test_search_speed_still_sweeps_every_point_within_the_look_distance():
    # spinning at YAW_RATE_MAX, flying straight at search_speed: a point SEARCH_LOOK of the detector's
    # range off the track comes into its 120 deg, 6 m view whatever the heading we start at
    import numpy as np
    import agents.planner as P
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": [0, 0, 0], "detector_range": 6.0,
                               "detector_half_fov_deg": 60.0, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3})
    v, dt, pt = ag.search_speed(), 0.1, np.array([0.0, 0.97 * P.SEARCH_LOOK * 6.0])   # the bound is a tangency
    for yaw0 in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        seen = False
        for t in np.arange(-20, 20, dt):
            rel = pt - np.array([v * t, 0.0])
            yaw = yaw0 + P.YAW_RATE_MAX * t
            bearing = np.arctan2(rel[1], rel[0]) - yaw
            seen |= np.linalg.norm(rel) <= 6.0 and abs((bearing + np.pi) % (2 * np.pi) - np.pi) <= np.radians(60)
        assert seen, yaw0
    assert v > 1.2                                                # faster than the 1 m/s it was
