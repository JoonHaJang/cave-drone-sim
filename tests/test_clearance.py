"""Clearance map (certified tube): a voxel is tight by its distance to the seen rock surface."""
import numpy as np
import pytest

import agents.planner as P


@pytest.fixture(autouse=True)
def clearance_by_distance(monkeypatch):
    monkeypatch.setattr(P, "CLEARANCE", "dist")


def wall_map(x=1.0, sweeps=P.SURF_N):
    m = P.VoxelMap()
    y, z = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
    pts = np.c_[np.full(y.size, x), y.ravel(), z.ravel()]
    for _ in range(sweeps):
        m.add_surface(pts)
    return m


def key(m, p):
    return int(P.keys_of(m.voxel(p)))


def test_tight_by_distance_to_surface():
    m = wall_map(1.0)                                  # surface cells centred at x = 1.125
    assert key(m, [0.25, 0.1, 0.1]) not in m.blocked   # 0.875 m off
    assert key(m, [0.75, 0.1, 0.1]) in m.blocked       # 0.375 m off
    assert key(m, [1.25, 0.1, 0.1]) in m.blocked       # in the rock


def test_one_sweep_is_not_rock():
    m = wall_map(1.0, sweeps=1)                        # dust or a ghost: not yet surface
    assert key(m, [0.75, 0.1, 0.1]) not in m.blocked


def test_new_surface_invalidates_cached_clearance():
    m = P.VoxelMap()
    k = key(m, [0.75, 0.1, 0.1])
    assert k not in m.blocked                          # nothing seen: clear (and cached)
    y, z = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
    pts = np.c_[np.full(y.size, 1.0), y.ravel(), z.ravel()]
    m.add_surface(pts)
    m.add_surface(pts)
    assert k in m.blocked


def test_guard_sees_mapped_rock_out_of_lidar_view():
    m = wall_map(1.0)
    q = m.surface_near(np.array([0.3, 0.0, 0.0]), 1.2)
    assert len(q) and np.all(np.linalg.norm(q - [0.3, 0.0, 0.0], axis=1) < 1.2)


def test_way_back_leads_out_of_the_margin_under_a_ceiling(monkeypatch):
    monkeypatch.setattr(P, "CLEARANCE", None)
    m = P.VoxelMap()
    for x in np.arange(-3, 3, 0.5):                      # free box, rock slab at z = 1.25 as the ceiling
        for y in np.arange(-3, 3, 0.5):
            for z in np.arange(-3, 1.0, 0.5):
                m.free.add(key(m, [x + 0.1, y + 0.1, z + 0.1]))
            k = key(m, [x + 0.1, y + 0.1, 1.1])
            m.update([k], 10.0)
    m.blocked = m.block_n
    d = m.way_back(np.array([0.1, 0.1, 0.6]))            # in the margin right under it
    assert d is not None and d[2] < -0.8                 # down, out of the margin
    assert m.way_back(np.array([0.1, 0.1, -1.4])) is None


def test_open_way_leaves_a_dead_end_through_its_mouth():
    # a pipe along +x closed at x = -1: rock all round within 0.6 m, open only toward +x
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.c_[np.zeros(24), 0.6 * np.cos(a), 0.6 * np.sin(a)]
    wall = np.vstack([ring + [x, 0, 0] for x in np.arange(-1.0, 6.0, 0.25)])
    cap = np.c_[np.full(25, -1.0), *np.meshgrid(np.linspace(-.6, .6, 5), np.linspace(-.6, .6, 5))].reshape(-1, 3) \
        if False else np.array([[-1.0, y, z] for y in np.linspace(-.6, .6, 5) for z in np.linspace(-.6, .6, 5)])
    w = P.open_way(np.vstack([wall, cap, [[8.0, 0, 0]]]))
    assert w is not None and w[0] > 0.9


def test_open_way_keeps_the_preferred_way_when_it_is_open_and_drops_it_when_not():
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.c_[np.zeros(24), 0.6 * np.cos(a), 0.6 * np.sin(a)]
    pipe = np.vstack([ring + [x, 0, 0] for x in np.arange(-6.0, 6.0, 0.25)])   # open both ways along x
    assert P.open_way(pipe, np.array([-1.0, 0, 0]))[0] < -0.9
    assert P.open_way(pipe, np.array([0, 0, 1.0]))[0] ** 2 > 0.8                # up is rock: along the pipe


@pytest.mark.parametrize("half, ok", [(0.6, True), (0.4, False)])
def test_measured_clearance_lets_a_margin_voxel_go_only_when_the_rock_is_far_enough(monkeypatch, half, ok):
    monkeypatch.setattr(P, "CLEARANCE", "measured")
    m = P.VoxelMap()
    x, z = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
    for y in (0.25 - half, 0.25 + half):                 # two walls, either side of the voxel centred at y = 0.25
        pts = np.c_[x.ravel(), np.full(x.size, y), z.ravel()]
        for _ in range(P.SURF_N):
            m.add_surface(pts)
        m.update(np.unique(P.keys_of(m.voxel(pts))).tolist(), 10.0)
    k = key(m, [0.1, 0.1, 0.1])
    assert k in m.block_n                                # margin by the 3x3x3 rule either way
    assert (k not in m.blocked) == ok


@pytest.mark.parametrize("range_down, track, sinks", [(10.0, True, True), (1.05, True, False), (10.0, False, False)])
def test_guard_trusts_the_rangefinder_over_mapped_cells_below(monkeypatch, range_down, track, sinks):
    # a mapped slab 1 m below: stale only if the rangefinder sees past it and we came up that way (a hole),
    # else a floor (or a ledge whose crack the beam slips through)
    from types import SimpleNamespace
    monkeypatch.setattr(P, "GUARD_MAP", True)
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3})
    x, y = np.meshgrid(np.arange(-1.2, 1.2, 0.1), np.arange(-1.2, 1.2, 0.1))
    slab = np.c_[x.ravel(), y.ravel(), np.full(x.size, -1.0)]
    for _ in range(P.SURF_N):
        ag.map.add_surface(slab)
    ag.mode = "home"                                        # the rule is for the way home only
    if track:
        ag.nodes.append(np.array([0.3, 0.0, -3.0]))
    obs = SimpleNamespace(pos=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0, lidar=np.zeros((0, 3)), range_down=range_down)
    v = ag.guard(obs, np.array([0.0, 0.0, -3.0]))
    monkeypatch.setattr(P, "RF_BEAM_R", 0.0)               # the rule off
    v_off = ag.guard(obs, np.array([0.0, 0.0, -3.0]))
    assert (v[2] < v_off[2] - 0.05) == sinks               # faster down only where the beam saw no slab


def test_escape_with_no_tube_open_goes_back_where_a_way_out_was_last_open():
    # wedged in a crack, rock 0.25 m off all round: the pick among closed tubes flipped every tick
    # (hard 36, notes 13); we flew in, so we leave the way we came
    from types import SimpleNamespace
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3})
    u = np.random.default_rng(0).normal(size=(2000, 3))
    crack = 0.25 * u / np.linalg.norm(u, axis=1)[:, None]
    obs = SimpleNamespace(pos=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0, lidar=crack)
    ag.open_at = np.array([0.0, 0.0, -2.0])
    assert ag.escape_way(obs, np.array([1.0, 0, 0]))[2] < -0.99
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)               # a pipe along x: open, so open_way decides
    pipe = np.vstack([np.c_[np.zeros(24), 0.6 * np.cos(a), 0.6 * np.sin(a)] + [x, 0, 0] for x in np.arange(-6.0, 6.0, 0.25)])
    ag.escape = None
    assert abs(ag.escape_way(SimpleNamespace(pos=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0, lidar=pipe), np.array([1.0, 0, 0]))[0]) > 0.9
    assert np.allclose(ag.open_at, 0.0)


def test_search_keeps_the_way_to_its_goal_when_a_replan_misses_it(monkeypatch):
    # a neck's voxels flickering free / rock made every other replan fail: the partial plan sent the
    # drone 15 m off after another frontier and back (hard 16, 22 at 20 m error, notes 13)
    from types import SimpleNamespace
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": [0, 0, 0], "target_error_m": 20.0, "detector_range": 6.0, "detector_half_fov_deg": 60.0,
                               "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3})
    goal = np.array([0.25, -12.25, 0.25])
    ag.frontiers = {(0, -12, 0): {"pos": goal, "key": int(P.keys_of(ag.map.voxel(goal))), "node": 0}}
    ag.next_survey = np.inf
    ahead = [np.array([0.25, -y, 0.25]) for y in (4.25, 8.25, 12.25)]
    away = [np.array([0.25, y, 0.25]) for y in (4.25, 8.25, 15.25)]
    plans = iter([(ahead, True), (away, False)])
    def plan(start, g, stop_m=0.9, relax=False):
        path, ag.map.reached_goal = next(plans)
        return [start] + path
    monkeypatch.setattr(ag.map, "plan", plan)
    monkeypatch.setattr(P, "shortcut", lambda path, m: path)
    obs = SimpleNamespace(pos=np.zeros(3), t=100.0, roll=0.0, pitch=0.0, yaw=0.0, lidar=np.zeros((0, 3)),
                          range_down=np.inf, vel=np.zeros(3))
    monkeypatch.setattr(ag, "follow", lambda obs: np.zeros(3))
    monkeypatch.setattr(ag, "guard", lambda obs, v: v)
    ag.search(obs)
    obs.t = 104.0
    ag.search(obs)
    assert ag.path[-1][1] < -12


def test_search_does_not_go_back_to_a_frontier_it_gave_up_on(monkeypatch):
    # dropped after GOAL_PATIENCE_S without a gain, the next survey met it again and it ranked first
    # again: 360 s wedged 5 m short of it (hard 52, 20 m error, notes 13)
    from types import SimpleNamespace
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": [0, 0, 0], "target_error_m": 20.0, "detector_range": 6.0, "detector_half_fov_deg": 60.0,
                               "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3})
    stuck, other, third = np.array([0.25, -5.25, 0.25]), np.array([0.25, 9.25, 0.25]), np.array([12.25, 9.25, 0.25])
    f = lambda p: {"pos": p, "key": int(P.keys_of(ag.map.voxel(p))), "node": 0}
    ag.frontiers = {(0, -5, 0): f(stuck), (0, 9, 0): f(other), (12, 9, 0): f(third)}
    ag.next_survey = np.inf
    def plan(start, g, stop_m=0.9, relax=False):
        ag.map.reached_goal = True
        return [start, g]
    monkeypatch.setattr(ag.map, "plan", plan)
    monkeypatch.setattr(P, "shortcut", lambda path, m: path)
    monkeypatch.setattr(ag, "follow", lambda obs: np.zeros(3))
    monkeypatch.setattr(ag, "guard", lambda obs, v: v)
    obs = SimpleNamespace(pos=np.zeros(3), t=100.0, roll=0.0, pitch=0.0, yaw=0.0, lidar=np.zeros((0, 3)),
                          range_down=np.inf, vel=np.zeros(3))
    ag.search(obs)
    assert np.allclose(ag.search_goal[1], stuck)                 # nearest the given position: first
    obs.t += P.GOAL_PATIENCE_S + 1                               # wedged: no gain
    ag.search(obs)
    assert np.allclose(ag.search_goal[1], other)                 # the same replan does not pick it again
    ag.frontiers[(0, -5, 0)] = f(stuck)                          # the next survey meets it again
    obs.pos, obs.t = other.copy(), obs.t + P.REPLAN_S + 1        # at the other one: a new pick
    ag.search(obs)
    assert np.allclose(ag.search_goal[1], third)


def test_rotor_dust_does_not_put_rock_where_the_drone_is(monkeypatch):
    # low over the floor, the rotors lift dust: returns 0.3 m off, in the drone's own voxel, where no
    # beam passes to clear them. Mapped, they made it rock: "trapped", the drone crept round the spot
    # in the dust for 120-300 s on the way home (hard, 20 m error, notes 13)
    from cavesim.api import Observation
    monkeypatch.setattr(P, "SCAN_MATCH", False)
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.1})
    u = np.random.default_rng(0).normal(size=(400, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    wall = np.c_[np.full(200, 3.0), np.linspace(-1, 1, 200), np.zeros(200)]
    dirs = np.vstack([u, wall / 3.0])
    ranges = np.r_[np.full(400, 0.3), np.full(200, 3.0)]
    obs = Observation(t=1.0, drone_id=0, pos=np.array([0.1, 0.1, 0.1]), vel=np.zeros(3), yaw=0.0, roll=0.0, pitch=0.0,
                      imu=np.zeros((0, 7)), lidar=dirs * ranges[:, None], lidar_dirs=dirs, lidar_ranges=ranges,
                      range_down=0.4, baro_alt=0.0, battery=1.0, detections=[], inbox=[], neighbors=[], base_link=False,
                      report_results=[], landed=False)
    seen = []
    monkeypatch.setattr(ag, "mission_step", lambda obs, pts: (seen.append(pts), P.Command())[1])
    for _ in range(3):
        ag.step(obs)
    assert all(np.linalg.norm(p - obs.pos, axis=1).min() > P.DUST_R for p in seen)   # the wall, not the dust
    assert all(len(p) >= 200 for p in seen)


@pytest.mark.parametrize("w, off", [(0.7, 0.0), (0.7, 0.08), (0.8, 0.0)])
def test_guard_lets_the_drone_fly_along_a_slot_it_fits_through(w, off):
    # a 0.4 m drone in a 0.7 m slot: every wall point just ahead counted as rock being closed in on,
    # and the guard cut the command along the slot to nothing -- 240 s wedged, three times in the
    # search near the survivor (hard, 20 m error, notes 13). Moving along a wall does not close on it.
    from types import SimpleNamespace
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.1})
    y, z = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-1, 1, 0.1))
    slot = np.vstack([np.c_[np.full(y.size, s * w / 2), y.ravel(), z.ravel()] for s in (-1, 1)]) - [off, 0, 0]
    obs = SimpleNamespace(pos=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0, lidar=slot, range_down=np.inf)
    v = ag.guard(obs, np.array([0.0, 1.0, 0.0]), use_map=False)
    assert v[1] > 0.4                                        # on along it ...
    assert v[0] * off <= 0                                   # pushed back to the middle, not into the wall
    assert ag.guard(obs, np.array([0.0, 3.0, 0.0]), use_map=False)[1] < 0.7   # ... slowly: off the line at speed, crashes (2/12)


def test_guard_still_stops_short_of_rock_ahead():
    from types import SimpleNamespace
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.1})
    x, z = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(-1, 1, 0.1))
    for d in (0.45, 0.7, 1.0):                                  # head on: no faster than we can still stop SAFE_M short
        wall = np.c_[x.ravel(), np.full(x.size, d), z.ravel()]
        obs = SimpleNamespace(pos=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0, lidar=wall, range_down=np.inf)
        assert ag.guard(obs, np.array([0.0, 3.0, 0.0]), use_map=False)[1] <= P.stop_speed(d - P.SAFE_M) + 0.01
    corner = np.c_[x.ravel() * 0 + 0.3, x.ravel(), z.ravel()]   # a wall 0.3 m to the side, flying at it at 45 deg
    obs = SimpleNamespace(pos=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0, lidar=corner, range_down=np.inf)
    assert ag.guard(obs, np.array([2.0, 2.0, 0.0]), use_map=False)[0] <= 0.0



def test_sweep_finds_the_way_on_that_leads_away_from_the_goal():
    # the plan toward the goal never expands the other way: the J1 chamber's exit, east of the drone
    # with the survivor west, was mapped but never remembered (hard, notes 13.14)
    m = P.VoxelMap()
    for x in np.arange(-15.25, 6.5, P.RES):                       # a tube, 6.5 m on east, 15.5 m west
        for y in (-0.25, 0.25, 0.75):
            for z in (-0.25, 0.25, 0.75):
                m.free.add(key(m, np.array([x, y, z])))
    found = m.sweep(np.array([0.25, 0.25, 0.25]), 10.0)
    assert any(m.centre(k)[0] > 6.5 for k, _ in found)            # the east end
    assert all(m.centre(k)[0] > -11.0 for k, _ in found)          # nothing past r


def test_a_beam_that_goes_on_clears_the_stale_surface_it_passes():
    # surface laid at x = 0.3 on the way out under another pose error; now the beams reach the wall at x = 1
    m = wall_map(0.3, sweeps=50)
    origin = np.zeros(3)
    assert len(m.surface_near(origin, 0.5)) > 0
    y, z = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-3, 3, 0.05))
    wall = np.c_[np.full(y.size, 1.0), y.ravel(), z.ravel()]
    m.insert(origin, wall)
    assert len(m.surface_near(origin, 0.8)) == 0          # the stale cells in front are gone
    m.insert(origin, wall)
    assert len(m.surface_near(origin, 1.2)) > 0           # the wall the beams hit is rock (twice seen)
    m.insert(origin, wall)
    assert len(m.surface_near(origin, 1.2)) > 0           # ... and stays rock


def test_the_way_home_is_planned_no_further_than_the_course_rule_allows(monkeypatch):
    # past the cave mouth the road runs over the staging area: a plan to a node there fails and its
    # partial end lay back in the cave (easy 6, notes 13.16)
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3})
    road = [np.array([x, 0.0, 1.5]) for x in (14.9, 13.1, 10.6, 8.0, 5.4, 2.6, 0.0)]
    monkeypatch.setattr(ag.map, "allowed", lambda k: ag.map.centre(k)[0] > 9.0)
    assert ag.home_leg_end(road) == 2                      # 10.6 m: the last node inside
    monkeypatch.setattr(ag.map, "allowed", lambda k: True)
    assert ag.home_leg_end(road) == 5                      # all inside: the first node HOME_LEG_M (12 m) along
    monkeypatch.setattr(ag.map, "allowed", lambda k: ag.map.centre(k)[0] > 14.0)
    assert ag.home_leg_end(road) == 0                      # the next node is outside: fly the road


def test_a_climb_that_does_not_happen_maps_the_rock_above_and_stops_the_creep():
    # creeping up into a ceiling pocket the LiDAR cannot see pinned the drone there 35 s (easy 0, notes 13.19)
    from types import SimpleNamespace
    ag = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3,
                               "max_speed_xy": 3.0, "max_speed_z": 1.5})
    above = int(P.keys_of(ag.map.voxel(np.array([0.1, 0.1, 0.1 + P.RES]))))
    ag.last_cmd = np.array([0.0, 0.0, P.CREEP_Z])
    obs = lambda t, vz: SimpleNamespace(t=t, pos=np.array([0.1, 0.1, 0.1]), vel=np.array([0.0, 0.0, vz]), gps=None, range_down=3.0,
                                        roll=0.0, pitch=0.0, yaw=0.0, lidar=np.zeros((0, 3)))
    for t in (0.0, 0.5, 1.2):
        ag.feel_ceiling(obs(t, 0.3))                          # rising as commanded: nothing felt
    assert above not in ag.map.occ
    for t in (2.0, 2.5, 3.1):
        ag.feel_ceiling(obs(t, 0.0))                          # commanded up, not rising
    assert above in ag.map.occ
    assert ag.drop_until > 3.1                                # ... and a full descent out of its pull
    assert ag.cap_speed(obs(3.2, 0.0), np.array([0.0, 0.0, 1.0]))[2] <= 0.0    # no creeping into known rock
    ag2 = P.PlannerAgent(0, 1, {"base": [0, 0, 0], "target": None, "lidar_vfov_deg": (-45.0, 45.0), "lidar_min_range": 0.3,
                                "max_speed_xy": 3.0, "max_speed_z": 1.5})
    assert ag2.cap_speed(obs(0.0, 0.0), np.array([0.0, 0.0, 1.0]))[2] > 0.2     # unknown above: the creep stands
