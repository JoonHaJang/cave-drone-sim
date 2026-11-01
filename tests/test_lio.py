import numpy as np

from agents.lio import OLD_S, ScanMatcher
from cavesim.api import body_rotation


def room(rng, n=6000, corridor=False):
    """Points on the walls of a 20 x 6 x 4 m room with a pillar (or an endless smooth corridor)."""
    u = rng.uniform(-1, 1, (n, 2))
    walls = [np.c_[u[:, 0] * 30, np.full(n, 3.0), u[:, 1] * 2], np.c_[u[:, 0] * 30, np.full(n, -3.0), u[:, 1] * 2],
             np.c_[u[:, 0] * 30, u[:, 1] * 3, np.full(n, 2.0)], np.c_[u[:, 0] * 30, u[:, 1] * 3, np.full(n, -2.0)]]
    if not corridor:
        walls += [np.c_[np.full(n, 10.0), u[:, 0] * 3, u[:, 1] * 2], np.c_[np.full(n, -10.0), u[:, 0] * 3, u[:, 1] * 2],
                  np.c_[np.full(n, 4.0) + 0.5 * (u[:, 0] > 0), u[:, 0] * 0.5 + 1.5, u[:, 1] * 2]]
        walls = [w[np.abs(w[:, 0]) <= 10] for w in walls]
    return np.vstack(walls)


def revisit(corridor):
    """Map a place, come back after OLD_S with the odometry 0.5 m off: what pose do we end at?"""
    world = room(np.random.default_rng(0), corridor=corridor)
    R, true_p = body_rotation(0.01, -0.01, 0.3), np.array([0.5, -0.2, 0.1])
    body = (world - true_p) @ R
    m = ScanMatcher()
    m.correct(R, true_p, body, 0.0)                                # first pass: maps the place
    odom_p = true_p + np.array([0.3, 0.35, -0.2])
    m.next_build = 0.0
    for t in OLD_S + 5 + np.arange(40) * 0.1:                     # the filter closes in over ~1 / GAIN sweeps
        p = m.correct(R, odom_p, body, t)
    return p, true_p, odom_p, m


def test_revisit_pulls_back_onto_the_old_map():
    p, true_p, _, m = revisit(corridor=False)
    assert np.linalg.norm(p - true_p) < 0.03 and m.degenerate == 0


def test_corridor_axis_is_left_to_the_odometry():
    p, true_p, odom_p, m = revisit(corridor=True)
    assert m.degenerate >= 1
    assert abs(p[1] - true_p[1]) < 0.03 and abs(p[2] - true_p[2]) < 0.03   # across the tunnel: corrected
    assert abs(p[0] - odom_p[0]) < 0.05                                    # along it: odometry kept


def test_fresh_ground_is_left_to_the_odometry():
    world = room(np.random.default_rng(1))
    m = ScanMatcher()
    R = np.eye(3)
    for t in np.arange(30) * 0.1:                                 # nothing OLD_S old yet
        p = m.correct(R, np.array([0.2, 0.0, 0.0]), world, t)
    assert np.allclose(p, [0.2, 0.0, 0.0])


def test_drift_gate_drops_a_correction_the_odometry_cannot_have_drifted():
    m = ScanMatcher()                                            # cells mapped 30 s and 20 m ago, at tc = 0
    m.matched, m.flown = np.arange(10), 20.0
    assert m.drift_ok(np.array([0.6, 0.0, 0.0]), 30.0)           # ~0.7 m allowed
    assert not m.drift_ok(np.array([1.5, 0.0, 0.0]), 30.0)       # a look-alike wall


def test_cell_means_stay_in_their_voxel_over_many_dense_sweeps():
    # 300 m out, many returns per cell per sweep: the running mean's cap must not inflate the mean
    # (it scaled the sum as if the count were exactly 20: cells drifted metres off, research_notes 13)
    from agents.lio import VOX
    rng = np.random.default_rng(0)
    m, p = ScanMatcher(), np.array([300.0, -150.0, -40.0])
    for t in np.arange(40) * 0.5:
        u = rng.uniform(-1, 1, (4000, 2))
        m.add(np.c_[u[:, 0] * 3, np.full(len(u), 1.5), u[:, 1] * 2] + p, t)
    n = len(m.index)
    mean = m.sums[:n] / m.counts[:n, None]
    assert np.abs(mean[:, 1] - (p[1] + 1.5)).max() < 1e-6       # the wall stays where it is
    assert np.abs(mean - p).max() < 3.0 + VOX


def test_time_counts_as_degenerate_only_while_a_registration_finds_it_so():
    # the flag stayed at the last registration's value through new ground: 300 s of "degenerate" time
    # widened the drift gate from 0.5 to 3.1 m (static audit, research_notes 13)
    m = ScanMatcher()
    m.degenerate = 1                                             # the last fit, in a tunnel
    rng = np.random.default_rng(0)
    for t in np.arange(0, 300, 1.0):                             # new ground: nothing old to register to
        m.correct(np.eye(3), np.array([t, 0.0, 0.0]), rng.uniform(-5, 5, (500, 3)), t)
    assert m.degen_s < 2.0


def test_drift_gate_allows_for_the_heading_turning_what_was_flown_since():
    # 1200 m and 650 s out and back with ~300 rad turned: the odometry was 10 m off its first pass
    # (seeds 14, 51 hard), the gate without a heading term allowed 8 m and dropped the right fits
    m = ScanMatcher()
    m.matched, m.flown, m.turned = np.arange(10), 1200.0, 300.0
    assert m.drift_ok(np.array([10.0, 0.0, 0.0]), 650.0)
