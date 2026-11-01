"""Re-localisation against the map: the agent's own correction on top of the odometry.

The odometry is a LIO already (cavesim.config.OdomParams: ~0.2-1 % drift), but unlike a real one
it never looks back at its map: flying again through a place mapped minutes ago, the pose no
longer matches what was mapped there, the map grows double walls, and the way home closes.

So each sweep is registered against map cells first seen more than OLD_S ago -- the places the
drift has had time to separate us from. Through new ground there is nothing that old within reach
and the odometry stands (registering against what was mapped a second ago would only trade its
drift for the registration's own: centimetres of bias per sweep, fed back through the map, walk
off by metres). Old cells are frozen, so a pass at a wrong pose cannot drag them along, and only
returns within MAP_R build them: a wall glimpsed from 20 m is too far off to anchor a pose, and
"old ground" should mean a place we actually flew through.

What is corrected is a translation, pose = odometry pose + tc. Roll and pitch are tied to gravity
(the flight controller's 0.3 deg is better than a 16-beam registration); the heading drifts about
a degree a mission, and what that does to the position shows, wherever we fly again, as an offset
that varies slowly along the way -- which is what tc follows. A rotation correction would also
turn about the origin, hundreds of metres away, so a wrong tenth of a degree would throw the pose.

Registration: N_PTS returns, ITERS Gauss-Newton steps of point-to-plane ICP (plane = the K nearest
map points, one point per VOX voxel), Huber-weighted. Unknowns are the translation only, so the
normal matrix is sum n n^T. Degeneracy (Zhang, Kaess & Singh, ICRA 2016): down a smooth tunnel, a
slide along its axis does not change the scan; the step is kept to the eigen-directions of the
mean n n^T above DEGEN_LAMBDA, and along the others the odometry stands. The pose moves GAIN of
the way to the registered one per sweep (a complementary filter: the odometry is exact over a
sweep, the registration is not), and a registration that would move it more than MAX_D is not
the same place: dropped.

Drift gate: the odometry's error between now and when a reference cell was mapped is a random
walk (config.OdomParams: 0.01 m/sqrt(m) flown, 0.2 % scale, 0.05 m/sqrt(s) while degenerate --
here: while our own registration finds a degenerate direction -- and the heading's own walk, 0.0005
rad/sqrt(s) and 0.002 rad/sqrt(rad turned), which turns everything flown since: 1200 m out and
back, a degree is 10 m, the larger part by far; left out, the gate dropped right fits on the way
home and the pose was lost, research_notes 13). So the correction may differ
from the one the cell was mapped with by no more than GATE_SIGMA of that, plus GATE_SLACK; a registration most of
its matched cells disagree with is a look-alike wall, not the place we mapped -- dropped. Without
it, at a multi-level junction a wrong fit taken GAIN at a time slid the pose 14 m in four minutes
while the odometry stayed within 0.8 m (research_notes 10).
"""
import numpy as np
from scipy.spatial import cKDTree

VOX = 0.5             # m: map point spacing (FAST-LIO keeps its map at about this)
SUBMAP_M = 30.0       # m around the drone indexed for matching
REBUILD_S = 1.0       # s between k-d tree rebuilds
N_PTS = 800           # returns matched per sweep
ITERS = 4             # Gauss-Newton steps per sweep
K = 5                 # map points per plane
MAX_D = 1.0           # m: farther pairs are not the same surface
HUBER = 0.1           # m
DEGEN_LAMBDA = 0.05   # as the simulator's own degeneracy model (config.OdomParams)
MIN_PAIRS = 100
GAIN = 0.2            # share of the registered offset taken per sweep
OLD_S = 20.0          # s: cells first seen longer ago than this are the reference
GATE = True          # drift gate (above)
DRIFT_PER_SQRT_M = 0.01
DRIFT_PER_SQRT_S = 0.05
DRIFT_SCALE = 0.002
YAW_RW = 0.0005       # rad/sqrt(s): heading random walk (config.OdomParams.yaw_rw_std)
YAW_PER_SQRT_RAD = 0.002   # rad/sqrt(rad turned) (config.OdomParams.yaw_noise_per_rad)
GATE_SIGMA = 3.0
GATE_SLACK = 0.5      # m
MAP_R = 8.0           # m: only nearer returns build the reference (farther ones land ~r/60 off: planner.HIT_R0)


class ScanMatcher:
    def __init__(self):
        self.tc = np.zeros(3)                           # correction: pose = odometry + tc
        self.index = {}                                 # voxel key -> row
        self.sums = np.zeros((1024, 3))
        self.counts = np.zeros(1024)
        self.born = np.zeros(1024)                      # time each cell was first seen
        self.born_d = np.zeros(1024)                    # ... the distance flown then
        self.born_tc = np.zeros((1024, 3))              # ... and the correction it was mapped with
        self.born_g = np.zeros(1024)                    # ... and the time spent degenerate till then
        self.born_y = np.zeros(1024)                    # ... and the heading turned till then
        self.turned, self.last_yaw = 0.0, None
        self.degen_s, self.last_t = 0.0, None
        self.flown, self.last_odom = 0.0, None
        self.tree_rows = None
        self.rejected = 0                               # registrations the drift gate dropped
        self.tree, self.tree_pts, self.next_build = None, None, -np.inf
        self.degenerate = 0                             # directions left to the odometry, last sweep
        self.pairs = 0

    def correct(self, R, p_odom, pts_body, t):
        """Corrected position for this sweep (R: body -> odom); adds the sweep to the map."""
        if self.last_odom is not None:
            self.flown += float(np.linalg.norm(p_odom - self.last_odom))
        self.last_odom = np.array(p_odom, dtype=float)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if self.last_yaw is not None:
            self.turned += abs((yaw - self.last_yaw + np.pi) % (2 * np.pi) - np.pi)
        self.last_yaw = yaw
        if self.last_t is not None and self.degenerate:
            self.degen_s += t - self.last_t
        self.last_t = t
        p = p_odom + self.tc
        self.pairs = self.degenerate = 0                # no registration this sweep: nothing is known degenerate
        if self.tree is not None and len(pts_body) >= MIN_PAIRS:
            s = pts_body[np.linspace(0, len(pts_body) - 1, min(N_PTS, len(pts_body))).astype(int)] @ R.T
            p1 = self.register(s, p)
            if p1 is not None and np.linalg.norm(p1 - p) < MAX_D:
                q = p + GAIN * (p1 - p)
                if self.drift_ok(q - p_odom, t):
                    p = q
                    self.tc = p - p_odom
                else:
                    self.rejected += 1
        self.add(pts_body[np.einsum("ij,ij->i", pts_body, pts_body) < MAP_R ** 2] @ R.T + p, t)
        if t >= self.next_build:
            self.next_build = t + REBUILD_S
            n = len(self.index)
            mean = self.sums[:n] / self.counts[:n, None]
            rows = np.flatnonzero((np.linalg.norm(mean - p, axis=1) < SUBMAP_M) & (self.born[:n] < t - OLD_S))
            near = mean[rows]
            self.tree, self.tree_pts, self.tree_rows = (cKDTree(near), near, rows) if len(near) >= K * 10 else (None, None, None)
        return p

    def register(self, s, p):
        """Translation that puts the rotated returns s (about p) on the old map, or None."""
        for _ in range(ITERS):
            w = s + p
            d, nb = self.tree.query(w, k=K, distance_upper_bound=MAX_D)
            ok = np.isfinite(d).all(1)
            if ok.sum() < MIN_PAIRS:
                return None
            Q = self.tree_pts[nb[ok]]                     # (M, K, 3)
            mu = Q.mean(1)
            C = np.einsum("mki,mkj->mij", Q - mu[:, None], Q - mu[:, None]) / K
            ev, V = np.linalg.eigh(C)
            n = V[:, :, 0]                               # plane normal: least spread
            r = np.einsum("mi,mi->m", n, w[ok] - mu)
            use = (ev[:, 0] < 0.1 * ev[:, 1]) & (np.abs(r) < MAX_D)
            if use.sum() < MIN_PAIRS:
                return None
            n, r = n[use], r[use]
            self.matched = self.tree_rows[nb[ok][use, 0]]  # the reference cells this fit rests on
            wt = np.minimum(1.0, HUBER / np.maximum(np.abs(r), 1e-9))
            H = (n * wt[:, None]).T @ n / wt.sum()       # mean n n^T, as in the degeneracy model
            g = (n * wt[:, None]).T @ r / wt.sum()
            ev, V = np.linalg.eigh(H)
            keep = ev > DEGEN_LAMBDA
            self.degenerate, self.pairs = int((~keep).sum()), int(use.sum())
            step = -V[:, keep] @ ((V[:, keep].T @ g) / ev[keep])
            p = p + step
            if np.linalg.norm(step) < 1e-3:
                break
        return p

    def drift_ok(self, tc, t):
        """Does the correction tc agree with most of the matched cells, given the drift since?"""
        if not GATE:
            return True
        m = self.matched
        dd = np.maximum(self.flown - self.born_d[m], 0.0)
        dg = np.maximum(self.degen_s - self.born_g[m], 0.0)
        # heading error since the cell was mapped turns every metre flown since: about dd / 2 of lever
        # for an out-and-back (a turn at the far end), dd / sqrt(12) for a steady random walk
        dpsi2 = YAW_RW ** 2 * np.maximum(t - self.born[m], 0.0) + YAW_PER_SQRT_RAD ** 2 * np.maximum(self.turned - self.born_y[m], 0.0)
        allow = GATE_SLACK + GATE_SIGMA * np.sqrt(DRIFT_PER_SQRT_M ** 2 * dd + (DRIFT_SCALE * dd) ** 2 + DRIFT_PER_SQRT_S ** 2 * dg
                                                  + dpsi2 * (dd / 2) ** 2)
        return float(np.median(np.linalg.norm(tc - self.born_tc[m], axis=1) - allow)) <= 0.0

    def add(self, pts, t):
        keys = np.floor(pts / VOX).astype(np.int64)
        keys = ((keys[:, 0] + (1 << 20)) << 42) | ((keys[:, 1] + (1 << 20)) << 21) | (keys[:, 2] + (1 << 20))
        uk, inv = np.unique(keys, return_inverse=True)
        sums = np.zeros((len(uk), 3))
        np.add.at(sums, inv, pts)
        cnt = np.bincount(inv, minlength=len(uk)).astype(float)
        rows = np.empty(len(uk), dtype=np.int64)
        index = self.index
        for i, k in enumerate(uk.tolist()):
            j = index.get(k)
            if j is None:
                j = index[k] = len(index)
                if j >= len(self.counts):
                    grow = 2 * len(self.counts)
                    self.sums = np.vstack([self.sums, np.zeros((grow - len(self.sums), 3))])
                    self.counts = np.concatenate([self.counts, np.zeros(grow - len(self.counts))])
                    self.born = np.concatenate([self.born, np.zeros(grow - len(self.born))])
                    self.born_d = np.concatenate([self.born_d, np.zeros(grow - len(self.born_d))])
                    self.born_tc = np.vstack([self.born_tc, np.zeros((grow - len(self.born_tc), 3))])
                    self.born_g = np.concatenate([self.born_g, np.zeros(grow - len(self.born_g))])
                    self.born_y = np.concatenate([self.born_y, np.zeros(grow - len(self.born_y))])
                self.born[j], self.born_d[j], self.born_tc[j], self.born_g[j], self.born_y[j] = t, self.flown, self.tc, self.degen_s, self.turned
            rows[i] = j
        young = self.born[rows] >= t - OLD_S             # old cells are the reference: frozen
        rows, sums, cnt = rows[young], sums[young], cnt[young]
        # Running mean, capped so a cell settles but a few bad sweeps cannot pin it.
        full = rows[self.counts[rows] >= 20]
        self.sums[full] *= (19 / self.counts[full])[:, None]   # a sweep adds many returns: the count is often past 20
        self.counts[full] = 19
        self.sums[rows] += sums
        self.counts[rows] += cnt
