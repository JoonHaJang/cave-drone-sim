"""Team agent: map-based search, shared finds, early and fast return.

Each drone keeps a 1 m voxel map (odom frame, log-odds from LiDAR) and a "searched" layer
marking cells the artifact detector has looked at from close enough. It flies the cheapest
path (skimage MCP over free cells) to the nearest frontier or unsearched floor that no
teammate has claimed or already covered, and shares claims, coverage and sightings by radio.
Any drone with a base link reports the whole team's finds. A drone heads home as soon as the
team has seen every artifact, nothing is left for it to search, or battery/time run short.

    python -m cavesim.run --agent agents.team1
    python -m agents.eval --agent agents.team1 --maps small_2 medium_1
"""
from collections import deque

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from cavesim.api import Agent, Command

RES = 1.0                                   # m per voxel
ORIGIN = np.array([-40.0, -120.0, -30.0])   # odom-frame corner of the grid; fits the large preset
SHAPE = (240, 240, 60)
HIT, MISS, L_MAX, L_MIN = 3, 1, 12, -6      # log-odds steps and clamps
BLOCK = 4.0                                 # m, coarse cells shared as team coverage
SEARCH_R = 5.0                              # m, how far we trust the 6 m detector
SEARCH_FOV = np.radians(55)                 # detector half FOV is 60 deg
CLAIM_R = 8.0                               # m, radius around a teammate's claimed target
CLAIM_COST = 50.0                           # m-equivalent penalty for going there anyway
REPLAN_S = 1.5
V_MAX = 3.0
BRAKE = 2.0                                 # m/s^2 assumed when limiting speed by free distance ahead
V_HOME = 2.5                                # m/s assumed when budgeting the trip home
ANCHOR_S = 5.0                              # s of early scans (odometry still ~exact) kept as the base reference
ICP_RANGE = 25.0                            # m from home where we start correcting drift against it


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def merge(sightings, typ, pos, n):
    """Fold a sighting into a list of {type, pos, n}; same type within 4 m is the same artifact."""
    for s in sightings:
        if s["type"] == typ and np.linalg.norm(s["pos"] - pos) < 4.0:
            if n > s["n"]:
                s["pos"], s["n"] = np.array(pos, dtype=float), n
            return s
    s = {"type": typ, "pos": np.array(pos, dtype=float), "n": n}
    sightings.append(s)
    return s


class TeamAgent(Agent):
    def __init__(self, drone_id, n_drones, info):
        super().__init__(drone_id, n_drones, info)
        self.L = np.zeros(SHAPE, np.int8)
        self.known = np.zeros(SHAPE, bool)
        self.searched = np.zeros(SHAPE, bool)
        self.lo = np.array(SHAPE) - 1
        self.hi = np.zeros(3, int)
        self.mode = "explore"
        self.path, self.path_i, self.target = [], 0, None
        self.next_plan = 0.1 * (1 + 2 * drone_id)   # stagger first plans so claims reach teammates
        self.last_plan = -1.0
        self.home_cost = 0.0
        self.mine = []           # own sightings {type, pos (avg), n}
        self.team = []           # own + heard, for counting and reporting
        self.confirmed = []      # (type, pos) accepted by the base
        self.claims = []         # every target a teammate announced
        self.blocks, self.team_blocks = set(), set()
        self.mates = {}
        self.crumbs = []
        self.blacklist = []
        self.stuck = 0
        self.tick = 0
        a = 2 * np.pi * drone_id / max(n_drones, 1)
        self.land_spot = np.array(info["base"]) + (np.array([np.cos(a), np.sin(a), 0.0]) if n_drones > 1 else 0.0)
        r = int(np.ceil(SEARCH_R / RES))
        g = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.cone = g[np.linalg.norm(g * RES, axis=1) <= SEARCH_R]
        self.cone_az = np.arctan2(self.cone[:, 1], self.cone[:, 0])
        self.cone_below = np.hypot(self.cone[:, 0], self.cone[:, 1]) <= 1
        self.el = None
        self.why, self.stats = "", {}
        self.anchor_pts, self.anchor = [], None
        self.recent = deque(maxlen=10)       # last second of scans, odom frame
        self.corr = (0.0, np.zeros(3))       # drift estimate: true ~= rot_z(yaw) @ odom + t

    # ---------------------------------------------------------------- map
    def cell(self, p):
        return np.floor((np.asarray(p) - ORIGIN) / RES).astype(int)

    def inside(self, c):
        return np.all((c >= 0) & (c < SHAPE), axis=-1)

    def world(self, c):
        return (np.asarray(c) + 0.5) * RES + ORIGIN

    def integrate(self, obs):
        R = rot_z(obs.yaw)
        dirs = obs.lidar_dirs[::2] @ R.T
        rng = obs.lidar_ranges[::2]
        hit = np.isfinite(rng)
        reach = np.where(hit, rng - 0.5, 14.0)
        ts = np.arange(0.5, 15.0, 0.7)
        free = (obs.pos + dirs[:, None, :] * ts[None, :, None])[ts[None, :] < reach[:, None]]
        hits = obs.pos + dirs[hit] * rng[hit, None]
        for pts, step in ((free, -MISS), (hits, HIT)):
            c = self.cell(pts)
            c = c[self.inside(c)]
            if not len(c):
                continue
            i = tuple(c.T)
            self.L[i] = np.clip(self.L[i].astype(np.int16) + step, L_MIN, L_MAX)
            self.known[i] = True
            self.lo = np.minimum(self.lo, c.min(0))
            self.hi = np.maximum(self.hi, c.max(0))
        here = self.cell(obs.pos)
        if self.inside(here):
            self.L[tuple(here)] = L_MIN          # we are flying here, so it is free
            self.known[tuple(here)] = True

    def mark_searched(self, obs):
        ok = (np.abs(wrap(self.cone_az - obs.yaw)) <= SEARCH_FOV) | self.cone_below
        c = self.cell(obs.pos) + self.cone[ok]
        c = c[self.inside(c)]
        self.searched[tuple(c.T)] = True

    # ---------------------------------------------------------------- planning
    def plan(self, obs, home=False):
        """Cheapest path to a search target (or home). Returns (path, target, home_cost)."""
        lo, hi = np.maximum(self.lo - 1, 0), np.minimum(self.hi + 2, SHAPE)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        known = self.known[sl]
        free = known & (self.L[sl] <= 0)
        clear = ndimage.distance_transform_edt(free)
        cost = np.where(free, 1.0 + 2.0 * np.clip(2.0 - clear, 0, None), np.inf)
        start = self.cell(obs.pos) - lo
        if not np.all((start >= 0) & (start < free.shape)):
            return [], None, np.inf
        cost[tuple(start)] = 1.0
        mcp = MCP_Geometric(cost, fully_connected=True)
        acc, _ = mcp.find_costs([tuple(start)])

        hc = self.cell(self.land_spot) - lo
        home_path, home_cost = [], np.inf
        if np.all((hc >= 0) & (hc < free.shape)) and np.isfinite(acc[tuple(hc)]):
            home_path = self.trace(mcp, hc, lo)
            home_cost = float(sum(np.linalg.norm(a - b) for a, b in zip(home_path, home_path[1:])))
        if home:
            return home_path, self.land_spot, home_cost

        # Frontier: free next to never-seen. Floor: free right above rock, not yet searched.
        unknown = ~known
        nb = np.zeros_like(free)
        for ax in range(3):
            for s in (1, -1):
                nb |= np.roll(unknown, s, axis=ax)
        frontier = free & nb
        below = np.roll(known & (self.L[sl] > 0), 1, axis=2)
        floor = free & below & ~self.searched[sl]
        cand = (frontier | floor) & np.isfinite(acc)
        self.stats = {"frontier": int(frontier.sum()), "floor": int(floor.sum()), "reach": int(cand.sum())}
        lab, n = ndimage.label(cand, structure=np.ones((3, 3, 3)))
        if n == 0:
            return [], None, home_cost
        sizes = ndimage.sum(cand, lab, index=np.arange(1, n + 1))
        cand &= np.isin(lab, np.flatnonzero(sizes >= 3) + 1)
        idx = np.argwhere(cand)
        self.stats["clustered"] = len(idx)
        if not len(idx):
            return [], None, home_cost
        pos = self.world(idx + lo)
        keep = np.ones(len(idx), bool)
        if self.team_blocks:
            blocks = np.floor(pos / BLOCK).astype(int)
            keep &= np.array([tuple(b) not in self.team_blocks for b in blocks])
        if self.blacklist:
            keep &= np.linalg.norm(pos[:, None, :] - np.array(self.blacklist)[None], axis=2).min(1) > 2.5
        self.stats["kept"] = int(keep.sum())
        if not keep.any():
            return [], None, home_cost
        idx, pos = idx[keep], pos[keep]
        score = acc[tuple(idx.T)] * RES
        if self.claims:     # teammate's area: allowed (to help out), but only if nothing else is near
            d = np.linalg.norm(pos[:, None, :] - np.array(self.claims)[None], axis=2).min(1)
            score = score + CLAIM_COST * (d < CLAIM_R)
        if self.target is not None:      # a little hysteresis so we don't flip between equal targets
            score = score - 4.0 * (np.linalg.norm(pos - self.target, axis=1) < 3.0)
        k = int(np.argmin(score))
        return self.trace(mcp, idx[k], lo), pos[k], home_cost

    def trace(self, mcp, end, lo):
        return [self.world(np.array(c) + lo) for c in mcp.traceback(tuple(end))]

    # ---------------------------------------------------------------- motion
    def free_ahead(self, obs, d):
        """Distance to the nearest LiDAR point inside a 0.7 m corridor along odom direction d."""
        db = rot_z(obs.yaw).T @ d
        pts = obs.lidar
        proj = pts @ db
        lat2 = (pts * pts).sum(1) - proj ** 2
        m = (proj > 0.3) & (lat2 < 0.49)
        return float(proj[m].min()) if m.any() else self.info["lidar_max_range"]

    def avoid(self, obs):
        pts = obs.lidar
        d = np.linalg.norm(pts, axis=1)
        close = d < 1.4
        if not close.any():
            return np.zeros(3)
        push = -(pts[close] / d[close, None]) * ((1.4 - d[close]) / 1.4)[:, None]
        return rot_z(obs.yaw) @ (2.5 * push.sum(0) / max(1, close.sum() ** 0.5))

    def floor_dist(self, obs):
        r = np.nan_to_num(obs.lidar_ranges, posinf=self.info["lidar_max_range"])
        down = self.el <= np.radians(-30)
        return float((r[down] * -obs.lidar_dirs[down, 2]).min())

    def follow(self, obs, v_cap=V_MAX):
        P = self.path
        near = range(self.path_i, min(self.path_i + 8, len(P)))
        self.path_i = min(near, key=lambda i: np.linalg.norm(P[i] - obs.pos))
        look = P[min(self.path_i + 3, len(P) - 1)]
        to = look - obs.pos
        dist = np.linalg.norm(to)
        if dist < 1e-6:
            return np.zeros(3)
        d = to / dist
        speed = min(v_cap, max(0.4, np.sqrt(2 * BRAKE * max(self.free_ahead(obs, d) - 1.0, 0.0))))
        if self.mode == "return":       # arrive at the landing spot instead of overshooting it
            speed = min(speed, 0.8 * np.linalg.norm(P[-1] - obs.pos) + 0.3)
        return speed * d

    # ---------------------------------------------------------------- team
    def broadcast(self, obs):
        msg = {"pos": obs.pos, "claim": self.target if self.mode == "explore" else None,
               "seen": [(s["type"], s["pos"], s["n"]) for s in self.mine],
               "ok": self.confirmed}
        if self.tick % 20 == 0:
            msg["blocks"] = list(self.blocks)
        return [("all", msg)]

    def receive(self, obs):
        for sender, msg in obs.inbox:
            self.mates[sender] = msg["pos"]
            c = msg.get("claim")
            if c is not None and not any(np.linalg.norm(c - x) < 2.0 for x in self.claims):
                self.claims.append(np.array(c))
            for typ, pos, n in msg["seen"]:
                merge(self.team, typ, pos, n)
            for typ, pos in msg["ok"]:
                if not any(t == typ and np.linalg.norm(p - pos) < 4.0 for t, p in self.confirmed):
                    self.confirmed.append((typ, pos))
            self.team_blocks.update(msg.get("blocks", ()))

    def remember(self, obs):
        for det in obs.detections:
            pos = obs.pos + rot_z(obs.yaw) @ det.rel_pos
            s = next((s for s in self.mine if s["type"] == det.type and np.linalg.norm(s["pos"] - pos) < 4.0), None)
            if s is None:
                s = {"type": det.type, "pos": pos, "n": 0}
                self.mine.append(s)
            s["pos"] = (s["pos"] * s["n"] + pos) / (s["n"] + 1)
            s["n"] += 1
            merge(self.team, s["type"], s["pos"], s["n"])

    def reports(self, obs):
        for (typ, pos), ok, _ in obs.report_results:
            pos = np.asarray(pos)
            if ok:
                self.confirmed.append((typ, pos))
            for s in self.team:
                if s["type"] == typ and np.linalg.norm(s["pos"] - pos) < 0.5:
                    s["done"] = True        # accepted or not, don't spend budget on it again
        if not obs.base_link:
            return []
        out = []
        for s in self.team:
            if s.get("done") or s.get("sent") or s["n"] < 2:
                continue
            if any(t == s["type"] and np.linalg.norm(p - s["pos"]) < 4.0 for t, p in self.confirmed):
                continue
            s["sent"] = True
            out.append((s["type"], s["pos"].tolist()))
        return out

    def all_seen(self):
        return sum(s["n"] >= 2 for s in self.team) >= self.info["n_artifacts"]

    # ---------------------------------------------------------------- step
    def step(self, obs):
        if self.el is None:
            self.el = np.arcsin(obs.lidar_dirs[:, 2])
        self.tick += 1
        if obs.landed:
            return Command()
        self.receive(obs)
        self.integrate(obs)
        self.mark_searched(obs)
        self.remember(obs)
        self.blocks.add(tuple(np.floor(obs.pos / BLOCK).astype(int)))
        self.add_crumb(obs.pos)
        self.track_drift(obs)

        if self.mode == "explore":
            home = self.home_cost if np.isfinite(self.home_cost) else self.crumb_len(obs.pos)
            need = home / V_HOME + 15
            short = (obs.battery < need * 1.3 / self.info["hover_endurance_s"] + 0.06
                     or obs.t + need > self.info["mission_time_s"] - 10)
            if self.all_seen() or short:
                self.go_home(obs, "all seen" if self.all_seen() else "battery/time")
            elif obs.t >= self.next_plan or (self.last_plan >= 0 and obs.t >= self.last_plan + 0.3
                                             and (not self.path or self.path_i >= len(self.path) - 3)):
                self.next_plan, self.last_plan = obs.t + REPLAN_S, obs.t
                path, target, self.home_cost = self.plan(obs)
                if target is None:
                    self.go_home(obs, "nothing left")
                else:
                    self.path, self.path_i, self.target = path, 0, target
        elif self.mode == "return" and obs.t >= self.next_plan:
            self.next_plan = obs.t + 3.0
            path, _, cost = self.plan(obs, home=True)
            if path:
                self.path, self.path_i = path, 0

        cmd = Command(reports=self.reports(obs), send=self.broadcast(obs))
        if self.mode == "descend":
            to = self.home_spot() - obs.pos
            fd = self.floor_dist(obs)
            if fd < 1.0:
                cmd.land = True
                return cmd
            v = np.array([0.8 * to[0], 0.8 * to[1], -min(1.2, fd - 0.6)])
        else:
            v = self.follow(obs) if self.path else np.zeros(3)
            if self.mode == "return" and np.linalg.norm((self.land_spot - obs.pos)[:2]) < 3.0:
                self.path, self.path_i = [obs.pos, self.home_spot()], 0     # last metres: drift-corrected spot
            if self.mode == "return" and np.linalg.norm((self.home_spot() - obs.pos)[:2]) < 0.6:
                self.mode = "descend"
        push = self.avoid(obs)
        if self.mode == "descend":
            push[2] = 0.0               # the floor is where we want to go
        v = v + push

        # Stuck against something: drop the target and let the planner pick again.
        if self.mode == "explore" and np.linalg.norm(v) > 0.6 and np.linalg.norm(obs.vel) < 0.25:
            self.stuck += 1
            if self.stuck > 25 and self.target is not None:
                self.blacklist.append(self.target)
                self.path, self.stuck = [], 0
        else:
            self.stuck = 0

        if np.hypot(v[0], v[1]) > 0.4:
            cmd.yaw_rate = 2.0 * wrap(np.arctan2(v[1], v[0]) - obs.yaw)
        cmd.vel = v
        return cmd

    # ---------------------------------------------------------------- drift
    def track_drift(self, obs):
        pts = (obs.pos + obs.lidar @ rot_z(obs.yaw).T)[::2]
        self.recent.append(pts)
        if obs.t < ANCHOR_S:
            self.anchor_pts.append(pts)
        elif self.anchor is None and self.anchor_pts:
            a = np.vstack(self.anchor_pts)
            _, keep = np.unique(np.floor(a / 0.25).astype(int), axis=0, return_index=True)
            self.anchor = cKDTree(a[keep])
        if (self.anchor is not None and self.mode != "explore" and self.tick % 10 == 0
                and np.linalg.norm(obs.pos - self.land_spot) < ICP_RANGE):
            self.icp(np.vstack(self.recent))

    def icp(self, src):
        """Align recent scans to the early base scans; yaw about z plus a 3D shift."""
        yaw, t = self.corr
        for gate in (3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.5):
            moved = src @ rot_z(yaw).T + t
            d, j = self.anchor.query(moved, distance_upper_bound=gate)
            ok = np.isfinite(d)
            if ok.sum() < 200:
                return
            a, b = src[ok], self.anchor.data[j[ok]]
            ca, cb = a.mean(0), b.mean(0)
            H = (a[:, :2] - ca[:2]).T @ (b[:, :2] - cb[:2])
            yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
            t = cb - rot_z(yaw) @ ca
        if abs(yaw) < 0.3 and np.linalg.norm(t) < 8.0 and np.median(d[ok]) < 0.3:
            self.corr = (yaw, t)

    def home_spot(self):
        """Landing spot in odom coordinates, after undoing the estimated drift."""
        yaw, t = self.corr
        return rot_z(-yaw) @ (self.land_spot - t)

    def add_crumb(self, pos):
        """Breadcrumbs home with loops cut out: the fallback path and distance when the map has none."""
        if self.crumbs and np.linalg.norm(pos - self.crumbs[-1]) < 2.0:
            return
        for k, c in enumerate(self.crumbs[:-2]):
            if np.linalg.norm(pos - c) < 2.0:
                del self.crumbs[k + 1:]
                break
        self.crumbs.append(pos.copy())

    def crumb_len(self, pos):
        pts = [pos, *self.crumbs[::-1], self.land_spot]
        return float(sum(np.linalg.norm(a - b) for a, b in zip(pts, pts[1:])))

    def go_home(self, obs, why):
        self.mode, self.target, self.why = "return", None, why
        path, _, _ = self.plan(obs, home=True)
        # Map path if there is one, else retrace our own breadcrumbs.
        self.path = path or [*self.crumbs[::-1], self.land_spot]
        self.path_i = 0
        self.next_plan = obs.t + 3.0

    def debug(self):
        return {"mode": f"{self.mode} ({self.why})" if self.why else self.mode, "path": [p.tolist() for p in self.path[self.path_i:]][::-1],
                "goal": None if self.target is None else self.target.tolist(),
                "points": [s["pos"].tolist() for s in self.team]}


Agent = TeamAgent
