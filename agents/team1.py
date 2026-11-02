"""Team agent: map-based search, shared finds, early and fast return.

Each drone keeps a 1 m voxel map (odom frame, log-odds from LiDAR) and a "searched" layer
marking cells the artifact detector has looked at from close enough. It flies the cheapest
path (skimage MCP over free cells) to the nearest frontier or unsearched floor that no
teammate has claimed or already covered, and shares claims, coverage and sightings by radio.
Locally it steers around rock the map doesn't show (drift, gaps narrower than a voxel) and gives
up on goals it stops making progress toward. A drone heads home as soon as the team has seen
every artifact, nothing is left for it to search, or battery/time run short; on the way it
realigns its frame to scans it took on the way out, and near home to the scans from takeoff.
Drones report the whole team's finds from home, one after another, and stay up until every
teammate has heard what was sent, so teammates don't report the same find twice.

    python -m cavesim.run --agent agents.team1
    python -m agents.eval --agent agents.team1 --maps small_2 medium_1
"""
from collections import deque
from dataclasses import replace

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from cavesim.api import Agent, Command, body_rotation, body_to_odom

RES = 1.0                                   # m per voxel
ORIGIN = np.array([-40.0, -120.0, -30.0])   # odom-frame corner of the grid; fits the large preset
SHAPE = (240, 240, 60)
HIT, MISS, L_MAX, L_MIN = 3, 1, 12, -6      # log-odds steps and clamps
BLOCK = 4.0                                 # m, coarse cells shared as team coverage
SEARCH_R = 5.0                              # m, how far we trust the 6 m detector
SEARCH_FOV = np.radians(55)                 # detector half FOV is 60 deg
CLAIM_R = 8.0                               # m, radius around a teammate's claimed target
CLAIM_COST = 50.0                           # m-equivalent penalty for going there anyway
TEAM_COST = 30.0                            # m-equivalent penalty for a block a teammate already flew through
REPLAN_S = 1.5
V_MAX = 3.0
BRAKE = 2.0                                 # m/s^2 assumed when limiting speed by free distance ahead
V_HOME = 2.5                                # m/s, most we assume when budgeting the trip home
ANCHOR_S = 5.0                              # s of early scans (odometry still ~exact) kept as the base reference
ICP_RANGE = 25.0                            # m from home where we start correcting drift against it
KEY_GAP = 8.0                               # m between keyframe scans kept on the way out, to realign against on the way back
STALL_S = 20.0                              # s without getting closer to the landing spot before we land where we are
PROG_S = 8.0                                # s without the path getting 0.5 m shorter before we give up on that goal
MAP_R = 15.0                                # m, LiDAR range used for mapping (far returns are sparse)
MAP_BEAMS = (16, 128)                       # channels x azimuths used for mapping, whatever the LiDAR preset
LINK_GRACE = 1.5                            # s of base link before reporting, so a teammate's "already sent" reaches us
REPORT_WAIT = 8.0                           # s of base link before reporting away from home (teammates there may have sent it)
MATE_R = 6.0                                # m; a teammate this close (radio position): slow, keep right, split heights
LAND_R = 1.5                                # m from base to each drone's landing spot
SAME_R = 7.0                                # m; same-type sightings closer than this are one artifact (generator keeps 8 m)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def merge(sightings, typ, pos, n):
    """Fold a sighting into a list of {type, pos, n}; same type within SAME_R is the same artifact."""
    for s in sightings:
        if s["type"] == typ and np.linalg.norm(s["pos"] - pos) < SAME_R:
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
        self.flown = np.zeros(SHAPE, bool)   # cells we flew through: passable whatever the 1 m map says
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
        self.mates = {}          # teammate -> (pos, time heard, mode)
        self.mate_msg = {}       # latest message from each teammate
        self.my_sent = []        # (type, pos) we reported ourselves
        self.link_t, self.hold_t = None, 0.0
        self.crumbs = []
        self.blacklist = []
        self.prog = (np.inf, 0.0)           # shortest path left to the current goal, and when
        self.retrace = False                # returning along our own trail, the map route didn't work
        self.tick = 0
        self.odo, self.last_pos = 0.0, None   # m flown so far: our average speed budgets the trip home
        a = 2 * np.pi * drone_id / max(n_drones, 1)
        self.land_spot = np.array(info["base"]) + (LAND_R * np.array([np.cos(a), np.sin(a), 0.0]) if n_drones > 1 else 0.0)
        r = int(np.ceil(SEARCH_R / RES))
        g = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.cone = g[np.linalg.norm(g * RES, axis=1) <= SEARCH_R]
        self.cone_az = np.arctan2(self.cone[:, 1], self.cone[:, 0])
        self.cone_below = np.hypot(self.cone[:, 0], self.cone[:, 1]) <= 1
        C, A = info["lidar_channels"], info["lidar_azimuth_steps"]
        ch, az = max(1, C // MAP_BEAMS[0]), max(1, A // MAP_BEAMS[1])
        self.beams = (np.arange(0, C, ch)[:, None] * A + np.arange(0, A, az)[None]).ravel()
        self.why, self.stats = "", {}
        self.anchor_pts, self.anchor = [], None
        self.recent = deque(maxlen=10)       # last second of scans, odom frame
        self.corr = (0.0, np.zeros(3))       # drift estimate: world ~= rot_z(yaw) @ odom + t
        self.keys = []                       # (position, KD-tree of scans there), in the frame we had then
        self.best = (np.inf, 0.0)            # closest we got to the landing spot while returning, and when

    # ---------------------------------------------------------------- map
    def cell(self, p):
        return np.floor((np.asarray(p) - ORIGIN) / RES).astype(int)

    def inside(self, c):
        return np.all((c >= 0) & (c < SHAPE), axis=-1)

    def world(self, c):
        return (np.asarray(c) + 0.5) * RES + ORIGIN

    def integrate(self, obs):
        R = body_rotation(obs.roll, obs.pitch, obs.yaw)
        dirs = obs.lidar_dirs[self.beams] @ R.T
        rng = obs.lidar_ranges[self.beams]
        hit = rng <= MAP_R
        # No return is grazing rock, a dropout or far away: carve nothing rather than through a wall.
        reach = np.where(hit, rng - 0.5, np.where(np.isfinite(rng), MAP_R - 0.5, 0.0))
        ts = np.arange(0.5, MAP_R, 0.7)
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
        # Gaps between speleothems can be narrower than a voxel; where we have flown, we can fly again.
        free = (known & (self.L[sl] <= 0)) | self.flown[sl]
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
        if self.blacklist:
            keep &= np.linalg.norm(pos[:, None, :] - np.array(self.blacklist)[None], axis=2).min(1) > 2.5
        self.stats["kept"] = int(keep.sum())
        if not keep.any():
            return [], None, home_cost
        idx, pos = idx[keep], pos[keep]
        score = acc[tuple(idx.T)] * RES
        if self.team_blocks:    # a teammate has been there: leftovers, worth it only when nothing fresh is near
            blocks = np.floor(pos / BLOCK).astype(int)
            score = score + TEAM_COST * np.array([tuple(b) in self.team_blocks for b in blocks])
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
    def free_ahead(self, obs, dirs):
        """Distance to the nearest LiDAR point inside a 0.7 m corridor along each direction (rows of dirs)."""
        db = dirs @ body_rotation(obs.roll, obs.pitch, obs.yaw)      # each row R.T @ d: into the body frame
        pts = obs.lidar
        proj = pts @ db.T
        lat2 = (pts * pts).sum(1)[:, None] - proj ** 2
        near = np.where((proj > 0.3) & (lat2 < 0.49), proj, np.inf).min(0)
        return np.minimum(near, self.info["lidar_max_range"])

    def steer(self, obs, d):
        """Path heading blocked (drift moves old map cells onto rock, gaps narrower than a voxel):
        take the heading closest to it that is clear for 2 m."""
        h, e = np.arctan2(d[1], d[0]), np.arcsin(np.clip(d[2], -1, 1))
        a, b = np.meshgrid(np.radians([0, 25, -25, 50, -50, 75, -75]), np.radians([0, 25, -25, 50, -50]))
        el = np.clip(e + b.ravel(), -1.2, 1.2)
        az = h + a.ravel()
        dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
        free = self.free_ahead(obs, dirs)
        ok = free >= 2.0
        k = int(np.argmax(np.where(ok, dirs @ d, -2.0))) if ok.any() else int(np.argmax(free))
        return dirs[k], free[k]

    def avoid(self, obs):
        pts = obs.lidar
        d = np.linalg.norm(pts, axis=1)
        close = d < 1.4
        if not close.any():
            return np.zeros(3)
        # Weighted mean direction away from close rock, strength set by the closest point, so the push
        # doesn't grow with the number of LiDAR beams.
        w = (1.4 - d[close]) / 1.4
        away = -(pts[close] / d[close, None] * w[:, None]).sum(0)
        n = np.linalg.norm(away)
        if n < 1e-9:
            return np.zeros(3)
        return body_rotation(obs.roll, obs.pitch, obs.yaw) @ (2.5 * w.max() * away / n)

    def climb_cap(self, obs):
        """Fastest safe climb. Ceiling suction near the roof leaves the autopilot almost nothing to brake
        with, so slow down when rock is overhead; the beams reach only 45 deg up, so the steepest returns
        stand in for the roof (a wall close beside us looks like one too, hence the 0.3 m/s floor)."""
        p = obs.lidar @ body_rotation(obs.roll, obs.pitch, obs.yaw).T
        m = (p[:, 2] > 0.2) & (np.hypot(p[:, 0], p[:, 1]) < 1.05 * p[:, 2])
        up = float(p[m, 2].min()) if m.any() else np.inf
        return max(0.3, np.sqrt(3.0 * max(up - 1.1, 0.0)))      # brake at 1.5 m/s^2 short of the suction zone

    def floor_dist(self, obs):
        return obs.range_down if np.isfinite(obs.range_down) else self.info["rangefinder_max_range"]

    def follow(self, obs, v_cap=V_MAX):
        P = self.path
        near = range(self.path_i, min(self.path_i + 8, len(P)))
        self.path_i = min(near, key=lambda i: np.linalg.norm(P[i] - obs.pos))
        # Look up to 3 points ahead, but within 3 m: breadcrumbs are 2 m apart, map cells 1 m.
        k = min(self.path_i + 1, len(P) - 1)
        while k + 1 < len(P) and k < self.path_i + 3 and np.linalg.norm(P[k + 1] - obs.pos) < 3.0:
            k += 1
        look = P[k]
        to = look - obs.pos
        dist = np.linalg.norm(to)
        if dist < 1e-6:
            return np.zeros(3)
        d = to / dist
        # Near-wall turbulence pushes sideways too, where the brake-distance limit below doesn't look.
        rock = float(np.linalg.norm(obs.lidar, axis=1).min()) if len(obs.lidar) else np.inf
        v_cap = min(v_cap, 0.6 + 1.5 * max(rock - 0.5, 0.0))
        free = self.free_ahead(obs, d[None])[0]
        if free < min(2.0, dist - 0.5):          # rock before the look point (not the floor we land on)
            d, free = self.steer(obs, d)
        speed = min(v_cap, max(0.4, np.sqrt(2 * BRAKE * max(free - 1.0, 0.0))))
        if d[2] > 0:                    # keep the heading, so a slow climb doesn't fly us into the slope
            speed = min(speed, self.climb / d[2])
        if self.mode == "return":       # arrive at the landing spot instead of overshooting it
            speed = min(speed, 0.8 * np.linalg.norm(P[-1] - obs.pos) + 0.3)
        return speed * d

    def avoid_mates(self, obs, v):
        """Teammates are thin targets for the LiDAR until a couple of metres away, so use their radio
        positions: no closing faster than 0.5 m/s, and both drift a little to their own right."""
        v = v.copy()
        for m, (p, heard, mode) in self.mates.items():
            if obs.t - heard > 0.5 or mode in ("hold", "descend"):
                continue
            rel = (obs.pos - p)[:2]
            d = np.linalg.norm(rel)
            if d > MATE_R:
                continue
            u = rel / max(d, 1e-6)                  # away from the teammate
            closing = -(v[:2] @ u)
            if closing > 0.5:
                v[:2] += (closing - 0.5) * u
            v[:2] += 0.4 * (MATE_R - d) / MATE_R * np.array([-u[1], u[0]])
        return v

    # ---------------------------------------------------------------- team
    def broadcast(self, obs):
        msg = {"pos": obs.pos, "mode": self.mode, "claim": self.target if self.mode == "explore" else None,
               "seen": [(s["type"], s["pos"], s["n"]) for s in self.team],     # relayed, not just our own
               "sent": [(s["type"], s["pos"]) for s in self.team if s.get("sent")],
               "ok": self.confirmed}
        if self.tick % 20 == 0:
            msg["blocks"] = list(self.blocks)
        return [("all", msg)]

    def receive(self, obs):
        for sender, msg in obs.inbox:
            self.mates[sender] = (msg["pos"], obs.t, msg["mode"])
            self.mate_msg[sender] = msg
            c = msg.get("claim")
            if c is not None and not any(np.linalg.norm(c - x) < 2.0 for x in self.claims):
                self.claims.append(np.array(c))
            for typ, pos, n in msg["seen"]:
                merge(self.team, typ, pos, n)
            for typ, pos in msg.get("sent", ()):       # a teammate already spent budget on it
                merge(self.team, typ, pos, 0)["sent"] = True
            for typ, pos in msg["ok"]:
                if not any(t == typ and np.linalg.norm(p - pos) < SAME_R for t, p in self.confirmed):
                    self.confirmed.append((typ, pos))
            self.team_blocks.update(msg.get("blocks", ()))

    def remember(self, obs):
        for det in obs.detections:
            pos = obs.pos + body_rotation(obs.roll, obs.pitch, obs.yaw) @ det.rel_pos
            s = next((s for s in self.mine if s["type"] == det.type and np.linalg.norm(s["pos"] - pos) < SAME_R), None)
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
            self.link_t = None
            return []
        if self.link_t is None:
            self.link_t = obs.t
        # Teammates holding at home hear each other's "sent" list, but the base link on the way in can come
        # well before a link to them does: report at home, after REPORT_WAIT of base link, or at once when
        # short of battery, since a drone that dies takes its finds with it. Drones go one after another
        # (all of them have a base link from the start), so each hears what the one before sent.
        wait = 0.5 * self.id
        home = self.mode in ("hold", "descend") and obs.t - self.hold_t >= wait
        if obs.t - self.link_t < LINK_GRACE + wait or not (home or self.why == "battery/time"
                                                           or obs.t - self.link_t >= REPORT_WAIT + wait):
            return []
        out = []
        for s in self.unsent():
            s["sent"] = True
            self.my_sent.append((s["type"], s["pos"]))
            out.append((s["type"], s["pos"].tolist()))
        return out

    def unsent(self):
        return [s for s in self.team if not (s.get("done") or s.get("sent") or s["n"] < 2)
                and not any(t == s["type"] and np.linalg.norm(p - s["pos"]) < SAME_R for t, p in self.confirmed)]

    def must_hold(self, obs):
        """Stay up at home while a teammate still out there may not know what we reported.

        The base tells only the reporter what it accepted, and a landed drone goes silent, so a
        teammate arriving later would report the same artifacts again. Waiting costs no lap time: the
        mission ends when the last drone lands, and if that teammate never comes back it fails anyway."""
        if obs.battery < 0.06 or obs.t > self.info["mission_time_s"] - 20:
            return False
        if self.unsent():        # we report from home only, so ours go out first
            return True
        if not self.my_sent:
            return False
        for m in range(self.n):
            msg = self.mate_msg.get(m)
            if m == self.id or (msg and msg["mode"] == "descend"):
                continue
            known = list(msg["ok"]) + list(msg["sent"]) if msg else []
            if not all(any(t == typ and np.linalg.norm(p - pos) < SAME_R for t, p in known) for typ, pos in self.my_sent):
                return True
        return False

    def all_seen(self):
        return sum(s["n"] >= 2 for s in self.team) >= self.info["n_artifacts"]

    # ---------------------------------------------------------------- step
    def step(self, obs):
        self.tick += 1
        if obs.landed:
            return Command()
        self.track_drift(obs)
        obs = self.correct(obs)              # everything below works in the drift-corrected frame
        self.receive(obs)
        self.integrate(obs)
        self.mark_searched(obs)
        self.remember(obs)
        self.blocks.add(tuple(np.floor(obs.pos / BLOCK).astype(int)))
        self.add_crumb(obs.pos)
        here = self.cell(obs.pos)
        if self.inside(here):
            self.flown[tuple(here)] = True
        if self.last_pos is not None:
            step = np.linalg.norm(obs.pos - self.last_pos)
            self.odo += step if step < 1.0 else 0.0      # not the jump when a drift correction lands
        self.last_pos = obs.pos.copy()

        if self.mode == "explore":
            home = self.home_cost if np.isfinite(self.home_cost) else self.crumb_len(obs.pos)
            v_home = np.clip(0.9 * self.odo / max(obs.t, 1.0), 0.8, V_HOME)    # caves differ: tight ones are slow
            need = home / v_home + 15
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
                    if self.target is None or np.linalg.norm(target - self.target) > 1.0:
                        self.prog = (np.inf, obs.t)
                    self.path, self.path_i, self.target = path, 0, target
        elif self.mode == "return" and obs.t >= self.next_plan:
            self.next_plan = obs.t + 3.0
            path = [] if self.retrace else self.plan(obs, home=True)[0]
            # No map route home: retrace our own loop-free trail, which we know is flyable.
            self.path, self.path_i = path or [*self.crumbs[::-1], self.land_spot], 0

        cmd = Command(reports=self.reports(obs), send=self.broadcast(obs))
        self.climb = self.climb_cap(obs)
        if self.mode == "descend":
            to = self.land_spot - obs.pos
            fd = self.floor_dist(obs)
            # Our own slow final descent, then the sim's fixed 0.5 m/s touchdown for only the last 10 cm
            # (hull radius 0.2 m): floor gusts on top of that 0.5 m/s can exceed the 1 m/s crash speed.
            if fd < 0.3:
                cmd.land = True
                return cmd
            vz = -min(1.2, max(0.25, 0.8 * (fd - 0.4)))
            if fd < 1.0:                 # gusts are strongest by the floor: brake harder than the autopilot does
                vz += 1.5 * (vz - obs.vel[2])
            xy = 0.8 * to[:2]
            v = np.array([*xy * min(1.0, 0.5 / max(np.linalg.norm(xy), 1e-6)), vz])
        else:
            v = self.follow(obs) if self.path else np.zeros(3)
            if self.mode == "return":
                d = np.linalg.norm((self.land_spot - obs.pos)[:2])
                if d < self.best[0] - 0.5:
                    self.best = (d, obs.t)
                elif obs.t - self.best[1] > STALL_S and d < 10.0:
                    # Can't get closer: land right here, it's near base.
                    self.mode, self.why, self.land_spot = "descend", "stalled", obs.pos.copy()
            if self.mode == "return" and np.linalg.norm((self.land_spot - obs.pos)[:2]) < 3.0:
                self.path, self.path_i = [obs.pos, self.land_spot], 0       # last metres: straight in
            if self.mode == "return" and np.linalg.norm((self.land_spot - obs.pos)[:2]) < 0.6:
                self.mode, self.hold_t = "hold", obs.t
            if self.mode == "hold":
                to = self.land_spot - obs.pos
                v = np.array([0.8 * to[0], 0.8 * to[1], 0.0])
                if not self.must_hold(obs):
                    self.mode = "descend"
        push = self.avoid(obs)
        if self.mode == "descend":
            push[2] = 0.0               # the floor is where we want to go
        v = v + push
        if self.mode in ("explore", "return"):
            v = self.avoid_mates(obs, v)
        v[2] = min(v[2], self.climb)

        # No progress along the path (a gap we can't fit, a target behind rock): give up on that goal.
        left = self.togo(obs)
        if self.mode not in ("explore", "return") or left < self.prog[0] - 0.5:
            self.prog = (left, obs.t)
        elif obs.t - self.prog[1] > PROG_S:
            if self.mode == "explore" and self.target is not None:
                self.blacklist.append(self.target)
                self.path = []
            elif self.mode == "return" and not self.retrace:
                self.retrace, self.path, self.path_i = True, [*self.crumbs[::-1], self.land_spot], 0
            self.prog = (np.inf, obs.t)

        if np.hypot(v[0], v[1]) > 0.4:
            cmd.yaw_rate = 2.0 * wrap(np.arctan2(v[1], v[0]) - obs.yaw)
        elif self.mode == "explore" and self.target is not None:     # creeping: look at what we came to search
            to = self.target - obs.pos
            cmd.yaw_rate = 2.0 * wrap(np.arctan2(to[1], to[0]) - obs.yaw)
        cmd.vel = rot_z(-self.corr[0]) @ v       # back to the odom frame the autopilot flies in
        return cmd

    # ---------------------------------------------------------------- drift
    def correct(self, obs):
        """Observation moved from the odom frame into our drift-corrected frame."""
        yaw, t = self.corr
        R = rot_z(yaw)
        return replace(obs, pos=R @ obs.pos + t, yaw=wrap(obs.yaw + yaw), vel=R @ obs.vel)

    def track_drift(self, obs):
        """obs is raw (odom frame). Near home, align the last second of scans to the scans taken in the
        first seconds, when odometry was still exact; the result moves our whole frame back onto the world.
        Further out, align to the keyframe scans we took there on the way out, which puts us back in the
        frame our map and breadcrumbs were drawn in."""
        pts = body_to_odom(obs, obs.lidar[::8])
        self.recent.append(pts)
        if obs.t < ANCHOR_S:
            self.anchor_pts.append(pts)
        elif self.anchor is None and self.anchor_pts:
            self.anchor = self.cloud(np.vstack(self.anchor_pts))
        if self.tick % 10:
            return
        R, t = rot_z(self.corr[0]), self.corr[1]
        here = R @ obs.pos + t
        if self.mode == "explore":
            if obs.t > ANCHOR_S and all(np.linalg.norm(here - k) > KEY_GAP for k, _ in self.keys):
                self.keys.append((here, self.cloud(np.vstack(self.recent) @ R.T + t)))
        elif self.anchor is not None and np.linalg.norm(here - self.land_spot) < ICP_RANGE:
            self.icp(np.vstack(self.recent)[::4], self.anchor)
        elif self.keys:
            gap = [np.linalg.norm(here - k) for k, _ in self.keys]
            if min(gap) < KEY_GAP:
                self.icp(np.vstack(self.recent)[::4], self.keys[int(np.argmin(gap))][1])

    @staticmethod
    def cloud(a):
        _, keep = np.unique(np.floor(a / 0.25).astype(int), axis=0, return_index=True)
        return cKDTree(a[keep])

    def icp(self, src, ref):
        """Yaw about z plus a 3D shift from odom onto the reference scans. Wide first gates so several metres
        of drift still converge; a bad fit (large residual or implausible jump) is thrown away."""
        yaw, t = self.corr
        for gate in (6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.5):
            moved = src @ rot_z(yaw).T + t
            d, j = ref.query(moved, distance_upper_bound=gate)
            ok = np.isfinite(d)
            if ok.sum() < 200:
                return
            a, b = src[ok], ref.data[j[ok]]
            ca, cb = a.mean(0), b.mean(0)
            H = (a[:, :2] - ca[:2]).T @ (b[:, :2] - cb[:2])
            yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
            t = cb - rot_z(yaw) @ ca
        if abs(yaw) < 0.3 and np.linalg.norm(t) < 12.0 and np.median(d[ok]) < 0.3:
            # Caves repeat themselves: take a fit only if it explains most of the scan, and no fewer points
            # than the correction we already have.
            def fit(y, s):
                return np.isfinite(ref.query(src @ rot_z(y).T + s, distance_upper_bound=0.3)[0]).mean()
            if fit(yaw, t) >= max(0.5, fit(*self.corr) - 0.02):
                self.corr = (yaw, t)

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

    def togo(self, obs):
        P = self.path[self.path_i:]
        if not P:
            return np.inf
        return float(np.linalg.norm(P[0] - obs.pos) + np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

    def go_home(self, obs, why):
        self.mode, self.target, self.why = "return", None, why
        self.prog = (np.inf, obs.t)
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
