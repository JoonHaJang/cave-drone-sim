"""Baseline agent: reactive exploration + breadcrumb return. No map, no cooperation.

It wanders toward open space it has not seen, keeps a breadcrumb chain home, turns
back when the battery says so, and reports whatever it saw once it reaches base
radio range. Every drone acts alone, so they tend to explore the same tunnels.
Beating this baseline is the point of the term project.

    python -m cavesim.run --agent agents.explorer --drones 3
"""
import numpy as np

from cavesim.api import Agent, Command

CRUMB_STEP = 2.0      # m between breadcrumbs
CELL = 2.5            # m, visited-cell size for the novelty bonus


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ExplorerAgent(Agent):
    def __init__(self, drone_id, n_drones, info):
        super().__init__(drone_id, n_drones, info)
        self.mode = "explore"
        self.crumbs = []          # loop-erased path from base to here, odom frame
        self.visited = set()
        self.heading = 2 * np.pi * drone_id / n_drones   # fan out at the start
        self.artifacts = []       # dicts: type, pos (odom), n, reported
        self.goal = None
        self.rng = np.random.default_rng(drone_id)
        self.stall = 0            # ticks without progress (stuck against a wall)
        az = np.linspace(-np.pi, np.pi, info["lidar_azimuth_steps"], endpoint=False)
        el = np.radians(np.linspace(*info["lidar_vfov_deg"], info["lidar_channels"]))
        self.az = az
        self.level = np.abs(el) <= np.radians(10)
        self.up = el >= np.radians(30)
        self.down = el <= np.radians(-30)
        self.el = el

    # ---------------------------------------------------------------- helpers
    def cell(self, p):
        return (int(np.floor(p[0] / CELL)), int(np.floor(p[1] / CELL)), int(np.floor(p[2] / CELL)))

    def add_crumb(self, pos):
        if self.crumbs and np.linalg.norm(pos - self.crumbs[-1]) < CRUMB_STEP:
            return
        # Loop erasure: if we are back near an older crumb, cut the loop out.
        for k, c in enumerate(self.crumbs[:-2]):
            if np.linalg.norm(pos - c) < CRUMB_STEP:
                del self.crumbs[k + 1:]
                break
        self.crumbs.append(pos.copy())

    def path_home(self, pos):
        pts = [pos] + self.crumbs[::-1] + [np.zeros(3)]
        return float(sum(np.linalg.norm(a - b) for a, b in zip(pts, pts[1:])))

    def avoid(self, obs):
        """Repulsion from nearby LiDAR points, in odom frame."""
        pts = obs.lidar
        d = np.linalg.norm(pts, axis=1)
        close = d < 1.6
        if not close.any():
            return np.zeros(3)
        push = -(pts[close] / d[close, None]) * ((1.6 - d[close]) / 1.6)[:, None]
        return rot_z(obs.yaw) @ (2.0 * push.sum(0) / max(1, close.sum() ** 0.5))

    def hold_middle(self, obs):
        """Vertical speed that keeps the drone halfway between floor and ceiling."""
        r = np.nan_to_num(obs.lidar_ranges.reshape(len(self.el), -1), posinf=self.info["lidar_max_range"])
        up = (r[self.up] * np.sin(self.el[self.up])[:, None]).min()
        down = (r[self.down] * -np.sin(self.el[self.down])[:, None]).min()
        return float(np.clip(0.6 * (up - down) / 2, -0.8, 0.8))

    # ------------------------------------------------------------------ modes
    def explore(self, obs):
        # Stuck in a dead end: pick a fresh heading behind us.
        self.stall = self.stall + 1 if np.linalg.norm(obs.vel[:2]) < 0.3 else 0
        if self.stall > 20:
            self.heading = wrap(self.heading + np.pi + self.rng.uniform(-0.8, 0.8))
            self.stall = 0
        r = np.nan_to_num(obs.lidar_ranges.reshape(len(self.el), -1), posinf=self.info["lidar_max_range"])
        free = r[self.level].min(0)                       # per azimuth, body frame
        best, best_score = self.heading, -np.inf
        for a, f in zip(self.az, free):
            h = wrap(obs.yaw + a)
            dirv = np.array([np.cos(h), np.sin(h), 0.0])
            novelty = sum(self.cell(obs.pos + k * dirv) not in self.visited for k in (4.0, 8.0))
            score = min(f, 10.0) + 3.0 * novelty - 2.5 * abs(wrap(h - self.heading))
            if f > 2.0 and score > best_score:
                best, best_score = h, score
        self.heading = best
        ahead = free[np.argmin(np.abs(wrap(self.az - wrap(best - obs.yaw))))]
        speed = self.info["max_speed_xy"] * 0.7 * np.clip((ahead - 1.5) / 4.0, 0.25, 1.0)
        v = speed * np.array([np.cos(best), np.sin(best), 0.0])
        self.goal = obs.pos + 4 * v / max(speed, 1e-6)
        return v

    def go_home(self, obs):
        # Drift can put the next crumb behind a wall; skip it when we stop making progress.
        self.stall = self.stall + 1 if np.linalg.norm(obs.vel[:2]) < 0.3 else 0
        if self.stall > 20 and self.crumbs:
            self.crumbs.pop()
            self.stall = 0
        while self.crumbs and np.linalg.norm(obs.pos - self.crumbs[-1]) < 2.0:
            self.crumbs.pop()
        target = self.crumbs[-1] if self.crumbs else np.array([0.0, 0.0, obs.pos[2]])
        self.goal = target
        to = target - obs.pos
        dist = np.linalg.norm(to)
        if not self.crumbs and np.linalg.norm(to[:2]) < 0.8:
            self.mode = "land"
            return np.zeros(3)
        speed = self.info["max_speed_xy"] * 0.6 * min(1.0, dist / 2.0)
        self.heading = np.arctan2(to[1], to[0])
        return speed * to / max(dist, 1e-6)

    # ------------------------------------------------------------- artifacts
    def remember(self, obs):
        for det in obs.detections:
            pos = obs.pos + rot_z(obs.yaw) @ det.rel_pos
            for a in self.artifacts:
                if a["type"] == det.type and np.linalg.norm(a["pos"] - pos) < 4.0:
                    a["pos"] = (a["pos"] * a["n"] + pos) / (a["n"] + 1)
                    a["n"] += 1
                    break
            else:
                self.artifacts.append({"type": det.type, "pos": pos, "n": 1, "reported": False})

    def reports(self, obs):
        if not obs.base_link:
            return []
        out = []
        for a in self.artifacts:
            if not a["reported"] and a["n"] >= 3:
                a["reported"] = True
                out.append((a["type"], a["pos"].tolist()))
        return out

    # ------------------------------------------------------------------- step
    def step(self, obs):
        if obs.landed:
            return Command()
        self.add_crumb(obs.pos)
        self.visited.add(self.cell(obs.pos))
        self.remember(obs)

        if self.mode == "explore":
            need = self.path_home(obs.pos) / (0.6 * self.info["max_speed_xy"])
            spare_batt = obs.battery - (need / self.info["hover_endurance_s"] * 1.8 + 0.10)
            spare_time = self.info["mission_time_s"] - obs.t - (need * 1.8 + 20)
            if spare_batt < 0 or spare_time < 0:
                self.mode = "return"

        if self.mode == "land":
            return Command(land=True, reports=self.reports(obs))
        v = self.explore(obs) if self.mode == "explore" else self.go_home(obs)
        v = v + self.avoid(obs)
        v[2] += self.hold_middle(obs)
        yaw_rate = 2.0 * wrap(self.heading - obs.yaw)
        return Command(vel=v, yaw_rate=yaw_rate, reports=self.reports(obs))

    def debug(self):
        return {"mode": self.mode, "path": [c.tolist() for c in self.crumbs],
                "goal": None if self.goal is None else self.goal.tolist(),
                "points": [a["pos"].tolist() for a in self.artifacts]}


Agent = ExplorerAgent
