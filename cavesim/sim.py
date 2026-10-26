"""Simulator core: PyBullet world, drone dynamics, sensors, comms and scoring.

The drone is a point-mass sphere driven by a velocity controller whose thrust is
limited the way a real quadrotor's is (max tilt, max climb). That keeps the physics
honest about inertia and collisions without making students tune attitude loops.
"""
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

from . import config as C
from .api import Command, Detection, Observation

CAVE_GROUP, DRONE_GROUP = 1, 2


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Drone:
    def __init__(self, pb, i, body, pos, rng, odom: C.OdomParams):
        self.pb = pb
        self.id = i
        self.body = body
        self.rng = rng
        self.op = odom
        self.yaw = 0.0
        self.status = "flying"          # flying | landing | landed | crashed | dead
        self.battery = 1.0
        self.cmd = Command()
        # Odometry state
        self.odom_pos = np.array(pos, dtype=float)
        self.yaw_err = 0.0
        self.scale = 1.0 + rng.normal(0, odom.scale_error_std)
        self.last_true = np.array(pos, dtype=float)
        self.last_yaw = 0.0
        self.distance = 0.0
        self.trail = [list(np.round(pos, 2))]
        self.odom_trail = [list(np.round(pos, 2))]
        # Mailboxes
        self.inbox = []
        self.report_results = []
        self.error = None

    @property
    def active(self):
        return self.status in ("flying", "landing")

    def true_pos(self):
        return np.array(self.pb.getBasePositionAndOrientation(self.body)[0])

    def true_vel(self):
        return np.array(self.pb.getBaseVelocity(self.body)[0])

    def update_odom(self, dt):
        """Integrate true motion since the last call into the drifting odometry."""
        cur = self.true_pos()
        d = cur - self.last_true
        dist = float(np.linalg.norm(d))
        dyaw = abs(wrap(self.yaw - self.last_yaw))
        self.yaw_err += self.rng.normal(0, self.op.yaw_rw_std * np.sqrt(dt))
        self.yaw_err += self.rng.normal(0, self.op.yaw_noise_per_rad * dyaw + 1e-12)
        noise = self.rng.normal(0, self.op.pos_noise_per_m * np.sqrt(dist) + 1e-12, 3)
        self.odom_pos += rot_z(self.yaw_err) @ (self.scale * d) + noise
        self.distance += dist
        self.last_true, self.last_yaw = cur, self.yaw

    @property
    def odom_yaw(self):
        return wrap(self.yaw + self.yaw_err)


class Simulator:
    def __init__(self, map_dir, n_drones, agent_factory, cfg: C.SimConfig = None,
                 odom="default", seed=0):
        self.cfg = cfg or C.SimConfig()
        if isinstance(odom, str):
            self.cfg.odom = C.ODOM_PRESETS[odom]
        self.map_dir = Path(map_dir)
        self.meta = json.loads((self.map_dir / "map.json").read_text())
        self.artifacts = json.loads((self.map_dir / "truth.json").read_text())["artifacts"]
        self.base = np.array(self.meta["base"], dtype=float)
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.tick = 0

        self.pb = BulletClient(connection_mode=p.DIRECT)
        self.pb.setGravity(0, 0, -C.GRAVITY)
        self.pb.setTimeStep(self.cfg.dt)
        mesh = np.load(self.map_dir / "mesh.npz")
        col = self.pb.createCollisionShape(p.GEOM_MESH, vertices=mesh["vertices"].tolist(),
                                     indices=mesh["faces"].reshape(-1).tolist(),
                                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        self.cave = self.pb.createMultiBody(0, col)
        self.pb.setCollisionFilterGroupMask(self.cave, -1, CAVE_GROUP, CAVE_GROUP | DRONE_GROUP)

        dp = self.cfg.drone
        sphere = self.pb.createCollisionShape(p.GEOM_SPHERE, radius=dp.radius)
        self.drones = []
        for i, pos in enumerate(self.spawn_points(n_drones)):
            body = self.pb.createMultiBody(dp.mass, sphere, basePosition=pos.tolist())
            self.pb.changeDynamics(body, -1, linearDamping=0, angularDamping=0, restitution=0.1)
            self.pb.setCollisionFilterGroupMask(body, -1, DRONE_GROUP, CAVE_GROUP | DRONE_GROUP)
            self.drones.append(Drone(self.pb, i, body, pos, np.random.default_rng([seed, i]), self.cfg.odom))

        lp = self.cfg.lidar
        el = np.radians(np.linspace(*lp.vfov_deg, lp.channels))
        az = np.linspace(-np.pi, np.pi, lp.azimuth_steps, endpoint=False)
        E, A = np.meshgrid(el, az, indexing="ij")
        self.lidar_dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)

        self.reports_left = self.cfg.reports_per_artifact * len(self.artifacts)
        self.found = {}               # artifact id -> (drone id, time)
        self.report_log = []
        nodes = np.array(self.meta["graph"]["nodes"])
        self.cover_nodes = nodes[1:]
        self.visited = np.zeros(len(self.cover_nodes), dtype=bool)
        self.links = []
        self.base_links = []

        info = self.agent_info()
        self.agents = [agent_factory(d.id, n_drones, dict(info)) for d in self.drones]

    # ------------------------------------------------------------------ setup
    def spawn_points(self, n):
        if not 1 <= n <= 9:
            raise ValueError("1..9 drones fit in the staging chamber")
        cols = int(np.ceil(np.sqrt(n)))
        pts = []
        for k in range(n):
            r, c = divmod(k, cols)
            rows = int(np.ceil(n / cols))
            pts.append(self.base + np.array([(r - (rows - 1) / 2) * 1.2, (c - (cols - 1) / 2) * 1.2, 0.0]))
        return pts

    def agent_info(self):
        c = self.cfg
        return {
            "agent_hz": c.agent_hz,
            "mission_time_s": c.mission_time_s,
            "home_radius": c.home_radius,
            "max_speed_xy": c.drone.max_speed_xy,
            "max_speed_z": c.drone.max_speed_z,
            "max_yaw_rate": c.drone.max_yaw_rate,
            "drone_radius": c.drone.radius,
            "crash_speed": c.drone.crash_speed,
            "hover_endurance_s": c.drone.hover_endurance_s,
            "lidar_max_range": c.lidar.max_range,
            "lidar_channels": c.lidar.channels,
            "lidar_azimuth_steps": c.lidar.azimuth_steps,
            "lidar_vfov_deg": c.lidar.vfov_deg,
            "comms_range": c.comms.max_range,
            "max_msg_bytes": c.comms.max_msg_bytes,
            "detector_range": c.detector.max_range,
            "detector_half_fov_deg": c.detector.half_fov_deg,
            "artifact_types": list(C.ARTIFACT_TYPES),
            "n_artifacts": len(self.artifacts),
            "report_budget": self.reports_left,
            "base": self.base.tolist(),
        }

    # ---------------------------------------------------------------- sensing
    def los(self, a, b):
        """True if nothing but air lies between points a and b."""
        r = self.pb.rayTest(a.tolist(), b.tolist(), collisionFilterMask=CAVE_GROUP)[0]
        return r[0] != self.cave

    def lidar(self, d: Drone):
        lp = self.cfg.lidar
        pos = d.true_pos()
        dirs_w = self.lidar_dirs @ rot_z(d.yaw).T
        start = pos + dirs_w * (self.cfg.drone.radius + 0.02)
        end = pos + dirs_w * lp.max_range
        ranges = np.full(len(dirs_w), np.inf)
        for k in range(0, len(dirs_w), 16000):
            hits = self.pb.rayTestBatch(start[k:k + 16000].tolist(), end[k:k + 16000].tolist())
            for m, h in enumerate(hits):
                if h[0] >= 0:
                    ranges[k + m] = np.linalg.norm(np.array(h[3]) - pos)
        hit = np.isfinite(ranges)
        ranges[hit] += d.rng.normal(0, lp.range_noise_std, hit.sum())
        return ranges

    def detect(self, d: Drone):
        dp = self.cfg.detector
        pos = d.true_pos()
        fwd = np.array([np.cos(d.yaw), np.sin(d.yaw)])
        out = []
        for a in self.artifacts:
            rel = np.array(a["pos"]) - pos
            dist = np.linalg.norm(rel)
            if dist > dp.max_range or dist < 1e-6:
                continue
            h = rel[:2] / (np.linalg.norm(rel[:2]) + 1e-9)
            if np.degrees(np.arccos(np.clip(h @ fwd, -1, 1))) > dp.half_fov_deg:
                continue
            if not self.los(pos, np.array(a["pos"]) + [0, 0, 0.2]):
                continue
            body = rot_z(-d.yaw) @ rel + d.rng.normal(0, dp.pos_noise_std, 3)
            out.append(Detection(a["type"], body))
        return out

    def update_links(self):
        cr = self.cfg.comms.max_range
        live = [d for d in self.drones if d.status != "crashed" and d.status != "dead"]
        pos = {d.id: d.true_pos() for d in live}
        self.links = []
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if np.linalg.norm(pos[a.id] - pos[b.id]) <= cr and self.los(pos[a.id], pos[b.id]):
                    self.links.append((a.id, b.id))
        self.base_links = [d.id for d in live
                           if np.linalg.norm(pos[d.id] - self.base) <= self.cfg.comms.base_range
                           and self.los(pos[d.id], self.base)]

    def neighbors(self, i):
        return [b if a == i else a for a, b in self.links if i in (a, b)]

    # ---------------------------------------------------------------- control
    def apply_thrust(self, d: Drone):
        dp = self.cfg.drone
        v = d.true_vel()
        if d.status in ("landed", "crashed", "dead"):
            self.pb.applyExternalForce(d.body, -1, (-dp.linear_drag * v).tolist(), d.true_pos().tolist(), p.WORLD_FRAME)
            return
        if d.status == "landing":
            v_cmd = np.array([0.0, 0.0, -0.5])
        else:
            v_cmd = rot_z(-d.yaw_err) @ np.asarray(d.cmd.vel, dtype=float)
            sxy = np.linalg.norm(v_cmd[:2])
            if sxy > dp.max_speed_xy:
                v_cmd[:2] *= dp.max_speed_xy / sxy
            v_cmd[2] = np.clip(v_cmd[2], -dp.max_speed_z, dp.max_speed_z)
        a = 3.0 * (v_cmd - v)
        amax = C.GRAVITY * np.tan(np.radians(dp.max_tilt_deg))
        axy = np.linalg.norm(a[:2])
        if axy > amax:
            a[:2] *= amax / axy
        a[2] = np.clip(a[2], -4.0, 4.0)
        f = dp.mass * (a + [0, 0, C.GRAVITY]) + dp.linear_drag * v_cmd - dp.linear_drag * v
        self.pb.applyExternalForce(d.body, -1, f.tolist(), d.true_pos().tolist(), p.WORLD_FRAME)

    def check_contacts(self, d: Drone, v_before):
        if d.status in ("crashed", "dead"):
            return
        for c in self.pb.getContactPoints(bodyA=d.body):
            n = np.array(c[7])  # normal on B, pointing towards A (the drone)
            impact = -float(v_before @ n)
            if impact > self.cfg.drone.crash_speed:
                d.status = "crashed"
                return
            if d.status == "landing" and c[2] == self.cave and n[2] > 0.6 and np.linalg.norm(v_before) < 0.6:
                d.status = "landed"

    # ------------------------------------------------------------------- loop
    def observe(self, d: Drone):
        ranges = self.lidar(d)
        d.last_ranges = ranges
        hit = np.isfinite(ranges)
        v_body = rot_z(-d.yaw) @ d.true_vel() + d.rng.normal(0, 0.03, 3)
        obs = Observation(
            t=self.t, drone_id=d.id, pos=d.odom_pos.copy(), yaw=d.odom_yaw, vel=v_body,
            battery=d.battery, lidar=self.lidar_dirs[hit] * ranges[hit, None],
            lidar_dirs=self.lidar_dirs, lidar_ranges=ranges, detections=self.detect(d),
            inbox=d.inbox, neighbors=self.neighbors(d.id), base_link=d.id in self.base_links,
            report_results=d.report_results, landed=d.status == "landed")
        d.inbox, d.report_results = [], []
        return obs

    def step(self):
        """Advance one agent tick (1 / agent_hz seconds). Returns False when the mission is over."""
        cfg = self.cfg
        self.update_links()
        outgoing = []
        for d, agent in zip(self.drones, self.agents):
            if not d.active:
                continue
            try:
                cmd = agent.step(self.observe(d))
                if not isinstance(cmd, Command):
                    raise TypeError(f"step() must return a Command, got {type(cmd).__name__}")
            except Exception:
                err = traceback.format_exc()
                if err != d.error:
                    print(f"[drone {d.id}] agent error:\n{err}")
                d.error = err
                cmd = Command()
            d.cmd = cmd
            if cmd.land and d.status == "flying":
                d.status = "landing"
            outgoing.append((d, cmd))
        self.deliver(outgoing)

        tick_dt = 1.0 / cfg.agent_hz
        for d in self.drones:
            d.yaw = wrap(d.yaw + np.clip(d.cmd.yaw_rate, -cfg.drone.max_yaw_rate, cfg.drone.max_yaw_rate)
                         * tick_dt) if d.status == "flying" else d.yaw
        for _ in range(cfg.steps_per_tick):
            vels = [d.true_vel() for d in self.drones]
            for d in self.drones:
                self.apply_thrust(d)
            self.pb.stepSimulation()
            for d, v in zip(self.drones, vels):
                self.check_contacts(d, v)
        self.t += tick_dt
        self.tick += 1

        for d in self.drones:
            if d.active:
                speed = np.linalg.norm(d.true_vel()) / cfg.drone.max_speed_xy
                d.battery -= tick_dt / cfg.drone.hover_endurance_s * (1 + 0.3 * speed ** 2)
                if d.battery <= 0:
                    d.battery, d.status = 0.0, "dead"
            self.pb.resetBaseVelocity(d.body, angularVelocity=[0, 0, 0])
            d.update_odom(tick_dt)
            tp = d.true_pos()
            if self.tick % 5 == 0:
                d.trail.append(list(np.round(tp, 2)))
                d.odom_trail.append(list(np.round(d.odom_pos, 2)))
            if d.status != "crashed":
                near = np.linalg.norm(self.cover_nodes - tp, axis=1) < 4.0
                self.visited |= near
        return self.t < cfg.mission_time_s and any(d.active for d in self.drones)

    def deliver(self, outgoing):
        limit = self.cfg.comms.max_msg_bytes
        for d, cmd in outgoing:
            nbrs = self.neighbors(d.id)
            for dest, payload in cmd.send:
                blob = pickle.dumps(payload)
                if len(blob) > limit:
                    print(f"[drone {d.id}] message of {len(blob)} bytes dropped (limit {limit})")
                    continue
                for j in (nbrs if dest == "all" else [dest]):
                    if j in nbrs:
                        self.drones[j].inbox.append((d.id, pickle.loads(blob)))
            for rep in cmd.reports:
                d.report_results.append((rep, *self.judge_report(d, rep)))

    def judge_report(self, d, rep):
        if d.id not in self.base_links:
            return False, "no base link"
        if self.reports_left <= 0:
            return False, "report budget exhausted"
        self.reports_left -= 1
        typ, pos = rep
        pos = np.asarray(pos, dtype=float)
        best = None
        for a in self.artifacts:
            if a["type"] != typ or a["id"] in self.found:
                continue
            dist = np.linalg.norm(np.array(a["pos"]) - pos)
            if dist <= self.cfg.report_match_dist and (best is None or dist < best[1]):
                best = (a, dist)
        self.report_log.append({"t": round(self.t, 1), "drone": d.id, "type": typ,
                                "pos": np.round(pos, 2).tolist(), "hit": best is not None})
        if best is None:
            return False, "no match"
        self.found[best[0]["id"]] = (d.id, self.t)
        return True, f"matched artifact at {best[1]:.1f} m"

    # ---------------------------------------------------------------- results
    def score(self):
        returned = [d.id for d in self.drones if d.status == "landed"
                    and np.linalg.norm((d.true_pos() - self.base)[:2]) <= self.cfg.home_radius]
        coverage = float(self.visited.mean()) if len(self.visited) else 0.0
        drift = [float(np.linalg.norm(d.odom_pos - d.true_pos())) for d in self.drones]
        total = 10 * len(self.found) + 5 * len(returned) + 20 * coverage
        return {
            "score": round(total, 2),
            "artifacts_found": len(self.found),
            "artifacts_total": len(self.artifacts),
            "returned": returned,
            "crashed": [d.id for d in self.drones if d.status == "crashed"],
            "battery_dead": [d.id for d in self.drones if d.status == "dead"],
            "coverage": round(coverage, 3),
            "reports_used": len(self.report_log),
            "final_drift_m": [round(x, 2) for x in drift],
            "distance_m": [round(d.distance, 1) for d in self.drones],
            "time_s": round(self.t, 1),
        }

    def viewer_state(self, lidar=True):
        drones = []
        for d, agent in zip(self.drones, self.agents):
            entry = {
                "id": d.id, "pos": np.round(d.true_pos(), 2).tolist(), "yaw": round(d.yaw, 3),
                "odom": np.round(d.odom_pos, 2).tolist(), "odom_yaw": round(d.odom_yaw, 3),
                "status": d.status, "battery": round(d.battery, 3),
            }
            r = getattr(d, "last_ranges", None)
            if lidar and d.active and r is not None:
                hit = np.isfinite(r)
                pts = (self.lidar_dirs[hit] * r[hit, None]) @ rot_z(d.yaw).T + d.true_pos()
                entry["lidar"] = np.round(pts[::2], 2).reshape(-1).tolist()
            try:
                dbg = agent.debug() or {}
                entry["debug"] = json.loads(json.dumps(dbg, default=lambda o: np.asarray(o).tolist()))
            except Exception:
                entry["debug"] = {}
            drones.append(entry)
        return {"type": "state", "t": round(self.t, 2), "drones": drones, "links": self.links,
                "base_links": self.base_links, "found": sorted(self.found), "score": self.score()}

    def viewer_init(self):
        return {"type": "init", "map": self.map_dir.name, "base": self.base.tolist(),
                "artifacts": self.artifacts, "n_drones": len(self.drones),
                "home_radius": self.cfg.home_radius, "comms_range": self.cfg.comms.max_range,
                "mission_time_s": self.cfg.mission_time_s}

    def trails(self):
        return {"type": "trails", "true": [d.trail for d in self.drones],
                "odom": [d.odom_trail for d in self.drones]}

    def close(self):
        if self.pb is not None:
            self.pb.disconnect()
            self.pb = None


def run_mission(map_dir, agent_factory, n_drones=3, odom="default", seed=0, cfg=None, verbose=False):
    """Headless run for experiments: returns the score dict."""
    sim = Simulator(map_dir, n_drones, agent_factory, cfg=cfg, odom=odom, seed=seed)
    try:
        while sim.step():
            if verbose and sim.tick % (10 * sim.cfg.agent_hz) == 0:
                print(f"t={sim.t:5.0f}s  {sim.score()}")
        return sim.score()
    finally:
        sim.close()
