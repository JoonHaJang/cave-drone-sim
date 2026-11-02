"""Simulator core: PyBullet world, drone dynamics, sensors, comms and scoring.

The drone is a point-mass sphere driven by a velocity controller whose thrust is
limited the way a real quadrotor's is (max tilt, max climb). The airframe attitude
follows the thrust direction with a short lag, so body-fixed sensors tilt and the
IMU sees realistic specific force and angular rate. That keeps the physics honest
about inertia and collisions without making students tune attitude loops.

Near rock the air is not still: gusts, wall turbulence and the ceiling/ground effect
push the drone around (config.AeroParams). The autopilot only corrects them after
the fact, so threading between stalactites needs clearance margin and slow speed.
"""
import json
import pickle
import traceback
from pathlib import Path

import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial.transform import Rotation

from . import config as C
from .api import Command, Detection, Observation
from .onboard import FlightController, mixer
from .sensors import (Barometer, Imu, Lidar, Rangefinder, angular_velocity,
                      degeneracy, roll_pitch, rot_z)

CAVE_GROUP, DRONE_GROUP, FENCE_GROUP = 1, 2, 4   # fences: drones bump into them, sensors do not see them
RESCUE_RADIUS = 3.0              # m from the survivor (with line of sight) that counts as reaching it
# Course boundary (SubT worlds): the tunnels hang in empty space, which in reality is solid rock,
# and the staging terrain is a sheet with empty space under it, which in reality is soil.
# In the staging box a drone must have ground below; outside it, rock above and below, within ENCLOSED_M;
# more than OUT_GRACE_S outside that ends its run as "out" (out of course), like a crash.
ENCLOSED_M = 30.0
OUT_GRACE_S = 1.0
DOWN_STATES = ("landed", "crashed", "dead", "out")   # motors off
MESH_CHUNK = 40_000             # triangles per PyBullet mesh shape
ATTITUDE_EST_STD = 0.005   # rad, flight-controller roll/pitch estimate noise


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Drone:
    def __init__(self, pb, i, body, pos, rng, odom, cfg: C.SimConfig):
        self.pb = pb
        self.id = i
        self.body = body
        self.rng = rng
        self.odom_on = odom is not None
        self.op = odom or C.ODOM_PRESETS["default"]   # heading still drifts when odom is off
        self.fc = FlightController(cfg.drone)      # the onboard flight controller
        self.R = np.eye(3)                          # body -> world
        self.omega = np.zeros(3)                    # body rates, rad/s
        self.last_hit = 0.0                         # m/s, the latest impact (names a flip)
        self.motors = np.full(4, cfg.drone.mass * C.GRAVITY / 4)   # thrust of each rotor, N (hovering)
        self.status = "flying"          # flying | landing | landed | crashed | dead
        self.battery = 1.0
        self.cmd = Command()
        self.imu = Imu(cfg.imu, rng)
        self.baro = Barometer(cfg.baro, pos[2])
        self.imu_buf = []
        self.pose_buf = []              # (pos, R) per physics step since the last sweep
        self.last_obs = None
        self.last_scan = None           # (ranges, R, pos) of the latest LiDAR sweep
        self.last_truth = None          # ground truth at the moment of last_obs
        # Odometry state
        self.odom_pos = np.array(pos, dtype=float)
        self.yaw_err = 0.0
        self.scale = 1.0 + rng.normal(0, self.op.scale_error_std)
        self.last_true = np.array(pos, dtype=float)
        self.last_yaw = 0.0
        self.distance = 0.0
        self.gust = np.zeros(3)         # unit-variance Gauss-Markov state of the air disturbance
        self.out_s = 0.0                # s spent outside the course boundary
        self.aero_acc = np.zeros(3)     # last disturbance acceleration, for the viewer
        self.mix = [0.0, 0.0, 0.0, 0.0]  # [thrust / weight, roll, pitch, yaw torque demand] for the viewer's rotors
        self.clearance = np.inf         # m from the hull to the nearest rock (6 axis rays)
        self.floor_gap = np.inf         # m from the hull down to the floor (lifts dust)
        self.scan_normals = np.zeros((0, 3))  # rock normals of the last sweep (odometry degeneracy)
        self.hits = []                  # impact speeds (m/s) of every collision
        self.touching = False           # a counted hit whose contact has not yet ended
        self.gps_err = np.zeros(3)      # slowly wandering GNSS error (multipath, atmosphere)
        self.crash_cause = None
        self.degen = np.zeros(3)        # last degeneracy weights, for tests and debugging
        self.trail = [list(np.round(pos, 2))]
        self.odom_trail = [list(np.round(pos, 2))]
        # Mailboxes
        self.inbox = []
        self.report_results = []
        self.error = None

    @property
    def yaw(self):
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    @property
    def thrust_dir(self):
        return self.R[:, 2]

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
        self.yaw_err += self.rng.normal(0, self.op.yaw_noise_per_rad * np.sqrt(dyaw) + 1e-12)   # variance ~ angle turned
        # Along directions no wall constrains, scan matching fails: more noise per metre,
        # and the IMU-only estimate walks even while hovering.
        V, w = degeneracy(self.scan_normals, self.op.degen_lambda)
        self.degen = w
        sigma = self.op.pos_noise_per_m * np.sqrt(dist) * (1 + (self.op.degen_gain - 1) * w)
        walk = self.op.degen_drift * np.sqrt(dt) * w
        noise = V @ (self.rng.normal(0, 1, 3) * (sigma + walk))
        self.odom_pos += rot_z(self.yaw_err) @ (self.scale * d) + noise
        self.distance += dist
        self.last_true, self.last_yaw = cur, self.yaw

    @property
    def odom_yaw(self):
        # The ZYX yaw of the airframe, as Observation promises -- not the heading the attitude is
        # built from (attitude_from_thrust): tilted, the two differ by degrees, and every return
        # placed with body_rotation(roll, pitch, yaw) would land that far off.
        return wrap(np.arctan2(self.R[1, 0], self.R[0, 0]) + self.yaw_err)


class Simulator:
    def __init__(self, map_dir, n_drones, agent_factory, cfg: C.SimConfig = None,
                 odom="default", lidar=None, seed=0, aero=None, imu=None):
        self.cfg = cfg or C.SimConfig()
        if isinstance(odom, str):
            self.cfg.odom = C.ODOM_PRESETS[odom]
        if isinstance(lidar, str):
            self.cfg.lidar = C.LIDAR_PRESETS[lidar]
        if isinstance(aero, str):
            self.cfg.aero = C.AERO_PRESETS[aero]
        if isinstance(imu, str):
            self.cfg.imu = C.IMU_PRESETS[imu]
        self.map_dir = Path(map_dir)
        self.meta = json.loads((self.map_dir / "map.json").read_text())
        # Rescue courses (subt_import --survivor) carry their own rules: one survivor to reach and
        # come home from, timed as a lap; battery and time limit sized for the course.
        self.mission = self.meta.get("mission", {"type": "explore"})
        if self.mission["type"] == "rescue":
            self.cfg.drone.hover_endurance_s = self.mission["battery_s"]
            self.cfg.mission_time_s = self.mission["time_s"]
        self.staging_box = np.array(self.meta["staging_box"]) if "staging_box" in self.meta else None
        self.rescued = {}             # drone id -> time it reached the survivor
        self.laps = {}                # drone id -> time it landed home after a rescue
        self.artifacts = json.loads((self.map_dir / "truth.json").read_text())["artifacts"]
        self.base = np.array(self.meta["base"], dtype=float)
        self.rng = np.random.default_rng(seed)
        u = np.random.default_rng([seed, 7919]).normal(size=3)        # own stream: the flight noise stays as it was
        self.target_hint_offset = self.cfg.target_error_m * u / np.linalg.norm(u)
        self.t = 0.0
        self.tick = 0

        self.pb = BulletClient(connection_mode=p.DIRECT)
        self.pb.setGravity(0, 0, -C.GRAVITY)
        self.pb.setTimeStep(self.cfg.dt)
        mesh = np.load(self.map_dir / "mesh.npz")
        # PyBullet takes at most 2^17 vertices per mesh shape, so big maps (the SubT
        # worlds) are loaded as several static bodies of MESH_CHUNK triangles each.
        V, F = mesh["vertices"], mesh["faces"]
        self.caves = set()
        for k in range(0, len(F), MESH_CHUNK):
            used, f = np.unique(F[k:k + MESH_CHUNK], return_inverse=True)
            col = self.pb.createCollisionShape(p.GEOM_MESH, vertices=V[used].tolist(),
                                               indices=f.reshape(-1).tolist(), flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            body = self.pb.createMultiBody(0, col)
            self.pb.setCollisionFilterGroupMask(body, -1, CAVE_GROUP, CAVE_GROUP | DRONE_GROUP)
            self.caves.add(body)
        # Course fences (SubT staging area): collision-only boxes, as in Gazebo, where LiDAR only
        # sees what is rendered.
        for f in self.meta.get("fences", []):
            T = np.array(f["pose"])
            col = self.pb.createCollisionShape(p.GEOM_BOX, halfExtents=(np.array(f["size"]) / 2).tolist())
            q = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
            body = self.pb.createMultiBody(0, col, basePosition=T[:3, 3].tolist(), baseOrientation=q)
            self.pb.setCollisionFilterGroupMask(body, -1, FENCE_GROUP, DRONE_GROUP)

        dp = self.cfg.drone
        # A thin disc like the airframe (a ball of the same radius would land 0.2 m up on its belly).
        disc = self.pb.createCollisionShape(p.GEOM_CYLINDER, radius=dp.radius, height=dp.height)
        self.mixer = mixer(dp)                                    # motor thrusts -> [T, torques], body
        self.drones = []
        for i, pos in enumerate(self.spawn_points(n_drones)):
            body = self.pb.createMultiBody(dp.mass, disc, basePosition=pos.tolist())
            self.pb.changeDynamics(body, -1, linearDamping=0, angularDamping=0, restitution=0.1)
            self.pb.setCollisionFilterGroupMask(body, -1, DRONE_GROUP, CAVE_GROUP | DRONE_GROUP | FENCE_GROUP)
            self.drones.append(Drone(self.pb, i, body, pos, np.random.default_rng([seed, i]),
                                     self.cfg.odom, self.cfg))

        self.lidar = Lidar(self.cfg.lidar, dp.radius + 0.02)
        self.lidar_dirs = self.lidar.dirs
        self.rangefinder = Rangefinder(self.cfg.rangefinder)

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
            "lidar_min_range": c.lidar.min_range,
            "lidar_channels": c.lidar.channels,
            "lidar_azimuth_steps": c.lidar.azimuth_steps,
            "lidar_vfov_deg": c.lidar.vfov_deg,
            "lidar_rate_hz": c.lidar.rate_hz,
            "lidar_mount": [0.0, 0.0, c.lidar.mount_z],      # sensor origin in the body frame
            "imu_rate_hz": c.imu.rate_hz,
            "rangefinder_max_range": c.rangefinder.max_range,
            "odom_available": c.odom is not None,
            "comms_range": c.comms.max_range,
            "max_msg_bytes": c.comms.max_msg_bytes,
            "detector_range": c.detector.max_range,
            "detector_half_fov_deg": c.detector.half_fov_deg,
            "artifact_types": list(C.ARTIFACT_TYPES),
            "n_artifacts": len(self.artifacts),
            "report_budget": self.reports_left,
            "base": self.base.tolist(),
            # Where the cave starts when the drones take off outside it (SubT worlds), else None.
            "entrance": self.meta.get("entrance"),
            # Straight legs from the base into the cave (the outdoor approach is flown on GPS).
            "entrance_route": self.meta.get("entrance_route"),
            "mission": self.mission["type"],
            # Outdoor staging area [[xmin, ymin, zmin], [xmax, ymax, zmax]]; beyond it the course is the cave.
            "staging_box": None if self.staging_box is None else self.staging_box.tolist(),
            # Rescue: where the survivor is (the route is for the agent to find), else None.
            # Approximate: off by SimConfig.target_error_m; reaching the true survivor is what counts.
            "target_error_m": c.target_error_m,             # how far off "target" may be
            "target": (self.target() + self.target_hint_offset).tolist() if self.mission["type"] == "rescue" else None,
        }

    def target(self):
        return np.array(self.artifacts[self.mission["target"]]["pos"], dtype=float)

    def check_course(self, dt):
        """Out of course: in the staging box with no ground below (under the terrain sheet), or
        outside it without rock both above and below (outside the tunnels)."""
        live = [d for d in self.drones if d.active]
        if self.staging_box is None or not live:
            return
        pos = [d.true_pos() for d in live]
        starts = [p.tolist() for p in pos for _ in (0, 1)]
        ends = [(p + [0, 0, s * ENCLOSED_M]).tolist() for p in pos for s in (1, -1)]
        hits = self.pb.rayTestBatch(starts, ends, collisionFilterMask=CAVE_GROUP)
        lo, hi = self.staging_box
        for k, (d, p) in enumerate(zip(live, pos)):
            inside = np.all(p >= lo) and np.all(p <= hi)
            above, below = hits[2 * k][0] in self.caves, hits[2 * k + 1][0] in self.caves
            ok = below and (inside or above)
            d.out_s = 0.0 if ok else d.out_s + dt
            if d.out_s > OUT_GRACE_S:
                d.status = "out"

    def update_rescue(self):
        """Rescue: within RESCUE_RADIUS of the survivor with a clear line counts as a rescue;
        landing home afterwards stops that drone's lap clock."""
        goal = self.target()
        for d in self.drones:
            p = d.true_pos()
            if d.id not in self.rescued and d.active and np.linalg.norm(p - goal) < RESCUE_RADIUS \
                    and self.los(p, goal + [0, 0, 0.3]):
                self.rescued[d.id] = self.t
                self.found.setdefault(self.mission["target"], (d.id, self.t))
            if d.id in self.rescued and d.id not in self.laps and d.status == "landed" \
                    and np.linalg.norm((p - self.base)[:2]) <= self.cfg.home_radius:
                self.laps[d.id] = self.t

    # ---------------------------------------------------------------- sensing
    def los(self, a, b):
        """True if nothing but air lies between points a and b."""
        r = self.pb.rayTest(a.tolist(), b.tolist(), collisionFilterMask=CAVE_GROUP)[0]
        return r[0] not in self.caves

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
            body = d.R.T @ rel + d.rng.normal(0, dp.pos_noise_std, 3)
            out.append(Detection(a["type"], body))
        return out

    def update_links(self):
        cr = self.cfg.comms.max_range
        live = [d for d in self.drones if d.status not in ("crashed", "dead", "out")]
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
    AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)

    def aero_disturbance(self, dt):
        """Gusts, wall turbulence, ceiling and ground effect for every flying drone (m/s^2)."""
        ap, r = self.cfg.aero, self.cfg.drone.radius
        live = [d for d in self.drones if d.active]
        if not live:
            return
        reach = r + max(ap.wall_range, ap.ceiling_range, ap.ground_range)
        hull = np.array([r, r, r, r, self.cfg.drone.height / 2, self.cfg.drone.height / 2])   # disc: thin up and down
        starts, ends = [], []
        for d in live:
            pos = d.true_pos()
            starts += [pos.tolist()] * 6
            ends += (pos + self.AXES * reach).tolist()
        hits = self.pb.rayTestBatch(starts, ends, collisionFilterMask=CAVE_GROUP)
        phi = np.exp(-dt / ap.gust_tau)
        for k, d in enumerate(live):
            h = hits[6 * k: 6 * k + 6]
            gap = np.array([x[2] * reach if x[0] in self.caves else np.inf for x in h]) - hull
            d.clearance = max(float(gap.min()), 0.0)
            d.floor_gap = max(float(gap[5]), 0.0)
            near = np.clip(1 - d.clearance / ap.wall_range, 0, 1) if np.isfinite(d.clearance) else 0.0
            d.gust = phi * d.gust + np.sqrt(1 - phi ** 2) * d.rng.normal(0, 1, 3)
            acc = (ap.draft_std + ap.wall_turb_std * near ** 2) * d.gust
            up, down = gap[4], gap[5]
            if up < ap.ceiling_range:
                acc[2] += ap.ceiling_gain * C.GRAVITY * (1 - max(up, 0) / ap.ceiling_range) ** 2
            if down < ap.ground_range:
                acc[2] += ap.ground_gain * C.GRAVITY * (1 - max(down, 0) / ap.ground_range) ** 2
            d.aero_acc = acc

    def apply_thrust(self, d: Drone, dt):
        dp = self.cfg.drone
        v = d.true_vel()
        if d.status in DOWN_STATES:
            self.pb.applyExternalForce(d.body, -1, (-dp.linear_drag * v).tolist(), d.true_pos().tolist(), p.WORLD_FRAME)
            return
        # The onboard flight controller (cavesim.onboard): setpoints in, four motor commands out.
        if d.status == "landing":
            v_cmd, yaw_rate = np.array([0.0, 0.0, -0.5]), 0.0
        else:
            v_cmd, yaw_rate = d.fc.limit(rot_z(-d.yaw_err) @ np.asarray(d.cmd.vel, dtype=float)), d.cmd.yaw_rate
        f_cmd = d.fc.update(v_cmd, yaw_rate, v, d.R, d.omega, dt)
        # The world: motors lag their commands; thrust and torques turn and push the rigid body.
        d.motors += (f_cmd - d.motors) * min(1.0, dt / dp.motor_tau)
        wrench = self.mixer @ d.motors                       # [T, tau_x, tau_y, tau_z], body frame
        J = np.diag(dp.inertia)
        d.omega = d.omega + dt * np.linalg.solve(J, wrench[1:] - np.cross(d.omega, J @ d.omega))
        d.R = d.R @ Rotation.from_rotvec(d.omega * dt).as_matrix()
        u, _, vt = np.linalg.svd(d.R)                        # keep it a rotation
        d.R = u @ vt
        d.mix = [float(x) for x in d.fc.demand]
        f = wrench[0] * d.R[:, 2] - dp.linear_drag * v + dp.mass * d.aero_acc
        self.pb.applyExternalForce(d.body, -1, f.tolist(), d.true_pos().tolist(), p.WORLD_FRAME)
        if d.status == "flying" and d.R[2, 2] < np.cos(np.radians(dp.upset_limit_deg)):
            d.status, d.crash_cause = "crashed", f"flipped by a {d.last_hit:.1f} m/s hit"

    def check_contacts(self, d: Drone, v_before):
        if d.status in ("crashed", "dead", "out"):
            return
        dp = self.cfg.drone
        worst, wn = 0.0, None
        contacts = self.pb.getContactPoints(bodyA=d.body)
        was_touching = d.touching = d.touching and bool(contacts)
        for c in contacts:
            n = np.array(c[7])  # normal on B, pointing towards A (the drone)
            impact = -float(v_before @ n)
            if d.status == "landing" and c[2] in self.caves and n[2] > 0.6 and np.linalg.norm(v_before) < 0.6:
                d.status = "landed"
                return
            if impact > worst:
                worst, wn = impact, n
        if worst < 0.3:                  # pressed or sliding against the rock, not a hit
            return
        if was_touching and d.hits:      # the same collision, still going on: one hit, at its worst
            d.hits[-1] = max(d.hits[-1], worst)
        else:
            d.hits.append(worst)
        d.touching = True
        # PyBullet has already bounced the hull. What it cannot see: the airframe is a disc, not a
        # ball, so the rim point nearest the rock takes the impulse J off-centre and tips it.
        zb = d.R[:, 2]
        if worst > dp.crash_speed:
            d.status, d.crash_cause = "crashed", f"frame broken at {worst:.1f} m/s"
            return
        if not dp.prop_guards and abs(wn @ zb) < 0.7 and worst > dp.prop_strike_speed:
            d.status, d.crash_cause = "crashed", f"prop strike at {worst:.1f} m/s"
            return
        rim = -wn - (-wn @ zb) * zb
        if np.linalg.norm(rim) < 1e-6:   # flat face on: no lever arm
            return
        rim *= dp.radius / np.linalg.norm(rim)
        inertia = dp.mass * dp.radius ** 2 / 4                    # thin disc, in-plane axis
        arm = np.cross(rim, wn)
        # Rigid-body impulse at an off-centre point: part of the hit goes into spin, so the
        # effective mass is 1 / (1/m + |r x n|^2 / I). Restitution 0.1, as the hull.
        # ponytail: PyBullet's ball already took the linear impulse with the full mass.
        J = (1 + 0.1) * worst / (1 / dp.mass + arm @ arm / inertia)
        dw = J * arm / inertia                                    # world frame
        if not was_touching:                                      # one impulse per collision, not per physics step
            d.omega = d.omega + d.R.T @ dw                        # the spin it takes; the rate loop catches it
        d.last_hit = worst                                        # if it does not: apply_thrust calls the flip

    # ------------------------------------------------------------------- loop
    def observe(self, d: Drone):
        pos = d.true_pos()
        ranges, d.scan_normals = self.lidar.scan(self.pb, d.pose_buf or [(pos, d.R)], d.rng, d.floor_gap)
        d.last_scan = (ranges, d.R.copy(), pos)
        hit = np.isfinite(ranges)
        roll, pitch = roll_pitch(d.R)
        odom = d.odom_on
        obs = Observation(
            t=self.t, drone_id=d.id,
            pos=d.odom_pos.copy() if odom else None,
            vel=rot_z(d.yaw_err) @ (d.scale * d.true_vel()) + d.rng.normal(0, 0.03, 3) if odom else None,
            yaw=d.odom_yaw,
            roll=roll + d.rng.normal(0, ATTITUDE_EST_STD), pitch=pitch + d.rng.normal(0, ATTITUDE_EST_STD),
            imu=np.array(d.imu_buf).reshape(-1, 7),
            lidar=self.lidar_dirs[hit] * ranges[hit, None] + self.lidar.mount, lidar_dirs=self.lidar_dirs, lidar_ranges=ranges,
            lidar_mount=self.lidar.mount,
            range_down=self.rangefinder.measure(self.pb, pos, d.R, d.rng),
            baro_alt=d.baro.measure(pos[2], 1.0 / self.cfg.agent_hz, d.rng),
            battery=d.battery, detections=self.detect(d),
            inbox=d.inbox, neighbors=self.neighbors(d.id), base_link=d.id in self.base_links,
            report_results=d.report_results, landed=d.status == "landed",
            lidar_t=self.lidar.ray_times(self.t))
        obs.gps = self.gps_fix(d, pos)
        d.inbox, d.report_results, d.imu_buf, d.pose_buf = [], [], [], []
        d.last_obs = obs
        d.last_truth = {"pos": np.round(pos, 3).tolist(), "yaw": round(float(np.arctan2(d.R[1, 0], d.R[0, 0])), 4),
                        "rp": [round(roll, 4), round(pitch, 4)], "z": round(float(pos[2]), 3)}
        return obs

    def gps_fix(self, d: Drone, pos):
        """GNSS under open sky: no rock straight up. The odom frame starts aligned with the world
        at the base, so the fix is given in it. Error: Gauss-Markov, 30 s correlation time."""
        if self.pb.rayTest(pos.tolist(), (pos + [0, 0, 200.0]).tolist(), collisionFilterMask=CAVE_GROUP)[0][0] in self.caves:
            return None
        phi = np.exp(-1.0 / self.cfg.agent_hz / 30.0)
        d.gps_err = phi * d.gps_err + np.sqrt(1 - phi ** 2) * d.rng.normal(0, [1.0, 1.0, 2.0])
        return pos + d.gps_err

    def step(self):
        """Advance one agent tick (1 / agent_hz seconds). Returns False when the mission is over."""
        cfg = self.cfg
        self.update_links()
        outgoing = []
        for d, agent in zip(self.drones, self.agents):
            if not d.active:
                continue
            obs = self.observe(d)
            try:
                cmd = agent.step(obs)
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

        dt = cfg.dt
        g = np.array([0.0, 0.0, C.GRAVITY])
        for k in range(cfg.steps_per_tick):
            vels = [d.true_vel() for d in self.drones]
            prev_R = [d.R for d in self.drones]
            self.aero_disturbance(dt)
            for d in self.drones:
                self.apply_thrust(d, dt)
            self.pb.stepSimulation()
            for d, v, R0 in zip(self.drones, vels, prev_R):
                self.check_contacts(d, v)
                if d.active:
                    accel = (d.true_vel() - v) / dt
                    d.imu_buf.append(d.imu.sample(self.t + (k + 1) * dt, d.R.T @ (accel + g),
                                                  angular_velocity(R0, d.R, dt)))
                    d.pose_buf.append((d.true_pos(), d.R))
        tick_dt = 1.0 / cfg.agent_hz
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
        self.check_course(tick_dt)
        if self.mission["type"] == "rescue":
            self.update_rescue()
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
        if self.mission["type"] == "rescue":
            best = min(self.laps.values()) if self.laps else None
            return {
                "mission": "rescue", "difficulty": self.mission.get("difficulty"),
                "score": round(best, 1) if best is not None else None,     # lap time, lower is better
                "lap_time_s": round(best, 1) if best is not None else None,
                "success": best is not None,
                "rescued_at_s": {i: round(t, 1) for i, t in self.rescued.items()},
                "laps_s": {i: round(t, 1) for i, t in self.laps.items()},
                "artifacts_found": len(self.found), "artifacts_total": len(self.artifacts),
                "returned": returned,
                "crashed": [d.id for d in self.drones if d.status == "crashed"],
                "crash_cause": {d.id: d.crash_cause for d in self.drones if d.crash_cause},
                "collisions": [len(d.hits) for d in self.drones],
                "worst_hit_ms": [round(max(d.hits, default=0.0), 2) for d in self.drones],
                "out_of_course": [d.id for d in self.drones if d.status == "out"],
                "battery_dead": [d.id for d in self.drones if d.status == "dead"],
                "coverage": round(coverage, 3),
                "final_drift_m": [round(x, 2) for x in drift],
                "distance_m": [round(d.distance, 1) for d in self.drones],
                "time_s": round(self.t, 1),
            }
        return {
            "score": round(total, 2),
            "artifacts_found": len(self.found),
            "artifacts_total": len(self.artifacts),
            "returned": returned,
            "crashed": [d.id for d in self.drones if d.status == "crashed"],
            "out_of_course": [d.id for d in self.drones if d.status == "out"],
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
                "rp": [round(x, 3) for x in roll_pitch(d.R)],
                "odom": np.round(d.odom_pos, 2).tolist() if d.odom_on else None,
                "odom_yaw": round(d.odom_yaw, 3), "status": d.status, "battery": round(d.battery, 3),
                "vel": np.round(d.true_vel(), 2).tolist(),
                "clear": round(d.clearance, 2) if np.isfinite(d.clearance) else None,
                "aero": np.round(d.aero_acc, 2).tolist(),
                "mix": [round(float(x), 3) for x in d.mix] if d.active else [0, 0, 0, 0],
            }
            if lidar and d.active and d.last_scan is not None:
                r, R, pos = d.last_scan
                hit = np.isfinite(r)
                pts = (self.lidar_dirs[hit] * r[hit, None] + self.lidar.mount) @ R.T + pos
                step = max(1, len(pts) // 3000)
                entry["lidar"] = np.round(pts[::step], 2).reshape(-1).tolist()
            try:
                dbg = agent.debug() or {}
                entry["debug"] = json.loads(json.dumps(dbg, default=lambda o: np.asarray(o).tolist()))
            except Exception:
                entry["debug"] = {}
            drones.append(entry)
        return {"type": "state", "t": round(self.t, 2), "drones": drones, "links": self.links,
                "base_links": self.base_links, "found": sorted(self.found), "score": self.score()}

    def sensor_packet(self, i):
        """Everything one drone sensed on its latest tick, for the sensor windows.
        Includes ground truth so the instructor can judge estimation error."""
        d = self.drones[i]
        o = d.last_obs
        lp = self.cfg.lidar
        out = {"type": "sensors", "drone": i, "t": round(self.t, 2), "status": d.status,
               "battery": round(d.battery, 3),
               "truth": d.last_truth,       # taken at the same instant as the observation
               "lidar_meta": {"channels": lp.channels, "azimuth_steps": lp.azimuth_steps,
                              "vfov_deg": list(lp.vfov_deg), "max_range": lp.max_range}}
        if o is None:
            return out
        r = o.lidar_ranges
        out.update({
            "obs_t": round(o.t, 2),
            "lidar_cm": np.where(np.isfinite(r), np.round(r * 100), 0).astype(int).tolist(),
            "imu": np.round(o.imu, 4).tolist(),
            "range_down": None if not np.isfinite(o.range_down) else round(o.range_down, 3),
            "baro_alt": round(o.baro_alt, 3),
            "roll": round(o.roll, 4), "pitch": round(o.pitch, 4), "yaw": round(o.yaw, 4),
            "odom": None if o.pos is None else np.round(o.pos, 3).tolist(),
            "detections": [{"type": det.type, "rel": np.round(det.rel_pos, 2).tolist()} for det in o.detections],
            "neighbors": o.neighbors, "base_link": o.base_link,
        })
        return out

    def viewer_init(self):
        return {"type": "init", "map": self.map_dir.name, "base": self.base.tolist(),
                "base_station": self.meta.get("base_station"),
                "artifacts": self.artifacts, "n_drones": len(self.drones),
                "home_radius": self.cfg.home_radius, "comms_range": self.cfg.comms.max_range,
                "mission_time_s": self.cfg.mission_time_s, "mission": self.mission,
                "lidar_mount": self.lidar.mount.tolist()}

    def trails(self):
        return {"type": "trails", "true": [d.trail for d in self.drones],
                "odom": [d.odom_trail if d.odom_on else [] for d in self.drones]}

    def close(self):
        if self.pb is not None:
            self.pb.disconnect()
            self.pb = None


def run_mission(map_dir, agent_factory, n_drones=3, odom="default", lidar=None, seed=0, cfg=None,
                verbose=False, aero=None, imu=None):
    """Headless run for experiments: returns the score dict."""
    sim = Simulator(map_dir, n_drones, agent_factory, cfg=cfg, odom=odom, lidar=lidar, seed=seed, aero=aero, imu=imu)
    try:
        while sim.step():
            if verbose and sim.tick % (10 * sim.cfg.agent_hz) == 0:
                print(f"t={sim.t:5.0f}s  {sim.score()}")
        return sim.score()
    finally:
        sim.close()
