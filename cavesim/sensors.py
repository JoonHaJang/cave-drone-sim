"""Sensor models. Parameters and their sources live in config.py.

Every sensor is body-fixed: when the drone tilts to accelerate, the LiDAR, IMU and
rangefinder tilt with it. Frames: body x forward, y left, z up; R maps body -> world.
"""
import numpy as np

from . import config as C

CAVE_GROUP, DRONE_GROUP = 1, 2
# Threads for a LiDAR sweep's ray batch. Keep 1: on Windows any value above 1 (or 0 = all cores)
# made PyBullet hang as soon as two simulations ran at the same time.
RAY_THREADS = 1   # sensors see rock and drones; course fences (group 4) stay invisible


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def attitude_from_thrust(thrust_dir, yaw):
    """Rotation whose z axis is the thrust direction and whose heading is yaw."""
    zb = thrust_dir / np.linalg.norm(thrust_dir)
    xc = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])


def roll_pitch(R):
    """ZYX Euler roll and pitch of a body->world rotation."""
    return float(np.arctan2(R[2, 1], R[2, 2])), float(-np.arcsin(np.clip(R[2, 0], -1, 1)))


def angular_velocity(R_prev, R, dt):
    """Body-frame angular rate that takes R_prev to R in dt (small-angle)."""
    dR = R_prev.T @ R
    return np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * dt)


class Lidar:
    def __init__(self, lp: C.LidarParams, start_offset):
        self.p = lp
        el = np.radians(np.linspace(*lp.vfov_deg, lp.channels))
        az = np.linspace(-np.pi, np.pi, lp.azimuth_steps, endpoint=False)
        E, A = np.meshgrid(el, az, indexing="ij")
        self.elevation, self.azimuth = el, az
        self.dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.start = max(lp.min_range, start_offset)
        self.mount = np.array([0.0, 0.0, lp.mount_z])           # sensor origin in the body frame
        # Beams through the rotor disc (body frame, sensor on the disc's axis, above or below it)
        # hit the frame: dropped.
        rise = lp.prop_z - lp.mount_z
        toward = self.dirs[:, 2] * np.sign(rise)
        with np.errstate(divide="ignore", invalid="ignore"):
            reach = np.where(toward > 0, abs(rise) / toward * np.hypot(self.dirs[:, 0], self.dirs[:, 1]), np.inf)
        self.blocked = (reach <= lp.prop_r) if (lp.self_occlusion and rise != 0) else np.zeros(len(self.dirs), bool)
        # Firing order of each ray's column. The real OS0 spins clockwise seen from above,
        # starting behind the drone (azimuth -180 deg) with azimuth decreasing (CTU SubT data).
        n = lp.azimuth_steps
        self.fire = (n - np.arange(len(self.dirs)) % n) % n
        bounds, sig = zip(*lp.precision)
        self.prec_bounds, self.prec_sigma = np.array(bounds), np.array(sig)

    def scan(self, pb, poses, rng, floor_gap=np.inf):
        """One sweep. poses: [(pos, R), ...] in time order over the sweep, last = now.

        Returns ranges (channels*azimuth_steps,) in metres, np.inf where there is no
        return, and the world-frame surface normals of the rock returns."""
        lp = self.p
        if not lp.motion_distortion:
            poses = poses[-1:]
        # The f-th column to fire does so at fraction (f+1)/azimuth_steps of the sweep,
        # from the pose then. ponytail: nearest physics step (1/240 s), no interpolation
        k = self.fire * len(poses) // lp.azimuth_steps
        Rs = np.array([q[1] for q in poses])[k]
        P = np.array([q[0] for q in poses])[k] + Rs @ self.mount       # the sensor, not the body centre
        dirs_w = np.einsum("nij,nj->ni", Rs, self.dirs)
        start = (P + dirs_w * self.start).tolist()
        end = (P + dirs_w * lp.max_range).tolist()
        hits = []
        for i in range(0, len(start), 16000):
            hits += pb.rayTestBatch(start[i:i + 16000], end[i:i + 16000], numThreads=RAY_THREADS,
                                    collisionFilterMask=CAVE_GROUP | DRONE_GROUP)
        obj = np.fromiter((h[0] for h in hits), int, len(hits))
        frac = np.fromiter((h[2] for h in hits), float, len(hits))
        hit = (obj >= 0) & ~self.blocked
        r = np.full(len(hits), np.inf)
        r[hit] = self.start + frac[hit] * (lp.max_range - self.start)

        # Return probability: range falloff, grazing incidence, random dropouts.
        idx = np.flatnonzero(hit)
        normals = np.array([hits[i][4] for i in idx]).reshape(-1, 3)
        prob = np.ones(len(idx))
        far = np.clip((r[idx] - lp.full_detect_range) / max(lp.max_range - lp.full_detect_range, 1e-9), 0, 1)
        prob *= 1 - far * (1 - lp.min_detect_prob)
        if lp.grazing_cos > 0:
            cos_inc = np.abs(np.einsum("ij,ij->i", normals, dirs_w[idx]))
            prob *= np.clip(cos_inc / lp.grazing_cos, 0, 1)
        prob *= 1 - lp.dropout
        kept = rng.random(len(idx)) < prob
        r[idx[~kept]] = np.inf
        normals = normals[kept]

        # Rotor dust: the first particle along the beam, if nearer than the rock, answers.
        density = lp.dust_density * np.clip(1 - floor_gap / lp.dust_height, 0, 1) if lp.dust_height > 0 else 0.0
        if density > 0:
            d = self.start + rng.exponential(1 / density, len(r))
            dust = d < np.minimum(r, lp.dust_radius)
            r[dust] = d[dust]
            normals = normals[~dust[idx[kept]]]
        ok = np.isfinite(r)
        sigma = self.prec_sigma[np.searchsorted(self.prec_bounds, r[ok])]
        r[ok] += rng.normal(0, 1, ok.sum()) * sigma
        return r, normals

    def ray_times(self, t_end):
        """Firing time of every ray for a sweep that ended at t_end."""
        lp = self.p
        if not lp.motion_distortion:
            return np.full(len(self.dirs), t_end)
        sweep = 1.0 / lp.rate_hz
        return t_end - sweep + (self.fire + 1) * sweep / lp.azimuth_steps


def degeneracy(normals, lam_min):
    """Zhang et al. 2016: eigen-directions of the scan's mean n n^T and how degenerate
    each is (0 = well constrained, 1 = no wall faces that way)."""
    if len(normals) < 10:
        return np.eye(3), np.ones(3)
    lam, V = np.linalg.eigh(normals.T @ normals / len(normals))
    return V, np.clip(1 - lam / lam_min, 0, 1)


class Imu:
    """White noise + Gauss-Markov bias + constant turn-on bias (RotorS model), plus
    scale/misalignment, rotor vibration, saturation and quantization."""

    def __init__(self, ip: C.ImuParams, rng):
        self.p = ip
        self.rng = rng
        self.dt = 1.0 / ip.rate_hz
        self.gyro_on = rng.normal(0, ip.gyro_turn_on_bias, 3)
        self.accel_on = rng.normal(0, ip.accel_turn_on_bias, 3)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        # Sensed = M @ true: diagonal scale error, off-diagonal misalignment.
        self.M_acc, self.M_gyr = (np.eye(3) + np.diag(rng.normal(0, ip.scale_std, 3))
                                  + (1 - np.eye(3)) * rng.normal(0, ip.misalign_std, (3, 3)) for _ in range(2))
        self.vib_phase = 0.0
        self.vib_offset = rng.uniform(0, 2 * np.pi, (2, 3))   # per-axis phase, accel and gyro

    def _bias_step(self, bias, rw, tau):
        phi = np.exp(-self.dt / tau)
        sigma = np.sqrt(-rw ** 2 * tau / 2 * (np.exp(-2 * self.dt / tau) - 1))
        return phi * bias + sigma * self.rng.normal(0, 1, 3)

    @staticmethod
    def _digitize(x, lim, lsb):
        x = np.clip(x, -lim, lim)
        return np.round(x / lsb) * lsb if lsb > 0 else x

    def sample(self, t, specific_force, omega):
        ip = self.p
        self.gyro_bias = self._bias_step(self.gyro_bias, ip.gyro_random_walk, ip.gyro_bias_corr_time)
        self.accel_bias = self._bias_step(self.accel_bias, ip.accel_random_walk, ip.accel_bias_corr_time)
        # Blade-pass frequency follows rotor speed ~ sqrt(thrust); thrust/mass ~ |specific force|.
        f = ip.vib_freq_hover * np.sqrt(np.linalg.norm(specific_force) / C.GRAVITY)
        self.vib_phase = (self.vib_phase + 2 * np.pi * f * self.dt) % (2 * np.pi)
        vib = np.sin(self.vib_phase + self.vib_offset)
        acc = self.M_acc @ specific_force + self.accel_on + self.accel_bias +             ip.accel_noise_density / np.sqrt(self.dt) * self.rng.normal(0, 1, 3) +             np.multiply(ip.vib_accel_amp, vib[0]) + np.multiply(ip.vib_accel_std, self.rng.normal(0, 1, 3))
        gyr = self.M_gyr @ omega + self.gyro_on + self.gyro_bias +             ip.gyro_noise_density / np.sqrt(self.dt) * self.rng.normal(0, 1, 3) +             np.multiply(ip.vib_gyro_amp, vib[1]) + np.multiply(ip.vib_gyro_std, self.rng.normal(0, 1, 3))
        return np.concatenate([[t], self._digitize(acc, ip.accel_range, ip.accel_lsb),
                               self._digitize(gyr, ip.gyro_range, ip.gyro_lsb)])


class Rangefinder:
    def __init__(self, rp: C.RangefinderParams):
        self.p = rp

    def measure(self, pb, pos, R, rng):
        rp = self.p
        down = -R[:, 2]
        hit = pb.rayTest(pos.tolist(), (pos + down * rp.max_range).tolist(), collisionFilterMask=CAVE_GROUP)[0]
        if hit[0] < 0:
            return np.inf
        r = hit[2] * rp.max_range
        if r < rp.min_range:
            return np.inf
        sigma = rp.sigma_near if r <= 5.0 else rp.rel_sigma_far * r
        return float(r + rng.normal(0, sigma))


class Barometer:
    def __init__(self, bp: C.BaroParams, z0):
        self.p = bp
        self.z0 = z0
        self.drift = 0.0

    def measure(self, z, dt, rng):
        self.drift += rng.normal(0, self.p.drift_rw * np.sqrt(dt))
        return float(z - self.z0 + self.drift + rng.normal(0, self.p.noise_std))
