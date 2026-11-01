"""Sensor models. Parameters and their sources live in config.py.

Every sensor is body-fixed: when the drone tilts to accelerate, the LiDAR, IMU and
rangefinder tilt with it. Frames: body x forward, y left, z up; R maps body -> world.
"""
import numpy as np

from . import config as C

CAVE_GROUP = 1


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
        bounds, sig = zip(*lp.precision)
        self.prec_bounds, self.prec_sigma = np.array(bounds), np.array(sig)

    def scan(self, pb, pos, R, rng):
        """Ranges (channels*azimuth_steps,) in metres, np.inf where there is no return."""
        lp = self.p
        dirs_w = self.dirs @ R.T
        start = (pos + dirs_w * self.start).tolist()
        end = (pos + dirs_w * lp.max_range).tolist()
        hits = []
        for k in range(0, len(start), 16000):
            hits += pb.rayTestBatch(start[k:k + 16000], end[k:k + 16000], numThreads=1)
        obj = np.fromiter((h[0] for h in hits), int, len(hits))
        frac = np.fromiter((h[2] for h in hits), float, len(hits))
        hit = obj >= 0
        r = np.full(len(hits), np.inf)
        r[hit] = self.start + frac[hit] * (lp.max_range - self.start)

        # Return probability: range falloff, grazing incidence, random dropouts.
        idx = np.flatnonzero(hit)
        prob = np.ones(len(idx))
        far = np.clip((r[idx] - lp.full_detect_range) / (lp.max_range - lp.full_detect_range), 0, 1)
        prob *= 1 - far * (1 - lp.min_detect_prob)
        if lp.grazing_cos > 0:
            normals = np.array([hits[i][4] for i in idx]).reshape(-1, 3)
            cos_inc = np.abs(np.einsum("ij,ij->i", normals, dirs_w[idx]))
            prob *= np.clip(cos_inc / lp.grazing_cos, 0, 1)
        prob *= 1 - lp.dropout
        lost = idx[rng.random(len(idx)) >= prob]
        r[lost] = np.inf

        ok = np.isfinite(r)
        sigma = self.prec_sigma[np.searchsorted(self.prec_bounds, r[ok])]
        r[ok] += rng.normal(0, 1, ok.sum()) * sigma
        return r


class Imu:
    """White noise + Gauss-Markov bias + constant turn-on bias (RotorS model)."""

    def __init__(self, ip: C.ImuParams, rng):
        self.p = ip
        self.rng = rng
        self.dt = 1.0 / ip.rate_hz
        self.gyro_on = rng.normal(0, ip.gyro_turn_on_bias, 3)
        self.accel_on = rng.normal(0, ip.accel_turn_on_bias, 3)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def _bias_step(self, bias, rw, tau):
        phi = np.exp(-self.dt / tau)
        sigma = np.sqrt(-rw ** 2 * tau / 2 * (np.exp(-2 * self.dt / tau) - 1))
        return phi * bias + sigma * self.rng.normal(0, 1, 3)

    def sample(self, t, specific_force, omega):
        ip = self.p
        self.gyro_bias = self._bias_step(self.gyro_bias, ip.gyro_random_walk, ip.gyro_bias_corr_time)
        self.accel_bias = self._bias_step(self.accel_bias, ip.accel_random_walk, ip.accel_bias_corr_time)
        acc = specific_force + self.accel_on + self.accel_bias + \
            ip.accel_noise_density / np.sqrt(self.dt) * self.rng.normal(0, 1, 3)
        gyr = omega + self.gyro_on + self.gyro_bias + \
            ip.gyro_noise_density / np.sqrt(self.dt) * self.rng.normal(0, 1, 3)
        return np.concatenate([[t], acc, gyr])


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
