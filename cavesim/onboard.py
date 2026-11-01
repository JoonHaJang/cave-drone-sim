"""The drone's onboard flight stack, kept apart from the world it flies in.

A real airframe splits three ways: the world (physics and raw sensors, `sim.py`), the onboard
stack (state estimation and the flight controller, here), and the autonomy that sets goals for it
(`agents/`). The flight controller is a PX4-style cascade, all at the physics rate:

    velocity setpoint --PI--> acceleration --(+g, drag)--> thrust vector
        --> attitude setpoint (thrust direction + heading) --P (geometric error, Lee et al. 2010)-->
        body-rate setpoint --PI--> torque --> mixer (quad X) --> four motor thrusts

The world turns the motor commands into thrust through a first-order motor lag and integrates the
rigid-body rotation (sim.apply_thrust). The controller closes its loops on the velocity, attitude
and body rates it is given; today those are still the true ones -- an onboard estimator replaces
them next.
"""
import numpy as np

from . import config as C
from .sensors import attitude_from_thrust


def mixer(dp: C.DroneParams):
    """(4, 4) matrix taking motor thrusts to [collective thrust, roll, pitch, yaw torque] (body).
    Quad X: motors at 45, 135, 225, 315 deg, spinning alternately (+, -, +, -)."""
    th = np.radians([45.0, 135.0, 225.0, 315.0])
    spin = np.array([1.0, -1.0, 1.0, -1.0])
    return np.vstack([np.ones(4), dp.arm * np.sin(th), -dp.arm * np.cos(th), dp.yaw_moment * spin])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class VelocityController:
    """Velocity setpoint (m/s, world frame) -> acceleration setpoint: PI, limited to the
    airframe's speeds, its tilt (horizontal) and +-4 m/s^2 (vertical). The integral takes out
    steady pushes the P term alone would leave as an error (ceiling suction, drafts)."""
    KP = 3.0              # 1/s: a velocity error closes in about a third of a second
    KI = 1.0              # 1/s^2
    I_MAX = 2.0           # m/s^2 the integral may contribute
    I_BAND = 0.3          # m/s: integrate only errors smaller than this
    A_Z = 4.0             # m/s^2 vertical acceleration limit

    def __init__(self, dp: C.DroneParams):
        self.dp = dp
        self.a_xy = C.GRAVITY * np.tan(np.radians(dp.max_tilt_deg))
        self.i = np.zeros(3)

    def limit(self, v_cmd):
        """The setpoint clipped to the airframe's speed limits."""
        v_cmd = np.array(v_cmd, dtype=float)
        sxy = np.linalg.norm(v_cmd[:2])
        if sxy > self.dp.max_speed_xy:
            v_cmd[:2] *= self.dp.max_speed_xy / sxy
        v_cmd[2] = np.clip(v_cmd[2], -self.dp.max_speed_z, self.dp.max_speed_z)
        return v_cmd

    def acceleration(self, v_cmd, v, dt):
        e = v_cmd - v
        a = self.KP * e + self.i
        axy = np.linalg.norm(a[:2])
        sat = axy > self.a_xy or abs(a[2]) > self.A_Z
        if axy > self.a_xy:
            a[:2] *= self.a_xy / axy
        a[2] = np.clip(a[2], -self.A_Z, self.A_Z)
        if not sat:                       # anti-windup: integrate only while not saturated, and only
            near = np.abs(e) < self.I_BAND   # the small error P leaves; a step is P's to close
            self.i = np.clip(self.i + self.KI * e * near * dt, -self.I_MAX, self.I_MAX)
        return a


class FlightController:
    """The cascade above. update() returns the four motor thrust commands (N)."""
    K_ATT = 8.0           # 1/s, attitude error -> body rate
    K_RATE = 25.0         # 1/s, body-rate error -> angular acceleration
    KI_RATE = 10.0        # 1/s^2
    RATE_MAX = 6.0        # rad/s roll / pitch rate setpoint limit

    def __init__(self, dp: C.DroneParams):
        self.dp = dp
        self.vel = VelocityController(dp)
        self.J = np.diag(dp.inertia)
        self.unmix = np.linalg.inv(mixer(dp))
        self.f_max = dp.thrust_to_weight * dp.mass * C.GRAVITY / 4
        self.i_rate = np.zeros(3)
        self.yaw_sp = None
        self.demand = np.zeros(4)         # [thrust / weight, roll, pitch, yaw] torque demand, for the viewer

    def limit(self, v_cmd):
        return self.vel.limit(v_cmd)

    def desaturate(self, f, f_tau):
        """Motor commands within [0, f_max], keeping the attitude torque first (PX4's mixer does
        the same): shift the collective thrust to fit the torque spread, and only if the spread
        itself does not fit, scale the torque down."""
        spread = f_tau.max() - f_tau.min()
        if spread > self.f_max:
            f_tau = f_tau * (self.f_max / spread)
            f = f_tau + np.mean(f - f_tau)
        shift = max(0.0, -f.min()) - max(0.0, f.max() - self.f_max)
        return np.clip(f + shift, 0.0, self.f_max)

    def update(self, v_sp, yaw_rate_sp, v, R, omega, dt):
        dp = self.dp
        # Velocity -> thrust vector. The drag the airframe will meet at the setpoint is fed forward.
        a = self.vel.acceleration(v_sp, v, dt)
        F = dp.mass * (a + [0.0, 0.0, C.GRAVITY]) + dp.linear_drag * v_sp
        # Attitude setpoint: thrust along F, nose at the integrated heading (kept near the actual one).
        yaw = np.arctan2(R[1, 0], R[0, 0])
        yaw_rate_sp = float(np.clip(yaw_rate_sp, -dp.max_yaw_rate, dp.max_yaw_rate))
        if self.yaw_sp is None:
            self.yaw_sp = yaw
        self.yaw_sp = yaw + np.clip(np.angle(np.exp(1j * (self.yaw_sp + yaw_rate_sp * dt - yaw))), -0.5, 0.5)
        R_sp = attitude_from_thrust(F, self.yaw_sp)
        e_R = 0.5 * vee(R_sp.T @ R - R.T @ R_sp)
        w_sp = -self.K_ATT * e_R + R.T @ [0.0, 0.0, yaw_rate_sp]
        w_sp[:2] = np.clip(w_sp[:2], -self.RATE_MAX, self.RATE_MAX)
        w_sp[2] = np.clip(w_sp[2], -dp.max_yaw_rate, dp.max_yaw_rate)
        # Body rates -> torque (rigid-body feed-forward of the gyroscopic term).
        e_w = w_sp - omega
        self.i_rate = np.clip(self.i_rate + e_w * dt, -0.5, 0.5)
        tau = self.J @ (self.K_RATE * e_w + self.KI_RATE * self.i_rate) + np.cross(omega, self.J @ omega)
        T = float(F @ R[:, 2])            # collective thrust along the airframe's own z
        f = self.desaturate(self.unmix @ np.r_[T, tau], self.unmix @ np.r_[0.0, tau])
        self.demand = np.r_[T / (dp.mass * C.GRAVITY),
                            np.clip(tau / np.array([2 * self.f_max * dp.arm, 2 * self.f_max * dp.arm,
                                                    2 * self.f_max * dp.yaw_moment]), -1, 1)]
        return f
