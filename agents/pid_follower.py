"""Step 2: fly a recorded route with a PID position loop. Copy to agents/<team>_pid.py and fill the TODOs.

    python -m agents.waypoints runs/manual_....jsonl            # writes runs/route.json
    python -m cavesim.run --agent agents.pid_follower --map rescue_easy --drones 1 --lidar lite

The route is a list of positions in the odometry frame (from agents.waypoints). The drone aims at
one waypoint at a time and moves on once within ACCEPT_R of it; after the last one it lands.

What is left to you is the controller: PID.update() turns the position error into a velocity
command. Until it is written the command is zero and the drone just hovers at the start.
Every tick goes to runs/pid_log.csv (time, waypoint, error, command) to plot and tune the gains.
"""
import csv
import json
from pathlib import Path

import numpy as np

from cavesim.api import Agent, Command

ROUTE = Path("runs/route.json")
LOG = Path("runs/pid_log.csv")
ACCEPT_R = 1.0        # m: this close to a waypoint, go on to the next
YAW_RATE_MAX = 0.8    # rad/s
KP, KI, KD = 0.0, 0.0, 0.0   # TODO: tune (start with KP alone)


class PID:
    """Three-axis PID on the position error; returns a velocity command, m/s."""

    def __init__(self, kp, ki, kd, i_limit=2.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit          # m*s: cap on the integral, so it cannot wind up while blocked
        self.reset()

    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = None

    def update(self, error, dt):
        """error: waypoint - position (3,), odom frame. dt: s since the last call."""
        # TODO 1: P term  -- kp * error
        # TODO 2: I term  -- add error * dt to self.integral, clip it to +-i_limit, ki * integral
        # TODO 3: D term  -- kd * (error - prev_error) / dt  (skip it on the first call)
        # Hint: obs.vel is measured velocity; -kd * obs.vel is a smoother D term than differencing error.
        self.prev_error = error
        return np.zeros(3)


class PIDFollower(Agent):
    def __init__(self, drone_id, n_drones, info):
        super().__init__(drone_id, n_drones, info)
        route = json.loads(ROUTE.read_text())
        self.wps = np.array(route["waypoints"], dtype=float)
        self.i = 0
        self.pid = PID(KP, KI, KD)
        self.vmax = np.array([info["max_speed_xy"], info["max_speed_xy"], info["max_speed_z"]])
        self.last_t = None
        LOG.parent.mkdir(parents=True, exist_ok=True)
        self.log = csv.writer(LOG.open("w", newline=""))
        self.log.writerow(["t", "wp", "ex", "ey", "ez", "vx", "vy", "vz"])

    def step(self, obs):
        if obs.landed:
            return Command()
        dt = 0.1 if self.last_t is None else obs.t - self.last_t
        self.last_t = obs.t

        error = self.wps[self.i] - obs.pos
        if np.linalg.norm(error) < ACCEPT_R:
            if self.i == len(self.wps) - 1:
                return Command(land=True)
            self.i += 1
            self.pid.reset()
            error = self.wps[self.i] - obs.pos

        v = np.clip(self.pid.update(error, dt), -self.vmax, self.vmax)
        self.log.writerow([round(obs.t, 1), self.i, *np.round(error, 3), *np.round(v, 3)])

        # Nose toward the waypoint, so the LiDAR and the viewer's FPV camera look where we go.
        want = np.arctan2(error[1], error[0])
        turn = (want - obs.yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = np.clip(1.5 * turn, -YAW_RATE_MAX, YAW_RATE_MAX) if np.hypot(error[0], error[1]) > 0.5 else 0.0
        return Command(vel=v, yaw_rate=yaw_rate)

    def debug(self):
        return {"path": self.wps[self.i:].tolist(), "goal": self.wps[self.i].tolist(), "mode": f"wp {self.i}/{len(self.wps)}"}


Agent = PIDFollower
