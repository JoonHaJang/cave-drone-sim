"""Fly it yourself, from the viewer's keyboard -- a human route to set beside the planner's.

    python -m cavesim.run --agent agents.manual --map rescue_easy --drones 1 --lidar lite --record runs/me_easy.json

Up / Down arrows: forward / back along the nose.   Left / Right arrows: turn.
W / S: climb / sink (without S, the drone keeps 1.2 m over the floor).   A / D: sideways.   Shift: twice as fast.   L: land (ends the lap at home).

The planner's stopping-distance governor and push-off still ride on top, and the sweeps are
re-localised and mapped as for the planner: the record is the route you chose, not your reflexes.
Every tick goes to LOG (one JSON line: time, position, heading, keys held, survivor in view), so the
choices -- where you slowed, looked around, turned -- can be read back next to the planner's.
"""
import json
from pathlib import Path

import numpy as np

from agents.planner import PlannerAgent
from cavesim.api import PILOT, Command

SPEED = 1.5           # m/s forward / sideways (Shift: twice)
CLIMB = 1.0           # m/s
TURN = 0.8            # rad/s
HOVER_AGL = 1.2       # m over the floor held unless S is down (lower, the rotors lift dust the LiDAR sees)
LOG = Path("runs/manual_log.jsonl")


class ManualAgent(PlannerAgent):
    def __init__(self, *args):
        super().__init__(*args)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        self.log = LOG.open("w")

    def mission_step(self, obs, pts):
        self.map.insert(obs.pos, pts)
        keys = PILOT["keys"]
        self.log.write(json.dumps({"t": round(obs.t, 1), "pos": np.round(obs.pos, 2).tolist(), "yaw": round(float(obs.yaw), 3),
                                   "keys": sorted(keys), "survivor": any(d.type == "survivor" for d in obs.detections)}) + "\n")
        self.log.flush()
        if "l" in keys:
            return Command(land=True)
        fwd = ("arrowup" in keys) - ("arrowdown" in keys)
        side = ("a" in keys) - ("d" in keys)
        c, s = np.cos(obs.yaw), np.sin(obs.yaw)
        v = np.array([c * fwd - s * side, s * fwd + c * side, 0.0]) * SPEED * (2 if "shift" in keys else 1)
        v[2] = CLIMB * (("w" in keys) - ("s" in keys))
        if "s" not in keys and np.isfinite(obs.range_down):   # keep out of the rotor dust unless told to go down
            v[2] = max(v[2], HOVER_AGL - obs.range_down)
        if np.linalg.norm(v) > 0:
            v = self.cap_speed(obs, v)
        turn = ("arrowleft" in keys) - ("arrowright" in keys)
        return Command(vel=self.guard(obs, v), yaw_rate=TURN * turn)


Agent = ManualAgent
