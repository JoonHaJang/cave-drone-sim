"""Starting point for your team's agent. Copy to agents/<team>.py and fill in the TODOs.

    python -m cavesim.run --agent agents.<team>

Everything you receive and may send is defined in cavesim/api.py.
"""
import numpy as np

from cavesim.api import Agent, Command


class TeamAgent(Agent):
    def __init__(self, drone_id, n_drones, info):
        super().__init__(drone_id, n_drones, info)
        # info holds limits and sensor specs, e.g. info["max_speed_xy"], info["lidar_max_range"]
        # TODO: your map (occupancy grid / voxels), frontier list, task assignment state

    def step(self, obs):
        # 1. Localisation: obs.pos / obs.yaw drift over time. Correct them if you can.
        # 2. Mapping: obs.lidar is (N, 3) in the body frame. Rotate by obs.yaw, add obs.pos.
        # 3. Cooperation: read obs.inbox, share map/frontier/assignment with Command.send.
        # 4. Decide: explore a frontier, relay, or go home before the battery runs out.
        # 5. Report artifacts in obs.detections once obs.base_link is True.
        return Command(vel=np.zeros(3), yaw_rate=0.0)

    def debug(self):
        # Shown in the viewer (odom frame): {"path": [...], "goal": [x, y, z], "points": [...], "mode": "..."}
        return {}


Agent = TeamAgent
