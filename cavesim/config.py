"""Simulation constants. Change these in one place; every module reads from here."""
from dataclasses import dataclass, field


@dataclass
class DroneParams:
    mass: float = 1.0                # kg
    radius: float = 0.2              # m, collision sphere
    max_speed_xy: float = 3.0        # m/s
    max_speed_z: float = 1.5         # m/s
    max_yaw_rate: float = 1.5        # rad/s
    max_tilt_deg: float = 25.0
    linear_drag: float = 0.15        # N per (m/s)
    hover_endurance_s: float = 480.0 # battery 100% -> 0% while hovering
    crash_speed: float = 1.0         # impact speed (m/s) that destroys the drone


@dataclass
class LidarParams:
    channels: int = 16               # vertical beams
    azimuth_steps: int = 90          # horizontal beams (4 deg spacing)
    vfov_deg: tuple = (-40.0, 40.0)
    max_range: float = 15.0
    range_noise_std: float = 0.02
    rate_hz: float = 10.0


@dataclass
class OdomParams:
    """GPS-denied odometry: drift grows with distance flown and with time."""
    pos_noise_per_m: float = 0.01     # std of position noise per metre flown
    scale_error_std: float = 0.01     # per-drone fixed scale error
    yaw_rw_std: float = 0.002         # rad / sqrt(s), yaw random walk
    yaw_noise_per_rad: float = 0.01   # extra yaw error per rad rotated


ODOM_PRESETS = {
    "perfect": OdomParams(0.0, 0.0, 0.0, 0.0),
    "default": OdomParams(),
    "hard": OdomParams(0.03, 0.03, 0.006, 0.03),
}


@dataclass
class CommsParams:
    max_range: float = 25.0          # m, needs line of sight as well
    base_range: float = 25.0
    max_msg_bytes: int = 200_000     # per message, after pickling


@dataclass
class DetectorParams:
    max_range: float = 6.0
    half_fov_deg: float = 60.0
    pos_noise_std: float = 0.2


@dataclass
class SimConfig:
    physics_hz: int = 240
    agent_hz: int = 10
    mission_time_s: float = 600.0
    home_radius: float = 3.0         # land within this distance of base = returned
    report_match_dist: float = 5.0   # artifact report counted if within this distance
    reports_per_artifact: int = 2    # report budget = this * number of artifacts
    drone: DroneParams = field(default_factory=DroneParams)
    lidar: LidarParams = field(default_factory=LidarParams)
    odom: OdomParams = field(default_factory=OdomParams)
    comms: CommsParams = field(default_factory=CommsParams)
    detector: DetectorParams = field(default_factory=DetectorParams)

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def steps_per_tick(self) -> int:
        return self.physics_hz // self.agent_hz


ARTIFACT_TYPES = ["survivor", "backpack", "phone", "helmet", "rope"]
GRAVITY = 9.81
