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
    attitude_tau: float = 0.08       # s, how fast the airframe tilts to the commanded thrust direction
    linear_drag: float = 0.15        # N per (m/s)
    hover_endurance_s: float = 480.0 # battery 100% -> 0% while hovering
    crash_speed: float = 1.0         # impact speed (m/s) that destroys the drone


@dataclass
class LidarParams:
    """Spinning 3D LiDAR, body-fixed (it tilts with the drone).

    Default follows the Ouster OS0-32 (Rev D datasheet), the sensor flown on the
    DARPA SubT CERBERUS aerial scouts: 90 deg vertical FOV, 512 columns, 10 Hz,
    min range 0.3 m, precision 2 cm (0.3-1 m), 1 cm (1-10 m), 1.5 cm (10-15 m), 5 cm (>15 m).
    """
    channels: int = 32
    azimuth_steps: int = 512
    vfov_deg: tuple = (-45.0, 45.0)
    min_range: float = 0.3
    max_range: float = 35.0
    full_detect_range: float = 20.0   # beyond this the return probability falls linearly...
    min_detect_prob: float = 0.3      # ...to this value at max_range (dark wet rock)
    grazing_cos: float = 0.15         # returns fade out below this |cos(incidence)|
    dropout: float = 0.005            # random missing returns
    precision: tuple = ((1.0, 0.02), (10.0, 0.01), (15.0, 0.015), (1e9, 0.05))  # (upper range, 1-sigma)
    rate_hz: float = 10.0


LIDAR_PRESETS = {
    "os0-32": LidarParams(),
    # For slow PCs: same optics, a quarter of the beams.
    "lite": LidarParams(channels=16, azimuth_steps=256),
    # Old idealised sensor (no dropouts, flat noise), handy for debugging.
    "ideal": LidarParams(channels=16, azimuth_steps=90, vfov_deg=(-40.0, 40.0), min_range=0.0,
                         max_range=15.0, full_detect_range=15.0, min_detect_prob=1.0,
                         grazing_cos=0.0, dropout=0.0, precision=((1e9, 0.0),)),
}


@dataclass
class ImuParams:
    """6-axis IMU. Defaults are the RotorS/Gazebo values for an ADIS16448."""
    rate_hz: int = 240                     # one sample per physics step
    gyro_noise_density: float = 3.39e-4    # rad/s/sqrt(Hz)
    gyro_random_walk: float = 3.88e-5      # rad/s^2/sqrt(Hz)
    gyro_bias_corr_time: float = 1000.0    # s
    gyro_turn_on_bias: float = 8.73e-3     # rad/s, 1-sigma
    accel_noise_density: float = 4.0e-3    # m/s^2/sqrt(Hz)
    accel_random_walk: float = 6.0e-3      # m/s^3/sqrt(Hz)
    accel_bias_corr_time: float = 300.0    # s
    accel_turn_on_bias: float = 0.196      # m/s^2, 1-sigma


@dataclass
class RangefinderParams:
    """Downward single-point LiDAR, body-fixed. Benewake TFmini Plus datasheet."""
    min_range: float = 0.1
    max_range: float = 12.0
    sigma_near: float = 0.03       # 1-sigma repeatability up to 5 m
    rel_sigma_far: float = 0.006   # ~1% (as +-2 sigma) beyond 5 m
    rate_hz: float = 100.0


@dataclass
class BaroParams:
    """Barometric altitude relative to take-off. BMP388-class sensor."""
    noise_std: float = 0.10        # m, sample noise
    drift_rw: float = 0.02         # m/sqrt(s), weather/temperature drift
    rate_hz: float = 50.0


@dataclass
class OdomParams:
    """Stand-in for an onboard LiDAR-inertial odometry (LIO) output.

    Drift grows with distance flown and with time. Preset "off" withholds it so a
    team can run its own odometry on the raw IMU + LiDAR."""
    pos_noise_per_m: float = 0.01     # std of position noise per metre flown
    scale_error_std: float = 0.01     # per-drone fixed scale error
    yaw_rw_std: float = 0.002         # rad / sqrt(s), yaw random walk
    yaw_noise_per_rad: float = 0.01   # extra yaw error per rad rotated


ODOM_PRESETS = {
    "perfect": OdomParams(0.0, 0.0, 0.0, 0.0),
    "default": OdomParams(),
    "hard": OdomParams(0.03, 0.03, 0.006, 0.03),
    "off": None,
}


@dataclass
class AeroParams:
    """Air the onboard autopilot has to fight, as an external acceleration.

    Rotor wash bouncing off nearby rock turns into turbulence, a close ceiling pulls
    the drone up into it (ceiling effect), a close floor gives extra lift (ground
    effect), and caves carry a weak gusty draft everywhere. Gusts are a first-order
    Gauss-Markov process; their strength grows as the clearance to rock shrinks."""
    draft_std: float = 0.25        # m/s^2, gust strength in open air
    gust_tau: float = 0.5          # s, gust correlation time
    wall_turb_std: float = 1.5     # m/s^2, extra gust strength right at a wall
    wall_range: float = 1.2        # m of clearance below which the wall turbulence starts
    ceiling_gain: float = 0.35     # x g upward pull at zero ceiling clearance
    ceiling_range: float = 0.8     # m
    ground_gain: float = 0.15      # x g extra lift at zero floor clearance
    ground_range: float = 0.6      # m


AERO_PRESETS = {
    "off": AeroParams(0.0, 0.5, 0.0, 1.2, 0.0, 0.8, 0.0, 0.6),
    "default": AeroParams(),
    "hard": AeroParams(0.5, 0.5, 3.0, 1.6, 0.5, 1.0, 0.2, 0.8),
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
    imu: ImuParams = field(default_factory=ImuParams)
    rangefinder: RangefinderParams = field(default_factory=RangefinderParams)
    baro: BaroParams = field(default_factory=BaroParams)
    odom: OdomParams = field(default_factory=OdomParams)
    aero: AeroParams = field(default_factory=AeroParams)
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
