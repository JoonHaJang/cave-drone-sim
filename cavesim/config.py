"""Simulation constants. Change these in one place; every module reads from here."""
from dataclasses import dataclass, field


@dataclass
class DroneParams:
    mass: float = 1.0                # kg
    radius: float = 0.2              # m, collision disc: radius ...
    height: float = 0.18             # ... and thickness (the viewer's model, rotor tips 0.5 m apart)
    max_speed_xy: float = 3.0        # m/s
    max_speed_z: float = 1.5         # m/s
    max_yaw_rate: float = 1.5        # rad/s
    max_tilt_deg: float = 25.0
    # Airframe for the rotational dynamics (sim) and the onboard cascade (onboard.py): a 1 kg-class quad X.
    arm: float = 0.177               # m, motor to centre (rotor tips 0.25 m out)
    inertia: tuple = (0.010, 0.010, 0.018)  # kg m^2 about body x, y, z
    thrust_to_weight: float = 2.2    # all four motors flat out
    motor_tau: float = 0.03          # s, rotor thrust response
    yaw_moment: float = 0.016        # m: rotor drag torque per newton of thrust
    linear_drag: float = 0.15        # N per (m/s)
    hover_endurance_s: float = 480.0 # battery 100% -> 0% while hovering
    # Collisions (sim.check_contacts). The airframe is a thin disc of this radius (arms + guards);
    # an off-centre hit spins it (an angular impulse), and the attitude loop catches it.
    crash_speed: float = 2.0         # m/s into rock that breaks a guarded frame (2 J for 1 kg)
    upset_limit_deg: float = 75.0    # knocked past this tilt = flipped, out of control
    prop_guards: bool = True         # False: any side hit faster than prop_strike_speed breaks a rotor
    prop_strike_speed: float = 0.3   # m/s


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
    # Mounting: under the frame (mount_z below the body centre: the 0.18 m body, then the puck's
    # half height). Beams that cross the rotor disc (prop_z, tips at prop_r) hit the props and
    # arms; as real pipelines do (self-filter), those returns are dropped -- above
    # atan((prop_z - mount_z) / prop_r), about 33 deg here, the sensor sees nothing. The bottom
    # mount sees the floor down to -45 deg, which the cave benchmark needs (research_notes 9.3);
    # "lite-top" is the SubT scouts' top mount, blind below about -16 deg.
    mount_z: float = -0.13
    prop_z: float = 0.03
    prop_r: float = 0.25
    self_occlusion: bool = True
    min_range: float = 0.3
    max_range: float = 35.0
    # CTU SubT Finals (cave uav21_3 + mine uav22_1): returns stay ~97 % to about 10 m,
    # then sit at 0.86-0.91 from 12 to 35 m. A line from 8 m to 0.8 at 35 m fits every
    # range bin of both flights within 6 percentage points.
    full_detect_range: float = 8.0    # beyond this the return probability falls linearly...
    min_detect_prob: float = 0.8      # ...to this value at max_range
    # CTU SubT Finals cave data (OS0-128 on a UAV, docs/sensors.md): rough rock returns
    # 97 % even 84-90 deg off the normal, and 1.2 % of beams at 3-8 m come back empty.
    grazing_cos: float = 0.03         # returns fade out below this |cos(incidence)|
    dropout: float = 0.015            # random missing returns
    precision: tuple = ((1.0, 0.02), (10.0, 0.01), (15.0, 0.015), (1e9, 0.05))  # (upper range, 1-sigma)
    rate_hz: float = 10.0             # one sweep per agent tick (agent_hz must match)
    # The real sensor fires column by column over the whole sweep, so a moving drone
    # smears the scan (motion distortion). Per-ray timestamps come in obs.lidar_t.
    motion_distortion: bool = True
    # Rotor downwash lifts dust off the floor. Beer-Lambert: a beam meets a particle
    # after an exponential distance with rate = density; that particle returns instead.
    # CTU mine flight uav22_1: isolated returns in front of the mapped wall rise above the
    # no-dust baseline below 1 m of floor clearance, implying 0.013 /m at the floor (a lower
    # bound: dust inside 0.5 m and dense clumps escape the detector).
    dust_density: float = 0.015       # 1/m inside the cloud with the drone at the floor
    dust_height: float = 1.0          # m of floor clearance above which no dust is lifted
    dust_radius: float = 2.0          # m, cloud extent around the drone


LIDAR_PRESETS = {
    "os0-32": LidarParams(),
    # For slow PCs: same optics, a quarter of the beams.
    "lite": LidarParams(channels=16, azimuth_steps=256),
    "lite-top": LidarParams(channels=16, azimuth_steps=256, mount_z=0.10),
    # The idealised mount: at the body centre, nothing in the way (the full +-45 deg both ways).
    "lite-center": LidarParams(channels=16, azimuth_steps=256, mount_z=0.0, self_occlusion=False),
    # Old idealised sensor (no dropouts, flat noise), handy for debugging.
    "ideal": LidarParams(channels=16, azimuth_steps=90, vfov_deg=(-40.0, 40.0), min_range=0.0,
                         max_range=15.0, full_detect_range=15.0, min_detect_prob=1.0,
                         grazing_cos=0.0, dropout=0.0, precision=((1e9, 0.0),),
                         motion_distortion=False, dust_density=0.0, mount_z=0.0, self_occlusion=False),
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
    # Deterministic errors, fixed per drone at power-on (typical MEMS, cf. Genesis IMU model)
    scale_std: float = 0.003               # scale-factor error, 1-sigma (0.3 %)
    misalign_std: float = 0.0017           # rad, axis misalignment / cross-axis coupling (~0.1 deg)
    # ADIS16448 datasheet: +-1000 deg/s, +-18 g, 0.04 deg/s and 0.833 mg per LSB
    gyro_range: float = 17.45              # rad/s
    accel_range: float = 176.6             # m/s^2
    gyro_lsb: float = 6.98e-4              # rad/s
    accel_lsb: float = 8.17e-3             # m/s^2
    # Rotor vibration (absent from datasheets, dominant on a real multirotor): a sine at the
    # once-per-rev rotor imbalance frequency, which scales with sqrt(thrust) and aliases when
    # sampled, plus broadband shake (cf. gnss-ins-sim sinusoidal + random vibration).
    # Calibrated on EuRoC V1_01_easy (ADIS16448 on an AscTec Firefly, docs/sensors.md):
    # 20-95 Hz band RMS per body axis x, y, z, split half sine, half broadband
    # (40-65 % of the real band power sits in narrow peaks). 108 Hz aliases to the 91.6 Hz
    # peak of the 200 Hz log.
    vib_freq_hover: float = 108.0                      # Hz at hover
    vib_accel_amp: tuple = (0.56, 0.56, 0.88)          # m/s^2, sine amplitude (z = thrust axis)
    vib_gyro_amp: tuple = (0.028, 0.028, 0.017)        # rad/s (roll, pitch, yaw)
    vib_accel_std: tuple = (0.457, 0.457, 0.718)       # m/s^2, broadband = amp * sqrt(2/3)
    vib_gyro_std: tuple = (0.0229, 0.0229, 0.0139)     # rad/s

IMU_PRESETS = {
    "flight": ImuParams(),
    # Bench datasheet noise only: no vibration, scale, misalignment or quantization.
    "datasheet": ImuParams(scale_std=0.0, misalign_std=0.0, gyro_lsb=0.0, accel_lsb=0.0,
                           vib_accel_amp=0.0, vib_gyro_amp=0.0, vib_accel_std=0.0, vib_gyro_std=0.0),
}


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
    # Calibrated to published LIO behaviour in SubT caves (~0.2-1 % drift, dominated by
    # geometric degeneracy): LiDAR ranges are metric, so there is next to no scale error,
    # and heading stays tied to the local map, so its random walk is small.
    pos_noise_per_m: float = 0.01     # std of position noise per sqrt(metre) flown
    scale_error_std: float = 0.002    # per-drone fixed scale error
    yaw_rw_std: float = 0.0005        # rad / sqrt(s), yaw random walk
    yaw_noise_per_rad: float = 0.002  # rad / sqrt(rad rotated), extra yaw error when turning
    # Geometric degeneracy (Zhang et al., ICRA 2016): scan matching cannot see motion
    # along a direction no wall faces, e.g. down a smooth straight tunnel. Eigenvalues
    # of the mean n n^T of LiDAR hit normals below degen_lambda mark such directions;
    # there the noise grows by up to degen_gain and the estimate also walks with time.
    degen_lambda: float = 0.05
    degen_gain: float = 8.0
    degen_drift: float = 0.05         # m / sqrt(s) along a fully degenerate direction


ODOM_PRESETS = {
    "perfect": OdomParams(0.0, 0.0, 0.0, 0.0, degen_gain=1.0, degen_drift=0.0),
    "default": OdomParams(),
    # Weak odometry (about 2 % of distance): what the old default did.
    "hard": OdomParams(0.02, 0.01, 0.002, 0.02, degen_gain=15.0, degen_drift=0.1),
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
    target_error_m: float = 0.0      # rescue: the survivor position given to the agent is off by this much
                                     # (a random direction per seed), as a phone fix or a caller's report would be
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
