"""Check that each sensor model produces the statistics its parameters promise.

    python -m cavesim.validate

    python -m cavesim.validate --euroc data/euroc/V1_01_easy
    python -m cavesim.validate --ctu data/ctu_subt/uav21_3      (LiDAR, see validate_ctu.py)

The first form verifies the implementation against its own spec (datasheet values in
config.py). --euroc compares the IMU with a real flight of the same sensor (ADIS16448 on
an AscTec Firefly, EuRoC MAV dataset, imu0/data.csv): motion lives below ~10 Hz, so the
20-95 Hz band holds sensor noise plus rotor vibration, which is what we calibrate.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import welch

from . import config as C
from .api import Agent
from .sensors import Imu, Lidar
from .sim import Simulator

MAP = Path(__file__).resolve().parent.parent / "maps" / "small_2"


def allan_dev(x, rate, taus):
    """Overlapping Allan deviation of a rate signal sampled at `rate` Hz."""
    theta = np.cumsum(x) / rate
    out = []
    for tau in taus:
        m = int(round(tau * rate))
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        out.append(np.sqrt(np.mean(d ** 2) / (2 * tau ** 2)))
    return np.array(out)


def imu_report(seconds=600):
    taus = np.array([0.01, 0.1, 1.0, 10.0])
    print(f"IMU, hovering still for {seconds} s (z gyro, x accel).")
    print("  model = white noise N and bias random walk K: sqrt(N^2/tau + K^2*tau/3)")
    for name, ip in C.IMU_PRESETS.items():
        imu = Imu(ip, np.random.default_rng(0))
        s = np.array([imu.sample(0, np.array([0, 0, C.GRAVITY]), np.zeros(3)) for _ in range(seconds * ip.rate_hz)])
        g, a = allan_dev(s[:, 6], ip.rate_hz, taus), allan_dev(s[:, 1], ip.rate_hz, taus)
        gm = np.sqrt(ip.gyro_noise_density ** 2 / taus + ip.gyro_random_walk ** 2 * taus / 3)
        am = np.sqrt(ip.accel_noise_density ** 2 / taus + ip.accel_random_walk ** 2 * taus / 3)
        print(f"  [{name}]  tau(s)   gyro adev      model   accel adev      model")
        for row in zip(taus, g, gm, a, am):
            print("           {:6.2f}   {:9.2e}   {:8.2e}   {:10.2e}   {:8.2e}".format(*row))


def lidar_report(sweeps=30):
    sim = Simulator(MAP, 1, Agent, cfg=C.SimConfig())
    d = sim.drones[0]
    pose = [(d.true_pos(), np.eye(3))]
    lp = sim.cfg.lidar
    perfect = replace(lp, min_detect_prob=1.0, grazing_cos=0.0, dropout=0.0, precision=((1e9, 0.0),),
                      dust_density=0.0)
    exact, _ = Lidar(perfect, sim.lidar.start).scan(sim.pb, pose, d.rng)
    rng = np.random.default_rng(0)
    scans = np.array([sim.lidar.scan(sim.pb, pose, rng, floor_gap=np.inf)[0] for _ in range(sweeps)])

    print(f"\nLiDAR (OS0-32), {sweeps} sweeps from a fixed pose, compared with exact geometry.")
    print("  range bin    rays   return rate   noise sigma   spec sigma")
    edges = [lp.min_range, 1, 5, 10, 15, 20, 25, lp.max_range]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (exact >= lo) & (exact < hi)
        if sel.sum() < 20:
            continue
        got = scans[:, sel]
        ok = np.isfinite(got)
        err = (got - exact[sel])[ok]
        spec = sim.lidar.prec_sigma[np.searchsorted(sim.lidar.prec_bounds, (lo + hi) / 2)]
        print(f"  {lo:5.1f}-{hi:4.1f} m {sel.sum():6d}   {ok.mean():10.2f}   {err.std():9.3f} m   {spec:8.3f} m")

    dusty = np.array([sim.lidar.scan(sim.pb, pose, rng, floor_gap=0.0)[0] for _ in range(sweeps)])
    far = exact > lp.dust_radius
    frac = np.mean(dusty[:, far] < lp.dust_radius)
    theory = 1 - np.exp(-lp.dust_density * (lp.dust_radius - sim.lidar.start))
    print(f"\n  rotor dust at the floor: {frac:.3f} of beams answered by dust, Beer-Lambert predicts {theory:.3f}")
    sim.close()


BAND = (20.0, 95.0)   # Hz: above the flight motion, below the 100 Hz Nyquist of a 200 Hz log


def band_stats(x, rate):
    """Per-axis RMS inside BAND and the frequency of the strongest PSD peak there."""
    f, P = welch(x, fs=rate, nperseg=1024, axis=0)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    rms = np.sqrt(np.sum(P[sel], axis=0) * (f[1] - f[0]))
    return rms, f[sel][np.argmax(P[sel], axis=0)]


def euroc_report(folder):
    raw = np.loadtxt(Path(folder) / "data.csv", delimiter=",", comments="#")
    rate = 1e9 / np.median(np.diff(raw[:, 0]))
    # EuRoC imu0 has x up (thrust); reorder to our body axes so z is the thrust axis.
    real = {"gyro": raw[:, [2, 3, 1]], "accel": raw[:, [5, 6, 4]]}
    print(f"EuRoC {Path(folder).name}: {len(raw)} samples at {rate:.0f} Hz, {len(raw) / rate:.0f} s of flight")
    print(f"  band {BAND[0]:.0f}-{BAND[1]:.0f} Hz      RMS x / y / z            strongest peak (Hz) x / y / z")
    rows = [("real", real)]
    for name, ip in C.IMU_PRESETS.items():
        imu = Imu(replace(ip, rate_hz=int(round(rate))), np.random.default_rng(0))
        s = np.array([imu.sample(0, np.array([0, 0, C.GRAVITY]), np.zeros(3)) for _ in range(len(raw))])
        rows.append((name, {"gyro": s[:, 4:7], "accel": s[:, 1:4]}))
    for kind, unit in (("accel", "m/s^2"), ("gyro", "rad/s")):
        for name, sig in rows:
            rms, pk = band_stats(sig[kind], rate)
            print(f"  {kind:5s} {name:9s} " + " ".join(f"{v:7.4f}" for v in rms) + f" {unit:6s}   "
                  + " ".join(f"{v:5.1f}" for v in pk))


if __name__ == "__main__":
    import sys
    if "--euroc" in sys.argv:
        euroc_report(sys.argv[sys.argv.index("--euroc") + 1])
    elif "--ctu" in sys.argv:
        from .validate_ctu import ctu_report
        ctu_report(sys.argv[sys.argv.index("--ctu") + 1])
    else:
        imu_report()
        lidar_report()
