"""Calibrate the LiDAR model against real OS0-128 flights from the CTU SubT Finals dataset.

    python -m cavesim.validate --ctu data/ctu_subt/uav21_3

Data (github.com/ctu-mrs/slam_datasets, subt_finals/): one folder per sequence holding
rosbag.bag, trajectory_groundtruth.txt and fcu_in_map.mat, next to map/subt_finals.pcd.
Needs `pip install rosbags` (only for this analysis, not for the simulator).

Method: place the sensor at the ground-truth pose of every column's firing time, cast
the same beams into the ground-truth map, and compare with what the sensor returned.
Every 4th of the 128 rows is used, giving 32 beams like the simulated OS0-32.
Pixels the airframe blocks (props, legs: under 90 % returns over the flight) are left out.
"""
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

VOX, STEP, RMAX = 0.10, 0.05, 35.0
T_FCU_LIDAR = np.array([0, 0, 0.1362])   # /tf_static: fcu -> os_sensor -> os_lidar; points are in os_sensor axes
CLEAN = 0.9                              # min per-pixel return rate of an unobstructed pixel
RAW = np.dtype({"names": ["x", "y", "z", "t", "range"], "formats": ["<f4", "<f4", "<f4", "<u4", "<u4"],
                "offsets": [0, 4, 8, 20, 32], "itemsize": 48})


def read_scans(bag, every):
    """(stamp, organized 128 x 512 record array) of every `every`-th raw sweep."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        raise SystemExit("this analysis reads ROS bags: pip install rosbags")
    out = []
    with AnyReader([bag]) as r:
        con = [c for c in r.connections if c.topic.endswith("os_cloud_nodelet/points")]
        for k, (c, _, raw) in enumerate(r.messages(connections=con)):
            if k % every == 0:
                m = r.deserialize(raw, c.msgtype)
                out.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                            np.frombuffer(m.data, RAW).reshape(m.height, m.width).copy()))
    return out


def extract(seq, every=10):
    """Raycast one sequence into the map; cached in seq/rays.npz."""
    gt = np.loadtxt(seq / "trajectory_groundtruth.txt")
    T0 = np.loadtxt(seq / "fcu_in_map.mat")
    slerp, R0 = Slerp(gt[:, 0], Rotation.from_quat(gt[:, 4:8])), Rotation.from_matrix(T0[:3, :3])

    def pose(tt):
        tt = np.clip(tt, gt[0, 0], gt[-1, 0])
        p = np.stack([np.interp(tt, gt[:, 0], gt[:, n]) for n in (1, 2, 3)], -1) @ T0[:3, :3].T + T0[:3, 3]
        return p, R0 * slerp(tt)

    M = np.loadtxt(seq.parent / "map/subt_finals.pcd", skiprows=11)
    M = M[cKDTree(pose(gt[:, 0])[0]).query(M)[0] < RMAX + 2]
    tree = cKDTree(M)
    nb = M[tree.query(M, k=12)[1]]
    nb -= nb.mean(1, keepdims=True)
    normals = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))[1][:, :, 0]
    lo = M.min(0) - 1
    dims = np.ceil((M.max(0) + 1 - lo) / VOX).astype(np.int64)

    def key(p):
        i = np.floor((p - lo) / VOX).astype(np.int64)
        return (i[..., 0] * dims[1] + i[..., 1]) * dims[2] + i[..., 2]

    occ = np.unique(key(M))
    steps = np.arange(0.3, RMAX, STEP)

    def cast(origin, d):
        r = np.full(len(d), np.inf)
        for b in range(0, len(d), 4000):
            kk = key(origin[b:b + 4000, None] + d[b:b + 4000, None] * steps[None, :, None])
            hit = occ[np.clip(np.searchsorted(occ, kk), 0, len(occ) - 1)] == kk
            r[b:b + 4000] = np.where(hit.any(1), steps[hit.argmax(1)], np.inf)
        return r

    scans = read_scans(seq / "rosbag.bag", every)
    # Beam direction per pixel: median unit vector of its returns. This picks up the
    # per-row azimuth offsets of the real sensor without needing its calibration file.
    xyz = np.stack([np.stack([a["x"], a["y"], a["z"]], -1) for _, a in scans]).astype(np.float32)
    rg = np.stack([a["range"] for _, a in scans]) / 1000.0
    u = xyz / np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-9)
    u[rg < 0.5] = np.nan          # under 0.5 m is the airframe (dataset README)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # pixels the airframe always blocks
        dirs = np.nanmedian(u, 0)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    pixel_rate = (rg >= 0.5).mean(0)
    del xyz, u, rg
    use = np.isfinite(dirs).all(-1) & (np.arange(128)[:, None] % 4 == 0)

    rays, per_scan = [], []
    for stamp, a in scans:
        ii, jj = np.nonzero(use)
        tt = stamp + a["t"][0][jj] * 1e-9
        if tt.min() <= gt[0, 0] or tt.max() >= gt[-1, 0]:
            continue
        p, R = pose(tt)
        origin, d = p + R.apply(T_FCU_LIDAR), R.apply(dirs[ii, jj])
        r_exp = cast(origin, d)
        hitp = origin + d * np.where(np.isfinite(r_exp), r_exp, 0)[:, None]
        cos_inc = np.abs(np.einsum("ij,ij->i", normals[tree.query(hitp)[1]], d))
        full = a["range"] / 1000.0
        # Dust speckle: fewer than 2 of its 8 neighbours within 0.2 m of its own range.
        pad = np.pad(full, 1)
        similar = sum(np.abs(pad[1 + di:full.shape[0] + 1 + di, 1 + dj:full.shape[1] + 1 + dj] - full) < 0.2
                      for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj)
        # Floor clearance: nearest map point below the drone inside a 0.3 m radius cylinder.
        pc = pose(np.array([stamp + 0.05]))[0][0]
        cand = M[tree.query_ball_point(pc - [0, 0, 2.5], 2.6)]
        below = cand[(np.hypot(*(cand[:, :2] - pc[:2]).T) < 0.3) & (cand[:, 2] < pc[2])]
        speed = np.linalg.norm(pose(np.array([stamp + 0.25]))[0] - pose(np.array([stamp - 0.25]))[0]) / 0.5
        per_scan.append((stamp, pc[2] - below[:, 2].max() if len(below) else np.inf, speed))
        rays.append(np.stack([np.full(len(ii), len(per_scan) - 1), ii, jj, r_exp, full[ii, jj], cos_inc,
                              (similar < 2)[ii, jj]], -1))
    np.savez(seq / "rays.npz", rays=np.concatenate(rays), scans=np.array(per_scan), pixel_rate=pixel_rate)


def plane_noise(seq, pixel_rate, every=20):
    """Range residual of each point against a plane fitted to its 5 x 5 neighbours (flat patches only)."""
    res, rng = [], []
    for _, a in read_scans(seq / "rosbag.bag", every):
        P = np.stack([a["x"], a["y"], a["z"]], -1).astype(np.float64)
        R = a["range"] / 1000.0
        ok = (R >= 0.5) & (pixel_rate >= CLEAN)
        for i in range(2, R.shape[0] - 2, 2):
            for j in range(2, R.shape[1] - 2, 2):
                if not ok[i - 2:i + 3, j - 2:j + 3].all():
                    continue
                rr = R[i - 2:i + 3, j - 2:j + 3]
                if rr.max() - rr.min() > 0.3 * R[i, j]:
                    continue                                   # depth edge
                nb = np.delete(P[i - 2:i + 3, j - 2:j + 3].reshape(-1, 3), 12, 0)
                mu = nb.mean(0)
                w, V = np.linalg.eigh((nb - mu).T @ (nb - mu))
                if w[0] > 0.05 * w[1]:
                    continue                                   # not flat enough
                u = P[i, j] / R[i, j]
                res.append(((P[i, j] - mu) @ V[:, 0]) / max(abs(u @ V[:, 0]), 0.2))
                rng.append(R[i, j])
    return np.array(res), np.array(rng)


def ctu_report(folder, every=10):
    seq = Path(folder)
    if not (seq / "rays.npz").exists():
        print(f"raycasting {seq.name} into the ground-truth map (cached in rays.npz) ...", flush=True)
        extract(seq, every)
    z = np.load(seq / "rays.npz")
    rays, scans, pixel_rate = z["rays"], z["scans"], z["pixel_rate"]
    _, ii, jj, r_exp, r_meas, cos_inc, iso = rays.T
    clean = pixel_rate[ii.astype(int), jj.astype(int)] >= CLEAN
    v = clean & np.isfinite(r_exp) & (r_exp >= 0.5)
    ret = r_meas >= 0.5
    both = v & ret
    res = r_meas[both] - r_exp[both]
    print(f"CTU {seq.name}: {len(scans)} sweeps, {v.sum()} beams on unobstructed pixels "
          f"({(pixel_rate >= CLEAN).mean():.0%} of pixels return >= {CLEAN:.0%} of the time; "
          f"the airframe partly or fully blocks most others)")
    print(f"  registration: |measured - map range| < 0.2 m for {np.mean(np.abs(res) < 0.2):.0%}, "
          f"< 0.5 m for {np.mean(np.abs(res) < 0.5):.0%} of returns")

    print("  return rate by map range:")
    for lo, hi in [(0.5, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 35)]:
        s = v & (r_exp >= lo) & (r_exp < hi)
        if s.sum() > 200:
            print(f"    {lo:4.1f}-{hi:4.1f} m  n={s.sum():7d}  {ret[s].mean():.3f}")
    print("  return rate by angle from the wall normal (1-8 m):")
    for lo, hi in [(0, .05), (.05, .1), (.1, .15), (.15, .25), (.25, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        s = v & (r_exp >= 1) & (r_exp < 8) & (cos_inc >= lo) & (cos_inc < hi)
        if s.sum() > 200:
            print(f"    {np.degrees(np.arccos(min(hi, 1))):4.1f}-{np.degrees(np.arccos(lo)):4.1f} deg  "
                  f"n={s.sum():7d}  {ret[s].mean():.3f}")

    # Dust: isolated returns well in front of the mapped wall and close to the drone.
    dust = v & ret & (r_meas < 3.0) & (r_exp - r_meas > 0.5) & (iso > 0)
    sc = rays[:, 0].astype(int)
    floor = scans[sc, 1]
    print(f"  dust-like returns (< 3 m, >= 0.5 m before the wall, isolated): {dust.sum() / v.sum():.4f} of beams")
    print("    by floor clearance:")
    for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, np.inf)]:
        s = v & (floor >= lo) & (floor < hi)
        if s.sum() > 2000:
            print(f"      {lo:3.1f}-{hi:3.1f} m  sweeps={len(np.unique(sc[s])):4d}  {dust[s].sum() / s.sum():.4f}")
    if dust.sum() > 200:
        d = r_meas[dust]
        h, e = np.histogram(d, bins=np.arange(0.5, 3.01, 0.25))
        print("    dust range histogram (0.5-3 m, 0.25 m bins):", h.tolist())

    res_n, rng_n = plane_noise(seq, pixel_rate)
    print("  range noise from flat 5x5 patches (robust sigma; beyond ~2 m rock shape dominates):")
    for lo, hi in [(0.5, 1), (1, 2), (2, 5), (5, 10)]:
        s = (rng_n >= lo) & (rng_n < hi)
        if s.sum() > 50:
            x = res_n[s]
            print(f"    {lo:4.1f}-{hi:4.1f} m  n={s.sum():5d}  {1.4826 * np.median(np.abs(x - np.median(x))) * 100:5.2f} cm")
