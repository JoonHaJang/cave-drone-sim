"""Procedural cave generator.

A random tunnel graph is grown from a staging chamber at the origin, turned into a
signed distance field (negative = air), roughened with value noise and meshed with
marching cubes. The same mesh feeds PyBullet (collision + LiDAR) and Three.js (render).

    python -m cavesim.cavegen --seed 7 --name seed7
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

from .config import ARTIFACT_TYPES

PRESETS = {
    # bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    "small": dict(n_segments=18, bounds=(-6, 70, -35, 35, -8, 8), n_artifacts=6),
    "medium": dict(n_segments=34, bounds=(-6, 110, -50, 50, -12, 12), n_artifacts=10),
    "large": dict(n_segments=60, bounds=(-6, 160, -75, 75, -15, 15), n_artifacts=16),
}

STAGING_RADIUS = 4.5
RESOLUTION = 0.5


def grow_graph(rng, n_segments, bounds):
    """Return node positions, node radii, segment index pairs and chamber flags."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    lo = np.array([xmin + 4, ymin + 4, zmin + 3])
    hi = np.array([xmax - 4, ymax - 4, zmax - 3])
    center = (lo + hi) / 2

    nodes = [np.zeros(3)]
    radii = [STAGING_RADIUS]
    chamber = [True]
    segs = []
    tips = [(0, 0.0, 0.0)]  # (node index, heading, pitch)
    fails = [0]             # consecutive failed growth attempts per tip

    attempts = 0
    while len(segs) < n_segments and attempts < n_segments * 40:
        attempts += 1
        if not tips:
            # Every tip is boxed in: sprout a side branch from a random tunnel node.
            j = int(rng.integers(1, len(nodes))) if len(nodes) > 1 else 0
            tips.append((j, rng.uniform(-np.pi, np.pi), 0.0))
            fails.append(0)
        # Depth-first most of the time so tunnels get long; sometimes resume an old tip.
        k = len(tips) - 1 if rng.random() < 0.7 else int(rng.integers(len(tips)))
        idx, heading, pitch = tips[k]
        h = heading + rng.normal(0, 0.55)
        p = np.clip(0.5 * pitch + rng.normal(0, 0.18), -0.35, 0.35)
        length = rng.uniform(8, 14)
        d = np.array([np.cos(h) * np.cos(p), np.sin(h) * np.cos(p), np.sin(p)])
        new = nodes[idx] + length * d

        if np.any(new < lo) or np.any(new > hi):
            # Steer back toward the middle of the box and try again later.
            to_c = center - nodes[idx]
            tips[k] = (idx, np.arctan2(to_c[1], to_c[0]), 0.0)
            continue
        dists = np.linalg.norm(np.array(nodes) - new, axis=1)
        dists[idx] = np.inf
        if dists.min() < 7.0:
            if rng.random() < 0.15:  # occasionally allow a loop back into the network
                j = int(np.argmin(dists))
                if (idx, j) not in segs and (j, idx) not in segs:
                    segs.append((idx, j))
            # Turn away and retry a few times before giving up on this tip.
            fails[k] += 1
            if fails[k] >= 4:
                tips.pop(k)
                fails.pop(k)
            else:
                tips[k] = (idx, heading + rng.choice([-1, 1]) * rng.uniform(0.6, 1.4), pitch)
            continue

        is_chamber = rng.random() < 0.12
        nodes.append(new)
        radii.append(rng.uniform(4.0, 6.0) if is_chamber else rng.uniform(1.8, 3.2))
        chamber.append(is_chamber)
        j = len(nodes) - 1
        segs.append((idx, j))

        tips[k] = (j, h, p)
        fails[k] = 0
        if rng.random() < 0.3:
            tips.append((j, h + rng.choice([-1, 1]) * rng.uniform(0.8, 1.6), 0.0))
            fails.append(0)

    return np.array(nodes), np.array(radii), segs, np.array(chamber)


def value_noise(rng, shape, cell):
    """Smooth noise in [-1, 1] by cubic upsampling of a coarse random grid."""
    coarse = [max(2, int(np.ceil(s / cell)) + 2) for s in shape]
    g = rng.uniform(-1, 1, coarse)
    zoom = [s / c for s, c in zip(shape, coarse)]
    return ndimage.zoom(g, zoom, order=3)[: shape[0], : shape[1], : shape[2]]


def build_sdf(rng, nodes, radii, segs, chamber, bounds, res):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    origin = np.array([xmin, ymin, zmin], dtype=float)
    shape = tuple(int(np.ceil((b - a) / res)) + 1 for a, b in [(xmin, xmax), (ymin, ymax), (zmin, zmax)])
    field = np.full(shape, 10.0, dtype=np.float32)

    def block(lo, hi):
        i0 = np.maximum(np.floor((lo - origin) / res).astype(int), 0)
        i1 = np.minimum(np.ceil((hi - origin) / res).astype(int) + 1, shape)
        axes = [origin[a] + res * np.arange(i0[a], i1[a]) for a in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2])), P

    pad = 7.0
    for i, j in segs:
        a, b = nodes[i], nodes[j]
        # Chamber radii only apply at the chamber itself, tunnels leave at tunnel width.
        ra = radii[i] if not chamber[i] else 2.6
        rb = radii[j] if not chamber[j] else 2.6
        sl, P = block(np.minimum(a, b) - pad, np.maximum(a, b) + pad)
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        closest = a + t[..., None] * ab
        d = np.linalg.norm(P - closest, axis=-1) - (ra + t * (rb - ra))
        field[sl] = np.minimum(field[sl], d)

    for n, r, c in zip(nodes, radii, chamber):
        if not c:
            continue
        sl, P = block(n - r - 2, n + r + 2)
        # Squash chambers vertically so they look like halls, not bubbles.
        q = (P - n) / np.array([1.0, 1.0, 0.6])
        field[sl] = np.minimum(field[sl], np.linalg.norm(q, axis=-1) - r)

    field += 0.8 * value_noise(rng, shape, 4.0 / res) + 0.3 * value_noise(rng, shape, 1.5 / res)
    return field, origin


def sample_field(field, origin, res, pts):
    idx = ((np.atleast_2d(pts) - origin) / res).T
    return ndimage.map_coordinates(field, idx, order=1, mode="nearest")


def mesh_from_sdf(field, origin, res):
    padded = np.pad(field, 1, constant_values=10.0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.0, spacing=(res, res, res))
    verts = verts - res + origin
    mesh = trimesh.Trimesh(verts, faces, process=True)
    parts = mesh.split(only_watertight=False)
    mesh = max(parts, key=lambda m: len(m.faces))
    return mesh


def place_artifacts(rng, field, origin, res, nodes, segs, n):
    # Graph distance from the staging chamber so artifacts sit deep in the cave.
    adj = {k: [] for k in range(len(nodes))}
    for i, j in segs:
        w = float(np.linalg.norm(nodes[i] - nodes[j]))
        adj[i].append((j, w))
        adj[j].append((i, w))
    dist = np.full(len(nodes), np.inf)
    dist[0] = 0.0
    todo = [0]
    while todo:
        u = todo.pop()
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                todo.append(v)

    candidates = [(i, j) for i, j in segs if min(dist[i], dist[j]) > 12.0]
    rng.shuffle(candidates)
    artifacts = []
    for i, j in candidates:
        if len(artifacts) >= n:
            break
        p = nodes[i] + rng.uniform(0.2, 0.8) * (nodes[j] - nodes[i])
        if sample_field(field, origin, res, p)[0] > -0.8:
            continue
        # Drop onto the floor.
        z = p[2]
        while z > p[2] - 8 and sample_field(field, origin, res, [p[0], p[1], z - 0.2])[0] < 0:
            z -= 0.2
        pos = np.array([p[0], p[1], z + 0.3])
        if any(np.linalg.norm(pos - np.array(a["pos"])) < 8 for a in artifacts):
            continue
        artifacts.append({
            "id": len(artifacts),
            "type": ARTIFACT_TYPES[len(artifacts) % len(ARTIFACT_TYPES)],
            "pos": [round(float(v), 3) for v in pos],
        })
    return artifacts


def generate(seed, size="medium", res=RESOLUTION):
    preset = PRESETS[size]
    rng = np.random.default_rng(seed)
    nodes, radii, segs, chamber = grow_graph(rng, preset["n_segments"], preset["bounds"])
    field, origin = build_sdf(rng, nodes, radii, segs, chamber, preset["bounds"], res)
    # Keep the staging chamber smooth so drones never spawn inside rock.
    sl_pts = np.stack(np.meshgrid(*[np.arange(n) for n in field.shape], indexing="ij"), -1) * res + origin
    near = np.linalg.norm(sl_pts, axis=-1) < STAGING_RADIUS - 0.5
    field[near] = np.minimum(field[near], -1.0)
    mesh = mesh_from_sdf(field, origin, res)
    artifacts = place_artifacts(rng, field, origin, res, nodes, segs, preset["n_artifacts"])
    meta = {
        "seed": seed,
        "size": size,
        "resolution": res,
        "bounds": list(preset["bounds"]),
        "base": [0.0, 0.0, 0.0],
        "graph": {"nodes": nodes.round(3).tolist(), "segments": [list(s) for s in segs]},
    }
    return mesh, meta, artifacts


def save(mesh, meta, artifacts, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mesh.npz",
                        vertices=mesh.vertices.astype(np.float32),
                        faces=mesh.faces.astype(np.int32))
    mesh.export(out / "cave.glb")
    (out / "map.json").write_text(json.dumps(meta, indent=1))
    # Ground truth for scoring. Agents never receive this file.
    (out / "truth.json").write_text(json.dumps({"artifacts": artifacts}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", choices=PRESETS, default="medium")
    ap.add_argument("--name", default=None, help="output folder under maps/ (default: <size>_<seed>)")
    ap.add_argument("--res", type=float, default=RESOLUTION)
    args = ap.parse_args()
    name = args.name or f"{args.size}_{args.seed}"
    mesh, meta, artifacts = generate(args.seed, args.size, args.res)
    out = Path(__file__).resolve().parent.parent / "maps" / name
    save(mesh, meta, artifacts, out)
    print(f"{out}: {len(mesh.faces)} triangles, {len(meta['graph']['segments'])} tunnels, "
          f"{len(artifacts)} artifacts")


if __name__ == "__main__":
    main()
