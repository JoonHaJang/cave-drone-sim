"""Procedural cave generator.

A random tunnel graph is grown from a staging chamber at the origin, turned into a
signed distance field (negative = air), roughened with value noise and meshed with
marching cubes. Stalactites, stalagmites and columns are then added as separate
cone meshes, because thin spikes would not survive the 0.5 m voxel grid. The same
geometry feeds PyBullet (collision + LiDAR) and Three.js (render).

    python -m cavesim.cavegen --seed 7 --name seed7
    python -m cavesim.cavegen --size small --seed 2 --name small_2 --speleothems 1.5
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

# Speleothems per metre of tunnel (scaled by --speleothems) and per chamber.
SPELEO_PER_M = 0.45
SPELEO_PER_CHAMBER = 14
SPELEO_KEEP_CLEAR = 7.0   # m around the base: take-off and landing stay free
SPELEO_MIN_GAP = 0.9      # m of air always left between a spike and the opposite surface


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


def surface_above_below(field, origin, res, p, reach=12.0, step=0.1):
    """Ceiling and floor heights straight above and below air point p (None if out of reach)."""
    dz = np.arange(step, reach, step)
    out = []
    for sign in (1, -1):
        col = np.column_stack([np.full_like(dz, p[0]), np.full_like(dz, p[1]), p[2] + sign * dz])
        f = sample_field(field, origin, res, col)
        k = np.flatnonzero(f >= 0)
        if len(k) == 0:
            out.append(None)
            continue
        k = k[0]
        f0 = sample_field(field, origin, res, p)[0] if k == 0 else f[k - 1]
        z0 = p[2] + sign * (dz[k - 1] if k else 0.0)
        out.append(z0 + sign * step * (-f0) / max(f[k] - f0, 1e-6))
    return out[0], out[1]


def lathe(rng, rings, n_seg=12):
    """Closed mesh through horizontal rings [(center xyz, radius), ...] sorted by height."""
    ang = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    verts = []
    for c, r in rings:
        rr = r * rng.uniform(0.88, 1.12, n_seg)        # knobbly, not lathe-perfect
        verts.append(np.column_stack([c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang), np.full(n_seg, c[2])]))
    verts = np.vstack(verts + [rings[0][0], rings[-1][0]])
    bottom, top = len(verts) - 2, len(verts) - 1
    faces = []
    for i in range(len(rings) - 1):
        for j in range(n_seg):
            a, b = i * n_seg + j, i * n_seg + (j + 1) % n_seg
            faces += [(a, b, b + n_seg), (a, b + n_seg, a + n_seg)]
    last = (len(rings) - 1) * n_seg
    for j in range(n_seg):
        faces += [(bottom, (j + 1) % n_seg, j), (top, last + j, last + (j + 1) % n_seg)]
    return trimesh.Trimesh(verts, np.array(faces), process=False)


def speleothem_mesh(rng, s, embed=0.6, n_rings=8):
    """Mesh for one speleothem dict. Bases sink `embed` m into the rock."""
    x, y = s["base"][:2]
    drift = rng.normal(0, 0.04 * s["length"], (n_rings, 2)).cumsum(0)   # slightly crooked
    drift -= drift[0]
    if s["kind"] == "column":
        z0, z1 = s["floor"] - embed, s["ceiling"] + embed
        t = np.linspace(0, 1, n_rings)
        r = s["radius"] * (0.45 + 0.55 * np.abs(2 * t - 1) ** 1.6)
        r[0] = r[-1] = s["radius"] * 1.1
    else:
        t = np.linspace(0, 1, n_rings)                  # 0 at the root, 1 at the tip
        r = np.maximum(s["radius"] * (1 - t) ** 1.4, 0.015)
        sign = 1 if s["kind"] == "stalagmite" else -1
        root = s["base"][2] - sign * embed
        z = root + sign * (embed + s["length"]) * t
        if sign < 0:                                    # rings must go bottom to top
            t, r, z, drift = t[::-1], r[::-1], z[::-1], drift[::-1]
        z0, z1 = z[0], z[-1]
    zs = np.linspace(z0, z1, n_rings)
    rings = [((x + dx, y + dy, zz), rr) for (dx, dy), zz, rr in zip(drift, zs, r)]
    return lathe(rng, rings)


def place_speleothems(rng, field, origin, res, nodes, radii, segs, chamber, artifacts, density=1.0):
    """Stalactites hang from the ceiling, stalagmites rise from the floor, columns join both.

    Every spike leaves at least SPELEO_MIN_GAP of air to the opposite surface, so no
    tunnel is sealed, but a drone has to thread between them."""
    cands = []
    for i, j in segs:
        a, b = nodes[i], nodes[j]
        ab = b - a
        n = rng.poisson(SPELEO_PER_M * density * np.linalg.norm(ab))
        side = np.array([-ab[1], ab[0], 0.0]) / (np.linalg.norm(ab[:2]) + 1e-9)
        for _ in range(n):
            t = rng.uniform()
            r = (radii[i] if not chamber[i] else 2.6) * (1 - t) + (radii[j] if not chamber[j] else 2.6) * t
            cands.append((a + t * ab + side * rng.uniform(-0.6, 0.6) * r, False))
    for k in np.flatnonzero(chamber):
        for _ in range(rng.poisson(SPELEO_PER_CHAMBER * density)):
            ang, rr = rng.uniform(0, 2 * np.pi), radii[k] * np.sqrt(rng.uniform()) * 0.85
            cands.append((nodes[k] + rr * np.array([np.cos(ang), np.sin(ang), 0.0]), True))

    art_xy = np.array([a["pos"][:2] for a in artifacts]).reshape(-1, 2)
    placed = []
    for p, in_chamber in cands:
        if np.linalg.norm(p) < SPELEO_KEEP_CLEAR or sample_field(field, origin, res, p)[0] > -0.4:
            continue
        if len(art_xy) and np.linalg.norm(art_xy - p[:2], axis=1).min() < 2.0:
            continue
        if any(np.linalg.norm(np.array(s["base"][:2]) - p[:2]) < 1.2 + s["radius"] for s in placed):
            continue
        ceil, floor = surface_above_below(field, origin, res, p)
        if ceil is None or floor is None or ceil - floor < 1.6:
            continue
        gap = ceil - floor
        around = [p + 1.6 * np.array([np.cos(a), np.sin(a), 0.0]) for a in np.arange(4) * np.pi / 2]
        roomy = (sample_field(field, origin, res, np.array(around)) < -0.3).all()
        u = rng.uniform()
        if in_chamber and roomy and gap > 2.5 and u < 0.2:
            s = {"kind": "column", "base": [p[0], p[1], floor], "floor": floor, "ceiling": ceil,
                 "length": gap, "radius": rng.uniform(0.3, 0.65)}
        else:
            kind = "stalactite" if u < 0.6 else "stalagmite"
            frac = rng.uniform(0.25, 0.65) if kind == "stalactite" else rng.uniform(0.2, 0.5)
            length = min(frac * gap, gap - SPELEO_MIN_GAP, 4.5 if kind == "stalactite" else 3.0)
            if length < 0.4:
                continue
            surf = ceil if kind == "stalactite" else floor
            s = {"kind": kind, "base": [p[0], p[1], surf], "length": length,
                 "radius": float(np.clip(length * rng.uniform(0.1, 0.18), 0.08, 0.5))}
        placed.append(s)
    meshes = [speleothem_mesh(rng, s) for s in placed]
    info = [{"kind": s["kind"], "base": [round(float(v), 3) for v in s["base"]],
             "length": round(float(s["length"]), 3), "radius": round(float(s["radius"]), 3)} for s in placed]
    return (trimesh.util.concatenate(meshes) if meshes else None), info


def generate(seed, size="medium", res=RESOLUTION, speleothems=1.0):
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
    # Own random stream, so the tunnels and artifacts of a seed stay the same with or without spikes.
    speleo, speleo_info = place_speleothems(np.random.default_rng([seed, 1]), field, origin, res,
                                            nodes, radii, segs, chamber, artifacts, speleothems)
    meta = {
        "seed": seed,
        "size": size,
        "resolution": res,
        "bounds": list(preset["bounds"]),
        "base": [0.0, 0.0, 0.0],
        "graph": {"nodes": nodes.round(3).tolist(), "segments": [list(s) for s in segs]},
        "speleothem_density": speleothems,
        "speleothems": speleo_info,
    }
    return {"rock": mesh, "speleothems": speleo}, meta, artifacts


def save(meshes, meta, artifacts, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = {k: m for k, m in meshes.items() if m is not None}
    # PyBullet gets one concave body; the viewer gets the parts so it can shade them apart.
    full = trimesh.util.concatenate(list(parts.values()))
    np.savez_compressed(out / "mesh.npz",
                        vertices=full.vertices.astype(np.float32),
                        faces=full.faces.astype(np.int32))
    scene = trimesh.Scene()
    for name, m in parts.items():
        scene.add_geometry(m, geom_name=name, node_name=name)
    scene.export(out / "cave.glb")
    (out / "map.json").write_text(json.dumps(meta, indent=1))
    # Ground truth for scoring. Agents never receive this file.
    (out / "truth.json").write_text(json.dumps({"artifacts": artifacts}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", choices=PRESETS, default="medium")
    ap.add_argument("--name", default=None, help="output folder under maps/ (default: <size>_<seed>)")
    ap.add_argument("--res", type=float, default=RESOLUTION)
    ap.add_argument("--speleothems", type=float, default=1.0,
                    help="stalactite/stalagmite density multiplier, 0 = none")
    args = ap.parse_args()
    name = args.name or f"{args.size}_{args.seed}"
    meshes, meta, artifacts = generate(args.seed, args.size, args.res, args.speleothems)
    out = Path(__file__).resolve().parent.parent / "maps" / name
    save(meshes, meta, artifacts, out)
    kinds = [s["kind"] for s in meta["speleothems"]]
    print(f"{out}: {len(meshes['rock'].faces)} rock triangles, {len(meta['graph']['segments'])} tunnels, "
          f"{len(artifacts)} artifacts, {kinds.count('stalactite')} stalactites, "
          f"{kinds.count('stalagmite')} stalagmites, {kinds.count('column')} columns")


if __name__ == "__main__":
    main()
