"""Where a drone can fly: one rigid definition, from the course mesh.

Every triangle of the mesh is a boundary that nothing passes: cave walls, floors and ceilings,
rocks, and the outdoor ground. A point is flyable when

  1. it is at least R_FLY from every triangle (drone radius 0.2 m + 0.1 m margin),
  2. it is on the course, by the referee's own ray test (sim.check_course): in the staging box
     with ground below, or elsewhere with rock both above and below within ENCLOSED_M,
  3. it is reached from the base through flyable points (6-connected voxels of RES).

3 is what keeps rock and soil out: the fill cannot cross a surface, so it never gets under the
ground or behind a tunnel wall -- the empty space the SubT tiles hang in, which in reality is rock.
(The triangle normals would say which side is air, but the SubT tile meshes are not wound
consistently, so the definition does not rely on them.) The fill can only leave the tunnels through
a hole in the mesh wider than 2 R_FLY; `leaks` reports flyable voxels that see out of the cave.

Distance: exact point-triangle distance to candidate triangles found with a k-d tree over points
sampled on every triangle at most SAMPLE_M apart.

    python -m cavesim.flyable maps/rescue_hard      # build (cached next to the map) and summarise
"""
import json
import sys
from pathlib import Path

import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient
from scipy.spatial import cKDTree

from .sim import ENCLOSED_M, MESH_CHUNK

R_FLY = 0.3          # m: drone collision radius 0.2 + 0.1 margin
RES = 0.5            # m voxel (the reference planner's resolution)
SAMPLE_M = 1.0       # m between the points that index a triangle in the k-d tree
K_CAND = 32          # candidate sample points per distance query
OFF = 1 << 20


def keys_of(idx):
    idx = np.asarray(idx, dtype=np.int64) + OFF
    return (idx[..., 0] << 42) | (idx[..., 1] << 21) | idx[..., 2]


def idx_of(keys):
    keys = np.asarray(keys, dtype=np.int64)
    return np.stack([(keys >> 42) - OFF, ((keys >> 21) & 0x1FFFFF) - OFF, (keys & 0x1FFFFF) - OFF], -1)


def closest_on_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (N, 3) (Ericson, Real-Time Collision
    Detection, 5.1.5)."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        out = a + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None]          # interior
        cases = [
            ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
             b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]),         # edge BC
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / (d2 - d6))[:, None]),  # edge AC
            ((d6 >= 0) & (d5 <= d6), c),                                             # vertex C
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / (d1 - d3))[:, None]),  # edge AB
            ((d3 >= 0) & (d4 <= d3), b),                                             # vertex B
            ((d1 <= 0) & (d2 <= 0), a),                                              # vertex A
        ]
    for m, q in cases:                     # later cases win: same priority as Ericson's early returns
        out[m] = q[m]
    return out


class Course:
    """The course mesh: distance to it, and the referee's ray tests on it."""

    def __init__(self, map_dir):
        map_dir = Path(map_dir)
        self.meta = json.loads((map_dir / "map.json").read_text())
        box = self.meta.get("staging_box")
        self.box = None if box is None else np.asarray(box, dtype=float)
        m = np.load(map_dir / "mesh.npz")
        V, F = m["vertices"], m["faces"]
        Vd = V.astype(np.float64)
        self.a, self.b, self.c = Vd[F[:, 0]], Vd[F[:, 1]], Vd[F[:, 2]]
        edge = np.max([np.linalg.norm(self.b - self.a, axis=1), np.linalg.norm(self.c - self.a, axis=1),
                       np.linalg.norm(self.c - self.b, axis=1)], axis=0)
        steps = np.maximum(1, np.ceil(edge / SAMPLE_M)).astype(int)
        pts, owner = [], []
        for s in np.unique(steps):
            i, j = np.meshgrid(np.arange(s + 1), np.arange(s + 1), indexing="ij")
            keep = i + j <= s
            u, v = i[keep] / s, j[keep] / s
            f = np.flatnonzero(steps == s)
            a, b, c = self.a[f], self.b[f], self.c[f]
            pts.append((a[:, None] + u[None, :, None] * (b - a)[:, None] + v[None, :, None] * (c - a)[:, None]).reshape(-1, 3))
            owner.append(np.repeat(f, len(u)))
        self.owner = np.concatenate(owner)
        self.tree = cKDTree(np.concatenate(pts))
        # The same collision mesh the simulator flies in, for the ray tests.
        self.pb = BulletClient(connection_mode=p.DIRECT)
        for k in range(0, len(F), MESH_CHUNK):
            used, f = np.unique(F[k:k + MESH_CHUNK], return_inverse=True)
            col = self.pb.createCollisionShape(p.GEOM_MESH, vertices=V[used].tolist(),
                                               indices=f.reshape(-1).tolist(), flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            self.pb.createMultiBody(0, col)

    def distance(self, q, chunk=20_000):
        """Distance of points q (N, 3) to the nearest triangle."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        out = np.empty(len(q))
        for s in range(0, len(q), chunk):
            x = q[s:s + chunk]
            _, nb = self.tree.query(x, k=K_CAND)
            t = self.owner[nb].reshape(-1)
            cp = closest_on_triangles(np.repeat(x, K_CAND, axis=0), self.a[t], self.b[t], self.c[t])
            out[s:s + chunk] = np.linalg.norm(cp.reshape(len(x), K_CAND, 3) - x[:, None], axis=2).min(1)
        return out

    def hits(self, starts, ends):
        """Does each ray hit the mesh?"""
        r = []
        for s in range(0, len(starts), 16_000):           # rayTestBatch takes at most 16384
            r += self.pb.rayTestBatch(starts[s:s + 16_000].tolist(), ends[s:s + 16_000].tolist(), numThreads=1)
        return np.array([h[0] >= 0 for h in r], dtype=bool)

    def on_course(self, q):
        """sim.check_course for points q (N, 3)."""
        up, down = self.hits(q, q + [0, 0, ENCLOSED_M]), self.hits(q, q - [0, 0, ENCLOSED_M])
        inside = np.zeros(len(q), bool) if self.box is None else np.all((q >= self.box[0]) & (q <= self.box[1]), axis=1)
        return down & (inside | up)

    def leaks(self, q, reach=1000.0, n_dirs=26):
        """Points that see out of the cave: some ray from them hits nothing within `reach`. In the
        staging area that is the sky; anywhere else it is a hole in the tunnels' mesh."""
        d = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], float)
        d /= np.linalg.norm(d, axis=1)[:, None]
        starts = np.repeat(q, n_dirs, axis=0)
        h = self.hits(starts, starts + np.tile(d * reach, (len(q), 1))).reshape(len(q), n_dirs)
        return ~h.all(1)


def flyable_voxels(map_dir, r_fly=R_FLY, cache=True):
    """{voxel key: clearance of its centre (m)} for every flyable voxel."""
    map_dir = Path(map_dir)
    path = map_dir / f"flyable_{int(round(r_fly * 100))}cm.npz"
    if cache and path.exists():
        z = np.load(path)
        return dict(zip(z["keys"].tolist(), z["clearance"].tolist()))
    course = Course(map_dir)
    start = int(keys_of(np.floor(np.asarray(course.meta["base"]) / RES)))
    six = np.array([1 << 42, -(1 << 42), 1 << 21, -(1 << 21), 1, -1], dtype=np.int64)
    seen, good = {start}, {}
    layer = np.array([start], dtype=np.int64)
    while len(layer):
        centres = (idx_of(layer) + 0.5) * RES
        d = course.distance(centres)
        keep = d >= r_fly
        keep[keep] = course.on_course(centres[keep])
        good.update(zip(layer[keep].tolist(), d[keep].tolist()))
        nxt = np.unique((layer[keep][:, None] + six[None, :]).reshape(-1)).tolist()
        nxt = [k for k in nxt if k not in seen]
        seen.update(nxt)
        layer = np.array(nxt, dtype=np.int64)
    if cache:
        np.savez(path, keys=np.array(list(good), dtype=np.int64), clearance=np.array(list(good.values())))
    return good


if __name__ == "__main__":
    import time
    t0 = time.time()
    fly = flyable_voxels(sys.argv[1])
    keys = np.array(list(fly))
    clr = np.array(list(fly.values()))
    course = Course(sys.argv[1])
    centres = (idx_of(keys) + 0.5) * RES
    outside = course.box is None or ~np.all((centres >= course.box[0]) & (centres <= course.box[1]), axis=1)
    cave = centres[outside]
    leak = course.leaks(cave)
    print(f"{sys.argv[1]}: {len(fly)} flyable voxels ({len(fly) * RES ** 3:.0f} m3), {len(cave)} of them in the cave; "
          f"clearance median {np.median(clr):.2f} m; {time.time() - t0:.0f} s")
    print(f"  cave voxels that see out (holes in the mesh): {leak.sum()} ({leak.mean():.1%})")
    if leak.any():
        from scipy.cluster.vq import kmeans2
        c, lab = kmeans2(cave[leak], min(8, int(leak.sum())), seed=0, minit="++")
        for i in np.argsort(-np.bincount(lab, minlength=len(c))):
            print(f"    around {np.round(c[i], 1)}: {np.sum(lab == i)} voxels")
