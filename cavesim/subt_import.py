"""Import an official DARPA SubT virtual cave world as a cave-sim map.

    python -m cavesim.subt_import simple_cave_01

The world file (tile layout and artifacts) comes from github.com/osrf/subt and the tile
meshes and textures from Gazebo Fuel (OpenRobotics, CC BY 4.0). Downloads are cached in
data/fuel/. Writes maps/<world>/ in the same format cavegen produces, plus textures/:

    mesh.npz      every tile placed and merged, metres (PyBullet collision + LiDAR)
    cave.glb      one mesh per tile kind, instanced per placement, with UVs (viewer)
    textures/     the shared PBR rock textures (albedo, normal, roughness)
    map.json      base (staging area) and the coverage graph (tile centres)
    truth.json    artifacts

Needs `pip install pycollada` to read the tiles' COLLADA meshes (import only).
"""
import argparse
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "fuel"
WORLD_URL = "https://raw.githubusercontent.com/osrf/subt/master/subt_ign/worlds/{}.{}"   # .sdf layout, .dot graph
MODEL_URL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/{}/tip/{}.zip"
TEXTURE_MODEL = "Cave Straight Type A"        # Type B tiles point at this model's textures
# Tile part name prefix -> shared texture set (part names differ a little between tiles)
MATERIALS = {"Striated": "StriatedRock", "RockPile": "RockPile", "CaveWall": "CaveWall", "Grass": "Grass",
             "Gravel": "Gravel"}
ARTIFACTS = {"Rescue Randy Sitting": "survivor", "Samsung J8 Black": "phone", "JanSport Backpack Red": "backpack",
             "Climbing Helmet With Light": "helmet", "Climbing Rope": "rope"}
BASE_HEIGHT = 1.5                              # m above the staging-area floor


def fetch(url):
    req = urllib.request.Request(urllib.parse.quote(url, safe=":/"), headers={"User-Agent": "cave-sim"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def model_dir(name):
    """Download a Fuel model once; returns its cached folder."""
    out = CACHE / "models" / name
    if not out.exists():
        print(f"  downloading {name}")
        zipfile.ZipFile(io.BytesIO(fetch(MODEL_URL.format(name, name)))).extractall(out)
    return out


def pose_matrix(text):
    x, y, z, roll, pitch, yaw = map(float, text.split())
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = x, y, z
    return T


def parse_world(sdf):
    """[(instance name, model name, 4x4 pose)] for every <include>."""
    out = []
    for inc in re.findall(r"<include>(.*?)</include>", sdf, re.S):
        uri = re.search(r"<uri>\s*([^<]*?)\s*</uri>", inc).group(1)
        name = re.search(r"<name>\s*([^<]*?)\s*</name>", inc)
        pose = re.search(r"<pose>\s*([^<]*?)\s*</pose>", inc)
        out.append((name.group(1) if name else "", uri.rsplit("/", 1)[-1].strip(),
                    pose_matrix(pose.group(1)) if pose else np.eye(4)))
    return out


def mesh_model(name):
    """The model that holds a tile's mesh: "... Lights" tiles only <include> their plain tile and
    add light sources around it."""
    d = model_dir(name)
    if (d / "meshes").exists():
        return name
    inc = re.search(r"<include>.*?<uri>\s*([^<]*?)\s*</uri>", (d / "model.sdf").read_text(encoding="utf-8"), re.S)
    return mesh_model(inc.group(1).rsplit("/", 1)[-1])


def texture_sets(name):
    """The tile's own part -> texture set table, from the albedo map of each <visual> in its
    model.sdf: {submesh name: set} plus "" for a visual without submesh (the whole mesh)."""
    sdf = (model_dir(name) / "model.sdf").read_text(encoding="utf-8")
    table = {}
    for vis in re.findall(r"<visual[^>]*>(.*?)</visual>", sdf, re.S):
        albedo = re.search(r"<albedo_map>[^<]*/(\w+?)_Albedo\.\w+</albedo_map>", vis)
        if albedo:
            sub = re.search(r"<submesh>\s*<name>\s*([^<]*?)\s*</name>", vis)
            table.setdefault(sub.group(1) if sub else "", albedo.group(1))
    return table


def first_pose(block):
    """The element's own <pose> (the first one before any child element), as a 4x4 matrix."""
    head = re.split(r"<(?:collision|visual|geometry|link|light)\b", block, maxsplit=1)[0]
    pose = re.search(r"<pose>\s*([^<]*?)\s*</pose>", head)
    return pose_matrix(pose.group(1)) if pose else np.eye(4)


def mesh_pose(name):
    """Link pose x visual pose of the tile's mesh inside its model (the staging area, for one,
    sits 90 degrees turned in its model)."""
    sdf = (model_dir(name) / "model.sdf").read_text(encoding="utf-8")
    for link in re.findall(r"<link[^>]*>(.*?)</link>", sdf, re.S):
        vis = re.search(r"<visual[^>]*>(.*?)</visual>", link, re.S)
        if vis and ".dae" in vis.group(1):
            return first_pose(link) @ first_pose(vis.group(1))
    return np.eye(4)


def fences(name, T):
    """Collision-only boxes of a model (links without visuals), e.g. the staging-area fence that
    keeps robots from leaving except through the cave mouth. [(4x4 world pose, size)]."""
    out = []
    sdf = (model_dir(name) / "model.sdf").read_text(encoding="utf-8")
    for link in re.findall(r"<link[^>]*>(.*?)</link>", sdf, re.S):
        if "<visual" in link:
            continue
        for col in re.findall(r"<collision[^>]*>(.*?)</collision>", link, re.S):
            size = re.search(r"<box>\s*<size>\s*([^<]*?)\s*</size>", col)
            if size:
                out.append((T @ first_pose(link) @ first_pose(col), [float(x) for x in size.group(1).split()]))
    return out


def load_tile(name):
    """[(texture set, part)] of one tile, in metres, in the tile's model frame."""
    name = mesh_model(name)
    dae = next((model_dir(name) / "meshes").glob("*.dae"))
    scene = trimesh.load(dae, force="scene")
    placed = mesh_pose(name)
    scale = 0.01 if str(scene.units).startswith("0.01") else 1.0   # SubT tiles are authored in cm
    table = texture_sets(name)
    parts = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname].copy()
        g.apply_transform(T)
        g.apply_scale(scale)
        g.apply_transform(placed)
        part = gname.removesuffix("-lib")
        # The tile's model.sdf says which texture each submesh wears; fall back on the shared
        # naming (StriatedRock / Striated ...), then on the tile's whole-mesh texture.
        mat = (table.get(part) or next((m for prefix, m in MATERIALS.items() if part.startswith(prefix)), None)
               or table.get(""))
        if mat is None:
            raise ValueError(f"{name}: no texture set for part {gname}")
        g.visual = trimesh.visual.TextureVisuals(uv=g.visual.uv, material=trimesh.visual.material.PBRMaterial(name=mat))
        parts.append((mat, g))
    return parts


def find_entrance(cave, tile_mesh, base, box, staging_box, res=0.25):
    """Route from the base into the first cave tile, as a few straight legs.

    Search space follows the simulator's course rule: ground below within 30 m, and either in
    the staging box or rock above too (the terrain is a one-sided sheet with empty space under
    it, and the tunnels hang in empty space). The goal is any free voxel inside the first tile's
    own tube, judged from that tile's mesh alone so the terrain sheet cannot fake a tunnel.
    Breadth-first search, then greedy straight legs."""
    lo, hi = np.floor(box[0]) - 2, np.ceil(box[1]) + 2
    shape = np.ceil((hi - lo) / res).astype(int)

    def voxels(mesh, n):
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
        pts = pts[np.all((pts >= lo) & (pts < hi), 1)]
        grid = np.zeros(shape, bool)
        grid[tuple(((pts - lo) / res).astype(int).T)] = True
        return grid

    def above_below(grid, reach):
        up, down = np.zeros(shape, bool), np.zeros(shape, bool)
        for k in range(shape[2]):
            down[:, :, k] = grid[:, :, max(k - reach, 0):k].any(2)
            up[:, :, k] = grid[:, :, k + 1:k + 1 + reach].any(2)
        return up, down

    rock = voxels(cave, 30_000_000)
    roof, ground = above_below(rock, int(30 / res))
    cell_lo = ((np.asarray(staging_box[0]) - lo) / res).astype(int)
    cell_hi = ((np.asarray(staging_box[1]) - lo) / res).astype(int)
    in_box = np.zeros(shape, bool)
    in_box[cell_lo[0]:cell_hi[0] + 1, cell_lo[1]:cell_hi[1] + 1, cell_lo[2]:cell_hi[2] + 1] = True
    off_course = ~(ground & (in_box | roof))
    # Rock voxels themselves only: a 0.5 m voxel centre is >= 0.25 m from rock, more than the 0.2 m
    # airframe radius. The rock piles inside SubT tiles leave ~1 m gaps that one voxel of margin closes.
    wall = rock | off_course
    tile_up, tile_down = above_below(voxels(tile_mesh, 4_000_000), int(15 / res))
    inside = tile_up & tile_down & roof & ~wall
    open_air = ground & ~roof & ~wall                     # the staging terrain under open sky
    depth = ndimage.distance_transform_edt(~open_air)

    # Breadth-first from the base to the nearest voxel of the first tile that is at least 3 m in
    # from open air: the nearest real way in, not some seam between the terrain sheet and a tube.
    target = inside & (depth * res >= 3.0)
    start = tuple(((np.asarray(base) - lo) / res).astype(int))
    centre = lambda c: lo + (np.array(c) + 0.5) * res
    prev, q, goal = {start: None}, deque([start]), None
    while q:
        c = q.popleft()
        if target[c]:
            goal = c
            break
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if n not in prev and all(0 <= n[k] < shape[k] for k in range(3)) and not wall[n]:
                prev[n] = c
                q.append(n)
    if goal is None:
        raise SystemExit("no free-space route from the base into the first cave tile")
    cells = []
    while goal is not None:
        cells.append(goal)
        goal = prev[goal]

    # Heights stay as found: the agent itself holds a minimum height above the ground below it.
    path = [centre(c) for c in cells[::-1]]

    def clear(a, b, grid):
        n = int(np.linalg.norm(b - a) / (res / 2)) + 2
        idx = ((a + (b - a) * np.linspace(0, 1, n)[:, None] - lo) / res).astype(int)
        return not grid[tuple(idx.T)].any()

    for grid in (ndimage.binary_dilation(rock, iterations=2) | off_course, ndimage.binary_dilation(rock) | off_course, wall):
        route, i = [], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not clear(path[i], path[j], grid):
                j -= 1
            if not clear(path[i], path[j], grid):
                break                                   # too tight for this clearance: try the next
            route.append(path[j])
            i = j
        else:
            return [p.round(2).tolist() for p in route]
    raise SystemExit("entrance route could not be straightened")


def floor_below(mesh, xy, top):
    """Height of the highest surface under (x, y) below `top` (vertical ray, barycentric test)."""
    a, b, c = (mesh.vertices[mesh.faces[:, k]] for k in range(3))
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    with np.errstate(all="ignore"):
        u = ((b[:, 1] - c[:, 1]) * (xy[0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (xy[1] - c[:, 1])) / d
        v = ((c[:, 1] - a[:, 1]) * (xy[0] - c[:, 0]) + (a[:, 0] - c[:, 0]) * (xy[1] - c[:, 1])) / d
    w = 1 - u - v
    inside = (u >= 0) & (v >= 0) & (w >= 0) & (np.abs(d) > 1e-12)
    z = u * a[:, 2] + v * b[:, 2] + w * c[:, 2]
    z = z[inside & (z < top)]
    return float(z.max()) if len(z) else 0.0


RESCUE_BATTERY_S = 1200.0        # rescue courses: battery and time limit long enough for the farthest survivor


def import_world(world, out_dir, survivor=None, difficulty=None):
    sdf_path = CACHE / "worlds" / f"{world}.sdf"
    if not sdf_path.exists():
        sdf_path.parent.mkdir(parents=True, exist_ok=True)
        sdf_path.write_bytes(fetch(WORLD_URL.format(world, "sdf")))
    dot_path = sdf_path.with_suffix(".dot")
    if not dot_path.exists():
        dot_path.write_bytes(fetch(WORLD_URL.format(world, "dot")))
    includes = parse_world(sdf_path.read_text(encoding="utf-8"))
    tiles = [(n, m, T) for n, m, T in includes if m.startswith("Cave")]
    print(f"{world}: {len(tiles)} tiles of {len({m for _, m, _ in tiles})} kinds")

    kinds = {m: load_tile(m) for m in sorted({m for _, m, _ in tiles})}
    scene = trimesh.Scene()
    merged = []
    staging = []
    for n, m, T in tiles:
        for k, (mat, g) in enumerate(kinds[m]):
            scene.add_geometry(g, geom_name=f"{m}|{k}", node_name=f"{n}|{k}|{mat}", transform=T)
            merged.append(g.copy().apply_transform(T))
            if "Starting Area" in m:
                staging.append(merged[-1].bounds)
    # Course boundary, part 1: the outdoor staging area (the cave proper is checked by the simulator).
    staging_box = [np.min([b[0] for b in staging], 0).round(2).tolist(), np.max([b[1] for b in staging], 0).round(2).tolist()]
    fence = [f for n, m, T in tiles if "Starting Area" in m for f in fences(m, T)]
    if fence:
        # The fence encloses the staging area: use its inside as the staging box (open towards the cave).
        corners = np.array([(T @ np.append(np.multiply(c, s) / 2, 1))[:3] for T, s in fence
                            for c in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)])
        staging_box = [[*corners[:, :2].min(0).round(2).tolist(), staging_box[0][2]],
                       [*corners[:, :2].max(0).round(2).tolist(), float(corners[:, 2].max().round(2))]]
    cave = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in merged])
    cave.merge_vertices()

    out = Path(out_dir)
    (out / "textures").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mesh.npz", vertices=cave.vertices.astype(np.float32), faces=cave.faces.astype(np.int32))
    scene.export(out / "cave.glb", include_normals=True)
    tex = model_dir(TEXTURE_MODEL) / "materials" / "textures"
    for mat in set(MATERIALS.values()):
        for kind in ("Albedo", "Normal", "Roughness"):
            (out / "textures" / f"{mat}_{kind}.jpg").write_bytes((tex / f"{mat}_{kind}.jpg").read_bytes())

    staging = next(T for n, m, T in includes if "Starting Area" in m)[:3, 3]
    floor = floor_below(cave, staging[:2], staging[2] + 20)
    base = [float(staging[0]), float(staging[1]), float(floor + BASE_HEIGHT)]
    nodes = [base] + [T[:3, 3].round(2).tolist() for n, m, T in tiles if "Starting Area" not in m]
    # SubT robots start outside the cave mouth; the entrance is the first cave tile.
    # The tile the world's connectivity graph joins to the base station (node 0) is where the cave starts.
    dot = dot_path.read_text(encoding="utf-8")
    labels = dict(re.findall(r'^\s*(\d+)\s*\[label="\d+::[^:]*::([^"]+)"\]', dot, re.M))
    joined = [b if a == "0" else a for a, b in re.findall(r"^\s*(\d+)\s*--\s*(\d+)", dot, re.M) if "0" in (a, b)]
    n0, m0, T0 = next(t for t in tiles if t[0] == labels[joined[0]])
    first = T0[:3, 3].tolist()
    tile_mesh = trimesh.util.concatenate([g.copy().apply_transform(T0) for _, g in kinds[m0]])
    box = [np.minimum(staging_box[0], np.subtract(first, 15)), np.maximum(staging_box[1], np.add(first, 15))]
    entrance_route = find_entrance(cave, tile_mesh, base, box, staging_box)
    entrance = entrance_route[-1]
    yaw = lambda T: round(float(np.arctan2(T[1, 0], T[0, 0])), 4)
    artifacts = [{"id": k, "type": ARTIFACTS[m], "pos": T[:3, 3].round(3).tolist(), "yaw": yaw(T), "name": n}
                 for k, (n, m, T) in enumerate((x for x in includes if x[1] in ARTIFACTS))]
    station = next(T for n, m, T in includes if m == "Base Station")
    mission = {"type": "explore"}
    if survivor:
        # Rescue course: that one survivor is the whole job.
        artifacts = [dict(a, id=0) for a in artifacts if a["name"] == survivor]
        if not artifacts:
            raise SystemExit(f"{world} has no artifact named {survivor}")
        mission = {"type": "rescue", "target": 0, "difficulty": difficulty,
                   "battery_s": RESCUE_BATTERY_S, "time_s": RESCUE_BATTERY_S}
    lo, hi = cave.bounds
    (out / "map.json").write_text(json.dumps({
        "source": f"DARPA SubT virtual world '{world}' (github.com/osrf/subt), tiles by Open Robotics "
                  "on Gazebo Fuel, CC BY 4.0",
        "world": world, "bounds": [lo.round(2).tolist(), hi.round(2).tolist()], "base": base, "entrance": entrance,
        "entrance_route": entrance_route,       # straight legs from the base into the cave (outdoors: GPS)
        "base_station": station[:3, 3].round(3).tolist() + [yaw(station)],      # viewer model pose
        "staging_box": staging_box,
        # Collision-only boxes (no visuals): drones bump into them, LiDAR does not see them.
        "fences": [{"pose": T.round(4).tolist(), "size": s} for T, s in fence],
        "graph": {"nodes": nodes}, "mission": mission}, indent=1))
    (out / "truth.json").write_text(json.dumps({"artifacts": artifacts}, indent=1))
    export_assets()
    print(f"wrote {out}: {len(cave.faces)} triangles, extent {np.round(hi - lo, 1).tolist()} m, "
          f"{len(artifacts)} artifacts, base {np.round(base, 2).tolist()}")


# Viewer models for the artifacts and the base station (Open Robotics on Fuel, CC0 / CC BY 4.0).
ASSET_MODELS = {"survivor": "Rescue Randy Sitting", "backpack": "JanSport Backpack Red",
                "phone": "Samsung J8 Black", "helmet": "Climbing Helmet With Light",
                "rope": "Climbing Rope", "base_station": "Base Station"}


def export_assets(out=ROOT / "viewer" / "assets"):
    """One GLB per artifact type plus the base station, with their albedo textures."""
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    for key, name in ASSET_MODELS.items():
        d = model_dir(name)
        scene = trimesh.Scene()
        for vis in re.findall(r"<visual[^>]*>(.*?)</visual>", (d / "model.sdf").read_text(encoding="utf-8"), re.S):
            rel = re.search(r"<uri>\s*([^<]*\.dae)\s*</uri>", vis).group(1).rsplit("/", 1)[-1]
            pose = re.search(r"<pose>\s*([^<]*?)\s*</pose>", vis)
            T = pose_matrix(pose.group(1)) if pose else np.eye(4)
            albedo = re.search(r"<albedo_map>\s*([^<]*?)\s*</albedo_map>", vis)
            tex = d / "materials" / "textures" / albedo.group(1).rsplit("/", 1)[-1] if albedo else None
            if tex is None:
                # The COLLADA files point at their image themselves; trimesh does not always
                # follow, so look for <mesh name>.jpg or a *Diffuse*/*Albedo* image (no thermal).
                images = [f for f in (d / "materials").rglob("*") if f.suffix.lower() in (".jpg", ".png")
                          and "thermal" not in f.name.lower()]
                stem = Path(rel).stem.lower()
                tex = next((f for f in images if f.stem.lower() == stem), None) or                     next((f for f in images if any(k in f.name.lower() for k in ("diffuse", "albedo"))), None)
            mesh = trimesh.load(d / "meshes" / rel, force="scene")
            units = str(mesh.units).split(" ")[0]
            scale = float(units) if units.replace(".", "").isdigit() else 1.0
            for node in mesh.graph.nodes_geometry:
                M, gname = mesh.graph[node]
                g = mesh.geometry[gname].copy()
                g.apply_transform(M)
                g.apply_scale(scale)
                g.apply_transform(T)
                image = getattr(getattr(g.visual, "material", None), "image", None)
                if image is None and tex is not None and getattr(g.visual, "uv", None) is not None:
                    image = Image.open(tex)
                if image is not None:
                    g.visual = trimesh.visual.TextureVisuals(
                        uv=g.visual.uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=image))
                scene.add_geometry(g)
        scene.export(out / f"{key}.glb", include_normals=True)
        ext = scene.extents.round(2).tolist()
        print(f"  {key}: {name}, {sum(len(g.faces) for g in scene.geometry.values())} triangles, extent {ext} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world", help="world name under osrf/subt subt_ign/worlds, e.g. simple_cave_01")
    ap.add_argument("--out", help="output folder (default maps/<world>)")
    ap.add_argument("--survivor", help="make a rescue course around this survivor, e.g. rescue_randy_4")
    ap.add_argument("--difficulty", help="label shown with a rescue course, e.g. 하 / 상")
    args = ap.parse_args()
    import_world(args.world, args.out or ROOT / "maps" / args.world, args.survivor, args.difficulty)


if __name__ == "__main__":
    main()
