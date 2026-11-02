"""Builds drone_3d_model/drone.glb, the viewer's quadrotor, from scratch in Blender.
Made for this project, not derived from any third-party model; released under CC0 1.0.

    "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe" --background --factory-startup --python drone_3d_model/build_drone.py

The layout follows the collision-tolerant cave scouts of the DARPA SubT Challenge (e.g. CERBERUS's
RMF-Owl, arXiv 2202.11055): a quad-X inside a horizontal bumper ring, a vertical hoop over the top
and under the belly, frames cut from carbon-foam sandwich plate (black skins, tan core on the cut
edges), the front left open for the camera. Sizes follow cavesim/config.py (DroneParams.arm,
LidarParams.prop_z / prop_r / mount_z), in metres, Blender z-up with the nose on +x (the exporter
writes glTF y-up; the viewer turns it back). The LiDAR puck is not part of the model: the viewer
draws it where the simulator mounts it, under the belly, inside the hoop.

Each rotor is its own object named prop_ccw_* / prop_cw_* (seen from above), origin on the motor
shaft, so the viewer can spin it about its local up axis. PX4 quad-X order: front-right and
rear-left turn counter-clockwise.
"""
import math
from pathlib import Path

import bmesh
import bpy

ARM = 0.177              # m, motor to centre
PROP_R = 0.073           # m, blade tip radius (tips 0.25 m from the centre)
PROP_Z = 0.03            # m, rotor plane above the body centre
RING_R = 0.265           # m, bumper ring centre line, just outside the rotor tips
HOOP_Z = 0.19            # m, hoop half height: clears the LiDAR puck under the belly (mount_z -0.13)
PLATE = 0.010            # m, sandwich plate thickness
OUT = Path(__file__).resolve().parent / "drone.glb"


def material(name, rgb, metal=0.0, rough=0.5, emit=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    return m


def finish(obj, name, mats, bevel=0.0, smooth=True):
    obj.name = obj.data.name = name
    for m in mats if isinstance(mats, list) else [mats]:
        obj.data.materials.append(m)
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments = bevel, 3
    if smooth:
        bpy.ops.object.shade_smooth()
    return obj


def box(name, size, loc, mat, rot=(0, 0, 0), bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    bpy.context.object.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return finish(bpy.context.object, name, mat, bevel, smooth=bool(bevel))


def cyl(name, r, h, loc, mat, rot=(0, 0, 0), n=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=n, radius=r, depth=h, location=loc, rotation=rot)
    return finish(bpy.context.object, name, mat)


def band(name, loop, width, thick, axis, skin, core):
    """A closed band cut from sandwich plate. loop(u) gives (centre, in-plane outward normal) for
    u in [0, 1) (the normal need not be unit); the plate is `width` wide in the plane and `thick` along `axis` (its normal).
    The two broad faces are carbon skin, the cut edges show the foam core."""
    n = 96
    bm = bmesh.new()
    rings = []
    for k in range(n):
        c, o = loop(k / n)
        o = [x / math.hypot(*o) for x in o]
        rings.append([bm.verts.new([c[i] + o[i] * dw + axis[i] * dt for i in range(3)])
                      for dw, dt in ((width / 2, thick / 2), (-width / 2, thick / 2),
                                     (-width / 2, -thick / 2), (width / 2, -thick / 2))])
    for k in range(n):
        a, b = rings[k], rings[(k + 1) % n]
        for j in range(4):
            f = bm.faces.new((a[j], b[j], b[(j + 1) % 4], a[(j + 1) % 4]))
            f.material_index = 0 if j in (0, 2) else 1      # j = 0, 2: broad faces; 1, 3: cut edges
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    me.materials.append(skin)
    me.materials.append(core)
    return obj


def rotor(name, loc, ccw, mat):
    """Three twisted blades, origin on the shaft; the leading edge is higher, as on a real prop."""
    parts = []
    for k in range(3):
        a = 2 * math.pi * k / 3
        bpy.ops.mesh.primitive_cube_add(size=1)
        b = bpy.context.object
        b.scale = (PROP_R - 0.008, 0.016, 0.002)
        b.location = ((PROP_R + 0.008) / 2, 0, 0)
        b.rotation_euler = (math.radians(14) * (1 if ccw else -1), 0, 0)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        b.rotation_euler = (0, 0, a)
        bpy.ops.object.transform_apply(rotation=True)
        parts.append(b)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.010, depth=0.010)
    parts.append(bpy.context.object)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.007, radius2=0.002, depth=0.008, location=(0, 0, 0.009))
    parts.append(bpy.context.object)
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = parts[-1]
    bpy.ops.object.join()
    p = bpy.context.object
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")                # the cursor is at the shaft (world origin)
    p.location = loc
    return finish(p, name, mat)


bpy.ops.wm.read_factory_settings(use_empty=True)

carbon = material("carbon", (0.03, 0.032, 0.036), metal=0.2, rough=0.3)
foam = material("foam", (0.62, 0.48, 0.32), rough=0.85)                # tan core on the cut edges
accent = material("accent", (0.95, 0.40, 0.04), rough=0.4)             # front props and arm tips: heading
prop_rear = material("prop_rear", (0.70, 0.72, 0.75), rough=0.4)
motor = material("motor", (0.08, 0.08, 0.09), metal=0.8, rough=0.3)
bell = material("bell", (0.75, 0.76, 0.78), metal=0.9, rough=0.25)
board = material("board", (0.10, 0.25, 0.12), rough=0.6)
battery_m = material("battery", (0.12, 0.13, 0.15), rough=0.55)
alu = material("alu", (0.62, 0.64, 0.67), metal=0.85, rough=0.35)
lens = material("lens", (0.01, 0.01, 0.015), metal=0.5, rough=0.05)
lamp = material("lamp", (1.0, 0.96, 0.88), emit=6.0)

# Frame: centre plate and X arms out to the bumper ring, all sandwich plate.
box("centre_plate", (0.13, 0.10, PLATE), (0, 0, 0), carbon, bevel=0.002)
for name, ang in (("fr", -45), ("fl", 45), ("rl", 135), ("rr", -135)):
    a = math.radians(ang)
    box(f"arm_{name}", (RING_R, 0.022, PLATE), (RING_R / 2 * math.cos(a), RING_R / 2 * math.sin(a), 0), carbon,
        rot=(0, 0, a))
band("ring", lambda u: ((RING_R * math.cos(2 * math.pi * u), RING_R * math.sin(2 * math.pi * u), 0.0),
                        (math.cos(2 * math.pi * u), math.sin(2 * math.pi * u), 0.0)),
     0.024, PLATE, (0, 0, 1), carbon, foam)
# Hoop across the drone (y-z plane), the nose and tail left open for the camera.
band("hoop", lambda u: ((0.0, RING_R * math.cos(2 * math.pi * u), HOOP_Z * math.sin(2 * math.pi * u)),
                        (0.0, math.cos(2 * math.pi * u) / RING_R, math.sin(2 * math.pi * u) / HOOP_Z)),  # ellipse normal
     0.020, PLATE, (1, 0, 0), carbon, foam)

# Hull: battery under the plate, flight controller and companion computer stacked on top.
box("battery", (0.12, 0.045, 0.032), (-0.01, 0, -0.022), battery_m, bevel=0.004)
box("stack_fc", (0.040, 0.040, 0.004), (-0.005, 0, 0.012), board)
box("stack_pc", (0.080, 0.058, 0.004), (-0.01, 0, 0.026), board)
for sx in (-0.045, 0.025):
    for sy in (-0.024, 0.024):
        cyl(f"standoff_{sx}_{sy}", 0.0025, 0.026, (sx, sy, 0.018), alu, n=8)
# Front: camera block on the plate's nose, LED above it.
box("camera", (0.030, 0.030, 0.030), (0.085, 0, -0.012), alu, bevel=0.002)
cyl("camera_lens", 0.010, 0.016, (0.106, 0, -0.012), lens, rot=(0, math.pi / 2, 0))
box("led", (0.012, 0.030, 0.012), (0.075, 0, 0.012), alu, bevel=0.002)
cyl("led_face", 0.008, 0.003, (0.0815, 0, 0.012), lamp, rot=(0, math.pi / 2, 0))

# Motors and rotors on the X diagonals.
for name, ang, ccw in (("fr", -45, True), ("fl", 45, False), ("rl", 135, True), ("rr", -135, False)):
    a = math.radians(ang)
    cx, cy = ARM * math.cos(a), ARM * math.sin(a)
    cyl(f"motor_{name}", 0.0135, 0.014, (cx, cy, PLATE / 2 + 0.007), motor)
    cyl(f"bell_{name}", 0.014, 0.006, (cx, cy, PLATE / 2 + 0.017), bell)
    rotor(f"prop_{'ccw' if ccw else 'cw'}_{name}", (cx, cy, PROP_Z), ccw, accent if name[0] == "f" else prop_rear)

bpy.ops.export_scene.gltf(filepath=str(OUT), export_format="GLB", export_yup=True, export_apply=True)
print("wrote", OUT, OUT.stat().st_size, "bytes")
