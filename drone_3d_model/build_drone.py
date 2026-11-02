"""Builds drone_3d_model/drone.glb, the viewer's quadrotor, from scratch in Blender.
Made for this project, not derived from any third-party model; released under CC0 1.0.

    "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe" --background --factory-startup --python drone_3d_model/build_drone.py

A research quad-X of the kind university labs fly: carbon tube arms out from a stacked plate body,
a landing leg straight down from each motor, a stereo camera bar on the nose, the battery box under
the belly, two-blade props. No GPS mast: there is no GPS underground. Sizes follow cavesim/config.py
(DroneParams.arm, LidarParams.prop_z / prop_r), in metres, Blender z-up with the nose on +x (the
exporter writes glTF y-up; the viewer turns it back). The LiDAR puck is not part of the model: the
viewer draws it where the simulator mounts it, under the belly, and the legs reach just past it
(longer than the 0.18 m collision body: drawing only).

Each rotor is its own object named prop_ccw_* / prop_cw_* (seen from above), origin on the motor
shaft, so the viewer can spin it about its local up axis. PX4 quad-X order: front-right and
rear-left turn counter-clockwise. Red lights on the front arms, green on the rear, mark the heading.
"""
import math
from pathlib import Path

import bpy

ARM = 0.177              # m, motor to centre
PROP_R = 0.073           # m, blade tip radius (tips 0.25 m from the centre)
PROP_Z = 0.03            # m, rotor plane above the body centre
TUBE_R = 0.009           # m, arm tube radius
FOOT_Z = -0.17           # m, feet just below the LiDAR puck the viewer hangs at mount_z -0.13
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


def rotor(name, loc, ccw, mat):
    """Two twisted blades, origin on the shaft; the leading edge is higher, as on a real prop."""
    parts = []
    for k in range(2):
        a = math.pi * k
        bpy.ops.mesh.primitive_cube_add(size=1)
        b = bpy.context.object
        b.scale = (PROP_R - 0.008, 0.013, 0.002)
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
plastic = material("plastic", (0.06, 0.06, 0.065), rough=0.6)
rubber = material("rubber", (0.02, 0.02, 0.02), rough=0.9)
prop_m = material("prop", (0.05, 0.05, 0.055), rough=0.35)
motor = material("motor", (0.08, 0.08, 0.09), metal=0.8, rough=0.3)
bell = material("bell", (0.75, 0.76, 0.78), metal=0.9, rough=0.25)
board = material("board", (0.10, 0.25, 0.12), rough=0.6)
battery_m = material("battery", (0.12, 0.13, 0.15), rough=0.55)
alu = material("alu", (0.62, 0.64, 0.67), metal=0.85, rough=0.35)
cam_body = material("cam_body", (0.80, 0.81, 0.83), metal=0.6, rough=0.3)
lens = material("lens", (0.01, 0.01, 0.015), metal=0.5, rough=0.05)
red = material("led_red", (1.0, 0.05, 0.03), emit=8.0)
green = material("led_green", (0.1, 1.0, 0.2), emit=8.0)

# Body: two carbon plates on standoffs, the flight controller and companion computer stacked on top.
box("plate_low", (0.13, 0.13, 0.004), (0, 0, -0.008), carbon, bevel=0.002)
box("plate_high", (0.11, 0.11, 0.004), (0, 0, 0.012), carbon, bevel=0.002)
for sx in (-0.045, 0.045):
    for sy in (-0.045, 0.045):
        cyl(f"standoff_{sx}_{sy}", 0.0025, 0.020, (sx, sy, 0.002), alu, n=8)
box("stack_fc", (0.045, 0.045, 0.012), (-0.01, 0, 0.020), plastic, bevel=0.002)
box("stack_pc", (0.085, 0.060, 0.004), (-0.01, 0, 0.034), board)
box("stack_pc_sink", (0.050, 0.045, 0.010), (-0.015, 0, 0.041), alu, bevel=0.001)
box("battery", (0.11, 0.07, 0.045), (-0.005, 0, -0.035), battery_m, bevel=0.004)
# Nose: stereo camera bar on a bracket.
box("cam_bracket", (0.020, 0.040, 0.004), (0.060, 0, 0.016), alu)
box("cam_bar", (0.026, 0.120, 0.024), (0.072, 0, 0.030), cam_body, bevel=0.005)
for sy in (-0.040, 0.040):
    cyl(f"cam_lens_{sy}", 0.008, 0.004, (0.0855, sy, 0.030), lens, rot=(0, math.pi / 2, 0))

# Arms, motors, legs and rotors on the X diagonals.
for name, ang, ccw in (("fr", -45, True), ("fl", 45, False), ("rl", 135, True), ("rr", -135, False)):
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    cx, cy = ARM * c, ARM * s
    r0 = 0.055                                            # tube leaves the body here
    cyl(f"arm_{name}", TUBE_R, ARM - r0, ((ARM + r0) / 2 * c, (ARM + r0) / 2 * s, 0), carbon,
        rot=(0, math.pi / 2, a))
    box(f"arm_clamp_{name}", (0.022, 0.026, 0.024), (r0 * c, r0 * s, 0), plastic, rot=(0, 0, a), bevel=0.002)
    box(f"motor_mount_{name}", (0.034, 0.030, 0.012), (cx, cy, 0), plastic, rot=(0, 0, a), bevel=0.002)
    cyl(f"motor_{name}", 0.0135, 0.012, (cx, cy, 0.012), motor)
    cyl(f"bell_{name}", 0.014, 0.005, (cx, cy, 0.0205), bell)
    rotor(f"prop_{'ccw' if ccw else 'cw'}_{name}", (cx, cy, PROP_Z), ccw, prop_m)
    leg = (-0.006 - FOOT_Z - 0.012)                       # mount bottom to the foot pad top
    cyl(f"leg_{name}", 0.0065, leg, (cx, cy, -0.006 - leg / 2), carbon)
    cyl(f"foot_{name}", 0.010, 0.016, (cx, cy, FOOT_Z + 0.008), rubber)
    # Brace from the leg's middle back up to the arm, as on the folding legs of a lab quad.
    m, k = (cx, cy, -0.006 - leg / 2), 0.11
    t = (k * c, k * s, 0)
    d = [t[i] - m[i] for i in range(3)]
    ln = math.sqrt(sum(x * x for x in d))
    cyl(f"brace_{name}", 0.0045, ln, tuple((m[i] + t[i]) / 2 for i in range(3)), carbon,
        rot=(0, math.acos(d[2] / ln), a + math.pi))
    box(f"led_{name}", (0.010, 0.010, 0.006), ((ARM - 0.025) * c, (ARM - 0.025) * s, -0.011),
        red if name[0] == "f" else green)

bpy.ops.export_scene.gltf(filepath=str(OUT), export_format="GLB", export_yup=True, export_apply=True)
print("wrote", OUT, OUT.stat().st_size, "bytes")
