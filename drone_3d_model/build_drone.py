"""Builds drone_3d_model/drone.glb, the viewer's quadrotor, from scratch in Blender.
Made for this project, not derived from any third-party model; released under CC0 1.0.

    "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe" --background --factory-startup --python drone_3d_model/build_drone.py

Sizes follow cavesim/config.py (DroneParams.arm, LidarParams.prop_z / prop_r), in metres, Blender
z-up with the nose on +x (the exporter writes glTF y-up; the viewer turns it back). The LiDAR puck
is not part of the model: the viewer draws it where the simulator mounts it.

Each rotor is its own object named prop_ccw_* / prop_cw_* (seen from above), origin on the motor
shaft, so the viewer can spin it about its local up axis. PX4 quad-X order: front-right and
rear-left turn counter-clockwise.
"""
import math
from pathlib import Path

import bpy

ARM = 0.177              # m, motor to centre
PROP_R = 0.073           # m, blade tip radius (tips 0.25 m from the centre)
PROP_Z = 0.03            # m, rotor plane above the body centre
GUARD_R = PROP_R + 0.008
OUT = Path(__file__).resolve().parent / "drone.glb"


def material(name, rgb, metal=0.0, rough=0.5, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    if emit:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emit
    return m


def finish(obj, name, mat, bevel=0.0):
    obj.name = obj.data.name = name
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments = bevel, 3
    bpy.ops.object.shade_smooth()
    return obj


def box(name, size, loc, mat, rot_z=0.0, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=(0, 0, rot_z))
    bpy.context.object.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return finish(bpy.context.object, name, mat, bevel)


def cyl(name, r, h, loc, mat, rot=(0, 0, 0), n=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=n, radius=r, depth=h, location=loc, rotation=rot)
    return finish(bpy.context.object, name, mat)


def rotor(name, loc, ccw, mat):
    """Two twisted blades, origin on the shaft. The leading edge is higher, as on a real prop."""
    blades = []
    for k in (0, 1):
        bpy.ops.mesh.primitive_cube_add(size=1)
        b = bpy.context.object
        b.scale = (PROP_R - 0.006, 0.014, 0.0025)
        b.location = ((PROP_R + 0.006) / 2 * (1 if k == 0 else -1), 0, 0)
        pitch = math.radians(12) * (1 if ccw else -1)
        b.rotation_euler = (pitch if k == 0 else -pitch, 0, 0)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        blades.append(b)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.009, depth=0.012)
    blades.append(bpy.context.object)
    for o in blades:
        o.select_set(True)
    bpy.context.view_layer.objects.active = blades[-1]
    bpy.ops.object.join()
    p = bpy.context.object
    p.location = loc
    return finish(p, name, mat)


bpy.ops.wm.read_factory_settings(use_empty=True)

carbon = material("carbon", (0.035, 0.037, 0.042), metal=0.3, rough=0.35)
shell = material("shell", (0.80, 0.82, 0.85), rough=0.4)
accent = material("accent", (0.95, 0.42, 0.05), rough=0.45)            # nose side: front arms and guards
motor = material("motor", (0.55, 0.57, 0.60), metal=0.9, rough=0.25)
prop_m = material("prop", (0.12, 0.13, 0.15), rough=0.6)
guard_m = material("guard", (0.20, 0.22, 0.25), rough=0.5)
lens = material("lens", (0.02, 0.02, 0.03), metal=0.5, rough=0.05)
lamp = material("lamp", (1.0, 0.95, 0.85), emit=4.0)

# Hull: carbon plate, battery and electronics under a light shell, FPV camera and lamps on the nose.
box("frame_plate", (0.17, 0.10, 0.008), (0, 0, 0.0), carbon, bevel=0.003)
box("battery", (0.13, 0.065, 0.032), (-0.005, 0, -0.022), carbon, bevel=0.006)
box("shell", (0.15, 0.085, 0.03), (-0.01, 0, 0.019), shell, bevel=0.012)
box("nose", (0.035, 0.05, 0.03), (0.09, 0, 0.012), shell, bevel=0.008)
cyl("camera_lens", 0.009, 0.006, (0.109, 0, 0.012), lens, rot=(0, math.pi / 2, 0))
for side in (1, -1):
    cyl(f"lamp_{'l' if side > 0 else 'r'}", 0.006, 0.004, (0.108, 0.017 * side, 0.012), lamp, rot=(0, math.pi / 2, 0))

# Arms, motors, guards and rotors on the X diagonals.
for name, ang, ccw in (("fr", -45, True), ("fl", 45, False), ("rl", 135, True), ("rr", -135, False)):
    a = math.radians(ang)
    cx, cy = ARM * math.cos(a), ARM * math.sin(a)
    front = name[0] == "f"
    box(f"arm_{name}", (ARM, 0.02, 0.012), (cx / 2, cy / 2, 0.0), accent if front else carbon, rot_z=a, bevel=0.004)
    cyl(f"motor_{name}", 0.014, 0.022, (cx, cy, PROP_Z - 0.016), motor)
    bpy.ops.mesh.primitive_torus_add(major_radius=GUARD_R, minor_radius=0.004, location=(cx, cy, PROP_Z),
                                     major_segments=48, minor_segments=8)
    finish(bpy.context.object, f"guard_{name}", accent if front else guard_m)
    for k in range(3):                                   # spokes from the motor bell to the ring
        s = math.radians(120 * k + ang + 60)
        box(f"guard_spoke_{name}_{k}", (GUARD_R - 0.012, 0.004, 0.003),
            (cx + (GUARD_R + 0.012) / 2 * math.cos(s), cy + (GUARD_R + 0.012) / 2 * math.sin(s), PROP_Z - 0.008),
            guard_m, rot_z=s)
    rotor(f"prop_{'ccw' if ccw else 'cw'}_{name}", (cx, cy, PROP_Z + 0.008), ccw, prop_m)

bpy.ops.export_scene.gltf(filepath=str(OUT), export_format="GLB", export_yup=True, export_apply=True)
print("wrote", OUT, OUT.stat().st_size, "bytes")
