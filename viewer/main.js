// Cave Drone Sim viewer. Draws whatever the Python server sends; holds no sim logic.
// World frame is z-up (same as the simulator), so the camera uses z as "up".
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const COLORS = [0xff6b6b, 0x4dabf7, 0x69db7c, 0xffd43b, 0xda77f2, 0xff922b, 0x38d9a9, 0xf783ac, 0xa9e34b];
const $ = (id) => document.getElementById(id);
// Scales every light; the cave textures read near-black at 1. Override with ?light=N.
const LIGHT_GAIN = Number(new URLSearchParams(location.search).get("light")) || 5;

// Drone model (glTF, y-up): Sketchfab "Bionic dragonfly four-axis UAV aircraft" by boxgizmo
// (Sketchfab Standard licence, so not part of the repo). Without it the drones stay spheres.
const DRONE_MODEL = "/drone_3d_model/scene.gltf";
const DRONE_SPAN = 0.5;          // m, rotor tip to rotor tip after scaling
const MODEL_YAW = -Math.PI / 2;  // turns the model's nose onto body +x
const FPV_OFFSET = new THREE.Vector3(0.14, 0, 0.03);   // camera position in the body frame
// The model's two propeller meshes each hold one diagonal pair; the mirrored pair spins the
// other way, as on a real quadrotor. They are split into one pivot per rotor at load time.
// GLTFLoader sanitises node names ("Cong5_03 - Default_0" -> "Cong5_03_-_Default_0"), so match prefixes.
const PROP_MESHES = [[/^Cong5_03/, 1], [/^Symtrie1_03/, -1]];
const propSpin = (name) => (PROP_MESHES.find(([re]) => re.test(name)) || [null, 0])[1];
const PROP_SPEED = 45;           // rad/s on screen at hover (real rotors are far faster; this reads as spinning)
const MIX_TILT = 0.35;           // share of a rotor's thrust the roll/pitch demand may add or take
const MIX_YAW = 0.25;            // same for the yaw demand

// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.autoClear = false;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
const SKY = new THREE.Color(0x0d1014);
// Fog is always present so materials never recompile; the overview pushes it out of reach.
scene.fog = new THREE.Fog(0x000000, 1e5, 1e5 + 1);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.up.set(0, 0, 1);
camera.position.set(-30, -60, 70);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(40, 0, 0);
controls.update();

// First-person camera, rigidly mounted on the selected drone. Camera looks down its -z,
// so its basis in the body frame is right = -y, up = +z, back = -x.
const fpvCam = new THREE.PerspectiveCamera(78, 16 / 9, 0.03, 120);
const CAM_IN_BODY = new THREE.Quaternion().setFromRotationMatrix(
  new THREE.Matrix4().makeBasis(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)));
// Onboard LEDs: a wide flood plus a narrow spot, both bolted to the camera.
const flood = new THREE.SpotLight(0xfff4e0, 0, 45, Math.PI / 3, 0.8, 1.6);
const beam = new THREE.SpotLight(0xffffff, 0, 60, Math.PI / 9, 0.5, 1.4);
for (const l of [flood, beam]) {
  l.position.set(0, 0, -0.35);   // a little ahead of the nose, so the drone does not light itself
  l.target.position.set(0, 0, -1);
  fpvCam.add(l, l.target);
}
scene.add(fpvCam);

// Third-person chase camera: behind and above the selected drone, swinging with its heading.
const chaseCam = new THREE.PerspectiveCamera(70, 16 / 9, 0.05, 300);
chaseCam.up.set(0, 0, 1);
// Where the chase camera sits, relative to the drone's heading: drag to orbit, wheel to zoom,
// double-click to reset. yaw 0 = straight behind; pitch = elevation above the drone.
const CHASE_DEFAULT = { yaw: 0, pitch: Math.atan2(0.9, 2.2), dist: Math.hypot(2.2, 0.9) };
const chase = { ...CHASE_DEFAULT };
const chaseRay = new THREE.Raycaster();
// SubT worlds start outside the cave mouth: under open sky the camera views use daylight,
// under rock they use darkness and the drone's lights. `daylight` eases between 0 and 1.
const roofRay = new THREE.Raycaster(undefined, undefined, 0, 60);
// Upper-hemisphere directions for "how much sky can the camera see": straight up, 8 at 45 deg, 4 at 20 deg.
const SKY_DIRS = [new THREE.Vector3(0, 0, 1)];
for (let k = 0; k < 8; k++) SKY_DIRS.push(new THREE.Vector3().setFromSphericalCoords(1, Math.PI / 4, k * Math.PI / 4));
for (let k = 0; k < 4; k++) SKY_DIRS.push(new THREE.Vector3().setFromSphericalCoords(1, 7 * Math.PI / 18, k * Math.PI / 2 + Math.PI / 4));
SKY_DIRS.forEach((d) => d.set(d.x, d.z, d.y));   // three's spherical coords are y-up; ours are z-up
let skyOpen = 0, skyCheckIn = 0;
const DAY_SKY = new THREE.Color(0x9fb3c8);
let daylight = 0;

const hemi = new THREE.HemisphereLight(0xcfd8e3, 0x302820, 1.4 * LIGHT_GAIN);
const sun = new THREE.DirectionalLight(0xffffff, 1.2 * LIGHT_GAIN);
sun.position.set(30, -40, 80);
scene.add(hemi, sun);

addEventListener("resize", () => {
  renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- state
const caveMeshes = [];      // {mesh, over, fpv}: overview (see-through) and FPV (solid) materials
const caveMats = [];        // see-through materials, driven by the opacity slider
const artifactMeshes = {};
const drones = [];          // per drone: {anchor, marker, craft, ghost, lidar, trail, odomTrail, path, goal, pts, pose}
const overlays = [];        // things the FPV view hides (trails, debug, links, ...)
let linkLines = null;
let selected = 0;
let initMsg = null;
let rescue = null;          // mission settings of a rescue course, else null
let lastState = null;
let ws = null;
let droneModel = null;
let lidarMount = [0, 0, -0.13];   // sensor origin in the body frame (m), from the init message
let stateDt = 100;          // ms between state messages, smoothed
let lastStateAt = 0;

const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const EULER = new THREE.Euler(0, 0, 0, "ZYX");
const quatOf = (yaw, rp) => new THREE.Quaternion().setFromEuler(EULER.set(rp ? rp[0] : 0, rp ? rp[1] : 0, yaw, "ZYX"));

function makeLine(color, dashed = false, opacity = 1) {
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.6, gapSize: 0.4, transparent: true, opacity })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(new THREE.BufferGeometry(), mat);
  line.frustumCulled = false;
  scene.add(line);
  overlays.push(line);
  return line;
}

function setLine(line, pts) {
  const flat = new Float32Array(pts.length * 3);
  pts.forEach((p, i) => flat.set(p, i * 3));
  line.geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
  line.geometry.computeBoundingSphere();
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}

// Split a mesh holding two rotors into two pivots, each spinning about its own hub (mesh-local y).
function splitRotors(mesh, dir) {
  const geo = mesh.geometry.index ? mesh.geometry.toNonIndexed() : mesh.geometry;
  const pos = geo.attributes.position;
  const nTri = pos.count / 3;
  const centre = new THREE.Vector3(), tri = [];
  for (let t = 0; t < nTri; t++) {
    const c = new THREE.Vector3();
    for (let k = 0; k < 3; k++) c.add(new THREE.Vector3().fromBufferAttribute(pos, 3 * t + k));
    tri.push(c.divideScalar(3));
    centre.add(c);
  }
  centre.divideScalar(nTri);
  // The two rotors sit on a diagonal: split along the direction from the centre to the farthest triangle.
  const axis = tri.reduce((a, c) => (c.distanceTo(centre) > a.distanceTo(centre) ? c : a)).clone().sub(centre);
  const groups = [[], []];
  tri.forEach((c, t) => groups[c.clone().sub(centre).dot(axis) > 0 ? 0 : 1].push(t));
  for (const ids of groups) {
    const g = new THREE.BufferGeometry();
    for (const name of Object.keys(geo.attributes)) {
      const src = geo.attributes[name], n = src.itemSize;
      const out = new Float32Array(ids.length * 3 * n);
      ids.forEach((t, i) => { for (let k = 0; k < 3 * n; k++) out[i * 3 * n + k] = src.array[3 * t * n + k]; });
      g.setAttribute(name, new THREE.BufferAttribute(out, n));
    }
    // Hub = area-weighted centroid: a three-blade rotor is symmetric about its shaft, so this lands
    // exactly on the motor axis (its bounding-box centre is off by ~4 % of the span).
    const hub = new THREE.Vector3(), a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    let area = 0;
    const P = g.attributes.position;
    for (let t = 0; t < P.count; t += 3) {
      a.fromBufferAttribute(P, t); b.fromBufferAttribute(P, t + 1); c.fromBufferAttribute(P, t + 2);
      const w = b.clone().sub(a).cross(c.clone().sub(a)).length();
      hub.addScaledVector(a.clone().add(b).add(c), w / 3);
      area += w;
    }
    hub.divideScalar(area);
    g.translate(-hub.x, -hub.y, -hub.z);
    const pivot = new THREE.Object3D();
    pivot.position.copy(hub);
    pivot.userData.spin = dir;
    pivot.add(new THREE.Mesh(g, mesh.material));
    mesh.add(pivot);
  }
  mesh.geometry = new THREE.BufferGeometry();     // the pivots carry the rotors now
}

function loadDroneModel() {
  new GLTFLoader().load(DRONE_MODEL, (gltf) => {
    const m = gltf.scene;
    const props = [];
    m.traverse((o) => { if (o.isMesh && propSpin(o.name)) props.push(o); });
    props.forEach((o) => splitRotors(o, propSpin(o.name)));
    if (props.length !== 2) console.warn(`drone model: found ${props.length} of 2 propeller meshes; rotors will not spin`);
    console.log(`drone model: ${props.length} propeller meshes split into ${2 * props.length} rotors`);
    m.rotation.x = Math.PI / 2;                 // glTF y-up -> z-up
    const wrap = new THREE.Group();
    wrap.add(m);
    wrap.rotation.z = MODEL_YAW;
    const holder = new THREE.Group();
    holder.add(wrap);
    holder.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(holder);
    const size = box.getSize(new THREE.Vector3());
    const s = DRONE_SPAN / Math.max(size.x, size.y);
    holder.scale.setScalar(s);
    holder.position.copy(box.getCenter(new THREE.Vector3()).multiplyScalar(-s));
    droneModel = holder;
    drones.forEach(attachModel);
  }, undefined, () => console.warn(`drone model not found at ${DRONE_MODEL}; drawing spheres`));
}

function attachModel(g) {
  if (!droneModel || g.craft.children.length) return;
  const m = droneModel.clone();
  g.rotors = [];
  m.traverse((o) => { if (o.userData.spin) g.rotors.push(o); });
  g.craft.add(m);
  g.anchor.updateMatrixWorld(true);
  const toBody = g.anchor.matrixWorld.clone().invert();
  const arm = g.rotors.map((r) => r.getWorldPosition(new THREE.Vector3()).applyMatrix4(toBody));
  const span = Math.max(...arm.map((p) => Math.hypot(p.x, p.y)), 1e-6);
  g.rotors.forEach((r, k) => { r.userData.x = arm[k].x / span; r.userData.y = arm[k].y / span; r.userData.w = 1; });
  // Status LED in the drone's colour, so several drones stay apart up close.
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.025, 10, 8),
                             new THREE.MeshBasicMaterial({ color: g.color }));
  led.position.set(-0.12, 0, 0.06);
  g.craft.add(led);
  // The LiDAR where the simulator mounts it: an Ouster OS0 puck (87 mm across, 58 mm tall)
  // with its dark optical window band.
  const puck = new THREE.Group();
  const shell = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, metalness: 0.6, roughness: 0.4 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.0435, 0.0435, 0.058, 24), shell);
  const window_ = new THREE.Mesh(new THREE.CylinderGeometry(0.0440, 0.0440, 0.026, 24),
                                 new THREE.MeshStandardMaterial({ color: 0x111418, metalness: 0.2, roughness: 0.15 }));
  body.rotation.x = window_.rotation.x = Math.PI / 2;    // cylinders stand on three's y; the body frame is z-up
  puck.add(body, window_);
  puck.position.set(...lidarMount);
  // Standoff from the frame to the puck (the drawn frame is thinner than the 0.18 m hull).
  const gap = Math.abs(lidarMount[2]) - 0.029;
  if (gap > 0.005) {
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, gap, 10), shell);
    stem.rotation.x = Math.PI / 2;
    stem.position.set(lidarMount[0], lidarMount[1], Math.sign(lidarMount[2]) * gap / 2);
    g.craft.add(stem);
  }
  g.craft.add(puck);
}

// SubT maps ship PBR rock textures in maps/<name>/textures/<set>_{Albedo,Normal,Roughness}.jpg,
// and their cave.glb names each mesh's material after its texture set.
const ROCK_TEXTURES = ["CaveWall", "StriatedRock", "RockPile", "Grass", "Gravel"];
const texLoader = new THREE.TextureLoader();

function rockMaterials(map, set) {
  const tex = (kind, srgb) => {
    const t = texLoader.load(`/maps/${map}/textures/${set}_${kind}.jpg`);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;           // tile UVs run well past 1
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  const maps = { map: tex("Albedo", true), normalMap: tex("Normal"), roughnessMap: tex("Roughness") };
  const over = new THREE.MeshStandardMaterial({
    ...maps, metalness: 0, side: THREE.DoubleSide, fog: false,
    transparent: true, opacity: $("opacity").value / 100, depthWrite: $("opacity").value > 90,
  });
  caveMats.push(over);
  // Camera views draw front faces only, as Gazebo does: the tiles are hollow tubes whose faces
  // point inward, so from the staging area outside their shells stay invisible instead of black.
  return { over, fpv: new THREE.MeshStandardMaterial({ ...maps, metalness: 0, side: THREE.FrontSide }) };
}

// Artifact and base-station models (viewer/assets/<type>.glb, from cavesim.subt_import;
// Open Robotics on Gazebo Fuel, CC0 / CC BY 4.0). Missing files fall back to simple shapes.
const assetCache = {};
function loadAsset(name, onLoad) {
  assetCache[name] ??= new Promise((res) =>
    new GLTFLoader().load(`/viewer/assets/${name}.glb`, (g) => res(g.scene), undefined, () => res(null)));
  assetCache[name].then((m) => m && onLoad(m.clone()));
}

function onInit(msg) {
  if (msg.lidar_mount) lidarMount = msg.lidar_mount;
  initMsg = msg;
  $("mapname").textContent = msg.map;
  new GLTFLoader().load(`/maps/${msg.map}/cave.glb`, (gltf) => {
    for (const c of caveMeshes) scene.remove(c.mesh);
    caveMeshes.length = caveMats.length = 0;
    const box = new THREE.Box3();
    const textured = {};          // SubT maps: one textured material pair per rock texture set
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
      const set = o.material?.name;
      if (ROCK_TEXTURES.includes(set)) {
        textured[set] ??= rockMaterials(msg.map, set);
        o.material = textured[set].over;
        caveMeshes.push({ mesh: o, ...textured[set] });
      } else {
        const spike = /speleo/i.test(o.name) || /speleo/i.test(o.parent?.name || "");
        const over = new THREE.MeshStandardMaterial({
          color: spike ? 0xe8dcc4 : 0xc9b496, roughness: 0.9, side: THREE.DoubleSide, fog: false,
          transparent: true, opacity: $("opacity").value / 100, depthWrite: false,
        });
        const fpv = new THREE.MeshStandardMaterial({
          color: spike ? 0xd9ccb2 : 0x8f7e69, roughness: spike ? 0.55 : 0.95, metalness: 0,
          side: THREE.DoubleSide,
        });
        o.material = over;
        caveMats.push(over);
        caveMeshes.push({ mesh: o, over, fpv });
      }
      box.expandByObject(o);
    });
    scene.add(gltf.scene);
    caveMeshes.forEach((c) => (c.root = gltf.scene));
    const c = box.getCenter(new THREE.Vector3());
    controls.target.copy(c);
    camera.position.set(c.x - 0.4 * (box.max.x - box.min.x), c.y - 0.9 * (box.max.y - box.min.y), c.z + 70);
    controls.update();
  });

  // Base station (tent + antenna where the world puts it; a puck at the base otherwise) and home circle
  if (msg.base_station) {
    loadAsset("base_station", (m) => {
      m.position.copy(v3(msg.base_station));
      m.rotation.z = msg.base_station[3];
      scene.add(m);
    });
  } else {
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 0.3, 24),
                                new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x333333 }));
    base.rotation.x = Math.PI / 2;
    base.position.copy(v3(msg.base));
    scene.add(base);
  }
  const ring = new THREE.Mesh(new THREE.RingGeometry(msg.home_radius - 0.08, msg.home_radius, 64),
                              new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.5 }));
  ring.position.copy(v3(msg.base)).add(new THREE.Vector3(0, 0, -2.5));
  scene.add(ring);
  overlays.push(ring);

  // Rescue course: relabel the HUD and mark the survivor with a beacon visible from afar.
  rescue = msg.mission?.type === "rescue" ? msg.mission : null;
  $("lScore").textContent = rescue ? "랩 타임" : "점수";
  $("lArts").textContent = rescue ? "구조" : "목표물";
  $("lCov").textContent = rescue ? "난이도" : "탐사율";
  if (rescue) {
    const t = msg.artifacts[rescue.target];
    const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 40, 16, 1, true),
                                  new THREE.MeshBasicMaterial({ color: 0xffa94d, transparent: true, opacity: 0.25,
                                                                side: THREE.DoubleSide, depthWrite: false }));
    beacon.rotation.x = Math.PI / 2;
    beacon.position.copy(v3(t.pos)).add(new THREE.Vector3(0, 0, 20));
    scene.add(beacon);
    overlays.push(beacon);
  }

  for (const a of msg.artifacts) {
    // Like the drones: the real model up close and in the camera views, a marker from afar.
    const group = new THREE.Group();
    group.position.copy(v3(a.pos));
    group.rotation.z = a.yaw || 0;
    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.45),
                                  new THREE.MeshStandardMaterial({ color: 0x868e96, emissive: 0x222222 }));
    marker.position.z = 0.6;
    const art = { group, marker, model: null };
    group.add(marker);
    loadAsset(a.type, (m) => {
      // Stand the model on its lowest point: the artifact's position is where it touches the floor,
      // and some models (the survivor) have their origin well below their feet.
      m.position.z -= new THREE.Box3().setFromObject(m).min.z;
      art.model = m;
      group.add(m);
    });
    group.userData = a;
    scene.add(group);
    artifactMeshes[a.id] = art;
  }

  for (let i = 0; i < msg.n_drones; i++) {
    const color = COLORS[i % COLORS.length];
    const anchor = new THREE.Group();          // true pose; the model and the marker hang off it
    const craft = new THREE.Group();
    // Far away the 0.5 m model is invisible, so a coloured marker with a heading cone takes over.
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14),
                                  new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4 }));
    const heading = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.7, 10),
                                   new THREE.MeshBasicMaterial({ color }));
    heading.geometry.rotateZ(-Math.PI / 2);
    heading.geometry.translate(0.6, 0, 0);
    marker.add(heading);
    anchor.add(craft, marker);
    const ghost = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 8),
                                 new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.6 }));
    const lidar = new THREE.Points(new THREE.BufferGeometry(),
                                   new THREE.PointsMaterial({ color, size: 0.12, transparent: true, opacity: 0.7 }));
    lidar.frustumCulled = false;
    const pts = new THREE.Points(new THREE.BufferGeometry(),
                                 new THREE.PointsMaterial({ color: 0xffffff, size: 0.5 }));
    pts.frustumCulled = false;
    const goal = new THREE.Mesh(new THREE.SphereGeometry(0.2, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    // The agent's own map (rock it has seen), as the agent sends it: cyan dots.
    const map = new THREE.Points(new THREE.BufferGeometry(),
                                 new THREE.PointsMaterial({ color: 0x3bc9db, size: 0.35, transparent: true, opacity: 0.55 }));
    map.frustumCulled = false;
    // A tall beam over the drone, so it can be found on the whole-cave map (like the survivor's).
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 30, 12, 1, true),
                                new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.45,
                                                              side: THREE.DoubleSide, depthWrite: false }));
    beam.rotation.x = Math.PI / 2;
    scene.add(anchor, ghost, lidar, pts, goal, map, beam);
    overlays.push(ghost, lidar, pts, goal, map, beam);
    const g = { anchor, craft, marker, ghost, lidar, pts, goal, map, beam, color, pose: null,
                trail: makeLine(color, false, 0.9), odomTrail: makeLine(color, true, 0.6),
                path: makeLine(0xffffff, false, 0.5) };
    drones.push(g);
    attachModel(g);
  }
  linkLines = new THREE.LineSegments(new THREE.BufferGeometry(),
                                     new THREE.LineBasicMaterial({ color: 0x51cf66, transparent: true, opacity: 0.7 }));
  linkLines.frustumCulled = false;
  scene.add(linkLines);
  overlays.push(linkLines);
  buildDroneHud(msg.n_drones);
}

function buildDroneHud(n) {
  const box = $("drones");
  box.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const el = document.createElement("div");
    el.className = "drone" + (i === selected ? " sel" : "");
    el.innerHTML = `<div class="row"><span><span class="dot" style="background:#${COLORS[i % COLORS.length].toString(16).padStart(6, "0")}"></span>드론 ${i}</span>
      <span class="st muted"></span></div><div class="row muted"><span class="mode"></span><span class="drift"></span></div>
      <div class="bar"><div class="batt"></div></div>`;
    el.onclick = () => { selected = i; [...box.children].forEach((c, k) => c.classList.toggle("sel", k === i)); };
    box.appendChild(el);
  }
}

const STATUS_KO = { flying: "비행", landing: "착륙 중", landed: "착륙", crashed: "충돌", dead: "배터리 방전",
                    out: "코스 이탈" };

// ---------------------------------------------------------------- replay
// Every state message is kept (LiDAR thinned) so the run can be scrubbed back and replayed.
const history = [];
let replayAt = null;        // index into history while replaying, null when live
let replayPlaying = false;
let scrubbing = false;      // the replay bar's thumb is held
const REPLAY_LIDAR_PTS = 600;

function remember(msg) {
  const slim = { ...msg, drones: msg.drones.map((d) => {
    if (!d.lidar) return d;
    const step = Math.max(1, Math.ceil(d.lidar.length / 3 / REPLAY_LIDAR_PTS));
    const pts = [];
    for (let k = 0; k < d.lidar.length; k += 3 * step) pts.push(d.lidar[k], d.lidar[k + 1], d.lidar[k + 2]);
    return { ...d, lidar: pts };
  }) };
  history.push(slim);
}

function onHistory(msg) {
  // The server's record of the run so far (a reload or a late join): LiDAR and debug come every
  // few ticks, so carry each forward to the ticks between.
  history.length = 0;
  let last = [];
  for (const st of msg.states) {
    st.drones.forEach((d, i) => {
      if (d.lidar) last[i] = { lidar: d.lidar, debug: d.debug };
      else if (last[i]) { d.lidar = last[i].lidar; d.debug = last[i].debug; }
    });
    remember(st);
  }
  updateReplayBar();
}

function trailsUpTo(k) {
  // Rebuild trails from the stored states (every 5th, like the server's trails).
  const tru = drones.map(() => []), odo = drones.map(() => []);
  for (let j = 0; j <= k; j += 5) {
    history[j].drones.forEach((d, i) => { tru[i].push(d.pos); if (d.odom) odo[i].push(d.odom); });
  }
  onTrails({ true: tru, odom: odo });
}

function showReplay(k, instant = true) {
  replayAt = Math.max(0, Math.min(k, history.length - 1));
  applyState(history[replayAt], instant);
  trailsUpTo(replayAt);
  updateReplayBar();
}

function goLive() {
  replayAt = null;
  replayPlaying = false;
  if (history.length) applyState(history[history.length - 1], true);
  updateReplayBar();
}

function updateReplayBar() {
  const bar = $("rpBar");
  bar.max = Math.max(history.length - 1, 0);
  const k = replayAt == null ? history.length - 1 : replayAt;
  if (!scrubbing) bar.value = k;
  const t = history[k] ? history[k].t : 0, end = history.length ? history[history.length - 1].t : 0;
  $("rpTime").textContent = replayAt == null ? `${end.toFixed(1)} s` : `${t.toFixed(1)} / ${end.toFixed(1)} s`;
  $("rpLive").className = replayAt == null ? "live" : "";
  $("rpPlay").textContent = replayPlaying ? "❚❚" : "▶";
}

// Live, every state (10 a second) put the thumb back at the end and the drag never took:
// holding the thumb leaves live at once, and nothing moves it while it is held.
$("rpBar").addEventListener("pointerdown", () => {
  scrubbing = true;
  replayPlaying = false;
  if (replayAt == null) replayAt = Math.max(history.length - 1, 0);
});
addEventListener("pointerup", () => { scrubbing = false; });
$("rpBar").oninput = (e) => { replayPlaying = false; showReplay(Number(e.target.value)); };
$("rpLive").onclick = goLive;
$("rpPlay").onclick = () => {
  if (replayAt == null) showReplay(history.length - 1);
  replayPlaying = !replayPlaying;
  updateReplayBar();
};
setInterval(() => {
  // Replay at the recorded pace (one state per 0.1 s of sim time); catching up means live again.
  if (replayAt == null || !replayPlaying) return;
  if (replayAt >= history.length - 1) { goLive(); return; }
  showReplay(replayAt + Number($("rpSpeed").value), false);
}, 100);

function onState(msg) {
  if (!initMsg) return;
  remember(msg);
  if (replayAt == null) applyState(msg, false);
  updateReplayBar();
}

function applyState(msg, instant) {
  const now = performance.now();
  if (lastStateAt && !instant) stateDt += (Math.min(Math.max(now - lastStateAt, 20), 500) - stateDt) * 0.2;
  lastStateAt = now;
  lastState = msg;
  $("time").textContent = `${msg.t.toFixed(1)} / ${initMsg.mission_time_s} s`;
  if (rescue) {
    const s = msg.score, saved = Object.entries(s.rescued_at_s || {});
    $("score").textContent = s.lap_time_s != null ? `${s.lap_time_s.toFixed(1)} s` : "-";
    $("arts").textContent = saved.length ? `드론 ${saved[0][0]} · ${saved[0][1].toFixed(1)} s` : "아직";
    $("cov").textContent = rescue.difficulty || "-";
  } else {
    $("score").textContent = msg.score.score;
    $("arts").textContent = `${msg.score.artifacts_found} / ${msg.score.artifacts_total}`;
    $("cov").textContent = `${(100 * msg.score.coverage).toFixed(0)} %`;
  }

  const show = (id) => $(id).checked;
  msg.drones.forEach((d, i) => {
    const g = drones[i];
    // Interpolate from wherever the drone is drawn now to the new pose over one message interval.
    const target = { pos: v3(d.pos), q: quatOf(d.yaw, d.rp) };
    g.pose = g.pose && !instant ? { from: { pos: g.anchor.position.clone(), q: g.anchor.quaternion.clone() }, to: target, t0: now }
                                : { from: target, to: target, t0: now };
    g.marker.material.emissiveIntensity = ["crashed", "dead", "out"].includes(d.status) ? 0 : 0.4;
    g.status = d.status;
    g.mix = d.mix;
    if (d.odom) g.ghost.position.copy(v3(d.odom));
    g.ghost.visible = show("cOdom") && !!d.odom;
    if (d.lidar && show("cLidar")) {
      g.lidar.geometry.setAttribute("position", new THREE.Float32BufferAttribute(d.lidar, 3));
      g.lidar.visible = true;
    } else {
      g.lidar.visible = false;
    }
    const dbg = d.debug || {};
    const dv = show("cDebug");
    if (dbg.path && dbg.path.length) { setLine(g.path, [d.odom || d.pos, ...dbg.path.slice().reverse()]); }
    g.path.visible = dv && !!(dbg.path && dbg.path.length);
    g.goal.visible = dv && !!dbg.goal;
    if (dbg.goal) g.goal.position.copy(v3(dbg.goal));
    if (dbg.points && dbg.points.length) {
      g.pts.geometry.setAttribute("position", new THREE.Float32BufferAttribute(dbg.points.flat(), 3));
    }
    g.pts.visible = dv && !!(dbg.points && dbg.points.length);
    if (dbg.map) g.map.geometry.setAttribute("position", new THREE.Float32BufferAttribute(dbg.map.flat(), 3));
    g.map.visible = show("cMap") && !!g.map.geometry.attributes.position;
    g.beam.position.copy(v3(d.pos)).add(new THREE.Vector3(0, 0, 15));
    g.beam.visible = i === selected;

    const el = $("drones").children[i];
    el.querySelector(".st").textContent = STATUS_KO[d.status] || d.status;
    el.querySelector(".mode").textContent = dbg.mode ? `모드: ${dbg.mode}` : "";
    el.querySelector(".drift").textContent = d.odom
      ? `추정 오차 ${Math.hypot(d.pos[0] - d.odom[0], d.pos[1] - d.odom[1], d.pos[2] - d.odom[2]).toFixed(2)} m`
      : "오도메트리 꺼짐";
    const b = el.querySelector(".batt");
    b.style.width = `${100 * d.battery}%`;
    b.style.background = d.battery > 0.4 ? "#51cf66" : d.battery > 0.2 ? "#fcc419" : "#ff6b6b";
  });

  const segs = [];
  for (const [a, b] of msg.links) segs.push(...msg.drones[a].pos, ...msg.drones[b].pos);
  for (const a of msg.base_links) segs.push(...msg.drones[a].pos, ...initMsg.base);
  linkLines.geometry.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
  linkLines.visible = show("cLinks");

  for (const [id, a] of Object.entries(artifactMeshes)) {
    const found = msg.found.includes(Number(id));
    a.marker.material.color.setHex(found ? 0x51cf66 : 0x868e96);
    a.group.visible = found || show("cTruth");
  }
  updateFpvHud();
}

function onTrails(msg) {
  msg.true.forEach((t, i) => { if (drones[i]) setLine(drones[i].trail, t); });
  msg.odom.forEach((t, i) => { if (drones[i]) setLine(drones[i].odomTrail, t); });
  drones.forEach((g) => { g.trail.visible = $("cTrail").checked; g.odomTrail.visible = $("cTrail").checked && $("cOdom").checked; });
}

function onDone(msg) {
  $("final").style.display = "block";
  $("finalText").textContent = JSON.stringify(msg.score, null, 1);
}

// ---------------------------------------------------------------- views
// The main window and the inset each show one view: "map" (orbit camera over the whole cave),
// "chase" (third person behind the selected drone) or "fpv" (its onboard camera).
const mainView = () => $("mainView").value;
const insetView = () => $("insetView").value;
const inCave = (v) => v === "chase" || v === "fpv";

function updateFpvHud() {
  const d = lastState && lastState.drones[selected];
  const v = inCave(mainView()) ? mainView() : insetView();
  $("fpvhud").style.display = inCave(v) && drones[selected] ? "block" : "none";
  if (!d) return;
  const speed = Math.hypot(...(d.vel || [0, 0, 0]));
  const gust = Math.hypot(...(d.aero || [0, 0, 0]));
  $("fpvTitle").textContent = `${v === "fpv" ? "1인칭" : "3인칭"} · 드론 ${selected}`;
  $("fpvSpd").textContent = `${speed.toFixed(2)} m/s`;
  $("fpvClr").textContent = d.clear == null ? "> 1.4 m" : `${d.clear.toFixed(2)} m`;
  $("fpvClr").className = d.clear != null && d.clear < 0.4 ? "warn" : "";
  $("fpvGust").textContent = `${gust.toFixed(2)} m/s²`;
  $("fpvAtt").textContent = `R ${(d.rp[0] * 57.3).toFixed(0)}° P ${(d.rp[1] * 57.3).toFixed(0)}°`;
  $("fpvBat").textContent = `${(100 * d.battery).toFixed(0)} %`;
  $("fpvHorizonBox").style.display = v === "fpv" ? "block" : "none";
  const dead = ["crashed", "dead", "out"].includes(d.status);
  $("fpvAlert").textContent = dead ? (STATUS_KO[d.status] || d.status) : "";
  // Where the survivor is from here -- the coordinates every agent is given, so a pilot gets them too.
  // ... and home, for the way back.
  if (rescue && initMsg) {
    const arrow = (name, t) => {
      const dx = t[0] - d.pos[0], dy = t[1] - d.pos[1], dz = t[2] - d.pos[2];
      const rel = Math.atan2(dy, dx) - d.yaw;               // bearing off the nose, counter-clockwise
      return `${name} <span style="display:inline-block;transform:rotate(${-rel}rad);font-size:20px">⬆</span> ` +
        `${Math.hypot(dx, dy, dz).toFixed(0)} m · 높이 ${dz >= 0 ? "+" : ""}${dz.toFixed(0)} m`;
    };
    $("fpvGoal").innerHTML = arrow("생존자", initMsg.artifacts[rescue.target].pos) + "<br>" + arrow("홈", initMsg.base);
  }
}

function insetRect() {
  const w = Math.min(560, Math.round(innerWidth * 0.38)), h = Math.round(w * 9 / 16);
  return { x: innerWidth - w - 10, y: 44, w, h };
}

function placeCameras(dt) {
  const g = drones[selected];
  if (!g) return;
  // Onboard camera (and the drone's lights, which hang off it) follow the selected drone.
  fpvCam.position.copy(FPV_OFFSET).applyQuaternion(g.anchor.quaternion).add(g.anchor.position);
  fpvCam.quaternion.copy(g.anchor.quaternion).multiply(CAM_IN_BODY);
  // Chase camera: heading-only frame, eased so it swings smoothly, pulled in front of rock.
  const yaw = new THREE.Euler().setFromQuaternion(g.anchor.quaternion, "ZYX").z;
  const look = g.anchor.position.clone().add(new THREE.Vector3(0, 0, 0.15));
  // Up against a ceiling the view from above-behind is rock: the camera ended up inside it and showed
  // a grey screen. Try the chosen elevation, then level, then from below; take the first with most
  // of the distance free, else the one with the most room, and stop short of the rock.
  const rock = caveMeshes.map((c) => c.mesh);
  let want = null, best = null;
  for (const pitch of [chase.pitch, 0, -0.3]) {
    const c = Math.cos(pitch);
    const dir = new THREE.Vector3(-c * Math.cos(chase.yaw), -c * Math.sin(chase.yaw), Math.sin(pitch))
      .applyAxisAngle(new THREE.Vector3(0, 0, 1), yaw);
    chaseRay.set(look, dir);
    chaseRay.far = chase.dist;
    const hit = chaseRay.intersectObjects(rock, false)[0];
    const free = hit ? hit.distance : chase.dist;
    if (!best || free > best.free) best = { dir, free };
    if (free >= 0.8 * chase.dist) { want = look.clone().addScaledVector(dir, free < chase.dist ? free - 0.25 : free); break; }
  }
  if (!want) want = look.clone().addScaledVector(best.dir, Math.max(best.free - 0.25, 0.1));
  // Ease when following; jump when far (first use, or another drone was selected).
  if (chaseCam.position.distanceTo(want) > Math.max(5, 2 * chase.dist)) chaseCam.position.copy(want);
  else chaseCam.position.lerp(want, 1 - Math.exp(-dt * 6));
  chaseCam.lookAt(look);
}

// Mouse input goes to whichever view is under the pointer: the orbit controls for the map,
// our own orbit around the drone for a chase view, nothing for the onboard camera.
function viewAt(x, y) {
  const r = insetRect();
  const inInset = insetView() !== "off" && x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
  return inInset ? insetView() : mainView();
}
let chaseDrag = null;
const canvas = renderer.domElement;
canvas.addEventListener("pointermove", (e) => {
  if (chaseDrag) {
    chase.yaw -= (e.clientX - chaseDrag.x) * 0.006;
    chase.pitch = THREE.MathUtils.clamp(chase.pitch + (e.clientY - chaseDrag.y) * 0.006, -0.35, 1.45);
    chaseDrag = { x: e.clientX, y: e.clientY };
  } else {
    controls.enabled = viewAt(e.clientX, e.clientY) === "map";
  }
}, { capture: true });
canvas.addEventListener("pointerdown", (e) => {
  controls.enabled = viewAt(e.clientX, e.clientY) === "map";
  if (viewAt(e.clientX, e.clientY) === "chase") {
    chaseDrag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  }
}, { capture: true });
canvas.addEventListener("pointerup", () => { chaseDrag = null; }, { capture: true });
canvas.addEventListener("wheel", (e) => {
  if (viewAt(e.clientX, e.clientY) !== "chase") return;
  e.preventDefault();
  chase.dist = THREE.MathUtils.clamp(chase.dist * Math.exp(e.deltaY * 0.0012), 0.6, 15);
}, { capture: true, passive: false });
canvas.addEventListener("dblclick", (e) => {
  if (viewAt(e.clientX, e.clientY) === "chase") Object.assign(chase, CHASE_DEFAULT);
});

function drawMap(r) {
  camera.aspect = r.w / r.h;
  camera.updateProjectionMatrix();
  scene.background = SKY;
  renderer.setViewport(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.setScissor(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.clear();
  renderer.render(scene, camera);
}

function updateDaylight(dt) {
  if (!caveMeshes.length || !drones[selected]) return;
  // Daylight follows the share of the sky the camera can see, so a cave mouth stays half lit
  // and it darkens gradually further in. Re-measured five times a second.
  skyCheckIn -= dt;
  if (skyCheckIn <= 0) {
    skyCheckIn = 0.2;
    const rock = caveMeshes.map((c) => c.mesh);
    const free = SKY_DIRS.filter((d) => { roofRay.set(chaseCam.position, d); return !roofRay.intersectObjects(rock, false).length; });
    skyOpen = Math.min(1.5 * free.length / SKY_DIRS.length, 1);
  }
  daylight += (skyOpen - daylight) * (1 - Math.exp(-dt * 3));
}

function drawInCave(view, r) {
  const g = drones[selected];
  const cam = view === "fpv" ? fpvCam : chaseCam;
  cam.aspect = r.w / r.h;
  cam.updateProjectionMatrix();

  // Swap the scene into "camera" mode: solid rock, darkness, onboard lights, drones as models only.
  const hidden = overlays.map((o) => o.visible);
  overlays.forEach((o) => (o.visible = false));
  const markers = drones.map((d) => d.marker.visible);
  drones.forEach((d) => (d.marker.visible = !d.craft.children.length));
  const artMarkers = Object.values(artifactMeshes).map((a) => a.marker.visible);
  Object.values(artifactMeshes).forEach((a) => (a.marker.visible = !a.model));
  const ownVis = g.anchor.visible;
  if (view === "fpv") g.anchor.visible = false;
  for (const c of caveMeshes) c.mesh.material = c.fpv;
  const hemiI = hemi.intensity, sunI = sun.intensity;
  const day = daylight;
  hemi.intensity = THREE.MathUtils.lerp(0.35, 1.3, day) * LIGHT_GAIN;   // some fill light in the cave too
  sun.intensity = THREE.MathUtils.lerp(0, 1.4, day) * LIGHT_GAIN;
  flood.intensity = THREE.MathUtils.lerp(70, 8, day) * LIGHT_GAIN;    // onboard LEDs matter in the dark only
  beam.intensity = THREE.MathUtils.lerp(120, 12, day) * LIGHT_GAIN;
  scene.fog.near = THREE.MathUtils.lerp(12, 60, day);
  scene.fog.far = THREE.MathUtils.lerp(80, 600, day);
  scene.fog.color.setRGB(0, 0, 0).lerp(DAY_SKY, day);
  scene.background = day > 0.02 ? new THREE.Color(0, 0, 0).lerp(DAY_SKY, day) : null;

  renderer.setViewport(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.setScissor(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.setClearColor(0x000000, 1);
  renderer.clear();
  renderer.render(scene, cam);

  overlays.forEach((o, k) => (o.visible = hidden[k]));
  drones.forEach((d, k) => (d.marker.visible = markers[k]));
  Object.values(artifactMeshes).forEach((a, k) => (a.marker.visible = artMarkers[k]));
  g.anchor.visible = ownVis;
  for (const c of caveMeshes) c.mesh.material = c.over;
  hemi.intensity = hemiI;
  sun.intensity = sunI;
  flood.intensity = beam.intensity = 0;
  scene.fog.near = 1e5;
  scene.fog.far = 1e5 + 1;
  scene.fog.color.setRGB(0, 0, 0);
}

function drawView(view, r) {
  if (view === "map") drawMap(r);
  else if (drones[selected]) drawInCave(view, r);
}

function placeFpvHud(r) {
  const el = $("fpvhud");
  Object.assign(el.style, { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
  el.classList.toggle("full", r.w === innerWidth);
  const d = lastState && lastState.drones[selected];
  if (d) {
    // Artificial horizon: tilts with roll, slides with pitch (vertical FOV maps to the view height).
    const px = (d.rp[1] / THREE.MathUtils.degToRad(fpvCam.fov)) * r.h;
    $("fpvHorizon").style.transform = `translateY(${px}px) rotate(${-d.rp[0]}rad)`;
  }
}

// ---------------------------------------------------------------- socket
function connect() {
  const port = new URLSearchParams(location.search).get("ws") || "8765";
  ws = new WebSocket(`ws://${location.hostname}:${port}`);
  ws.onopen = () => { $("status").textContent = "연결됨"; ws.send(JSON.stringify({ cmd: "viewer" })); };
  ws.onclose = () => { $("status").textContent = "연결 끊김 (서버 종료?)"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "init") onInit(msg);
    else if (msg.type === "state") onState(msg);
    else if (msg.type === "history") onHistory(msg);
    else if (msg.type === "trails") { if (replayAt == null) onTrails(msg); }
    else if (msg.type === "done") onDone(msg);
  };
}
loadDroneModel();
connect();

let paused = false;
$("pause").onclick = () => {
  paused = !paused;
  ws.send(JSON.stringify({ cmd: "pause", value: paused }));
  $("pause").textContent = paused ? "재개" : "일시정지";
};
$("speed").onchange = (e) => ws.send(JSON.stringify({ cmd: "speed", value: Number(e.target.value) }));
$("opacity").oninput = (e) => caveMats.forEach((m) => { m.opacity = e.target.value / 100; m.depthWrite = e.target.value > 90; });
$("final").onclick = () => { $("final").style.display = "none"; };
// Start views can come from the URL, e.g. /viewer/?main=chase&inset=map
if (new URLSearchParams(location.search).get("follow") === "1") $("cFollow").checked = true;
for (const k of ["main", "inset"]) {
  const v = new URLSearchParams(location.search).get(k);
  if (v) $(`${k}View`).value = v;
}
$("mainView").onchange = $("insetView").onchange = updateFpvHud;
const swapViews = () => {
  if (insetView() === "off") return;
  [$("mainView").value, $("insetView").value] = [insetView(), mainView()];
  updateFpvHud();
};
$("swapView").onclick = swapViews;
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "v" || e.key === "V") swapViews();
  const n = Number(e.key);
  if (n >= 1 && n <= drones.length) $("drones").children[n - 1].click();
});

// Keys held down go to the server, for the keyboard pilot (agents.manual); other agents ignore them.
const PILOT_KEYS = new Set(["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "s", "a", "d", "l", "shift"]);
const held = new Set();
const sendKeys = () => { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ cmd: "keys", keys: [...held] })); };
const pilotKey = (e, down) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (!PILOT_KEYS.has(k)) return;
  if (k.startsWith("arrow")) e.preventDefault();       // no page scrolling
  const had = held.has(k);
  down ? held.add(k) : held.delete(k);
  if (had !== down) sendKeys();
};
addEventListener("keydown", (e) => pilotKey(e, true));
addEventListener("keyup", (e) => pilotKey(e, false));
addEventListener("blur", () => { held.clear(); sendKeys(); });

// One separate window per sensor, for the drone selected in the HUD.
const SENSOR_WIN = { lidar: [960, 820], imu: [900, 560], alt: [900, 560], odom: [1000, 480], status: [760, 420] };
document.querySelectorAll("[data-sensor]").forEach((b) => {
  b.onclick = () => {
    const kind = b.dataset.sensor, [w, h] = SENSOR_WIN[kind];
    const port = new URLSearchParams(location.search).get("ws") || "8765";
    window.open(`sensors.html?sensor=${kind}&drone=${selected}&ws=${port}`, `sensor_${kind}_${selected}`,
                `width=${w},height=${h}`);
  };
});

renderer.setScissorTest(true);
let lastFrame = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  for (const g of drones) {
    if (!g.pose) continue;
    const a = Math.min((now - g.pose.t0) / stateDt, 1);
    g.anchor.position.lerpVectors(g.pose.from.pos, g.pose.to.pos, a);
    g.anchor.quaternion.slerpQuaternions(g.pose.from.q, g.pose.to.q, a);
    const far = camera.position.distanceTo(g.anchor.position) > 7 || !g.craft.children.length;
    g.marker.visible = far;
    // Quadrotor mixer: each rotor gets a quarter of the thrust, more or less by where it sits
    // (roll: left/right, pitch: front/back) and by its spin direction (yaw). Speed ~ sqrt(thrust).
    const [T, roll, pitch, yaw] = g.mix || [0, 0, 0, 0];
    for (const r of g.rotors || []) {
      const u = r.userData;
      const share = Math.max(T * (1 + MIX_TILT * (roll * u.y - pitch * u.x) - MIX_YAW * yaw * u.spin), 0);
      u.w += (Math.sqrt(share) - u.w) * (1 - Math.exp(-dt * 8));   // motors spin up and down, not instantly
      r.rotation.y += u.spin * PROP_SPEED * u.w * dt;
    }
  }
  for (const a of Object.values(artifactMeshes)) {
    a.marker.visible = !a.model || camera.position.distanceTo(a.group.position) > 15;
  }
  if ($("cFollow").checked && drones[selected]) {
    const delta = drones[selected].anchor.position.clone().sub(controls.target);
    controls.target.add(delta);
    camera.position.add(delta);
  }
  controls.update();
  placeCameras(dt);
  updateDaylight(dt);

  const full = { x: 0, y: 0, w: innerWidth, h: innerHeight };
  const main = mainView(), inset = insetView();
  drawView(main, full);
  if (inset !== "off") drawView(inset, insetRect());
  if (inCave(main)) placeFpvHud(full);
  else if (inCave(inset)) placeFpvHud(insetRect());
});
