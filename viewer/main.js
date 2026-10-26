// Cave Drone Sim viewer. Draws whatever the Python server sends; holds no sim logic.
// World frame is z-up (same as the simulator), so the camera uses z as "up".
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const COLORS = [0xff6b6b, 0x4dabf7, 0x69db7c, 0xffd43b, 0xda77f2, 0xff922b, 0x38d9a9, 0xf783ac, 0xa9e34b];
const $ = (id) => document.getElementById(id);

// Drone model (glTF, y-up). Not part of the repo; without it the drones stay spheres.
const DRONE_MODEL = "/bionic_dragonfly_four-axis_uav_aircraft/scene.gltf";
const DRONE_SPAN = 0.5;          // m, rotor tip to rotor tip after scaling
const MODEL_YAW = -Math.PI / 2;  // turns the model's nose onto body +x
const FPV_OFFSET = new THREE.Vector3(0.14, 0, 0.03);   // camera position in the body frame

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
  l.position.set(0, 0, 0);
  l.target.position.set(0, 0, -1);
  fpvCam.add(l, l.target);
}
scene.add(fpvCam);

const hemi = new THREE.HemisphereLight(0xcfd8e3, 0x302820, 1.4);
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
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
let lastState = null;
let ws = null;
let droneModel = null;
let stateDt = 100;          // ms between state messages, smoothed
let lastStateAt = 0;
let fpvBig = false;

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

function loadDroneModel() {
  new GLTFLoader().load(DRONE_MODEL, (gltf) => {
    const m = gltf.scene;
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
  // Status LED in the drone's colour, so several drones stay apart up close.
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.025, 10, 8),
                             new THREE.MeshBasicMaterial({ color: g.color }));
  led.position.set(-0.12, 0, 0.06);
  g.craft.add(m, led);
}

function onInit(msg) {
  initMsg = msg;
  $("mapname").textContent = msg.map;
  new GLTFLoader().load(`/maps/${msg.map}/cave.glb`, (gltf) => {
    for (const c of caveMeshes) scene.remove(c.mesh);
    caveMeshes.length = caveMats.length = 0;
    const box = new THREE.Box3();
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      const spike = /speleo/i.test(o.name) || /speleo/i.test(o.parent?.name || "");
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
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
      box.expandByObject(o);
    });
    scene.add(gltf.scene);
    caveMeshes.forEach((c) => (c.root = gltf.scene));
    const c = box.getCenter(new THREE.Vector3());
    controls.target.copy(c);
    camera.position.set(c.x - 0.4 * (box.max.x - box.min.x), c.y - 0.9 * (box.max.y - box.min.y), c.z + 70);
    controls.update();
  });

  // Base station and home circle
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 0.3, 24),
                              new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x333333 }));
  base.rotation.x = Math.PI / 2;
  base.position.copy(v3(msg.base));
  scene.add(base);
  const ring = new THREE.Mesh(new THREE.RingGeometry(msg.home_radius - 0.08, msg.home_radius, 64),
                              new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.5 }));
  ring.position.copy(v3(msg.base)).add(new THREE.Vector3(0, 0, -2.5));
  scene.add(ring);
  overlays.push(ring);

  for (const a of msg.artifacts) {
    const m = new THREE.Mesh(new THREE.OctahedronGeometry(0.45),
                             new THREE.MeshStandardMaterial({ color: 0x868e96, emissive: 0x222222 }));
    m.position.copy(v3(a.pos));
    m.userData = a;
    scene.add(m);
    artifactMeshes[a.id] = m;
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
    scene.add(anchor, ghost, lidar, pts, goal);
    overlays.push(ghost, lidar, pts, goal);
    const g = { anchor, craft, marker, ghost, lidar, pts, goal, color, pose: null,
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

const STATUS_KO = { flying: "비행", landing: "착륙 중", landed: "착륙", crashed: "충돌", dead: "배터리 방전" };

function onState(msg) {
  if (!initMsg) return;
  const now = performance.now();
  if (lastStateAt) stateDt += (Math.min(Math.max(now - lastStateAt, 20), 500) - stateDt) * 0.2;
  lastStateAt = now;
  lastState = msg;
  $("time").textContent = `${msg.t.toFixed(1)} / ${initMsg.mission_time_s} s`;
  $("score").textContent = msg.score.score;
  $("arts").textContent = `${msg.score.artifacts_found} / ${msg.score.artifacts_total}`;
  $("cov").textContent = `${(100 * msg.score.coverage).toFixed(0)} %`;

  const show = (id) => $(id).checked;
  msg.drones.forEach((d, i) => {
    const g = drones[i];
    // Interpolate from wherever the drone is drawn now to the new pose over one message interval.
    const target = { pos: v3(d.pos), q: quatOf(d.yaw, d.rp) };
    g.pose = g.pose ? { from: { pos: g.anchor.position.clone(), q: g.anchor.quaternion.clone() }, to: target, t0: now }
                    : { from: target, to: target, t0: now };
    g.marker.material.emissiveIntensity = d.status === "crashed" || d.status === "dead" ? 0 : 0.4;
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

  for (const [id, m] of Object.entries(artifactMeshes)) {
    const found = msg.found.includes(Number(id));
    m.material.color.setHex(found ? 0x51cf66 : 0x868e96);
    m.visible = found || show("cTruth");
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

// ---------------------------------------------------------------- FPV
const fpvOn = () => $("cFpv").checked && drones[selected];

function updateFpvHud() {
  const d = lastState && lastState.drones[selected];
  $("fpvhud").style.display = fpvOn() ? "block" : "none";
  if (!d) return;
  const speed = Math.hypot(...(d.vel || [0, 0, 0]));
  const gust = Math.hypot(...(d.aero || [0, 0, 0]));
  $("fpvTitle").textContent = `FPV · 드론 ${selected}`;
  $("fpvSpd").textContent = `${speed.toFixed(2)} m/s`;
  $("fpvClr").textContent = d.clear == null ? "> 1.4 m" : `${d.clear.toFixed(2)} m`;
  $("fpvClr").className = d.clear != null && d.clear < 0.4 ? "warn" : "";
  $("fpvGust").textContent = `${gust.toFixed(2)} m/s²`;
  $("fpvAtt").textContent = `R ${(d.rp[0] * 57.3).toFixed(0)}° P ${(d.rp[1] * 57.3).toFixed(0)}°`;
  $("fpvBat").textContent = `${(100 * d.battery).toFixed(0)} %`;
  const dead = d.status === "crashed" || d.status === "dead";
  $("fpvAlert").textContent = dead ? (STATUS_KO[d.status] || d.status) : "";
}

function fpvRect() {
  // Inset in the top-right corner, or the whole window when enlarged.
  if (fpvBig) return { x: 0, y: 0, w: innerWidth, h: innerHeight };
  const w = Math.min(560, Math.round(innerWidth * 0.38)), h = Math.round(w * 9 / 16);
  return { x: innerWidth - w - 10, y: 44, w, h };
}

function renderOverview(r) {
  camera.aspect = r.w / r.h;
  camera.updateProjectionMatrix();
  scene.background = SKY;
  renderer.setViewport(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.setScissor(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.clear();
  renderer.render(scene, camera);
}

function renderFpv(r) {
  const g = drones[selected];
  fpvCam.position.copy(FPV_OFFSET).applyQuaternion(g.anchor.quaternion).add(g.anchor.position);
  fpvCam.quaternion.copy(g.anchor.quaternion).multiply(CAM_IN_BODY);
  fpvCam.aspect = r.w / r.h;
  fpvCam.updateProjectionMatrix();

  // Swap the scene into "camera" mode: solid rock, darkness, onboard lights, no overlays.
  const hidden = overlays.map((o) => o.visible);
  overlays.forEach((o) => (o.visible = false));
  const ownVis = g.anchor.visible;
  g.anchor.visible = false;
  for (const c of caveMeshes) c.mesh.material = c.fpv;
  const hemiI = hemi.intensity, sunI = sun.intensity;
  hemi.intensity = 0.05;
  sun.intensity = 0;
  flood.intensity = 70;
  beam.intensity = 120;
  scene.fog.near = 6;
  scene.fog.far = 45;
  scene.background = null;

  renderer.setViewport(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.setScissor(r.x, innerHeight - r.y - r.h, r.w, r.h);
  renderer.setClearColor(0x000000, 1);
  renderer.clear();
  renderer.render(scene, fpvCam);

  overlays.forEach((o, k) => (o.visible = hidden[k]));
  g.anchor.visible = ownVis;
  for (const c of caveMeshes) c.mesh.material = c.over;
  hemi.intensity = hemiI;
  sun.intensity = sunI;
  flood.intensity = beam.intensity = 0;
  scene.fog.near = 1e5;
  scene.fog.far = 1e5 + 1;
}

function placeFpvHud(r) {
  const el = $("fpvhud");
  Object.assign(el.style, { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
  const d = lastState && lastState.drones[selected];
  if (d) {
    // Artificial horizon: tilts with roll, slides with pitch (vertical FOV maps to the inset height).
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
    else if (msg.type === "trails") onTrails(msg);
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
$("cFpv").onchange = updateFpvHud;
const toggleBig = () => { fpvBig = !fpvBig; $("fpvBig").textContent = fpvBig ? "FPV 작게 (V)" : "FPV 크게 (V)"; };
$("fpvBig").onclick = toggleBig;
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "v" || e.key === "V") toggleBig();
  const n = Number(e.key);
  if (n >= 1 && n <= drones.length) $("drones").children[n - 1].click();
});

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
renderer.setAnimationLoop(() => {
  const now = performance.now();
  for (const g of drones) {
    if (!g.pose) continue;
    const a = Math.min((now - g.pose.t0) / stateDt, 1);
    g.anchor.position.lerpVectors(g.pose.from.pos, g.pose.to.pos, a);
    g.anchor.quaternion.slerpQuaternions(g.pose.from.q, g.pose.to.q, a);
    const far = camera.position.distanceTo(g.anchor.position) > 7 || !g.craft.children.length;
    g.marker.visible = far;
  }
  if ($("cFollow").checked && drones[selected]) {
    const delta = drones[selected].anchor.position.clone().sub(controls.target);
    controls.target.add(delta);
    camera.position.add(delta);
  }
  controls.update();

  const full = { x: 0, y: 0, w: innerWidth, h: innerHeight };
  const fpv = fpvOn();
  if (fpv && fpvBig) {
    renderFpv(full);
  } else {
    renderOverview(full);
    if (fpv) renderFpv(fpvRect());
  }
  if (fpv) placeFpvHud(fpvBig ? full : fpvRect());
});
