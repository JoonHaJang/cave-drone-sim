// Cave Drone Sim viewer. Draws whatever the Python server sends; holds no sim logic.
// World frame is z-up (same as the simulator), so the camera uses z as "up".
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const COLORS = [0xff6b6b, 0x4dabf7, 0x69db7c, 0xffd43b, 0xda77f2, 0xff922b, 0x38d9a9, 0xf783ac, 0xa9e34b];
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- scene
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1014);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.up.set(0, 0, 1);
camera.position.set(-30, -60, 70);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(40, 0, 0);
controls.update();

scene.add(new THREE.HemisphereLight(0xcfd8e3, 0x302820, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(30, -40, 80);
scene.add(sun);

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- state
let cave = null, caveMat = null;
const artifactMeshes = {};
const drones = [];          // per drone: {body, ghost, heading, lidar, trail, odomTrail, path, goal, pts}
let linkLines = null;
let selected = 0;
let initMsg = null;
let ws = null;

const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function makeLine(color, dashed = false, opacity = 1) {
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: 0.6, gapSize: 0.4, transparent: true, opacity })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(new THREE.BufferGeometry(), mat);
  line.frustumCulled = false;
  scene.add(line);
  return line;
}

function setLine(line, pts) {
  const flat = new Float32Array(pts.length * 3);
  pts.forEach((p, i) => flat.set(p, i * 3));
  line.geometry.setAttribute("position", new THREE.BufferAttribute(flat, 3));
  line.geometry.computeBoundingSphere();
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}

function onInit(msg) {
  initMsg = msg;
  $("mapname").textContent = msg.map;
  new GLTFLoader().load(`/maps/${msg.map}/cave.glb`, (gltf) => {
    if (cave) scene.remove(cave);
    cave = gltf.scene;
    caveMat = new THREE.MeshStandardMaterial({
      color: 0x9c8b76, roughness: 0.95, side: THREE.DoubleSide,
      transparent: true, opacity: $("opacity").value / 100, depthWrite: false,
    });
    cave.traverse((o) => { if (o.isMesh) o.material = caveMat; });
    scene.add(cave);
    const box = new THREE.Box3().setFromObject(cave);
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
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.35, 20, 14),
                                new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4 }));
    const heading = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.7, 10),
                                   new THREE.MeshBasicMaterial({ color }));
    heading.geometry.rotateZ(-Math.PI / 2);
    heading.geometry.translate(0.6, 0, 0);
    body.add(heading);
    const ghost = new THREE.Mesh(new THREE.SphereGeometry(0.35, 12, 8),
                                 new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.6 }));
    const lidar = new THREE.Points(new THREE.BufferGeometry(),
                                   new THREE.PointsMaterial({ color, size: 0.12, transparent: true, opacity: 0.7 }));
    lidar.frustumCulled = false;
    const pts = new THREE.Points(new THREE.BufferGeometry(),
                                 new THREE.PointsMaterial({ color: 0xffffff, size: 0.5 }));
    pts.frustumCulled = false;
    const goal = new THREE.Mesh(new THREE.SphereGeometry(0.2, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    scene.add(body, ghost, lidar, pts, goal);
    drones.push({ body, ghost, lidar, pts, goal, color,
                  trail: makeLine(color, false, 0.9), odomTrail: makeLine(color, true, 0.6),
                  path: makeLine(0xffffff, false, 0.5) });
  }
  linkLines = new THREE.LineSegments(new THREE.BufferGeometry(),
                                     new THREE.LineBasicMaterial({ color: 0x51cf66, transparent: true, opacity: 0.7 }));
  linkLines.frustumCulled = false;
  scene.add(linkLines);
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
  $("time").textContent = `${msg.t.toFixed(1)} / ${initMsg.mission_time_s} s`;
  $("score").textContent = msg.score.score;
  $("arts").textContent = `${msg.score.artifacts_found} / ${msg.score.artifacts_total}`;
  $("cov").textContent = `${(100 * msg.score.coverage).toFixed(0)} %`;

  const show = (id) => $(id).checked;
  msg.drones.forEach((d, i) => {
    const g = drones[i];
    g.body.position.copy(v3(d.pos));
    g.body.rotation.z = d.yaw;
    g.body.material.emissiveIntensity = d.status === "crashed" || d.status === "dead" ? 0 : 0.4;
    g.ghost.position.copy(v3(d.odom));
    g.ghost.visible = show("cOdom");
    if (d.lidar && show("cLidar")) {
      g.lidar.geometry.setAttribute("position", new THREE.Float32BufferAttribute(d.lidar, 3));
      g.lidar.visible = true;
    } else {
      g.lidar.visible = false;
    }
    const dbg = d.debug || {};
    const dv = show("cDebug");
    if (dbg.path && dbg.path.length) { setLine(g.path, [d.odom, ...dbg.path.slice().reverse()]); }
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
    const drift = Math.hypot(d.pos[0] - d.odom[0], d.pos[1] - d.odom[1], d.pos[2] - d.odom[2]);
    el.querySelector(".drift").textContent = `추정 오차 ${drift.toFixed(2)} m`;
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

  if (show("cFollow") && msg.drones[selected]) {
    const p = v3(msg.drones[selected].pos);
    const delta = p.clone().sub(controls.target);
    controls.target.add(delta);
    camera.position.add(delta);
  }
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

// ---------------------------------------------------------------- socket
function connect() {
  const port = new URLSearchParams(location.search).get("ws") || "8765";
  ws = new WebSocket(`ws://${location.hostname}:${port}`);
  ws.onopen = () => { $("status").textContent = "연결됨"; };
  ws.onclose = () => { $("status").textContent = "연결 끊김 (서버 종료?)"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "init") onInit(msg);
    else if (msg.type === "state") onState(msg);
    else if (msg.type === "trails") onTrails(msg);
    else if (msg.type === "done") onDone(msg);
  };
}
connect();

let paused = false;
$("pause").onclick = () => {
  paused = !paused;
  ws.send(JSON.stringify({ cmd: "pause", value: paused }));
  $("pause").textContent = paused ? "재개" : "일시정지";
};
$("speed").onchange = (e) => ws.send(JSON.stringify({ cmd: "speed", value: Number(e.target.value) }));
$("opacity").oninput = (e) => { if (caveMat) { caveMat.opacity = e.target.value / 100; caveMat.depthWrite = e.target.value > 90; } };
$("final").onclick = () => { $("final").style.display = "none"; };

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
