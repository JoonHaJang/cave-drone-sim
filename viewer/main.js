// Cave Drone Sim viewer. Draws whatever the Python server sends; holds no sim logic.
// World frame is z-up (same as the simulator), so the camera uses z as "up".
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

// ---------------------------------------------------------------- constants
const COLORS = [0xff6b6b, 0x4dabf7, 0x69db7c, 0xffd43b, 0xda77f2, 0xff922b, 0x38d9a9, 0xf783ac, 0xa9e34b];
const ARTIFACTS = {
  survivor: { name: "Survivor", color: 0xff6b6b },
  backpack: { name: "Backpack", color: 0xff922b },
  phone: { name: "Phone", color: 0x4dabf7 },
  helmet: { name: "Helmet", color: 0xffd43b },
  rope: { name: "Rope", color: 0xda77f2 },
};
const DRONE_SCALE = 1.6;                    // drawn bigger than the 0.2 m collision sphere so it reads from afar
const MAP_MAX = 400_000, MAP_VOXEL = 0.3;   // accumulated LiDAR map: at most one point per 30 cm voxel
const FOG = { outside: 0.004, inside: 0.06 };
const LAMP = 30;                           // headlamp candela; walls 2 m away should stay under the bloom threshold
const HOLO = [0x8b5cf6, 0x3b82f6, 0x22d3ee].map((c) => new THREE.Color(c));    // cave tint, low -> high
const MAP_RAMP = [0xd9480f, 0xf59f00, 0xfff3bf].map((c) => new THREE.Color(c)); // mapped surface tint

const $ = (id) => document.getElementById(id);
const on = (id) => $(id).checked;
const hex = (c) => "#" + c.toString(16).padStart(6, "0");
const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const damp = (dt, rate) => 1 - Math.exp(-rate * dt);          // frame-rate independent lerp factor
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function ramp(stops, h, out) {
  const x = THREE.MathUtils.clamp(h, 0, 1) * (stops.length - 1);
  const k = Math.min(Math.floor(x), stops.length - 2);
  return out.lerpColors(stops[k], stops[k + 1], x - k);
}

// ---------------------------------------------------------------- renderer, camera, post
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.prepend(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(innerWidth, innerHeight);
labelRenderer.domElement.id = "labels";
renderer.domElement.after(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = backdrop();
scene.fog = new THREE.FogExp2(0x05080d, FOG.outside);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.05, 2000);
camera.up.set(0, 0, 1);
camera.position.set(-30, -60, 70);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(40, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.setPixelRatio(renderer.getPixelRatio());
composer.setSize(innerWidth, innerHeight);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.6, 0.45, 0.72);
composer.addPass(bloom);
composer.addPass(new OutputPass());

const hemi = new THREE.HemisphereLight(0xbcd2ff, 0x2a1d14, 1.1);
const sun = new THREE.DirectionalLight(0xfff0dd, 1.6);
sun.position.set(30, -40, 80);
scene.add(hemi, sun);

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
  labelRenderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------- textures
function canvasTexture(size, paint) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  paint(c.getContext("2d"), size);
  return new THREE.CanvasTexture(c);
}

function backdrop() {
  const t = canvasTexture(512, (g, s) => {
    const grd = g.createRadialGradient(s / 2, s * 0.35, 10, s / 2, s * 0.6, s * 0.85);
    grd.addColorStop(0, "#15233a");
    grd.addColorStop(0.5, "#090e17");
    grd.addColorStop(1, "#020306");
    g.fillStyle = grd;
    g.fillRect(0, 0, s, s);
  });
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

const dot = (hard) => canvasTexture(64, (g, s) => {
  const grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(hard ? 0.6 : 0.2, hard ? "rgba(255,255,255,1)" : "rgba(255,255,255,.5)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, s, s);
});
const DOT = dot(true), GLOW = dot(false);

// ---------------------------------------------------------------- cave materials
const caveU = {
  uTime: { value: 0 }, uOpacity: { value: 0.7 },
  uZ: { value: new THREE.Vector2(-12, 9) }, uBase: { value: new THREE.Vector3() },
};

// Hologram: glowing silhouettes tinted by height, contour lines, and a sonar ping from the base.
const holoMat = new THREE.ShaderMaterial({
  uniforms: { ...caveU, uC0: { value: HOLO[0] }, uC1: { value: HOLO[1] }, uC2: { value: HOLO[2] } },
  vertexShader: /* glsl */ `
    varying vec3 vW; varying vec3 vN;
    void main() {
      vec4 w = modelMatrix * vec4(position, 1.0);
      vW = w.xyz;
      vN = normalize(mat3(modelMatrix) * normal);
      gl_Position = projectionMatrix * viewMatrix * w;
    }`,
  fragmentShader: /* glsl */ `
    uniform float uTime, uOpacity; uniform vec2 uZ; uniform vec3 uBase, uC0, uC1, uC2;
    varying vec3 vW; varying vec3 vN;
    void main() {
      float h = clamp((vW.z - uZ.x) / (uZ.y - uZ.x), 0.0, 1.0);
      vec3 col = h < 0.5 ? mix(uC0, uC1, h * 2.0) : mix(uC1, uC2, h * 2.0 - 1.0);
      float rim = pow(1.0 - abs(dot(normalize(vN), normalize(cameraPosition - vW))), 2.2);
      float q = vW.z / 1.5;                                        // contour every 1.5 m of height
      float contour = 1.0 - smoothstep(0.0, 1.2, abs(fract(q - 0.5) - 0.5) / fwidth(q));
      float r = mod(uTime * 18.0, 140.0);
      float ping = exp(-pow((distance(vW.xy, uBase.xy) - r) / 2.0, 2.0)) * (1.0 - r / 140.0);
      float a = 0.035 + 0.75 * rim + 0.16 * contour + 0.8 * ping;
      gl_FragColor = vec4(col * (0.6 + 1.6 * ping) + 0.4 * ping, a * uOpacity);
    }`,
  transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
});

// Rock: lit stone with procedural grain and faint sediment bands, darker deeper down.
const NOISE = /* glsl */ `
  float hash3(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
  float noise3(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + vec3(1, 0, 0)), f.x), mix(hash3(i + vec3(0, 1, 0)), hash3(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash3(i + vec3(0, 0, 1)), hash3(i + vec3(1, 0, 1)), f.x), mix(hash3(i + vec3(0, 1, 1)), hash3(i + vec3(1, 1, 1)), f.x), f.y), f.z);
  }
`;
const rockMat = new THREE.MeshStandardMaterial({ roughness: 0.9, side: THREE.DoubleSide, transparent: true, envMapIntensity: 0 });
rockMat.onBeforeCompile = (s) => {
  s.uniforms.uZ = caveU.uZ;
  s.vertexShader = "varying vec3 vRock;\n" + s.vertexShader.replace("#include <begin_vertex>",
    "#include <begin_vertex>\nvRock = (modelMatrix * vec4(transformed, 1.0)).xyz;");
  s.fragmentShader = "varying vec3 vRock;\nuniform vec2 uZ;\n" + NOISE + s.fragmentShader.replace("#include <color_fragment>", `
    #include <color_fragment>
    float n = 0.55 * noise3(vRock * 0.7) + 0.3 * noise3(vRock * 2.3) + 0.15 * noise3(vRock * 7.1);
    float h = clamp((vRock.z - uZ.x) / (uZ.y - uZ.x), 0.0, 1.0);
    vec3 rock = mix(vec3(0.05, 0.045, 0.04), vec3(0.26, 0.2, 0.15), h);
    rock *= 0.65 + 0.7 * n;
    rock *= 0.9 + 0.1 * sin(vRock.z * 5.0 + n * 7.0);
    diffuseColor.rgb = rock;`).replace("#include <normal_fragment_maps>", `
    #include <normal_fragment_maps>
    vec3 bp = vRock * 1.8, grad = vec3(0.0);
    for (int o = 0; o < 2; o++) {                                // two octaves of bump from the noise gradient
      float b0 = noise3(bp);
      grad += (vec3(noise3(bp + vec3(0.1, 0, 0)), noise3(bp + vec3(0, 0.1, 0)), noise3(bp + vec3(0, 0, 0.1))) - b0) * (o == 0 ? 4.0 : 2.0);
      bp *= 3.1;
    }
    vec3 gv = mat3(viewMatrix) * grad;
    normal = normalize(normal - 0.6 * (gv - dot(gv, normal) * normal));`);
};

// ---------------------------------------------------------------- state
let initMsg = null, lastState = null, ws = null;
let cave = null, grid = null, linkLine = null;
let selected = 0, camMode = "orbit", caveStyle = "holo", saved = null, intro = null;
let lastPulse = 0;
const drones = [];          // per drone: meshes, lines, smoothed pose, HUD card refs
const artifactObjs = {};
const found = new Set();
let seeded = false;         // first state after connecting only seeds `found`/status, without log spam
const effects = [];
const ray = new THREE.Raycaster();

// ---------------------------------------------------------------- helpers
function makeLine2(color, { width = 2, dashed = false, opacity = 1, vertexColors = false, segments = false } = {}) {
  const mat = new LineMaterial({ color, linewidth: width, transparent: true, opacity, dashed, vertexColors,
                                 dashSize: 0.6, gapSize: 0.4, depthWrite: false });
  const line = segments ? new LineSegments2(new LineSegmentsGeometry(), mat) : new Line2(new LineGeometry(), mat);
  line.frustumCulled = false;
  line.visible = false;
  scene.add(line);
  return line;
}

function setLine2(line, flat, colors) {
  line.geometry.dispose();
  const g = line.isLine2 ? new LineGeometry() : new LineSegmentsGeometry();
  g.setPositions(flat);
  if (colors) g.setColors(colors);
  line.geometry = g;
  if (line.material.dashed) line.computeLineDistances();
}

// Points are sized in metres; cap their on-screen size so close-ups don't turn into a wall of discs.
function capPointSize(mat, px) {
  const max = (px * renderer.getPixelRatio()).toFixed(1);
  mat.onBeforeCompile = (s) => {
    s.vertexShader = s.vertexShader.replace("#include <logdepthbuf_vertex>",
      `gl_PointSize = min(gl_PointSize, ${max});
#include <logdepthbuf_vertex>`);
  };
  mat.customProgramCacheKey = () => `cap${max}`;
  return mat;
}

function makePoints(color, size, map, additive) {
  const p = new THREE.Points(new THREE.BufferGeometry(), capPointSize(new THREE.PointsMaterial({
    color, size, map, transparent: true, depthWrite: !additive, alphaTest: additive ? 0 : 0.5,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  }), additive ? 6 : 5));
  p.frustumCulled = false;
  scene.add(p);
  return p;
}

function setPoints(points, flat) {
  let attr = points.geometry.getAttribute("position");
  if (!attr || attr.array.length < flat.length) {
    attr = new THREE.BufferAttribute(new Float32Array(Math.max(flat.length, 3000)), 3).setUsage(THREE.DynamicDrawUsage);
    points.geometry.setAttribute("position", attr);
  }
  attr.array.set(flat);
  attr.needsUpdate = true;
  points.geometry.setDrawRange(0, flat.length / 3);
}

function tag(text, cls, color) {
  const el = document.createElement("div");
  el.className = `tag ${cls}`;
  if (color !== undefined) el.style.setProperty("--c", hex(color));
  el.textContent = text;
  return new CSS2DObject(el);
}

function burst(pos, color, { from = 0.5, to = 6, life = 1.4 } = {}) {
  const m = new THREE.Mesh(RING, new THREE.MeshBasicMaterial({
    color, transparent: true, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
  m.position.copy(pos);
  scene.add(m);
  effects.push({ m, t: 0, from, to, life });
}
const RING = new THREE.RingGeometry(0.9, 1, 64);

// ---------------------------------------------------------------- accumulated LiDAR map
const mapPos = new Float32Array(MAP_MAX * 3), mapCol = new Float32Array(MAP_MAX * 3);
const mapGeom = new THREE.BufferGeometry();
mapGeom.setAttribute("position", new THREE.BufferAttribute(mapPos, 3).setUsage(THREE.DynamicDrawUsage));
mapGeom.setAttribute("color", new THREE.BufferAttribute(mapCol, 3).setUsage(THREE.DynamicDrawUsage));
mapGeom.setDrawRange(0, 0);
const mapCloud = new THREE.Points(mapGeom, capPointSize(new THREE.PointsMaterial({ size: 0.2, vertexColors: true, map: DOT, alphaTest: 0.5 }), 3));
mapCloud.frustumCulled = false;
scene.add(mapCloud);
const mapSeen = new Set();
let mapCount = 0, mapUploaded = 0;
const _c = new THREE.Color();

function addToMap(flat) {
  const z0 = caveU.uZ.value.x, z1 = caveU.uZ.value.y;
  for (let k = 0; k + 2 < flat.length && mapCount < MAP_MAX; k += 3) {
    const x = flat[k], y = flat[k + 1], z = flat[k + 2];
    const key = (Math.round(x / MAP_VOXEL) + 2048) + (Math.round(y / MAP_VOXEL) + 2048) * 4096
              + (Math.round(z / MAP_VOXEL) + 2048) * 16777216;
    if (mapSeen.has(key)) continue;
    mapSeen.add(key);
    ramp(MAP_RAMP, (z - z0) / (z1 - z0), _c);
    const o = mapCount * 3;
    mapPos[o] = x; mapPos[o + 1] = y; mapPos[o + 2] = z;
    mapCol[o] = _c.r; mapCol[o + 1] = _c.g; mapCol[o + 2] = _c.b;
    mapCount++;
  }
}

function flushMap() {
  if (mapCount === mapUploaded) return;
  for (const name of ["position", "color"]) {
    const a = mapGeom.attributes[name];
    a.addUpdateRange(mapUploaded * 3, (mapCount - mapUploaded) * 3);
    a.needsUpdate = true;
  }
  mapGeom.setDrawRange(0, mapCount);
  mapUploaded = mapCount;
}

// ---------------------------------------------------------------- drones
const shellMat = new THREE.MeshStandardMaterial({ color: 0x1b222c, metalness: 0.7, roughness: 0.35 });
const armMat = new THREE.MeshStandardMaterial({ color: 0x2b3440, metalness: 0.5, roughness: 0.5 });
const rotorMat = new THREE.MeshBasicMaterial({ color: 0xcfe3ff, transparent: true, opacity: 0.14,
                                               side: THREE.DoubleSide, depthWrite: false });
const BEAM_LEN = 7;
const beamGeo = new THREE.ConeGeometry(BEAM_LEN * Math.tan(0.42), BEAM_LEN, 32, 1, true)
  .rotateZ(Math.PI / 2).translate(BEAM_LEN / 2, 0, 0);       // apex at the nose, opening along +x
const beamMat = new THREE.ShaderMaterial({
  uniforms: { uColor: { value: new THREE.Color(0xfff1dc) } },
  vertexShader: /* glsl */ `
    varying float vX; varying vec3 vN; varying vec3 vV;
    void main() {
      vX = position.x;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: /* glsl */ `
    uniform vec3 uColor; varying float vX; varying vec3 vN; varying vec3 vV;
    void main() {
      float t = clamp(vX / ${BEAM_LEN.toFixed(1)}, 0.0, 1.0);
      gl_FragColor = vec4(uColor, 0.11 * pow(1.0 - t, 1.6) * abs(dot(normalize(vN), normalize(vV))));
    }`,
  transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
});

function makeDrone(i) {
  const color = COLORS[i % COLORS.length];
  const root = new THREE.Group(), tilt = new THREE.Group(), model = new THREE.Group();
  root.add(tilt);
  tilt.add(model);
  model.scale.setScalar(DRONE_SCALE);
  const led = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 3 });

  model.add(new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.16, 0.08), shellMat));
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.075, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2), led);
  canopy.rotation.x = Math.PI / 2;
  canopy.position.z = 0.04;
  model.add(canopy);
  for (const a of [Math.PI / 4, -Math.PI / 4]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.03, 0.025), armMat);
    arm.rotation.z = a;
    model.add(arm);
  }
  const rotors = [];
  for (const [x, y] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.05, 12), armMat);
    motor.rotation.x = Math.PI / 2;
    motor.position.set(0.2 * x, 0.2 * y, 0.01);
    const disc = new THREE.Mesh(new THREE.CircleGeometry(0.13, 28), rotorMat);
    disc.position.set(0.2 * x, 0.2 * y, 0.045);
    const rim = new THREE.Mesh(new THREE.RingGeometry(0.122, 0.132, 28), led);
    rim.position.copy(disc.position);
    const blade = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.018, 0.004), armMat);
    blade.position.set(0.2 * x, 0.2 * y, 0.047);
    model.add(motor, disc, rim, blade);
    rotors.push({ disc, blade, dir: x * y });
  }
  const nose = new THREE.Mesh(new THREE.SphereGeometry(0.025, 8, 6), new THREE.MeshBasicMaterial({ color: 0xfff4e0 }));
  nose.position.set(0.155, 0, 0);
  model.add(nose);

  const lamp = new THREE.SpotLight(0xfff1dc, LAMP, 30, 0.6, 0.55, 2);
  lamp.position.set(0.3, 0, 0);
  lamp.target.position.set(6, 0, -0.8);
  const beam = new THREE.Mesh(beamGeo, beamMat);
  beam.position.set(0.3, 0, 0);
  beam.rotation.y = 0.13;
  tilt.add(lamp, lamp.target, beam);

  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW, color, transparent: true, depthWrite: false,
                                                           depthTest: false, sizeAttenuation: false,
                                                           blending: THREE.AdditiveBlending }));
  halo.scale.setScalar(0.05);
  halo.renderOrder = 10;
  const label = tag(`D${i}`, "", color);
  label.position.set(0, 0, 0.9);
  root.add(halo, label);

  const ghost = new THREE.Mesh(new THREE.IcosahedronGeometry(0.32, 1),
                               new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.5, depthWrite: false }));
  const errLine = new THREE.Line(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3)),
                                 new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.45 }));
  errLine.frustumCulled = false;
  const goal = new THREE.Mesh(new THREE.OctahedronGeometry(0.25), new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true }));
  scene.add(root, ghost, errLine, goal);

  return {
    i, color, root, tilt, model, led, rotors, lamp, beam, halo, label, ghost, errLine, goal,
    scan: makePoints(color, 0.16, GLOW, true),
    pts: makePoints(0xffffff, 0.4, DOT, false),
    trail: makeLine2(0xffffff, { width: 2.5, vertexColors: true, opacity: 0.95 }),
    odomTrail: makeLine2(color, { width: 1.2, dashed: true, opacity: 0.55 }),
    path: makeLine2(0xffffff, { width: 1.5, opacity: 0.6 }),
    target: new THREE.Vector3(), odom: new THREE.Vector3(), vel: new THREE.Vector3(),
    yaw: 0, targetYaw: 0, lastT: null, status: "flying", alive: true, battery: 1, lowWarned: false,
    hasTrail: false,
  };
}

const selRing = new THREE.Mesh(new THREE.RingGeometry(0.8, 0.9, 48),
                               new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.8, side: THREE.DoubleSide,
                                                             depthWrite: false, blending: THREE.AdditiveBlending }));
scene.add(selRing);

// ---------------------------------------------------------------- base station
const baseGroup = new THREE.Group(), baseGround = new THREE.Group();   // baseGround sits on the cave floor
baseGroup.add(baseGround);
scene.add(baseGroup);

function beaconGeometry(h) {
  const g = new THREE.CylinderGeometry(0.08, 0.35, h, 24, 1, true).rotateX(Math.PI / 2).translate(0, 0, h / 2);
  const pos = g.attributes.position, col = new Float32Array(pos.count * 4);
  for (let k = 0; k < pos.count; k++) col.set([1, 1, 1, 0.55 * (1 - pos.getZ(k) / h)], k * 4);
  g.setAttribute("color", new THREE.BufferAttribute(col, 4));
  return g;
}

function buildBase(msg) {
  baseGroup.position.copy(v3(msg.base));
  baseGround.position.z = -2.5;
  caveU.uBase.value.copy(baseGroup.position);
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.15, 0.12, 48).rotateX(Math.PI / 2),
                             new THREE.MeshStandardMaterial({ color: 0x18202b, metalness: 0.6, roughness: 0.4 }));
  pad.position.z = 0.06;
  const padRing = new THREE.Mesh(new THREE.RingGeometry(0.72, 0.84, 48), new THREE.MeshBasicMaterial({ color: 0x22d3ee }));
  padRing.position.z = 0.125;
  const home = new THREE.Mesh(new THREE.RingGeometry(msg.home_radius - 0.07, msg.home_radius, 96),
                              new THREE.MeshBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.5,
                                                            side: THREE.DoubleSide, depthWrite: false }));
  home.position.z = 0.03;
  const beacon = new THREE.Mesh(beaconGeometry(9), new THREE.MeshBasicMaterial({
    color: 0x22d3ee, vertexColors: true, transparent: true, depthWrite: false, side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending }));
  const label = tag("BASE", "base");
  label.position.z = 2.2;
  baseGround.add(pad, padRing, home, beacon, label);

  const ring = [];
  for (let k = 0; k <= 160; k++) {
    const a = (k / 160) * Math.PI * 2;
    ring.push(Math.cos(a) * msg.comms_range, Math.sin(a) * msg.comms_range, 0);
  }
  const comms = makeLine2(0x22d3ee, { width: 1, dashed: true, opacity: 0.28 });
  comms.material.dashSize = 1.6;
  comms.material.gapSize = 1.0;
  baseGroup.add(comms);
  setLine2(comms, ring);
  comms.visible = true;
}

// ---------------------------------------------------------------- artifacts
const ART_GEO = {
  survivor: new THREE.CapsuleGeometry(0.2, 0.55, 4, 12).rotateX(Math.PI / 2),
  backpack: new THREE.BoxGeometry(0.45, 0.32, 0.55),
  phone: new THREE.BoxGeometry(0.36, 0.06, 0.6),
  helmet: new THREE.SphereGeometry(0.34, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2).rotateX(Math.PI / 2),
  rope: new THREE.TorusGeometry(0.28, 0.07, 10, 28),
};

function buildArtifact(a) {
  const info = ARTIFACTS[a.type] || { name: a.type, color: 0xadb5bd };
  const mat = new THREE.MeshStandardMaterial({ color: info.color, emissive: info.color, emissiveIntensity: 0.25,
                                               transparent: true, opacity: 0.5, roughness: 0.5 });
  const mesh = new THREE.Mesh(ART_GEO[a.type] || new THREE.OctahedronGeometry(0.45), mat);
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW, color: info.color, transparent: true,
                                                           depthWrite: false, blending: THREE.AdditiveBlending }));
  glow.scale.setScalar(2.4);
  glow.visible = false;
  const label = tag(info.name, "art");
  label.position.z = 0.9;
  const group = new THREE.Group();
  group.add(mesh, glow, label);
  group.position.copy(v3(a.pos));
  scene.add(group);
  artifactObjs[a.id] = { a, info, group, mesh, glow, label, phase: a.id * 1.7 };
}

function markFound(id, celebrate) {
  const o = artifactObjs[id];
  if (!o) return;
  o.mesh.material.opacity = 1;
  o.mesh.material.emissiveIntensity = 1.6;
  o.glow.visible = true;
  o.label.element.textContent = `✓ ${o.info.name}`;
  o.label.element.classList.add("found");
  if (celebrate) {
    burst(o.group.position, o.info.color, { to: 7, life: 1.6 });
    burst(o.group.position, 0x4ade80, { to: 4, life: 1.1 });
    logEvent(`Found ${o.info.name}`, "good");
  }
}

// ---------------------------------------------------------------- HUD
const ui = {};   // cached per-drone card elements

function buildCards(n) {
  const box = $("drones");
  box.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const el = document.createElement("div");
    el.className = "drone" + (i === selected ? " sel" : "");
    el.style.setProperty("--c", hex(COLORS[i % COLORS.length]));
    el.innerHTML = `<i class="swatch"></i><div class="dmain">
      <div class="row"><span class="dname">Drone ${i}</span>
        <span class="row" style="gap:6px"><span class="link">● base</span><span class="chip st">-</span></span></div>
      <div class="row muted small"><span class="mode"></span><span class="drift mono"></span></div>
      <div class="row small"><div class="bar" style="flex:1"><div class="batt"></div></div><span class="pct mono muted"></span></div>
    </div>`;
    el.onclick = () => select(i);
    el.ondblclick = () => { select(i); setCamMode("chase"); };
    box.appendChild(el);
    ui[i] = { el, st: el.querySelector(".st"), link: el.querySelector(".link"), mode: el.querySelector(".mode"),
              drift: el.querySelector(".drift"), batt: el.querySelector(".batt"), pct: el.querySelector(".pct") };
  }
}

function logEvent(text, cls = "") {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t"></span><span class="${cls}"></span>`;
  li.firstChild.textContent = lastState ? `${lastState.t.toFixed(0)}s` : "--";
  li.lastChild.textContent = text;
  const list = $("events");
  list.prepend(li);
  while (list.children.length > 9) list.lastChild.remove();
}

function select(i) {
  if (i >= drones.length) return;
  selected = i;
  Object.values(ui).forEach((u, k) => u.el.classList.toggle("sel", k === i));
  selRing.material.color.setHex(drones[i].color);
}

// ---------------------------------------------------------------- messages
function onInit(msg) {
  initMsg = msg;
  $("mapname").textContent = msg.map;
  $("tmax").textContent = msg.mission_time_s;
  buildBase(msg);
  msg.artifacts.forEach(buildArtifact);
  for (let i = 0; i < msg.n_drones; i++) drones.push(makeDrone(i));
  linkLine = makeLine2(0x5effc0, { width: 1.8, dashed: true, opacity: 0.85, segments: true });
  linkLine.material.dashSize = 0.7;
  linkLine.material.gapSize = 0.45;
  buildCards(msg.n_drones);
  select(0);
  logEvent(`Connected · ${msg.map} · ${msg.n_drones} drones`);

  new GLTFLoader().load(`/maps/${msg.map}/cave.glb`, (gltf) => {
    cave = gltf.scene;
    cave.traverse((o) => { if (o.isMesh && !o.geometry.attributes.normal) o.geometry.computeVertexNormals(); });
    scene.add(cave);
    cave.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(cave);
    const c = box.getCenter(new THREE.Vector3());
    caveU.uZ.value.set(box.min.z, box.max.z);
    applyCave();

    const size = Math.ceil((Math.max(box.max.x - box.min.x, box.max.y - box.min.y) * 1.5) / 10) * 10;
    grid = new THREE.GridHelper(size, size / 5, 0x2b4a6b, 0x15263a);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(c.x, c.y, box.min.z - 0.5);
    grid.material.transparent = true;
    grid.material.opacity = 0.4;
    grid.material.depthWrite = false;
    scene.add(grid);

    // Put the pad on the cave floor right under the base.
    ray.set(baseGroup.position, new THREE.Vector3(0, 0, -1));
    ray.far = 20;
    const hit = ray.intersectObject(cave, true)[0];
    if (hit) baseGround.position.z = hit.point.z - baseGroup.position.z + 0.03;

    controls.target.copy(c);
    const off = new THREE.Vector3(-0.4 * (box.max.x - box.min.x), -0.9 * (box.max.y - box.min.y), 70);
    intro = { c, off, t: 0 };
  });
}

function onState(msg) {
  if (!initMsg) return;
  lastState = msg;
  const s = msg.score;
  $("time").textContent = msg.t.toFixed(1);
  $("tbar").style.width = `${Math.min(100, (100 * msg.t) / initMsg.mission_time_s)}%`;
  $("score").textContent = s.score.toFixed(1);
  $("arts").textContent = `${s.artifacts_found}/${s.artifacts_total}`;
  $("cov").textContent = `${Math.round(100 * s.coverage)}%`;
  $("ret").textContent = `${s.returned.length}/${initMsg.n_drones}`;

  msg.drones.forEach((d, i) => {
    const g = drones[i];
    const p = v3(d.pos);
    if (g.lastT === null) {
      g.root.position.copy(p);
      g.yaw = d.yaw;
      g.ghost.position.copy(v3(d.odom));
    } else if (msg.t > g.lastT) {
      g.vel.copy(p).sub(g.target).divideScalar(msg.t - g.lastT);
    }
    g.lastT = msg.t;
    g.target.copy(p);
    g.targetYaw = d.yaw;
    g.odom.copy(v3(d.odom));
    g.battery = d.battery;

    if (d.status !== g.status) {
      if (seeded) {
        const home = Math.hypot(d.pos[0] - initMsg.base[0], d.pos[1] - initMsg.base[1]) <= initMsg.home_radius;
        if (d.status === "crashed") logEvent(`Drone ${i} crashed`, "bad");
        else if (d.status === "dead") logEvent(`Drone ${i} battery dead`, "bad");
        else if (d.status === "landing") logEvent(`Drone ${i} landing`);
        else if (d.status === "landed") logEvent(home ? `Drone ${i} returned home` : `Drone ${i} landed away from base`, home ? "good" : "warn");
      }
      g.status = d.status;
      g.alive = d.status === "flying" || d.status === "landing";
      g.led.emissiveIntensity = g.alive ? 3 : d.status === "landed" ? 1 : 0.1;
      g.lamp.intensity = g.alive ? LAMP : 0;
      g.rotors.forEach((r) => { r.disc.visible = g.alive; });
      if (d.status === "crashed") g.tilt.rotation.set(1.1, 0.5, 0);
      g.label.element.classList.toggle("dead", d.status === "crashed" || d.status === "dead");
    }
    if (g.alive && d.battery < 0.2 && !g.lowWarned) {
      g.lowWarned = true;
      if (seeded) logEvent(`Drone ${i} battery below 20%`, "warn");
    }

    if (d.lidar) addToMap(d.lidar);
    g.scan.visible = !!d.lidar && on("cLidar");
    if (g.scan.visible) setPoints(g.scan, d.lidar);

    const dbg = d.debug || {};
    const dv = on("cDebug");
    const hasPath = !!(dbg.path && dbg.path.length);
    if (hasPath) setLine2(g.path, [d.odom, ...dbg.path.slice().reverse()].flat());
    g.path.visible = dv && hasPath;
    g.goal.visible = dv && !!dbg.goal;
    if (dbg.goal) g.goal.position.copy(v3(dbg.goal));
    const hasPts = !!(dbg.points && dbg.points.length);
    if (hasPts) setPoints(g.pts, dbg.points.flat());
    g.pts.visible = dv && hasPts;

    const u = ui[i];
    u.st.textContent = d.status;
    u.st.className = `chip st ${d.status}`;
    u.link.classList.toggle("on", msg.base_links.includes(i));
    u.mode.textContent = dbg.mode ? `mode · ${dbg.mode}` : "";
    u.drift.textContent = `drift ${Math.hypot(d.pos[0] - d.odom[0], d.pos[1] - d.odom[1], d.pos[2] - d.odom[2]).toFixed(2)} m`;
    u.batt.style.width = `${100 * d.battery}%`;
    u.batt.style.background = d.battery > 0.4 ? "var(--good)" : d.battery > 0.2 ? "var(--warn)" : "var(--bad)";
    u.pct.textContent = `${Math.round(100 * d.battery)}%`;
  });

  for (const id of msg.found) {
    if (found.has(id)) continue;
    found.add(id);
    markFound(id, seeded);
  }
  seeded = true;
}

function onTrails(msg) {
  msg.true.forEach((tr, i) => {
    const g = drones[i];
    if (!g || tr.length < 2) return;
    const c = new THREE.Color(g.color), col = new Float32Array(tr.length * 3);
    for (let k = 0; k < tr.length; k++) {
      const f = 0.12 + 0.88 * (k / (tr.length - 1)) ** 1.5;     // older segments fade out
      col[k * 3] = c.r * f; col[k * 3 + 1] = c.g * f; col[k * 3 + 2] = c.b * f;
    }
    setLine2(g.trail, tr.flat(), col);
    g.hasTrail = true;
  });
  msg.odom.forEach((tr, i) => { if (drones[i] && tr.length >= 2) setLine2(drones[i].odomTrail, tr.flat()); });
  applyToggles();
}

function onDone(msg) {
  const s = msg.score;
  const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  const rows = [
    ["Artifacts found", `${s.artifacts_found} / ${s.artifacts_total}`, `+${10 * s.artifacts_found}`],
    ["Drones home", `${s.returned.length}`, `+${5 * s.returned.length}`],
    ["Coverage", `${(100 * s.coverage).toFixed(1)}%`, `+${(20 * s.coverage).toFixed(1)}`],
    null,
    ["Crashed / battery dead", `${s.crashed.length} / ${s.battery_dead.length}`, ""],
    ["Reports used", `${s.reports_used}`, ""],
    ["Total distance", `${s.distance_m.reduce((x, y) => x + y, 0).toFixed(0)} m`, ""],
    ["Mean odometry drift", `${avg(s.final_drift_m).toFixed(2)} m`, ""],
    ["Mission time", `${s.time_s} s`, ""],
  ];
  $("fScore").textContent = s.score.toFixed(1);
  $("fGrid").innerHTML = rows.map((r) => (r ? `<span>${r[0]}</span><span class="v">${r[1]}</span><span class="pts">${r[2]}</span>` : "<hr>")).join("");
  $("final").style.display = "block";
  logEvent(`Mission over · score ${s.score}`, "good");
}

// ---------------------------------------------------------------- view modes
function applyCave() {
  const o = $("opacity").value / 100;
  caveU.uOpacity.value = o;
  rockMat.opacity = o;
  rockMat.depthWrite = o >= 0.95;
  if (!cave) return;
  cave.visible = caveStyle !== "off";
  const mat = caveStyle === "rock" ? rockMat : holoMat;
  cave.traverse((m) => { if (m.isMesh) m.material = mat; });
}

function setCaveStyle(style, opacity) {
  caveStyle = style;
  segSelect("caveStyle", style);
  if (opacity !== undefined) $("opacity").value = opacity;
  applyCave();
}

function setCamMode(mode) {
  if (mode === camMode) return;
  const inside = (m) => m === "chase" || m === "fpv";
  if (inside(mode) && !inside(camMode)) {
    saved = { style: caveStyle, opacity: $("opacity").value, pos: camera.position.clone(), target: controls.target.clone() };
    setCaveStyle("rock", 100);
    lookAt.copy(controls.target);
  }
  if (!inside(mode) && inside(camMode)) {
    if (saved) {
      setCaveStyle(saved.style, saved.opacity);
      camera.position.copy(saved.pos);
      controls.target.copy(saved.target);
    }
  }
  camMode = mode;
  intro = null;
  segSelect("cam", mode);
  controls.enabled = !inside(mode);
  scene.fog.density = inside(mode) ? FOG.inside : FOG.outside;
  hemi.intensity = inside(mode) ? 0.05 : 1.1;
  sun.intensity = inside(mode) ? 0 : 1.6;
  bloom.threshold = inside(mode) ? 0.9 : 0.72;
  if (grid) grid.visible = !inside(mode);
}

function applyToggles() {
  drones.forEach((g) => {
    g.trail.visible = on("cTrail") && g.hasTrail;
    g.odomTrail.visible = on("cTrail") && on("cOdom") && g.hasTrail;
  });
  mapCloud.visible = on("cMap");
  bloom.enabled = on("cBloom");
  document.body.classList.toggle("no-labels", !on("cLabels"));
}

// ---------------------------------------------------------------- per frame
const lookAt = new THREE.Vector3(), _v = new THREE.Vector3(), _w = new THREE.Vector3();

function updateDrones(dt, t) {
  const k = damp(dt, 12);
  const inside = camMode === "chase" || camMode === "fpv";
  drones.forEach((g, i) => {
    g.root.position.lerp(g.target, k);
    g.yaw += wrap(g.targetYaw - g.yaw) * k;
    g.root.rotation.z = g.yaw;
    if (g.alive) {
      const c = Math.cos(g.yaw), s = Math.sin(g.yaw);
      const fwd = c * g.vel.x + s * g.vel.y, side = -s * g.vel.x + c * g.vel.y;
      g.tilt.rotation.y += (THREE.MathUtils.clamp(fwd * 0.1, -0.35, 0.35) - g.tilt.rotation.y) * damp(dt, 5);
      g.tilt.rotation.x += (THREE.MathUtils.clamp(-side * 0.1, -0.35, 0.35) - g.tilt.rotation.x) * damp(dt, 5);
      g.rotors.forEach((r) => { r.blade.rotation.z += r.dir * dt * 38; });
      if (g.battery < 0.2) g.led.emissiveIntensity = Math.sin(t * 9) > 0 ? 3 : 0.3;
    }
    const odomOn = on("cOdom");
    g.ghost.position.lerp(g.odom, k);
    g.ghost.visible = g.errLine.visible = odomOn && !inside;
    if (odomOn) {
      g.errLine.geometry.attributes.position.array.set([...g.root.position.toArray(), ...g.ghost.position.toArray()]);
      g.errLine.geometry.attributes.position.needsUpdate = true;
    }
    g.goal.rotation.z += dt * 1.5;

    const me = camMode === "fpv" && i === selected;
    g.model.visible = !me;
    g.beam.visible = !me && g.alive;
    g.label.visible = !me;
    const dist = camera.position.distanceTo(g.root.position);
    g.halo.visible = !me;
    g.halo.material.opacity = THREE.MathUtils.smoothstep(dist, 8, 30) * (g.alive ? 0.9 : 0.3);
  });

  const g = drones[selected];
  selRing.visible = !!g && !inside;
  if (g) {
    selRing.position.copy(g.root.position);
    selRing.position.z -= 0.35;
    selRing.scale.setScalar(1 + 0.12 * Math.sin(t * 4));
  }
}

function updateLinks(dt) {
  if (!linkLine) return;
  const arr = [];
  if (lastState && on("cLinks")) {
    const p = (a) => drones[a].root.position.toArray();
    for (const [a, b] of lastState.links) arr.push(...p(a), ...p(b));
    for (const a of lastState.base_links) arr.push(...p(a), ...initMsg.base);
  }
  linkLine.visible = arr.length > 0;
  if (!arr.length) return;
  setLine2(linkLine, arr);
  linkLine.material.dashOffset -= dt * 2.5;
}

function updateScene(dt, t) {
  const inside = camMode === "chase" || camMode === "fpv";
  for (const o of Object.values(artifactObjs)) {
    const isFound = found.has(o.a.id);
    o.group.visible = isFound || on("cTruth");
    o.label.visible = o.group.visible && (!inside || camera.position.distanceTo(o.group.position) < 15);
    o.mesh.rotation.z = t * 0.6 + o.phase;
    o.mesh.position.z = 0.35 + Math.sin(t * 1.6 + o.phase) * 0.08;
    if (isFound) o.glow.material.opacity = 0.55 + 0.25 * Math.sin(t * 3 + o.phase);
  }
  if (initMsg && t - lastPulse > 2.4) {
    lastPulse = t;
    burst(baseGround.getWorldPosition(_v).add(new THREE.Vector3(0, 0, 0.05)), 0x22d3ee, { from: 1, to: initMsg.home_radius * 1.8, life: 2.2 });
  }
  for (let k = effects.length - 1; k >= 0; k--) {
    const e = effects[k];
    e.t += dt;
    const u = e.t / e.life;
    if (u >= 1) {
      scene.remove(e.m);
      e.m.material.dispose();
      effects.splice(k, 1);
      continue;
    }
    e.m.scale.setScalar(e.from + (e.to - e.from) * (1 - (1 - u) ** 3));
    e.m.material.opacity = (1 - u) ** 2;
  }
}

function updateCamera(dt) {
  const g = drones[selected];
  if (camMode === "orbit" || !g) {
    if (intro) {
      intro.t += dt;
      const u = Math.min(intro.t / 3, 1), e = 1 - (1 - u) ** 3;
      _v.copy(intro.off).applyAxisAngle(camera.up, -1.1 * (1 - e)).multiplyScalar(1 + 1.2 * (1 - e));
      camera.position.copy(intro.c).add(_v);
      if (u >= 1) intro = null;
    }
    controls.update();
    return;
  }
  const p = g.root.position;
  if (camMode === "follow") {
    _v.copy(p).sub(controls.target).multiplyScalar(damp(dt, 6));
    controls.target.add(_v);
    camera.position.add(_v);
    controls.update();
    return;
  }
  const fwd = _w.set(Math.cos(g.yaw), Math.sin(g.yaw), 0);
  if (camMode === "chase") {
    const want = p.clone().addScaledVector(fwd, -3.4);
    want.z += 1.4;
    if (cave) {                       // pull in when a wall is between the drone and the camera
      const dir = want.clone().sub(p);
      const len = dir.length();
      ray.set(p, dir.divideScalar(len));
      ray.far = len;
      const hit = ray.intersectObject(cave, true)[0];
      if (hit) want.copy(p).addScaledVector(dir, Math.max(hit.distance - 0.4, 0.6));
    }
    camera.position.lerp(want, damp(dt, 5));
    lookAt.lerp(_v.copy(p).addScaledVector(fwd, 2.5), damp(dt, 8));
  } else {
    camera.position.copy(p).addScaledVector(fwd, 0.45);
    camera.position.z += 0.08;
    lookAt.copy(p).addScaledVector(fwd, 6);
    lookAt.z -= 0.4;
  }
  camera.lookAt(lookAt);
}

// ---------------------------------------------------------------- socket and controls
function send(obj) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }

function connect() {
  const port = new URLSearchParams(location.search).get("ws") || "8765";
  ws = new WebSocket(`ws://${location.hostname}:${port}`);
  ws.onopen = () => { $("status").textContent = "connected"; $("status").className = "conn on"; };
  ws.onclose = () => { $("status").textContent = "disconnected"; $("status").className = "conn off"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "init") onInit(msg);
    else if (msg.type === "state") onState(msg);
    else if (msg.type === "trails") onTrails(msg);
    else if (msg.type === "done") onDone(msg);
  };
}
connect();

function seg(id, pick) {
  $(id).addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) pick(b.dataset.v); });
}
function segSelect(id, v) { for (const b of $(id).children) b.classList.toggle("on", b.dataset.v === String(v)); }

let paused = false;
$("pause").onclick = () => {
  paused = !paused;
  send({ cmd: "pause", value: paused });
  $("pause").textContent = paused ? "Resume" : "Pause";
  $("pause").classList.toggle("paused", paused);
};
seg("speed", (v) => { send({ cmd: "speed", value: Number(v) }); segSelect("speed", v); });
seg("cam", setCamMode);
seg("caveStyle", (v) => setCaveStyle(v));
$("opacity").oninput = applyCave;
document.querySelectorAll("#dock input[type=checkbox]").forEach((c) => c.addEventListener("change", applyToggles));
$("final").onclick = () => { $("final").style.display = "none"; };
controls.addEventListener("start", () => { intro = null; });

const CAMS = ["orbit", "follow", "chase", "fpv"];
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  if (/^Digit[1-9]$/.test(e.code)) select(Number(e.code.slice(5)) - 1);
  else if (e.code === "KeyC") setCamMode(CAMS[(CAMS.indexOf(camMode) + 1) % CAMS.length]);
  else if (e.code === "KeyH") document.body.classList.toggle("ui-hidden");
  else if (e.code === "Space") { e.preventDefault(); $("pause").click(); }
});

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1), t = clock.elapsedTime;
  caveU.uTime.value = t;
  updateDrones(dt, t);
  updateLinks(dt);
  updateScene(dt, t);
  flushMap();
  updateCamera(dt);
  composer.render(dt);
  labelRenderer.render(scene, camera);
});
