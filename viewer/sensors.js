// Sensor windows. One page, five views picked by ?sensor=lidar|imu|alt|odom|status.
// Each window opens its own WebSocket, asks for one drone's sensor stream, and draws it.
const params = new URLSearchParams(location.search);
const KIND = params.get("sensor") || "lidar";
let droneId = Number(params.get("drone") || 0);
const css = getComputedStyle(document.documentElement);
const C = (name) => css.getPropertyValue(name).trim();
const COL = { s1: C("--series-1"), s2: C("--series-2"), s3: C("--series-3"), truth: C("--truth"),
              grid: C("--grid"), muted: C("--text-muted"), text: C("--text-secondary"), surface: C("--surface-1") };
const TITLES = { lidar: "LiDAR (Ouster OS0-32)", imu: "IMU (accelerometer · gyro)", alt: "Altitude (barometer · downward rangefinder)",
                 odom: "Localisation (odometry vs truth)", status: "Status (comms · detections · attitude)" };
const main = document.getElementById("main");
document.title = `${TITLES[KIND]} - drone ${droneId}`;

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

function card(title, sub) {
  const c = el("div", "card");
  c.append(el("h2", null, title));
  if (sub) c.append(el("div", "sub", sub));
  main.append(c);
  return c;
}

function sizeCanvas(cv, h) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  cv.style.height = `${h}px`;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function niceStep(span, n = 4) {
  const raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
}

// ------------------------------------------------------------ time series chart
class TimeChart {
  constructor(parent, { series, unit, windowS = 10, height = 170, fixed = null }) {
    this.series = series;
    this.unit = unit;
    this.windowS = windowS;
    this.height = height;
    this.fixed = fixed;
    this.data = [];          // [t, v0, v1, ...]
    const lg = el("div", "legend");
    for (const s of series) lg.append(el("span", null, `<i class="${s.dash ? "dash" : ""}" style="background:${s.color}"></i>${s.name}`));
    parent.append(lg);
    this.cv = el("canvas");
    parent.append(this.cv);
    this.tip = el("div", "tip");
    parent.append(this.tip);
    this.hoverX = null;
    this.cv.addEventListener("mousemove", (e) => { this.hoverX = e.offsetX; this.draw(); });
    this.cv.addEventListener("mouseleave", () => { this.hoverX = null; this.tip.style.display = "none"; this.draw(); });
  }

  push(t, vals) {
    this.data.push([t, ...vals]);
    const cut = t - this.windowS;
    let k = 0;
    while (k < this.data.length && this.data[k][0] < cut) k++;
    if (k) this.data.splice(0, k);
  }

  clear() { this.data = []; }

  draw() {
    const { ctx, w, h } = sizeCanvas(this.cv, this.height);
    const L = 52, R = 10, T = 6, B = 22;
    ctx.clearRect(0, 0, w, h);
    if (!this.data.length) return;
    const tEnd = this.data[this.data.length - 1][0], t0 = tEnd - this.windowS;
    let lo = Infinity, hi = -Infinity;
    for (const row of this.data) for (let i = 1; i < row.length; i++) {
      const v = row[i];
      if (v == null || !isFinite(v)) continue;
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
    if (this.fixed) [lo, hi] = [Math.min(lo, this.fixed[0]), Math.max(hi, this.fixed[1])];
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-6) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const X = (t) => L + (t - t0) / this.windowS * (w - L - R);
    const Y = (v) => T + (hi - v) / (hi - lo) * (h - T - B);

    // recessive grid + axis labels in muted ink
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = COL.muted;
    ctx.strokeStyle = COL.grid;
    ctx.lineWidth = 1;
    const step = niceStep(hi - lo);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
      ctx.fillText(`${+v.toFixed(4)}`, L - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    const tStep = niceStep(this.windowS, 6);
    for (let s = 0; s <= this.windowS; s += tStep) ctx.fillText(s === 0 ? "now" : `-${s}s`, X(tEnd - s), h - B + 5);
    ctx.save(); ctx.translate(11, (h - B) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
    ctx.fillText(this.unit, 0, 0); ctx.restore();

    this.series.forEach((s, i) => {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.setLineDash(s.dash ? [5, 4] : []);
      ctx.beginPath();
      let pen = false;
      for (const row of this.data) {
        const v = row[i + 1];
        if (v == null || !isFinite(v)) { pen = false; continue; }
        const x = X(row[0]), y = Y(v);
        pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        pen = true;
      }
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // crosshair + tooltip
    if (this.hoverX != null && this.hoverX > L) {
      const tq = t0 + (this.hoverX - L) / (w - L - R) * this.windowS;
      let best = this.data[0];
      for (const row of this.data) if (Math.abs(row[0] - tq) < Math.abs(best[0] - tq)) best = row;
      const x = X(best[0]);
      ctx.strokeStyle = COL.text; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke();
      this.series.forEach((s, i) => {
        const v = best[i + 1];
        if (v == null || !isFinite(v)) return;
        ctx.fillStyle = s.color; ctx.strokeStyle = COL.surface; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, Y(v), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      });
      const rows = this.series.map((s, i) => {
        const v = best[i + 1];
        return `<div><span class="status-dot" style="background:${s.color}"></span>${s.name}: <b>${v == null || !isFinite(v) ? "none" : v.toFixed(3)}</b> ${this.unit}</div>`;
      }).join("");
      this.tip.innerHTML = `<div class="muted">${(best[0] - tEnd).toFixed(2)} s</div>${rows}`;
      this.tip.style.display = "block";
      const left = Math.min(x + 12, w - 190);
      this.tip.style.left = `${left + 10}px`;
      this.tip.style.top = `${this.cv.offsetTop + 8}px`;
    }
  }
}

// ------------------------------------------------------------ sequential blue ramp (dark mode: near = dark)
const RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
              "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"].map((h) =>
  [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
function rampColor(f) {
  const x = Math.max(0, Math.min(1, f)) * (RAMP.length - 1), i = Math.floor(x), u = x - i;
  const a = RAMP[i], b = RAMP[Math.min(i + 1, RAMP.length - 1)];
  return a.map((v, k) => Math.round(v + (b[k] - v) * u));
}

// ------------------------------------------------------------ views
const views = {
  lidar() {
    const c1 = card("Range image", "Horizontal: azimuth −180° to 180° (centre is the nose). Vertical: 32 channels from +45° (top) to −45° (bottom). Darker is closer; background colour means no return.");
    const img = el("canvas"); c1.append(img);
    const scale = el("canvas"); c1.append(scale);
    const tip = el("div", "tip"); c1.append(tip);
    const stats = el("div", "stats"); c1.append(stats);
    const c2 = card("Planar scan (body frame, top view)", "Only beams within ±6° of horizontal. Rings every 5 m, nose points up. The LiDAR is fixed to the airframe, so the scan tilts when the drone tilts.");
    const scan = el("canvas"); c2.append(scan);
    let last = null, hover = null;
    img.addEventListener("mousemove", (e) => { hover = [e.offsetX, e.offsetY]; render(); });
    img.addEventListener("mouseleave", () => { hover = null; tip.style.display = "none"; });

    function render() {
      if (!last || !last.lidar_cm) return;
      const m = last.lidar_meta, ch = m.channels, n = m.azimuth_steps, r = last.lidar_cm;
      const { ctx, w } = sizeCanvas(img, Math.max(120, Math.round(ch * 6)));
      const h = Math.max(120, Math.round(ch * 6));
      const off = new OffscreenCanvas(n, ch), octx = off.getContext("2d"), id = octx.createImageData(n, ch);
      const sbg = [21, 26, 33];
      let hits = 0, minR = Infinity;
      for (let row = 0; row < ch; row++) {
        const src = ch - 1 - row;           // top row = highest elevation
        for (let col = 0; col < n; col++) {
          const v = r[src * n + col] / 100, k = (row * n + col) * 4;
          const rgb = v > 0 ? rampColor(v / m.max_range) : sbg;
          if (v > 0) { hits++; minR = Math.min(minR, v); }
          id.data[k] = rgb[0]; id.data[k + 1] = rgb[1]; id.data[k + 2] = rgb[2]; id.data[k + 3] = 255;
        }
      }
      octx.putImageData(id, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(off, 0, 0, w, h);
      ctx.strokeStyle = COL.text; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(w / 2 + 0.5, 0); ctx.lineTo(w / 2 + 0.5, h); ctx.stroke(); ctx.setLineDash([]);

      // scale bar = legend for the sequential encoding
      const s = sizeCanvas(scale, 30);
      s.ctx.clearRect(0, 0, s.w, 30);
      for (let x = 0; x < s.w - 80; x++) { const c = rampColor(x / (s.w - 80)); s.ctx.fillStyle = `rgb(${c})`; s.ctx.fillRect(x, 6, 1, 8); }
      s.ctx.fillStyle = COL.muted; s.ctx.font = "11px system-ui"; s.ctx.textBaseline = "top";
      s.ctx.textAlign = "left"; s.ctx.fillText("0 m", 0, 17);
      s.ctx.textAlign = "right"; s.ctx.fillText(`${m.max_range} m`, s.w - 80, 17);
      s.ctx.textAlign = "left"; s.ctx.fillStyle = COL.muted; s.ctx.fillRect(s.w - 70, 6, 10, 8);
      s.ctx.fillStyle = "rgb(21,26,33)"; s.ctx.fillRect(s.w - 69, 7, 8, 6);
      s.ctx.fillStyle = COL.muted; s.ctx.fillText("no return", s.w - 56, 4);

      stats.innerHTML = `<span>returns <b>${hits}</b> / ${ch * n} (${(100 * hits / (ch * n)).toFixed(1)}%)</span>` +
        `<span>nearest <b>${isFinite(minR) ? minR.toFixed(2) : "-"}</b> m</span>` +
        `<span>beams ${ch} × ${n}, vertical FOV ${m.vfov_deg[0]}° to ${m.vfov_deg[1]}°</span>`;

      if (hover) {
        const col = Math.floor(hover[0] / w * n), row = Math.floor(hover[1] / h * ch), src = ch - 1 - row;
        if (col >= 0 && col < n && row >= 0 && row < ch) {
          const v = r[src * n + col] / 100;
          const az = -180 + 360 * col / n, elv = m.vfov_deg[0] + (m.vfov_deg[1] - m.vfov_deg[0]) * src / (ch - 1);
          tip.innerHTML = `azimuth <b>${az.toFixed(1)}°</b> · elevation <b>${elv.toFixed(1)}°</b><br>range <b>${v > 0 ? v.toFixed(2) + " m" : "no return"}</b>`;
          tip.style.display = "block";
          tip.style.left = `${Math.min(hover[0] + 24, w - 150)}px`;
          tip.style.top = `${img.offsetTop + hover[1] + 12}px`;
        }
      }

      // top-down scan of near-horizontal beams
      const sc = sizeCanvas(scan, 360), cx = sc.w / 2, cy = 180, px = 160 / 20;
      sc.ctx.clearRect(0, 0, sc.w, 360);
      sc.ctx.strokeStyle = COL.grid; sc.ctx.fillStyle = COL.muted; sc.ctx.font = "11px system-ui";
      for (let rr = 5; rr <= 20; rr += 5) {
        sc.ctx.beginPath(); sc.ctx.arc(cx, cy, rr * px, 0, 2 * Math.PI); sc.ctx.stroke();
        sc.ctx.fillText(`${rr} m`, cx + rr * px * 0.71 + 3, cy - rr * px * 0.71);
      }
      sc.ctx.fillStyle = COL.s1;
      for (let src = 0; src < ch; src++) {
        const elv = m.vfov_deg[0] + (m.vfov_deg[1] - m.vfov_deg[0]) * src / (ch - 1);
        if (Math.abs(elv) > 6) continue;
        for (let col = 0; col < n; col++) {
          const v = r[src * n + col] / 100;
          if (!(v > 0)) continue;
          const az = -Math.PI + 2 * Math.PI * col / n, d = v * Math.cos(elv * Math.PI / 180);
          sc.ctx.fillRect(cx - d * Math.sin(az) * px - 1, cy - d * Math.cos(az) * px - 1, 2.5, 2.5);
        }
      }
      sc.ctx.fillStyle = COL.text;
      sc.ctx.beginPath(); sc.ctx.moveTo(cx, cy - 9); sc.ctx.lineTo(cx - 6, cy + 6); sc.ctx.lineTo(cx + 6, cy + 6); sc.ctx.fill();
    }
    return { update(p) { last = p; render(); }, reset() { last = null; } };
  },

  imu() {
    const a = card("Accelerometer (specific force, body frame)", "z ≈ +9.81 m/s² in hover. x and y stay small even when the drone tilts to accelerate, because thrust is along the body z axis. Raw 240 Hz samples.");
    const acc = new TimeChart(a, { unit: "m/s²", series: [{ name: "x (forward)", color: COL.s1 }, { name: "y (left)", color: COL.s2 }, { name: "z (up)", color: COL.s3 }] });
    const g = card("Gyro (angular rate, body frame)", "White noise + turn-on bias + slowly wandering bias (ADIS16448 model).");
    const gyr = new TimeChart(g, { unit: "rad/s", series: [{ name: "x (roll)", color: COL.s1 }, { name: "y (pitch)", color: COL.s2 }, { name: "z (yaw)", color: COL.s3 }] });
    return {
      update(p) { for (const s of p.imu || []) { acc.push(s[0], [s[1], s[2], s[3]]); gyr.push(s[0], [s[4], s[5], s[6]]); } acc.draw(); gyr.draw(); },
      reset() { acc.clear(); gyr.clear(); },
    };
  },

  alt() {
    const b = card("Barometric altitude (relative to take-off)", "Barometer reading and true altitude (instructor ground truth, dashed). 0.1 m sample noise plus weather/temperature drift.");
    const baro = new TimeChart(b, { unit: "m", windowS: 30, series: [{ name: "barometer", color: COL.s1 }, { name: "true altitude", color: COL.truth, dash: true }] });
    const r = card("Downward rangefinder (distance to floor)", "TFmini Plus model: 0.1–12 m, tilts with the airframe. The line breaks when out of range.");
    const rng = new TimeChart(r, { unit: "m", windowS: 30, series: [{ name: "range down", color: COL.s2 }] });
    return {
      update(p) { baro.push(p.obs_t, [p.baro_alt, p.truth.z]); rng.push(p.obs_t, [p.range_down]); baro.draw(); rng.draw(); },
      reset() { baro.clear(); rng.clear(); },
    };
  },

  odom() {
    const grid = el("div", "two"); main.append(grid);
    const c1 = el("div", "card"); grid.append(c1);
    c1.append(el("h2", null, "Trajectory (top view)"));
    c1.append(el("div", "sub", "Position estimate the agent receives vs true position (instructor ground truth)."));
    const lg = el("div", "legend", `<span><i style="background:${COL.s1}"></i>estimate (odometry)</span><span><i class="dash"></i>truth</span>`);
    c1.append(lg);
    const top = el("canvas"); c1.append(top);
    const c2 = el("div", "card"); grid.append(c2);
    c2.append(el("h2", null, "Localisation error"));
    c2.append(el("div", "sub", "|estimate − truth|. Grows with distance flown and time."));
    const err = new TimeChart(c2, { unit: "m", windowS: 120, height: 300, fixed: [0, 0.5], series: [{ name: "error", color: COL.s1 }] });
    const note = el("div", "card", ""); note.style.display = "none"; main.append(note);
    let est = [], tru = [];
    function drawTop() {
      const { ctx, w, h } = sizeCanvas(top, 300);
      ctx.clearRect(0, 0, w, h);
      const all = est.concat(tru);
      if (!all.length) return;
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of all) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
      const span = Math.max(x1 - x0, y1 - y0, 10) * 1.1, s = Math.min(w, h) / span;
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      const P = (p) => [w / 2 + (p[0] - mx) * s, h / 2 - (p[1] - my) * s];
      const line = (pts, color, dash) => {
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash ? [5, 4] : []);
        ctx.beginPath(); pts.forEach((p, i) => { const [x, y] = P(p); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
      };
      line(tru, COL.truth, true);
      line(est, COL.s1, false);
      ctx.setLineDash([]);
      ctx.fillStyle = COL.muted; ctx.font = "11px system-ui";
      const bar = niceStep(span, 5);
      ctx.fillRect(10, h - 14, bar * s, 2); ctx.fillText(`${bar} m`, 10, h - 20);
      const [bx, by] = P([0, 0]);
      ctx.strokeStyle = COL.text; ctx.lineWidth = 1; ctx.strokeRect(bx - 4, by - 4, 8, 8);
      ctx.fillText("base", bx + 7, by - 6);
    }
    return {
      update(p) {
        if (!p.odom) {
          note.style.display = "block";
          note.innerHTML = "<b>Odometry is off</b> (<code>--odom off</code>). The agent must estimate its own position from IMU and LiDAR. Showing the true trajectory only.";
        }
        tru.push(p.truth.pos);
        if (p.odom) { est.push(p.odom); err.push(p.obs_t, [Math.hypot(...p.odom.map((v, i) => v - p.truth.pos[i]))]); }
        if (tru.length > 6000) { tru = tru.slice(-6000); est = est.slice(-6000); }
        drawTop(); err.draw();
      },
      reset() { est = []; tru = []; err.clear(); },
    };
  },

  status() {
    const c = card("Current status");
    const body = el("div"); c.append(body);
    const d = card("Artifact detections (this tick)", "Type and body-frame relative position reported by the detector (noisy).");
    const det = el("div"); d.append(det);
    const deg = (r) => (r * 180 / Math.PI).toFixed(1);
    return {
      update(p) {
        const ok = (b) => `<span class="status-dot" style="background:${b ? C("--good") : C("--critical")}"></span>${b ? "up" : "down"}`;
        body.innerHTML = `<table>
          <tr><th>Status</th><td>${p.status}</td><th>Battery</th><td>${(100 * p.battery).toFixed(0)}%</td></tr>
          <tr><th>Base link</th><td>${ok(p.base_link)}</td><th>Neighbours</th><td>${(p.neighbors || []).join(", ") || "none"}</td></tr>
          <tr><th>Roll (est / true)</th><td>${deg(p.roll)}° / ${deg(p.truth.rp[0])}°</td>
              <th>Pitch (est / true)</th><td>${deg(p.pitch)}° / ${deg(p.truth.rp[1])}°</td></tr>
          <tr><th>Yaw (est / true)</th><td>${deg(p.yaw)}° / ${deg(p.truth.yaw)}°</td>
              <th>Range down / baro alt</th><td>${p.range_down == null ? "out of range" : p.range_down.toFixed(2) + " m"} / ${p.baro_alt.toFixed(2)} m</td></tr>
        </table>`;
        det.innerHTML = (p.detections || []).length
          ? `<table><tr><th>Type</th><th>x (forward)</th><th>y (left)</th><th>z (up)</th></tr>${p.detections.map((x) =>
              `<tr><td>${x.type}</td><td>${x.rel[0]}</td><td>${x.rel[1]}</td><td>${x.rel[2]}</td></tr>`).join("")}</table>`
          : `<span class="muted">no artifact in view</span>`;
      },
      reset() {},
    };
  },
};

const view = views[KIND]();
document.getElementById("title").textContent = TITLES[KIND];

// ------------------------------------------------------------ socket
let ws, lastObsT = null, nDrones = 0;
const sel = document.getElementById("drone");
sel.onchange = () => {
  droneId = Number(sel.value);
  document.title = `${TITLES[KIND]} - drone ${droneId}`;
  lastObsT = null;
  view.reset();
  ws.send(JSON.stringify({ cmd: "sensors", drone: droneId }));
};

function connect() {
  const port = params.get("ws") || "8765";
  ws = new WebSocket(`ws://${location.hostname}:${port}`);
  ws.onopen = () => ws.send(JSON.stringify({ cmd: "sensors", drone: droneId }));
  ws.onclose = () => { document.getElementById("clock").textContent = "disconnected (server stopped?)"; };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "init" && msg.n_drones !== nDrones) {
      nDrones = msg.n_drones;
      sel.innerHTML = [...Array(nDrones).keys()].map((i) => `<option value="${i}" ${i === droneId ? "selected" : ""}>${i}</option>`).join("");
    } else if (msg.type === "sensors" && msg.drone === droneId) {
      document.getElementById("clock").textContent = `t = ${msg.t.toFixed(1)} s · ${msg.status}`;
      if (msg.obs_t == null || msg.obs_t === lastObsT) return;   // drone stopped sensing (landed/crashed) or duplicate
      lastObsT = msg.obs_t;
      view.update(msg);
    }
  };
}
connect();
addEventListener("resize", () => lastObsT = null);
