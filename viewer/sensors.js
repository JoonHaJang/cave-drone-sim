// Sensor windows. One page, five views picked by ?sensor=lidar|imu|alt|odom|status.
// Each window opens its own WebSocket, asks for one drone's sensor stream, and draws it.
const params = new URLSearchParams(location.search);
const KIND = params.get("sensor") || "lidar";
let droneId = Number(params.get("drone") || 0);
const css = getComputedStyle(document.documentElement);
const C = (name) => css.getPropertyValue(name).trim();
const COL = { s1: C("--series-1"), s2: C("--series-2"), s3: C("--series-3"), truth: C("--truth"),
              grid: C("--grid"), muted: C("--text-muted"), text: C("--text-secondary"), surface: C("--surface-1") };
const TITLES = { lidar: "LiDAR (Ouster OS0-32)", imu: "IMU (가속도·자이로)", alt: "고도 (기압계·하방 거리센서)",
                 odom: "위치 추정 (오도메트리 vs 실제)", status: "상태 (통신·탐지·자세)" };
const main = document.getElementById("main");
document.title = `${TITLES[KIND]} - 드론 ${droneId}`;

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
    for (let s = 0; s <= this.windowS; s += tStep) ctx.fillText(s === 0 ? "지금" : `-${s}s`, X(tEnd - s), h - B + 5);
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
        return `<div><span class="status-dot" style="background:${s.color}"></span>${s.name}: <b>${v == null || !isFinite(v) ? "없음" : v.toFixed(3)}</b> ${this.unit}</div>`;
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
    const c1 = card("거리 영상 (range image)", "가로: 방위각 −180°~180° (가운데가 기체 앞), 세로: 32채널 +45°(위)~−45°(아래). 어두울수록 가깝고, 바탕색은 반사 없음.");
    const img = el("canvas"); c1.append(img);
    const scale = el("canvas"); c1.append(scale);
    const tip = el("div", "tip"); c1.append(tip);
    const stats = el("div", "stats"); c1.append(stats);
    const c2 = card("평면 스캔 (기체 기준, 위에서 본 모습)", "수평 ±6° 안의 빔만 표시. 원은 5 m 간격, 위쪽이 기체 앞. LiDAR가 기체에 고정돼 있어 기울면 스캔도 기울어집니다.");
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
      s.ctx.fillStyle = COL.muted; s.ctx.fillText("반사 없음", s.w - 56, 4);

      stats.innerHTML = `<span>반사 <b>${hits}</b> / ${ch * n} (${(100 * hits / (ch * n)).toFixed(1)}%)</span>` +
        `<span>최단 거리 <b>${isFinite(minR) ? minR.toFixed(2) : "-"}</b> m</span>` +
        `<span>빔 ${ch} × ${n}, 수직 시야 ${m.vfov_deg[0]}°~${m.vfov_deg[1]}°</span>`;

      if (hover) {
        const col = Math.floor(hover[0] / w * n), row = Math.floor(hover[1] / h * ch), src = ch - 1 - row;
        if (col >= 0 && col < n && row >= 0 && row < ch) {
          const v = r[src * n + col] / 100;
          const az = -180 + 360 * col / n, elv = m.vfov_deg[0] + (m.vfov_deg[1] - m.vfov_deg[0]) * src / (ch - 1);
          tip.innerHTML = `방위 <b>${az.toFixed(1)}°</b> · 고각 <b>${elv.toFixed(1)}°</b><br>거리 <b>${v > 0 ? v.toFixed(2) + " m" : "반사 없음"}</b>`;
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
    const a = card("가속도계 (비력, 기체 좌표)", "정지 비행 중 z ≈ +9.81 m/s². 기체가 기울어 가속해도 x·y는 작게 나옵니다(추력이 기체 z축 방향이기 때문). 240 Hz 원시 샘플.");
    const acc = new TimeChart(a, { unit: "m/s²", series: [{ name: "x (앞)", color: COL.s1 }, { name: "y (왼쪽)", color: COL.s2 }, { name: "z (위)", color: COL.s3 }] });
    const g = card("자이로 (각속도, 기체 좌표)", "백색잡음 + 켜질 때의 바이어스 + 천천히 변하는 바이어스 (ADIS16448 기준).");
    const gyr = new TimeChart(g, { unit: "rad/s", series: [{ name: "x (롤)", color: COL.s1 }, { name: "y (피치)", color: COL.s2 }, { name: "z (요)", color: COL.s3 }] });
    return {
      update(p) { for (const s of p.imu || []) { acc.push(s[0], [s[1], s[2], s[3]]); gyr.push(s[0], [s[4], s[5], s[6]]); } acc.draw(); gyr.draw(); },
      reset() { acc.clear(); gyr.clear(); },
    };
  },

  alt() {
    const b = card("기압 고도 (이륙 지점 기준)", "기압계 측정값과 실제 고도(교수자용 정답, 점선). 측정 잡음 0.1 m에 날씨·온도로 인한 드리프트가 쌓입니다.");
    const baro = new TimeChart(b, { unit: "m", windowS: 30, series: [{ name: "기압계", color: COL.s1 }, { name: "실제 고도", color: COL.truth, dash: true }] });
    const r = card("하방 거리센서 (바닥까지 거리)", "TFmini Plus 기준: 0.1~12 m, 기체와 함께 기울어집니다. 범위를 벗어나면 선이 끊깁니다.");
    const rng = new TimeChart(r, { unit: "m", windowS: 30, series: [{ name: "하방 거리", color: COL.s2 }] });
    return {
      update(p) { baro.push(p.obs_t, [p.baro_alt, p.truth.z]); rng.push(p.obs_t, [p.range_down]); baro.draw(); rng.draw(); },
      reset() { baro.clear(); rng.clear(); },
    };
  },

  odom() {
    const grid = el("div", "two"); main.append(grid);
    const c1 = el("div", "card"); grid.append(c1);
    c1.append(el("h2", null, "궤적 (위에서 본 모습)"));
    c1.append(el("div", "sub", "에이전트가 받는 추정 위치와 실제 위치(교수자용 정답)."));
    const lg = el("div", "legend", `<span><i style="background:${COL.s1}"></i>추정 (오도메트리)</span><span><i class="dash"></i>실제</span>`);
    c1.append(lg);
    const top = el("canvas"); c1.append(top);
    const c2 = el("div", "card"); grid.append(c2);
    c2.append(el("h2", null, "위치 추정 오차"));
    c2.append(el("div", "sub", "|추정 − 실제|. 이동 거리와 시간에 따라 커집니다."));
    const err = new TimeChart(c2, { unit: "m", windowS: 120, height: 300, fixed: [0, 0.5], series: [{ name: "오차", color: COL.s1 }] });
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
      ctx.fillText("기지", bx + 7, by - 6);
    }
    return {
      update(p) {
        if (!p.odom) {
          note.style.display = "block";
          note.innerHTML = "<b>오도메트리가 꺼져 있습니다</b> (<code>--odom off</code>). 에이전트가 IMU와 LiDAR로 직접 위치를 추정해야 합니다. 실제 궤적만 표시합니다.";
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
    const c = card("현재 상태");
    const body = el("div"); c.append(body);
    const d = card("목표물 탐지 (이번 틱)", "탐지기가 보고한 종류와 기체 기준 상대 위치(잡음 포함).");
    const det = el("div"); d.append(det);
    const deg = (r) => (r * 180 / Math.PI).toFixed(1);
    return {
      update(p) {
        const ok = (b) => `<span class="status-dot" style="background:${b ? C("--good") : C("--critical")}"></span>${b ? "연결" : "끊김"}`;
        body.innerHTML = `<table>
          <tr><th>상태</th><td>${p.status}</td><th>배터리</th><td>${(100 * p.battery).toFixed(0)}%</td></tr>
          <tr><th>기지 통신</th><td>${ok(p.base_link)}</td><th>통신 중인 드론</th><td>${(p.neighbors || []).join(", ") || "없음"}</td></tr>
          <tr><th>롤 (추정 / 실제)</th><td>${deg(p.roll)}° / ${deg(p.truth.rp[0])}°</td>
              <th>피치 (추정 / 실제)</th><td>${deg(p.pitch)}° / ${deg(p.truth.rp[1])}°</td></tr>
          <tr><th>요 (추정 / 실제)</th><td>${deg(p.yaw)}° / ${deg(p.truth.yaw)}°</td>
              <th>하방 거리 / 기압 고도</th><td>${p.range_down == null ? "범위 밖" : p.range_down.toFixed(2) + " m"} / ${p.baro_alt.toFixed(2)} m</td></tr>
        </table>`;
        det.innerHTML = (p.detections || []).length
          ? `<table><tr><th>종류</th><th>x (앞)</th><th>y (왼쪽)</th><th>z (위)</th></tr>${p.detections.map((x) =>
              `<tr><td>${x.type}</td><td>${x.rel[0]}</td><td>${x.rel[1]}</td><td>${x.rel[2]}</td></tr>`).join("")}</table>`
          : `<span class="muted">시야에 목표물 없음</span>`;
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
  document.title = `${TITLES[KIND]} - 드론 ${droneId}`;
  lastObsT = null;
  view.reset();
  ws.send(JSON.stringify({ cmd: "sensors", drone: droneId }));
};

function connect() {
  const port = params.get("ws") || "8765";
  ws = new WebSocket(`ws://${location.hostname}:${port}`);
  ws.onopen = () => ws.send(JSON.stringify({ cmd: "sensors", drone: droneId }));
  ws.onclose = () => { document.getElementById("clock").textContent = "연결 끊김 (서버 종료?)"; };
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
