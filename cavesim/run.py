"""Run a mission with the browser viewer.

    python -m cavesim.run --agent agents.explorer --map medium_1 --drones 3
    then open http://localhost:8000/viewer/

--agent takes a module (its `Agent` class is used) or module:ClassName.
Use --headless to skip the viewer and just print the score.
"""
import argparse
import asyncio
import functools
import importlib
import json
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.asyncio.server import serve

from .api import PILOT
from .config import SimConfig
from .sim import Simulator

ROOT = Path(__file__).resolve().parent.parent
HISTORY_FULL = 10         # replay history: LiDAR and debug kept every this many ticks
HISTORY_LIDAR_PTS = 600   # ... thinned to this many points (as the viewer's own replay)


def save_record(sim, args):
    """--record: the flown trails and the score, as JSON."""
    if args.record:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"map": args.map, "agent": args.agent, "seed": args.seed, "t": sim.t,
                                   "score": sim.score(), **sim.trails()}))
        print("recorded", out)


def load_agent(spec):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls or "Agent")


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def start_http(port):
    handler = functools.partial(QuietHandler, directory=str(ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


class Session:
    def __init__(self, sim, speed):
        self.sim = sim
        self.speed = speed          # 1.0 = real time, 0 = as fast as possible
        self.paused = False
        self.clients = set()        # 3D viewer windows
        self.sensor_clients = {}    # sensor windows: websocket -> drone id
        self.done = False
        self.history = []           # every state so far, for a viewer that joins late or reloads

    def keep(self, state):
        """Store a state for replay: the LiDAR (thinned) and the agent's debug only every
        HISTORY_FULL-th tick, the poses every tick (a 1200 s run: ~25 MB, not ~150)."""
        full = self.sim.tick % HISTORY_FULL == 0
        drones = []
        for d in state["drones"]:
            d = dict(d)
            if full and "lidar" in d:
                pts = d["lidar"]
                step = max(1, -(-len(pts) // 3 // HISTORY_LIDAR_PTS))
                d["lidar"] = [v for k in range(0, len(pts) - 2, 3 * step) for v in pts[k:k + 3]]
            elif not full:
                d.pop("lidar", None)
                d.pop("debug", None)
            drones.append(d)
        self.history.append({**state, "drones": drones})

    async def handler(self, ws):
        try:
            await ws.send(json.dumps(self.sim.viewer_init()))
            async for raw in ws:
                msg = json.loads(raw)
                cmd = msg.get("cmd")
                if cmd == "viewer":
                    if self.history:
                        await ws.send(json.dumps({"type": "history", "states": self.history}))
                    self.clients.add(ws)
                    await ws.send(json.dumps(self.sim.trails()))
                    await ws.send(json.dumps(self.sim.viewer_state()))
                elif cmd == "sensors":
                    self.sensor_clients[ws] = int(msg.get("drone", 0))
                    await ws.send(json.dumps(self.sim.sensor_packet(self.sensor_clients[ws])))
                elif cmd == "pause":
                    self.paused = bool(msg.get("value", not self.paused))
                elif cmd == "speed":
                    self.speed = float(msg["value"])
                elif cmd == "keys":
                    PILOT["keys"] = frozenset(msg.get("keys", []))
        finally:
            self.clients.discard(ws)
            self.sensor_clients.pop(ws, None)

    def broadcast(self, msg):
        data = json.dumps(msg)
        for ws in list(self.clients):
            asyncio.ensure_future(self._send(ws, data))

    def send_sensors(self):
        packets = {}
        for ws, i in list(self.sensor_clients.items()):
            if i not in packets:
                packets[i] = json.dumps(self.sim.sensor_packet(i))
            asyncio.ensure_future(self._send(ws, packets[i]))

    async def _send(self, ws, data):
        try:
            await ws.send(data)
        except Exception:
            self.clients.discard(ws)
            self.sensor_clients.pop(ws, None)

    async def loop(self):
        sim = self.sim
        tick_dt = 1.0 / sim.cfg.agent_hz
        while not self.done:
            if self.paused:
                await asyncio.sleep(0.05)
                continue
            t0 = time.perf_counter()
            running = sim.step()
            state = sim.viewer_state()
            self.keep(state)
            if self.clients:
                self.broadcast(state)
                if sim.tick % 10 == 0:
                    self.broadcast(sim.trails())
            if self.sensor_clients:
                self.send_sensors()
            if not running:
                self.done = True
                score = sim.score()
                self.broadcast(sim.trails())
                self.broadcast({"type": "done", "score": score})
                print("mission over:", json.dumps(score, indent=1))
                break
            wait = tick_dt / self.speed - (time.perf_counter() - t0) if self.speed > 0 else 0
            await asyncio.sleep(max(wait, 0))


async def serve_viewer(sim, args):
    session = Session(sim, args.speed)
    httpd = start_http(args.http_port)
    url = f"http://localhost:{args.http_port}/viewer/?ws={args.ws_port}"
    async with serve(session.handler, "127.0.0.1", args.ws_port, max_size=None):
        print(f"viewer: {url}   (Ctrl+C to quit)")
        if not args.no_browser:
            webbrowser.open(url)
        if args.wait:
            print("waiting for the viewer to connect...")
            while not (session.clients or session.sensor_clients):
                await asyncio.sleep(0.1)
        await session.loop()
        save_record(sim, args)
        print("mission finished; viewer stays up until Ctrl+C")
        await asyncio.Future()
    httpd.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="agents.explorer")
    ap.add_argument("--map", default="medium_1", help="folder name under maps/ or a path")
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--odom", choices=["perfect", "default", "hard", "off"], default="default",
                    help="LiDAR-inertial odometry quality; off = no pos/vel, build your own")
    ap.add_argument("--lidar", choices=["os0-32", "lite", "lite-top", "lite-center", "ideal"], default="os0-32",
                    help="os0-32 = Ouster OS0-32 spec (slow), lite = 16x256 beams (about 4x faster)")
    ap.add_argument("--imu", choices=["flight", "datasheet"], default="flight",
                    help="flight = with rotor vibration and calibration errors, datasheet = bench noise only")
    ap.add_argument("--aero", choices=["off", "default", "hard"], default="default",
                    help="gusts, wall turbulence, ceiling/ground effect near rock")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="sim speed x real time, 0 = max")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--wait", action="store_true", help="start only after the viewer connects")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--http-port", type=int, default=8000)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--target-error", type=float, default=0.0,
                    help="rescue: the survivor position given to the agent is off by this many metres")
    ap.add_argument("--record", help="save the flown trails and the score to this JSON file at the end")
    args = ap.parse_args()

    map_dir = Path(args.map)
    if not map_dir.exists():
        map_dir = ROOT / "maps" / args.map
    sim = Simulator(map_dir, args.drones, load_agent(args.agent), cfg=SimConfig(target_error_m=args.target_error),
                    odom=args.odom, lidar=args.lidar,
                    seed=args.seed, aero=args.aero, imu=args.imu)
    try:
        if args.headless:
            while sim.step():
                if sim.tick % (30 * sim.cfg.agent_hz) == 0:
                    print(f"t={sim.t:4.0f}s score={sim.score()['score']}")
            print(json.dumps(sim.score(), indent=1))
        else:
            asyncio.run(serve_viewer(sim, args))
    except KeyboardInterrupt:
        pass
    finally:
        save_record(sim, args)          # also when stopped early (Ctrl+C)
        sim.close()


if __name__ == "__main__":
    main()
