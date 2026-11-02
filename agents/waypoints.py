"""Turn a keyboard flight log (agents.manual) into a waypoint route for agents.pid_follower.

    python -m agents.waypoints runs/manual_20261102_101500.jsonl
    python -m agents.waypoints runs/manual_20261102_101500.jsonl --step 3 --out runs/route.json

Keeps one logged position every STEP metres flown, plus the last one, and stops at the landing
(the log goes on after touchdown). The positions are obs.pos, the odometry the agent itself sees,
so the follower can aim at them directly.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_log(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path}: empty log")
    return rows


def thin(points, step):
    """Indices of points at least `step` metres apart along the flown path, first and last included."""
    keep, run = [0], 0.0
    for i in range(1, len(points)):
        run += float(np.linalg.norm(points[i] - points[i - 1]))
        if run >= step:
            keep.append(i)
            run = 0.0
    if keep[-1] != len(points) - 1:
        keep.append(len(points) - 1)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="runs/manual_*.jsonl written by agents.manual")
    ap.add_argument("--step", type=float, default=2.0, help="metres between waypoints")
    ap.add_argument("--out", default="runs/route.json")
    args = ap.parse_args()

    rows = load_log(args.log)
    # Cut the log where the drone stopped for good (landed): from there on the position no longer changes.
    pos = np.array([r["pos"] for r in rows], dtype=float)
    moved = np.flatnonzero(np.linalg.norm(np.diff(pos, axis=0), axis=1) > 0.01)
    end = int(moved[-1]) + 2 if len(moved) else 1
    rows, pos = rows[:end], pos[:end]

    keep = thin(pos, args.step)
    seen = [i for i, r in enumerate(rows) if r.get("survivor")]
    survivor_wp = next((k for k, i in enumerate(keep) if seen and i >= seen[0]), None)
    route = {
        "source": str(args.log),
        "step_m": args.step,
        "waypoints": pos[keep].round(2).tolist(),
        "times": [rows[i]["t"] for i in keep],
        "survivor_waypoint": survivor_wp,     # first waypoint at or after the survivor came into view
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(route, indent=1))
    flown = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    print(f"{len(keep)} waypoints, {flown:.0f} m flown in {rows[-1]['t']:.0f} s -> {out}")
    if survivor_wp is None:
        print("note: the survivor was never in view in this log")


if __name__ == "__main__":
    main()
