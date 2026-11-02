"""Batch headless runs in parallel, one row per mission, plus a lap-time summary.

    python -m agents.eval --agent agents.team1 --maps small_2 medium_1 --seeds 0 1 --odom default hard

A run "finishes" only if every artifact was accepted and every drone landed home; its lap
time is the sim time when the last drone stopped. Anything else is a DNF.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cavesim.run import load_agent
from cavesim.sim import run_mission

ROOT = Path(__file__).resolve().parent.parent


def one(job):
    agent, map_name, odom, seed, n, lidar, aero = job
    t0 = time.time()
    s = run_mission(ROOT / "maps" / map_name, load_agent(agent), n_drones=n, odom=odom, seed=seed,
                    lidar=lidar, aero=aero)
    return job, s, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="agents.explorer")
    ap.add_argument("--maps", nargs="+", default=["small_2", "medium_1"])
    ap.add_argument("--odom", nargs="+", default=["default"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--lidar", default="os0-32", help="os0-32, lite (about 4x faster) or ideal")
    ap.add_argument("--aero", default="default", help="off, default or hard")
    args = ap.parse_args()

    jobs = [(args.agent, m, o, s, args.drones, args.lidar, args.aero)
            for m in args.maps for o in args.odom for s in args.seeds]
    print(f"{args.agent}: {len(jobs)} runs on {args.jobs} processes, lidar {args.lidar}, aero {args.aero}")
    print(f"{'map':<10}{'odom':<9}{'seed':>4}  {'lap s':>7}{'score':>8}{'found':>7}{'home':>6}"
          f"{'crash':>6}{'wrong':>6}{'drift':>7}{'wall s':>8}")
    laps = []
    with ProcessPoolExecutor(args.jobs) as ex:
        for (_, m, o, seed, n, _, _), s, wall in ex.map(one, jobs):
            done = s["artifacts_found"] == s["artifacts_total"] and len(s["returned"]) == n
            laps.append(s["time_s"] if done else None)
            print(f"{m:<10}{o:<9}{seed:>4}  {s['time_s'] if done else 'DNF':>7}{s['score']:>8.1f}"
                  f"{s['artifacts_found']:>4}/{s['artifacts_total']:<2}{len(s['returned']):>4}/{n}"
                  f"{len(s['crashed']) + len(s['battery_dead']):>6}{s['reports_used'] - s['artifacts_found']:>6}"
                  f"{max(s['final_drift_m']):>7.2f}{wall:>8.0f}", flush=True)
    ok = [x for x in laps if x is not None]
    print(f"finished {len(ok)}/{len(laps)}" + (f", mean lap {sum(ok) / len(ok):.1f} s" if ok else ""))


if __name__ == "__main__":
    main()
