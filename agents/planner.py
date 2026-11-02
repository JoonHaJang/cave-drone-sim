"""Demo agent vDEMO: build a voxel map from LiDAR, plan with A*, follow the path with a position loop.

    python -m cavesim.run --agent agents.planner --map rescue_easy --drones 1 --lidar lite

Students may read it for ideas, not copy it (docs/student_guide.md).

Mapping. Every sweep is first aligned to the map built so far (agents.lio: point-to-plane
ICP on top of the odometry, as LIO front ends do), so a place flown through again lines up
with what was mapped there before. Voxels a return lands in are occupied; voxels a beam
crossed on its way are free. Space within one voxel of an occupied one is too tight for
the airframe.

Planning. A* over free voxels. When the goal is not yet reachable through mapped space,
the plan ends at the frontier voxel (unknown, next to free space) that brings the drone
closest to it. Unknown space is never crossed: outside the cave's shell there is no rock
to see, so "unknown = free" would send the drone off into the void around the course.
The plan is redone every REPLAN_S as the map grows.

Control. A carrot CARROT_M ahead on the path; the velocity command is a proportional
term on the position error to it plus damping on the measured velocity (the simulator's
autopilot closes the velocity loop), slowed near rock, with a short-range push-off as a
last line of defence.

Mission. Rescue courses: entrance -> survivor (re-aimed at the detector's fix once the
survivor is in view) -> home and land. Explore courses: entrance, then the nearest
frontier each time, and home on battery.
"""
import bisect
import collections
import dataclasses
import heapq
import itertools
import math

import numpy as np

from cavesim.api import Agent, Command, body_rotation
from agents.lio import ScanMatcher

SCAN_MATCH = True       # align every sweep to the map (agents.lio) instead of taking the odometry as it is
RES = 0.5               # m, voxel edge
CARVE_RAYS = 6000       # carve free space along at most this many beams per sweep (all of them for 'lite')
CARVE_M = 15.0          # ... and only this far along each beam
# Log-odds occupancy (OctoMap-style): a return adds L_HIT, a beam passing through adds L_MISS.
# One stray return (noise, rotor dust) is cleared by the next beam through it; rock seen again
# and again saturates at L_MAX and stays.
L_HIT, L_MISS, L_MIN, L_MAX = 0.85, -0.4, -2.0, 3.5
HIT_R0 = 8.0            # m: farther returns count L_HIT * HIT_R0 / r. The attitude estimate is ~1 deg off, so
                        # a return r away lands ~r/60 off the rock: past HIT_R0 it may lie in the next voxel,
                        # and one such return must not make rock in open space (two are needed by 10 m)
OCC_T, FREE_T = 0.7, -0.2
REPLAN_S = 3.0          # s between routine replans (sooner when the path ends, is blocked, or the goal moves)
AHEAD_M = 8.0           # replan once less than this is left of a path that is not the final one: the next leg is
                        # ready before we reach the edge of the map, so we never stop there (receding horizon)
SURVEY_S = 3.0          # s between wide searches toward the goal that refresh the frontier memory
SWEEP_S = 9.0           # s between sweeps for frontiers all round (VoxelMap.sweep) ...
SWEEP_M = 20.0          # m ... this far over passable voxels
CARROT_M = 2.5          # m ahead on the path
LOOK_M = 6.0            # m ahead on the path the nose (and camera) points at
HOME_LEG_M = 12.0       # homing: plan locally to the node of the road home this far along
HOME_STUCK_S = 30.0     # ... but with no metre gained for this long, fly the road itself
KP = 1.2                # final approach: velocity command per metre of position error
YAW_RATE_MAX = 0.8      # rad/s: slower turns smear the LiDAR less and drift the heading less
TUBE_R = 0.5            # m: the tube we are about to fly through (drone radius 0.2 + margin); rock in it sets the speed
BRAKE_A = 3.5           # m/s^2 we count on to stop: 3/4 of g*tan(25 deg), the most the airframe has
BRAKE_LAG = 0.3         # s before braking bites: a tick to see it, ~0.2 s for the cascade to tilt back
BRAKE_MARGIN = 0.6      # m left between us and the rock once stopped
GUARD_M = 1.2           # m: returns this close limit how fast we may close in on them (guard)
SAFE_M = 0.45           # m: ... down to zero here (drone radius 0.2 + margin); closer, we are pushed off
GUARD_TUBE_R = 0.25     # m: ... counting only rock this near the line we fly along (drone radius 0.2 + 0.05;
                        # keeping off walls beside it is the push-off's job)
SLEW = 0.6              # m/s the velocity command may change per tick (6 m/s^2, above what the airframe can do)
DUST_GAP = 1.0          # m: lower than this over the floor, the rotors lift dust, and returns nearer than
DUST_R = 0.6            # this are mostly dust: they are left out of braking, push-off and the ceiling check
FOOTPRINT = [(0.0, 0.0), (0.25, 0.0), (-0.25, 0.0), (0.0, 0.25), (0.0, -0.25)]   # m, the rotor disc
CREEP_Z = 0.3           # m/s we may climb into space not mapped free (the simulator counts no hit below 0.3)
FEEL_S = 1.0            # s commanding up (or down) without moving that way: rock felt above, mapped
FEEL_DROP_S = 1.5       # ... and this long at full descent out of the ceiling effect's reach
CEIL_M = 1.0            # m: keep at least this far below the ceiling -- closer, the rotor wash sucks the drone
                        # up into it (ceiling effect, ~0.8 m reach) faster than the autopilot pulls back
NODE_BUDGET = 12_000    # A* expansions per plan (long legs go over the navigation graph)
WINDOW_M = 30.0         # plan in detail only this far around the drone (cf. TARE / GBPlanner local planners)
REACH_M = 1.5           # m from the *detected* survivor counts as there (the rules allow 3 m in sight) ...
FRESH_S = 1.0           # ... by a fix this recent: an old fix moves with every pose correction since (easy, 20 m
                        # error: turned home 4 m short)
LOST_S = 10.0           # at the fix and no sight of the survivor for this long: the fix was wrong, search again
STOP_M = 2.0            # plans to the survivor end this far off: it lies on the floor and is itself an obstacle
LAND_RD = 0.7           # m on the rangefinder: below this (and still) the autopilot's own landing takes over
RESCUE_FIRST = True     # the rescue counts, the way home comes second (Q3): give up only once even the
                        # lower bound on the rescue (straight line at full speed) no longer fits the
                        # flight time left -- t + LB <= T. False: GBPlanner's homing rule.
FINISH_M = 40.0         # this close to the survivor (or with it in view), the rescue comes before the battery rule
SEARCH_M = 2.5          # m from the given coordinates: start searching for the survivor
SEARCH_STALL_S = 60.0   # s without a metre gained on the given position: search from farther out
SEARCH_S = 90.0         # s of searching before giving up and heading home
SEARCH_DRIFT = 0.03     # search radius: our odometry may be off by this fraction of the distance flown
SEARCH_MIN_M = 5.0      # ... but at least this
SEARCH_LOOK = 0.6       # share of the detector's range a search point covers
CRUMB_M = 2.5           # m between nodes of the navigation graph (places flown through)
FRONTIER_CELL = 4       # voxels per side of the cells frontiers are remembered in (2 m)
GOAL_REACHED_M = 2.0    # at a remembered frontier: look around, then pick the next one
UNREACHED_TRIES = 5     # frontiers a replan may drop as unreachable ...
UNREACHED_M = 10.0      # ... unreachable: no plan reaches it though it is this close (the local window covers it)
GOAL_PATIENCE_S = 15.0  # no progress towards the committed frontier for this long: drop it
NEAR_FRONTIER_M = 4.0   # closer frontiers are mostly the LiDAR's blind cones above/below us and get
                        # mapped in passing; chasing them made the drone dither (picked only if nothing else)
GRAPH_LEG_M = 18.0      # farther than this along the graph: travel the graph first, plan locally at the end
ENCLOSED_M = 30.0       # course rule: beyond the staging area, rock above and below within this
BOX_MARGIN_M = 3.0      # keep this far inside the staging box: our position estimate drifts
COLUMN = 4              # voxels per side of the columns used for "rock above / below" (2 m: ceilings
                        # and floors get only sparse LiDAR hits per 0.5 m column)
ENTER_M = 3.0           # m from the given cave entrance counts as inside
AGL_MIN_M = 1.5         # never below this height above the ground under the drone (rangefinder), entering
FLOOR_MIN_M = 0.5       # ... and in the cave (low passages: 0.5 m keeps the airframe off the floor)
HOLE_NBRS = 13          # an unknown voxel with more free neighbours (of 26) is a gap between beams, not a frontier
DZ_W = 5.0              # height counts this much more in the distance a frontier leaves to the goal: caves are
                        # stacked levels, and a level above the goal is close only as the crow flies
                        # (hard 64: 54 vs 46 rescued at 3.0 -- the level over the survivor stopped drawing it)
# Doubling commitment (online search with predictions; Angelopoulos 2023, Banerjee et al. 2023):
# the frontier score is a prediction and may be wrong, so no region of the cave gets more than a
# budget of wasted time -- time not spent bringing us closer to the goal (the lower bound on what is
# left) at full speed; hovering and dithering count too -- before the others get their turn; every
# full round doubles it.
DOUBLING = False        # measured: the budgets cost hard its junction; kept for the experiment
BUDGET_S0 = 30.0        # s of waste a region gets in the first round (larger: trust the prediction longer)
BUDGET_R = 2.0          # budget growth per round
REGION_M = 16.0         # m: frontiers are grouped into regions of this size
CLEARANCE = None        # tight = next to an occupied voxel (None); within R_CLEAR of the seen rock
                        # surface ("dist"); or either ("union"); or next to an occupied
                        # voxel *and* within R_CLEAR of the surface measured round it ("measured")
R_SEEN = 1.0            # "measured": a margin voxel is let go only with measured surface this near it
GUARD_MAP = True        # the guard also brakes for rock surface cells the map holds (blind sides)
SURF_RES = 0.25         # m, rock surface cells (returns)
SURF_SCALE = 2          # surface cells per voxel edge (RES / SURF_RES)
SURF_N = 2              # sweeps a surface cell must be seen in (a single one is often dust or a ghost)
SURF_CLEAR_M = 3.0      # m: surface cells this near that a live beam passes through are dropped (stale)
R_CLEAR = 0.5           # m from a voxel centre (where paths run) to the nearest rock surface cell centre
ALT_MODE = "high"       # experiment: None, "high" (hug the ceiling, CEIL_M + 0.5 below it), "ceil" (the same, from the
                        # ceiling alone) or "mid" (halfway up)
ALT_W = 1.0             # ... step cost x (1 + ALT_W x the normalised height error)
FRONTIER_W = 1.0        # weight of the distance still to go: 1 = optimistic A* (unknown space assumed open); more is greedy and skips branches that first lead away
RECOVERY = False        # stuck (no certified path, or no headway): back off to a place flown through,
                        # look around, drop the frontier that failed, replan
STUCK_M = 1.0           # m: moving less than this ...
STUCK_S = 10.0          # ... for this long, in the rescue, is being stuck
RETREAT_M = 4.0         # m: back off to the nearest graph node at least this far
LOOK_S = 2 * math.pi / 0.8 + 1.0   # s of spinning at the yaw-rate limit: one full turn
ESCAPE_COST = 4.0       # cost factor for margin voxels next to rock, allowed only within 1.5 m of the start
RELAX = False           # no way on through certified voxels: plan again across margin voxels (not rock);
                        # measured: hard 9/16 vs 11/16 off (circling in margin before the survivor)
RF_TRACK_M = 1.5         # m: ... and only over a graph node this close, below (we came up that way)
RF_BEAM_R = 0.3          # m: map cells this close to the vertical below are checked against the rangefinder
GUARD_SEEN_UP_DEG = 30.0  # above this the bottom mount's view is the airframe (props at ~33 deg): map fills in
SAFE_SET = True         # outside the passable set (in rock or margin by the map): fly straight back into it
SAFE_SET_R = 4          # voxels: look this far for passable ones (2 m)
SAFE_SET_V = 0.5        # m/s
OPEN_WAY = True         # trapped: of the directions open OPEN_MIN on the live scan *and* the mapped surface,
                        # leave by the one nearest the map's way back (the map, built on a drifted pose, put
                        # its way back into real rock -- the live returns blocked it, or rock inside the
                        # LiDAR's 0.3 m / above its cone was pushed against: 6 hard runs sat 600+ s)
OPEN_MIN = 1.0          # m of free tube that counts as open
OPEN_R = 0.35           # m: radius of the tube a direction must keep clear (the airframe is 0.2)
OPEN_SEEN_DEG = 15.0    # a direction with no return this close to it is the LiDAR's blind side: not open

OFF = 1 << 20
NEIGHBOURS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if i or j or k])
STEP_COST = np.linalg.norm(NEIGHBOURS, axis=1)
# The packed key is linear in the indices, so a neighbour is one integer addition away.
FACE_DELTAS = [(int(i) << 42) + (int(j) << 21) + int(k) for i, j, k in NEIGHBOURS if abs(i) + abs(j) + abs(k) == 1]
DELTAS = [((int(i) << 42) + (int(j) << 21) + int(k), float(c)) for (i, j, k), c in zip(NEIGHBOURS, STEP_COST)]
# Margin round rock: the neighbours (by L1 offset: 1 face, 2 edge, 3 corner) an occupied voxel blocks.
# From a voxel centre, rock in a face neighbour can be 0.25 m off, in an edge one 0.35 m, a corner 0.43 m.
MARGIN_L1 = 3
def block_deltas(l1):
    return [0] + [d for (d, _), o in zip(DELTAS, NEIGHBOURS) if np.abs(o).sum() <= l1]
BLOCK_DELTAS = block_deltas(MARGIN_L1)
SOFT_MARGIN = True      # the corner neighbours (rock 0.35-0.43 m off) may be crossed at SOFT_COST: wide
SOFT_COST = 4.0         # ways first, the narrow one when it is the only one (the last 35 m to the hard
                        # survivor are margin by the full 3x3x3 rule but flyable)
HARD_DELTAS = block_deltas(2)
SOFT_NEAR_M = 120.0     # ... only this close to the survivor's coordinates (plus their stated error): everywhere,
                        # it opened narrow wrong ways at the 340 m junction, 180 m off (hard 32: 17 vs 20
                        # rescued); at 60 m the neck 70 m short of it stayed margin and sent the search up
                        # a chimney the drone wedged in
SOFT_LOCAL_M = 0.0      # ... and this close to the drone, where the live LiDAR is looking at the neck itself
                        # (the 340 m junction's way on is a 7 m neck). At 6 m: hard 60 vs 59 /64 without an
                        # error, 51 vs 58 with 20 m -- it opens wrong narrow ways too; off
SOFT_FAR_COST = 0.0     # farther off: this factor (0 = not at all). The 340 m junction's way on is such a
                        # neck, 180 m off: blocked, the search had only the upper level left
ESCALATE_S = 180.0       # s in the rescue without a metre gained on the height-weighted distance left: the way on
                        # was classed shut somewhere behind us. Open the soft margin round the place we were
                        # nearest (and go back there). Rescued hard runs stall <= 120 s, the 340 m junction's
                        # failures >= 480 s (64 seeds). Off: easy 20 m error
                        # 25 vs 31 /32 rescued; a global version 58 vs 59 /64 hard
ESCALATE_R = 30.0       # m round that place
CELL_OFFSETS = np.array(np.meshgrid(*[np.arange(FRONTIER_CELL)] * 3, indexing="ij")).reshape(3, -1).T
# Clearance: a voxel centre is the corner of 8 surface cells; the cells whose centres lie within
# R_CLEAR of it, as key deltas from the cell above that corner. Voxels to recheck around a new
# surface cell: all whose centres may lie within R_CLEAR of it.
_g = np.arange(-4, 4)
_o = np.array(np.meshgrid(_g, _g, _g, indexing="ij")).reshape(3, -1).T
CLEAR_DELTAS = [int((int(i) << 42) + (int(j) << 21) + int(k)) for i, j, k in _o
                if np.linalg.norm((np.array([i, j, k]) + 0.5) * SURF_RES) < R_CLEAR]
CLEAR_DIRTY = [int((int(i) << 42) + (int(j) << 21) + int(k))       # as key deltas, like CLEAR_DELTAS
               for i, j, k in np.array(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij")).reshape(3, -1).T]


def keys_of(idx):
    """Pack integer voxel indices (N, 3) into int64 keys."""
    idx = np.asarray(idx, dtype=np.int64) + OFF
    return (idx[..., 0] << 42) | (idx[..., 1] << 21) | idx[..., 2]


def idx_of(key):
    return np.array([(key >> 42) - OFF, ((key >> 21) & 0x1FFFFF) - OFF, (key & 0x1FFFFF) - OFF])


class Tight:
    """`key in map.blocked` for the clearance map: rock, or too near the rock surface."""
    def __init__(self, m):
        self.m = m

    def __contains__(self, key):
        m = self.m
        if CLEARANCE == "measured":
            # The 3x3x3 rule blocks a voxel for rock anywhere in the next voxel, up to 0.75 m off: it
            # closed 1-1.5 m necks the drone flies (the 340 m junction's way on). Let a margin voxel
            # go where the measured surface is R_CLEAR off -- only beside rock actually measured, so a
            # gap in the sparse returns never certifies anything (that sank "dist").
            return key in m.occ or (key in m.block_n and not m.clear(key))
        return key in m.occ or (CLEARANCE == "union" and key in m.block_n) or not m.clear(key)


class VoxelMap:
    def __init__(self, staging_box=None):
        self.logodds = {}       # voxel -> log-odds of being rock
        self.occ = set()        # log-odds above OCC_T
        self.block_n = {}       # voxel -> number of occupied voxels in its 3x3x3 block
        self.block_h = {}       # ... in its face and edge neighbours (the hard margin)
        self.soft_at = None     # voxel index the soft margin is allowed round, within soft_r m
        self.soft_r = SOFT_NEAR_M
        self.soft_extra = []    # voxel indices the soft margin is also allowed round, within ESCALATE_R m
        # Too tight to fly: within R_CLEAR of the rock surface we have seen (CLEARANCE), or else
        # next to an occupied voxel. Read with `in`.
        self.blocked = Tight(self) if CLEARANCE else self.block_n
        self.surface = {}       # SURF_RES cell -> sweeps that put a return in it
        self.clear_cache = {}   # voxel -> its centre is R_CLEAR from every surface cell
        self.free = set()       # log-odds below FREE_T: a beam passed through, more than hit it
        self.columns = {}       # (i, j) column key -> sorted k of occupied voxels, for the course rule
        self.box = None if staging_box is None else [self.voxel(np.add(staging_box[0], BOX_MARGIN_M)),
                                                      self.voxel(np.subtract(staging_box[1], BOX_MARGIN_M))]
        self.fence = None if staging_box is None else [self.voxel(staging_box[0]), self.voxel(staging_box[1])]
        self.allowed_cache = {}
        self.cave_only = False  # once inside: stay under rock (no going back out over the hills)
        self.entrance = None    # ... and out of the staging box, but for the way in

    def allowed(self, key):
        """Course rule as seen in our own map: ground mapped below within ENCLOSED_M, and either
        inside the staging box or rock mapped above too. Keeps us inside the tunnels and above
        the staging terrain (both hang in empty space in the simulator)."""
        if self.box is None:
            return True
        hit = self.allowed_cache.get(key)
        if hit is not None:
            return hit
        i, j, k = (key >> 42) - OFF, ((key >> 21) & 0x1FFFFF) - OFF, (key & 0x1FFFFF) - OFF
        lo, hi = self.box
        inside = lo[0] <= i <= hi[0] and lo[1] <= j <= hi[1] and lo[2] <= k <= hi[2]
        flo, fhi = self.fence
        in_box = flo[0] <= i <= fhi[0] and flo[1] <= j <= fhi[1] and flo[2] <= k <= fhi[2]
        if self.cave_only and in_box and np.linalg.norm(self.centre(key) - self.entrance) > ENTER_M + 2.0:
            return False          # inside, the staging area is outside: cliffs beside it pass for cave walls
        if in_box and (i < lo[0] or not lo[1] <= j <= hi[1] or k > hi[2]):
            return False          # along the staging fences (all sides but the cave's, +x, and the ground): the LiDAR does not see them
        # Rock in the column or the eight around it (6 m): a frontier is at the edge of what we have
        # seen, and close up the rock right above it is in the LiDAR's blind cone -- the ceiling a
        # metre beside it is what we have mapped. One column alone called passage mouths "off course".
        kk, reach = key & 0x1FFFFF, ENCLOSED_M / RES
        below = above = False
        ci, cj = i // COLUMN, j // COLUMN
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                col = self.columns.get(((ci + di) << 21) | (cj + dj & 0x1FFFFF))
                if col:
                    n = bisect.bisect_left(col, kk)
                    below = below or (n > 0 and kk - col[n - 1] <= reach)
                    above = above or (n < len(col) and col[n] - kk <= reach)
        ok = below and above if self.cave_only else below and (inside or above)
        if ok:                    # "not allowed" can change as rock gets mapped; "allowed" cannot
            self.allowed_cache[key] = True
        return ok

    def voxel(self, p):
        return np.floor(np.asarray(p) / RES).astype(np.int64)

    def centre(self, key):
        return (idx_of(key) + 0.5) * RES

    def insert(self, origin, pts):
        """origin: sensor position (3,); pts: returns in the same (odom) frame (N, 3)."""
        if not len(pts):
            return
        # Misses first: sample beams at voxel steps up to a voxel short of the return (capped).
        rays = pts[::max(1, -(-len(pts) // CARVE_RAYS))] - origin
        dist = np.linalg.norm(rays, axis=1)
        reach = np.minimum(dist - RES, CARVE_M)
        steps = np.arange(0.0, max(reach.max(), 0.0), RES)
        ok = steps[None, :] < reach[:, None]
        samples = origin + rays[:, None, :] * (steps[None, :] / np.maximum(dist[:, None], 1e-6))[..., None]
        self.update(np.unique(keys_of(self.voxel(samples[ok]))).tolist(), L_MISS)
        w = np.minimum(1.0, HIT_R0 / np.maximum(np.linalg.norm(pts - origin, axis=1), 1e-6))
        order = np.argsort(w)                              # dict keeps the last, i.e. the surest return per voxel
        hits = dict(zip(keys_of(self.voxel(pts[order])).tolist(), (L_HIT * w[order]).tolist()))
        self.update(list(hits), list(hits.values()))
        if CLEARANCE or GUARD_MAP:
            self.add_surface(pts)
            if SURF_CLEAR_M:
                self.clear_surface(origin, pts)

    # ---------------------------------------------------------------- clearance (certified tube)
    def add_surface(self, pts):
        """Count the sweeps that put a return in each SURF_RES cell; a cell seen in SURF_N sweeps
        is rock surface, and the clearance of the voxels around it is recomputed when asked."""
        surf, cache = self.surface, self.clear_cache
        for c in np.unique(keys_of(np.floor(pts / SURF_RES).astype(np.int64))).tolist():
            n = surf.get(c, 0) + 1
            surf[c] = n
            if n == SURF_N:
                base = np.floor(idx_of(c) * SURF_RES / RES).astype(np.int64)     # the voxel it lies in
                kb = int(keys_of(base))
                for d in CLEAR_DIRTY:
                    cache.pop(kb + d, None)

    def clear_surface(self, origin, pts):
        """A surface cell within SURF_CLEAR_M that a live beam passes through -- within half a cell of
        its centre, on to a return beyond it -- is not rock: a beam went on. Cells were never cleared,
        and those laid on the way out under another pose error stood 0.2-0.3 m off the drone on the
        way back, in the free space the beams went through: the guard's push-offs cancelled the
        command at a neck the drone had flown, 90 s (hard). Counts run to hundreds, so
        it drops to one short of rock: real rock is hit again the next sweep, and is rock again."""
        q = self.surface_near(origin, SURF_CLEAR_M)
        if not len(q) or not len(pts):
            return
        rays = pts - origin
        dl = np.linalg.norm(rays, axis=1)
        u = rays / np.maximum(dl, 1e-6)[:, None]
        w = q - origin
        along = u @ w.T                                    # (beams, cells)
        perp2 = np.einsum("ij,ij->i", w, w)[None, :] - along ** 2
        through = ((along > 0) & (perp2 < (SURF_RES / 2) ** 2) & (dl[:, None] > along + SURF_RES / 2)).any(0)
        surf, cache = self.surface, self.clear_cache
        for c in keys_of(np.floor(q[through] / SURF_RES).astype(np.int64)).tolist():
            surf[c] = SURF_N - 1
            base = np.floor(idx_of(c) * SURF_RES / RES).astype(np.int64)   # the clearance round it changes
            kb = int(keys_of(base))
            for d in CLEAR_DIRTY:
                cache.pop(kb + d, None)

    def clear(self, key):
        """Is the voxel centre at least R_CLEAR from every rock surface cell?"""
        hit = self.clear_cache.get(key)
        if hit is None:
            surf = self.surface
            base = int(keys_of(idx_of(key) * SURF_SCALE + SURF_SCALE // 2))  # the surface cell above the centre
            hit = not any(surf.get(base + d, 0) >= SURF_N for d in CLEAR_DELTAS)
            if hit and CLEARANCE == "measured":
                hit = len(self.surface_near(self.centre(key), R_SEEN)) > 0
            self.clear_cache[key] = hit
        return hit

    def surface_near(self, p, r):
        """Centres of the rock surface cells within r of p."""
        c = np.floor(np.asarray(p) / SURF_RES).astype(np.int64)
        n = int(np.ceil(r / SURF_RES))
        g = np.arange(-n, n + 1)
        o = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
        ks = keys_of(c + o).tolist()
        surf = self.surface
        sel = [i for i, k in enumerate(ks) if surf.get(k, 0) >= SURF_N]
        q = (c + o[sel] + 0.5) * SURF_RES
        return q[np.linalg.norm(q - p, axis=1) < r]

    def update(self, keys, delta):
        """Add delta (one value, or one per key) to the log-odds of keys."""
        lo, occ, free = self.logodds, self.occ, self.free
        for k, dl in zip(keys, delta if isinstance(delta, list) else itertools.repeat(delta)):
            old = lo.get(k, 0.0)
            new = min(max(old + dl, L_MIN), L_MAX)
            lo[k] = new
            if (old > OCC_T) != (new > OCC_T):
                self.set_occupied(k, new > OCC_T)
            if new < FREE_T:
                free.add(k)
            else:
                free.discard(k)

    def set_occupied(self, k, on):
        col = ((((k >> 42) - OFF) // COLUMN) << 21) | ((((k >> 21) & 0x1FFFFF) - OFF) // COLUMN & 0x1FFFFF)
        if on:
            self.occ.add(k)
            for d in BLOCK_DELTAS:
                self.block_n[k + d] = self.block_n.get(k + d, 0) + 1
            for d in HARD_DELTAS:
                self.block_h[k + d] = self.block_h.get(k + d, 0) + 1
            bisect.insort(self.columns.setdefault(col, []), k & 0x1FFFFF)
        else:
            self.occ.discard(k)
            for d in BLOCK_DELTAS:
                n = self.block_n[k + d] - 1
                if n:
                    self.block_n[k + d] = n
                else:
                    del self.block_n[k + d]
            for d in HARD_DELTAS:
                n = self.block_h[k + d] - 1
                if n:
                    self.block_h[k + d] = n
                else:
                    del self.block_h[k + d]
            c = self.columns[col]
            del c[bisect.bisect_left(c, k & 0x1FFFFF)]

    def is_hole(self, key):
        free = self.free
        return sum((key + d) in free for d, _ in DELTAS) > HOLE_NBRS

    def passable(self, key):
        return key in self.free and key not in self.blocked and self.allowed(key)

    def plan(self, start, goal, stop_m=0.9, relax=False):
        """Path (list of points) from start toward goal through free voxels; ends at the goal or
        at the frontier voxel that gets closest to it. With relax, margin voxels (never rock) may be
        crossed anywhere, at ESCAPE_COST: the second tier, when the certified ones lead nowhere."""
        s = int(keys_of(self.voxel(start)))
        gi, gj, gk = (int(x) + OFF for x in self.voxel(goal))
        si, sj, sk = (int(x) + OFF for x in self.voxel(start))
        win2 = (WINDOW_M / RES) ** 2
        free, blocked, allowed = self.free, self.blocked, self.allowed
        soft = SOFT_MARGIN and CLEARANCE in (None, "measured") and self.soft_at is not None
        if soft:
            ci, cj, ck = (int(x) + OFF for x in self.soft_at)
            soft2 = (self.soft_r / RES) ** 2
            local2 = (SOFT_LOCAL_M / RES) ** 2
            extra = [tuple(int(x) + OFF for x in c) for c in self.soft_extra]
            esc2 = (ESCALATE_R / RES) ** 2

        def soft_cost(m):                                  # a corner-margin voxel: its cost factor, 0 = not allowed
            if not soft or m in self.block_h:
                return 0.0
            near = ((m >> 42) - ci) ** 2 + (((m >> 21) & 0x1FFFFF) - cj) ** 2 + ((m & 0x1FFFFF) - ck) ** 2 <= soft2
            here = ((m >> 42) - si) ** 2 + (((m >> 21) & 0x1FFFFF) - sj) ** 2 + ((m & 0x1FFFFF) - sk) ** 2 <= local2
            if near or here:
                return SOFT_COST
            i, j, k = m >> 42, (m >> 21) & 0x1FFFFF, m & 0x1FFFFF
            if any((i - a) ** 2 + (j - b) ** 2 + (k - c) ** 2 <= esc2 for a, b, c in extra):
                return SOFT_COST
            return SOFT_FAR_COST

        def h(k):                                          # voxels to the goal
            return math.sqrt(((k >> 42) - gi) ** 2 + (((k >> 21) & 0x1FFFFF) - gj) ** 2 + ((k & 0x1FFFFF) - gk) ** 2)

        openq = [(h(s), 0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        self.frontiers_seen = []                           # (voxel key, travel cost) of every frontier met
        # Best frontier by travel + FRONTIER_W * distance left: a greedy "closest to the goal" pick
        # gets stuck in dead ends where every way out first leads away from the goal.
        best = (math.inf, s)
        expanded = 0
        alt = {}
        while openq and expanded < NODE_BUDGET:
            _, gc, k = heapq.heappop(openq)
            if gc > cost[k]:
                continue
            expanded += 1
            if h(k) <= stop_m / RES:
                best = (-1.0, k)
                break
            for d, c in DELTAS:
                m = k + d
                if m in blocked and soft_cost(m):
                    c *= soft_cost(m)
                elif m in blocked:
                    # Squeezed against rock, the drone's own neighbourhood is all margin: let the plan
                    # cross margin voxels (never rock) near the start, at a premium, to get out.
                    near = ((m >> 42) - si) ** 2 + (((m >> 21) & 0x1FFFFF) - sj) ** 2 + ((m & 0x1FFFFF) - sk) ** 2 <= 9
                    if m in self.occ or not (near or relax):
                        continue
                    c *= ESCAPE_COST
                if ((m >> 42) - si) ** 2 + (((m >> 21) & 0x1FFFFF) - sj) ** 2 + ((m & 0x1FFFFF) - sk) ** 2 > win2:
                    # Edge of the local window: k is as far as this plan goes in that direction.
                    f = gc + FRONTIER_W * h(k)
                    if f < best[0]:
                        best = (f, k)
                    continue
                if ALT_MODE:
                    c *= 1.0 + ALT_W * self.alt_cost(m, alt)
                nc = gc + c
                if m in free and not allowed(m):
                    continue                    # course rule on mapped space (unknown space has no rock yet)
                if m not in free:
                    # Frontier: unknown voxel on the edge of unmapped space. A goal for this plan, never crossed.
                    # Only off a passable voxel: seen from the escape margin it is the blind cone over a
                    # drone squeezed under the ceiling, and creeping into it pinned the drone there.
                    if m not in came and not self.is_hole(m) and (k not in blocked or soft_cost(k)):
                        came[m], cost[m] = k, nc
                        self.frontiers_seen.append((m, nc))
                        f = nc + FRONTIER_W * h(m)
                        if f < best[0]:
                            best = (f, m)
                    continue
                if nc < cost.get(m, math.inf):
                    cost[m], came[m] = nc, k
                    heapq.heappush(openq, (nc + 1.5 * h(m), nc, m))   # weighted A*: fewer expansions
        self.reached_goal = best[0] < 0
        if RELAX and not relax and best[1] == s:
            # Nothing but margin round the certified voxels we can reach: a passage narrower than the
            # 3x3x3 margin asks for (21-31 % of flyable voxels near the track are margin).
            return self.plan(start, goal, stop_m, relax=True)
        path, k = [], best[1]
        while k is not None:
            path.append(self.centre(k))
            k = came[k]
        return path[::-1]

    def alt_cost(self, key, cache, reach=20):
        """Height error in [0, 1] of voxel key against ALT_MODE, from the rock mapped straight above
        and below it (within reach voxels); 0 where either is unmapped."""
        hit = cache.get(key)
        if hit is None:
            occ = self.occ
            up = next((n for n in range(1, reach) if key + n in occ), None)
            dn = next((n for n in range(1, reach) if key - n in occ), None)
            if ALT_MODE == "ceil":           # the ceiling alone: a top-mounted LiDAR barely maps the floor
                hit = 0.0 if up is None else min(1.0, max(0.0, up * RES - CEIL_M - 0.5) / 5.0)
            elif up is None or dn is None:
                hit = 0.0
            elif ALT_MODE == "high":
                hit = max(0.0, up * RES - CEIL_M - 0.5) / ((up + dn) * RES)
            else:
                hit = abs(up - dn) / (up + dn)
            cache[key] = hit
        return hit

    def way_back(self, p):
        """Unit vector from p toward the nearest passable voxels (the mean over the nearest shell),
        or None if p's voxel is passable or none lies within SAFE_SET_R."""
        c = self.voxel(p)
        if self.passable(int(keys_of(c))):
            return None
        for r in range(1, SAFE_SET_R + 1):
            g = np.arange(-r, r + 1)
            o = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
            o = o[np.abs(o).max(1) == r]
            ok = [i for i, k in enumerate(keys_of(c + o).tolist()) if self.passable(int(k))]
            if ok:
                d = np.mean([self.centre(int(keys_of(c + o[i]))) for i in ok], axis=0) - p
                n = np.linalg.norm(d)
                return d / n if n > 1e-6 else None
        return None

    def clear_line(self, a, b):
        n = int(np.linalg.norm(b - a) / (RES / 2)) + 1
        ks = keys_of(self.voxel(a + (b - a) * np.linspace(0, 1, n)[:, None])).tolist()
        return all(self.passable(k) for k in ks[1:])

    def sweep(self, start, r):
        """Every frontier within r m of start over passable voxels, whichever way the goal lies:
        [(voxel key, travel m)]. plan() is pulled toward its goal and never expands the other way, so
        a way on that first leads away was never remembered: the J1 chamber's exit lay east of the
        drone, the survivor west -- 1031 frontier voxels mapped there, none in memory (hard)."""
        s = int(keys_of(self.voxel(start)))
        si, sj, sk = (int(x) + OFF for x in self.voxel(start))
        r2 = (r / RES) ** 2
        dist, q, out = {s: 0.0}, collections.deque([s]), []
        while q:
            k = q.popleft()
            for d in FACE_DELTAS:
                m = k + d
                if m in dist:
                    continue
                dist[m] = dist[k] + RES
                if m in self.free:
                    if self.passable(m) and ((m >> 42) - si) ** 2 + (((m >> 21) & 0x1FFFFF) - sj) ** 2 + ((m & 0x1FFFFF) - sk) ** 2 <= r2:
                        q.append(m)
                elif m not in self.occ and m not in self.blocked and not self.is_hole(m):
                    out.append((m, dist[m]))
        return out


def _sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    a = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(a), r * np.sin(a), z]


OPEN_DIRS = _sphere(256)


def tube_free(rel):
    """Free length (m) of the OPEN_R tube along each of OPEN_DIRS, -1 where no point says; None without points."""
    d = np.linalg.norm(rel, axis=1)
    rel, d = rel[d > 1e-6], d[d > 1e-6]
    if not len(rel):
        return None
    t = OPEN_DIRS @ rel.T                                    # (dirs, pts) distance along each direction
    lat2 = d[None] ** 2 - t ** 2
    hit = (t > 0) & (lat2 < OPEN_R ** 2)
    free = np.where(hit, t, np.inf).min(1)
    seen = (t / d[None] > np.cos(np.radians(OPEN_SEEN_DEG))).any(1)
    return np.where(seen, np.minimum(free, d.max()), -1.0)


def open_way(rel, prefer=None):
    """Unit direction (world frame) for a tube of OPEN_R, among the directions the points rel
    (rock relative to us, world frame) cover: of those open OPEN_MIN, the nearest to prefer; else
    (or without prefer) the one open farthest. None without points."""
    free = tube_free(rel)
    if free is None:
        return None
    if prefer is not None and (free >= OPEN_MIN).any():
        return OPEN_DIRS[int(np.argmax(np.where(free >= OPEN_MIN, OPEN_DIRS @ prefer, -np.inf)))]
    i = int(np.argmax(free))
    return OPEN_DIRS[i] if free[i] > 0 else None


def stop_speed(gap):
    """Fastest speed from which we still stop within gap (m), braking BRAKE_A after BRAKE_LAG:
    v*lag + v^2 / 2a = gap."""
    gap = np.maximum(gap, 0.0)
    return -BRAKE_A * BRAKE_LAG + np.sqrt((BRAKE_A * BRAKE_LAG) ** 2 + 2 * BRAKE_A * gap)


def deskew(obs):
    """LiDAR returns in the body frame at the end of the sweep.

    The sensor fires column by column over the sweep (obs.lidar_t), each from the pose at that
    instant. Undo the motion since then with the mean gyro rate and the odometry velocity, as LIO
    front ends do: p_end = Exp(-w dt) p - R^T v dt."""
    hit = np.isfinite(obs.lidar_ranges)
    p = obs.lidar_dirs[hit] * obs.lidar_ranges[hit, None] + np.asarray(obs.lidar_mount)
    if obs.lidar_t is None or not len(obs.imu):
        return p
    dt = (obs.t - obs.lidar_t[hit])[:, None]
    w = obs.imu[:, 4:7].mean(0)
    theta = np.linalg.norm(w)
    if theta > 1e-6:                                   # Rodrigues, angle -theta*dt about w/|w|
        k = w / theta
        a = -theta * dt
        p = p * np.cos(a) + np.cross(k, p) * np.sin(a) + k * (p @ k)[:, None] * (1 - np.cos(a))
    if obs.vel is not None:
        R = body_rotation(obs.roll, obs.pitch, obs.yaw)
        p = p - (R.T @ obs.vel)[None, :] * dt
    return p


def shortcut(path, vmap):
    """Drop waypoints that can be skipped in a straight, collision-free line."""
    if len(path) < 3:
        return path
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not vmap.clear_line(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


class PlannerAgent(Agent):
    def __init__(self, drone_id, n_drones, info):
        super().__init__(drone_id, n_drones, info)
        self.map = VoxelMap(info.get("staging_box"))
        self.lio = ScanMatcher()
        self.last_cmd = np.zeros(3)       # the velocity command of the last tick (SLEW)
        self.held = None                  # since when a climb / descent commanded is not happening (feel_ceiling)
        self.drop_until = -np.inf         # full descent till then: out of a felt ceiling's pull
        self.home = np.array(info["base"], dtype=float)
        self.home_odom = self.home.copy()   # where home is in our drifted odom frame
        self.gps_off = None
        self.entrance = None if info.get("entrance") is None else np.array(info["entrance"], dtype=float)
        self.target = None if info.get("target") is None else np.array(info["target"], dtype=float)
        if self.target is not None:
            self.map.soft_at = self.map.voxel(self.target)
            self.map.soft_r = SOFT_NEAR_M + info.get("target_error_m", 0.0)
        # Courses entered from an outdoor staging area: reach the cave entrance first (teams know
        # where the course entrance is), then go for the survivor.
        self.route = [np.array(p, dtype=float) for p in info.get("entrance_route") or []]
        self.mode = "enter" if self.route else ("rescue" if self.target is not None else "home")
        self.path, self.next_plan, self.planned_at = [], 0.0, -math.inf
        self.path_final = False           # does the path end where we actually want to stop?
        self.next_survey = 0.0
        self.goal = None
        self.prev_vel = np.zeros(3)
        self.seen = False                 # survivor confirmed by the detector
        self.seen_t = -math.inf           # ... last at this time
        self.closest = None               # (t, distance) of the last metre gained on the given position
        self.escape = None                # unit direction we are leaving a trap by
        self.open_at = None               # last place with a way out: certified, or an open tube
        self.search_goal = None           # [frontier cell or None, place, t of last gain, best distance]
        self.search_path_to = None        # the search goal self.path was last planned all the way to
        self.next_sweep = 0.0
        self.given_up = set()             # frontier cells the search set out for and stopped gaining on
        self.search_until = None
        self.searched = []                # where the search has taken us
        self.flown, self.prev_pos = 0.0, None
        self.next_battery_check = 0.0
        self.battery0 = self.t0 = None    # battery and time at the first check (drain rate as flown)
        # Navigation graph: places flown through (guaranteed-passable roads, cf. GBPlanner's global
        # graph, CTU's tree of visited positions) plus the frontiers still to visit, remembered by
        # 2 m cell with the graph node they were seen from. Long-term planning picks a frontier
        # on this graph and commits to it; short-term planning (voxel A*) covers the last stretch.
        self.nodes, self.edges, self.last = [], {}, None
        self.frontiers = {}               # cell -> {"pos": point, "node": graph node}
        self.committed = None             # cell of the frontier we are heading for
        self.commit_best, self.commit_since = math.inf, 0.0
        self.waste = {}                   # region -> time wasted while committed to it (s)
        self.round = 0                    # doubling round: budgets are BUDGET_M0 * BUDGET_R ** round
        self.recover = None               # recovery: {"phase": "back" | "look", "until": t}
        self.waste_from = None            # (t, lower bound) at the last tick, for the waste count
        self.stuck_ref = None             # (t, position) of the last headway
        self.best_lb = None               # (t, height-weighted distance left, place) of the last metre gained
        self.revisit = None               # escalated: go back to this place first
        self.home_ref = None              # (t, road length left) of the last metre gained on the way home

    # ---------------------------------------------------------------- graph home
    def add_crumb(self, pos):
        if self.last is not None and np.linalg.norm(pos - self.nodes[self.last]) < CRUMB_M:
            return
        near = [j for j, q in enumerate(self.nodes) if j != self.last and np.linalg.norm(pos - q) < 0.75 * CRUMB_M]
        if near:                          # back at a known place: join the loop
            j = near[0]
        else:
            j = len(self.nodes)
            self.nodes.append(pos.copy())
            self.edges[j] = set()
        if self.last is not None and j != self.last:
            self.edges[self.last].add(j)
            self.edges[j].add(self.last)
        self.last = j

    def graph_from_here(self):
        """Dijkstra over the navigation graph from the current node: (distances, predecessors)."""
        dist, prev, q = {self.last: 0.0}, {self.last: None}, [(0.0, self.last)]
        while q:
            d, u = heapq.heappop(q)
            if d > dist[u]:
                continue
            for v in self.edges[u]:
                nd = d + float(np.linalg.norm(self.nodes[u] - self.nodes[v]))
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(q, (nd, v))
        return dist, prev

    def graph_path(self, prev, node):
        out = []
        while node is not None:
            out.append(self.nodes[node])
            node = prev[node]
        return out[::-1]

    def home_leg_end(self, road):
        """Index of the road node the local planner takes us to: HOME_LEG_M along, but no further
        than the last node inside the course rule. Past the cave mouth the road runs over the staging
        area, which no plan may enter: the plan to a node there failed, and its partial end lay 15 m
        back in the cave -- in and out of the mouth for 35 s, the GNSS fix coming and going (easy 6). 0: none, fly the road itself."""
        along, j = 0.0, 0
        while j < len(road) - 1 and along < HOME_LEG_M and self.map.allowed(int(keys_of(self.map.voxel(road[j + 1])))):
            along += np.linalg.norm(road[j + 1] - road[j])
            j += 1
        return j

    def road_home(self):
        """Shortest way back to the first node over the graph of places flown through."""
        if self.last is None:
            return []
        _, prev = self.graph_from_here()
        return self.graph_path(prev, 0 if 0 in prev else self.last) + [self.home_odom]

    # ---------------------------------------------------------------- frontier memory
    def remember_frontiers(self, pos):
        """Fold the frontiers the last local search met into the memory, by 2 m cell, and forget
        remembered ones whose voxel has since been mapped (explored, or closed off). Not meeting
        one is no reason: the search is pulled toward the goal and never looks the other way."""
        seen = {}
        for k, g in getattr(self.map, "frontiers_seen", []):
            c = self.map.centre(k)
            cell = tuple((np.floor(c / (RES * FRONTIER_CELL))).astype(int))
            if cell not in seen or g < seen[cell][1]:
                seen[cell] = (c, g, k)
        m = self.map
        for cell, f in list(self.frontiers.items()):
            # Gone: mapped since, a gap between beams, against rock, or (seen from close by) off the course.
            k = f["key"]
            off = np.linalg.norm(f["pos"] - pos) < 10.0 and not m.allowed(k)
            if k in m.free or k in m.occ or k in m.blocked or m.is_hole(k) or off:
                # One voxel of the cell is gone, not necessarily the cell's frontier: move to another
                # frontier voxel of the same cell (else a way on that the goal-pulled searches never
                # look at again is forgotten for good).
                k = self.frontier_in_cell(cell, pos)
                if k is None:
                    del self.frontiers[cell]
                else:
                    f["key"], f["pos"] = k, m.centre(k)
        for cell, (c, g, k) in seen.items():
            if cell not in self.frontiers:
                self.frontiers[cell] = {"pos": c, "node": self.last, "key": k}

    def frontier_in_cell(self, cell, pos):
        m = self.map
        base = np.array(cell) * FRONTIER_CELL
        for o in CELL_OFFSETS:
            k = int(keys_of(base + o))
            if k in m.free or k in m.occ or k in m.blocked or m.is_hole(k):
                continue
            if np.linalg.norm(m.centre(k) - pos) < 10.0 and not m.allowed(k):
                continue
            if any((k + d) in m.free and (k + d) not in m.blocked for d, _ in DELTAS):
                return k
        return None

    def region(self, p):
        return tuple(np.floor(np.asarray(p) / REGION_M).astype(int))

    def exhausted(self, region):
        return DOUBLING and self.waste.get(region, 0.0) >= BUDGET_S0 * BUDGET_R ** self.round

    def count_waste(self, obs):
        """Charge the committed region with the time that did not lower the distance left."""
        lb = float(np.linalg.norm(obs.pos - self.target))
        if self.waste_from is not None and self.committed in self.frontiers:
            t0, lb0 = self.waste_from
            r = self.region(self.frontiers[self.committed]["pos"])
            gain = max(0.0, lb0 - lb) / self.info["max_speed_xy"]
            self.waste[r] = self.waste.get(r, 0.0) + max(0.0, (obs.t - t0) - gain)
        self.waste_from = (obs.t, lb)

    def pick_frontier(self, obs, goal):
        """Long-term choice: the remembered frontier minimising graph travel + FRONTIER_W x the
        distance it leaves to the goal (the prediction), among regions whose waste budget is not
        spent. Kept until reached, gone, stalled, or its region's budget runs out."""
        dist, prev = self.graph_from_here()
        if self.committed in self.frontiers and self.exhausted(self.region(self.frontiers[self.committed]["pos"])):
            self.committed = None                             # budget spent: the other regions' turn
        if self.committed in self.frontiers:
            f = self.frontiers[self.committed]
            d = float(np.linalg.norm(f["pos"] - obs.pos))
            if d < self.commit_best - 0.5:
                self.commit_best, self.commit_since = d, obs.t
            if d < GOAL_REACHED_M or obs.t - self.commit_since > GOAL_PATIENCE_S:
                del self.frontiers[self.committed]            # visited, or not getting closer
                self.committed = None
            else:
                return f, dist, prev
        best, best_f = math.inf, None
        scale = np.array([1.0, 1.0, DZ_W])
        open_ = [c for c, f in self.frontiers.items() if f["node"] in dist]
        while open_ and self.round < 30 and all(self.exhausted(self.region(self.frontiers[c]["pos"])) for c in open_):
            self.round += 1                                   # every region had its turn: double the budgets
        for cell, f in self.frontiers.items():
            if f["node"] not in dist or self.exhausted(self.region(f["pos"])):
                continue
            score = dist[f["node"]] + float(np.linalg.norm(f["pos"] - self.nodes[f["node"]])) \
                + FRONTIER_W * float(np.linalg.norm((f["pos"] - goal) * scale))
            if np.linalg.norm(f["pos"] - obs.pos) < NEAR_FRONTIER_M:
                score += 1e6                                  # only if there is nothing farther
            if score < best:
                best, best_f = score, cell
        self.committed = best_f
        self.commit_best, self.commit_since = math.inf, obs.t
        return (self.frontiers[best_f] if best_f else None), dist, prev

    def escalate(self, obs):
        """Evidence the way on was classed shut behind us: no metre gained on the height-weighted
        distance left for ESCALATE_S. Open the soft margin round the place we were nearest, go back."""
        w = float(np.linalg.norm((obs.pos - self.target) * np.array([1.0, 1.0, DZ_W])))
        if self.best_lb is None or w < self.best_lb[1] - 1.0:
            self.best_lb = (obs.t, w, obs.pos.copy())
        elif obs.t - self.best_lb[0] > ESCALATE_S:
            self.map.soft_extra.append(self.map.voxel(self.best_lb[2]))
            self.revisit = self.best_lb[2]
            self.best_lb = (obs.t, self.best_lb[1], self.best_lb[2])
            self.committed, self.next_plan = None, 0.0

    def route_to(self, obs, goal):
        """Path toward the goal: straight from the local search when it reaches it, else to the
        committed frontier -- over the graph when far, by local A* when near. The wide search
        toward the goal (which also refreshes the frontier memory) runs every SURVEY_S only."""
        if self.revisit is not None and self.last is not None:
            if np.linalg.norm(obs.pos - self.revisit) > 4.0:
                dist, prev = self.graph_from_here()
                j = min(dist, key=lambda n: np.linalg.norm(self.nodes[n] - self.revisit))
                self.path_final = False
                return self.graph_path(prev, j)[1:] + [self.revisit]
            self.revisit = None
        if obs.t >= self.next_survey or self.committed is None:
            self.next_survey = obs.t + SURVEY_S
            stop = STOP_M if self.mode in ("rescue", "search") else 0.9
            local = shortcut(self.map.plan(obs.pos, goal, stop), self.map)
            if obs.t >= self.next_sweep:
                self.next_sweep = obs.t + SWEEP_S
                self.map.frontiers_seen += self.map.sweep(obs.pos, SWEEP_M)
            self.remember_frontiers(obs.pos)
            if self.map.reached_goal:
                self.committed = None
                self.path_final = True
                return local[1:]
        else:
            local = [obs.pos]
        self.path_final = False
        for _ in range(UNREACHED_TRIES):
            f, dist, prev = self.pick_frontier(obs, goal)
            if f is None:
                return local[1:]
            self.goal = f["pos"]
            if dist.get(f["node"], 0.0) > GRAPH_LEG_M:
                return self.graph_path(prev, f["node"])[1:] + [f["pos"]]
            path = shortcut(self.map.plan(obs.pos, f["pos"]), self.map)[1:]
            if path and (self.map.reached_goal or np.linalg.norm(f["pos"] - obs.pos) > UNREACHED_M):
                return path
            # Close by and out of reach (behind the mapped floor, say): drop it now, not after
            # GOAL_PATIENCE_S of hovering -- the next one was the same, for 550 s (easy).
            del self.frontiers[self.committed]
            self.committed = None
        # Not the local search toward the goal either: with the given position 9 m off, that led up
        # through a hole in the roof and out of the course (easy). The next replan goes on.
        return []

    def path_blocked(self):
        """Has rock been mapped onto the next few metres of the path since it was planned?"""
        prev = None
        for w in self.path[:4]:
            if prev is not None:
                n = int(np.linalg.norm(w - prev) / RES) + 1
                ks = keys_of(self.map.voxel(prev + (w - prev) * np.linspace(0, 1, n)[:, None])).tolist()
                if any(k in self.map.occ for k in ks):
                    return True
            prev = w
        return False

    # ---------------------------------------------------------------- mission
    def current_goal(self, obs):
        if self.mode == "enter":
            if np.linalg.norm(obs.pos - self.route[-1]) < ENTER_M:
                self.mode = "rescue" if self.target is not None else "home"
                self.next_plan = 0.0
                self.path = []
                self.map.cave_only, self.map.allowed_cache = True, {}
                self.map.entrance = self.route[-1]
            else:
                return self.route[-1]
        if self.mode in ("rescue", "search"):
            for det in obs.detections:
                if det.type == "survivor":          # the detector's fix beats the drifted coordinates
                    self.target = obs.pos + body_rotation(obs.roll, obs.pitch, obs.yaw) @ det.rel_pos
                    self.seen, self.seen_t, self.mode = True, obs.t, "rescue"
            dist = np.linalg.norm(obs.pos - self.target)
            if self.seen and dist < REACH_M and obs.t - self.seen_t < FRESH_S:
                self.mode = "home"                  # confirmed by sight, not by drifted coordinates
                self.next_plan = 0.0
            elif self.seen and dist < 2 * REACH_M and obs.t - self.seen_t > LOST_S:
                self.seen, self.mode = False, "search"
            elif not self.seen and (self.mode == "search" or self.search_now(obs, dist)):
                # Rescue first: keep searching while t + LB <= T allows. (Starting already inside the
                # drift radius, ~30 m late in a run, swept the known ground round a corner from the
                # survivor till the battery died: hard 4/16.)
                if self.search_until is None:
                    self.search_until = obs.t + SEARCH_S
                self.mode = "search" if RESCUE_FIRST or obs.t < self.search_until else "home"
                return self.target
            else:
                return self.target
        return self.home_odom

    # ---------------------------------------------------------------- control
    def follow(self, obs):
        """Velocity command along the path toward a carrot CARROT_M ahead.

        Cruising: full speed toward the carrot (the simulator's autopilot closes the velocity
        loop), capped by the stopping distance to rock in the tube ahead -- not by the floor or a wall beside us --
        sinking no faster than we can stop above the floor below, and backing off a ceiling closer than CEIL_M.
        Only on the final approach does the command shrink with the distance left."""
        if not self.path:
            return np.zeros(3)
        pos = obs.pos
        # Drop waypoints reached, or nearby and already passed (a fresh plan often starts just behind us).
        while len(self.path) > 1:
            off = pos - self.path[0]
            if np.linalg.norm(off) < 1.0 or (np.linalg.norm(off) < CRUMB_M and off @ (self.path[1] - self.path[0]) > 0):
                self.path.pop(0)
            else:
                break
        carrot = self.along(pos, CARROT_M)
        to_end = self.path[-1] - pos
        if carrot is None:                       # path shorter than the carrot distance
            if self.path_final:
                v = KP * to_end                  # arrive and stop
            else:                                # keep going: the next plan extends the path
                v = to_end / max(np.linalg.norm(to_end), 1e-6) * self.info["max_speed_xy"]
        else:
            d = carrot - pos
            v = d / max(np.linalg.norm(d), 1e-6) * self.info["max_speed_xy"]
        return self.cap_speed(obs, v)

    def cap_speed(self, obs, v):
        """v, slowed to what we can still stop from before the first rock in the tube ahead
        (v^2 = 2 a d). Walls beside the tube, the floor and the ceiling do not slow us: we are not
        flying into them."""
        v = np.array(v, dtype=float)
        pos = obs.pos
        R = body_rotation(obs.roll, obs.pitch, obs.yaw)
        pts = obs.lidar
        free = 10.0
        if len(pts) and np.linalg.norm(v) > 1e-3:
            u = R.T @ (v / np.linalg.norm(v))
            along = pts @ u
            side = np.linalg.norm(pts - along[:, None] * u, axis=1)
            hit = (along > 0) & (side < TUBE_R)
            if hit.any():
                free = float(along[hit].min())
            # Climbing or sinking steeply, the tube leaves the LiDAR's +-45 deg fan: the map still knows that rock.
            u = v / np.linalg.norm(v)
            for s in np.arange(RES, free, RES / 2):
                if int(keys_of(self.map.voxel(pos + s * u))) in self.map.occ:
                    free = float(s)
                    break
        vmax = float(np.clip(stop_speed(free - BRAKE_MARGIN), 0.3, self.info["max_speed_xy"]))
        sp = np.linalg.norm(v)
        if sp > vmax:
            v *= vmax / sp
        # Ceiling: the LiDAR sees no higher than 45 deg, so a ceiling h above shows in the top beams
        # about h out. Only one closer than CEIL_M matters, so only returns that steep and that near
        # count -- a tunnel wall a metre or two beside us is not a ceiling.
        up = pts @ R.T if len(pts) else np.zeros((0, 3))
        side = np.hypot(up[:, 0], up[:, 1])
        over = (up[:, 2] >= 0.9 * side) & (side < CEIL_M)
        if over.any() and obs.range_down >= DUST_GAP:      # in rotor dust, near returns cannot be trusted
            v[2] = min(v[2], up[over, 2].min() - CEIL_M)
        # Unknown is not free: climb no faster than we can stop CEIL_M short of the end of the space
        # mapped free above -- straight up is the bottom mount's blind cone (above ~33 deg), where
        # frontiers are rock as often as not. Into the unknown we creep (0.3 m/s is no hit).
        if v[2] > 0:
            free, clear = self.map.free, 0.0
            while clear < 4.0 and all(int(keys_of(self.map.voxel(pos + [dx, dy, clear + RES]))) in free
                                      for dx, dy in FOOTPRINT):
                clear += RES
            if int(keys_of(self.map.voxel(pos + [0.0, 0.0, clear + RES]))) in self.map.occ:
                v[2] = min(v[2], max(clear - CEIL_M, -SAFE_SET_V))   # known rock: no creeping into it, back off
            else:
                v[2] = min(v[2], max(CREEP_Z, float(stop_speed(max(clear - CEIL_M, 0.0)))))
        v[2] = np.clip(v[2], -self.info["max_speed_z"], self.info["max_speed_z"])
        if np.isfinite(obs.range_down):          # never sink faster than we can stop above the floor below
            v[2] = max(v[2], -float(stop_speed(obs.range_down - BRAKE_MARGIN)))
        return v

    def along(self, pos, dist):
        """The point dist metres ahead along the path from pos, or None if the path is shorter."""
        prev = pos
        for w in self.path:
            seg = np.linalg.norm(w - prev)
            if seg >= dist:
                return prev + (w - prev) * (dist / seg)
            dist -= seg
            prev = w
        return None

    def came_from_below(self, pos):
        return any(q[2] < pos[2] - 1.0 and np.hypot(*(q[:2] - pos[:2])) < RF_TRACK_M for q in self.nodes)

    def feel_ceiling(self, obs):
        """Rock right above that no beam can show (the bottom mount is blind above ~30 deg, and closer
        than its minimum range it sees nothing at all): felt instead. Commanding a climb that does not
        happen, or a descent the ceiling effect holds us back from, for FEEL_S -- clear of the floor --
        means rock within a voxel above: mapped so. Creeping up into a ceiling pocket at the cave mouth
        pinned the drone there 35 s, its frontiers never seen nor dropped (easy 0)."""
        if obs.gps is not None or obs.vel is None or not np.isfinite(obs.range_down) or obs.range_down < 1.0:
            self.held = None
            return
        cz, vz = float(self.last_cmd[2]), float(obs.vel[2])
        if not ((cz > 0.1 and vz < 0.05) or (cz < -0.2 and vz > -0.05)):
            self.held = None
        elif self.held is None:
            self.held = obs.t
        elif obs.t - self.held >= FEEL_S:
            self.held = None
            self.map.update([int(keys_of(self.map.voxel(obs.pos + [dx, dy, RES]))) for dx, dy in FOOTPRINT], L_MAX)
            # Within ~0.8 m the rotor wash sucks us up harder than a path's few tenths of a m/s down pull
            # back (commanding -0.4, rising +0.1 to +0.3, 36 s along a ceiling: easy 5): full descent.
            self.drop_until = obs.t + FEEL_DROP_S

    def guard(self, obs, v, use_map=True):
        """Last line of defence, a continuous one (a control-barrier filter): toward any return
        within GUARD_M, v may close in no faster than we could still stop SAFE_M short of it; inside
        SAFE_M we are pushed off in proportion. The old on/off 1 m/s push flipped the command from
        tick to tick, which a real airframe cannot follow."""
        R = body_rotation(obs.roll, obs.pitch, obs.yaw)
        pts = obs.lidar
        if GUARD_MAP and use_map and obs.range_down >= DUST_GAP:
            # The rock the map has seen, too: the side the LiDAR is blind to (the ceiling for a
            # bottom mount) is where the drone hit what it knew was there.
            # In rotor dust, near returns -- and the surface cells they made -- are not trusted.
            # Only where the LiDAR cannot see: elsewhere its live returns are the truth, and mapped
            # cells laid down under another pose error stood 0.3 m off with the nearest live return
            # 2 m away, and stopped the drone for 700 s, 26 m from the survivor (hard).
            m = (self.map.surface_near(obs.pos, GUARD_M) - obs.pos) @ R
            d = np.linalg.norm(m, axis=1)
            el = np.degrees(np.arcsin(np.clip(m[:, 2] / np.maximum(d, 1e-6), -1.0, 1.0)))
            lo, hi = self.info["lidar_vfov_deg"]
            keep = (el > min(hi, GUARD_SEEN_UP_DEG)) | (el < lo) | (d < self.info["lidar_min_range"] + 0.1)
            if self.mode == "home" and np.isfinite(obs.range_down) and self.came_from_below(obs.pos):
                # Straight down the rangefinder is the truth: cells on its beam short of its reading are
                # stale (a ceiling seen from below, the drone since up through a hole in it: 700 s
                # hovering over the hole, easy 20 m error). Only over our own track: a beam
                # through a crack in a ledge we never came through set the drone down in it (hard).
                # Only on the way home: on the way in it wedged the drone in a ceiling slot 4 times /64.
                w = m @ R.T
                keep &= ~((np.hypot(w[:, 0], w[:, 1]) < RF_BEAM_R) & (w[:, 2] < 0) & (w[:, 2] > -(obs.range_down - 0.3)))
            pts = np.vstack([pts, m[keep]])
        d = np.linalg.norm(pts, axis=1)
        near = (d < GUARD_M) & (d > 1e-6)
        v = np.array(v, dtype=float)
        if not near.any():
            return v
        n = R @ (pts[near] / d[near, None]).T                # (3, k) toward rock
        allow = stop_speed(d[near] - SAFE_M)
        # Rock beside the tube we sweep is not closed on by flying along it, though each of its points
        # ahead is nearer each tick: along a 0.7 m slot (0.4 m drone) they cut the command to nothing,
        # wedged 240 s (hard, 20 m error). So they still slow us, but only down to SAFE_SET_V:
        # ignored outright, the drone grazed walls at speed off its commanded line and crashed (2 of 12).
        sp = np.linalg.norm(v)
        if sp > 1e-6:
            along = (v / sp) @ n * d[near]
            allow = np.where(d[near] ** 2 - along ** 2 < GUARD_TUBE_R ** 2, allow, np.maximum(allow, SAFE_SET_V))
        for _ in range(3):                                   # the worst violations, a few at a time
            over = v @ n - allow
            i = int(np.argmax(over))
            if over[i] <= 0:
                break
            v -= over[i] * n[:, i]
        inside = d[near] < SAFE_M
        if inside.any():
            v -= n[:, inside] @ (2.0 * (SAFE_M - d[near][inside])) / inside.sum()
        return v

    def step(self, obs):
        """Re-localise the sweep against the old map (agents.lio) and fly the mission from there.
        The correction is a translation, so velocities and headings are the odometry's own."""
        if obs.landed:
            return Command()
        pts = deskew(obs)
        R = body_rotation(obs.roll, obs.pitch, obs.yaw)
        if obs.range_down < DUST_GAP:
            obs = dataclasses.replace(obs, lidar=obs.lidar[np.einsum("ij,ij->i", obs.lidar, obs.lidar) > DUST_R ** 2])
            # Nor into the map: dust in the drone's own voxel, where no beam passes to clear it, made it
            # rock -- "trapped", the drone crept round the spot in the dust for minutes (hard).
            pts = pts[np.einsum("ij,ij->i", pts, pts) > DUST_R ** 2]
        if SCAN_MATCH:
            p = self.lio.correct(R, np.asarray(obs.pos, dtype=float), pts, obs.t)
            obs = dataclasses.replace(obs, pos=p)   # the GNSS fix is in the world frame, which the correction is to
        cmd = self.mission_step(obs, pts @ R.T + obs.pos)
        if not cmd.land:                                     # no faster change than SLEW per tick
            dv = np.asarray(cmd.vel, dtype=float) - self.last_cmd
            if np.linalg.norm(dv) > SLEW:
                cmd.vel = self.last_cmd + dv * (SLEW / np.linalg.norm(dv))
            self.last_cmd = np.asarray(cmd.vel, dtype=float)
        return cmd

    def mission_step(self, obs, pts):
        """obs in the corrected frame; pts: this sweep's returns in it."""
        if self.prev_pos is not None:
            self.flown += float(np.linalg.norm(obs.pos - self.prev_pos))
        self.prev_pos = obs.pos.copy()
        if self.mode == "rescue" and self.target is not None:
            self.count_waste(obs)
            if ESCALATE_S:
                self.escalate(obs)
        self.map.insert(obs.pos, pts)
        if FEEL_S:
            self.feel_ceiling(obs)
        self.add_crumb(obs.pos)

        if self.mode in ("rescue", "search") and obs.t >= self.next_battery_check:
            # Turn back while the battery still covers the road home (GBPlanner's and CTU's homing rule).
            self.next_battery_check = obs.t + 1.0
            road = [obs.pos] + self.road_home()
            need = sum(float(np.linalg.norm(b - a)) for a, b in zip(road, road[1:])) / (0.6 * self.info["max_speed_xy"])
            near = self.seen or np.linalg.norm(obs.pos - self.target) < FINISH_M
            if self.battery0 is None:
                self.battery0, self.t0 = obs.battery, obs.t
            if RESCUE_FIRST:
                drain = (self.battery0 - obs.battery) / max(obs.t - self.t0, 1.0)      # per second, as flown
                left = min(obs.battery / max(drain, 1e-6), self.info["mission_time_s"] - obs.t)
                if np.linalg.norm(obs.pos - self.target) / self.info["max_speed_xy"] > left:
                    self.mode, self.next_plan = "home", 0.0
            elif (obs.battery < need / self.info["hover_endurance_s"] * 1.3 + 0.05
                    or obs.t > self.info["mission_time_s"] - need * 1.3 - 20) and not near:
                self.mode, self.next_plan = "home", 0.0
        goal = self.current_goal(obs)
        if RECOVERY and self.mode == "rescue" and not self.seen:
            cmd = self.recovery(obs)
            if cmd is not None:
                return cmd
        if self.mode == "home" and obs.gps is not None:
            # Back under open sky: GNSS (about 1 m) beats odometry that drifted through the cave.
            off = obs.gps - obs.pos
            self.gps_off = off if self.gps_off is None else self.gps_off + 0.05 * (off - self.gps_off)
            self.home_odom = self.home - self.gps_off
        if self.mode == "home" and np.linalg.norm((obs.pos - self.home_odom)[:2]) < 1.0:
            # The autopilot's landing holds no position, and the air near the ground pushes us about:
            # hold the pad ourselves down to a low hover, and only then let it touch down.
            if obs.range_down < LAND_RD and np.linalg.norm(np.asarray(obs.vel)[:2]) < 0.3:
                return Command(land=True)
            sink = -min(0.5, np.sqrt(2 * BRAKE_A * max(obs.range_down - (LAND_RD - 0.2), 0.0))) if np.isfinite(obs.range_down) else -0.5
            return Command(vel=np.r_[KP * (self.home_odom - obs.pos)[:2], sink])
        if self.mode == "search":
            return self.search(obs)
        if self.mode == "enter":
            # Outdoors on GPS: fly the known straight legs into the cave mouth, no planning.
            if not self.path:
                self.path = [p.copy() for p in self.route]
                self.path_final = False
        elif obs.t >= self.next_plan or obs.t - self.planned_at >= 1.0 and (
                not self.path or self.path_blocked() or
                (not self.path_final and self.along(obs.pos, AHEAD_M) is None) or
                (self.path_final and self.mode != "home" and np.linalg.norm(self.path[-1] - goal) > STOP_M + 1.0)):
            # (at most once a second off schedule: a path that stays blocked replanned every tick before)
            self.goal = goal
            self.planned_at = obs.t
            if self.mode == "home":
                road = self.road_home()
                self.path_final = True
                # The road was laid with the odometry of the way out; with drift, rock mapped since
                # sits on it. It only gives the direction: the local planner, on the map as it is now,
                # takes us to a node HOME_LEG_M down the road, and the road goes on from there.
                # Outdoors (GPS fix) the road is the entry route over open ground: fly it as is; A* there
                # takes the open sky for unexplored space and climbs into it.
                # So is the staging area out of the GNSS fix (under the cave mouth's overhang): the course
                # rule bars it to plans, and the drone hovered on an empty plan 4 m from the mouth till
                # the battery died (hard).
                outdoors = obs.gps is not None or not self.map.allowed(int(keys_of(self.map.voxel(obs.pos))))
                # No headway home for HOME_STUCK_S: the map has closed a neck we flew through (rock laid
                # down under another pose error -- 600 s at the 108 m neck, hard). The road
                # itself was flown: follow it as it is, the guard keeps us off the rock.
                left = sum(float(np.linalg.norm(b - a)) for a, b in zip(road, road[1:])) + float(np.linalg.norm(road[0] - obs.pos))
                if self.home_ref is None or left < self.home_ref[1] - 1.0:
                    self.home_ref = (obs.t, left)
                outdoors |= HOME_STUCK_S > 0 and obs.t - self.home_ref[0] > HOME_STUCK_S
                j = 0 if outdoors else self.home_leg_end(road)
                if j == 0:
                    self.path = road
                else:
                    leg = shortcut(self.map.plan(obs.pos, road[j], STOP_M if j < len(road) - 1 else 0.9), self.map)
                    # Not reached (the node may now lie in rock): still the best way that way; replanned soon.
                    self.path = leg[1:] + road[j + 1:] if self.map.reached_goal else leg[1:]
            else:
                self.path = self.route_to(obs, goal)
            self.next_plan = obs.t + REPLAN_S

        # Outside the passable set the plan has nowhere to start from (squeezed under a ceiling the
        # bottom mount never saw, the drone sat on an empty plan till the battery died).
        # Being there is itself the evidence: go back to the nearest certified voxels first.
        # Only when trapped -- in mapped rock, or with no plan at all: squeezing through margin voxels
        # is how narrow passages are flown, and pulling out of every one stalled 15 of 16 hard runs.
        trapped = not self.path or int(keys_of(self.map.voxel(obs.pos))) in self.map.occ
        back = self.map.way_back(obs.pos) if SAFE_SET and trapped and self.mode != "enter" else None
        # Way back on the live returns only: the map that says we sit in rock is what is wrong here,
        # and its surface cells all round cancelled the way back to nothing (seed 15).
        if back is not None and OPEN_WAY:
            back = self.escape_way(obs, back)
        else:
            self.escape = None
            if not trapped:
                self.open_at = obs.pos.copy()
        v = self.guard(obs, SAFE_SET_V * back, use_map=False) if back is not None else self.guard(obs, self.follow(obs))
        if back is not None and OPEN_WAY and np.linalg.norm(v) < 0.3 * SAFE_SET_V:
            # Wedged in a crack, rock 0.24 m off all round: the guard's push-offs cancel to nothing
            # (600 s at one spot, hard). The open_way tube is clear: leave along it, unguarded, at a
            # quarter of the crash speed.
            v = SAFE_SET_V * back
        if obs.t < self.drop_until and np.isfinite(obs.range_down) and obs.range_down > 1.5:
            v[2] = -self.info["max_speed_z"]
        if self.mode == "enter" and np.isfinite(obs.range_down):
            # Like an aircraft: height is judged against the ground below, not an absolute altitude.
            v[2] = max(v[2], 1.0 * (AGL_MIN_M - obs.range_down))
        elif np.isfinite(obs.range_down):
            # Never sink onto the floor: a given position below it pushed the drone down through a hole
            # in the floor mesh (easy, out of course), or onto it beside rock in rotor dust (hard).
            v[2] = max(v[2], 1.0 * (FLOOR_MIN_M - obs.range_down))
        # Nose toward where the path goes a few seconds on, not the momentary velocity: push-offs and
        # carrot swaps no longer swing it, and the camera looks round a bend before we get there.
        look = self.along(obs.pos, LOOK_M) if self.path else None
        if look is None and self.path:
            look = self.path[-1]
        ahead = (look - obs.pos)[:2] if look is not None else v[:2]
        heading = np.arctan2(ahead[1], ahead[0]) if np.linalg.norm(ahead) > 1.0 else obs.yaw
        yaw_rate = np.clip(2.0 * ((heading - obs.yaw + np.pi) % (2 * np.pi) - np.pi), -YAW_RATE_MAX, YAW_RATE_MAX)
        return Command(vel=v, yaw_rate=yaw_rate)

    def escape_way(self, obs, back):
        """Direction out of a trap, given the map's way back."""
        rel = np.vstack([obs.lidar @ body_rotation(obs.roll, obs.pitch, obs.yaw).T,
                         self.map.surface_near(obs.pos, 2.0) - obs.pos])
        free = tube_free(rel)
        is_open = free is not None and (free >= OPEN_MIN).any()
        to_open = None if self.open_at is None else self.open_at - obs.pos
        if is_open or to_open is None or np.linalg.norm(to_open) < 0.3:
            # Keep the way out we took while it stays open: the map's way back swings round as rock
            # is added, and the choice flipped every second, 180 deg, going nowhere (hard).
            self.escape = open_way(rel, back if self.escape is None else self.escape)
            if is_open:
                self.open_at = obs.pos.copy()
        else:
            # Rock 0.25 m off all round, no tube open anywhere: the pick among closed directions was
            # noise and flipped every tick, the drone crept up a crack for 600 s (hard 36).
            # We got in here, so the way we came is flyable: back to where a way out was last open.
            self.escape = to_open / np.linalg.norm(to_open)
        return self.escape

    def recovery(self, obs):
        """Stuck: STUCK_S without STUCK_M of headway. Back off over the graph of places flown
        through (certified by the drone itself) to a node RETREAT_M away, turn once round to map
        what the LiDAR's blind sides hid, drop the frontier we were stuck on, and replan.
        Returns the command while recovering, else None."""
        if self.stuck_ref is None or np.linalg.norm(obs.pos - self.stuck_ref[1]) > STUCK_M:
            self.stuck_ref = (obs.t, obs.pos.copy())
        r = self.recover
        if r is None:
            if obs.t - self.stuck_ref[0] < STUCK_S or self.last is None:
                return None
            dist, prev = self.graph_from_here()
            far = [j for j in dist if np.linalg.norm(self.nodes[j] - obs.pos) >= RETREAT_M]
            if not far:
                self.stuck_ref = (obs.t, obs.pos.copy())
                return None
            j = min(far, key=lambda j: dist[j])
            self.path = self.graph_path(prev, j)[1:] + [self.nodes[j].copy()]
            self.path_final = True
            r = self.recover = {"phase": "back", "until": obs.t + 20.0, "node": self.nodes[j].copy()}
        if r["phase"] == "back":
            if np.linalg.norm(obs.pos - r["node"]) < 1.0 or obs.t > r["until"]:
                r["phase"], r["until"] = "look", obs.t + LOOK_S
            else:
                return Command(vel=self.guard(obs, self.follow(obs)), yaw_rate=0.0)
        if obs.t < r["until"]:
            return Command(vel=self.guard(obs, np.zeros(3)), yaw_rate=YAW_RATE_MAX)
        if self.committed in self.frontiers:          # the way we were stuck on: tried
            del self.frontiers[self.committed]
        self.committed, self.recover = None, None
        self.stuck_ref = (obs.t, obs.pos.copy())
        self.path, self.next_plan, self.next_survey = [], 0.0, 0.0
        return None

    def search_now(self, obs, dist):
        """Start searching: within SEARCH_M of the given position; with an error, within it plus the
        detector's range (passed 6 m from the survivor, 22 m from the coordinates, and never came
        back: easy, 20 m error) -- or within twice that, if we have stopped getting closer: the given
        position may be in rock (25 m from the survivor, 38 m from it, for 500 s: hard).
        Twice the error outright started too early (hard 20 m: 9/24)."""
        err = self.info.get("target_error_m", 0.0)
        if err <= 0:
            return dist < SEARCH_M
        if self.closest is None or dist < self.closest[1] - 1.0:
            self.closest = (obs.t, dist)
        reach = err + self.info["detector_range"]
        return dist < reach or (dist < 2 * reach and obs.t - self.closest[0] > SEARCH_STALL_S)

    def search_speed(self):
        """Fastest sweep that still sees every point within SEARCH_LOOK of the track. A point l off it
        stays within the detector's range r for 2 sqrt(r^2 - l^2) / v, while its bearing swings by
        2 atan(sqrt(r^2 - l^2) / l); turning the same way at YAW_RATE_MAX, the view must still gain a
        full turn less its own fov on it. With 6 m, 120 deg: 1.27 m/s (it was 1 m/s).
        (Leaving out the swing gave 1.83 m/s, which a test sweep showed missing a third of the headings.)"""
        r = self.info["detector_range"]
        a = np.sqrt(r ** 2 - (SEARCH_LOOK * r) ** 2)
        lag = 2 * np.pi * (1 - self.info["detector_half_fov_deg"] / 180.0) + 2 * np.arctan2(a, SEARCH_LOOK * r)
        return 2 * a * YAW_RATE_MAX / lag

    def search_radius(self):
        return max(SEARCH_MIN_M, SEARCH_DRIFT * self.flown + self.info.get("target_error_m", 0.0))

    def search(self, obs):
        """Not at the coordinates we were given: our odometry has drifted (and so may theirs).
        The detector only sees SEARCH_LOOK of its range reliably, so cover the drift radius around
        the target: go to the nearest place we can fly to that no point of our search has yet come
        within SEARCH_LOOK of, turning the detector all the way round as we go. (Mapped is not
        searched: the LiDAR maps a chamber from afar, the detector needs us in it.)"""
        radius = self.search_radius()
        if not self.searched or np.linalg.norm(obs.pos - self.searched[-1]) > 1.0:
            self.searched.append(obs.pos.copy())
        if obs.t >= self.next_plan or not self.path:
            look = SEARCH_LOOK * self.info["detector_range"]
            n = int(radius / RES)
            grid = np.array(np.meshgrid(*[np.arange(-n, n + 1, 2)] * 3)).reshape(3, -1).T
            grid = grid[np.linalg.norm(grid, axis=1) * RES <= radius] + self.map.voxel(self.target)
            # In the course only: a high given position puts part of the sphere above ground (easy).
            ks = [k for k in keys_of(grid).tolist() if self.map.passable(k) and self.map.allowed(k)]
            seen = np.array(self.searched)
            cand = []                                  # (frontier cell, or None for a mapped place; place)
            if ks:
                c = np.array([self.map.centre(k) for k in ks])
                cand += [(None, g) for g in c[np.min(np.linalg.norm(c[:, None] - seen[None], axis=2), axis=1) > look]]
            # The unmapped is searched too, on the same footing: sweeping only what we had mapped kept
            # a drone 30-45 m from a given position in rock while the survivor sat 16 m from it, past a
            # neck we had not flown (hard, 20 m error: 17 of 20 failures).
            if obs.t >= self.next_survey:
                # Frontiers are met only by searches toward a goal, which stop at the given position:
                # the far side of the ball went unseen while the survivor sat 20 m from it there
                # (hard, 20 m error). Search toward it and six points round it.
                self.next_survey = obs.t + SURVEY_S
                for o in np.vstack([np.zeros(3), np.eye(3), -np.eye(3)]) * 0.7 * radius:
                    self.map.plan(obs.pos, self.target + o, STOP_M)
                    self.remember_frontiers(obs.pos)
            cand += [(cell, f["pos"]) for cell, f in self.frontiers.items()
                     if np.linalg.norm(f["pos"] - self.target) <= radius and cell not in self.given_up]
            # Likeliest first -- near the given position -- then near us: nearest-to-us alone swept
            # outward, 40 m off the coordinates and down a level, 14 m from the survivor at the start.
            cand.sort(key=lambda cg: np.linalg.norm(cg[1] - self.target) + 0.5 * np.linalg.norm(cg[1] - obs.pos))
            # Keep the place we set out for till we get there, it proves out of reach, or we stop
            # gaining on it: re-ranked every replan, the choice swung between one ahead and one behind
            # and the drone went to and fro for 300 s, 26 m from the survivor (hard).
            sg = self.search_goal
            if sg is not None:
                d = float(np.linalg.norm(sg[1] - obs.pos))
                if d < sg[3] - 0.5:
                    sg[2], sg[3] = obs.t, d
                live = sg[0] is None or sg[0] in self.frontiers
                if live and d > look and obs.t - sg[2] < GOAL_PATIENCE_S:
                    cand.insert(0, (sg[0], sg[1]))
                else:
                    if sg[0] is None:
                        self.searched.append(sg[1])
                    else:
                        # For good: the next survey met it again, it ranked first again, and the drone sat
                        # 5 m short of it for 360 s, wedged (hard 52, 20 m error).
                        self.given_up.add(sg[0])
                        self.frontiers.pop(sg[0], None)
                    # cand was ranked before this: without this, the same replan picked it again, patience anew
                    cand = [cg for cg in cand if not np.array_equal(cg[1], sg[1])]
                    self.search_goal = None
            sg = self.search_goal
            kept = self.path if sg is not None and self.search_path_to is not None and np.array_equal(sg[1], self.search_path_to) else []
            self.path, self.search_path_to = [], None
            for cell, g in cand[:UNREACHED_TRIES]:
                self.goal = g
                path = shortcut(self.map.plan(obs.pos, g), self.map)[1:]
                reached = self.map.reached_goal
                if not reached and kept and np.array_equal(g, sg[1])                         and not any(int(keys_of(self.map.voxel(p))) in self.map.occ for p in kept):
                    # The way there came and went with a neck's voxels flickering free and rock: each miss
                    # sent the drone 15 m off after another frontier and back, 100-160 s at the hard
                    # survivor (20 m error). Keep the way we had while no rock is mapped on it.
                    path, reached = kept, True
                if path and (reached or (cell is not None and np.linalg.norm(g - obs.pos) > UNREACHED_M)):
                    self.path = path
                    self.search_path_to = g if reached else None
                    if self.search_goal is None or self.search_goal[0] != cell or not np.array_equal(self.search_goal[1], g):
                        self.search_goal = [cell, g, obs.t, float(np.linalg.norm(g - obs.pos))]
                    break
                # Out of reach from here: count it searched (or forget the frontier), or the sweep
                # follows partial plans round the known ground for good (hard, 20 m error).
                if cell is None:
                    self.searched.append(g)
                elif cell in self.frontiers:
                    del self.frontiers[cell]
                self.search_goal = None
            self.path_final = True
            if not self.path and RESCUE_FIRST:
                # Nothing left to search inside the radius: explore toward it instead of hovering.
                self.path = self.route_to(obs, self.target)
            self.next_plan = obs.t + REPLAN_S
        v = self.follow(obs)
        sp = np.linalg.norm(v)
        if sp > self.search_speed():
            v *= self.search_speed() / sp
        v = self.guard(obs, v)
        if np.isfinite(obs.range_down):
            v[2] = max(v[2], 1.0 * (FLOOR_MIN_M - obs.range_down))
        return Command(vel=v, yaw_rate=YAW_RATE_MAX)

    def debug(self):
        out = {"mode": self.mode, "path": [p.tolist() for p in self.path[::-1]],
               "goal": None if self.goal is None else np.asarray(self.goal).tolist(),
               # white dots in the viewer: frontiers still to visit
               "points": [f["pos"].tolist() for f in self.frontiers.values()][:400]}
        # Our own map (rock voxels, thinned to one per metre), every 20th call: the viewer keeps it.
        self.debug_calls = getattr(self, "debug_calls", 0) + 1
        if self.debug_calls % 20 == 1 and self.map.occ:
            k = np.fromiter(self.map.occ, np.int64, len(self.map.occ))
            idx = np.stack([(k >> 42) - OFF, ((k >> 21) & 0x1FFFFF) - OFF, (k & 0x1FFFFF) - OFF], 1)
            out["map"] = (np.unique(idx // 2, axis=0) * 2 * RES + RES).round(1)
        return out


Agent = PlannerAgent
