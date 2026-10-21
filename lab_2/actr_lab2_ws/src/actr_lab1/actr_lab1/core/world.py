"""The map of the world: places, rooms, cubes, and the occupancy grid.

Layout   - reads config/places.json (names, polygons, waypoints)
OccupancyGrid - reads maps/rooms.yaml + rooms.pgm (the same files Nav2 uses)
Everything here is plain Python (no ROS), so it also works in the simulator.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------- small geometry
@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0

    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.yaw)


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting: count how many polygon edges a ray to the right crosses."""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


# ---------------------------------------------------------------------- layout
@dataclass
class Place:
    name: str
    kind: str                      # "room", "hall" or "restricted"
    polygon: List[List[float]]
    waypoints: List[Pose2D]
    aliases: List[str]

    def contains(self, x: float, y: float) -> bool:
        return point_in_polygon(x, y, self.polygon)


@dataclass
class CubeInfo:
    name: str
    start: Tuple[float, float]
    aliases: List[str]


class Layout:
    def __init__(self, data: dict, base_dir: str = "."):
        self.base_dir = base_dir
        self.frame = data.get("frame", "map")
        self.robot_start = Pose2D(*data["robot_start"])
        self.robot_radius = float(data.get("robot_radius", 0.105))
        self.cube_size = float(data.get("cube_size", 0.22))
        self.pallet_height = float(data.get("pallet_height", 0.0))     # every cube stands on a small pallet
        self.pallet_size = float(data.get("pallet_size", self.cube_size))
        # height of the centre of a cube that stands (on its pallet) on the floor
        self.cube_rest_z = self.pallet_height + self.cube_size / 2
        self.goal_room = data.get("goal_room", "room_a")
        self.forbidden = set(data.get("forbidden", []))
        self.map_yaml = data.get("map_yaml", "maps/rooms.yaml")
        self.places: Dict[str, Place] = {}
        for name, p in data["places"].items():
            self.places[name] = Place(
                name=name, kind=p["kind"], polygon=[list(pt) for pt in p["polygon"]],
                waypoints=[Pose2D(*wp) for wp in p["waypoints"]], aliases=list(p.get("aliases", [])))
        self.cubes: Dict[str, CubeInfo] = {}
        for name, c in data["cubes"].items():
            self.cubes[name] = CubeInfo(name, tuple(c["start"]), list(c.get("aliases", [name])))

    @classmethod
    def load(cls, path: str) -> "Layout":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, base_dir=os.path.dirname(os.path.dirname(os.path.abspath(path))))

    def map_yaml_path(self) -> str:
        return os.path.join(self.base_dir, self.map_yaml)

    # -- queries -----------------------------------------------------------------
    def room_at(self, x: float, y: float) -> str:
        """Name of the room or hall that contains the point ('' if none)."""
        for name, p in self.places.items():
            if p.kind in ("room", "hall") and p.contains(x, y):
                return name
        return ""

    def in_forbidden(self, x: float, y: float) -> bool:
        return any(self.places[n].contains(x, y) for n in self.forbidden if n in self.places)

    def is_forbidden_place(self, name: str) -> bool:
        return name in self.forbidden

    def is_room(self, name: str) -> bool:
        return name in self.places and self.places[name].kind in ("room", "hall")

    def waypoint(self, place: str, variant: int = 0) -> Optional[Pose2D]:
        p = self.places.get(place)
        if p is None or not p.waypoints:
            return None
        return p.waypoints[variant % len(p.waypoints)]

    def n_waypoints(self, place: str) -> int:
        p = self.places.get(place)
        return 0 if p is None else len(p.waypoints)


# ------------------------------------------------------------------------ grid
def _read_yaml_simple(path: str) -> dict:
    """The map yaml has only 'key: value' lines, so a tiny reader is enough."""
    try:
        import yaml  # type: ignore
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        out = {}
        with open(path) as f:
            for line in f:
                if ":" not in line or line.strip().startswith("#"):
                    continue
                k, v = line.split(":", 1)
                v = v.strip()
                if v.startswith("["):
                    out[k.strip()] = [float(t) for t in v.strip("[]").split(",")]
                else:
                    try:
                        out[k.strip()] = float(v)
                    except ValueError:
                        out[k.strip()] = v
        return out


def _read_pgm(path: str) -> Tuple[int, int, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    # header: magic, width height, maxval; '#' lines are comments
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1  # one whitespace byte after maxval
    if tokens[0] != b"P5":
        raise ValueError("only binary PGM (P5) is supported")
    width, height = int(tokens[1]), int(tokens[2])
    return width, height, data[pos:pos + width * height]


class OccupancyGrid:
    """Static map. Cell (ix, iy): ix grows with x, iy grows with y (y up)."""

    def __init__(self, width: int, height: int, resolution: float, origin: Tuple[float, float],
                 blocked: bytearray, unknown: bytearray):
        self.width, self.height, self.res = width, height, resolution
        self.ox, self.oy = origin
        self.blocked = blocked      # 1 = occupied
        self.unknown = unknown      # 1 = unknown
        self._inflated: Dict[float, bytearray] = {}
        self._paths: Dict[tuple, Optional[List[Tuple[float, float]]]] = {}

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OccupancyGrid":
        meta = _read_yaml_simple(yaml_path)
        pgm = os.path.join(os.path.dirname(yaml_path), str(meta["image"]))
        width, height, pix = _read_pgm(pgm)
        negate = int(meta.get("negate", 0))
        occ_t = float(meta.get("occupied_thresh", 0.65))
        free_t = float(meta.get("free_thresh", 0.196))
        blocked = bytearray(width * height)
        unknown = bytearray(width * height)
        for row in range(height):
            iy = height - 1 - row
            for col in range(width):
                v = pix[row * width + col]
                p = (v / 255.0) if negate else (255 - v) / 255.0
                idx = iy * width + col
                if p > occ_t:
                    blocked[idx] = 1
                elif p >= free_t:
                    unknown[idx] = 1
        origin = meta["origin"]
        return cls(width, height, float(meta["resolution"]), (float(origin[0]), float(origin[1])), blocked, unknown)

    # -- conversions -------------------------------------------------------------
    def to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor((x - self.ox) / self.res)), int(math.floor((y - self.oy) / self.res)))

    def to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        return (self.ox + (ix + 0.5) * self.res, self.oy + (iy + 0.5) * self.res)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height

    # -- obstacle inflation --------------------------------------------------------
    def inflated(self, radius: float) -> bytearray:
        """Cells closer than `radius` to an occupied cell (or unknown cell) are not usable."""
        key = round(radius, 3)
        if key in self._inflated:
            return self._inflated[key]
        r = int(math.ceil(radius / self.res))
        offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                   if math.hypot(dx, dy) * self.res <= radius]
        out = bytearray(self.width * self.height)
        w, h = self.width, self.height
        for iy in range(h):
            base = iy * w
            for ix in range(w):
                i = base + ix
                if self.unknown[i]:
                    out[i] = 1
                elif self.blocked[i]:
                    for dx, dy in offsets:
                        jx, jy = ix + dx, iy + dy
                        if 0 <= jx < w and 0 <= jy < h:
                            out[jy * w + jx] = 1
        self._inflated[key] = out
        return out

    def is_free(self, x: float, y: float, radius: float = 0.0) -> bool:
        ix, iy = self.to_cell(x, y)
        if not self.in_bounds(ix, iy):
            return False
        grid = self.inflated(radius) if radius > 0 else None
        i = iy * self.width + ix
        if grid is not None:
            return not grid[i]
        return not (self.blocked[i] or self.unknown[i])

    # -- path planning (A*), used only by the simulator --------------------------------
    def astar(self, start: Tuple[float, float], goal: Tuple[float, float], radius: float,
              step: int = 2) -> Optional[List[Tuple[float, float]]]:
        """Shortest path on a coarse grid (step x step cells), 8 neighbours.
        Returns a list of (x, y) points or None if there is no path."""
        blocked = self.inflated(radius)
        w, h = self.width, self.height
        cw, ch = w // step, h // step

        def coarse_free(cx, cy):
            fx, fy = cx * step + step // 2, cy * step + step // 2
            return 0 <= cx < cw and 0 <= cy < ch and not blocked[fy * w + fx]

        def nearest_free(c):
            if coarse_free(*c):
                return c
            for rad in range(1, 8):
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        if max(abs(dx), abs(dy)) == rad and coarse_free(c[0] + dx, c[1] + dy):
                            return (c[0] + dx, c[1] + dy)
            return None

        sx, sy = self.to_cell(*start)
        gx, gy = self.to_cell(*goal)
        s = nearest_free((sx // step, sy // step))
        g = nearest_free((gx // step, gy // step))
        if s is None or g is None:
            return None
        key = (s, g, round(radius, 3), step)
        if key in self._paths:
            return self._paths[key]

        def h_cost(c):
            dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
            return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

        open_heap = [(h_cost(s), 0.0, s)]
        best = {s: 0.0}
        parent = {}
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        found = False
        while open_heap:
            _, cost, cur = heapq.heappop(open_heap)
            if cur == g:
                found = True
                break
            if cost > best.get(cur, 1e18):
                continue
            for dx, dy, c in moves:
                nxt = (cur[0] + dx, cur[1] + dy)
                if not coarse_free(*nxt):
                    continue
                if dx and dy and not (coarse_free(cur[0] + dx, cur[1]) and coarse_free(cur[0], cur[1] + dy)):
                    continue  # do not cut corners
                new_cost = cost + c
                if new_cost < best.get(nxt, 1e18):
                    best[nxt] = new_cost
                    parent[nxt] = cur
                    heapq.heappush(open_heap, (new_cost + h_cost(nxt), new_cost, nxt))
        if not found:
            self._paths[key] = None
            return None
        cells = [g]
        while cells[-1] != s:
            cells.append(parent[cells[-1]])
        cells.reverse()
        path = [self.to_world(c[0] * step + step // 2, c[1] * step + step // 2) for c in cells]
        self._paths[key] = path
        return path


# ----------------------------------------------------- grounding helper functions
def approach_candidates(grid: OccupancyGrid, layout: Layout, cube_xy: Tuple[float, float],
                        robot_xy: Tuple[float, float], radii: Sequence[float] = (0.55, 0.65),
                        n_angles: int = 12, margin: float = 0.08) -> List[Pose2D]:
    """Free poses around a cube, facing it. The closest one to the robot comes first."""
    clearance = layout.robot_radius + margin
    cands = []
    for r in radii:
        for k in range(n_angles):
            a = 2 * math.pi * k / n_angles
            x, y = cube_xy[0] + r * math.cos(a), cube_xy[1] + r * math.sin(a)
            if not grid.is_free(x, y, clearance):
                continue
            if layout.in_forbidden(x, y):
                continue
            yaw = math.atan2(cube_xy[1] - y, cube_xy[0] - x)
            cands.append(Pose2D(x, y, yaw))
    cands.sort(key=lambda p: dist(p.xy(), robot_xy))
    return cands


def find_drop_point(grid: OccupancyGrid, layout: Layout, robot: Pose2D, room: str,
                    other_cubes: Sequence[Tuple[float, float]], reach: float = 0.55,
                    min_gap: float = 0.40,
                    accept: Optional[Callable[[Tuple[float, float]], bool]] = None) -> Optional[Tuple[float, float]]:
    """Where to put the carried cube: close to the robot, inside the room, on free floor,
    not on top of another cube. We try the direction the robot looks to first.
    accept(xy): the robot can say no to a point (the forklift checks the way of its load there)."""
    place = layout.places.get(room)
    if place is None:
        return None
    clearance = layout.cube_size / 2 + 0.05
    for radius in (reach, reach + 0.15, reach + 0.30):
        for deg in (0, 40, -40, 80, -80, 120, -120, 160, 180):
            a = robot.yaw + math.radians(deg)
            x, y = robot.x + radius * math.cos(a), robot.y + radius * math.sin(a)
            if not place.contains(x, y):
                continue
            if not grid.is_free(x, y, clearance):
                continue
            if any(dist((x, y), c) < min_gap for c in other_cubes):
                continue
            if layout.in_forbidden(x, y):
                continue
            if accept is not None and not accept((x, y)):
                continue
            return (x, y)
    return None
