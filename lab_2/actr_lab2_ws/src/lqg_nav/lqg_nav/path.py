"""The reference path: the points of the Nav2 planner, made smooth and evenly spaced.

The planner works on a grid (5 cm cells), so its path has small steps. The controller needs a smooth line with a
direction (heading) and a curvature at every point:
    1. drop repeated points,  2. moving average (keeps the first and the last point),
    3. resample every `ds` metres,  4. heading = direction between neighbours,  5. curvature = d(heading)/ds.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .lqg import wrap

Point = Tuple[float, float]


def _moving_average(points: List[Point], half: int) -> List[Point]:
    if half <= 0 or len(points) < 3:
        return list(points)
    out = [points[0]]
    for i in range(1, len(points) - 1):
        h = min(half, i, len(points) - 1 - i)          # a smaller window near the ends: the ends stay in place
        xs = [p[0] for p in points[i - h:i + h + 1]]
        ys = [p[1] for p in points[i - h:i + h + 1]]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    out.append(points[-1])
    return out


def _resample(points: List[Point], ds: float) -> List[Point]:
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = lengths[-1]
    out: List[Point] = []
    j = 0
    n = int(total / ds)
    for k in range(n + 1):
        s = k * ds
        while j < len(points) - 2 and lengths[j + 1] < s:
            j += 1
        seg = lengths[j + 1] - lengths[j]
        f = 0.0 if seg <= 0 else (s - lengths[j]) / seg
        a, b = points[j], points[j + 1]
        out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    if total - n * ds > 1e-6:
        out.append(points[-1])                         # the goal itself is always the last point
    return out


class RefPath:
    def __init__(self, points: Sequence[Sequence[float]], ds: float = 0.05, smooth: int = 3):
        pts: List[Point] = []
        for p in points:
            q = (float(p[0]), float(p[1]))
            if not pts or math.hypot(q[0] - pts[-1][0], q[1] - pts[-1][1]) > 1e-3:
                pts.append(q)
        if not pts:
            raise ValueError("empty path")
        self.ds = ds
        if len(pts) == 1:
            self.xs, self.ys, self.headings, self.curvatures = [pts[0][0]], [pts[0][1]], [0.0], [0.0]
            self.length = 0.0
            return
        pts = _resample(_moving_average(pts, smooth), ds)
        self.xs = [p[0] for p in pts]
        self.ys = [p[1] for p in pts]
        n = len(pts)
        self.headings = []
        for i in range(n):
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
            self.headings.append(math.atan2(b[1] - a[1], b[0] - a[0]))
        raw = [0.0] * n
        for i in range(1, n - 1):
            step = math.hypot(pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1])
            raw[i] = wrap(self.headings[i + 1] - self.headings[i - 1]) / max(step, 1e-6)
        self.curvatures = [sum(raw[max(0, i - 2):i + 3]) / len(raw[max(0, i - 2):i + 3]) for i in range(n)]
        self.length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(n - 1))

    def __len__(self) -> int:
        return len(self.xs)

    @property
    def end(self) -> Point:
        return self.xs[-1], self.ys[-1]

    def nearest(self, x: float, y: float, start: int = 0, back: int = 10, ahead: int = 80) -> int:
        """Index of the path point closest to (x, y), searched near the last index (the robot does not jump)."""
        lo, hi = max(0, start - back), min(len(self.xs), start + ahead + 1)
        return min(range(lo, hi), key=lambda i: (self.xs[i] - x) ** 2 + (self.ys[i] - y) ** 2)

    def errors(self, i: int, x: float, y: float, yaw: float) -> Tuple[float, float]:
        """(e_y, e_theta) at the path point i: left of the path = positive e_y."""
        th = self.headings[i]
        dx, dy = x - self.xs[i], y - self.ys[i]
        return -math.sin(th) * dx + math.cos(th) * dy, wrap(yaw - th)

    def remaining(self, i: int) -> float:
        """Path length from the point i to the end."""
        return max(0.0, self.length - i * self.ds) if i < len(self.xs) - 1 else 0.0

    def max_curvature(self, i: int, distance: float) -> float:
        """Largest |curvature| on the next `distance` metres (to slow down before a curve, not in it)."""
        j = min(len(self.curvatures), i + 1 + int(distance / self.ds))
        return max(abs(k) for k in self.curvatures[i:j]) if j > i else 0.0

    def ahead(self, i: int, distance: float) -> Point:
        """The point `distance` metres further along the path (or the end)."""
        j = min(len(self.xs) - 1, i + int(round(distance / self.ds)))
        return self.xs[j], self.ys[j]
