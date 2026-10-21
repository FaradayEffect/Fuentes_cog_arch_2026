"""Pictures of a run (needs matplotlib; everything else in the project works without it)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def have_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def draw_map(ax, layout, grid) -> None:
    import numpy as np
    img = np.full((grid.height, grid.width), 1.0)
    blocked = np.frombuffer(bytes(grid.blocked), dtype=np.uint8).reshape(grid.height, grid.width)
    unknown = np.frombuffer(bytes(grid.unknown), dtype=np.uint8).reshape(grid.height, grid.width)
    img[unknown == 1] = 0.88
    img[blocked == 1] = 0.25
    extent = [grid.ox, grid.ox + grid.width * grid.res, grid.oy, grid.oy + grid.height * grid.res]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=extent, interpolation="nearest")
    for name, place in layout.places.items():
        xs = [p[0] for p in place.polygon] + [place.polygon[0][0]]
        ys = [p[1] for p in place.polygon] + [place.polygon[0][1]]
        if place.kind == "restricted":
            ax.fill(xs, ys, color="red", alpha=0.18)
            ax.plot(xs, ys, color="red", lw=1, ls="--")
        cx = sum(xs[:-1]) / len(xs[:-1])
        cy = sum(ys[:-1]) / len(ys[:-1]) if place.kind == "restricted" else min(ys) + 0.45
        ax.text(cx, cy, name, ha="center", va="center", fontsize=9, color="#335", alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_run(path: str, layout, grid, trace: Sequence[Tuple[float, float, float]], start_cubes, final_cubes,
             events: Optional[List[dict]] = None, title: str = "") -> None:
    """Map + path of the robot + cubes (start: empty square, end: filled) + marks for problems."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    draw_map(ax, layout, grid)
    if trace:
        ax.plot([p[1] for p in trace], [p[2] for p in trace], color="#1565c0", lw=1.6, label="robot path")
        ax.plot(trace[0][1], trace[0][2], "o", color="#1565c0", ms=7)
    half = max(layout.cube_size, 0.16) / 2          # the cubes are small (8 cm): drawn bigger so that they can be seen
    colors = {"cube_0": "#f9a825", "cube_1": "#ad1457"}
    for name, (x, y) in start_cubes.items():
        c = colors.get(name, "#555")
        ax.add_patch(plt.Rectangle((x - half, y - half), 2 * half, 2 * half, fill=False, ec=c, lw=1.6))
        ax.text(x, y + 0.25, f"{name} (start)", ha="center", fontsize=7, color=c)
    for name, (x, y) in final_cubes.items():
        c = colors.get(name, "#555")
        ax.add_patch(plt.Rectangle((x - half, y - half), 2 * half, 2 * half, fc=c, ec="k", lw=0.8))
        ax.text(x, y - 0.38, f"{name} (end)", ha="center", fontsize=7, color=c)

    def where(t: float):
        if not trace:
            return None
        return min(trace, key=lambda p: abs(p[0] - t))[1:]

    marks = {"emergency": ("X", "#d50000", "emergency stop"), "gate": ("s", "#ff6d00", "refused by the gate"),
             "recover": ("^", "#6a1b9a", "recovery")}
    used = set()
    for e in events or []:
        kind = e.get("kind")
        msg = e.get("msg", "")
        if kind == "result" and "FAILED" in msg:
            key = ("v", "#c62828", "failed step")
        elif kind == "gate" and "DENIED" in msg:
            key = marks["gate"]
        elif kind in ("emergency", "recover"):
            key = marks[kind]
        else:
            continue
        pos = where(e.get("t", 0.0))
        if pos is None:
            continue
        ax.plot(pos[0], pos[1], key[0], color=key[1], ms=9, label=None if key[2] in used else key[2])
        used.add(key[2])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bar_chart(path: str, labels: Sequence[str], series: dict, title: str, ylabel: str, ylim=None) -> None:
    plt = _plt()
    import numpy as np
    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(series)
    width = 0.8 / max(1, n)
    x = np.arange(len(labels))
    for i, (name, values) in enumerate(series.items()):
        bars = ax.bar(x + (i - (n - 1) / 2) * width, values, width, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def line_chart(path: str, xs: Sequence[float], series: dict, title: str, xlabel: str, ylabel: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
