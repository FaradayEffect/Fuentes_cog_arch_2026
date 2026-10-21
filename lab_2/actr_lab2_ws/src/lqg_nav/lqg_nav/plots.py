"""The figure of the experiment (matplotlib; optional: without it the tables are still written)."""
from __future__ import annotations

import os
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# one colour per controller, always the same (validated categorical palette: blue, orange, aqua)
COLORS = {"lqg": "#2a78d6", "lqr": "#eb6834", "pure_pursuit": "#1baf7a"}
NAMES = {"lqg": "LQG (LQR + Kalman)", "lqr": "LQR alone", "pure_pursuit": "pure pursuit"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
EXAMPLE = ("roomA-roomB", "amcl", 0)          # the route, noise and seed of the command plot


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, summary: Dict[str, Dict[str, dict]], key: str, title: str, unit: str) -> None:
    noises = list(summary)
    controllers = [c for c in ("pure_pursuit", "lqr", "lqg") if any(c in summary[n] for n in noises)]
    width = 0.26
    for k, c in enumerate(controllers):
        xs = [i + (k - 1) * (width + 0.02) for i in range(len(noises))]
        ys = [summary[n][c][key] for n in noises]
        ax.bar(xs, ys, width=width, color=COLORS[c], label=NAMES[c])
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.1f}" if y < 10 else f"{y:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels([{"none": "no noise", "amcl": "1 cm, 1.1°", "high": "3 cm, 2.9°"}.get(n, n) for n in noises])
    ax.set_xlabel("noise of the measured pose", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(unit, color=INK2, fontsize=9)
    _style(ax)


def write_plots(rows: Sequence[dict], summary: Dict[str, Dict[str, dict]], out: str) -> str:
    from .experiment import load_world, routes, run_one          # the example run, with its commands

    layout, grid = load_world()
    route = next(r for r in routes(layout, grid) if r[0] == EXAMPLE[0])
    fig = plt.figure(figsize=(11, 7.2), facecolor=SURFACE)
    top = fig.add_subplot(2, 1, 1)
    for c in ("lqr", "lqg"):
        run = run_one(c, route, EXAMPLE[1], EXAMPLE[2], grid, keep_track=True)
        ts = [p[0] for p in run["track"]]
        ws = [p[3] for p in run["track"]]
        top.plot(ts, ws, color=COLORS[c], linewidth=1.6 if c == "lqr" else 2.0, label=NAMES[c],
                 solid_capstyle="round", solid_joinstyle="round")
    top.set_title("Turn command on one route (room A to room B) with a noisy pose (1 cm, 1.1°): "
                  "the Kalman filter removes the jitter", loc="left", color=INK, fontsize=11)
    top.set_xlabel("time, s", color=INK2, fontsize=9)
    top.set_ylabel("ω command, rad/s", color=INK2, fontsize=9)
    top.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    _style(top)
    _bars(fig.add_subplot(2, 2, 3), summary, "rms_cte_cm", "Lateral error to the path (RMS)", "cm")
    right = fig.add_subplot(2, 2, 4)
    _bars(right, summary, "jerk", "Turn jerk (RMS of dω/dt): smaller = smoother", "rad/s²")
    right.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.tight_layout(pad=1.2, h_pad=2.0)
    path = os.path.join(out, "compare.png")
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path
