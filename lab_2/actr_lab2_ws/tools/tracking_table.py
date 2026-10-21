#!/usr/bin/env python3
"""Table of the path tracking in Gazebo: Nav2's standard controller against the LQG of Lab 2.

    python3 tools/tracking_table.py                         # reads ~/actr_lab2_logs/tracking.jsonl
    python3 tools/tracking_table.py --save                  # ... and writes results/lqg/gazebo.md
    python3 tools/tracking_table.py other/tracking.jsonl

Every line of the file is one mission measured by the node track_monitor (lqg_nav/monitor_node.py).
Plain Python: it runs on your computer, without Docker or ROS.
"""
import json
import os
import statistics
import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def table(rows):
    lines = ["| controller | missions | success | navigate time, s | lateral error RMS, cm | "
             "lateral error max, cm | turn jerk RMS, rad/s^2 |",
             "|---|---|---|---|---|---|---|"]
    for name in sorted({r.get("controller", "unknown") for r in rows}):
        sel = [r for r in rows if r.get("controller", "unknown") == name]
        ok = [r for r in sel if str(r.get("result", "")).upper() == "SUCCESS"]

        def mean(key, subset):
            values = [float(r[key]) for r in subset if key in r]
            return f"{statistics.mean(values):.1f}" if values else "-"
        lines.append(f"| {name} | {len(sel)} | {len(ok)} | {mean('nav_time_s', ok)} | {mean('rms_cte_cm', sel)} | "
                     f"{mean('max_cte_cm', sel)} | {mean('jerk', sel)} |")
    return "\n".join(lines)


GAZEBO_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "lqg", "gazebo.md")


def main(argv):
    save = "--save" in argv
    args = [a for a in argv[1:] if a != "--save"]
    path = args[0] if args else os.path.expanduser("~/actr_lab2_logs/tracking.jsonl")
    if not os.path.isfile(path):
        print(f"no file {path}: run a mission first (the simulation of Lab 2 measures it)")
        return 1
    rows = load(path)
    if not rows:
        print(f"{path} is empty")
        return 1
    text = table(rows)
    print(text)
    if save:
        with open(GAZEBO_MD, "w", encoding="utf-8") as f:
            f.write("# Gazebo: the standard Nav2 controller against the LQG controller\n\n"
                    "Measured by the node track_monitor in my runs of the demo mission (one line per mission in "
                    "~/actr_lab2_logs/tracking.jsonl). Lateral error = distance of the robot to the latest path "
                    "of the Nav2 planner; turn jerk = RMS of d(omega)/dt of /cmd_vel; navigate time = the time of "
                    "the navigate steps (picking and placing are not counted).\n\n" + text + "\n")
        print(f"saved in {os.path.abspath(GAZEBO_MD)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
