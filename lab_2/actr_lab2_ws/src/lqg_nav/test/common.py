"""Shared helpers for the tests of lqg_nav. Run them from src/lqg_nav:

    python3 -m unittest discover -s test -v
"""
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # src/lqg_nav
LAB1 = os.path.join(os.path.dirname(PKG), "actr_lab1")                      # src/actr_lab1 (the agent of Lab 1)
WS = os.path.dirname(os.path.dirname(PKG))                                  # the workspace
for folder in (PKG, LAB1, os.path.join(LAB1, "test")):
    if folder not in sys.path:
        sys.path.insert(0, folder)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()
