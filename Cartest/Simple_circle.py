"""Frenet B-spline trajectory MPC demo on circular track.

Thin wrapper around Simple.run with the ``circle_track`` scenario.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Cartest.Simple import run


def _parse():
    p = argparse.ArgumentParser(description="Circular-track MPC demo")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(steps=args.steps, seed=args.seed, plot=not args.no_plot,
        scenario_name="circle_track")
