#!/usr/bin/env python3
"""
BattenSpline Quickstart — Minimal 5-Line Usage

Shows the absolute minimum to get a routing decision from BattenSpline.
Run:  python quickstart.py
"""
from batten_spline import BattenSpline

spline = BattenSpline()
spline.learn(embedding=[0.1, -0.2, 0.5, 0.3], quality=0.9)   # teach it a known-good region
print(spline.routing_decision([0.15, -0.15, 0.48, 0.35]))     # -> "LOCAL" (close to known-good)
print(spline.routing_decision([9.0, 9.0, 9.0, 9.0]))          # -> "CLOUD" (unfamiliar territory)
