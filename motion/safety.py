#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运动安全检查。"""

from __future__ import annotations

from pipeline.context import Point3D
from state.errors import SafetyViolation

WORKSPACE = {"x": (-0.80, 0.80), "y": (-0.80, 0.80), "z": (0.00, 0.80)}
MIN_GRASP_Z = 0.02


class SafetyGuard:
    def __init__(self, workspace: dict | None = None, min_z: float = MIN_GRASP_Z):
        self.ws = workspace or WORKSPACE
        self.min_z = min_z

    def check_target(self, p: Point3D) -> tuple[bool, str]:
        checks = [
            (self.ws["x"][0] <= p.x <= self.ws["x"][1], f"x={p.x:.3f} 超出 {self.ws['x']}"),
            (self.ws["y"][0] <= p.y <= self.ws["y"][1], f"y={p.y:.3f} 超出 {self.ws['y']}"),
            (self.ws["z"][0] <= p.z <= self.ws["z"][1], f"z={p.z:.3f} 超出 {self.ws['z']}"),
            (p.z >= self.min_z, f"z={p.z:.3f} < min_z={self.min_z}"),
        ]
        for ok, reason in checks:
            if not ok:
                return False, reason
        return True, ""

    def assert_safe(self, p: Point3D) -> None:
        ok, reason = self.check_target(p)
        if not ok:
            raise SafetyViolation(f"目标点不安全: {reason}  p={p}")
