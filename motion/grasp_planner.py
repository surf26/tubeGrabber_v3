#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 P_base 生成 pre / touch / lift。"""

from __future__ import annotations

import math
from typing import Optional

from pipeline.context import GraspPlan, Pose6D, Point3D


class GraspPlanner:
    def __init__(self, arm, grasp_cfg=None):
        self.arm = arm
        self.cfg = grasp_cfg

    def plan(self, p: Point3D) -> Optional[GraspPlan]:
        cfg = self.cfg
        pre_z = cfg.pre_offset_z if cfg else 0.08
        touch_z = cfg.touch_offset_z if cfg else 0.002
        lift_z = cfg.lift_offset_z if cfg else 0.12
        g_open = cfg.gripper_open if cfg else 0.85
        g_close = cfg.gripper_close if cfg else 0.0
        tool_z = cfg.tool_offset_z if cfg else 0.0
        rx_v, ry_v, rz_v = math.pi, 0.0, 0.0

        x, y, z_base = p.x, p.y, p.z
        z = z_base + tool_z
        pre = Pose6D(x=x, y=y, z=z + pre_z, rx=rx_v, ry=ry_v, rz=rz_v)
        touch = Pose6D(x=x, y=y, z=z + touch_z, rx=rx_v, ry=ry_v, rz=rz_v)
        lift = Pose6D(x=x, y=y, z=z + lift_z, rx=rx_v, ry=ry_v, rz=rz_v)
        return GraspPlan(pre=pre, touch=touch, lift=lift, gripper_open=g_open, gripper_close=g_close)
