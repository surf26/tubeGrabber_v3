#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 P_base 生成放置 pre / touch / retract。"""

from __future__ import annotations

import math
from typing import Optional

from pipeline.context import PlacePlan, Pose6D, Point3D


class PlacePlanner:
    def __init__(self, place_cfg=None):
        self.cfg = place_cfg

    def plan(self, p: Point3D) -> Optional[PlacePlan]:
        cfg = self.cfg
        pre_z = cfg.pre_offset_z if cfg else 0.08
        touch_z = cfg.touch_offset_z if cfg else 0.005
        retract_z = cfg.retract_offset_z if cfg else 0.10
        g_open = cfg.gripper_open if cfg else 0.85
        g_close = cfg.gripper_close if cfg else 0.0
        tool_z = cfg.tool_offset_z if cfg else 0.0
        rx_v, ry_v, rz_v = math.pi, 0.0, 0.0

        x, y, z_base = p.x, p.y, p.z
        z = z_base + tool_z
        pre = Pose6D(x=x, y=y, z=z + pre_z, rx=rx_v, ry=ry_v, rz=rz_v)
        touch = Pose6D(x=x, y=y, z=z + touch_z, rx=rx_v, ry=ry_v, rz=rz_v)
        retract = Pose6D(x=x, y=y, z=z + retract_z, rx=rx_v, ry=ry_v, rz=rz_v)
        return PlacePlan(pre=pre, touch=touch, retract=retract, gripper_open=g_open, gripper_close=g_close)
