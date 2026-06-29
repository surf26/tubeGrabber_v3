#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行放置轨迹。"""

from __future__ import annotations

import time

from pipeline.context import PlacePlan
from drivers.arm import MotionResult


class PlaceExecutor:
    def __init__(self, arm, place_cfg=None):
        self.arm = arm
        self.cfg = place_cfg

    def execute(self, plan: PlacePlan) -> MotionResult:
        cfg = self.cfg
        speed_approach = cfg.speed_approach if cfg else 20
        speed_place = cfg.speed_place if cfg else 10
        speed_retract = cfg.speed_retract if cfg else 25
        settle_s = cfg.settle_after_approach_s if cfg else 0.3

        r = self.arm.move_pose_cartesian(plan.pre.as_list(), speed=speed_approach, linear=False)
        if not r.success:
            return MotionResult(False, r.code, f"place pre 失败: {r.message}")
        time.sleep(settle_s)

        r = self.arm.move_pose_cartesian(plan.touch.as_list(), speed=speed_place, linear=True)
        if not r.success:
            return MotionResult(False, r.code, f"place touch 失败: {r.message}")

        print(f"[PlaceExecutor] 开爪 {plan.gripper_open:.2f}")
        self.arm.set_gripper(plan.gripper_open)
        time.sleep(0.3)

        r = self.arm.move_pose_cartesian(plan.retract.as_list(), speed=speed_retract, linear=True)
        if not r.success:
            return MotionResult(False, r.code, f"place retract 失败: {r.message}")

        print("[PlaceExecutor] ✓ 放置完成")
        return MotionResult(True, 0, "OK")
