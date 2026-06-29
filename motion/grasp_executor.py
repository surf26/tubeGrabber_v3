#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行三点抓取轨迹。"""

from __future__ import annotations

import time

from pipeline.context import GraspPlan
from drivers.arm import MotionResult


class GraspExecutor:
    def __init__(self, arm, grasp_cfg=None):
        self.arm = arm
        self.cfg = grasp_cfg

    def execute(self, plan: GraspPlan) -> MotionResult:
        cfg = self.cfg
        speed_approach = cfg.speed_approach if cfg else 25
        speed_grasp = cfg.speed_grasp if cfg else 12
        speed_lift = cfg.speed_lift if cfg else 25
        settle_s = cfg.settle_after_approach_s if cfg else 0.3

        print(f"[GraspExecutor] 开爪 {plan.gripper_open:.2f}")
        self.arm.set_gripper(plan.gripper_open)
        time.sleep(0.5)

        r = self.arm.move_pose_cartesian(plan.pre.as_list(), speed=speed_approach, linear=False)
        if not r.success:
            return MotionResult(False, r.code, f"pre 失败: {r.message}")
        time.sleep(settle_s)

        r = self.arm.move_pose_cartesian(plan.touch.as_list(), speed=speed_grasp, linear=True)
        if not r.success:
            return MotionResult(False, r.code, f"touch 失败: {r.message}")

        print(f"[GraspExecutor] 闭爪 {plan.gripper_close:.2f}")
        self.arm.set_gripper(plan.gripper_close)
        time.sleep(0.3)

        r = self.arm.move_pose_cartesian(plan.lift.as_list(), speed=speed_lift, linear=True)
        if not r.success:
            return MotionResult(False, r.code, f"lift 失败: {r.message}")

        print("[GraspExecutor] ✓ 夹取完成")
        return MotionResult(True, 0, "OK")
