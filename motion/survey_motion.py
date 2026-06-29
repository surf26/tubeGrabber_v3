#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""survey 初始位运动。"""

from __future__ import annotations

import time

from drivers.arm import MotionResult


class SurveyMotion:
    def __init__(self, arm, survey_cfg=None):
        self.arm = arm
        self.cfg = survey_cfg

    def goto_survey(self) -> MotionResult:
        joints = self.cfg.joints_deg if self.cfg else None
        speed = self.cfg.speed if self.cfg else 25
        if not joints:
            msg = (
                "survey.yaml 未配置 joints_deg。\n"
                "请运行: python main.py record-survey"
            )
            print(f"[SurveyMotion] {msg}")
            return MotionResult(False, -1, msg)
        print(f"[SurveyMotion] → survey 位  joints={joints}")
        return self.arm.move_joints(joints, speed=speed)

    def wait_settle(self, seconds: float | None = None) -> None:
        s = seconds if seconds is not None else (self.cfg.settle_s if self.cfg else 0.5)
        time.sleep(s)
