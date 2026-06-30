#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""survey 初始位运动。"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from drivers.arm import MotionResult


class SurveyMotion:
    def __init__(self, arm, survey_cfg=None):
        self.arm = arm
        self.cfg = survey_cfg
        self._move_thread: Optional[threading.Thread] = None
        self._moving = False
        self._last_result: Optional[MotionResult] = None

    @property
    def is_moving(self) -> bool:
        return self._moving

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

    def goto_survey_async(self, on_done: Optional[Callable[[MotionResult], None]] = None) -> bool:
        """后台移臂，预览循环可继续刷新画面。"""
        if self._moving:
            print("[SurveyMotion] 移臂进行中，请稍候")
            return False

        def worker():
            self._moving = True
            try:
                r = self.goto_survey()
                if r.success:
                    self.wait_settle()
                self._last_result = r
                if on_done:
                    on_done(r)
            finally:
                self._moving = False

        self._move_thread = threading.Thread(target=worker, daemon=True)
        self._move_thread.start()
        return True

    def wait_settle(self, seconds: float | None = None) -> None:
        s = seconds if seconds is not None else (self.cfg.settle_s if self.cfg else 0.5)
        time.sleep(s)

    def is_at_survey(self, tol_deg: float = 2.0) -> bool:
        """当前关节角是否仍在 survey 全景位（容差 tol_deg 度）。"""
        target = self.cfg.joints_deg if self.cfg else None
        if not target:
            return False
        current = self.arm.get_joint_angles_deg()
        if not current or len(current) < 6:
            return False
        return all(abs(float(a) - float(b)) <= tol_deg for a, b in zip(current, target))
