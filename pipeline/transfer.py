#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试管转移：查 BoardSnapshot → 夹取 → 放置 → 可选再 scan。
"""

from __future__ import annotations

import uuid
from typing import Optional

from pipeline.context import SlotId
from state.errors import TubeGrabberError


class TransferPipeline:
    def __init__(
        self,
        survey_pipeline,
        board_store,
        grasp_planner,
        grasp_executor,
        place_planner,
        place_executor,
        safety,
        survey_motion,
    ):
        self.survey = survey_pipeline
        self.store = board_store
        self.grasp_planner = grasp_planner
        self.grasp_executor = grasp_executor
        self.place_planner = place_planner
        self.place_executor = place_executor
        self.safety = safety
        self.survey_motion = survey_motion

    def run(
        self,
        from_slot: SlotId,
        to_slot: SlotId,
        *,
        rescan: bool = True,
        rescan_after: bool = True,
        init_gripper: bool = False,
    ) -> bool:
        task_id = uuid.uuid4().hex[:8]
        print(f"[Transfer] === {task_id}  {from_slot} → {to_slot} ===")

        if init_gripper and self.grasp_executor.arm:
            self.grasp_executor.arm.init_gripper(
                self.grasp_executor.arm.config.gripper
                if self.grasp_executor.arm.config else None
            )

        try:
            if rescan:
                snap = self.survey.run(move_arm=True, save=True, show_ui=False)
                if snap is None:
                    print("[Transfer] ✗ 扫描失败")
                    return False
            else:
                snap = self.store.latest()
                if snap is None:
                    print("[Transfer] ✗ 无 board_latest，请先 scan")
                    return False

            src = self.store.require_tube(from_slot)
            dst = self.store.require_empty(to_slot)
            print(f"[Transfer] 源 {from_slot}: {src.point_base}")
            print(f"[Transfer] 目标 {to_slot}: {dst.point_base}")

            self.safety.assert_safe(src.point_base)
            self.safety.assert_safe(dst.point_base)

            grasp_plan = self.grasp_planner.plan(src.point_base)
            if grasp_plan is None:
                return False
            r = self.grasp_executor.execute(grasp_plan)
            if not r.success:
                print(f"[Transfer] ✗ 夹取失败: {r.message}")
                return False

            place_plan = self.place_planner.plan(dst.point_base)
            if place_plan is None:
                return False
            r = self.place_executor.execute(place_plan)
            if not r.success:
                print(f"[Transfer] ✗ 放置失败: {r.message}")
                return False

            if rescan_after:
                print("[Transfer] 动作后刷新表...")
                self.survey.run(move_arm=True, save=True, show_ui=False)
            else:
                r = self.survey_motion.goto_survey()
                if r.success:
                    self.survey_motion.wait_settle()

            print(f"[Transfer] ✓ 成功 {from_slot} → {to_slot}")
            return True

        except TubeGrabberError as e:
            print(f"[Transfer] ✗ {e}")
            return False
