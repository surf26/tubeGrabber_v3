#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 A：survey 初始位拍一张全景 → YOLO → 24 孔位表落盘。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2

from drivers.camera import FramePacket
from drivers.paths import BOARD_SNAPSHOTS_DIR, OUTPUTS_DIR
from pipeline.context import BoardSnapshot
from vision.geometry import median_depth_frames
from ui.overlays import render_snapshot


class SurveyPipeline:
    def __init__(
        self,
        arm,
        camera,
        detector,
        survey_motion,
        board_store,
        survey_cfg=None,
        display_cfg=None,
    ):
        self.arm = arm
        self.camera = camera
        self.detector = detector
        self.survey_motion = survey_motion
        self.store = board_store
        self.survey_cfg = survey_cfg
        self.display_cfg = display_cfg

    def _grab_snapshot_frame(self) -> Optional[FramePacket]:
        n = self.survey_cfg.snapshot_median_frames if self.survey_cfg else 3
        frames: List[FramePacket] = []
        for _ in range(max(1, n)):
            fp = self.camera.grab()
            if fp is not None:
                frames.append(fp)
            time.sleep(0.03)

        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]

        med_depth = median_depth_frames(frames)
        base = frames[-1]
        return FramePacket(
            color=base.color.copy(),
            depth=med_depth,
            K=base.K.copy(),
            dist=base.dist.copy(),
            timestamp=time.time(),
            camera_id=base.camera_id,
        )

    def run(
        self,
        *,
        move_arm: bool = True,
        save: bool = True,
        show_ui: bool = False,
        pause_s: float | None = None,
    ) -> Optional[BoardSnapshot]:
        print("[Survey] === 阶段 A：初始全景快照 ===")

        if move_arm:
            r = self.survey_motion.goto_survey()
            if not r.success:
                print(f"[Survey] ✗ 移臂失败: {r.message}")
                return None
            self.survey_motion.wait_settle()

        fp = self._grab_snapshot_frame()
        if fp is None:
            print("[Survey] ✗ 采图失败")
            return None

        T = self.arm.get_T_base_end()
        if T is None:
            print("[Survey] ✗ FK 读取失败")
            return None

        joints = self.arm.get_joint_angles_deg() or (
            self.survey_cfg.joints_deg if self.survey_cfg else []
        )

        img_name = f"snapshot_{int(time.time())}.jpg"
        img_path = BOARD_SNAPSHOTS_DIR.parent / "snapshots" / img_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), fp.color)

        snap = self.detector.build_snapshot(
            fp, T, list(joints), image_path=str(img_path),
        )

        tubes = sum(1 for s in snap.slots if s.status == "tube")
        empties = sum(1 for s in snap.slots if s.status == "empty")
        print(f"[Survey] ✓ snapshot={snap.snapshot_id}  tube={tubes} empty={empties}")

        if save:
            self.store.save(snap)

        if show_ui and self.display_cfg:
            pause = pause_s if pause_s is not None else self.display_cfg.snapshot_pause_s
            color_vis, depth_vis = render_snapshot(fp.color, fp.depth, snap, self.display_cfg)
            cv2.imshow(self.display_cfg.survey_window, color_vis)
            if self.display_cfg.show_depth_panel:
                cv2.imshow(self.display_cfg.depth_window, depth_vis)
            cv2.waitKey(int(pause * 1000))

        return snap
