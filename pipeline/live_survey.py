#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 A + 阶段 B：先拍一张落表，再实时 YOLO 可视化。

离 survey 位时进入编号追踪模式：24 孔编号/坐标以快照表为准，FOV 内可见孔用 live 更新。
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import cv2

from pipeline.context import BoardSnapshot, SlotId
from pipeline.slot_tracker import merge_snapshot_live, visible_in_frame
from ui.overlays import render_snapshot


class LiveSurveyPipeline:
    def __init__(
        self,
        survey_pipeline,
        camera,
        detector,
        arm,
        board_store,
        survey_motion=None,
        display_cfg=None,
        transfer_fn: Optional[Callable] = None,
        transfer_from: Optional[SlotId] = None,
        transfer_to: Optional[SlotId] = None,
    ):
        self.survey = survey_pipeline
        self.camera = camera
        self.detector = detector
        self.arm = arm
        self.store = board_store
        self.survey_motion = survey_motion
        self.cfg = display_cfg
        self.transfer_fn = transfer_fn
        self.transfer_from = transfer_from
        self.transfer_to = transfer_to

    def run(self, *, no_rescan: bool = False) -> bool:
        print("[Live] 启动 hand survey 实时感知")
        print("  键: s 重拍落表 | t transfer | q 退出")
        print("  离 survey 位后自动切换编号追踪（表内 slot_id 不变）")

        snap: Optional[BoardSnapshot] = None
        if no_rescan:
            snap = self.store.latest()
            if snap is None:
                print("[Live] 无 board_latest，强制执行阶段 A")
                no_rescan = False

        if not no_rescan:
            snap = self.survey.run(move_arm=True, save=True, show_ui=True)
            if snap is None:
                return False

        assert snap is not None
        self._live_loop(snap)
        return True

    def _is_at_survey(self) -> bool:
        if self.survey_motion is None:
            return True
        return self.survey_motion.is_at_survey()

    def _live_loop(self, snap: BoardSnapshot) -> None:
        cfg = self.cfg
        interval = 1.0 / max(cfg.refresh_hz, 1) if cfg else 1.0 / 30
        last_t = time.time()
        frame_count = 0
        fps = 0.0
        was_at_survey = True

        while True:
            t0 = time.time()
            fp = self.camera.grab()
            if fp is None:
                time.sleep(0.02)
                continue

            T = self.arm.get_T_base_end()
            if T is None:
                time.sleep(0.02)
                continue

            frame_count += 1
            elapsed = t0 - last_t
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_t = t0

            at_survey = self._is_at_survey()
            if at_survey != was_at_survey:
                mode = "全景实时" if at_survey else "编号追踪（离 survey 位）"
                print(f"[Live] 模式切换 → {mode}")
                was_at_survey = at_survey

            live_raw = self.detector.build_live_frame(fp, T, snap.snapshot_id, fps=fps)

            if at_survey:
                merged = live_raw
                overlay = None
                show_panel = False
            else:
                merged = merge_snapshot_live(snap, live_raw)
                overlay = visible_in_frame(live_raw.slots)
                show_panel = True

            color_vis, depth_vis = render_snapshot(
                fp.color, fp.depth, snap, cfg,
                live_frame=merged,
                overlay_slots=overlay,
                at_survey=at_survey,
                show_table_panel=show_panel,
            )

            cv2.imshow(cfg.survey_window, color_vis)
            if cfg.show_depth_panel:
                cv2.imshow(cfg.depth_window, depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                print("[Live] 重拍落表...")
                new_snap = self.survey.run(
                    move_arm=not at_survey,
                    save=True,
                    show_ui=False,
                )
                if new_snap:
                    snap = new_snap
            if key == ord("t"):
                self._try_transfer(at_survey)

            sleep_t = interval - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

        cv2.destroyAllWindows()

    def _try_transfer(self, at_survey: bool) -> None:
        if self.transfer_fn is None:
            print("[Live] 未配置 transfer（请用 CLI: transfer --from ... --to ...）")
            return
        if self.transfer_from is None or self.transfer_to is None:
            print("[Live] 未指定 --from / --to")
            return
        print(f"[Live] transfer {self.transfer_from} → {self.transfer_to}")
        self.transfer_fn(
            from_slot=self.transfer_from,
            to_slot=self.transfer_to,
            rescan=not at_survey,
        )
