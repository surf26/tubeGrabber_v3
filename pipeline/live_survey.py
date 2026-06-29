#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 A + 阶段 B：先拍一张落表，再实时 YOLO 可视化。
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import cv2

from pipeline.context import BoardSnapshot, SlotId
from ui.overlays import render_snapshot


class LiveSurveyPipeline:
    def __init__(
        self,
        survey_pipeline,
        camera,
        detector,
        arm,
        board_store,
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
        self.cfg = display_cfg
        self.transfer_fn = transfer_fn
        self.transfer_from = transfer_from
        self.transfer_to = transfer_to

    def run(self, *, no_rescan: bool = False) -> bool:
        print("[Live] 启动 hand survey 实时感知")
        print("  键: s 重拍落表 | t transfer | q 退出")

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

    def _live_loop(self, snap: BoardSnapshot) -> None:
        cfg = self.cfg
        interval = 1.0 / max(cfg.refresh_hz, 1) if cfg else 1.0 / 30
        last_t = time.time()
        frame_count = 0
        fps = 0.0

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

            live = self.detector.build_live_frame(fp, T, snap.snapshot_id, fps=fps)
            color_vis, depth_vis = render_snapshot(
                fp.color, fp.depth, snap, cfg, live_frame=live,
            )

            cv2.imshow(cfg.survey_window, color_vis)
            if cfg.show_depth_panel:
                cv2.imshow(cfg.depth_window, depth_vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                print("[Live] 重拍落表...")
                new_snap = self.survey.run(move_arm=False, save=True, show_ui=False)
                if new_snap:
                    snap = new_snap
            if key == ord("t"):
                self._try_transfer()

            sleep_t = interval - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

        cv2.destroyAllWindows()

    def _try_transfer(self) -> None:
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
            rescan=False,
        )
