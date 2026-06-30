#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 A + 阶段 B：先相机预览 → 拍一张落表 → 实时 YOLO 可视化。

离 survey 位时进入编号追踪模式：24 孔编号/坐标以快照表为准，FOV 内可见孔用 live 更新。
"""

from __future__ import annotations

import time
from typing import Callable, Literal, Optional

import cv2

from pipeline.context import BoardSnapshot, SlotId
from pipeline.slot_tracker import merge_snapshot_live, visible_in_frame
from ui.overlays import (
    destroy_windows,
    render_detect_preview,
    render_snapshot,
    setup_windows,
    show_frame,
    show_mapping_table,
)

PreviewAction = Literal["quit", "scan", "capture"]


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
        self._move_status: Optional[str] = None

    def _split_x(self) -> float:
        sm = getattr(self.detector, "slot_mapper", None)
        if sm is not None:
            return float(sm.split_x)
        return 320.0

    def _key_pressed(self, key: int, *chars: str) -> bool:
        if key == 255:
            return False
        if key in (13, 10) and "\n" in chars:
            return True
        ch = chr(key) if 0 < key < 128 else ""
        return ch.lower() in {c.lower() for c in chars if len(c) == 1}

    def _on_move_done(self, result) -> None:
        if result.success:
            self._move_status = "✓ 已到达 survey 拍照位"
            print("[Move] ✓ 已到达 survey 拍照位")
        else:
            self._move_status = f"✗ 移臂失败: {result.message}"
            print(f"[Move] ✗ {result.message}")

    def _start_move_survey(self) -> None:
        if self.survey_motion is None:
            print("[Move] 未配置 survey_motion")
            return
        if self.survey_motion.is_moving:
            self._move_status = "移臂进行中..."
            return
        self._move_status = "移臂中 → survey 拍照位..."
        print("[Move] 后台移臂 → survey 位")
        self.survey_motion.goto_survey_async(on_done=self._on_move_done)

    def _detect_frame(self, fp, snapshot_id: str, fps: float = 0.0):
        """YOLO 推理 + 24 孔 live 表（预览/实时共用）。"""
        T = self.arm.get_T_base_end()
        raw = self.detector.detect_raw(fp)
        live = None
        if T is not None:
            live = self.detector.build_live_frame(fp, T, snapshot_id, fps=fps)
        return raw, live

    def preview_loop(self, *, allow_scan: bool = True) -> PreviewAction:
        """阶段 0：相机 + YOLO。返回 quit / scan(先移臂) / capture(当前位拍)。"""
        cfg = self.cfg
        if cfg is None or not cfg.preview_on_start:
            return "scan"

        setup_windows(cfg)
        split_x = self._split_x()
        print("[Preview] 相机预览 + YOLO")
        print("  Enter/s → 拍照落表（已在拍照位则直接拍）")
        print("  m       → 移到 survey 拍照位")
        print("  q       → 退出")

        last_t = time.time()
        frame_count = 0
        fps = 0.0

        while True:
            t0 = time.time()
            fp = self.camera.grab()
            if fp is None:
                time.sleep(0.02)
                continue

            frame_count += 1
            if t0 - last_t >= 1.0:
                fps = frame_count / (t0 - last_t)
                frame_count = 0
                last_t = t0

            status = self._move_status
            if self.survey_motion and self.survey_motion.is_moving:
                status = status or "移臂中..."

            raw, live = ([], None)
            if cfg.preview_detect:
                raw, live = self._detect_frame(fp, "preview", fps=fps)

            if cfg.preview_detect and (raw or live):
                slots = live.slots if live else []
                color_vis, depth_vis = render_detect_preview(
                    fp.color, fp.depth, cfg, raw,
                    live_slots=slots,
                    split_x=split_x,
                    fps=fps,
                    status_text=status,
                )
                key = show_frame(cfg, color_vis, depth_vis)
            else:
                from ui.overlays import render_preview
                panel = render_preview(fp.color, fp.depth, cfg)
                if status:
                    from ui.overlays import draw_status_banner
                    draw_status_banner(panel[:, : fp.color.shape[1]], status)
                cv2.imshow(cfg.survey_window, panel)
                key = cv2.waitKey(30) & 0xFF

            if self._key_pressed(key, "q"):
                return "quit"
            if allow_scan and self._key_pressed(key, "s", "\n"):
                if self._is_at_survey():
                    return "capture"
                return "scan"
            if self._key_pressed(key, "m"):
                self._start_move_survey()

    def run(self, *, no_rescan: bool = False, skip_preview: bool = False) -> bool:
        print("[Live] 启动 hand survey 实时感知")
        print("  键: s 回拍照位重拍 | m 移臂 | t transfer | q 退出")

        move_arm_for_scan = True
        if not skip_preview:
            action = self.preview_loop()
            if action == "quit":
                destroy_windows(self.cfg)
                return False
            move_arm_for_scan = action == "scan"

        snap: Optional[BoardSnapshot] = None
        if no_rescan:
            snap = self.store.latest()
            if snap is None:
                print("[Live] 无 board_latest，强制执行阶段 A")
                no_rescan = False

        if not no_rescan:
            snap = self.survey.run(
                move_arm=move_arm_for_scan,
                save=True,
                show_ui=True,
            )
            if snap is None:
                destroy_windows(self.cfg)
                return False

        assert snap is not None
        setup_windows(self.cfg)
        show_mapping_table(self.cfg, snap.slots, snap.snapshot_id)
        self._live_loop(snap)
        destroy_windows(self.cfg)
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
        split_x = self._split_x()

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

            raw, live_raw = self._detect_frame(fp, snap.snapshot_id, fps=fps)
            assert live_raw is not None

            if at_survey:
                merged = live_raw
                overlay = None
                show_panel = cfg.show_slot_panel
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
                split_x=split_x,
                stage="live",
                raw_dets=raw,
            )
            show_mapping_table(cfg, merged.slots, snap.snapshot_id)

            key = show_frame(cfg, color_vis, depth_vis)
            if self._key_pressed(key, "q"):
                break
            if self._key_pressed(key, "m"):
                self._start_move_survey()
            if self._key_pressed(key, "s"):
                print("[Live] 回拍照位并重拍落表...")
                if not at_survey:
                    self._start_move_survey()
                    if self.survey_motion:
                        while self.survey_motion.is_moving:
                            time.sleep(0.05)
                new_snap = self.survey.run(
                    move_arm=not self._is_at_survey(),
                    save=True,
                    show_ui=True,
                )
                if new_snap:
                    snap = new_snap
                    show_mapping_table(cfg, snap.slots, snap.snapshot_id)
            if self._key_pressed(key, "t"):
                self._try_transfer(at_survey)

            sleep_t = interval - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

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
