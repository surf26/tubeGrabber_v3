#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
编号追踪：survey 快照表 + 当前帧局部 YOLO 合并。

臂离开 survey 位失去全局视野时：
  - 24 孔编号/状态/基座坐标仍以 BoardSnapshot 为准（transfer 用）
  - 当前 FOV 内检测到的孔位用 live 结果覆盖显示
"""

from __future__ import annotations

from typing import List

from pipeline.context import BoardSnapshot, LiveFrame, SlotRecord


def merge_snapshot_live(snapshot: BoardSnapshot, live: LiveFrame) -> LiveFrame:
    """快照为底，live 可见孔位覆盖（编号 slot_id 不变）。"""
    slot_map = {str(r.slot_id): r for r in snapshot.slots}
    for lr in live.slots:
        if lr.status == "unknown" or lr.pixel is None:
            continue
        slot_map[str(lr.slot_id)] = lr

    merged = [slot_map[str(s.slot_id)] for s in snapshot.slots]
    return LiveFrame(
        timestamp=live.timestamp,
        fps=live.fps,
        slots=merged,
        snapshot_id=snapshot.snapshot_id,
    )


def visible_in_frame(slots: List[SlotRecord]) -> List[SlotRecord]:
    """当前帧 YOLO 实际看到的孔（有 pixel）。"""
    return [r for r in slots if r.pixel is not None and r.status != "unknown"]
