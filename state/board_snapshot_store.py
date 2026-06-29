#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""24 孔位 BoardSnapshot 持久化。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from pipeline.context import BoardSnapshot, SlotId, SlotRecord
from state.errors import (
    CoordMissingError, SlotEmptyError, SlotOccupiedError, SlotUnknownError,
)
from drivers.paths import BOARD_SNAPSHOTS_DIR, DATA_DIR


class BoardSnapshotStore:
    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or DATA_DIR
        self._snap_dir = self._data_dir / "board_snapshots"
        self._latest_path = self._data_dir / "board_latest.json"
        self._index_path = self._data_dir / "board_latest_id.json"
        self._lock = threading.Lock()
        self._snap_dir.mkdir(parents=True, exist_ok=True)

    def save(self, snap: BoardSnapshot) -> None:
        with self._lock:
            snap_path = self._snap_dir / f"{snap.snapshot_id}.json"
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, ensure_ascii=False, indent=2)
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump({"snapshot_id": snap.snapshot_id, "path": str(snap_path)}, f)
            with open(self._latest_path, "w", encoding="utf-8") as f:
                json.dump(snap.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[BoardStore] ✓ 已保存 snapshot={snap.snapshot_id}  ({len(snap.slots)} 孔)")

    def latest(self) -> Optional[BoardSnapshot]:
        with self._lock:
            if not self._latest_path.is_file():
                return None
            try:
                with open(self._latest_path, "r", encoding="utf-8") as f:
                    return BoardSnapshot.from_dict(json.load(f))
            except Exception as e:
                print(f"[BoardStore] 加载失败: {e}")
                return None

    def get_slot(self, slot: SlotId) -> Optional[SlotRecord]:
        snap = self.latest()
        if snap is None:
            return None
        return snap.get_slot(slot)

    def require_tube(self, slot: SlotId) -> SlotRecord:
        rec = self.get_slot(slot)
        if rec is None:
            raise SlotUnknownError(f"孔位 {slot} 不在表中")
        if rec.status == "unknown":
            raise SlotUnknownError(f"{slot} 状态 unknown，请重新 scan")
        if rec.status != "tube":
            raise SlotEmptyError(f"{slot} 状态为 {rec.status}，无法夹取")
        if rec.point_base is None:
            raise CoordMissingError(f"{slot} 缺少基座坐标")
        return rec

    def require_empty(self, slot: SlotId) -> SlotRecord:
        rec = self.get_slot(slot)
        if rec is None:
            raise SlotUnknownError(f"孔位 {slot} 不在表中")
        if rec.status == "unknown":
            raise SlotUnknownError(f"{slot} 状态 unknown，请重新 scan")
        if rec.status != "empty":
            raise SlotOccupiedError(f"{slot} 状态为 {rec.status}，无法放置")
        if rec.point_base is None:
            raise CoordMissingError(f"{slot} 缺少基座坐标")
        return rec
