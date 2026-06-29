#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测 → 槽位编号（hand survey 俯视）

hand 相机视角（与 rack.yaml 一致）：

  左架 left          |  右架 right
  列 3  2  1         |  列 1  2  3
      A  (v小/上)    |      D  (v小/上)
      B              |      C
      C              |      B
      D  (v大/下)    |      A  (v大/下)

行：v 大 → ri=0 → rows[0]（最下行）；v 小 → ri=3 → rows[3]（最上行）
列：左架 u 降序 → 3,2,1；右架 u 升序 → 1,2,3
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from drivers.config_loader import RackConfig
from pipeline.context import SlotId


@dataclass
class RawDetection:
    bbox: List[float]
    pixel: Tuple[float, float]
    confidence: float
    class_name: str
    class_id: int
    slot_id: Optional[SlotId] = None


class SlotMapper:
    def __init__(self, image_width: int = 640, rack_cfg: Optional[RackConfig] = None):
        self.image_width = image_width
        self.rack = rack_cfg or RackConfig()
        split = self.rack.board_split_x
        self.split_x = float(split) if split is not None else image_width / 2.0

    def assign_board(self, u: float) -> str:
        return "left" if u < self.split_x else "right"

    def map_detections(self, dets: List[RawDetection]) -> List[RawDetection]:
        if not dets or not self.rack.map_slot_on_detect:
            return dets

        left_d, right_d = [], []
        for d in dets:
            board = self.assign_board(d.pixel[0])
            (left_d if board == "left" else right_d).append(d)

        self._assign_group(left_d, "left")
        self._assign_group(right_d, "right")
        return dets

    def _assign_group(self, group: List[RawDetection], board: str) -> None:
        slot_map = self._compute_slots(group, board)
        for d in group:
            d.slot_id = slot_map.get(id(d))

    def _compute_slots(self, dets: List[RawDetection], board: str) -> Dict[int, SlotId]:
        rows = self.rack.rows_for_board(board)
        n_cols = self.rack.cols
        pts = [(d.pixel[0], d.pixel[1], d) for d in dets]
        if not pts:
            return {}

        vs = [p[1] for p in pts]
        v_max, v_min = max(vs), min(vs)
        v_span = max(v_max - v_min, 1.0)

        by_row: Dict[int, List] = defaultdict(list)
        for u, v, d in pts:
            ri = int((v_max - v) / v_span * (len(rows) - 0.001))
            ri = min(max(ri, 0), len(rows) - 1)
            by_row[ri].append((u, v, d))

        out: Dict[int, SlotId] = {}
        for ri in sorted(by_row.keys()):
            items = by_row[ri]
            if board == "right":
                items.sort(key=lambda x: x[0])
            else:
                items.sort(key=lambda x: -x[0])
            for ci, (_, _, d) in enumerate(items[:n_cols]):
                out[id(d)] = SlotId(board=board, row=rows[ri], col=ci + 1)
        return out

    def resolve_conflicts(self, dets: List[RawDetection]) -> List[RawDetection]:
        """同一 SlotId 多个检测 → 保留最高 conf。"""
        by_slot: Dict[str, RawDetection] = {}
        orphans: List[RawDetection] = []
        for d in dets:
            if d.slot_id is None:
                orphans.append(d)
                continue
            key = str(d.slot_id)
            if key not in by_slot or d.confidence > by_slot[key].confidence:
                by_slot[key] = d
        return list(by_slot.values()) + [d for d in orphans if d.slot_id is None]
