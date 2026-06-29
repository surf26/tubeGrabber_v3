#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tubeGrabber_v3 领域模型 DTO。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SlotId:
    board: str
    row: str
    col: int

    @classmethod
    def parse(cls, s: str) -> "SlotId":
        try:
            board, rest = s.strip().split(".", 1)
            row = rest[0].upper()
            col = int(rest[1:])
            return cls(board=board.lower(), row=row, col=col)
        except Exception as exc:
            raise ValueError(f"无效 SlotId: {s!r}") from exc

    def __str__(self) -> str:
        return f"{self.board}.{self.row}{self.col}"

    def slot_key(self) -> str:
        return f"{self.row}{self.col}"


def all_slot_ids() -> List[SlotId]:
    """24 孔位模板（left/right × A-D × 1-3）。"""
    out: List[SlotId] = []
    for board in ("left", "right"):
        for row in ("A", "B", "C", "D"):
            for col in (1, 2, 3):
                out.append(SlotId(board=board, row=row, col=col))
    return out


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    frame: str = "base"

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "frame": self.frame}

    @classmethod
    def from_dict(cls, d: dict) -> "Point3D":
        return cls(
            x=float(d["x"]), y=float(d["y"]), z=float(d["z"]),
            frame=d.get("frame", "base"),
        )

    def __repr__(self) -> str:
        return f"Point3D({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass
class Pose6D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]


@dataclass
class GraspPlan:
    pre: Pose6D
    touch: Pose6D
    lift: Pose6D
    gripper_open: float = 0.85
    gripper_close: float = 0.0


@dataclass
class PlacePlan:
    pre: Pose6D
    touch: Pose6D
    retract: Pose6D
    gripper_open: float = 0.85
    gripper_close: float = 0.0


@dataclass
class SlotRecord:
    slot_id: SlotId
    status: str = "unknown"
    confidence: float = 0.0
    pixel: Optional[Tuple[float, float]] = None
    bbox: Optional[List[float]] = None
    depth_m: Optional[float] = None
    point_base: Optional[Point3D] = None
    class_name: str = ""

    def to_dict(self) -> dict:
        return {
            "slot": str(self.slot_id),
            "status": self.status,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "pixel": list(self.pixel) if self.pixel else None,
            "bbox": self.bbox,
            "depth_m": self.depth_m,
            "point_base": self.point_base.to_dict() if self.point_base else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SlotRecord":
        pb = d.get("point_base")
        return cls(
            slot_id=SlotId.parse(d["slot"]),
            status=d.get("status", "unknown"),
            confidence=float(d.get("confidence", 0)),
            pixel=tuple(d["pixel"]) if d.get("pixel") else None,
            bbox=d.get("bbox"),
            depth_m=d.get("depth_m"),
            point_base=Point3D.from_dict(pb) if pb else None,
            class_name=d.get("class_name", ""),
        )


@dataclass
class BoardSnapshot:
    snapshot_id: str
    timestamp: float
    survey_joints: List[float]
    T_base_end: List[List[float]]
    slots: List[SlotRecord]
    image_path: Optional[str] = None

    @classmethod
    def new_empty(cls, joints: List[float], T: List[List[float]]) -> "BoardSnapshot":
        return cls(
            snapshot_id=uuid.uuid4().hex[:8],
            timestamp=time.time(),
            survey_joints=list(joints),
            T_base_end=T,
            slots=[
                SlotRecord(slot_id=s, status="unknown", class_name="")
                for s in all_slot_ids()
            ],
        )

    def get_slot(self, slot: SlotId) -> Optional[SlotRecord]:
        key = str(slot)
        for rec in self.slots:
            if str(rec.slot_id) == key:
                return rec
        return None

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "survey_joints": self.survey_joints,
            "T_base_end": self.T_base_end,
            "image_path": self.image_path,
            "slots": [s.to_dict() for s in self.slots],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoardSnapshot":
        return cls(
            snapshot_id=d["snapshot_id"],
            timestamp=float(d["timestamp"]),
            survey_joints=list(d.get("survey_joints", [])),
            T_base_end=d.get("T_base_end", []),
            image_path=d.get("image_path"),
            slots=[SlotRecord.from_dict(s) for s in d.get("slots", [])],
        )


@dataclass
class LiveFrame:
    timestamp: float
    fps: float
    slots: List[SlotRecord]
    snapshot_id: str
