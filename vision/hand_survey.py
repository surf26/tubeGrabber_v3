#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hand 相机 YOLO 双类全板检测 + 24 孔位表填充。

协调铁律：BoardSnapshot 坐标来自快照时刻 FK；LiveFrame 仅用于 UI。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from drivers.camera import FramePacket
from pipeline.context import (
    BoardSnapshot, LiveFrame, SlotId, SlotRecord, all_slot_ids,
)
from vision.geometry import get_bbox_depth, get_point_depth, pixel_to_base_hand
from vision.slot_mapper import RawDetection, SlotMapper


class HandSurveyDetector:
    def __init__(
        self,
        T_cam_end: np.ndarray,
        perception_cfg=None,
        slot_mapper: Optional[SlotMapper] = None,
        project_root: Optional[Path] = None,
    ):
        self.T_cam_end = T_cam_end
        self.cfg = perception_cfg
        self.slot_mapper = slot_mapper
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self._yolo = None
        self._class_names: List[str] = ["empty", "tube"]

    def _load_yolo(self):
        if self._yolo is not None:
            return
        from ultralytics import YOLO  # type: ignore

        ycfg = self.cfg.yolo if self.cfg else None
        model_path = ycfg.model if ycfg else "assets/model/survey_best.pt"
        mp = Path(model_path)
        if not mp.is_absolute():
            mp = self.project_root / mp
        if not mp.is_file():
            fallback = self.project_root / "assets/model/best.pt"
            if fallback.is_file():
                mp = fallback
                print(f"[HandSurvey] survey_best.pt 缺失，回退 {fallback.name}")
            else:
                raise FileNotFoundError(f"YOLO 模型不存在: {mp}")

        device = ycfg.device if ycfg else "cpu"
        self._yolo = YOLO(str(mp))
        self._yolo.to(device)
        if ycfg and ycfg.classes:
            self._class_names = list(ycfg.classes)
        print(f"[HandSurvey] YOLO 已加载: {mp.name}  device={device}")

    def _yolo_infer(self, color: np.ndarray) -> List[RawDetection]:
        self._load_yolo()
        ycfg = self.cfg.yolo if self.cfg else None
        conf = ycfg.conf if ycfg else 0.45
        iou = ycfg.iou if ycfg else 0.50
        imgsz = ycfg.imgsz if ycfg else 640

        results = self._yolo.predict(
            source=color, conf=conf, iou=iou, imgsz=imgsz, verbose=False,
        )
        out: List[RawDetection] = []
        if not results:
            return out

        r0 = results[0]
        names = r0.names or {}
        boxes = r0.boxes
        if boxes is None:
            return out

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cid = int(box.cls[0])
            cname = names.get(cid, self._class_names[cid] if cid < len(self._class_names) else str(cid))
            if cname not in ("tube", "empty"):
                cname = "empty" if cid == 0 else "tube"
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            out.append(RawDetection(
                bbox=[x1, y1, x2, y2],
                pixel=(u, v),
                confidence=float(box.conf[0]),
                class_name=cname,
                class_id=cid,
            ))
        return out

    def _depth_for_det(self, det: RawDetection, depth: np.ndarray, radius: int) -> Optional[float]:
        if det.class_name == "tube":
            return get_bbox_depth(depth, det.bbox)
        return get_point_depth(depth, det.pixel[0], det.pixel[1], r=radius)

    def _coords_for_det(
        self, det: RawDetection, fp: FramePacket, T_base_end: np.ndarray, radius: int,
    ) -> Tuple[Optional[float], Optional]:
        z = self._depth_for_det(det, fp.depth, radius)
        if z is None:
            return None, None
        pb = pixel_to_base_hand(
            det.pixel[0], det.pixel[1], z,
            fp.K, fp.dist, self.T_cam_end, T_base_end,
        )
        return z, pb

    def detect_raw(self, fp: FramePacket) -> List[RawDetection]:
        dets = self._yolo_infer(fp.color)
        if self.slot_mapper:
            self.slot_mapper.map_detections(dets)
            dets = self.slot_mapper.resolve_conflicts(dets)
        return dets

    def detect_to_records(
        self,
        fp: FramePacket,
        T_base_end: np.ndarray,
    ) -> List[SlotRecord]:
        radius = self.cfg.survey.depth_roi_radius_px if self.cfg else 12
        records: List[SlotRecord] = []
        for det in self.detect_raw(fp):
            if det.slot_id is None:
                continue
            z, pb = self._coords_for_det(det, fp, T_base_end, radius)
            status = det.class_name if det.class_name in ("tube", "empty") else "unknown"
            records.append(SlotRecord(
                slot_id=det.slot_id,
                status=status,
                confidence=det.confidence,
                pixel=det.pixel,
                bbox=det.bbox,
                depth_m=z,
                point_base=pb,
                class_name=det.class_name,
            ))
        return records

    def fill_24_slots(
        self,
        detections: List[SlotRecord],
        joints: List[float],
        T_base_end: np.ndarray,
        image_path: Optional[str] = None,
    ) -> BoardSnapshot:
        snap = BoardSnapshot.new_empty(joints, T_base_end.tolist())
        snap.image_path = image_path
        slot_map = {str(d.slot_id): d for d in detections if d.slot_id}
        filled: List[SlotRecord] = []
        for template in snap.slots:
            key = str(template.slot_id)
            if key in slot_map:
                filled.append(slot_map[key])
            else:
                filled.append(SlotRecord(
                    slot_id=template.slot_id,
                    status="unknown",
                    class_name="",
                ))
        snap.slots = filled
        return snap

    def build_snapshot(
        self,
        fp: FramePacket,
        T_base_end: np.ndarray,
        joints: List[float],
        image_path: Optional[str] = None,
    ) -> BoardSnapshot:
        dets = self.detect_to_records(fp, T_base_end)
        return self.fill_24_slots(dets, joints, T_base_end, image_path)

    def build_live_frame(
        self,
        fp: FramePacket,
        T_base_end: np.ndarray,
        snapshot_id: str,
        fps: float = 0.0,
    ) -> LiveFrame:
        dets = self.detect_to_records(fp, T_base_end)
        live_slots = self.fill_24_slots(dets, [], T_base_end).slots
        return LiveFrame(
            timestamp=time.time(),
            fps=fps,
            slots=live_slots,
            snapshot_id=snapshot_id,
        )
