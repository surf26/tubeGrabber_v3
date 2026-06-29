#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度采样、反投影、手眼变换。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline.context import Point3D

DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 2.00
BBOX_SHRINK_RATIO = 0.10
MIN_VALID_DEPTH_PIXELS = 10


def get_bbox_depth(depth: np.ndarray, bbox: tuple | list, shrink: float = BBOX_SHRINK_RATIO) -> Optional[float]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = depth.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    mx = max(1, int((x2 - x1) * shrink))
    my = max(1, int((y2 - y1) * shrink))
    roi = depth[y1 + my : y2 - my, x1 + mx : x2 - mx]
    if roi.size == 0:
        return None
    valid = roi[(roi > DEPTH_MIN_M) & (roi < DEPTH_MAX_M)]
    if len(valid) < MIN_VALID_DEPTH_PIXELS:
        return None
    return float(np.median(valid))


def get_point_depth(depth: np.ndarray, u: float, v: float, r: int = 12) -> Optional[float]:
    h, w = depth.shape[:2]
    uu, vv = int(round(u)), int(round(v))
    x1, x2 = max(0, uu - r), min(w, uu + r + 1)
    y1, y2 = max(0, vv - r), min(h, vv + r + 1)
    roi = depth[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    valid = roi[(roi > DEPTH_MIN_M) & (roi < DEPTH_MAX_M)]
    if len(valid) < MIN_VALID_DEPTH_PIXELS:
        return None
    return float(np.median(valid))


def pixel_to_camera(u: float, v: float, z: float, k: np.ndarray, dist: Optional[np.ndarray] = None) -> Optional[Point3D]:
    if z is None or z <= 0.0:
        return None
    k = np.asarray(k, dtype=np.float64)
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    if dist is not None and np.any(dist != 0):
        pt = np.array([[[u, v]]], dtype=np.float32)
        ud = cv2.undistortPoints(pt, k, dist, P=k)[0, 0]
        u, v = float(ud[0]), float(ud[1])
    xc = (u - cx) * z / fx
    yc = (v - cy) * z / fy
    return Point3D(xc, yc, z, frame="cam")


def transform_hand_to_base(p_cam: Point3D, t_cam_end: np.ndarray, t_base_end: np.ndarray) -> Optional[Point3D]:
    if None in (p_cam, t_cam_end, t_base_end):
        return None
    t_cam_target = np.eye(4, dtype=np.float64)
    t_cam_target[:3, 3] = [p_cam.x, p_cam.y, p_cam.z]
    t_world = t_base_end @ t_cam_end @ t_cam_target
    pb = t_world[:3, 3]
    return Point3D(pb[0], pb[1], pb[2], frame="base")


def pixel_to_base_hand(
    u: float, v: float, z: float,
    k: np.ndarray, dist: Optional[np.ndarray],
    t_cam_end: np.ndarray, t_base_end: np.ndarray,
) -> Optional[Point3D]:
    p_cam = pixel_to_camera(u, v, z, k, dist)
    if p_cam is None:
        return None
    return transform_hand_to_base(p_cam, t_cam_end, t_base_end)


def median_depth_frames(frames: list) -> np.ndarray:
    """多帧深度中位数融合。"""
    if not frames:
        raise ValueError("frames 为空")
    stack = np.stack([f.depth for f in frames], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def load_matrix_json(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        print(f"[geometry] 标定 JSON 不存在: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("matrix") or data.get("T") or data.get("transform")
    if raw is None:
        return None
    mat = np.array(raw, dtype=np.float64)
    if mat.shape != (4, 4):
        return None
    return mat


def load_calib_hand(project_root: Path, calib_cfg) -> Optional[np.ndarray]:
    return load_matrix_json(project_root / calib_cfg.hand)
