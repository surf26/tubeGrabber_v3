#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenCV 绘制（只读 DTO）。"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from drivers.config_loader import DisplayConfig
from pipeline.context import BoardSnapshot, LiveFrame, SlotRecord


def draw_cross(img: np.ndarray, u: int, v: int, color, size: int = 8) -> None:
    cv2.line(img, (u - size, v), (u + size, v), color, 1)
    cv2.line(img, (u, v - size), (u, v + size), color, 1)


def _color_for_record(rec: SlotRecord, cfg: DisplayConfig):
    colors = cfg.colors
    if rec.status == "tube":
        return tuple(colors.get("tube", [0, 255, 0]))
    if rec.status == "empty":
        return tuple(colors.get("empty", [255, 160, 0]))
    return tuple(colors.get("unknown", [0, 0, 255]))


def draw_slot_record(img: np.ndarray, rec: SlotRecord, cfg: DisplayConfig) -> None:
    color = _color_for_record(rec, cfg)
    if rec.bbox:
        x1, y1, x2, y2 = [int(v) for v in rec.bbox]
        thickness = cfg.line_thickness
        if rec.status == "unknown":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
    elif rec.pixel:
        u, v = int(rec.pixel[0]), int(rec.pixel[1])
    else:
        return

    draw_cross(img, u, v, color)
    lines: List[str] = []
    if cfg.show_slot_id:
        lines.append(str(rec.slot_id))
    if rec.class_name:
        lines.append(f"{rec.class_name} conf={rec.confidence:.2f}")
    elif rec.status != "unknown":
        lines.append(f"{rec.status} conf={rec.confidence:.2f}")
    if cfg.show_base_coords and rec.point_base:
        p = rec.point_base
        lines.append(f"base({p.x:.3f},{p.y:.3f},{p.z:.3f})")

    y_text = max(v - 8, 15)
    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (u - 40, y_text + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, color, 1, cv2.LINE_AA,
        )


def draw_board_header(
    img: np.ndarray,
    snapshot_id: str,
    slots: List[SlotRecord],
    fps: float = 0.0,
    live: bool = False,
    tracking: bool = False,
    at_survey: bool = True,
) -> None:
    tubes = sum(1 for s in slots if s.status == "tube")
    empties = sum(1 for s in slots if s.status == "empty")
    unknown = sum(1 for s in slots if s.status == "unknown")
    prefix = f"snapshot={snapshot_id}"
    if tracking and not at_survey:
        text = f"{prefix} | 追踪模式 | 表: tubes={tubes} empty={empties} | FPS={fps:.0f}"
    elif live:
        text = f"{prefix} | FPS={fps:.0f} | live: tubes={tubes} empty={empties} unk={unknown}"
    else:
        text = f"{prefix} | tubes={tubes} empty={empties} unk={unknown}"
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    if tracking and not at_survey:
        warn = "OFF-SURVEY: 全局视野丢失，编号以快照表追踪；仅绘制当前可见孔"
        cv2.putText(img, warn, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)


def draw_slot_table_panel(img: np.ndarray, slots: List[SlotRecord], x: int = 8, y0: int = 60) -> None:
    """右侧/左侧文字面板：24 孔编号 + 状态（追踪模式用）。"""
    line_h = 14
    for i, rec in enumerate(slots):
        y = y0 + i * line_h
        if y > img.shape[0] - 10:
            break
        pb = rec.point_base
        coord = f"({pb.x:.2f},{pb.y:.2f})" if pb else "—"
        text = f"{rec.slot_id}: {rec.status:7s} {coord}"
        color = (180, 180, 180)
        if rec.status == "tube":
            color = (0, 220, 0)
        elif rec.status == "empty":
            color = (0, 160, 255)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


def depth_to_colormap(depth_m: np.ndarray) -> np.ndarray:
    valid = (depth_m > 0.1) & (depth_m < 2.0)
    vis = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        d = depth_m.copy()
        d[~valid] = np.nan
        d_norm = np.nan_to_num((d - 0.1) / 1.9 * 255, nan=0).astype(np.uint8)
        vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
    return vis


def draw_depth_slots(depth_vis: np.ndarray, slots: List[SlotRecord], cfg: DisplayConfig) -> None:
    for rec in slots:
        if not rec.pixel:
            continue
        u, v = int(rec.pixel[0]), int(rec.pixel[1])
        color = _color_for_record(rec, cfg)
        cv2.circle(depth_vis, (u, v), 4, color, -1)
        if rec.depth_m is not None:
            cv2.putText(
                depth_vis, f"Z={rec.depth_m:.3f}m", (u + 6, v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )


def render_snapshot(
    color: np.ndarray,
    depth: np.ndarray,
    snap: BoardSnapshot,
    cfg: DisplayConfig,
    live_frame: Optional[LiveFrame] = None,
    overlay_slots: Optional[List[SlotRecord]] = None,
    at_survey: bool = True,
    show_table_panel: bool = False,
) -> tuple:
    """返回 (color_vis, depth_vis)。

    overlay_slots: 实际画框的孔位；None 则用 live_frame/snap 全部。
    at_survey=False 时只画当前可见孔，避免快照像素错位。
    """
    table_slots = live_frame.slots if live_frame else snap.slots
    draw_slots = overlay_slots if overlay_slots is not None else table_slots
    sid = live_frame.snapshot_id if live_frame else snap.snapshot_id
    fps = live_frame.fps if live_frame else 0.0
    is_live = live_frame is not None
    tracking = show_table_panel or (not at_survey and is_live)

    color_vis = color.copy()
    for rec in draw_slots:
        draw_slot_record(color_vis, rec, cfg)
    draw_board_header(
        color_vis, sid, table_slots, fps=fps, live=is_live,
        tracking=tracking, at_survey=at_survey,
    )
    if show_table_panel:
        draw_slot_table_panel(color_vis, table_slots, x=8, y0=58)

    depth_vis = depth_to_colormap(depth)
    draw_depth_slots(depth_vis, draw_slots, cfg)
    return color_vis, depth_vis
