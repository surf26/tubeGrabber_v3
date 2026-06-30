#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenCV 可视化：检测框、孔位表、合并面板。"""

from __future__ import annotations

from typing import List, Optional, Tuple

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


def draw_rack_split(img: np.ndarray, split_x: float) -> None:
    h, w = img.shape[:2]
    x = int(split_x)
    cv2.line(img, (x, 0), (x, h), (0, 255, 255), 2)
    cv2.putText(img, "left", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "right", (x + 12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def _draw_text_block(
    img: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int],
    font_scale: float,
    color=(0, 220, 255),
    bg_alpha: float = 0.55,
) -> None:
    if not lines:
        return
    x0, y0 = origin
    lh = int(22 * font_scale / 0.5)
    pad = 6
    max_w = max(cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1, cv2.LINE_AA)[0][0] for ln in lines)
    box_h = len(lines) * lh + pad * 2
    box_w = max_w + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)
    for i, ln in enumerate(lines):
        y = y0 + pad + (i + 1) * lh - 4
        cv2.putText(img, ln, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, ln, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def draw_help_bar(img: np.ndarray, cfg: DisplayConfig, stage: str) -> None:
    if not cfg.show_help:
        return
    hints = {
        "preview": "相机预览 | Enter/s=移臂拍照落表 | m=仅移臂 | q=退出",
        "snapshot": "全局映射完成 | Enter=进入实时 | q=退出",
        "live": "s=重拍 | t=transfer | q=退出",
        "tracking": "追踪模式 | s=重拍 | t=transfer | q=退出",
    }
    title = {
        "preview": "阶段0 — 相机预览",
        "snapshot": "阶段A — 全局检测映射",
        "live": "阶段B — 实时检测",
        "tracking": "阶段B — 编号追踪",
    }.get(stage, stage)
    _draw_text_block(
        img,
        [title, hints.get(stage, "")],
        (8, 8),
        max(cfg.font_scale, 0.45),
    )


def draw_slot_record(img: np.ndarray, rec: SlotRecord, cfg: DisplayConfig) -> None:
    color = _color_for_record(rec, cfg)
    if rec.bbox:
        x1, y1, x2, y2 = [int(v) for v in rec.bbox]
        thickness = cfg.line_thickness
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1 if rec.status == "unknown" else thickness)
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
    elif rec.pixel:
        u, v = int(rec.pixel[0]), int(rec.pixel[1])
    else:
        return

    cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 10, 2)
    label = str(rec.slot_id) if cfg.show_slot_id else rec.class_name
    if rec.status != "unknown":
        label = f"{rec.slot_id} {rec.class_name[:1].upper()} {rec.confidence:.2f}"
    ty = max((rec.bbox[1] if rec.bbox else v) - 6, 18)
    cv2.putText(img, label, (u - 48, ty), cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale * 0.85, color, 2, cv2.LINE_AA)


def draw_stats_bar(
    img: np.ndarray,
    snapshot_id: str,
    slots: List[SlotRecord],
    fps: float = 0.0,
    stage: str = "live",
) -> None:
    tubes = sum(1 for s in slots if s.status == "tube")
    empties = sum(1 for s in slots if s.status == "empty")
    unknown = sum(1 for s in slots if s.status == "unknown")
    mapped = tubes + empties
    h = img.shape[0]
    text = f"id={snapshot_id}  mapped={mapped}/24  T={tubes} E={empties} U={unknown}  FPS={fps:.0f}"
    cv2.putText(img, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1, cv2.LINE_AA)


def draw_slot_table_panel(
    img: np.ndarray,
    slots: List[SlotRecord],
    *,
    font_scale: float = 0.38,
) -> None:
    """右侧 24 孔面板（半透明底）。"""
    h, w = img.shape[:2]
    panel_w = 200
    x0 = w - panel_w - 8
    y0 = 52
    line_h = 15
    panel_h = min(24 * line_h + 12, h - y0 - 40)

    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 8), (w - 4, y0 + panel_h), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, "24-SLOT", (x0, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (180, 180, 180), 1, cv2.LINE_AA)

    for i, rec in enumerate(slots):
        y = y0 + 16 + i * line_h
        if y > h - 36:
            break
        color = (140, 140, 140)
        if rec.status == "tube":
            color = (0, 220, 0)
        elif rec.status == "empty":
            color = (0, 160, 255)
        conf = f"{rec.confidence:.2f}" if rec.confidence > 0 else "—"
        cv2.putText(
            img, f"{rec.slot_id} {rec.status[:1].upper()} {conf}",
            (x0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )


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


def label_panel(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    return out


def compose_panels(color: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """左右拼接：彩色 | 深度。"""
    dh, dw = depth.shape[:2]
    ch, cw = color.shape[:2]
    if (dh, dw) != (ch, cw):
        depth = cv2.resize(depth, (cw, ch))
    color = label_panel(color, "COLOR")
    depth = label_panel(depth, "DEPTH")
    return np.hstack([color, depth])


def render_preview(color: np.ndarray, depth: np.ndarray, cfg: DisplayConfig) -> np.ndarray:
    """原始相机画面（无检测框）。"""
    vis = color.copy()
    draw_help_bar(vis, cfg, "preview")
    if cfg.combined_panel and cfg.show_depth_panel:
        dvis = depth_to_colormap(depth)
        return compose_panels(vis, dvis)
    return vis


def render_snapshot(
    color: np.ndarray,
    depth: np.ndarray,
    snap: BoardSnapshot,
    cfg: DisplayConfig,
    live_frame: Optional[LiveFrame] = None,
    overlay_slots: Optional[List[SlotRecord]] = None,
    at_survey: bool = True,
    show_table_panel: Optional[bool] = None,
    split_x: Optional[float] = None,
    stage: str = "live",
) -> Tuple[np.ndarray, np.ndarray]:
    table_slots = live_frame.slots if live_frame else snap.slots
    draw_slots = overlay_slots if overlay_slots is not None else table_slots
    sid = live_frame.snapshot_id if live_frame else snap.snapshot_id
    fps = live_frame.fps if live_frame else 0.0
    panel = show_table_panel if show_table_panel is not None else cfg.show_slot_panel

    color_vis = color.copy()
    if split_x is not None:
        draw_rack_split(color_vis, split_x)

    for rec in draw_slots:
        draw_slot_record(color_vis, rec, cfg)

    stg = "tracking" if (stage == "live" and not at_survey) else stage
    draw_help_bar(color_vis, cfg, stg)
    draw_stats_bar(color_vis, sid, table_slots, fps=fps, stage=stg)
    if panel:
        draw_slot_table_panel(color_vis, table_slots, font_scale=cfg.font_scale * 0.75)

    depth_vis = depth_to_colormap(depth)
    draw_depth_slots(depth_vis, draw_slots, cfg)
    if split_x is not None:
        draw_rack_split(depth_vis, split_x)

    return color_vis, depth_vis


def show_frame(
    cfg: DisplayConfig,
    color_vis: np.ndarray,
    depth_vis: Optional[np.ndarray] = None,
) -> int:
    """显示一帧，返回 waitKey 结果。"""
    if cfg.combined_panel and cfg.show_depth_panel and depth_vis is not None:
        panel = compose_panels(color_vis, depth_vis)
        cv2.imshow(cfg.survey_window, panel)
    else:
        cv2.imshow(cfg.survey_window, color_vis)
        if cfg.show_depth_panel and depth_vis is not None:
            cv2.imshow(cfg.depth_window, depth_vis)
    return cv2.waitKey(1) & 0xFF


def setup_windows(cfg: DisplayConfig) -> None:
    cv2.namedWindow(cfg.survey_window, cv2.WINDOW_NORMAL)
    if not cfg.combined_panel and cfg.show_depth_panel:
        cv2.namedWindow(cfg.depth_window, cv2.WINDOW_NORMAL)


def destroy_windows(cfg: DisplayConfig) -> None:
    cv2.destroyWindow(cfg.survey_window)
    if not cfg.combined_panel and cfg.show_depth_panel:
        cv2.destroyWindow(cfg.depth_window)
