#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drivers/camera.py —— Orbbec Gemini 336L 单路相机驱动

参数固定（与 tube_detector 已验证配置一致）：
    彩色：640×480 BGR @ 60fps
    深度：HW D2C 对齐，d2c_list[1] profile
    对齐：OBAlignMode.HW_MODE
    深度单位：uint16 × scale / 1000 → float32（米）

后台线程持续采图，grab() 返回最新帧（daemon 线程）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ── FramePacket DTO ───────────────────────────────────────────────────────────

@dataclass
class FramePacket:
    """单次采图结果。"""
    color:      np.ndarray        # uint8 BGR H×W×3
    depth:      np.ndarray        # float32 H×W 米
    K:          np.ndarray        # 3×3 相机内参
    dist:       np.ndarray        # 畸变系数 [k1,k2,p1,p2,k3]
    timestamp:  float             # time.time()
    camera_id:  str               # "head" | "hand"


# ── pyorbbecsdk import ────────────────────────────────────────────────────────

def _import_orbbec():
    try:
        from pyorbbecsdk import (  # type: ignore
            Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
        )
        return Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
    except ImportError as e:
        raise ImportError(
            "请安装 pyorbbecsdk: pip install pyorbbecsdk"
        ) from e


def _frame_to_bgr(color_frame) -> Optional[np.ndarray]:
    """彩色帧 → BGR numpy（不依赖 Orbbec examples）。"""
    if color_frame is None:
        return None
    from pyorbbecsdk import OBFormat  # type: ignore
    w, h = color_frame.get_width(), color_frame.get_height()
    fmt  = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    print(f"[Camera] 不支持的彩色格式: {fmt}")
    return None


# ── CameraDriver ──────────────────────────────────────────────────────────────

class CameraDriver:
    """
    单路 Orbbec 相机驱动（后台采图线程）。

    Args:
        camera_id:  "head" | "hand"，用于标识
        serial:     USB 序列号（None=单相机时自动选择）
        width/height: 分辨率（默认 640×480）
    """

    def __init__(
        self,
        camera_id: str = "head",
        serial:    Optional[str] = None,
        width:     int = 640,
        height:    int = 480,
    ):
        self.camera_id  = camera_id
        self.serial     = serial
        self.width      = width
        self.height     = height

        self._pipeline      = None
        self._K:   Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None

        self._lock  = threading.Lock()
        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._ts    = 0.0
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ready = False

    # ── 开启 ──────────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """初始化并启动后台采图线程。"""
        Pipeline, Config, OBSensorType, OBFormat, OBAlignMode = _import_orbbec()
        try:
            self._pipeline = Pipeline()
            config = Config()

            # 彩色流：640×480 BGR 60fps
            color_pl = (
                self._pipeline
                .get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                .get_video_stream_profile(self.width, self.height, OBFormat.BGR, 60)
            )
            if color_pl is None:
                print(f"[{self.camera_id}] 找不到 {self.width}×{self.height} BGR 彩色流")
                return False

            # 深度流：HW D2C 对齐
            d2c_list = self._pipeline.get_d2c_depth_profile_list(
                color_pl, OBAlignMode.HW_MODE
            )
            if not d2c_list:
                print(f"[{self.camera_id}] 无 HW D2C 深度流")
                return False
            depth_pl = d2c_list[1]  # index 1 与 tube_detector 一致

            # 内参
            vsp = color_pl.as_video_stream_profile()
            intr = vsp.get_intrinsic()
            dist = vsp.get_distortion()
            self._K = np.array([
                [intr.fx, 0.0, intr.cx],
                [0.0, intr.fy, intr.cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            self._dist = np.array(
                [dist.k1, dist.k2, dist.p1, dist.p2, dist.k3],
                dtype=np.float64,
            )

            config.enable_stream(color_pl)
            config.enable_stream(depth_pl)
            config.set_align_mode(OBAlignMode.HW_MODE)
            self._pipeline.start(config)

            print(
                f"[{self.camera_id}] 相机就绪  "
                f"fx={intr.fx:.1f} fy={intr.fy:.1f} "
                f"serial={self.serial or 'auto'}"
            )

            self._stop.clear()
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True, name=f"cam-{self.camera_id}"
            )
            self._thread.start()

            # 等待首帧
            for _ in range(30):
                time.sleep(0.05)
                with self._lock:
                    if self._color is not None:
                        self._ready = True
                        break

            return self._ready
        except Exception as e:
            print(f"[{self.camera_id}] open 失败: {e}")
            return False

    def _capture_loop(self) -> None:
        """后台采图线程：持续 wait_for_frames 并更新最新帧。"""
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
                if frames is None:
                    continue
                cf = frames.get_color_frame()
                df = frames.get_depth_frame()
                if cf is None or df is None:
                    continue

                color = _frame_to_bgr(cf)
                if color is None:
                    continue

                depth_u16 = np.frombuffer(
                    df.get_data(), dtype=np.uint16
                ).reshape(df.get_height(), df.get_width())
                depth_m = depth_u16.astype(np.float32) * df.get_depth_scale() / 1000.0

                with self._lock:
                    self._color = color.copy()
                    self._depth = depth_m.copy()
                    self._ts    = time.time()
            except Exception as e:
                print(f"[{self.camera_id}] 采图错误: {e}")
                time.sleep(0.5)

    # ── 接口 ──────────────────────────────────────────────────────────────────

    def grab(self) -> Optional[FramePacket]:
        """返回最新一帧（线程安全）。"""
        with self._lock:
            if self._color is None or self._depth is None:
                return None
            return FramePacket(
                color     = self._color.copy(),
                depth     = self._depth.copy(),
                K         = self._K.copy(),
                dist      = self._dist.copy(),
                timestamp = self._ts,
                camera_id = self.camera_id,
            )

    def get_intrinsics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回 (K 3×3, dist 5)。"""
        return (
            self._K.copy()   if self._K   is not None else None,
            self._dist.copy() if self._dist is not None else None,
        )

    def is_ready(self) -> bool:
        return self._ready

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._ready = False
        print(f"[{self.camera_id}] 相机已关闭")
