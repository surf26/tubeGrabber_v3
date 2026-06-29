#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单 hand 相机管理器。"""

from __future__ import annotations

from typing import Optional

from drivers.camera import CameraDriver, FramePacket


class HandCameraManager:
    def __init__(self, serial: Optional[str], width: int = 640, height: int = 480):
        self._cam = CameraDriver(
            camera_id="hand",
            serial=serial,
            width=width,
            height=height,
        )

    def open(self) -> bool:
        return self._cam.open()

    def grab(self) -> Optional[FramePacket]:
        return self._cam.grab()

    def close(self) -> None:
        self._cam.close()

    def is_ready(self) -> bool:
        return self._cam.is_ready()
