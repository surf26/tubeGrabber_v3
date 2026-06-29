#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tubeGrabber_v3 路径与 SDK 定位。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

ASSETS_ROOT = PROJECT_ROOT / "assets"
ASSETS_CALIB = ASSETS_ROOT / "calib"
ASSETS_MODEL = ASSETS_ROOT / "model"

CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
BOARD_SNAPSHOTS_DIR = DATA_DIR / "board_snapshots"


def find_rm_api2_root() -> Path | None:
    candidates: list[Path] = []
    env = os.environ.get("TUBE_RM_API2")
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJECT_ROOT / "third_party" / "RM_API2",
        REPO_ROOT / "tubeGrabber_v2" / "third_party" / "RM_API2",
        REPO_ROOT / "Grabber" / "external" / "RM_API2",
        REPO_ROOT / "RM_API2",
    ]
    for p in candidates:
        if (p / "Python" / "Robotic_Arm").is_dir():
            return p.resolve()
    return None


def setup_sdk_paths() -> Path | None:
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    rm_root = find_rm_api2_root()
    if rm_root is not None:
        py_dir = str(rm_root / "Python")
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)
        _setup_native_lib(rm_root)
    return rm_root


def _setup_native_lib(rm_root: Path) -> None:
    import platform
    machine = platform.machine().lower()
    lib_sub = "linux_aarch64" if ("aarch64" in machine or "arm" in machine) else "linux_x86"
    lib_dir = rm_root / "Python" / "Robotic_Arm" / "libs" / lib_sub
    if not lib_dir.is_dir():
        return
    lib_str = str(lib_dir)
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_str not in ld:
        os.environ["LD_LIBRARY_PATH"] = lib_str + (":" + ld if ld else "")


def ensure_dirs() -> None:
    for d in [LOGS_DIR, OUTPUTS_DIR, DATA_DIR, BOARD_SNAPSHOTS_DIR, ASSETS_CALIB, ASSETS_MODEL]:
        d.mkdir(parents=True, exist_ok=True)
