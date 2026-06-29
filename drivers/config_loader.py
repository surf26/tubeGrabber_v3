#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YAML 配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from drivers.paths import CONFIG_DIR


class ConfigError(Exception):
    pass


@dataclass
class ConnectionsConfig:
    arm_ip: str = "192.168.1.18"
    arm_port: int = 8080


@dataclass
class GripperConfig:
    zero_speed: int = 25600
    init_speed: int = 51200
    run_speed: int = 51200


@dataclass
class ArmCfg:
    home_pose: Optional[List[float]] = None
    startup_init_pose: bool = False
    startup_init_gripper: bool = False


@dataclass
class CamerasConfig:
    hand_serial: Optional[str] = None
    width: int = 640
    height: int = 480
    fixed_exposure: Optional[int] = None


@dataclass
class SdkConfig:
    rm_api2_root: Optional[str] = None


@dataclass
class HardwareConfig:
    connections: ConnectionsConfig = field(default_factory=ConnectionsConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    arm: ArmCfg = field(default_factory=ArmCfg)
    cameras: CamerasConfig = field(default_factory=CamerasConfig)
    sdk: SdkConfig = field(default_factory=SdkConfig)


@dataclass
class GraspConfig:
    pre_offset_z: float = 0.08
    touch_offset_z: float = 0.002
    lift_offset_z: float = 0.12
    gripper_open: float = 0.85
    gripper_close: float = 0.0
    speed_approach: int = 25
    speed_grasp: int = 12
    speed_lift: int = 25
    tool_offset_z: float = 0.0
    settle_after_approach_s: float = 0.3
    approach_mode: str = "vertical"


@dataclass
class PlaceConfig:
    pre_offset_z: float = 0.08
    touch_offset_z: float = 0.005
    retract_offset_z: float = 0.10
    gripper_open: float = 0.85
    gripper_close: float = 0.0
    speed_approach: int = 20
    speed_place: int = 10
    speed_retract: int = 25
    settle_after_approach_s: float = 0.3
    tool_offset_z: float = 0.0


@dataclass
class SurveyConfig:
    joints_deg: Optional[List[float]] = None
    settle_s: float = 0.5
    speed: int = 25
    snapshot_median_frames: int = 3


@dataclass
class CalibPathsConfig:
    hand: str = "assets/calib/T_cam_end.json"


@dataclass
class DisplayConfig:
    survey_window: str = "tubeGrabber v3 — Survey"
    depth_window: str = "tubeGrabber v3 — Depth"
    show_depth_panel: bool = True
    font_scale: float = 0.45
    line_thickness: int = 2
    refresh_hz: int = 30
    show_base_coords: bool = True
    show_slot_id: bool = True
    snapshot_pause_s: float = 1.0
    colors: dict = field(default_factory=lambda: {
        "tube": [0, 255, 0],
        "empty": [255, 160, 0],
        "unknown": [0, 0, 255],
        "low_conf": [0, 255, 255],
    })


@dataclass
class RackConfig:
    cols: int = 3
    rows: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    # 图像下方→上方：左架 D C B A，右架 A B C D
    left_rows: List[str] = field(default_factory=lambda: ["D", "C", "B", "A"])
    right_rows: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    board_split_x: Optional[float] = None
    map_slot_on_detect: bool = True

    def rows_for_board(self, board: str) -> List[str]:
        if board == "left":
            return self.left_rows or self.rows
        return self.right_rows or self.rows


@dataclass
class YoloConfig:
    model: str = "assets/model/survey_best.pt"
    conf: float = 0.45
    iou: float = 0.50
    imgsz: int = 640
    device: str = "cpu"
    classes: List[str] = field(default_factory=lambda: ["tube", "empty"])


@dataclass
class SurveyPerceptionConfig:
    depth_roi_radius_px: int = 12


@dataclass
class PerceptionConfig:
    yolo: YoloConfig = field(default_factory=YoloConfig)
    survey: SurveyPerceptionConfig = field(default_factory=SurveyPerceptionConfig)


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_hardware(path: Path | None = None) -> HardwareConfig:
    p = path or (CONFIG_DIR / "hardware.yaml")
    d = _load_yaml(p)
    conn = d.get("connections", {})
    grip = d.get("gripper", {})
    arm = d.get("arm", {})
    cams = d.get("cameras", {})
    sdk = d.get("sdk", {})
    return HardwareConfig(
        connections=ConnectionsConfig(
            arm_ip=conn.get("arm_ip", "192.168.1.18"),
            arm_port=int(conn.get("arm_port", 8080)),
        ),
        gripper=GripperConfig(
            zero_speed=int(grip.get("zero_speed", 25600)),
            init_speed=int(grip.get("init_speed", 51200)),
            run_speed=int(grip.get("run_speed", 51200)),
        ),
        arm=ArmCfg(
            home_pose=arm.get("home_pose"),
            startup_init_pose=bool(arm.get("startup_init_pose", False)),
            startup_init_gripper=bool(arm.get("startup_init_gripper", False)),
        ),
        cameras=CamerasConfig(
            hand_serial=cams.get("hand_serial"),
            width=int(cams.get("width", 640)),
            height=int(cams.get("height", 480)),
            fixed_exposure=cams.get("fixed_exposure"),
        ),
        sdk=SdkConfig(rm_api2_root=sdk.get("rm_api2_root") or os.environ.get("TUBE_RM_API2")),
    )


def load_grasp(path: Path | None = None) -> GraspConfig:
    p = path or (CONFIG_DIR / "grasp.yaml")
    d = _load_yaml(p).get("grasp", {})
    return GraspConfig(
        pre_offset_z=float(d.get("pre_offset_z", 0.08)),
        touch_offset_z=float(d.get("touch_offset_z", 0.002)),
        lift_offset_z=float(d.get("lift_offset_z", 0.12)),
        gripper_open=float(d.get("gripper_open", 0.85)),
        gripper_close=float(d.get("gripper_close", 0.0)),
        speed_approach=int(d.get("speed_approach", 25)),
        speed_grasp=int(d.get("speed_grasp", 12)),
        speed_lift=int(d.get("speed_lift", 25)),
        tool_offset_z=float(d.get("tool_offset_z", 0.0)),
        settle_after_approach_s=float(d.get("settle_after_approach_s", 0.3)),
        approach_mode=str(d.get("approach_mode", "vertical")),
    )


def load_place(path: Path | None = None) -> PlaceConfig:
    p = path or (CONFIG_DIR / "place.yaml")
    try:
        d = _load_yaml(p).get("place", {})
    except ConfigError:
        d = {}
    return PlaceConfig(
        pre_offset_z=float(d.get("pre_offset_z", 0.08)),
        touch_offset_z=float(d.get("touch_offset_z", 0.005)),
        retract_offset_z=float(d.get("retract_offset_z", 0.10)),
        gripper_open=float(d.get("gripper_open", 0.85)),
        gripper_close=float(d.get("gripper_close", 0.0)),
        speed_approach=int(d.get("speed_approach", 20)),
        speed_place=int(d.get("speed_place", 10)),
        speed_retract=int(d.get("speed_retract", 25)),
        settle_after_approach_s=float(d.get("settle_after_approach_s", 0.3)),
        tool_offset_z=float(d.get("tool_offset_z", 0.0)),
    )


def load_survey(path: Path | None = None) -> SurveyConfig:
    p = path or (CONFIG_DIR / "survey.yaml")
    try:
        d = _load_yaml(p).get("survey", {})
    except ConfigError:
        d = {}
    joints = d.get("joints_deg")
    return SurveyConfig(
        joints_deg=list(joints) if joints else None,
        settle_s=float(d.get("settle_s", 0.5)),
        speed=int(d.get("speed", 25)),
        snapshot_median_frames=int(d.get("snapshot_median_frames", 3)),
    )


def load_calib_paths(path: Path | None = None) -> CalibPathsConfig:
    p = path or (CONFIG_DIR / "calib.yaml")
    try:
        d = _load_yaml(p).get("calib", {})
    except ConfigError:
        d = {}
    return CalibPathsConfig(hand=d.get("hand", "assets/calib/T_cam_end.json"))


def load_display(path: Path | None = None) -> DisplayConfig:
    p = path or (CONFIG_DIR / "display.yaml")
    try:
        d = _load_yaml(p).get("display", {})
    except ConfigError:
        d = {}
    return DisplayConfig(
        survey_window=d.get("survey_window", "tubeGrabber v3 — Survey"),
        depth_window=d.get("depth_window", "tubeGrabber v3 — Depth"),
        show_depth_panel=bool(d.get("show_depth_panel", True)),
        font_scale=float(d.get("font_scale", 0.45)),
        line_thickness=int(d.get("line_thickness", 2)),
        refresh_hz=int(d.get("refresh_hz", 30)),
        show_base_coords=bool(d.get("show_base_coords", True)),
        show_slot_id=bool(d.get("show_slot_id", True)),
        snapshot_pause_s=float(d.get("snapshot_pause_s", 1.0)),
        colors=d.get("colors", DisplayConfig().colors),
    )


def load_rack(path: Path | None = None) -> RackConfig:
    p = path or (CONFIG_DIR / "rack.yaml")
    try:
        d = _load_yaml(p).get("rack", {})
    except ConfigError:
        d = {}
    default_rows = ["A", "B", "C", "D"]
    return RackConfig(
        cols=int(d.get("cols", 3)),
        rows=list(d.get("rows", default_rows)),
        left_rows=list(d.get("left_rows", ["D", "C", "B", "A"])),
        right_rows=list(d.get("right_rows", default_rows)),
        board_split_x=d.get("board_split_x"),
        map_slot_on_detect=bool(d.get("map_slot_on_detect", True)),
    )


def load_perception(path: Path | None = None) -> PerceptionConfig:
    p = path or (CONFIG_DIR / "perception.yaml")
    d = _load_yaml(p)
    yolo_d = d.get("yolo", {})
    surv_d = d.get("survey", {})
    return PerceptionConfig(
        yolo=YoloConfig(
            model=yolo_d.get("model", "assets/model/survey_best.pt"),
            conf=float(yolo_d.get("conf", 0.45)),
            iou=float(yolo_d.get("iou", 0.50)),
            imgsz=int(yolo_d.get("imgsz", 640)),
            device=yolo_d.get("device", "cpu"),
            classes=list(yolo_d.get("classes", ["tube", "empty"])),
        ),
        survey=SurveyPerceptionConfig(
            depth_roi_radius_px=int(surv_d.get("depth_roi_radius_px", 12)),
        ),
    )
