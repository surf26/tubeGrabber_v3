#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tubeGrabber_v3 依赖注入容器。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from drivers.config_loader import (
    load_calib_paths,
    load_display,
    load_grasp,
    load_hardware,
    load_perception,
    load_place,
    load_rack,
    load_survey,
)
from drivers.paths import PROJECT_ROOT, ensure_dirs, setup_sdk_paths
from vision.geometry import load_calib_hand


class Container:
    def __init__(self):
        ensure_dirs()
        self.hw_cfg = load_hardware()
        self.grasp_cfg = load_grasp()
        self.place_cfg = load_place()
        self.survey_cfg = load_survey()
        self.perc_cfg = load_perception()
        self.display_cfg = load_display()
        self.rack_cfg = load_rack()
        self.calib_paths = load_calib_paths()

        self.T_cam_end: Optional[np.ndarray] = None
        self.arm = None
        self.camera = None
        self.detector = None
        self.survey_motion = None
        self.grasp_planner = None
        self.grasp_executor = None
        self.place_planner = None
        self.place_executor = None
        self.safety = None
        self.board_store = None
        self.survey = None
        self.live_survey = None
        self.transfer = None

    def load_calibration(self) -> bool:
        self.T_cam_end = load_calib_hand(PROJECT_ROOT, self.calib_paths)
        if self.T_cam_end is not None:
            print("[Container] ✓ hand 标定 T_cam_end 已加载")
            return True
        print("[Container] ✗ hand 标定缺失")
        return False

    def build_arm(self, init_pose: bool = False, init_gripper: bool = False):
        setup_sdk_paths()
        from drivers.arm import ArmDriver
        self.arm = ArmDriver(self.hw_cfg, init_pose=init_pose, init_gripper=init_gripper)
        return self.arm

    def build_camera(self) -> bool:
        from drivers.hand_camera import HandCameraManager
        cfg = self.hw_cfg.cameras
        self.camera = HandCameraManager(
            serial=cfg.hand_serial,
            width=cfg.width,
            height=cfg.height,
        )
        return self.camera.open()

    def build_vision(self):
        from vision.hand_survey import HandSurveyDetector
        from vision.slot_mapper import SlotMapper

        mapper = SlotMapper(
            image_width=self.hw_cfg.cameras.width,
            rack_cfg=self.rack_cfg,
        )
        self.detector = HandSurveyDetector(
            T_cam_end=self.T_cam_end,
            perception_cfg=self.perc_cfg,
            slot_mapper=mapper,
            project_root=PROJECT_ROOT,
        )

    def build_motion(self):
        from motion.survey_motion import SurveyMotion
        from motion.grasp_planner import GraspPlanner
        from motion.grasp_executor import GraspExecutor
        from motion.place_planner import PlacePlanner
        from motion.place_executor import PlaceExecutor
        from motion.safety import SafetyGuard

        self.survey_motion = SurveyMotion(self.arm, self.survey_cfg)
        self.grasp_planner = GraspPlanner(self.arm, self.grasp_cfg)
        self.grasp_executor = GraspExecutor(self.arm, self.grasp_cfg)
        self.place_planner = PlacePlanner(self.place_cfg)
        self.place_executor = PlaceExecutor(self.arm, self.place_cfg)
        self.safety = SafetyGuard()

    def build_state(self):
        from state.board_snapshot_store import BoardSnapshotStore
        self.board_store = BoardSnapshotStore()

    def build_pipelines(self, transfer_from=None, transfer_to=None):
        from pipeline.survey import SurveyPipeline
        from pipeline.live_survey import LiveSurveyPipeline
        from pipeline.transfer import TransferPipeline

        self.survey = SurveyPipeline(
            arm=self.arm,
            camera=self.camera,
            detector=self.detector,
            survey_motion=self.survey_motion,
            board_store=self.board_store,
            survey_cfg=self.survey_cfg,
            display_cfg=self.display_cfg,
        )

        self.transfer = TransferPipeline(
            survey_pipeline=self.survey,
            board_store=self.board_store,
            grasp_planner=self.grasp_planner,
            grasp_executor=self.grasp_executor,
            place_planner=self.place_planner,
            place_executor=self.place_executor,
            safety=self.safety,
            survey_motion=self.survey_motion,
        )

        self.live_survey = LiveSurveyPipeline(
            survey_pipeline=self.survey,
            camera=self.camera,
            detector=self.detector,
            arm=self.arm,
            board_store=self.board_store,
            display_cfg=self.display_cfg,
            transfer_fn=self.transfer.run,
            transfer_from=transfer_from,
            transfer_to=transfer_to,
        )

    def build(
        self,
        *,
        init_gripper: bool = False,
        init_pose: bool = False,
        transfer_from=None,
        transfer_to=None,
    ) -> "Container":
        if not self.load_calibration():
            raise RuntimeError("手眼标定未就绪")
        self.build_arm(init_pose=init_pose, init_gripper=init_gripper)
        if not self.build_camera():
            raise RuntimeError("hand 相机开启失败")
        self.build_vision()
        self.build_motion()
        self.build_state()
        self.build_pipelines(transfer_from=transfer_from, transfer_to=transfer_to)
        print("[Container] ✓ tubeGrabber_v3 就绪")
        return self

    def shutdown(self):
        print("[Container] 关闭...")
        if self.camera:
            self.camera.close()
        if self.arm:
            self.arm.disconnect()
        print("[Container] ✓ 已关闭")
