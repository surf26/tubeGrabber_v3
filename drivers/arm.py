#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
drivers/arm.py —— Realman RM-65B 机械臂驱动

职责：唯一接触 RM_API2 的模块。对上层暴露 ArmDriver 接口，
      不包含任何业务逻辑（无 YOLO / 无 HSV / 无 pipeline 状态）。

线程安全：所有 SDK 调用均在 threading.Lock 内执行。
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from drivers.paths import setup_sdk_paths

# ── SDK import（延迟，确保 path 已设置）──────────────────────────────────────

def _import_sdk():
    """返回 (RoboticArm, rm_modbus_rtu_write_params_t, rm_thread_mode_e)"""
    setup_sdk_paths()
    try:
        from Robotic_Arm.rm_robot_interface import (  # type: ignore
            RoboticArm,
            rm_modbus_rtu_write_params_t,
            rm_thread_mode_e,
        )
        return RoboticArm, rm_modbus_rtu_write_params_t, rm_thread_mode_e
    except ImportError as exc:
        raise ImportError(
            "无法导入 Realman SDK。\n"
            "  方法 1: pip install Robotic_Arm\n"
            "  方法 2: 设置环境变量 TUBE_RM_API2=/path/to/RM_API2\n"
            "  方法 3: 将 RM_API2 放到 tubeGrabber/third_party/RM_API2"
        ) from exc


# ── DTO ──────────────────────────────────────────────────────────────────────

class MotionResult:
    """运动指令结果 DTO。"""
    __slots__ = ("success", "code", "message")

    def __init__(self, success: bool, code: int = 0, message: str = ""):
        self.success = success
        self.code    = code
        self.message = message

    def __repr__(self):
        return f"MotionResult(success={self.success}, code={self.code}, msg='{self.message}')"

    @classmethod
    def ok(cls) -> "MotionResult":
        return cls(True, 0, "OK")

    @classmethod
    def fail(cls, code: int, msg: str = "") -> "MotionResult":
        return cls(False, code, msg or f"SDK error {code}")


# ── ArmDriver ─────────────────────────────────────────────────────────────────

class ArmDriver:
    """
    Realman RM-65B 机械臂驱动。

    Args:
        config:         HardwareConfig（来自 config_loader）
        init_pose:      连接后是否运动到安全位（生产=True，调试/标定=False）
        init_gripper:   是否初始化 RS485 夹爪（约 5 秒）
    """

    def __init__(
        self,
        config=None,
        *,
        init_pose: bool = False,
        init_gripper: bool = False,
    ):
        RoboticArm, self._ModbusParams, rm_thread_mode_e = _import_sdk()
        self.lock = threading.Lock()
        self.config = config
        self.openness = 0.0           # 当前夹爪开度

        # 连接
        if config is not None:
            ip   = config.connections.arm_ip
            port = config.connections.arm_port
        else:
            ip, port = "192.168.1.18", 8080

        self.arm    = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(ip, port)
        print(f"[ArmDriver] 已连接 {ip}:{port}  handle={self.handle.id}")

        if init_pose and config is not None:
            home = getattr(config.arm, "home_pose", None)
            if home:
                self.move_joints(home)

        if init_gripper:
            gc = config.gripper if config else None
            self.init_gripper(gc)

    # ── 状态读取 ──────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """句柄有效则认为已连接。"""
        try:
            return self.handle is not None and self.handle.id > 0
        except Exception:
            return False

    def get_T_base_end(self) -> Optional[np.ndarray]:
        """
        获取末端在基座系下的 4×4 位姿矩阵 T_end_base。

        SDK: rm_get_current_arm_state() → pose[x,y,z,rx,ry,rz] 米+弧度 ZYX
        """
        with self.lock:
            try:
                ret, state = self.arm.rm_get_current_arm_state()
            except Exception as e:
                print(f"[ArmDriver] get_T_base_end 异常: {e}")
                return None

        if ret != 0:
            print(f"[ArmDriver] 获取状态失败 code={ret}")
            return None

        pose = state.get("pose")
        if pose is None or len(pose) < 6:
            print("[ArmDriver] pose 数据异常")
            return None

        x, y, z   = pose[0], pose[1], pose[2]
        rx, ry, rz = pose[3], pose[4], pose[5]

        T = np.eye(4, dtype=np.float64)
        try:
            T[:3, :3] = Rotation.from_euler("ZYX", [rx, ry, rz], degrees=False).as_matrix()
        except Exception:
            # 如果 SDK 约定不同，也支持 XYZ 顺序（备用）
            T[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=False).as_matrix()
        T[:3, 3] = [x, y, z]
        return T

    def get_current_pose6d(self) -> Optional[List[float]]:
        """返回当前末端 Pose6D = [x,y,z,rx,ry,rz]（米+弧度）。"""
        with self.lock:
            try:
                ret, state = self.arm.rm_get_current_arm_state()
            except Exception:
                return None
        if ret != 0:
            return None
        return list(state.get("pose", []))

    def get_joint_angles_deg(self) -> Optional[List[float]]:
        """返回 6 个关节角（度）。用于 record-scan 示教。"""
        with self.lock:
            try:
                code, joints = self.arm.rm_get_joint_degree()
            except Exception:
                return None
        if code != 0:
            return None
        return [float(j) for j in joints]

    # ── 运动指令 ──────────────────────────────────────────────────────────────

    def move_joints(
        self,
        deg: List[float],
        speed: int = 30,
        radius: int = 0,
        wait: bool = True,
    ) -> MotionResult:
        """
        关节空间运动（rm_movej）。

        Args:
            deg:    6 个关节角（度）
            speed:  速度百分比 1~100
            radius: 轨迹圆滑半径
            wait:   True=阻塞直到完成
        """
        block = 1 if wait else 0
        with self.lock:
            try:
                code = self.arm.rm_movej(deg, speed, radius, 0, block)
            except Exception as e:
                return MotionResult.fail(-1, str(e))
        if code == 0:
            return MotionResult.ok()
        return MotionResult.fail(code)

    def move_pose_cartesian(
        self,
        pose6d: List[float],
        speed: int = 25,
        linear: bool = False,
        wait: bool = True,
    ) -> MotionResult:
        """
        笛卡尔运动。

        Args:
            pose6d: [x,y,z,rx,ry,rz] 米+弧度
            speed:  速度百分比
            linear: True=rm_movel（直线，竖直段必须用）；False=rm_movej_p（PTP）
            wait:   阻塞
        """
        block = 1 if wait else 0
        with self.lock:
            try:
                if linear:
                    code = self.arm.rm_movel(pose6d, speed, 0, 0, block)
                else:
                    code = self.arm.rm_movej_p(pose6d, speed, 0, 0, block)
            except Exception as e:
                return MotionResult.fail(-1, str(e))
        if code == 0:
            return MotionResult.ok()
        return MotionResult.fail(code)

    # ── 夹爪 ─────────────────────────────────────────────────────────────────

    def init_gripper(self, gripper_config=None) -> None:
        """初始化 RS485 夹爪（约 5 秒，首次抓取前调用）。"""
        gc = gripper_config or (self.config.gripper if self.config else None)
        if gc is None:
            print("[ArmDriver] 无 GripperConfig，跳过夹爪初始化")
            return

        self.arm.rm_set_tool_voltage(3)
        time.sleep(0.5)
        print("[ArmDriver] 设置 RS485 模式:", self.arm.rm_set_tool_rs485_mode(0, 9600))
        time.sleep(0.2)

        self._write_gripper_reg(36, gc.zero_speed)   # 找零速度
        time.sleep(0.2)
        self._write_gripper_reg(38, gc.init_speed)   # 初始速度
        time.sleep(0.2)
        self._write_gripper_reg(40, gc.run_speed)    # 运行速度
        time.sleep(0.2)
        self._write_gripper_reg(43, 256000)          # 全闭初始化
        time.sleep(5.0)
        self.openness = 0.0
        print("[ArmDriver] 夹爪初始化完成")

    def set_gripper(self, openness: float) -> int:
        """
        设置夹爪开度。

        Args:
            openness: 0.0=全闭，1.0=全开
        Returns:
            SDK 返回码（0=成功）
        """
        openness = float(np.clip(openness, 0.0, 1.0))
        pos = int((1.0 - openness) * 256000)

        gc = self.config.gripper if self.config else None
        run_speed = gc.run_speed if gc else 51200

        with self.lock:
            res = self._write_gripper_reg(43, pos)
            wait_t = abs(self.openness - openness) * 256000 / run_speed
            self.openness = openness

        time.sleep(wait_t)
        return res

    def _write_gripper_reg(self, address: int, value: int) -> int:
        """写 Modbus RTU 寄存器（工具端）。"""
        high = (value >> 16) & 0xFFFF
        low  = value & 0xFFFF
        param = self._ModbusParams(
            device=1, address=address, type=1, num=2, data=[high, low]
        )
        return self.arm.rm_write_modbus_rtu_registers(param)

    def disconnect(self) -> None:
        """释放资源（如 SDK 支持）。"""
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            pass
        print("[ArmDriver] 已断开")
