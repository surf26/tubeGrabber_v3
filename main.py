#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tubeGrabber_v3 CLI

命令：
    live [--no-rescan] [--from SLOT] [--to SLOT]
    scan [--hold]
    preview          仅相机预览（不移臂）
    transfer --from left.A1 --to right.B2
    show-board
    record-survey
    check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def cmd_live(args, c):
    return c.live_survey.run(no_rescan=args.no_rescan)


def cmd_preview(_args, c):
    from ui.overlays import destroy_windows
    ok = c.live_survey.preview_loop()
    destroy_windows(c.display_cfg)
    return ok


def cmd_scan(args, c):
    from ui.overlays import destroy_windows

    if c.display_cfg.preview_on_start:
        if not c.live_survey.preview_loop():
            destroy_windows(c.display_cfg)
            return False

    snap = c.survey.run(move_arm=True, save=True, show_ui=True, pause_s=0)
    if snap and args.hold:
        import cv2
        print("[scan] 按任意键关闭窗口...")
        cv2.waitKey(0)
    destroy_windows(c.display_cfg)
    return snap is not None


def cmd_transfer(args, c):
    from pipeline.context import SlotId
    return c.transfer.run(
        from_slot=SlotId.parse(args.transfer_from),
        to_slot=SlotId.parse(args.transfer_to),
        rescan=not args.no_rescan,
        rescan_after=not args.no_rescan_after,
        init_gripper=args.init_gripper,
    )


def cmd_show_board(_args, c):
    snap = c.board_store.latest()
    if snap is None:
        print("无 board_latest，请先运行 scan 或 live")
        return False
    print(f"snapshot_id={snap.snapshot_id}  time={snap.timestamp:.0f}")
    print(f"{'slot':<12} {'status':<8} {'conf':>6}  {'x':>8} {'y':>8} {'z':>8}")
    print("-" * 60)
    for rec in snap.slots:
        pb = rec.point_base
        xs = f"{pb.x:.4f}" if pb else "—"
        ys = f"{pb.y:.4f}" if pb else "—"
        zs = f"{pb.z:.4f}" if pb else "—"
        print(f"{str(rec.slot_id):<12} {rec.status:<8} {rec.confidence:>6.2f}  {xs:>8} {ys:>8} {zs:>8}")
    return True


def cmd_record_survey(_args, c):
    import yaml
    from drivers.config_loader import CONFIG_DIR

    joints = c.arm.get_joint_angles_deg()
    if joints is None:
        print("[record-survey] 无法读取关节角")
        return False

    path = CONFIG_DIR / "survey.yaml"
    data = {"survey": {
        "joints_deg": [round(j, 4) for j in joints],
        "settle_s": c.survey_cfg.settle_s,
        "speed": c.survey_cfg.speed,
        "snapshot_median_frames": c.survey_cfg.snapshot_median_frames,
    }}
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"[record-survey] ✓ 已写入 {path}")
    print(f"  joints_deg={data['survey']['joints_deg']}")
    return True


def cmd_check(_args, c):
    ok = True
    ok = c.arm.is_connected() and ok
    print(f"  机械臂: {'OK' if c.arm.is_connected() else 'FAIL'}")
    fp = c.camera.grab()
    print(f"  hand 相机: {'OK' if fp else 'FAIL'}")
    ok = fp is not None and ok
    calib_ok = c.T_cam_end is not None
    print(f"  T_cam_end: {'OK' if calib_ok else 'FAIL'}")
    try:
        c.detector._load_yolo()
        print("  YOLO 模型: OK")
    except Exception as e:
        print(f"  YOLO 模型: FAIL ({e})")
        ok = False
    survey_ok = c.survey_cfg.joints_deg is not None
    print(f"  survey 位: {'OK' if survey_ok else 'MISSING (record-survey)'}")
    return ok and calib_ok


def parse_args():
    p = argparse.ArgumentParser(prog="tubeGrabber_v3")
    sub = p.add_subparsers(dest="command")

    pl = sub.add_parser("live", help="阶段A落表 + 阶段B实时")
    pl.add_argument("--no-rescan", action="store_true")
    pl.add_argument("--from", dest="transfer_from", type=str, default=None)
    pl.add_argument("--to", dest="transfer_to", type=str, default=None)

    ps = sub.add_parser("scan", help="仅阶段A拍一张落表")
    ps.add_argument("--hold", action="store_true", help="定格窗口直到按键")

    pt = sub.add_parser("transfer", help="夹取并放置")
    pt.add_argument("--from", dest="transfer_from", type=str, required=True)
    pt.add_argument("--to", dest="transfer_to", type=str, required=True)
    pt.add_argument("--no-rescan", action="store_true")
    pt.add_argument("--no-rescan-after", action="store_true")
    pt.add_argument("--init-gripper", action="store_true")

    sub.add_parser("preview", help="仅相机预览，不移臂")
    sub.add_parser("show-board", help="打印 24 孔位表")
    sub.add_parser("record-survey", help="示教 survey 关节角")
    sub.add_parser("check", help="硬件自检")

    return p.parse_args()


def main():
    args = parse_args()
    if not args.command:
        print("子命令: live | scan | preview | transfer | show-board | record-survey | check")
        sys.exit(1)

    from container import Container
    c = Container()

    init_gripper = getattr(args, "init_gripper", False)

    try:
        if args.command == "check":
            c.load_calibration()
            c.build_arm()
            c.build_camera()
            c.build_vision()
            sys.exit(0 if cmd_check(args, c) else 1)

        if args.command == "show-board":
            c.build_state()
            sys.exit(0 if cmd_show_board(args, c) else 1)

        if args.command == "record-survey":
            c.load_calibration()
            c.build_arm()
            sys.exit(0 if cmd_record_survey(args, c) else 1)

        if args.command in ("live", "scan", "preview", "transfer", "record-survey"):
            from_slot = None
            to_slot = None
            if getattr(args, "transfer_from", None):
                from pipeline.context import SlotId
                from_slot = SlotId.parse(args.transfer_from)
            if getattr(args, "transfer_to", None):
                from pipeline.context import SlotId
                to_slot = SlotId.parse(args.transfer_to)
            c.build(
                init_gripper=init_gripper,
                transfer_from=from_slot,
                transfer_to=to_slot,
            )

        handlers = {
            "live": cmd_live,
            "scan": cmd_scan,
            "preview": cmd_preview,
            "transfer": cmd_transfer,
        }
        ok = handlers[args.command](args, c)
        sys.exit(0 if ok else 1)

    except KeyboardInterrupt:
        print("\n[main] 中断")
        sys.exit(1)
    finally:
        c.shutdown()


if __name__ == "__main__":
    main()
