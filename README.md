# tubeGrabber v3

单 hand 相机全板扫描 + YOLO 双类（tube/empty）+ 24 孔位表 + 夹取放置。

设计文档：[docs/单Hand相机全板扫描架构设计.md](docs/单Hand相机全板扫描架构设计.md)

## 流程

```
survey 初始位 → 拍一张全景 → 24 孔落表 (BoardSnapshot)
             → 实时 YOLO 预览 (LiveFrame，仅 UI)
             → transfer --from left.A1 --to right.B2 (读表坐标驱动机械臂)
```

## 快速开始

```bash
cd tubeGrabber_v3

# 1. 示教能拍到 24 孔全景的关节角
python main.py record-survey

# 2. 标准流程：落表 + 实时双窗口
python main.py live

# 3. 仅拍一张落表
python main.py scan --hold

# 4. 查看 24 孔表
python main.py show-board

# 5. 转移试管
python main.py transfer --from left.A1 --to right.B2 --init-gripper
```

## CLI

| 命令 | 说明 |
|------|------|
| `live` | 阶段 A 落表 + 阶段 B 实时（`s` 重拍，`t` transfer，`q` 退出） |
| `live --no-rescan` | 跳过阶段 A，直接实时（需已有 `board_latest`） |
| `scan` | 仅阶段 A |
| `transfer --from S --to D` | 扫描 → 夹取 → 放置 → 再扫描 |
| `show-board` | 打印权威表 |
| `record-survey` | 写入 `config/survey.yaml` |
| `check` | 臂 / 相机 / 标定 / YOLO 自检 |

## 目录

```
tubeGrabber_v3/
  main.py              CLI
  container.py         依赖注入
  config/              YAML 参数
  pipeline/            survey · live_survey · transfer
  vision/              hand_survey · slot_mapper · geometry
  motion/              survey · grasp · place · safety
  state/               board_snapshot_store
  ui/                  OpenCV overlays
  assets/calib/        T_cam_end.json（手眼标定）
  assets/model/        last.pt
  data/                board_latest.json · board_snapshots/
```

## 配置要点

- `config/survey.yaml` — survey 俯视关节角（必须 `record-survey`）
- `config/hardware.yaml` — 臂 IP、`hand_serial`
- `config/perception.yaml` — YOLO 模型与阈值
- `assets/calib/T_cam_end.json` — 手眼标定（唯一视觉标定）

## Python API

```python
from container import Container
from pipeline.context import SlotId

c = Container()
c.build(init_gripper=True)

c.live_survey.run()
c.transfer.run(
    SlotId.parse("left.A1"),
    SlotId.parse("right.B2"),
    rescan=True,
)
c.shutdown()
```

## 依赖

- Python 3.10+
- `numpy`, `opencv-python`, `pyyaml`, `scipy`
- `ultralytics` (YOLO)
- `pyorbbecsdk` (Orbbec 相机)
- Realman `RM_API2` / `Robotic_Arm`

## 与 v2 差异

| v2 | v3 |
|----|-----|
| head + hand 双相机 | 仅 hand |
| pick 单孔粗精 | transfer from/to |
| head YOLO + hand OpenCV | hand YOLO 双类 |
| slots.yaml 每孔示教 | survey.yaml 单一位 |
