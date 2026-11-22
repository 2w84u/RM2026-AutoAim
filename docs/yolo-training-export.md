# YOLO 装甲板模型训练与 ONNX 导出指南

> 目标：训练一个与本仓库 `tasks/auto_aim/yolos/yolo11.cpp` 适配器兼容的检测模型，
> 并导出为可被 OpenCV DNN（CUDA）直接加载的 `.onnx`。

## 1 输出契约（必须满足）

| 项目 | 要求 |
|---|---|
| 输入形状 | `1×3×640×640`（yolo11 适配器；yolov8 适配器是 `416`） |
| 颜色顺序 | **RGB**（代码 `blobFromImage(..., swapRB=true)`） |
| 归一化 | `/255`，即 0~1 的 float32 |
| 输出形状 | `1×(4+nc+8)×8400` |
| 单列内容 | `cx, cy, w, h, cls_0 ... cls_(nc-1), kpt1_x, kpt1_y, ..., kpt4_x, kpt4_y` |
| 坐标空间 | letterbox 后的 640×640 像素空间 |
| 关键点语义 | 装甲板四个**物理角点**（不能是检测框四角） |
| 关键点顺序 | 覆盖 TL/TR/BR/BL 即可；代码会按 y 再按 x 重排 |
| `nc` | 由 YAML 的 `class_num` 指定（旧模型 38，新赛季自定） |

> ⚠️ Ultralytics pose 导出的关键点维度默认是 3（`x, y, visibility`）。
> 必须训练/导出 `kpt_shape: [4, 2]`，否则输出通道变成 `4+nc+12`，与适配器不匹配。

## 2 类别定义（38 类旧模型）

类别 id 必须与 `tasks/auto_aim/armor.hpp` 的 `armor_properties` 顺序完全一致：

| id | 类别 | id | 类别 |
|---|---|---|---|
| 0 | blue sentry (small) | 19 | red outpost (small) |
| 1 | red sentry (small) | 20 | extinguish outpost (small) |
| 2 | extinguish sentry (small) | 21 | blue base (big) |
| 3 | blue one (small) | 22 | red base (big) |
| 4 | red one (small) | 23 | extinguish base (big) |
| 5 | extinguish one (small) | 24 | purple base (big) |
| 6 | blue two (small) | 25 | blue base (small) |
| 7 | red two (small) | 26 | red base (small) |
| 8 | extinguish two (small) | 27 | extinguish base (small) |
| 9 | blue three (small) | 28 | purple base (small) |
| 10 | red three (small) | 29 | blue three (big) |
| 11 | extinguish three (small) | 30 | red three (big) |
| 12 | blue four (small) | 31 | extinguish three (big) |
| 13 | red four (small) | 32 | blue four (big) |
| 14 | extinguish four (small) | 33 | red four (big) |
| 15 | blue five (small) | 34 | extinguish four (big) |
| 16 | red five (small) | 35 | blue five (big) |
| 17 | extinguish five (small) | 36 | red five (big) |
| 18 | blue outpost (small) | 37 | extinguish five (big) |

新赛季若改用简化类别（例如 `best2-sim.onnx` 的 3 类），只需保证：
`nc` 与训练时一致，并在 YAML 写 `class_num: N`；PnP 的几何点位仍需在
`tasks/auto_aim/armor_geometry.hpp` 的 `armor_object_points()` 中替换。

## 3 数据集准备

### 3.1 目录结构（YOLO pose 格式）

```
datasets/armor/
├── data.yaml
├── images/
│   ├── train/  *.jpg
│   └── val/    *.jpg
└── labels/
    ├── train/  *.txt
    └── val/    *.txt
```

标注文件一行一个目标，全部为归一化到 [0,1] 的值：

```
class_id cx cy w h  kx1 ky1  kx2 ky2  kx3 ky3  kx4 ky4
```

- `w/h` 取四个关键点的外接矩形
- 关键点顺序建议 **TL → TR → BR → BL**（代码会重排，但保持一致便于人工检查）
- 若使用 `kpt_shape: [4, 2]`，不要写 visibility 列

### 3.2 标注与采集规范

- 四点必须是装甲板灯条外侧的真实角点，不要用检测框四角代替
- 采集覆盖：1~8 m 距离、不同俯仰角/偏航角、高速旋转、不同曝光与增益、
  场地灯光变化、运动模糊、部分遮挡
- 每个编号/颜色至少数百个样本；建议保留 10% 作为验证集，按比赛视频切分避免同帧泄漏
- 加入负样本（场地灯、反光、观众屏幕、己方装甲板）可显著降低误检

## 4 训练

训练建议在带 NVIDIA GPU 的 PC/服务器上完成，Jetson 只做推理部署。

```bash
pip install ultralytics
```

`datasets/armor/data.yaml`：

```yaml
path: datasets/armor
train: images/train
val: images/val
kpt_shape: [4, 2]     # 4 个关键点，只有 x/y

names:
  0: blue_sentry
  1: red_sentry
  2: extinguish_sentry
  # ... 按第 2 节的顺序补齐到 37
```

开始训练：

```bash
yolo pose train \
  model=yolo11n-pose.pt \
  data=datasets/armor/data.yaml \
  imgsz=640 epochs=300 batch=32 workers=8 \
  project=runs/pose name=armor_yolo11n
```

说明：

- 预训练 pose 模型的关键点是 17 个，与 `[4,2]` 不匹配时 Ultralytics 会自动重建 head，
  属正常现象，backbone 权重仍可迁移
- 若追求小目标召回，可换 `yolo11s-pose.pt`；Orin NX 上 s 模型 FP16 约 15 ms，
  n 模型约 8~10 ms（640 输入，受功耗模式影响）
- 常用超参：`mosaic=1.0 close_mosaic=10 degrees=10 translate=0.1 scale=0.5
  fliplr=0.0`（左右翻转会破坏颜色/编号语义，建议关闭）

## 5 导出 ONNX

```python
from ultralytics import YOLO

m = YOLO("runs/pose/armor_yolo11n/weights/best.pt")
m.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
```

导出后必须校验形状：

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("best.onnx")
out = sess.run(None, {"images": np.zeros((1, 3, 640, 640), np.float32)})
print(out[0].shape)  # 期望 (1, 4+nc+8, 8400)
```

> 若输出是 `(1, 8400, 4+nc+8)`（转置过），说明导出配置不同；本仓库适配器只接受
> `(1, C, N)`，可加 `m.export(..., nms=False)` 或改用适配器的 `parse()`。

## 6 接入本仓库

1. 模型放到 `assets/`，例如 `assets/armor_yolo11n.onnx`
2. 新建配置（可复制 `configs/jetson.yaml`）：

```yaml
yolo_name: yolo11
yolo11_model_path: assets/armor_yolo11n.onnx
classify_model: assets/tiny_resnet.onnx
device: GPU          # GPU -> OpenCV DNN CUDA FP16；CPU -> OpenCV DNN CPU
class_num: 38        # 与训练类别数一致
```

3. 编译（CMake 自动选择已安装的 TensorRT 或 OpenVINO）：

```bash
cmake -S . -B build
cmake --build --preset default
```

4. 验证：

```bash
cmake --build --preset default --target detector_verify
./build/detector_verify path/to/video.avi --config-path=configs/jetson.yaml
```

日志出现 `[infer] OpenCV DNN CUDA FP16 backend enabled` 即表示走 GPU。

## 7 预处理约定（容易踩坑）

仓库推理前的预处理是：

1. 等比缩放原图，使其长边适配 640
2. 缩放后的图像**贴在 640×640 画布的左上角**（右下补黑边，不是居中补边）
3. `blobFromImage(scale=1/255, swapRB=true)` → RGB、0~1、NCHW

训练时的 letterbox 尽量采用相同的「左上角对齐」策略；若使用 Ultralytics 默认的
居中补边，在 4:3 画幅下会有约 80 像素的垂直偏移，表现为框/关键点系统性偏移。

## 8 可选：INT8 量化

- OpenCV DNN 路径使用 **FP16 CUDA**，不需要量化即可获得明显加速
- 若要进一步压缩，可在 x86 机器上用 NNCF 做 PTQ（需要校准集），再用 OpenVINO 运行；
  本仓库当前没有 INT8 的 GPU 执行器（TensorRT 适配器尚未实现）
- 仓库 `assets/` 里的 `.xml/.bin` 已经是 INT8 量化模型，仅供 OpenVINO 使用

## 9 关于 .xml/.bin 反向转 ONNX

**不推荐作为长期方案，但仓库已提供可用的转换工具：**

- `docs/ir-to-onnx.md`：完整流程、算子映射表、数值对齐与排错方法
- `tools/ir2onnx.py`：IR → ONNX 转换脚本（含 `--inspect` 自检、`--fq drop|exact`）
- 转换是**有损**的：IR 内卷积权重本身是 INT8 量化后再反量化的结果，
  只能还原“量化后”的行为；`yolo11.xml` 内为 96 个 `i8` 权重张量（≈2.5 MB）+
  110 个 `FakeQuantize` 层
- 最省事的路线仍是：从原始 `.pt` 重新 `export(format="onnx")`（无损、几分钟完成）
- 只有在 `.pt` 已丢失、且必须上 GPU 时才建议走反向转换

## 10 检查清单

- [ ] `kpt_shape = [4, 2]`，输出为 `(1, 4+nc+8, 8400)`
- [ ] 输入 RGB、`/255`，letterbox 左上角对齐
- [ ] 类别 id 与 `armor_properties` 一致（旧模型 38 类）
- [ ] 关键点为物理角点，非检测框角点
- [ ] `class_num` 与训练一致
- [ ] 用 `detector_video_test` 回放比赛视频，检查框/关键点无系统偏移
- [ ] 在实机上测 PnP 重投影误差与端到端帧率
