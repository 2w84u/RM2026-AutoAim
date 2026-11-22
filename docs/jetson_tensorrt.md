# Jetson NX TensorRT 后端（0708.onnx → .engine）

本文覆盖深圳大学 `0708.onnx`（`input [1,3,640,640]` RGB/255，`output [1,25200,22]`）
在 Jetson NX 上的 TensorRT 执行路径：**x86 主机只负责准备代码与模型，ONNX → Engine
的转换和最终运行都在 NX 上完成**。

普通 x86 主机（包括带独立显卡的机器）用 `trtexec` 生成的 Engine 不能保证在 Jetson 上
加载：架构（x86_64 → aarch64）、GPU 计算能力和 TensorRT 版本都不同。本仓库**不在
x86 上生成给 NX 用的 Engine**。

## 代码位置

| 文件 | 作用 |
|---|---|
| `tasks/auto_aim/yolos/yolov5_trt.hpp/.cpp` | `YOLOV5TRT : public YOLOBase`，TensorRT 加载、推理、前后处理 |
| `tasks/auto_aim/yolo.cpp` | 工厂新增 `yolo_name: yolov5_trt` |
| `CMakeLists.txt` | 自动查找并选择 TensorRT 或 OpenVINO |
| `tasks/auto_aim/CMakeLists.txt` | 选中 TensorRT 时编译/链接 TensorRT 后端 |
| `scripts/trt/prepare_model.sh` | x86：下载并校验 `0708.onnx` |
| `scripts/trt/build_engine.sh` | NX：`trtexec` 生成 `.engine`，默认 FP16 |
| `scripts/trt/deploy_nx.sh` | x86：`rsync` 同步源码/模型/脚本到 NX |

推理后端仍由 `yolo_name` 选择，原 `yolov5`（OpenVINO / OpenCV DNN）代码完全保留：

```yaml
yolo_name: yolov5_trt
yolov5_trt_model_path: assets/0708_fp16.engine
```

`Solver` / `Tracker` / `Aimer` 无需改动：`YOLO::detect()` 与返回的 `std::list<Armor>`
语义与 `yolov5` 完全一致（同样的 ROI 偏移、`min_confidence`、`check_name()` /
`check_type()`、NMS、`use_traditional` 角点矫正、`center_norm`）。

## 前后处理（与 OpenVINO 版一致）

- 预处理：BGR 原图按 ROI 裁剪 → 等比缩放到不超过 640×640 → 放在全黑画布的**左上角**
  （不是居中 letterbox，填充值不是 114）→ BGR→RGB → `/255` → NCHW FP32/FP16。
- 后处理：逐候选读取 `[1,25200,22]`；`0..7` 为四角点（原始顺序 `TL, BL, BR, TR`，
  转成 `Armor.points` 的 `TL, TR, BR, BL`）；`8` 为置信度 logit（做 sigmoid）；
  `9..12` 颜色分数 Argmax；`13..21` 数字类别分数 Argmax。
- 输出 dtype 不假设是 float32：加载时查询真实类型，FP16 输出会先转成 float32 再解析。

## 一次性部署流程

### 1. x86：准备模型

```bash
./scripts/trt/prepare_model.sh
# 可选：FORCE=1 重新下载；OUT=... MODEL_URL=... 覆盖默认路径/来源
```

脚本会下载 `assets/0708.onnx`，打印输入输出，并把来源、大小、SHA-256 写入
`assets/0708.onnx.download.txt`（只追踪本次下载的文件，不做模型身份对照）。

### 2. x86：同步到 NX

```bash
./scripts/trt/deploy_nx.sh nvidia@<NX_IP> ~/RM2026-AutoAim --dry-run   # 先看清单
./scripts/trt/deploy_nx.sh nvidia@<NX_IP> ~/RM2026-AutoAim
```

- 默认不覆盖 NX 上已有的 `configs/*.yaml`（使用 `--ignore-existing`），也不同步
  `*.engine`；需要时加 `--overwrite-config` / `--with-engine`。
- 没有 `rsync` 时会打印等价的 `scp` 命令。
- 传输内容：`tasks tools io src sp_msgs calibration tests scripts assets docs`、
  `CMakeLists.txt`、`CMakePresets.json`，以及 `configs/*.yaml`（仅补齐缺失文件）。

### 3. NX：确认环境（不要擅自升级 JetPack）

```bash
head -1 /etc/nv_tegra_release
trtexec --help | head          # 或 dpkg -l | grep -i nvinfer
ls /usr/include/aarch64-linux-gnu/NvInfer.h
ls /usr/lib/aarch64-linux-gnu/libnvinfer.so
```

### 4. NX：构建 Engine（只需一次）

```bash
cd ~/RM2026-AutoAim
./scripts/trt/build_engine.sh                  # assets/0708.onnx -> assets/0708_fp16.engine
./scripts/trt/build_engine.sh --precision fp32 # 需要 FP32 时
./scripts/trt/build_engine.sh --force          # 已存在时强制重建
```

- `trtexec` 默认从 `PATH` 或 `/usr/src/tensorrt/bin/trtexec` 查找，可用
  `--trtexec <path>` 或环境变量 `TRTEXEC` 指定。
- 实际执行的命令与 `trtexec` 日志分别打印到终端和 `assets/0708_fp16.build.log`；
  脚本使用 `--skipInference`，只构建 Engine，不使用 `trtexec` 的随机输入。
- Engine 已存在且没有 `--force` 时直接跳过，**启动视觉程序不会重新转换**。
- 转换失败会打印日志尾部并退出，ONNX 保持不变；不会退回 CPU 推理。
- 用下方视觉程序和实拍图片或相机画面检查 Engine 加载及推理。

### 5. NX：编译

```bash
cmake -S . -B build-trt
cmake --build build-trt --parallel 4 --target standard_mpc auto_aim_debug_mpc detector_verify
```

- CMake 自动查找 OpenVINO 和 TensorRT（`NvInfer.h`、`libnvinfer` 与 CUDA Toolkit）；
  Jetson 上两者都存在时优先选择 TensorRT，其他平台优先选择 OpenVINO。只找到一个时使用
  找到的后端，均不可用时配置报错。配置输出会显示最终选择。
- 选中 TensorRT 时，其它模块（YOLOv8/YOLO11、auto_buff 等）的原有 ONNX 路径仍由
  OpenCV DNN 处理；它们的 `.xml` 模型需要选中 OpenVINO。

### 6. NX：切换配置并运行

`configs/jetson_trt.yaml` 基于本机 `configs/sentry.yaml` 的相机、串口和标定参数，
单独选择 TensorRT 后端，不改变原有配置：

```yaml
yolo_name: yolov5_trt
yolov5_trt_model_path: assets/0708_fp16.engine
```

```bash
# 用已有实拍图片检查（结果保存在指定目录）
./build-trt/detector_verify -c configs/jetson_trt.yaml \
  -o out/detector_verify/trt-0708 /path/to/real-armor-image.png

# 在桌面环境使用相机画面，按 q 退出；不会发送云台控制或开火指令
./build-trt/detector_verify --camera -c configs/jetson_trt.yaml --save=false --show

# 相机、串口和标定匹配时，用 MPC 可视化入口观察；显式关闭控制输出
./build-trt/auto_aim_debug_mpc configs/jetson_trt.yaml --enable-control=false
```

检查相机参数、实际标定及串口与设备一致，并观察四角点、颜色和数字。
`detector_verify` 的标注显示颜色和类别；角点编号 `0,1,2,3` 分别表示左上、右上、
右下、左下。
启用 TensorRT 编译的 `auto_aim_debug_mpc` 在省略配置路径时默认使用
`configs/jetson_trt.yaml`；它的控制开关默认开启，仅观察时请显式传入
`--enable-control=false`。
`standard_mpc` 会与云台通信并使用独立的打符模型；使用前须按设备配置其他模块，
且不能仅凭 `auto_fire: false` 认为云台控制输出已关闭。

## 后续启动

只需运行程序：直接反序列化并加载已有 `.engine`，不再执行 ONNX 转换。

## 常见错误

| 现象 | 原因 / 处理 |
|---|---|
| CMake: `No usable inference backend found` | 未找到可用的 OpenVINO 或完整的 TensorRT/CUDA 开发环境；检查报错中的头文件、库和 CUDA Toolkit 路径 |
| 运行时报 `Cannot deserialize engine ...` | Engine 与当前 TensorRT 版本或 GPU 架构不匹配（例如从别的机器拷贝来的）；在 NX 上重新 `build_engine.sh --force` |
| `Unexpected input shape [...]` | 不是 `0708.onnx` 生成的 Engine（要求 `[1,3,640,640]`）；重新转换 |
| `Unexpected output shape [...]` | 输出需为 `[1,N,22]`（4 角点 + 置信度 + 4 颜色 + 9 数字）；说明用了别的检测头模型 |
| `yolov5_trt_model_path is missing` | 配置里没写该字段，或改的是另一份 yaml |
| `yolov5_trt needs TensorRT` | CMake 没有选择 TensorRT；检查安装并重新配置和编译 |
| `trtexec: command not found` | 用 `--trtexec /usr/src/tensorrt/bin/trtexec` |
| 检测框位置整体偏移 | 检查 `use_roi` / `roi` 是否与相机分辨率匹配（与 OpenVINO 后端同一套配置） |
| 置信度普遍偏低或框错乱 | 确认没有把居中 letterbox 或 114 填充引入预处理；本后端与 OpenVINO 版一样是左上角 + 黑边 |

## 设备差异

Engine 与生成它的设备和 TensorRT 环境有关；更换设备或版本后应在目标 NX 上重新构建。
实际检测效果、角点、颜色、数字和完整相机链路需要结合设备画面人工确认。

2026-11-15 在本机 Jetson Orin NX（L4T 36.4.3、CUDA 12.6、TensorRT 10.3）已生成
`assets/0708_fp16.engine`，并编译上述三个目标。Engine 的外部输入与输出均为 FP32，
形状分别为 `[1,3,640,640]` 和 `[1,25200,22]`；FP16 表示允许内部层使用半精度，
不意味着外部张量一定是 FP16。

仓库本机已有的实拍装甲板图片检测出一个 `red sentry`，四角点顺序为左上、右上、右下、左下。
海康相机已采集并处理 30 帧且正常退出；当时画面较暗，没有清晰的亮灯装甲板，未检出目标。
仍需将相机对准实际目标，在桌面画面上确认颜色、数字、角点和 Solver 链路。
