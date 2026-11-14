# OpenVINO IR → ONNX：已实施的转换流程

仓库的四个 IR 均已生成 ONNX，并在 x86 CPU 上通过 ONNX checker、ONNX Runtime 和
OpenCV DNN 4.10 的加载及前向检查。转换器为 `tools/ir2onnx.py`。

**可行范围：转换和浮点图执行已验证；原 INT8 模型的检测精度保持、Jetson CUDA/FP16
仍未验证。** `--fq drop` 会改变模型行为，不能按“误差通常很小”直接验收。

## 1 产物与接入

| IR 来源 | 已生成 ONNX | 输入 NCHW | 输出 | 大小 MiB |
|---|---|---|---|---|
| `assets/yolo11.xml` | `assets/yolo11_dnn.onnx` | 1×3×640×640 | 1×50×8400 | 10.38 |
| `assets/yolo11_buff_int8.xml` | `assets/yolo11_buff_int8.onnx` | 1×3×640×640 | 1×17×8400 | 10.35 |
| `assets/yolov8.xml` | `assets/yolov8.onnx` | 1×3×416×416 | 1×14×3549 | 13.00 |
| `assets/yolov5.xml` | `assets/yolov5.onnx` | 1×3×640×640 | 1×25200×22 | 5.44 |

以上均使用 `--fq drop --opset 13`，权重反量化后存为浮点 initializer，不是 INT8 ONNX。
`yolo11` 是 38 类和 4 个关键点，`yolov8` 是 2 类和 4 个关键点；能量机关模型是
1 类和 6 个关键点。`yolov5` 保留旧版 22 列布局，不按上述公式解码。

新增 [configs/jetson-ir.yaml](../configs/jetson-ir.yaml)，使用 `yolo11_dnn.onnx` 和
`class_num: 38`。`configs/jetson.yaml` 也已改用 `yolo11_dnn.onnx`。

> **注意：`assets/best2-sim.onnx` 不能直接用当前适配器解析。** 实测其输出为
> `4 框 + 1 类 + 5 关键点（10 通道）= 15`，而 `yolo11` 适配器假设
> `4 框 + nc 类 + 4 关键点`。若误把 `class_num` 设为 3，关键点坐标会被当成
> 类别分数，表现为置信度出现数百、满屏误检。使用 5 关键点模型需要新写适配器。
切换到 `yolov8` 时要同时设置 `yolo_name: yolov8` 和 `class_num: 2`。
能量机关配置中的 `model` 可改为 `assets/yolo11_buff_int8.onnx`。

在实际 Jetson、已构建对应可执行文件且有本地视频时执行：

```bash
./build/detector_video_test --config-path=configs/jetson-ir.yaml match.avi --tradition=false
```

CUDA 构建和后端条件见 [Jetson 构建说明](jetson-cuda-build.md)。启动日志应出现
`[infer] OpenCV DNN CUDA FP16 backend enabled`；出现 CPU 回退日志不能算 CUDA 验证通过。
本次未运行 Jetson 或完整 C++ 检测程序。

## 2 环境与复现

实测环境为 x86_64、Python 3.11、NumPy 1.26.4、ONNX 1.17.0、ONNX Runtime 1.20.1、
OpenVINO 2024.6.0、OpenCV 4.10.0。依赖已锁定在
[tools/requirements-ir2onnx.txt](../tools/requirements-ir2onnx.txt)。
OpenVINO 只在 `--validate` 时需要；单独转换仅依赖 NumPy 和 ONNX。

```bash
# 在仓库根目录，使用 Python 3.11 创建独立环境
python3.11 -m venv .venv-ir2onnx
source .venv-ir2onnx/bin/activate
python -m pip install -r tools/requirements-ir2onnx.txt

# 只解析并检查算子与卷积权重；不需要安装 onnx
python tools/ir2onnx.py assets/yolo11.xml --inspect

# 转换并用同一张图片验证，报告目录自动创建
python tools/ir2onnx.py assets/yolo11.xml assets/yolo11_dnn.onnx \
  --fq drop --validate --image assets/standard_fanblade.jpg \
  --report out/ir2onnx/yolo11.json

for model in yolo11_buff_int8 yolov8 yolov5; do
  python tools/ir2onnx.py "assets/${model}.xml" "assets/${model}.onnx" \
    --fq drop --validate --image assets/standard_fanblade.jpg \
    --report "out/ir2onnx/${model}.json" || break
done
```

本次已创建 `.venv-ir2onnx/`，可直接用 `.venv-ir2onnx/bin/python` 执行上述命令。
`--bin` 可指定独立权重路径；`--opset` 支持 13～17，已验证的是默认 13。
自检遇到未覆盖算子或无法折叠的卷积权重时返回非零状态。

`--image` 沿用项目预处理：保持比例缩放、左上对齐、右侧/下侧补零、BGR 转 RGB，
然后转换为 float32 NCHW 并除以 255。**模型输入范围是 0～1，不要再乘 255。**
不指定图片时使用固定种子的 `[0, 1)` 随机输入。

## 3 实测结果与验收含义

2026-11-13，使用仓库的 `assets/standard_fanblade.jpg` 和上述版本得到以下结果。
所有数值都是整份原始输出的最大绝对误差，包含低置信度候选的坐标和关键点。

| 模型 | 去 FQ 并折叠常量的 IR vs ORT | ORT vs OpenCV CPU | 原始 IR vs ORT |
|---|---:|---:|---:|
| yolo11 | 0.000717 | 0.001251 | 44.664948 |
| yolo11_buff_int8 | 0.005798 | 0.002869 | 129.252136 |
| yolov8 | 0.000397 | 0.001411 | 56.524719 |
| yolov5 | 0.001602 | 0.003326 | 0.001602 |

原始报告见 [out/ir2onnx](../out/ir2onnx)。其中：

- `ir_vs_ort` 始终比较**原始 IR** 与导出模型，保留量化移除的实际影响。
- `ir_without_fq_vs_ort` 比较同样去掉 FQ、折叠权重常量后的浮点 IR，用于检查转换映射。
  提前折叠也避免 OpenVINO CPU 继续对反量化权重链执行低精度融合，混淆比较基准。
- `ort_vs_opencv` 检查同一 ONNX 在两种 CPU 执行器中的差异。
- `checks` 保存各项结果。`drop` 模式的 `validation_passed` 要求浮点 IR/ORT 和
  ORT/OpenCV 均满足逐元素 `atol=1e-2, rtol=1e-3`，并检查输出数量、形状及有限值。
  它**不表示原始 INT8 IR 数值对齐或检测精度已经通过**。
- 数值对齐未通过时，脚本仍保留 ONNX 和 JSON 报告，但返回状态 1；所有报告均标记
  `accuracy_validated: false`。

本次图片在原能量机关 IR 和导出模型中均没有置信度超过 0.7 的候选，不能用它证明
真实目标的框、类别和关键点保持一致。还需要同一段有目标的视频或标注数据，比较
NMS 后的框 IoU、类别、置信度和关键点误差，再决定是否替换实际使用的模型。

## 4 量化处理与 exact 模式

`yolo11`、能量机关和 `yolov8` 包含 INT8 权重反量化及激活 `FakeQuantize`。
`yolov5` 的权重主要为 FP16，且没有 FQ；不能把这四个模型都称为 INT8 模型。
IR 元数据记录的生成版本分别为 2024.2、2025.2、2024.0、2023.3；本次均用
OpenVINO 2024.6 读取成功，该版本是验证环境版本，不是所有模型的生成版本。

- `--fq drop`：权重反量化后折叠为浮点常量，激活 FQ 视为恒等。兼容性已验证，
  但不能恢复原 FP32 训练权重，也不能保证原 INT8 模型的检测精度。
- `--fq exact`：按 FQ 公式展开浮点算术、Clip 和 Round，保留模拟量化。
  当前只支持 `input_high > input_low` 的量化区间。`exact` 表示采用量化公式，
  **不保证不同执行器间逐位一致**。

FQ 运算和舍入语义参考 [OpenVINO FQ 文档](https://docs.openvino.ai/2024/documentation/openvino-ir-format/operation-sets/operation-specs/quantization/fake-quantize-1.html)
及 [2024.6 参考实现](https://github.com/openvinotoolkit/openvino/blob/2024.6.0/src/core/reference/include/openvino/reference/fake_quantize.hpp)。
深层量化图中的浮点误差可能跨越舍入阈值；原 IR 的 CPU 量化内核与展开图也可能不同，
不能以公式一致推断整个检测头输出一致。

```bash
# 实验模式：此次 yolo11 的数值验收失败，预期返回 1
python tools/ir2onnx.py assets/yolo11.xml out/ir2onnx/yolo11_exact.onnx \
  --fq exact --validate --image assets/standard_fanblade.jpg \
  --report out/ir2onnx/yolo11_exact.json
```

该模式实测原 IR/ORT 最大误差为 19.720398，ORT/OpenCV 为 15.285400，未达到原方案
声称的 `1e-4～1e-3`。因此仅保留为实验入口，不作为 Jetson 默认产物；详细结果见
[yolo11_exact.json](../out/ir2onnx/yolo11_exact.json)。

## 5 转换器的关键修正与边界

本次修复了会阻止原脚本运行或导致错误图的映射：

| 项目 | 已实施处理 |
|---|---|
| 算术节点 | IR `Multiply/Subtract/Divide` 分别映射到 ONNX `Mul/Sub/Div` |
| Gather / VariadicSplit | 从常量输入读取轴；Gather 的轴不作为 ONNX 输入 |
| Split 长度 | 转为 int64，解析 `-1`；保留多输出端口映射 |
| 常量折叠 | 修正 Slice/StridedSlice 的输入索引，以及整数 Divide 的 dtype/取整 |
| Conv / GroupConv | 显式导出 kernel_shape，折叠反量化权重；GroupConv 重排权重 |
| FQ | 标量 Clip 上下界，按原公式运算顺序展开 |
| 导出 | 保留零维常量、裁剪未使用的 initializer、完整 ONNX checker、自动创建目录 |
| 验证 | 同一预处理输入、浮点 IR 基准、ORT/OpenCV 前向、JSON 和失败退出状态 |

这是针对仓库四个模型的静态图转换器。ShapeOf 和 Reshape 使用 XML 中的静态形状，
不能把输入尺寸改为动态，也不会补回训练图、原始高精度权重或额外 NMS。
新增模型出现未知算子、复杂切片掩码或 Gather 的非零 batch_dims 时需要另外适配。
本次四个产物已能直接由 OpenCV 加载，没有引入 onnxsim 作为必要步骤。

## 6 C++ 端识别验证（detector_verify）

`tests/vision/detector_verify.cpp` 是专门用于验证转换后模型的**无界面**工具，支持图片、
图片目录、视频和相机实时输入，并可导出 JSON 做 A/B 对比。它绕过 `Debug` 适配器
自带的弹窗，只走 `YOLO::detect()` 这条实际推理路径。

该目标在 Debug 和 Release 下都可用，但不随默认构建编译。先单独构建：

```bash
cmake --build --preset default --target detector_verify
```

### 6.1 图片 / 视频 / 目录

```bash
# 单张图片
./build/detector_verify assets/xxx.jpg --config-path=configs/jetson-ir.yaml

# 图片目录（标注结果写入 out/detector_verify/）
./build/detector_verify /data/armor_imgs --config-path=configs/jetson-ir.yaml -o out/detector_verify

# 视频（同时写出 out/detector_verify/annotated.avi 和 JSON）
./build/detector_verify match.avi --config-path=configs/jetson-ir.yaml --json cand.json
```

注意：OpenCV 的 `CommandLineParser` 对 `-c 路径` 这种“空格分隔”的短参数解析不可靠，
请统一写成 `--config-path=路径`。

每帧输出耗时、检出数、类别、置信度、框与第一个关键点；结束时输出帧数、
有检出帧比例、类别直方图和平均耗时/FPS。若整段没有任何检出，会打印排查提示。

### 6.2 相机实时

```bash
# camera_name / exposure_ms / gain 全部取自 config
./build/detector_verify --camera --config-path=configs/jetson-ir.yaml -m 300 -p 30
```

- `-m` 为最多处理帧数（0=不限，Ctrl+C 结束），`-p` 为每 N 帧打印一次
- 相机采集是阻塞读取；若 5 秒没有任何帧，工具会直接退出并打印排查提示

海康相机报 `MV_CC_EnumDevices failed: 0x80000006` 或 `Unable to open usb!` 时，
先确认设备节点权限（本机实测原因为 `crw-rw-r-- root root`，普通用户无写权限）：

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2bdf", MODE="0666", GROUP="plugdev"' \
  | sudo tee /etc/udev/rules.d/99-hikrobot.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
# 之后重新插拔相机
```

### 6.3 IR / ONNX A/B 对比

```bash
# 参考：x86 上用 OpenVINO 构建的同一工具 + .xml 配置
./build/detector_verify match.avi -c configs/ref_ir.yaml --json ref.json
# 候选：Jetson 上用转换后的 ONNX
./build/detector_verify match.avi --config-path=configs/jetson-ir.yaml --json cand.json

python3 tools/compare_detections.py ref.json cand.json --iou 0.5
```

比较脚本按类别 + IoU 贪心匹配，输出漏检、多余、类别不一致、平均 IoU、
关键点平均偏移；匹配率低于 `--min-match`（默认 95%）时返回非 0，可用于回归验收。
这是目前唯一能覆盖“NMS 后框/类别/关键点”的验证方式，建议录制一段有目标的
比赛视频作为固定基准。
