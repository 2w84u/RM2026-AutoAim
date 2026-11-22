# Orin NX 上的 YOLO 性能优化实施路线

更新日期：2026-11-14。

推荐顺序：**准备真实数据并固定运行条件 → 用现有模型接 TensorRT FP16 → 优化前后处理 → 必要时做 INT8 → 最后考虑更换网络和重新训练。** 如果仍需使用 OpenCV DNN，先修复 YOLOv8/v5 的全局平均池化导出；这个修复不是 YOLO11 接 TensorRT 的前置条件。

本文是待执行的工作说明。编写本文时只读取了源码、已有报告和已安装软件信息，没有修改模型、接入 TensorRT、改变功耗设置或运行测试。文中的命令分别标注了使用前提；后续执行测试仍须按用户明确要求的方式和范围进行。所有输入验证、校准和性能比较都使用真实图片、真实录像或真实相机帧，不使用随机输入、mock 或合成样本。

## 1. 当前起点与优先级

本机为 Jetson Orin NX 16GB / Seeed Super，L4T R36.4.3；项目与 OpenCV 4.10 均为 Release 构建。当前程序使用 OpenCV DNN CUDA FP16。

本机已经安装 TensorRT **10.3.0.30**，包括开发库、ONNX parser 和 Python bindings。命令行工具位于 `/usr/src/tensorrt/bin/trtexec`，不在当前 PATH 中。第一阶段使用这套安装，不要求先升级系统或 TensorRT。

2026-11-14 的真实相机结果如下，每个模型 120 帧，排除前 10 帧：

| 模型 | 输入 | forward 均值 | detect 均值 | CPU 计算层回退 |
|---|---|---:|---:|---|
| YOLO11 | 640×640 | 28.77 ms | 43.16 ms | 0 |
| YOLOv8 | 416×416 | 39.92 ms | 45.08 ms | 8 个 ReduceMean |
| YOLOv5 | 640×640 | 36.07 ms | 47.69 ms | 9 个 ReduceMean |

来源：[三模型报告](../out/detector_verify/headless_20261114_231856/comparison.json)。这轮没有检出目标，因此不能作为精度基线，也没有覆盖 YOLOv8 有候选时的数字分类开销。三者结构、输入和输出不同，不能用耗时相减直接计算 CPU 回退代价。

| 顺序 | 工作 | 完成后再决定什么 |
|---|---|---|
| 1 | 准备有目标的录像，建立固定运行条件下的 OpenCV 基线 | 当前模型的检测效果是否可用 |
| 可选 | 修复 v8/v5 的池化导出，保留原模型 | 保留 OpenCV 时能减少多少耗时 |
| 2 | 先接 YOLO11 TensorRT FP16，复用现有前后处理 | 后端更换是否已经满足延迟预算 |
| 3 | 按耗时优化预处理、输出布局和解码 | CPU 部分是否仍占主要时间 |
| 4 | 若仍由网络计算限制，进行真实数据 INT8 校准 | 速度收益是否值得精度代价 |
| 5 | 若仍不达标，换更小的网络或做 QAT/重新训练 | 模型结构与训练是否需要调整 |

达到预先约定的速度和精度目标即可停止推进，不必为了使用 INT8 而量化。

## 2. 先准备哪些材料

### 模型与配置

- 保留当前 ONNX、原始 XML/BIN、YAML 和转换报告，后续产物使用新文件名。
- 尽量获取原始训练权重 `.pt`、训练代码/版本、类别表和量化前 ONNX。拿到这些材料后通常可以直接重新导出，不必先重训。
- 只有现有 ONNX 也能开始 TensorRT FP16 和训练后量化；不能从反量化后的权重恢复已经丢失的原始训练精度。
- 记录模型文件哈希、输入尺寸、颜色顺序、归一化、关键点定义、类别映射、阈值和 ROI 设置。

当前模型契约：

| 模型 | ONNX | 输入 NCHW | 原始输出 | 适配要求 |
|---|---|---|---|---|
| YOLO11 | `assets/yolo11_dnn.onnx` | 1×3×640×640 | 1×50×8400 | 38 类、4 个二维关键点 |
| YOLOv8 | `assets/yolov8.onnx` | 1×3×416×416 | 1×14×3549 | 2 类、4 个二维关键点，另有数字分类器 |
| YOLOv5 | `assets/yolov5.onnx` | 1×3×640×640 | 1×25200×22 | 保持旧版列布局 |

预处理必须保持：等比例缩放、左上对齐、右侧/下侧补零、BGR 转 RGB、除以 255、NCHW。不要直接换成其他 YOLO 工具默认的居中 letterbox。

### 真实数据

- 准备一段固定的、有装甲板的录像作为初期 A/B 输入。保留相同起始帧和帧序号，逐帧比较，避免两次相机拍摄内容不同。
- 另留独立的精度评估数据，覆盖远近距离、小目标、不同角度、曝光、运动模糊、类别以及无目标背景。
- INT8 校准图像与精度评估数据按录像或场景分开。校准通常不要求标签；评价漏检、类别和四角点误差需要标注或可靠参考。
- 当前 `detector_verify --save-raw` 只保存前 5 帧原图，不能把它当成完整录像或校准数据采集功能。

### 先填写验收预算

在比较之前约定：`detect` 均值/P95 上限、实际处理 FPS、可接受的漏检与误检变化、四角点误差，以及对 PnP 位姿误差/抖动的要求。数值应来自自瞄系统需求，不在本文中虚构目标。

`detect` 延迟不含完整的相机采集、跟踪和控制链路；达到检测预算后仍需按实际部署任务评价完整链路。

## 3. 建立可比较的基线

以下命令在项目根目录执行，示例中的变量在同一个终端保留。`EVAL_VIDEO` 必须替换成已经存在的真实录像路径。

### 3.1 记录环境与原始状态

这些是已有工具支持的命令：

```bash
RUN_DIR="out/detector_verify/optimization_$(date +%Y%m%d_%H%M%S)"
EVAL_VIDEO="/absolute/path/to/armor_eval.avi"
mkdir -p "$RUN_DIR"
cp configs/sentry.yaml "$RUN_DIR/config.yaml"
sha256sum assets/yolo11_dnn.onnx "$EVAL_VIDEO" > "$RUN_DIR/inputs.sha256"
nvpmodel -q --verbose > "$RUN_DIR/power-before.txt"
sudo jetson_clocks --show > "$RUN_DIR/clocks-before.txt"
```

本机原测试选中了 40W、模式 4，GPU 上限为 1173 MHz，但主要采样频率是 408 MHz。当前查询还发现部分实际参数与该模式配置值不同。开始比较前，应对照 `/etc/nvpmodel.conf` 处理这些差异，记录实际 CPU/GPU/EMC 频率及可用核心；不要只记录“40W”标签。

不要照抄其他 Jetson 的模式编号。如果需要切换模式，先根据本机配置选择，并在允许重启的窗口处理工具提示；本文不安排自动重启。频率设置在模式确定之后进行。

### 3.2 比较动态频率与固定频率

先在原运行状态下执行一次下面的真实录像基线：

```bash
./build/detector_verify "$EVAL_VIDEO" \
  -c "$RUN_DIR/config.yaml" \
  --diagnose --warmup 10 -m 120 -p 30 --save=false \
  -o "$RUN_DIR/opencv_dynamic"
```

随后保存频率设置并固定当前功耗模式允许的最高频率：

```bash
sudo jetson_clocks --store "$RUN_DIR/clocks.conf"
sudo jetson_clocks
sudo jetson_clocks --show > "$RUN_DIR/clocks-fixed.txt"
```

保持模型、配置和录像不变，再执行：

```bash
./build/detector_verify "$EVAL_VIDEO" \
  -c "$RUN_DIR/config.yaml" \
  --diagnose --warmup 10 -m 120 -p 30 --save=false \
  -o "$RUN_DIR/opencv_fixed"
```

这两次才用于判断固定频率带来的变化。之后比较 OpenCV 与 TensorRT、FP16 与 INT8 时，继续使用相同的固定频率条件，不要把多个改动混在一次对比中。

另一个终端可用 `tegrastats --interval 200` 记录频率、负载、温度和功耗，结束后用 Ctrl+C 停止。监控需要覆盖真实推理时间；不能只看运行前后的空闲频率。记录两次运行的其他负载，避免同时编译或运行另一套视觉程序。

全部固定频率比较结束后，恢复先前保存的设置：

```bash
sudo jetson_clocks --restore "$RUN_DIR/clocks.conf"
```

该恢复步骤针对保存的时钟设置，不代替功耗模式恢复。固定频率的作用与边界见 [NVIDIA Jetson 功耗文档](https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#maximizing-jetson-orin-series-performance)。

**本阶段交付：** 有目标的检测输出、动态/固定频率下的阶段统计、运行环境和输入文件记录。若实际目标仍无法正常检出，先处理模型精度、类别/预处理契约或阈值问题，再进行量化。

## 4. 可选：修复 v8/v5 在 OpenCV 下的 CPU 回退

**状态：待实现。** 修改位置为 [tools/ir2onnx.py](../tools/ir2onnx.py) 的 `ReduceMean` 导出分支。

原始 v8/v5 的这 17 个节点全部是四维输入，归约轴 `[2, 3]`，`keepdims=true`，也就是全局平均池化。OpenCV 4.10 将通用 Reduce 放到 CPU，而平均池化有 CUDA 实现。

实施要求：

1. 仅在输入维数、归约轴与保留维度满足全局平均池化语义时，导出 `GlobalAveragePool`；其他 Reduce 保持原映射。
2. 生成独立候选文件，例如 `yolov8_gap.onnx`、`yolov5_gap.onnx`；复制对应配置并只修改模型路径。
3. 用同一真实录像比较原版和候选版，检查检测输出、角点偏差和 forward/detect 统计。
4. 从真实推理后的图和日志确认原先 8/9 个计算层回退已消失。`_input` 提示单独解释，不要求为了它修改模型。

数学等价不代表不同精度、不同计算顺序下逐位相同，因此仍需真实输入对比。这个改动对已经没有计算层回退的 YOLO11 没有直接收益。

**本阶段交付：** 转换器改动、独立候选模型、真实输入的结果和性能对比。达到 OpenCV 路径的目标后可以先部署这一版，无须等待后续全部完成。

## 5. 主线：先接入 YOLO11 TensorRT FP16

### 5.1 先生成 engine

以下构建命令使用本机已有 TensorRT 10.3 工具，先从现有 YOLO11 ONNX 开始：

```bash
mkdir -p out/tensorrt
/usr/src/tensorrt/bin/trtexec \
  --onnx=assets/yolo11_dnn.onnx \
  --fp16 \
  --saveEngine=out/tensorrt/yolo11_fp16.engine \
  --skipInference
```

`--skipInference` 用于只构建并保存 engine，跳过输入推理与性能测试。TensorRT 构建本身仍会进行内部实现选择/计时，构建完成不能作为真实图像精度或端到端速度的证明。不要去掉该选项后使用工具默认的随机输入做验证。

构建失败时依据 parser 的具体节点、算子和形状错误处理，保留日志，不通过随意删除节点绕过问题。`--fp16` 是允许选择 FP16 实现，不保证所有层和输入输出都是 FP16。

在目标 NX 上生成 engine，并记录模型哈希、TensorRT/CUDA 版本与构建选项。部署时不要假定普通 engine 可以跨不同 GPU 或软件版本直接复用。

### 5.2 接入项目，而不只是生成一个文件

**状态：待实现。** 当前 `.onnx` 路由是 OpenCV DNN，程序没有 TensorRT engine 加载分支。直接把 YAML 路径改成 `.engine`，或把 `device` 改成 `INT8`，都不会完成后端切换。

| 位置 | 需要做的工作 |
|---|---|
| 根目录及 `tasks/auto_aim/CMakeLists.txt` | 新增 TensorRT/CUDA Runtime 依赖与构建选项；保持现有构建可用 |
| `tools/infer_backend.hpp` 及新增执行器文件 | 新增明确的 engine 路由和 TensorRT 加载/执行封装，不复用 OpenCV 的 FP16 开关冒充 TensorRT |
| `tasks/auto_aim/yolos/yolo11.cpp` / `.hpp` | 接入 TensorRT 分支，最初复用现有预处理、输出解析与 Armor 构造 |
| `tasks/auto_aim/inference_diagnostics.hpp`、`tests/vision/detector_diagnostics.hpp` | 记录实际后端、engine 元信息与可比较的计时；不为 TensorRT 伪造 OpenCV DOT |
| 独立候选 YAML | 保留现有配置，候选配置在路由实现后指向 engine |

执行器应在初始化时创建 runtime、engine、execution context、CUDA stream 和缓冲区，并在各帧复用，避免逐帧加载模型或分配显存。

使用本机 TensorRT 10 的命名张量接口查询输入输出名称、形状、dtype 和布局，再绑定缓冲区并调用 `enqueueV3()`。不要假设输出一定为 FP32，也不要把 FP16 内存直接包装为 `CV_32F`。首版保留 640×640、batch=1 和现有输出契约，暂不同时改分辨率、检测头或 NMS。

计时必须等相应 GPU 工作和输出拷贝完成：只测异步 `enqueueV3()` 返回时间会严重低估推理延迟。保留与现有报告可比的主机 forward 耗时；若另记 CUDA event 耗时，应明确标为设备阶段，不能直接替代旧 forward 数值。

### 5.3 再运行真实数据对比

接入完成后，用同一 `detector_verify` 入口、同一录像、固定频率条件和相同阈值，对比 OpenCV FP16 与 TensorRT FP16。只变更推理后端和对应模型路径；先保留 CPU 前后处理，便于归因。

以实际 `detect` 均值和 P95 为主要性能结果，同时记录 forward 与前后处理阶段。候选模型必须能处理真实目标，框、类别和角点变化需在已约定范围内。TensorRT FP16 如果已经达标，就可以暂不做 INT8。

## 6. 处理前后处理耗时

YOLO11 原报告中，forward 之外约为 **14.39 ms**，主要包括：缩放 2.42 ms、`blobFromImage` 5.71 ms、绑定输入 0.63 ms、输出转置 2.07 ms、解码 3.55 ms。

**状态：待实现，先以新基线决定具体改动。** 优先考虑复用输入/输出内存，减少中间图像与张量复制、整块输出转置及逐候选的临时对象。如果 CPU 预处理仍占主要时间，再考虑将缩放、补边、RGB/NCHW 转换和归一化合并到 GPU 路径。

优化必须维持原始坐标变换、关键点顺序和阈值语义。GPU 预处理也有上传和同步成本，应比较整段 detect 的收益；降低 GPU kernel 时间而增加拷贝，不算完成优化。

每个有实际改动的候选版本按明确约定的真实数据范围比较。若目前已达性能预算，不为了优化而增加新的 GPU 处理模块。

## 7. 仍不够快时，再做 INT8

### 7.1 进入条件

只有在 TensorRT FP16 的识别效果已建立基线、运行状态一致，并且剩余主要开销确实来自网络计算时，才优先尝试 INT8。如果主要时间在 CPU 预处理、角点解码或数字分类，应先处理对应阶段。

当前 YOLO11/v8 来自 `--fq drop` 转换，已经去掉激活 FakeQuantize，权重也已反量化；v5 原 IR 没有这类 FQ。对现有浮点 ONNX 再量化可以尝试，但不会自动恢复原 OpenVINO INT8 的数值行为。具体转换边界见 [IR 转换说明](ir-to-onnx.md)。

本机 OpenCV 4.10 的 CUDA 路径不支持其 INT8 量化网络执行方式；本阶段的部署后端应为 TensorRT，不能只替换成量化 ONNX 后继续期待 OpenCV CUDA 获得 INT8 加速。

### 7.2 准备校准输入

使用第 2 节收集的真实场景，保持与部署完全一致的预处理、输入尺寸和归一化。数据应覆盖装甲板、远距离小目标、不同颜色/类别、亮暗变化、运动和真实背景；不要只抽取大量相邻、几乎相同的帧，也不要只用空场景。

先形成可追溯的图片列表和预处理配置；是否充分应由独立数据上的结果判断，不承诺某个固定图片数量必然足够。训练后量化（PTQ）使用代表性数据确定量化参数，不需要先重新训练；QAT 则需要训练流程参与。[NVIDIA 量化说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)

### 7.3 在 TensorRT 10.3 下实现校准

**状态：待实现，仓库目前没有对应校准工具。** 可针对本机 10.3 实现真实图片数据读取和 `IInt8EntropyCalibrator2` 校准流程，生成与当前模型、尺寸和预处理绑定的缓存，再构建 INT8 候选 engine。保留浮点实现处理不适合量化或精度敏感的部分。

不要直接运行无真实校准数据/有效缓存的 `trtexec --int8`，也不要将 `--loadInputs` 当成校准数据集参数；它提供的是推理输入。本文不提供会在校准材料缺失时退化为随机输入的 INT8 快捷命令。

另一条路线是由兼容的工具离线生成带 `QuantizeLinear/DequantizeLinear`（Q/DQ）的 ONNX，再交给 TensorRT。两种路线要分开实施：Q/DQ 模型的量化参数已在图中，不再叠加传统校准缓存。较新 TensorRT 版本推荐显式量化，但本机是 10.3，不能照搬 11.x 的 API 或升级要求；实现时以本机头文件和版本支持为准。

### 7.4 决定是否采用 INT8

在独立真实数据上与 FP16 比较：

- `detect` 均值/P95 是否达到目标，是否存在明显尾延迟。
- 远距离目标漏检、误检、类别和置信度变化。
- 四角点的平均误差、较大误差，以及固定输入下的 PnP 结果变化。
- engine 实际层精度；不能因为文件名包含 `int8` 就认定整网使用 INT8。

如果速度收益小，或角点/PnP 精度不达标，继续使用 FP16；必要时调整校准数据和敏感部分精度。INT8 达标后再考虑替换部署配置，保留可回退的 FP16 版本。

## 8. 什么时候才需要重新训练

| 情况 | 下一步 |
|---|---|
| 能拿到原始 `.pt`，只是希望避免 IR 反向转换 | 先直接导出符合现有契约的 ONNX，无需重训 |
| 相同结构精度足够，只是推理慢 | 先完成后端、图表达和前后处理优化；同结构重训一般不减少计算量 |
| FP16 精度合格，PTQ INT8 精度不合格且必须用 INT8 | 评估 QAT，需要可训练模型、训练代码和真实数据 |
| 优化后仍超出延迟预算 | 选择更小或在目标 GPU 上更合适的结构，再训练/微调 |
| 现有浮点模型本身漏检、类别或角点不合格 | 准备标注数据，修正训练与输出契约 |

降低输入分辨率也可能减少计算量，有原始模型时可先评估重新导出，不一定需要重训。但当前 IR 转出的图使用静态形状，适配器也固定了输入大小，不能只改 YAML 就把 640 改成 416。必须同时处理模型导出与适配器，并评估远距离小目标和关键点误差。

如果重训，仍需明确类别映射和四个物理角点。当前 YOLO11 适配器要求二维关键点布局，不能直接套用含可见性通道的默认 pose 输出，也不能拿检测框四角代替装甲板角点。训练准备见 [训练与导出指南](yolo-training-export.md)，最终以实际适配器契约为准。

## 9. 如何记录验收结果

每个候选版本保存：模型/engine、配置、来源与哈希、构建日志、输入录像标识、环境和监控记录、逐帧检测 JSON，以及阶段均值/P50/P95/最大值。保持原模型与候选产物可同时使用。

已有 `detector_verify` 的诊断文件位于各自输出目录的 `diagnostics/report.json`，其中也含逐帧检测。完成候选版本的真实输入运行后，可调用现有工具查看 A/B 差异：

```bash
python3 tools/compare_detections.py \
  "$RUN_DIR/opencv_fixed/diagnostics/report.json" \
  "$RUN_DIR/candidate/diagnostics/report.json"
```

这里的 `candidate` 是示例候选输出目录，需要由对应运行实际生成。两份记录必须来自同一录像的相同帧范围。该工具当前只比较共同帧，不能用它的 PASS 证明两份结果完整对齐。

工具默认 PASS 条件主要是匹配率至少 95% 且无类别不一致；多余检测数量和关键点误差没有作为完整通过门槛，参考目标为零时也可能 PASS。因此还需按约定范围评价漏检/误检与角点结果，不能拿空场景 PASS 或平均 IoU 代替自瞄精度验收。相对于旧模型的一致性也不等同于相对于标注的正确性。

建议使用下表记录实际结果，不填入估算或未经运行的数据：

| 版本 | 输入/帧范围 | 运行状态 | forward 均值 | detect 均值/P95 | 检测与角点结果 | 是否达标 |
|---|---|---|---|---|---|---|
| OpenCV 原运行状态 | 待填 | 待填 | 待填 | 待填 | 待填 | 待定 |
| OpenCV 固定频率 | 同上 | 待填 | 待填 | 待填 | 待填 | 待定 |
| TensorRT FP16 | 同上 | 同上 | 待填 | 待填 | 待填 | 待定 |
| FP16 + 前后处理优化 | 同上 | 同上 | 待填 | 待填 | 待填 | 待定 |
| TensorRT INT8（如需要） | 同上 | 同上 | 待填 | 待填 | 待填 | 待定 |

初期先沿用 120 帧、10 帧预热的比较范围。如果明确需要评估持续温升或尾延迟，再单独约定更长的真实运行范围；短段基线不能证明长期稳定。

## 10. 接下来可以直接拆分的工作

- [ ] 数据准备：收集有目标的固定录像，确定速度、检测和角点误差预算。
- [ ] 环境与基线：整理功耗状态，完成同一录像下的动态/固定频率比较。
- [ ] 可选修复：仅在保留 v8/v5 OpenCV 路径时实施池化导出修正。
- [ ] 主线开发：接入 YOLO11 TensorRT FP16，复用现有前后处理和真实输入诊断。
- [ ] 按实测结果决定是否优化前后处理、实施 INT8 或准备重新训练。

以上是后续任务清单，不表示本文已完成其中的实现或测试。
