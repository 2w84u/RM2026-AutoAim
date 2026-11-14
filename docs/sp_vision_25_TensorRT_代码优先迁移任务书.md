# sp_vision_25：深圳大学 YOLOv5 → Jetson NX TensorRT 代码优先迁移任务书

> 将本文件整体交给本地代码 AI 执行。**优先完成代码与脚本；暂时没有 Jetson NX、CUDA 或 TensorRT 环境时，不要因此停止编码，也不要假装运行成功。**

## 1. 目标与范围

项目是我们基于同济 `sp_vision_25` 修改过的本地仓库，以**本地现有代码**为准。

- 上游参考：https://github.com/TongjiSuperPower/sp_vision_25
- 深圳大学模型：https://github.com/broalantaps/RobotDetectionModel
- **本次唯一优先使用的模型**：`https://raw.githubusercontent.com/broalantaps/RobotDetectionModel/main/Model/0708.onnx`
- 同济现有 OpenVINO 推理：`tasks/auto_aim/yolos/yolov5.cpp`（若本地结构已变化，定位实际实现）

**目的**：无需训练、不转换成 `.pt`，使用深圳大学原始 `0708.onnx` 生成 TensorRT `.engine`，在 Jetson NX 上进行 CUDA 加速推理；让后续 `Solver`、`Tracker`、`Aimer` 等模块**基本无感知**。

**本次不做**：证明 ONNX 与同济 XML 的权重完全相同、开发新训练流程、建立复杂精度基准、长时间性能测试、强制实现多种运行时、为兼容所有 TensorRT 版本设计大而全的抽象层。实际效果由我在 NX 上人工看画面判断。

## 2. 必须先明确的部署策略

### 2.1 x86 端能完成什么

在普通 x86 Linux 主机（即使是 Intel CPU + Intel Arc 显卡）上，**先完成**：

1. 下载并保存 `0708.onnx`；可以用 ONNX 库静态检查输入输出，但不做额外模型等价性鉴定。
2. 编写 TensorRT C++ 后端、CMake 选项、配置文件和脚本。
3. 编写一键部署/转换/编译/运行指导；允许在 x86 上构建不依赖 TensorRT 的代码部分。
4. 使用现有代码风格整理提交，让工作区在无 NX 时也能清楚区分「已完成代码」和「尚待设备验证」。

**普通 TensorRT `trtexec` 不能依靠 Intel Arc 显卡生成用于 Jetson NX 的 Engine。** x86 即使装有 NVIDIA 显卡，其常规 Engine 也不保证可在 Jetson 上加载，尤其涉及 x86_64→aarch64、GPU 架构和 TensorRT 版本差异。不要将它作为主方案。不要用 TensorRT-RTX 的 CPU AOT 能力代替 Jetson 使用的标准 TensorRT。

### 2.2 NX 端必须做什么

**推荐在 NX 本机只做一次 ONNX→Engine 构建**。之后程序直接加载 Engine，不要每次启动重新转换：

```text
x86 主机：代码 + 0708.onnx + 构建脚本
                    │
                    └── rsync/scp → Jetson NX
                                   │
                             首次运行转换脚本
                                   ↓
                              0708_fp16.engine
                                   │
                             编译/运行视觉程序
                                   ↓
                              手动查看检测效果
```

Engine 构建脚本应默认生成 FP16，必要时支持参数切换 FP32。若 NX 暂时不可用，准备好脚本即可，**不阻塞本次代码工作**。

## 3. 优先实施的代码任务

### 任务 A：梳理本地改动（只查与迁移有关的部分）

阅读并定位本地对应实现：

- `tasks/auto_aim/yolo.hpp`、`yolo.cpp`
- `tasks/auto_aim/yolos/yolov5.hpp`、`yolov5.cpp`
- `tasks/auto_aim/armor.hpp`、`armor.cpp`
- `tasks/auto_aim/CMakeLists.txt`、顶层 `CMakeLists.txt`
- 实际使用的配置文件、相机/视频入口，以及涉及 OpenVINO 的其他源码

识别本地自定义改动，**不要直接复制同济上游覆盖**。只对推理后端和必要的构建开关动刀，优先保持原 OpenVINO 代码可用。

### 任务 B：新增 TensorRT 后端

优先新增 `YOLOV5TRT : public YOLOBase`（可按本地命名规范调整），并在原有 `YOLO` 工厂中增加一个明确的后端选项，如：

```yaml
yolo_name: yolov5_trt
yolov5_trt_model_path: assets/0708_fp16.engine
```

不要把原 `yolov5` 改名或默认切换。让原 `YOLO::detect(const cv::Mat&, int)` 与返回的 `std::list<Armor>` 接口语义保持不变。若基类或现有调用要求实现 `postprocess(...)`，一并处理而不是留下未实现的接口。必要时给 `YOLOBase` 补 `virtual ~YOLOBase() = default;`。

实现内容：

1. 加载已生成的 `.engine`，初始化 TensorRT Runtime/Engine/Context。
2. 查询输入输出名字、类型和维度，检查符合预期（或输出明确错误）；**不要假设 FP16 Engine 的输出内存一定是 float32**。
3. 初始化并复用输入/输出 GPU Buffer；避免每帧重新分配、重新创建 Engine。
4. 对单帧 `cv::Mat` 执行相同预处理，异步/同步执行推理，正确同步后读取结果。
5. 将推理输出交给与原实现一致的后处理，返回 `Armor`。
6. 添加清晰的错误日志及析构释放逻辑；不做复杂自动回退、监控、状态机或多层错误门控。

**TensorRT C++ API 以 NX 实际安装版本为准**：8.x/10.x 的 binding 查询、tensor 地址设置、enqueue 方法有差异。优先针对目标版本写清楚、少量封装的实现，不要在未知版本上猜 API、宣称已通过编译。

### 任务 C：与同济原版保持一致的前后处理

这部分非常关键，不能直接套标准 Ultralytics YOLO 解析器。

**预处理：**

- 输入 `cv::Mat` 为 BGR；按原配置支持 ROI。
- 等比例缩放至不超过 `640×640`，把图像放在全黑画布**左上角**（不是居中 letterbox，填充值也不是 114）。
- BGR→RGB，转换为 FP32，除以 255，变为 NCHW。
- 预期模型输入：`[1,3,640,640]`；裁剪、取整、缩放行为尽量沿用本地 OpenVINO 实现。

**后处理：**

- 预期原输出：`[1,25200,22]`，逐个候选读取。
- `0..7`：四角点，原始顺序 `TL, BL, BR, TR`；参考同济原解析顺序把 `Armor.points` 改成 `TL, TR, BR, BL`。
- `8`：置信度 **logit**，按原实现执行 sigmoid。
- `9..12`：颜色分数，Argmax；`13..21`：数字类别分数，Argmax。
- 保持现有 `Armor` 类别映射、ROI 偏移、NMS、`check_name()`、`check_type()`、`min_confidence`、`use_traditional`、`center_norm` 等行为。
- 注意原上游 YOLOv5 用 `score_threshold = 0.7`、`nms_threshold = 0.3`，还有独立的 `min_confidence`，但**若本地已经调整，以本地逻辑为准**。

可以抽出共用的 `preprocess()` / `parse()`，但只要重构范围开始过大，就直接在新后端复用/迁移必要逻辑；不要为了“架构更完美”改动整条自瞄链路。

### 任务 D：CMake 和可选依赖

当前同济上游 `tasks/auto_aim/CMakeLists.txt` 存在固定 OpenVINO 安装路径，且 YOLOv8、YOLO11、其他识别模块也可能引用 OpenVINO。**不能只改 YOLOv5 而忽略其他编译单元的依赖。**

- 增加简单明确的后端构建选项（命名按本地规范），Jetson 可以选择仅需要的 TensorRT 实现。
- 编译 TensorRT 后端时找到正确的 `NvInfer.h`、`libnvinfer`、CUDA Runtime 等依赖。
- 非 TensorRT 环境下不要强制包含 `NvInfer.h` 或链接 CUDA/TensorRT。
- 原 OpenVINO 模块仅在启用时参与编译；如果有其他功能仍依赖它们，请用最小范围的条件编译处理，不准顺手删掉功能。
- 对 x86 机器上的“未安装 TensorRT”提供清晰的配置/构建提示即可，不要写大量替代后端。

### 任务 E：脚本（优先完成，可复制执行）

脚本目录按本地仓库风格决定，建议：

```text
scripts/trt/
├── prepare_model.sh      # x86：下载 0708.onnx、检查下载结果
├── build_engine.sh       # NX：ONNX → TensorRT Engine，默认 FP16
└── deploy_nx.sh          # x86：以参数传入 NX SSH 地址，同步源码/模型/脚本
```

**`prepare_model.sh`**：

- `curl -fL --retry ...` / `wget` 从深圳大学官方 GitHub Raw 下载。
- 验证下载完成，非空文件，若可用 `onnx` 库则打印输入输出。
- 保存下载源、文件大小、SHA-256（用于追踪本次下载文件，不做模型身份对照）。
- 不强制要求 PyTorch/Ultralytics，也不安装与任务无关的工具。

**`build_engine.sh`**：

- 在 NX 上检查 `trtexec`，支持命令行或环境变量指定路径、精度、输出文件。
- 默认将 `0708.onnx` 生成为 `0708_fp16.engine`；必要时支持 FP32。
- 按 `trtexec --help` / 实际 TensorRT 版本选择有效参数，打印实际调用的命令与日志位置。
- TensorRT 8/10 下的典型命令（以本机工具接受的参数为准）：

```bash
trtexec --onnx=assets/0708.onnx \
        --saveEngine=assets/0708_fp16.engine \
        --fp16
```

- 如转换失败，显示错误并退出，保留 ONNX，不允许静默执行 CPU 推理。
- Engine 存在且用户未显式要求覆盖时可以跳过重复转换；不要让正常启动过程重新构建 Engine。
- 不要把不同 Jetson/不同 TensorRT 环境生成的 Engine 作为通用二进制提交到仓库。

**`deploy_nx.sh`**：

- 接收 `user@host` 和远程目录等参数，不写死 IP、账号、密码。
- 用 `rsync` 优先同步必要源码、配置、ONNX 和脚本；不存在 `rsync` 时给出 `scp` 手动命令。
- 可以提供 `--dry-run` 或简洁的确认信息，但不要做复杂交互菜单。
- 默认不覆盖 NX 上现有配置/Engine；明确说明本次传送的文件。

## 4. 预期 NX 运行步骤（在 Markdown 文档里写清楚）

以下是**参考流程**，本地 AI 应按实际仓库构建方式补全命令。

### 第一次部署

1. x86 执行 `scripts/trt/prepare_model.sh`，确认 `assets/0708.onnx` 存在。
2. x86 执行 `scripts/trt/deploy_nx.sh user@NX_IP /path/to/project`（具体调用按脚本实现）。
3. NX 检查 `/etc/nv_tegra_release`、`trtexec --help`、CUDA/TensorRT 安装；**不要擅自升级 JetPack**。
4. NX 执行 `scripts/trt/build_engine.sh`，生成 `assets/0708_fp16.engine`。
5. NX 按仓库 CMake/构建工具编译启用了 TensorRT 的视觉程序。
6. 配置 `yolo_name: yolov5_trt`，模型路径指向 `.engine`。
7. 用现有图片、录像或相机入口运行检测，人工观察四角点、数字和颜色是否正常。

### 后续启动

只需运行视觉程序，**直接加载现有 Engine**；不再执行 ONNX 转换。

### 人工快速检查

只保留以下最低限度验证：

- `trtexec --loadEngine=...` 能成功加载和执行（根据本机工具参数），或由程序加载成功。
- 摄像头/视频画面能正常检测装甲板，四个角点没有交换位置。
- 颜色与数字识别正常；Solver 能获取正确的 `Armor` 数据。
- 程序退出不崩溃，GPU 内存没有明显持续增加。

**不要新增复杂 benchmark、自动回归矩阵、精度差异报告或连环验收门槛。** 如发现效果确实有问题，再定位修复即可。上实车前可先禁止执行输出，查看检测画面和预测结果。

## 5. 精简交付要求

**优先直接实现代码，不要把时间花在写长篇方案上。** 最终只需：

1. 代码：TensorRT 后端、必要的工厂注册、CMake 配置修改。
2. 脚本：模型准备、NX 构建 Engine、文件部署。
3. **一份 Markdown 指南**：`docs/jetson_tensorrt.md`，说明 x86 → NX 的命令、配置开关和常见错误。
4. 最终在回复中简短列明：改了什么、现在哪些可用、哪些因未连 NX 还没验证。

不强制新增其他设计文档、性能报告或测试平台。请不要直接推送到远程仓库；可以在本地按合理范围提交 commit，等待我决定是否 push。

## 6. 实施顺序与停止条件

按以下顺序执行，**不要因为 NX 环境暂时不可用而停止前面的工作**：

1. 查看本地项目结构和已经修改的识别代码。
2. 下载原始 `0708.onnx`，创建转换/部署脚本。
3. 实现 C++ TensorRT 后端（把预处理和后处理对齐现有代码）。
4. 接入工厂、配置和 CMake，尽量完成 x86 可验证的部分。
5. 补齐 `docs/jetson_tensorrt.md`，列出 NX 上只需要执行的命令。
6. 能访问 NX 就实际转换与编译；不能访问则把缺失的环境信息和未验证项清楚写明，直接结束本次交付。

**禁止**：擅自替换为新 `0526.onnx`、改用标准 YOLO 框输出、把原 OpenVINO 后端删掉、为适配 TensorRT 改写 Solver/Tracker/Aimer、在 x86 Intel 主机上强行生成标称可用于 Jetson 的 Engine、用空实现或吞异常伪装推理成功。

---

### 参考资料

- 深圳大学模型：https://github.com/broalantaps/RobotDetectionModel
- 同济项目：https://github.com/TongjiSuperPower/sp_vision_25
- NVIDIA TensorRT Engine 兼容性：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html
- NVIDIA TensorRT 构建入门：https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/build-your-first-engine.html
- TensorRT 8→10 trtexec 参数迁移：https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-trtexec.html
