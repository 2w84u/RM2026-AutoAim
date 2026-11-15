# Jetson / CUDA DNN 构建说明

当前三模型的性能优化、TensorRT FP16 接入、INT8 校准与重训决策，见 [Orin NX YOLO 优化实施路线](jetson-yolo-optimization-roadmap.md)。

## 后端选择规则

模型后缀和 `yolo_name` 决定运行时路径。CMake 自动查找 OpenVINO 和 TensorRT：
Jetson 上两者都存在时优先选择 TensorRT，其他平台优先选择 OpenVINO；只找到一个时
选择该后端，都不可用时配置直接报错。

| 模型后缀 | 后端 | 需要 |
|---|---|---|
| `.onnx` | OpenCV DNN | 已找到 OpenVINO 或 TensorRT 的构建；`device: GPU` 请求 CUDA FP16 |
| `.xml` + `.bin` | OpenVINO Runtime | CMake 选择 OpenVINO |
| `.engine` | TensorRT | CMake 选择 TensorRT，且 `yolo_name: yolov5_trt` |

Jetson（ARM64）没有官方 OpenVINO GPU 插件，所以想用 Orin 的 GPU 必须同时满足：

1. 系统 OpenCV 带 CUDA 模块（`cv::cuda::getCudaEnabledDeviceCount() > 0`）；
2. 配置项 `ENABLE_CUDA_DNN=ON`（默认开启）；
3. YAML 中 `device: GPU`；
4. 模型为 `.onnx`。

## 自动选择后端构建

```bash
cmake -S . -B build-jetson
cmake --build build-jetson --parallel 8
```

- 配置日志显示实际选择的后端；OpenVINO 和 TensorRT 都不可用时会报错。
- 若配置文件里仍是 `.xml` 模型，运行时会抛出明确异常，提示改用 `.onnx`。
- `device: CPU` 时使用 OpenCV DNN CPU 后端，不需要 CUDA。

## `class_num` 配置项

`yolo11` / `yolov8` 适配器可以根据模型输出动态解析类别数：

```yaml
class_num: 3   # 输出通道 = 4 + class_num + 8(4 个关键点)
```

不配置时分别使用默认值（yolo11: 38，yolov8: 2），保持旧模型行为。

## 本机实测（Jetson Orin NX 16GB，15W 模式）

`assets/best2-sim.onnx`，640×640，20 次平均：

| 后端 | 单帧耗时 |
|---|---|
| OpenCV DNN CPU | ≈ 78 ms |
| OpenCV DNN CUDA FP16 | ≈ 15.5 ms |

启动时可能出现 `cuDNN reports version 90.3 which is not compatible with the version 9.3`
警告，属于 OpenCV 版本检查误报，不影响推理。若需更高性能，建议
`sudo nvpmodel -m 0 && sudo jetson_clocks` 切到 MAXN 模式，或后续移植 TensorRT 执行器。

## 参考配置文件

`configs/jetson.yaml`：ONNX 模型 + `device: GPU` 的示例。
