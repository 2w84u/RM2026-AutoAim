# Jetson / CUDA DNN 构建说明

## 后端选择规则

推理后端由**模型文件后缀**决定，编译期开关只控制哪些后端被编进二进制：

| 模型后缀 | 后端 | 需要 |
|---|---|---|
| `.onnx` | OpenCV DNN | 任意构建；`device: GPU` 时使用 CUDA FP16 |
| `.xml` + `.bin` | OpenVINO Runtime | 配置时 `-DENABLE_OPENVINO=ON` 且找到 OpenVINO |

Jetson（ARM64）没有官方 OpenVINO GPU 插件，所以想用 Orin 的 GPU 必须同时满足：

1. 系统 OpenCV 带 CUDA 模块（`cv::cuda::getCudaEnabledDeviceCount() > 0`）；
2. 配置项 `ENABLE_CUDA_DNN=ON`（默认开启）；
3. YAML 中 `device: GPU`；
4. 模型为 `.onnx`。

## 无 OpenVINO 构建

```bash
cmake -B build-jetson -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENVINO=OFF
cmake --build build-jetson -j$(nproc)
```

- `find_package(OpenVINO QUIET)`：找不到时自动关闭 OpenVINO 路径，不会配置失败。
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
