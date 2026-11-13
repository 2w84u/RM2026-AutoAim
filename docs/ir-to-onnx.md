# IR → ONNX 反向转换详细流程（方案 2）

> 目标：把 `assets/` 里只有 OpenVINO IR（`.xml` + `.bin`）的 YOLO 模型转回 ONNX，
> 以便在 Jetson 上用 OpenCV DNN（CUDA）运行。
>
> 仓库已提供转换脚本：`tools/ir2onnx.py`
> 建议在 **x86 Ubuntu 22.04** 上完成转换与验证，Jetson 只做最终部署验证。

## 0 先明确“有损”在哪里

| 来源 | 说明 | 影响 |
|---|---|---|
| 卷积权重 | IR 里本来就是 **INT8 量化**权重（`yolo11.xml`：96 个 `i8` 张量，≈2.5 MB），图中用 `Convert(i8→f32) → Multiply(per-channel scale)` 反量化 | 只能还原“量化后”的精度，不会恢复原始 FP32 训练权重 |
| 激活量化 | IR 中有 110 个 `FakeQuantize`（`levels=256`） | 默认 `--fq drop` 时忽略模拟量化，通常只带来很小误差；`--fq exact` 可保留 |
| 图结构 | IR 已把 NMS 之前的 decode（DFL、anchor、stride）固化进图 | 输出就是 `(1, 4+nc+8, N)`，与仓库适配器一致，不需要额外解码 |

**结论：转换后是“量化版浮点模型”，精度通常接近原 INT8 模型，低于原 FP32 模型。**

## 1 环境准备（x86）

```bash
python3 -m venv ~/ir2onnx && source ~/ir2onnx/bin/activate
pip install numpy onnx onnxruntime openvino==2024.6.0 onnxsim
```

- `openvino` 只用于读取 IR 做数值对齐，版本与生成 IR 时一致最稳（本仓库为 2024.6）
- Jetson 上没有官方 x86 包时不必装，转换产物直接拷贝过去即可

## 2 转换步骤

### 2.1 自检（不需要 onnx，可直接在任何机器跑）

```bash
python3 tools/ir2onnx.py assets/yolo11.xml --inspect
```

本仓库四个模型的自检结果（已在本机验证通过）：

| 模型 | 输入 | 输出 | Conv 数 | 权重+常量 |
|---|---|---|---|---|
| `yolo11.xml` | 1×3×640×640 | 1×**50**×8400（38 类 + 4 关键点） | 97 | ≈12.9 MB |
| `yolo11_buff_int8.xml` | 1×3×640×640 | 1×**17**×8400（1 类 + 6 关键点） | 97 | ≈12.9 MB |
| `yolov8.xml` | 1×3×**416**×416 | 1×**14**×3549（2 类 + 4 关键点） | 92 | ≈11.5 MB |
| `yolov5.xml` | 1×3×640×640 | 1×**25200×22**（旧格式） | 61 | ≈5.6 MB |

自检必须没有 `未支持算子` 和 `权重无法折叠为常量` 两类告警。

### 2.2 转换

```bash
mkdir -p out
python3 tools/ir2onnx.py assets/yolo11.xml out/yolo11.onnx --fq drop
# 需要更接近 IR 数值时：
python3 tools/ir2onnx.py assets/yolo11.xml out/yolo11_exact.onnx --fq exact
```

脚本会调用 `onnx.checker.check_model()` 并在成功时打印节点数、initializer 数和文件大小。

## 3 转换原理（排错时看）

### 3.1 IR 文件结构

- `.xml`：`<layers>` 中每个 `<layer id type name>` 有 `<data>`（属性）、`<input><port>`、`<output><port>`；
  `<edges>` 描述 `from-layer/from-port → to-layer/to-port`
- `.bin`：所有 `Const` 张量的原始字节，XML 中通过 `offset` + `size` 定位；
  `element_type` 为 `f32/f16/i8/u8/i32/i64/boolean`

### 3.2 权重反量化链（必须折叠）

```
Const(i8, [16,3,3,3]) → Convert(to f32) → Multiply(Const f32 [16,1,1,1]) → Convolution
```

`tools/ir2onnx.py` 的 `const_value()` 会递归求值所有输入均为常量的子图
（`Convert/Multiply/Add/Subtract/Divide/Reshape/Transpose/Concat/Gather/ShapeOf/
Slice/StridedSlice/Mod/Greater`），把整条链折叠成**一个 f32 initializer**。
若这一步失败，Conv 的权重就不是常量，OpenCV DNN 无法加载。

### 3.3 静态 shape 分支

DFL/decode 部分包含 `ShapeOf → Gather → Mod → Greater → StridedSlice` 的 shape 计算。
由于输入固定为 1×3×H×W，脚本直接用 XML 里每个 port 的 `<dim>` 作为 `ShapeOf` 的值，
整条分支会折叠为常量，最终只剩 `Reshape` 需要的常量 shape。

### 3.4 FakeQuantize

- `--fq drop`（默认）：视为恒等，直接让输出复用输入 tensor（OpenCV DNN 兼容性最好）
- `--fq exact`：按 OpenVINO 语义展开为
  `Clip((x-low)/step) → Round → *step_out + output_low`
  （需要 OpenCV DNN 支持 `Round`，若加载失败就换回 `drop`）

## 4 算子映射表

脚本已覆盖这四个 IR 中出现的全部算子：

| IR 算子 | ONNX 实现 | 关键属性/注意事项 |
|---|---|---|
| `Parameter` / `Result` | 图输入 / 输出 | 输入名固定为 `input`，f32 NCHW |
| `Const` | initializer | 按 `offset/size` 读取，按内容去重 |
| `Convolution` | `Conv` | `pads = pads_begin + pads_end`（ONNX 顺序 `[t,l,b,r]`） |
| `GroupConvolution` | `Conv(group=G)` | 权重 `[G, C/G, kH, kW]` → reshape 为 `[G*(C/G), C/G, kH, kW]` |
| `Convert` | `Cast` | 只在非常量分支出现 |
| `FakeQuantize` | 恒等 / 算术展开 | 见 3.4 |
| `Add/Subtract/Multiply/Divide` | 同名 | numpy 广播 |
| `Sigmoid` | `Sigmoid` | — |
| `Swish` | `Sigmoid` + `Mul` | `beta≠1` 时先乘 beta |
| `ReLU` | `Relu` | — |
| `HSigmoid` | `Add(3) → Clip(0,6) → Div(6)` | 不用 HardSigmoid，兼容性更好 |
| `HSwish` | `x * HSigmoid(x)` | 同上 |
| `Concat` | `Concat` | `axis` 属性 |
| `Reshape` | `Reshape` | 直接使用 XML 输出 dim，规避 `special_zero` |
| `VariadicSplit` | `Split` | `split` 作为第二个输入（opset 13） |
| `MaxPool` | `MaxPool` | `rounding_type=ceil → ceil_mode=1` |
| `MatMul` | `MatMul` | `transpose_a/b` 转成前置 `Transpose`（4D 用 `[0,1,3,2]`） |
| `SoftMax` | `Softmax` | `axis` 属性（本仓库为 -1） |
| `Interpolate` | `Resize` | `nearest` + `asymmetric` + `floor`；scales 按 axes 展开到整图 rank |
| `Transpose` | `Transpose` | `perm` 来自常量输入 |
| `StridedSlice` | `Slice` | 掩码为 1 的轴直接跳过 |
| `Slice` | `Slice` | IR 输入顺序 `start, stop, step, axes` → ONNX `starts, ends, axes, steps` |
| `Mod` | `Mod(fmod=0)` | 整数取模（shape 计算） |
| `Greater` | `Greater` | — |
| `Gather` | `Gather` | `axis`、`batch_dims` |
| `ShapeOf` | `Shape`(+`Cast`) | 静态 shape 下会被折叠掉 |
| `ReduceMean` | `ReduceMean` | `axes` 用属性（opset ≤ 17），`keepdims` |

> 若还有其他模型出现新算子，只需在 `SUPPORTED_EMIT` 与 `_emit()` 中补一项。

## 5 验证（三级，必须逐级通过）

### 5.1 IR vs ONNX 数值对齐

```python
import numpy as np, openvino as ov, onnxruntime as ort

x = np.random.rand(1, 3, 640, 640).astype(np.float32) * 255

ir_out = ov.Core().compile_model('assets/yolo11.xml', 'CPU')([x])[0]
sess = ort.InferenceSession('out/yolo11.onnx', providers=['CPUExecutionProvider'])
onnx_out = sess.run(None, {'input': x})[0]

print('shape:', ir_out.shape, onnx_out.shape)      # 都应为 (1, 50, 8400)
print('max abs diff:', np.abs(ir_out - onnx_out).max())
```

判定标准：

- `--fq exact`：`max abs diff` 应在 `1e-4 ~ 1e-3` 量级
- `--fq drop`：差异略大（量化舍入），建议再比较**检测结果**：
  取两条输出置信度 > 0.7 的列，比较框的 IoU 与类别 id；应完全一致或 IoU > 0.95
- 若差异很大：检查 `Resize` 的插值模式、`MatMul` 转置、`pads` 顺序、`Split` 轴

### 5.2 OpenCV DNN 加载

```python
import cv2, numpy as np
net = cv2.dnn.readNetFromONNX('out/yolo11.onnx')
img = np.zeros((640, 640, 3), np.uint8)
net.setInput(cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True))
print(net.forward().shape)   # 期望 (1, 50, 8400)
```

- 在 x86 上只验证“能加载并前向”
- 最终目标机是 Jetson 的系统 OpenCV 4.10（带 CUDA），务必在板子上再跑一次
- 若报 `Unsupported layer`：先试 `onnxsim out/yolo11.onnx out/yolo11_sim.onnx`，
  再检查是否用了 `Round`（换 `--fq drop`）

### 5.3 接入本仓库

```bash
cp out/yolo11.onnx assets/yolo11_dnn.onnx
```

配置（参考 `configs/jetson.yaml`）：

```yaml
yolo_name: yolo11
yolo11_model_path: assets/yolo11_dnn.onnx
device: GPU
class_num: 38        # yolo11.xml 是 38 类
```

```bash
./build/detector_video_test --config-path=configs/jetson.yaml match.avi --tradition=false
```

日志出现 `[infer] OpenCV DNN CUDA FP16 backend enabled` 即成功；再对比
IR（OpenVINO CPU）与 ONNX（CUDA）在同一段视频上的检测框，确认无系统偏移。

## 6 常见问题

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `未支持算子: {...}` | 脚本缺映射 | 在 `SUPPORTED_EMIT`/`_emit()` 补；先看 `--inspect` |
| `Conv ... 权重无法折叠为常量` | 反量化链里有未实现算子 | 检查该 Conv 权重输入的 `Convert/Multiply` 链 |
| `无法解析 tensor` | 某常量分支未折叠（多为 shape 分支） | 检查 `ShapeOf/Gather/StridedSlice/Mod/Greater` |
| ONNX 能建但 ORT 报 shape 错 | `Reshape/Split` 常量不对 | 用 XML 输出 dim 作为 Reshape 目标 |
| 输出通道数不对 | `class_num` 或模型选错 | 用 `--inspect` 确认输出 shape |
| OpenCV 输出全 0/异常 | 某层 OpenCV 不支持或语义不同 | 逐层对比（见 6.1） |

### 6.1 逐层定位

把中间 tensor 加为 IR 的输出，再与 ONNXRuntime 的同名中间结果比较：

```python
import openvino as ov
m = ov.Core().read_model('assets/yolo11.xml')
m.add_outputs(['<XML 中 port 的 names 属性>'])   # 例如 '1351'
cm = ov.Core().compile_model(m, 'CPU')
```

ONNX 侧把对应节点输出加入 `graph.output` 即可（或用 `onnx.helper` 临时改图）。

## 7 风险与工作量

| 项 | 评估 |
|---|---|
| 本机已完成的验证 | IR 解析、常量折叠、算子覆盖、IO 形状（`--inspect` 全绿） |
| 还需在 x86 完成 | 安装 onnx/openvino → 实际生成 ONNX → 数值对齐 → `onnxsim` |
| 预计调试时间 | 0.5 ~ 1 天（主要耗在数值对齐与 OpenCV DNN 算子兼容） |
| 主要风险 | ① FQ 丢弃导致精度略降；② OpenCV DNN 对部分算子支持有限；③ 仅能还原量化模型 |
| 更优替代 | 找到原始 `.pt` 直接 `export(format="onnx")`，无损且几分钟完成 |

## 8 已知限制

- `yolov5.xml` 输出是旧格式 `(1, 25200, 22)`，仓库 `yolov5` 适配器的 `parse()` 按
  固定列号解析，可继续用，但建议迁移到 yolo11 适配器
- `yolov8.xml` 输入是 416，适配器也硬编码 416；若要 640 需改代码或重新训练
- 转换后的模型仍需要正确设置 `class_num`（38 / 2 / 3 等），否则 `parse()` 解列会错位
