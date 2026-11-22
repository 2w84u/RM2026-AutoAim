# 自瞄 MPC 实机可视化

`auto_aim_debug_mpc` 使用真实相机、真实云台反馈、现有 YOLO / Solver / Tracker /
弹道与 MPC 规划器。云台跟随默认开启；传入 `--enable-control=false` 时只接收云台串口，
不发送控制或停控帧。所有路径的开火位都固定关闭，不转发 IMU 或导航帧。
板端处于空闲模式时同样处理图像和上行姿态。

在项目根目录运行；先关闭占用同一相机或云台串口的其他程序：

```bash
cmake --build --preset default --target auto_aim_debug_mpc
./build/auto_aim_debug_mpc configs/sentry.yaml --enable-control=false
```

NX 上使用 0708 TensorRT 模型（先按 [TensorRT 指南](../jetson_tensorrt.md) 构建）：

```bash
./build-trt/auto_aim_debug_mpc configs/jetson_trt.yaml --enable-control=false
```

TensorRT 构建在省略配置路径时也默认读取 `configs/jetson_trt.yaml`；其他构建默认读取
`configs/sentry.yaml`。显式指定的配置路径始终优先。

开启真实云台跟随（也可省略默认开启的开关）：

```bash
./build/auto_aim_debug_mpc configs/sentry.yaml --enable-control
```

控制模式下，现有规划线程约每 10 ms 发送一次 yaw/pitch、角速度和角加速度，开火位
固定为 false。板端反馈必须处于 `AUTO_AIM`（上行 mode=1），目标必须已进入
`tracking`，才会请求跟随。反馈超过 100 ms、图像超过 250 ms、姿态缺失、目标未稳定
跟踪、规划无效、模式切换或串口重连时改发停控帧。正常退出及主线程抛出异常时，
先停止规划线程并请求停控。GUI 右下角和终端显示 `TX FOLLOW` 或 `TX STOP` 的原因；
这是主机发送请求的状态，协议没有提供执行确认。

GUI 按 `q` / Esc 或终端 Ctrl+C 退出。支持窗口状态查询的后端也可关闭窗口退出；
不支持查询时会在终端提示，继续运行，不会因查询返回负值自行退出。
退出日志会注明按键、窗口关闭或 Ctrl+C 原因。可拖动窗口调整尺寸。
它覆盖相机到 MPC 的自瞄链路，可显式开启云台跟随；导航与哨兵决策不在这个入口内。

## 画面与轨迹

- 蓝色：检测出的实际装甲板关键点及类别、颜色、置信度。
- 绿色：跟踪器估计的装甲板重投影。
- 黄色：弹道飞行时间补偿后的目标预测位置，菱形标出当前参考瞄准位置。
- 紫色：MPC 规划方向在相机中的投影，十字为当前指令方向，箭头从当前云台方向
  指向期望方向。紫色路径按目标距离投影，用于观察角度指令，不表示实测弹丸轨迹。

下方两张曲线显示规划器实际计算出的 yaw / pitch 参考和 MPC 结果，单位为度。
黄色是参考，紫色是优化结果，青色横线是原始反馈换算到云台世界系后的当前角度。
横轴是同一次求解的预测时间窗 `-0.50 .. +0.49 s`，竖线 `0` 对应输出指令所在列。
这些是基于真实观测的规划结果，不是云台已经执行的运动记录。

右下角方向图从世界 Z 正侧斜俯视。青色是接收姿态的机体 X 正方向，紫色是期望
云台 X 正方向，绿色是当前观测装甲板的法向；灰色为世界 XYZ 轴。
沿 Z 负方向看时，正 yaw 为逆时针。

## Detect 分段耗时

调试入口默认启用分段计时，并打开 `Detect timing (ms)` 窗口；主画面第二行同步显示
预处理、`forward`、后处理及其他开销摘要。只观察检测耗时、不控制云台时显式运行：

```bash
./build/auto_aim_debug_mpc configs/sentry.yaml --enable-control=false
```

计时窗口约每 200 ms 更新一次，显示当前帧、最近 60 帧均值、P95、最大值及当前占比。
候选框数量按原始输出、置信度筛选、NMS、最终装甲板四步显示。
`T` 隐藏/重新打开计时窗口，`R` 清空统计；隐藏窗口时仍持续采集计时。
`--detect-timing=false` 完全关闭本入口的分段计时和计时窗口。

YOLO11、YOLOv5、YOLOv8 的 OpenCV DNN / OpenVINO 路径均包含以下阶段：

| 阶段 | 测量内容 |
|---|---|
| ROI / shape setup | ROI 选择、输入缩放尺寸计算与相关准备 |
| Resize / padding | 图像内存分配、补边、缩放 |
| Blob / tensor packing | ONNX 的 `blobFromImage`；OpenVINO 的输入张量包装 |
| Create infer request | OpenVINO 推理请求创建；OpenCV DNN 不执行此阶段 |
| Input binding | `setInput` / `set_input_tensor` 的调用耗时 |
| Network forward / infer | `net.forward()` / `infer_request.infer()` 的主机等待耗时 |
| Output view / copy | 输出张量获取、reshape 或 clone |
| Output transpose | YOLO11 / YOLOv8 输出转置 |
| Decode / score filtering | 遍历输出、找类别/置信度、还原框和物理关键点 |
| NMS | 非极大值抑制 |
| Armor / keypoint construction | 装甲板对象创建、角点排序和 ROI 偏移处理 |
| Classifier image crop | YOLOv8 的数字区域裁剪 |
| Number classification | YOLOv8 的数字分类器调用 |
| Traditional corner refinement | YOLOv5 在启用传统方法时的角点二次矫正 |
| Armor filtering / normalization | 类别、阈值等过滤与中心坐标归一化 |
| Detector debug drawing | 检测器自带的调试绘图；本入口使用 `debug=false`，因此不执行 |
| Other / dispatch | 检测封装、函数收尾和未归入上述阶段的剩余时间 |

阶段不执行时显示 `--`，不能将其理解成实际耗时为 0；各阶段统计只使用执行过该阶段的帧。
启动阶段不会被自动丢弃，初次推理较慢时可等待其移出 60 帧窗口，或按 R 重置统计。
当前阶段耗时均为互不重叠的主机墙钟区间，合计对应同一次 `detect()` 总耗时。
计时窗口、跟踪、规划、相机采集和主 GUI 绘制都不计入 `detect()`。

`forward()` 可能包含数据传输、同步、内部转换及不支持算子的 CPU 回退，不能将其当作
纯 GPU kernel 耗时；`setInput` 也不能直接命名为显存上传耗时。此实现未插入额外 CUDA
同步或 GPU 事件。后端栏明确显示请求的设备，实际后端选择另见启动日志。

公共 `YOLO` 接口提供 `set_timing_enabled(bool)` 和 `timing()`，其他入口默认不开启计时。
多线程检测器单独调用 `postprocess()` 时，统计仅涵盖后处理，不包含其外部推理过程。

## 四元数与欧拉角

四元数顺序统一为 `w x y z`；欧拉角按 ZYX 显示 `yaw / pitch / roll`，单位为度。

- `RECEIVED RAW q`：AB 包中原始四元数，显示原始模长和接收年龄。仅在转换和
  方向显示时归一化；无效四元数显示原始值及无效提示，不生成默认姿态。
- `PIPELINE @ IMAGE`：本帧解算实际使用的姿态，继续沿用 yaw/pitch 合成、roll=0
  的数据来源，并应用配置中的 IMU 安装变换。这个工具没有切换正式自瞄或 capture
  的姿态来源。
- `DESIRED GIMBAL q`：MPC 期望 yaw/pitch 转换出的期望云台姿态，roll=0。
  协议发送字段本身是角度、角速度和角加速度，这个四元数用于可视化。
- `OBSERVED ARMOR q`：当前跟踪目标对应的装甲板观测姿态，来自 PnP 与现有 yaw
  优化，表示装甲板到世界的旋转。它的轴与云台轴含义不同，不能直接把两组欧拉角
  相减当成跟踪误差。目标暂时不可见时显示 N/A，跟踪器保留的中心和朝向另行显示。

原始反馈是最新收到的 AB 数据；图像姿态按图像时间匹配。界面分别显示其年龄。
无有效匹配姿态时继续显示检测框，暂停世界坐标跟踪与规划；姿态恢复后重新建跟踪。
源图像超过 250 ms 或没有有效规划结果时，清除轨迹和指令姿态显示。

## 弹速与配置

反馈弹速为 0 或超出当前 MPC 接受的 `10 .. 25 m/s` 时，读取：

```yaml
default_bullet_speed: 22.0
```

22 m/s 沿用原 MPC 的默认值，可按实际使用弹速修改。界面明确区分 `BOARD` 和
`CONFIG DEFAULT`，同时显示板端原值；不是将配置值当成已测得的反馈。
自瞄调试入口要求配置含有这个字段，默认值也必须处于当前 MPC 接受范围内。

识别敌方颜色由 `enemy_color` 决定；模型、曝光和标定参数同样读取传入 YAML。
`yaw_offset` / `pitch_offset` 仍参与实际规划，界面会显示这两个补偿值。
`auto_fire` 不会启用这个入口的开火位，`planned fire` 只展示规划器的判定，
即使控制模式下显示 YES，发送的开火位仍然为 false。
