# 测试索引

所有交互/硬件测试均按目标单独编译，不随普通构建或 Debug 构建自动编译，
也不会被 CTest 自动启动。可执行文件仍位于 `build/<目标名>`。

```bash
cmake --build --preset default --target gimbal_test
./build/gimbal_test configs/sentry.yaml
```

在项目根目录使用 `default` 构建预设，默认 8 路并行；可用 `--parallel N` 临时覆盖。

| 目录 | 目标 | 用途 |
|---|---|---|
| `vision/` | `detector_verify` | 统一验证图片、目录、视频和实时相机；支持 JSON 输出 |
| `vision/` | `camera_test` | 工业相机采集、帧率和可选预览 |
| `vision/` | `auto_aim_test` | 使用录像和配套姿态文件回放自瞄 |
| `vision/` | `auto_buff_test` | 打符录像回放 |
| `hardware/` | `gimbal_test` | Gimbal AB 协议收发、模式/姿态/弹速观测，可选开火测试 |
| `hardware/` | `gimbal_feedback_test` | 实机原始四元数转欧拉角、三维姿态和 UDP 曲线显示，仅接收反馈 |
| `hardware/` | `dm_test` | 达妙 IMU 姿态观测 |
| `planning/` | `planner_test_offline` | 合成目标的 MPC 规划，不连接云台 |
| `planning/` | `planner_test` | 合成目标的实车 MPC 跟随 |
| `ros2/` | `topic_loop_test` | 自定义消息发布/订阅链路 |
| `ros2/` | `cmd_vel_publisher_test` | 导航速度话题发布 |
| `ros2/` | `cmd_vel_subscriber_test` | 导航速度话题接收 |

`ros2/` 的三个目标要求配置 CMake 前先加载 ROS2 和已编译的 `sp_msgs` 工作空间。
实时识别可直接使用 `detector_verify --camera --config-path=configs/sentry.yaml`；
详细选项见 [模型验证说明](../docs/ir-to-onnx.md)。

## 原始四元数实机诊断

这个入口是用于连接真实云台板的功能工具，不是模拟或自动验证代码。
直接读取 AB 上行帧中的原始 `q[4]`，按 **w,x,y,z、B→W、ZYX** 转换为
yaw/pitch/roll，仅接收反馈，不发送控制或开火指令。

在项目根目录运行：

```bash
cmake --preset default
cmake --build --preset default --target gimbal_feedback_test
./build/gimbal_feedback_test configs/sentry.yaml
```

默认读取配置文件中的 `com_port`，也可显式指定实际串口：

```bash
./build/gimbal_feedback_test --port=/dev/ttyACM0
```

窗口显示原始 wxyz、归一化前模长、ZYX 欧拉角、板端 yaw/pitch 及二者环绕角差，
右侧显示三维立方体和坐标轴。灰色为世界轴，红/绿/蓝分别是机体 X/Y/Z。
按 q 或 Esc 退出；无有效四元数或反馈已超时时不绘制姿态。

同时向本机 UDP **9870** 发送 JSON，供 PlotJuggler 接收。
可绘制 `q_yaw_deg/q_pitch_deg/q_roll_deg`、`feedback_yaw_deg/feedback_pitch_deg`、
`yaw_error_deg/pitch_error_deg`、`quaternion_norm`、`q_valid/q_unit`、`rx_hz/crc_errors/fresh`。
`--no-display` 关闭窗口，保留终端和 UDP 输出；`--plot-port` 可修改曲线端口。

测试时由此工具独占云台串口。按提供的电控分析，反馈约 100 Hz，`mode=0` 也会回传，
头部 IMU 无效时会停发，弹速/弹数尚未接入因而回传 0。
欧拉角差接近 0 表示两端转换约定一致；安装方向仍需结合实际转动观察。
该工具不切换自瞄或标定采集的姿态来源。

## 无硬件回归

`contracts/` 仅包含三个可自动执行的测试：装甲板几何、导航 AB 封包/CRC16、
伪终端串口断线重连。独立构建不需要相机 SDK、ROS2、Ceres 或推理模型，
还会编译真实 UART 实现以检查接口。

```bash
cmake -S tests/contracts -B build/contracts -DCMAKE_BUILD_TYPE=Debug
cmake --build build/contracts --parallel 8
ctest --test-dir build/contracts --output-on-failure
```

也可在主项目配置时传入 `-DBUILD_TESTING=ON`，正常构建后使用
`ctest --test-dir build --output-on-failure`。两种方式复用同一份目标定义；
Release 下也保留契约断言。

## 归档

`archive/legacy_cboard/` 保存旧 C 板 CAN/UART 测试；`archive/experiments/` 保存重复检测、
旧相机线程、整机拼接和 IMU 转发等实验。不再为这些文件注册目标，
`ENABLE_LEGACY_CBOARD=ON` 也不会自动恢复它们。

| 原入口 | 当前替代 |
|---|---|
| `camera_detect_test`、`detector_video_test`、`usbcamera_detect_test` | `detector_verify` 的相机/视频输入 |
| `camera_thread_test`、USB 相机示例 | 工业相机先用 `camera_test`；特殊 USB 实验源码归档 |
| `auto_aim_test_new`、`minimum_vision_system` | `auto_aim_debug_mpc` 或 `sentry` |
| `standard_mpc_test` | `standard_mpc`，旧 IMU 转发代码仅归档 |
| `fire_test`、`gimbal_control_test`、`nav_to_gimbal_test` | `gimbal_test`、`planner_test`；特殊轨迹实验源码归档 |
| `publish_test`、`subscribe_test` | `topic_loop_test` |
| 旧 C 板测试 | 当前通信使用 `gimbal_test`，旧协议源码仅供查阅 |
