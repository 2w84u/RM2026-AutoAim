# RM2026-AutoAim

当前默认构建哨兵及 MPC 版本，通信使用 `io/gimbal/` 的 AB 串口协议。
上游算法介绍保存在 [docs/upstream-guide.md](docs/upstream-guide.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `src/` | `sentry`、`standard_mpc`、`auto_aim_debug_mpc`、`auto_buff_debug_mpc` 入口 |
| `src/archive/` | 旧步兵、无人机、旧 C 板哨兵入口，不注册编译目标 |
| `tasks/`、`io/`、`tools/` | 算法、设备/通信、公共 C++ 工具及模型转换脚本 |
| `tests/` | 按用途分类的有效测试，见 [测试索引](tests/README.md) |
| `tests/archive/` | 重复示例和旧协议实验，不注册编译目标 |
| `calibration/` | 标定工具，按目标单独编译，见 [棋盘格标定说明](calibration/README.md) |
| `configs/`、`assets/` | 机器人参数、模型；PlotJuggler 布局位于 `configs/plotjuggler/` |
| `scripts/setup/`、`scripts/deploy/` | 依赖安装/交换空间脚本、自启/守护脚本 |
| `docs/protocols/` | 当前 AB 协议说明、裁判协议 PDF/文本、ROS 消息说明 |
| `docs/debugging/`、`logs/`、`out/` | 历史诊断快照、运行日志、生成结果 |

## 编译与运行

在项目根目录执行。依赖和 Jetson 推理配置见 [CUDA 构建说明](docs/jetson-cuda-build.md)；
依赖安装脚本为 `scripts/setup/install_dependences.sh`。

```bash
# 哨兵需要先加载 ROS2 和已编译的 sp_msgs 消息包。
source /opt/ros/humble/setup.bash
source ~/ros_ws/install/setup.bash

cmake --preset default
cmake --build --preset default
./build/sentry configs/sentry.yaml
```

`default` 预设为 Debug，使用 `build/` 和 4 路并行；`release` 预设为 Release，使用
独立的 `build-release/` 和 8 路并行。需要 CMake 3.21 及以上，可通过 `--parallel N`
临时覆盖并行数。Release 可运行 `cmake --preset release`，随后运行
`cmake --build --preset release`。

VS Code CMake Tools 检测到 `CMakePresets.json` 时使用预设模式，不显示原来的
“Select Variant”。在命令面板选择 **CMake: Select Configure Preset**，切换
`Debug (default)` 或 `Release`，然后使用对应的 Build Preset 构建。

默认应用共四个：`sentry`、`standard_mpc`、`auto_aim_debug_mpc`、`auto_buff_debug_mpc`。
`auto_aim_debug_mpc` 是实机 GUI，可显示识别、跟踪、MPC 轨迹、原始姿态和期望姿态。
默认允许云台跟随；`--enable-control=false` 仅接收反馈，开火位始终关闭，见
[自瞄 MPC 可视化说明](docs/debugging/auto-aim-mpc-gui.md)。
云台动态响应调节使用独立的 `gimbal_mpc_response_test`，支持可调矩形轨迹、内置波形图和
实测数据导出，见 [MPC 响应测试说明](docs/debugging/gimbal-mpc-response.md)。
缺少 ROS2 或 `sp_msgs` 时跳过 `sentry`，其余 MPC 目标仍可编译。
`sp_msgs/` 是独立的 ROS2 消息包，需要先放入 ROS2 工作空间并通过 colcon 构建。

Debug 和 Release 均不会自动编译交互测试或标定工具；需要哪个就构建哪个：

```bash
cmake --build --preset default --target detector_verify
cmake --build --preset default --target gimbal_test
cmake --build --preset default --target calibrate_camera
cmake --build --preset default --target capture
./build/detector_verify --help
```

目标程序仍输出到 `build/<目标名>`。`BUILD_TESTING` 默认关闭；开启后仅将三个无硬件
契约测试加入默认构建，不会启用历史示例。可独立运行这些测试：

```bash
cmake -S tests/contracts -B build/contracts -DCMAKE_BUILD_TYPE=Debug
cmake --build build/contracts --parallel 8
ctest --test-dir build/contracts --output-on-failure
```

旧 C 板 CAN/UART 实现默认关闭。`-DENABLE_LEGACY_CBOARD=ON` 仅恢复旧传输及
`CommandGener` 实现，不会注册归档应用或测试。标定采集 `capture` 使用 Gimbal AB 接口，
可同时保存图片和对应四元数，详见 [标定采集说明](calibration/README.md)。
当前协议字段和数据来源见 [AB 协议与数据链路](docs/protocols/gimbal-ab.md)。

## 部署与输出

自启入口为 `scripts/deploy/autostart.sh`，其调用 `scripts/deploy/watchdog.sh`，
两个脚本均按自身位置寻找项目根目录。现有桌面自启项的 `Exec` 需指向新的绝对路径。
守护脚本默认加载 `~/ros_ws/install/setup.bash`，也可用 `SPVISION_ROS_SETUP` 指定消息工作空间。

模型验证结果默认写入 `out/detector_verify/`；本地相机样本收拢至 `out/camera_samples/`。
原有运行日志保留在 `logs/`。串口配置见 [串口重连说明](docs/serial-hotplug.md)，
新赛季识别替换点见 [迁移说明](docs/2027-vision-migration.md)。
