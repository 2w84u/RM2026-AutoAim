# 当前 Gimbal AB 协议与自瞄数据来源

本页按现有代码整理，用于与电控核对。此次目录整理没有修改协议布局、单位或自瞄输入。
结构以 [`io/vision_protocol.hpp`](../../io/vision_protocol.hpp) 为准，
收发实现位于 [`io/gimbal/gimbal.cpp`](../../io/gimbal/gimbal.cpp)。

## 编译边界

- 默认编译 `Gimbal`、`NavSerial` 和 `DM_IMU`。
- `io/cboard.cpp`、`io/cboard_uart.cpp` 和旧 `multithread/commandgener.cpp` 仅在
  `ENABLE_LEGACY_CBOARD=ON` 时编译，默认关闭。
- 当前哨兵和三个 MPC 入口均使用 `Gimbal`，不使用 `CBoard`/`CBoardUART`。
- `src/archive/`、`tests/archive/` 不注册编译目标。

## 线格式

三种数据帧均为紧密打包、小端编码，头部是 ASCII `A`、`B`（`0x41 0x42`），
尾部是 `tools/crc.cpp` 实现的 CRC16，覆盖除末尾 CRC16 本身之外的全部字节。
云台帧使用 IEEE-754 `float32`，导航速度使用 `float64`。

| 方向 | 长度 | 字段顺序 |
|---|---|---|
| 云台 → 视觉 | 43 字节 | `head[2]`、`mode:u8`、`q[4]:f32`、`yaw`、`yaw_vel`、`pitch`、`pitch_vel`、`bullet_speed`、`bullet_count:u16`、`crc16:u16` |
| 视觉 → 云台 | 29 字节 | `head[2]`、`mode:u8`、`yaw`、`yaw_vel`、`yaw_acc`、`pitch`、`pitch_vel`、`pitch_acc`、`crc16:u16` |
| 视觉 → 导航 | 55 字节 | `head[2]`、`mode:u8`、`empty1:u8`、`empty2:u8`、六个 `f64` 速度分量、`crc16:u16` |

现有代码将角度/角速度/角加速度分别按 rad、rad/s、rad/s² 使用，弹速按 m/s 使用。
云台接收 `mode`：0 空闲、1 自瞄、2 小符、3 大符；视觉发送 `mode`：0 不控制、
1 控制不开火、2 控制且开火。两个方向的枚举含义不同。

云台串口由 `com_port` 指定，导航串口由 `nav_com_port` 指定，两者必须不同。
Gimbal 还接收 `0xA5` 开头的裁判系统帧；上述 AB 表只描述自瞄/导航数据。
完整裁判系统资料保存在同目录的 `communication_protocol.pdf` 和 `protocol.txt`，
ROS 消息见 [sp-msgs.md](sp-msgs.md)。

## 下次调整自瞄输入时的入口

| 数据 | 当前来源/处理 |
|---|---|
| 图像与采集时间 | `io::Camera::read()`，相机型号/参数来自 YAML |
| 识别结果 | `YOLO::detect()` → `Tracker::track()` |
| 解算姿态 | `Gimbal::q(timestamp)`；当前从 AB 反馈的 yaw/pitch 合成四元数，roll 固定为 0 |
| 标定采集姿态 | `Gimbal::try_q(timestamp)`，沿用相同姿态来源；无有效数据或时间差超出 50 ms 时拒绝配对，不返回默认姿态 |
| 包内四元数 | `GimbalToVision::q[4]`；自瞄/采集尚未使用，`gimbal_feedback_test` 实机工具直接读取并可视化 |
| 自瞄 GUI | `auto_aim_debug_mpc` 通过 `Gimbal::feedback()` 显示原始四元数、实际解算姿态、观测目标与期望云台姿态；默认无串口发送，`--enable-control` 开启跟随、固定不开火 |
| 弹速/计数/关节状态 | `Gimbal::state()`；弹速用于哨兵 Aimer 和 MPC Planner |
| 达妙 IMU | 哨兵创建并传给 `SentryDecider`，但当前导航状态机只保存指针、没有读取；自瞄 Solver 也未使用它 |
| 哨兵导航指令 | 来自 `SentryDecider` 的时间/血量状态机，当前主循环未采用 ROS `cmd_vel` |
| 输出 | 哨兵通过 `gimbal.send(Command)` 发角度；MPC 控制入口还发送角速度和角加速度；GUI 仅在显式指定 `--enable-control` 时发送跟随/停控帧，开火位固定关闭 |

后续替换数据来源时，需要一起确认姿态坐标系、roll 是否保留、时间戳对齐和弹速来源。
当前现状已定位，本轮没有切换这些输入。

## 板端回传说明

原始四元数实机诊断入口为 [`gimbal_feedback_test`](../../tests/README.md#原始四元数实机诊断)，
支持原始 wxyz 转 ZYX 欧拉角、三维姿态及 UDP 曲线，仅接收云台板反馈。

按提供的电控分析，板端经 USB CDC ACM 以约 100 Hz 回传头部 IMU 的 B→W 姿态，
yaw/pitch 同样由此四元数推导，弹速/弹数尚未接入。实际串口需要核对 `ttyACM*`
或稳定设备别名，不能直接沿用示例 `ttyUSB0` 而不检查连接对象。
