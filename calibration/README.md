# 相机标定

当前使用 checker 棋盘格：**8 列 × 11 行方格，方格边长 15 mm**。
OpenCV 配置填写内角点数，因此 `configs/calibration.yaml` 为：

```yaml
pattern_cols: 7
pattern_rows: 10
center_distance_mm: 15
```

相机内参、两种手眼标定和采集预览共用棋盘格检测与亚像素角点优化。
更换标定板时，修改上述内角点数量和间距。

## 同时采集图片和姿态

在 `configs/calibration.yaml` 中将 `com_port` 设置为云台实际串口，然后运行：

```bash
cmake --build --preset default --target capture
./build/capture configs/calibration.yaml --output-folder=assets/img_with_q
```

按 **s** 保存当前图像及其对应姿态，按 **q** 退出。每组数据保存为同编号的
`1.jpg` / `1.txt`、`2.jpg` / `2.txt`……；TXT 内容是 `w x y z`，重复采集会跳过已有编号。
只保存识别到完整 7×10 内角点、且有有效姿态的帧；写入失败时会清理本次不完整样本。
采集程序只读取云台反馈，不发送云台控制指令。

当前姿态沿用自瞄的 Gimbal yaw/pitch 合成四元数，**roll 为 0**，尚未使用 AB 包内的 `q[4]`。
姿态按图像采集时间查询，允许与相邻姿态样本最多相差 50 ms；断线、缺数据或超时会拒绝保存，
不会把无数据时的单位四元数当作标定数据。时间戳来自主机接收时间，属于软件时间匹配。
采样时保持云台短暂停稳，硬件曝光时间与 MCU 时钟同步仍需后续联调。

只采内参图片时可以跳过云台：

```bash
./build/capture configs/calibration.yaml --images-only --output-folder=assets/calibration_images
```

该模式只保存 JPG，不连接串口，也不生成四元数文件。

## 相机内参

只需要标定图片，不需要四元数。在项目根目录运行：

```bash
cmake --build --preset default --target calibrate_camera
./build/calibrate_camera assets/calibration_images --config-path=configs/calibration.yaml
```

图片连续命名为 `1.jpg`、`2.jpg`……，缺号即停止读取。
程序逐张显示角点检测结果，按键进入下一张。
终端输出的 `camera_matrix`、`distort_coeffs` 需要手动填入机器人配置，例如 `configs/sentry.yaml`。

## 手眼标定

`calibrate_handeye` / `calibrate_robotworld_handeye` 还需要每张图片对应的同名 `.txt`，
内容为拍摄时的四元数，顺序是 `w x y z`；配置中需先填写相机内参、畸变参数和 `R_gimbal2imubody`。

`capture` 已改用当前 Gimbal AB 接口，不再需要开启 `ENABLE_LEGACY_CBOARD`。
它与其它标定工具一样，按目标手动编译。
