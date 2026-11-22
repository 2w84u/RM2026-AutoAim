# 历史应用入口

本目录保留旧步兵、无人机、非 MPC 调试和旧 C 板哨兵代码，供历史行为和迁移对照。
这些文件不再注册为 CMake 目标；开启 `ENABLE_LEGACY_CBOARD` 也不会将它们加入构建。

`sentry_test_new.cpp` 是旧哨兵 MPC 实验，包含固定弹速等实验逻辑，也一并归档。

当前可构建的应用入口位于上一级目录：`sentry.cpp`、`standard_mpc.cpp`、
`auto_aim_debug_mpc.cpp` 和 `auto_buff_debug_mpc.cpp`。
