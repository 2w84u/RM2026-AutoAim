#!/usr/bin/env bash

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root" || exit 1
target_program="$project_root/build/sentry"

if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi
ros_workspace_setup="${SPVISION_ROS_SETUP:-$HOME/ros_ws/install/setup.bash}"
if [ -f "$ros_workspace_setup" ]; then
    source "$ros_workspace_setup"
fi

if [ ! -x "$target_program" ]; then
    echo "[Watchdog] 找不到可执行程序: $target_program"
    exit 1
fi

child_pid=""
shutdown() {
    trap - SIGINT SIGTERM
    if [ -n "$child_pid" ]; then
        kill "$child_pid" 2>/dev/null
        wait "$child_pid" 2>/dev/null
    fi
    exit 0
}
trap shutdown SIGINT SIGTERM

while true; do
    echo "[Watchdog] $(date '+%Y-%m-%d %H:%M:%S') 启动 $target_program"
    "$target_program" "$@" &
    child_pid=$!
    wait "$child_pid"
    exit_code=$?
    child_pid=""
    echo "[Watchdog] 程序退出，退出码: $exit_code；3 秒后重启"
    sleep 3 &
    child_pid=$!
    wait "$child_pid"
    child_pid=""
done
