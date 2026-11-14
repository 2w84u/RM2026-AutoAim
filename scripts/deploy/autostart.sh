#!/usr/bin/env bash

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root" || exit 1
mkdir -p logs
sleep 5

printf -v run_cmd 'cd %q && bash %q' "$project_root" "$project_root/scripts/deploy/watchdog.sh"

# 默认弹出可见终端，便于随时手动停止。
if [ -n "$DISPLAY" ]; then
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal -- bash -lc "$run_cmd; exec bash"
        exit $?
    elif command -v xfce4-terminal >/dev/null 2>&1; then
        xfce4-terminal --hold --execute bash -lc "$run_cmd"
        exit $?
    elif command -v konsole >/dev/null 2>&1; then
        konsole --hold -e bash -lc "$run_cmd"
        exit $?
    elif command -v x-terminal-emulator >/dev/null 2>&1; then
        x-terminal-emulator -e bash -lc "$run_cmd"
        exit $?
    fi
fi

screen -S autoaim -L -Logfile "logs/$(date '+%Y-%m-%d_%H-%M-%S').screenlog" \
    -d -m bash -lc "$run_cmd"
