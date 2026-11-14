#!/usr/bin/env bash
# x86 -> Jetson NX 同步：源码、0708.onnx、脚本。默认不覆盖 NX 上已有的配置与 Engine。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

WITH_ENGINE=0
OVERWRITE_CONFIG=0
DRY_RUN=0

usage() {
  cat <<'EOF'
用法: deploy_nx.sh <user@host> <远程目录> [选项]
  --with-engine       同时同步 *.engine（默认不同步，NX 本机构建）
  --overwrite-config  覆盖 NX 上已有的 configs/*.yaml（默认 --ignore-existing）
  --dry-run           只显示会传输什么
  -h, --help          显示帮助

示例: ./scripts/trt/deploy_nx.sh nvidia@192.168.1.50 ~/RM2026-AutoAim --dry-run
EOF
}

log() { printf '[deploy_nx] %s\n' "$*"; }
die() { printf '[deploy_nx] ERROR: %s\n' "$*" >&2; exit 1; }

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-engine) WITH_ENGINE=1; shift ;;
    --overwrite-config) OVERWRITE_CONFIG=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) usage; die "未知参数: $1" ;;
    *) positional+=("$1"); shift ;;
  esac
done

[[ ${#positional[@]} -eq 2 ]] || { usage; die "需要 <user@host> 和 <远程目录>"; }
TARGET="${positional[0]}"
REMOTE_DIR="${positional[1]}"

[[ -s "$REPO_ROOT/assets/0708.onnx" ]] || \
  log "警告：本地缺少 assets/0708.onnx，先运行 scripts/trt/prepare_model.sh"

log "目标：$TARGET:$REMOTE_DIR"
log "同步：tasks tools io src sp_msgs calibration tests scripts assets docs CMakeLists.txt"

if ! command -v rsync >/dev/null 2>&1; then
  log "本机没有 rsync，请手动执行："
  cat <<EOF
  ssh $TARGET 'mkdir -p $REMOTE_DIR'
  scp -r $REPO_ROOT/{tasks,tools,io,src,sp_msgs,calibration,tests,scripts,configs} \\
      $REPO_ROOT/CMakeLists.txt $REPO_ROOT/CMakePresets.json $TARGET:$REMOTE_DIR/
  scp $REPO_ROOT/assets/0708.onnx $TARGET:$REMOTE_DIR/assets/
EOF
  log "注意：手动 scp configs 会覆盖 NX 上的配置，请先备份"
  exit 0
fi

main_opts=(-av --human-readable)
[[ "$DRY_RUN" == "1" ]] && main_opts+=(--dry-run)
main_opts+=(
  --exclude '.git/'
  --exclude 'build/'
  --exclude 'build-*/'
  --exclude 'imgs/'
  --exclude '.venv-ir2onnx/'
  --exclude '__pycache__/'
  --exclude '*.part'
  --exclude '*.build.log'
  --exclude 'configs/'
)
[[ "$WITH_ENGINE" == "1" ]] || main_opts+=(--exclude '*.engine')

log "同步源码与模型（配置与 Engine 除外）"
rsync "${main_opts[@]}" "$REPO_ROOT/" "$TARGET:$REMOTE_DIR/"

cfg_opts=(-av --human-readable --include '*/' --include '*.yaml' --exclude '*')
[[ "$OVERWRITE_CONFIG" == "1" ]] || cfg_opts+=(--ignore-existing)
[[ "$DRY_RUN" == "1" ]] && cfg_opts+=(--dry-run)

log "同步 configs/*.yaml（默认只补齐 NX 上缺失的文件）"
rsync "${cfg_opts[@]}" "$REPO_ROOT/configs/" "$TARGET:$REMOTE_DIR/configs/"

log "完成。NX 上继续执行："
cat <<EOF
  ssh $TARGET
  cd $REMOTE_DIR
  ./scripts/trt/build_engine.sh            # 首次：ONNX -> assets/0708_fp16.engine
  cmake -S . -B build -DENABLE_TENSORRT=ON -DENABLE_OPENVINO=OFF
  cmake --build build -j\$(nproc)
  # ./build/standard_mpc configs/jetson.yaml（配置里 yolo_name: yolov5_trt）
EOF
