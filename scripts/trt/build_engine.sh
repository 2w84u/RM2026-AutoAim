#!/usr/bin/env bash
# 在 Jetson NX 上把 0708.onnx 转成 TensorRT Engine。默认 FP16，只需构建一次。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ONNX="${ONNX:-$REPO_ROOT/assets/0708.onnx}"
PRECISION="${PRECISION:-fp16}"
ENGINE="${ENGINE:-}"
TRTEXEC="${TRTEXEC:-}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
用法: build_engine.sh [选项]
  --onnx <path>       ONNX 模型（默认 assets/0708.onnx）
  --engine <path>     输出 Engine（默认 assets/0708_<precision>.engine）
  --precision fp16|fp32   （默认 fp16）
  --trtexec <path>    trtexec 路径（默认从 PATH 或 /usr/src/tensorrt/bin 查找）
  --force             已存在 Engine 时强制重建
  -h, --help          显示帮助
也可用环境变量 ONNX / ENGINE / PRECISION / TRTEXEC / FORCE=1。
EOF
}

log() { printf '[build_engine] %s\n' "$*"; }
die() { printf '[build_engine] ERROR: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) ONNX="${2:?--onnx 需要路径}"; shift 2 ;;
    --engine) ENGINE="${2:?--engine 需要路径}"; shift 2 ;;
    --precision) PRECISION="${2:?--precision 需要 fp16 或 fp32}"; shift 2 ;;
    --trtexec) TRTEXEC="${2:?--trtexec 需要路径}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "未知参数: $1" ;;
  esac
done

[[ "$PRECISION" == "fp16" || "$PRECISION" == "fp32" ]] || die "precision 只能是 fp16 或 fp32"
[[ -s "$ONNX" ]] || die "找不到 ONNX：$ONNX（先在 x86 上运行 scripts/trt/prepare_model.sh 并同步到 NX）"
[[ -n "$ENGINE" ]] || ENGINE="$REPO_ROOT/assets/0708_${PRECISION}.engine"

if [[ -f /etc/nv_tegra_release ]]; then
  log "Jetson: $(head -1 /etc/nv_tegra_release)"
else
  log "警告：未检测到 /etc/nv_tegra_release，本机可能不是 Jetson；这里构建的 Engine 不一定能用于 NX"
fi

if [[ -z "$TRTEXEC" ]]; then
  if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC="$(command -v trtexec)"
  elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
  else
    die "找不到 trtexec。请用 --trtexec 指定，或确认 JetPack 已安装 TensorRT（常见路径 /usr/src/tensorrt/bin/trtexec）"
  fi
fi
[[ -x "$TRTEXEC" ]] || die "trtexec 不可执行：$TRTEXEC"
log "trtexec: $TRTEXEC"

if [[ -s "$ENGINE" && "$FORCE" != "1" ]]; then
  log "已存在 $ENGINE，跳过转换（--force 可重建）"
  log "使用视觉程序与真实图片或相机画面检查 Engine"
  exit 0
fi

# trtexec 默认会生成随机输入并运行性能测试。构建时只生成 Engine，实际推理使用真实画面。
help_output="$("$TRTEXEC" --help 2>&1)" || die "无法读取 trtexec --help"
[[ "$help_output" == *--skipInference* ]] || \
  die "当前 trtexec 不支持 --skipInference，无法在不生成随机输入的情况下构建 Engine"

mkdir -p "$(dirname "$ENGINE")"
log_file="${ENGINE%.engine}.build.log"
args=(--onnx="$ONNX" --saveEngine="$ENGINE" --skipInference)
[[ "$PRECISION" == "fp16" ]] && args+=(--fp16)

log "执行：$TRTEXEC ${args[*]}"
log "日志：$log_file"
if ! "$TRTEXEC" "${args[@]}" 2>&1 | tee "$log_file"; then
  log "转换失败，日志尾部："
  tail -n 30 "$log_file" >&2
  die "Engine 构建失败；ONNX 保持不变，本次不生成 Engine"
fi

[[ -s "$ENGINE" ]] || die "trtexec 返回成功但没有生成 $ENGINE"
log "生成成功：$ENGINE ($(wc -c <"$ENGINE") bytes)"
log "使用视觉程序与真实图片或相机画面检查 Engine"
log "下一步：配置 yolo_name: yolov5_trt，yolov5_trt_model_path: ${ENGINE#$REPO_ROOT/}，然后编译并运行视觉程序"
