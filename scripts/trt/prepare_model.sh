#!/usr/bin/env bash
# 在 x86 主机上准备模型：下载深圳大学 0708.onnx（不训练、不转 .pt）。
# 只做下载与静态检查，不生成 Engine——Engine 必须在 Jetson NX 上构建。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

URL="${MODEL_URL:-https://raw.githubusercontent.com/broalantaps/RobotDetectionModel/main/Model/0708.onnx}"
OUT="${OUT:-$REPO_ROOT/assets/0708.onnx}"
FORCE="${FORCE:-0}"

log() { printf '[prepare_model] %s\n' "$*"; }
die() { printf '[prepare_model] ERROR: %s\n' "$*" >&2; exit 1; }

show_onnx_io() {
  local model="$1" py=""
  if python3 -c "import onnx" >/dev/null 2>&1; then
    py=python3
  elif [[ -x "$REPO_ROOT/.venv-ir2onnx/bin/python" ]] &&
       "$REPO_ROOT/.venv-ir2onnx/bin/python" -c "import onnx" >/dev/null 2>&1; then
    py="$REPO_ROOT/.venv-ir2onnx/bin/python"
  fi
  if [[ -n "$py" ]]; then
    "$py" - "$model" <<'PY'
import sys
import onnx

model = onnx.load(sys.argv[1])
for tensor, kind in [(t, "input") for t in model.graph.input] + [
    (t, "output") for t in model.graph.output
]:
    dims = [d.dim_value or d.dim_param for d in tensor.type.tensor_type.shape.dim]
    print(f"[prepare_model]   {kind:6s} {tensor.name:10s} {dims}")
PY
  else
    log "python onnx 不可用，跳过输入/输出静态检查（不影响下载）"
  fi
}

if [[ -s "$OUT" && "$FORCE" != "1" ]]; then
  log "已存在 $OUT，跳过下载（FORCE=1 可重新下载）"
else
  mkdir -p "$(dirname "$OUT")"
  tmp="$OUT.part"
  rm -f "$tmp"
  log "下载 $URL"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$tmp" "$URL" \
      || die "curl 下载失败"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp" "$URL" || die "wget 下载失败"
  else
    die "需要 curl 或 wget"
  fi
  [[ -s "$tmp" ]] || die "下载结果为空：$tmp"
  mv "$tmp" "$OUT"
fi

[[ -s "$OUT" ]] || die "模型文件不存在或为空：$OUT"

size=$(wc -c <"$OUT")
if command -v sha256sum >/dev/null 2>&1; then
  sha=$(sha256sum "$OUT" | cut -d' ' -f1)
else
  sha=$(shasum -a 256 "$OUT" | cut -d' ' -f1)
fi

log "文件: $OUT"
log "大小: $size bytes"
log "SHA-256: $sha"
show_onnx_io "$OUT"

{
  echo "source: $URL"
  echo "file: $OUT"
  echo "size_bytes: $size"
  echo "sha256: $sha"
  echo "downloaded_at: $(date -Iseconds)"
} >"$OUT.download.txt"
log "已记录下载信息到 $OUT.download.txt（仅追踪本次下载的文件，不做模型身份对照）"
log "下一步：./scripts/trt/deploy_nx.sh <user>@<nx-ip> <远程目录>"
