#!/usr/bin/env python3
"""对比两份 detector_verify 输出的 JSON，用于 IR(参考) 与 ONNX(候选) 的 A/B 验证。

用法:
  python3 tools/compare_detections.py ref.json cand.json [--iou 0.5] [--min-match 0.95]

判定:
  以参考结果为准，按类别 + IoU 贪心匹配；输出漏检、多余、类别不一致、
  平均 IoU、关键点平均偏移，并在匹配率低于阈值时返回非 0 退出码。
"""

import argparse
import json
import math
import sys


def iou(a, b):
  ax, ay, aw, ah = a
  bx, by, bw, bh = b
  if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
    return 0.0
  ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
  iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
  inter = ix * iy
  union = aw * ah + bw * bh - inter
  return inter / union if union > 0 else 0.0


def kpt_dist(pa, pb):
  if len(pa) != 4 or len(pb) != 4:
    return None
  return sum(math.dist(pa[i], pb[i]) for i in range(4)) / 4.0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('ref')
  ap.add_argument('cand')
  ap.add_argument('--iou', type=float, default=0.5, help='匹配 IoU 阈值')
  ap.add_argument('--min-match', type=float, default=0.95, help='最低匹配率')
  ap.add_argument('--max-report', type=int, default=10, help='最多打印多少条不匹配明细')
  args = ap.parse_args()

  ref = json.load(open(args.ref))
  cand = json.load(open(args.cand))

  def frame_map(doc):
    out = {}
    for f in doc.get('frames', []):
      out[f['index']] = f.get('detections', [])
    return out

  ref_frames, cand_frames = frame_map(ref), frame_map(cand)
  common = sorted(set(ref_frames) & set(cand_frames))

  n_ref = n_cand = matched = class_mismatch = 0
  ious, dists = [], []
  details = []

  for idx in common:
    r = ref_frames[idx]
    c = list(cand_frames[idx])
    n_ref += len(r)
    n_cand += len(c)
    used = [False] * len(c)

    for d in r:
      best, best_iou, best_same = -1, 0.0, False
      for j, e in enumerate(c):
        if used[j]:
          continue
        v = iou(d['box'], e['box'])
        same = d.get('name') == e.get('name')
        # 同类优先，其次 IoU
        if v >= args.iou and (v > best_iou or (same and not best_same and v >= best_iou)):
          best, best_iou, best_same = j, v, same
      if best >= 0:
        used[best] = True
        matched += 1
        ious.append(best_iou)
        if not best_same:
          class_mismatch += 1
        kd = kpt_dist(d.get('points', []), c[best].get('points', []))
        if kd is not None:
          dists.append(kd)
        if not best_same and len(details) < args.max_report:
          details.append('帧 %d: 类别 %s -> %s (IoU %.2f)' %
                         (idx, d.get('name'), c[best].get('name'), best_iou))
      elif len(details) < args.max_report:
        details.append('帧 %d: 漏检 %s conf=%.2f' % (idx, d.get('name'), d.get('confidence', 0)))

  missed = n_ref - matched
  extra = n_cand - matched
  rate = matched / n_ref if n_ref else 1.0
  mean_iou = sum(ious) / len(ious) if ious else 0.0
  mean_kpt = sum(dists) / len(dists) if dists else float('nan')

  print('参考帧数      : %d' % len(ref_frames))
  print('候选帧数      : %d' % len(cand_frames))
  print('共同帧数      : %d' % len(common))
  print('参考目标数    : %d' % n_ref)
  print('候选目标数    : %d' % n_cand)
  print('匹配成功      : %d (%.1f%%)' % (matched, 100 * rate))
  print('漏检/多余     : %d / %d' % (missed, extra))
  print('类别不一致    : %d' % class_mismatch)
  print('平均 IoU      : %.3f' % mean_iou)
  print('关键点平均偏移: %.2f px' % mean_kpt)

  if details:
    print('\n不匹配明细（最多 %d 条）:' % args.max_report)
    for d in details:
      print('  ' + d)

  ok = rate >= args.min_match and class_mismatch == 0
  print('\n判定: %s（匹配率阈值 %.0f%%）' % ('PASS' if ok else 'FAIL', args.min_match * 100))
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
