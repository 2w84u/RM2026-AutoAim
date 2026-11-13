#!/usr/bin/env python3
"""OpenVINO IR (.xml/.bin) -> ONNX 转换器（针对 assets/ 下的 YOLO 检测模型）。

背景
----
OpenVINO 只提供 ONNX/TF/Paddle -> IR 的单向转换，没有官方反向导出工具。
本脚本直接解析 IR 的图结构与权重文件，重建等价的 ONNX 图：

  * Const 节点：按 offset/size 从 .bin 读取（i8/f32/i32/i64/...）
  * 权重量化链 Convert(i8->f32) -> Multiply(per-channel scale) 会被常量折叠成
    单个 f32 initializer，供 Conv/GroupConv 使用
  * 静态 shape 分支（ShapeOf/Gather/StridedSlice/Mod/...）同样折叠成常量
  * FakeQuantize 默认按恒等处理（--fq drop），也可用 --fq exact 保留模拟量化

用法
----
  # 只解析、自检（不需要安装 onnx，可在任何机器上跑）
  python3 tools/ir2onnx.py assets/yolo11.xml --inspect

  # 转换
  python3 tools/ir2onnx.py assets/yolo11.xml assets/yolo11.onnx --fq drop

依赖
----
  numpy；转换时额外需要 onnx（pip install onnx）

注意
----
转换是"有损"的：IR 内的卷积权重本身就是 INT8 量化后反量化的结果，精度只会
保持/略低于原量化模型，不会恢复原始 FP32 训练权重。
"""

import argparse
import os
import sys
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np

DTYPES = {
  'f32': (np.float32, 'FLOAT'),
  'f16': (np.float16, 'FLOAT16'),
  'i8': (np.int8, 'INT8'),
  'u8': (np.uint8, 'UINT8'),
  'i32': (np.int32, 'INT32'),
  'i64': (np.int64, 'INT64'),
  'boolean': (np.bool_, 'BOOL'),
}

FOLDABLE = {
  'Convert', 'Multiply', 'Add', 'Subtract', 'Divide', 'Reshape', 'Transpose', 'Concat',
  'Gather', 'ShapeOf', 'Slice', 'StridedSlice', 'Mod', 'Greater', 'Squeeze', 'Unsqueeze',
}

SUPPORTED_EMIT = {
  'Parameter', 'Result', 'Const', 'FakeQuantize', 'Convolution', 'GroupConvolution', 'Add',
  'Multiply', 'Subtract', 'Divide', 'Sigmoid', 'Swish', 'Concat', 'Reshape', 'VariadicSplit',
  'MaxPool', 'MatMul', 'SoftMax', 'Interpolate', 'Transpose', 'StridedSlice', 'Slice', 'Mod',
  'Greater', 'Gather', 'ShapeOf', 'Convert', 'ReLU', 'HSwish', 'HSigmoid', 'ReduceMean',
}


class IRModel:
  def __init__(self, xml_path, bin_path=None):
    self.xml_path = xml_path
    self.bin_path = bin_path or os.path.splitext(xml_path)[0] + '.bin'
    self.root = ET.parse(xml_path).getroot()
    with open(self.bin_path, 'rb') as f:
      self.bin = f.read()

    self.layers = {}
    self.order = []
    for lay in self.root.iter('layer'):
      self.layers[lay.get('id')] = lay
      self.order.append(lay.get('id'))

    # (to_layer, to_port) -> (from_layer, from_port)
    self.producer = {}
    for e in self.root.iter('edge'):
      self.producer[(e.get('to-layer'), e.get('to-port'))] = (
        e.get('from-layer'), e.get('from-port'))

    self._const_cache = {}

  # ---------- graph queries ----------
  def layer(self, lid):
    return self.layers[lid]

  def type(self, lid):
    return self.layers[lid].get('type')

  def name(self, lid):
    return self.layers[lid].get('name')

  def attr(self, lid):
    d = self.layers[lid].find('data')
    return dict(d.attrib) if d is not None else {}

  def in_keys(self, lid):
    return [(lid, p.get('id')) for p in self.layers[lid].findall('input/port')]

  def in_key(self, lid, idx):
    return self.in_keys(lid)[idx]

  def src(self, key):
    return self.producer.get(key)

  def out_key(self, lid):
    p = self.layers[lid].find('output/port')
    return (lid, p.get('id'))

  def out_dims(self, lid):
    p = self.layers[lid].find('output/port')
    return [int(d.text) for d in p.findall('dim')] if p is not None else None

  def in_dims(self, lid, idx):
    p = self.layers[lid].findall('input/port')[idx]
    return [int(d.text) for d in p.findall('dim')]

  # ---------- constants ----------
  def read_const(self, lid):
    d = self.layers[lid].find('data')
    if d is None:
      raise ValueError('layer %s is not a Const' % lid)
    dt = d.get('element_type', 'f32')
    shape_txt = d.get('shape') or ''
    shape = tuple(int(x) for x in shape_txt.split(',')) if shape_txt else ()
    off = int(d.get('offset') or 0)
    size = int(d.get('size') or 0)
    npdt = DTYPES[dt][0]
    if size == 0:
      return np.zeros(shape, dtype=npdt)
    count = size // np.dtype(npdt).itemsize
    arr = np.frombuffer(self.bin, dtype=npdt, count=count, offset=off)
    return arr.reshape(shape).copy()

  def const_value(self, key):
    """返回该 tensor 的常量值（可递归折叠），不可折叠则返回 None。"""
    if key in self._const_cache:
      return self._const_cache[key]
    self._const_cache[key] = None  # 简单的环保护
    lid, _ = key
    t = self.type(lid)
    v = None
    if t == 'Const':
      v = self.read_const(lid)
    elif t == 'ShapeOf':
      # 输入是动态张量，但 XML 里记录了静态 shape，直接折叠成常量数组
      a = self.attr(lid)
      dtype = np.int64 if a.get('output_type', 'i64') == 'i64' else np.int32
      v = np.array(self.in_dims(lid, 0), dtype=dtype)
    elif t in FOLDABLE:
      vals = []
      for k in self.in_keys(lid):
        s = self.src(k)
        vals.append(self.const_value(s) if s is not None else None)
      if all(x is not None for x in vals):
        v = self._eval_fold(lid, t, vals)
    self._const_cache[key] = v
    return v

  def _eval_fold(self, lid, t, vals):
    a = self.attr(lid)
    if t == 'Convert':
      return vals[0].astype(DTYPES[a['destination_type']][0])
    if t == 'Multiply':
      return vals[0] * vals[1]
    if t == 'Add':
      return vals[0] + vals[1]
    if t == 'Subtract':
      return vals[0] - vals[1]
    if t == 'Divide':
      return vals[0] / vals[1]
    if t == 'Mod':
      return np.mod(vals[0], vals[1])
    if t == 'Greater':
      return np.greater(vals[0], vals[1])
    if t == 'Reshape':
      return vals[0].reshape(self.out_dims(lid))
    if t == 'Transpose':
      return np.transpose(vals[0], vals[1].astype(int).tolist())
    if t == 'Concat':
      return np.concatenate(vals, axis=int(a.get('axis', 0)))
    if t == 'Gather':
      axis = int(a.get('axis', 0))
      batch = int(a.get('batch_dims', 0))
      if batch:
        raise NotImplementedError('Gather batch_dims>0')
      return np.take(vals[0], vals[1].astype(int), axis=axis)
    if t == 'ShapeOf':
      dtype = np.int64 if a.get('output_type', 'i64') == 'i64' else np.int32
      return np.array(self.in_dims(lid, 0), dtype=dtype)
    if t == 'Slice':
      start, stop, step, axes = [v.astype(int).tolist() for v in vals[:4]]
      idx = [slice(None)] * vals[0].ndim
      for s, e, st, ax in zip(start, stop, step, axes):
        idx[ax] = slice(s, e, st)
      return vals[0][tuple(idx)]
    if t == 'StridedSlice':
      begin, end, strides = [v.astype(int).tolist() for v in vals[:3]]
      bm = [int(x) for x in a.get('begin_mask', '').split(',') if x != '']
      em = [int(x) for x in a.get('end_mask', '').split(',') if x != '']
      idx = []
      for i in range(vals[0].ndim):
        lo = None if (i < len(bm) and bm[i]) else begin[i]
        hi = None if (i < len(em) and em[i]) else end[i]
        st = strides[i] if i < len(strides) else 1
        idx.append(slice(lo, hi, st))
      return vals[0][tuple(idx)]
    if t == 'Squeeze':
      return np.squeeze(vals[0], axis=tuple(vals[1].astype(int)) if len(vals) > 1 else None)
    if t == 'Unsqueeze':
      return np.expand_dims(vals[0], axis=tuple(vals[1].astype(int)))
    raise NotImplementedError('fold op %s' % t)


def inspect(ir):
  types = Counter(ir.type(lid) for lid in ir.order)
  print('layers :', sum(types.values()))
  print('ops    :', dict(types.most_common()))
  unknown = set(types) - SUPPORTED_EMIT
  if unknown:
    print('!! 未支持算子:', unknown)
  params = [lid for lid in ir.order if ir.type(lid) == 'Parameter']
  for lid in params:
    a = ir.attr(lid)
    print('input  :', a.get('element_type'), a.get('shape'))
  for lid in ir.order:
    if ir.type(lid) == 'Result':
      src = ir.src(ir.in_key(lid, 0))
      print('output :', ir.out_dims(src[0]))
  n_conv = 0
  bytes_total = 0
  for lid in ir.order:
    if ir.type(lid) in ('Convolution', 'GroupConvolution'):
      n_conv += 1
      wkey = ir.src(ir.in_keys(lid)[1])
      w = ir.const_value(wkey)
      if w is None:
        print('!! Conv %s 的权重无法折叠为常量' % ir.name(lid))
      else:
        bytes_total += w.nbytes
    if ir.type(lid) == 'Const':
      bytes_total += ir.read_const(lid).nbytes
  print('conv   :', n_conv, '个，权重+常量约 %.2f MB' % (bytes_total / 1024 / 1024))


class Builder:
  def __init__(self, ir, fq='drop', opset=13):
    import onnx  # noqa: F401  (延迟导入，inspect 模式不需要)
    from onnx import helper, numpy_helper, TensorProto
    self.helper = helper
    self.numpy_helper = numpy_helper
    self.TensorProto = TensorProto
    self.ir = ir
    self.fq = fq
    self.opset = opset
    self.nodes = []
    self.inits = []
    self.init_cache = {}
    self.inputs = []
    self.outputs = []
    self.tensor = {}
    self.alias = {}
    self.n = 0

  def _name(self, prefix):
    self.n += 1
    return '%s_%d' % (prefix, self.n)

  def _tensor_proto(self, t):
    return getattr(self.TensorProto, DTYPES[t][1])

  def _init(self, arr):
    key = (arr.dtype.str, arr.shape, arr.tobytes())
    if key in self.init_cache:
      return self.init_cache[key]
    name = self._name('const')
    self.inits.append(self.numpy_helper.from_array(np.ascontiguousarray(arr), name))
    self.init_cache[key] = name
    return name

  def resolve(self, key):
    if key in self.tensor:
      return self.tensor[key]
    if key in self.alias:
      return self.resolve(self.alias[key])
    v = self.ir.const_value(key)
    if v is None:
      raise RuntimeError('无法解析 tensor %s (%s)' % (key, self.ir.type(key[0])))
    name = self._init(v)
    self.tensor[key] = name
    return name

  def _ins(self, lid):
    return [self.resolve(self.ir.src(k)) for k in self.ir.in_keys(lid)]

  def _pads(self, a):
    b = [int(x) for x in a.get('pads_begin', '0, 0').split(',')]
    e = [int(x) for x in a.get('pads_end', '0, 0').split(',')]
    return b + e

  def build(self):
    ir = self.ir
    for lid in ir.order:
      t = ir.type(lid)
      if t == 'Const':
        continue
      if t == 'Parameter':
        a = ir.attr(lid)
        shape = [int(x) for x in (a.get('shape') or '').split(',')]
        name = 'input'
        self.tensor[ir.out_key(lid)] = name
        self.inputs.append(self.helper.make_tensor_value_info(
          name, self._tensor_proto(a.get('element_type', 'f32')), shape))
        continue
      if t == 'Result':
        src = ir.src(ir.in_key(lid, 0))
        dims = ir.out_dims(src[0])
        self.outputs.append(self.helper.make_tensor_value_info(
          self.resolve(src), self.TensorProto.FLOAT, dims))
        continue
      if ir.const_value(ir.out_key(lid)) is not None:
        continue  # 常量折叠，按需物化
      if t not in SUPPORTED_EMIT:
        raise NotImplementedError('emit op %s (%s)' % (t, ir.name(lid)))
      self._emit(lid, t)

    graph = self.helper.make_graph(
      self.nodes, os.path.basename(self.ir.xml_path), self.inputs, self.outputs, self.inits)
    model = self.helper.make_model(graph, opset_imports=[self.helper.make_opsetid('', self.opset)])
    model.ir_version = 8
    return model

  def _emit(self, lid, t):
    ir = self.ir
    a = ir.attr(lid)
    H = self.helper

    if t == 'FakeQuantize':
      if self.fq == 'drop':
        # 恒等：输出直接复用上游 tensor（注意要取 producer，而不是本层的输入端口）
        self.alias[ir.out_key(lid)] = ir.src(ir.in_key(lid, 0))
        return
      ins = self._ins(lid)
      out = self._name('fq')
      levels = int(a.get('levels', 256))
      il = ir.const_value(ir.src(ir.in_key(lid, 1)))
      ih = ir.const_value(ir.src(ir.in_key(lid, 2)))
      ol = ir.const_value(ir.src(ir.in_key(lid, 3)))
      oh = ir.const_value(ir.src(ir.in_key(lid, 4)))
      step_in = self._init(((ih - il) / (levels - 1)).astype(np.float32))
      step_out = self._init(((oh - ol) / (levels - 1)).astype(np.float32))
      lo = self._init(np.zeros_like(il, dtype=np.float32))
      hi = self._init(np.full_like(il, levels - 1, dtype=np.float32))
      il_n, ol_n = self._init(il.astype(np.float32)), self._init(ol.astype(np.float32))
      n1, n2, n3, n4, n5 = [self._name('fq') for _ in range(5)]
      self.nodes += [
        H.make_node('Sub', [ins[0], il_n], [n1]),
        H.make_node('Div', [n1, step_in], [n2]),
        H.make_node('Clip', [n2, lo, hi], [n3]),
        H.make_node('Round', [n3], [n4]),
        H.make_node('Mul', [n4, step_out], [n5]),
        H.make_node('Add', [n5, ol_n], [out]),
      ]
      self.tensor[ir.out_key(lid)] = out
      return

    ins = self._ins(lid)
    out = self._name(t.lower())

    if t == 'Convolution':
      w = ir.const_value(ir.src(ir.in_key(lid, 1)))
      if w is None:
        raise RuntimeError('Conv %s 的权重不是常量，无法导出' % ir.name(lid))
      node = H.make_node(
        'Conv', ins, [out],
        strides=[int(x) for x in a.get('strides', '1, 1').split(',')],
        dilations=[int(x) for x in a.get('dilations', '1, 1').split(',')],
        pads=self._pads(a), group=1)
    elif t == 'GroupConvolution':
      w = ir.const_value(ir.src(ir.in_key(lid, 1)))
      g = w.shape[0]
      w = w.reshape((g * w.shape[1],) + w.shape[2:])
      ins = [ins[0], self._init(w)]
      node = H.make_node(
        'Conv', ins, [out],
        strides=[int(x) for x in a.get('strides', '1, 1').split(',')],
        dilations=[int(x) for x in a.get('dilations', '1, 1').split(',')],
        pads=self._pads(a), group=g)
    elif t in ('Add', 'Multiply', 'Subtract', 'Divide'):
      node = H.make_node(t, ins, [out])
    elif t == 'Sigmoid':
      node = H.make_node('Sigmoid', ins, [out])
    elif t == 'ReLU':
      node = H.make_node('Relu', ins, [out])
    elif t in ('HSigmoid', 'HSwish'):
      three = self._init(np.array(3.0, np.float32))
      six = self._init(np.array(6.0, np.float32))
      zero = self._init(np.array(0.0, np.float32))
      a1, a2, a3 = [self._name('hsw') for _ in range(3)]
      self.nodes += [
        H.make_node('Add', [ins[0], three], [a1]),
        H.make_node('Clip', [a1, zero, six], [a2]),
        H.make_node('Div', [a2, six], [a3]),
      ]
      if t == 'HSigmoid':
        self.tensor[ir.out_key(lid)] = a3
        return
      node = H.make_node('Mul', [ins[0], a3], [out])
    elif t == 'ReduceMean':
      axes = ir.const_value(ir.src(ir.in_key(lid, 1)))
      node = H.make_node(
        'ReduceMean', [ins[0]], [out],
        axes=[int(x) for x in np.atleast_1d(axes)],
        keepdims=1 if a.get('keep_dims') == 'true' else 0)
    elif t == 'Swish':
      beta = float(a.get('beta', 1.0))
      if beta != 1.0:
        b = self._init(np.array(beta, dtype=np.float32))
        m = self._name('swish')
        self.nodes.append(H.make_node('Mul', [ins[0], b], [m]))
        ins = [m]
      s = self._name('swish')
      self.nodes.append(H.make_node('Sigmoid', ins, [s]))
      node = H.make_node('Mul', [ins[0], s], [out])
    elif t == 'Concat':
      node = H.make_node('Concat', ins, [out], axis=int(a.get('axis', 1)))
    elif t == 'Reshape':
      shape = self._init(np.array(ir.out_dims(lid), dtype=np.int64))
      node = H.make_node('Reshape', [ins[0], shape], [out])
    elif t == 'VariadicSplit':
      axis = int(a.get('axis', 0))
      n_out = len(ir.layer(lid).findall('output/port'))
      outs = [self._name('split') for _ in range(n_out)]
      node = H.make_node('Split', [ins[0], ins[2]], outs, axis=axis)
      self.nodes.append(node)
      for p, n in zip(ir.layer(lid).findall('output/port'), outs):
        self.tensor[(lid, p.get('id'))] = n
      return
    elif t == 'MaxPool':
      node = H.make_node(
        'MaxPool', ins, [out],
        kernel_shape=[int(x) for x in a.get('kernel', '1, 1').split(',')],
        strides=[int(x) for x in a.get('strides', '1, 1').split(',')],
        pads=self._pads(a), ceil_mode=1 if a.get('rounding_type') == 'ceil' else 0)
    elif t == 'MatMul':
      lhs = ins[0]
      if a.get('transpose_a') == 'true':
        rank = len(ir.in_dims(lid, 0))
        perm = list(range(rank - 2)) + [rank - 1, rank - 2]
        tn = self._name('ta')
        self.nodes.append(H.make_node('Transpose', [lhs], [tn], perm=perm))
        lhs = tn
      rhs = ins[1]
      if a.get('transpose_b') == 'true':
        rank = len(ir.in_dims(lid, 1))
        perm = list(range(rank - 2)) + [rank - 1, rank - 2]
        tn = self._name('tb')
        self.nodes.append(H.make_node('Transpose', [rhs], [tn], perm=perm))
        rhs = tn
      node = H.make_node('MatMul', [lhs, rhs], [out])
    elif t == 'SoftMax':
      node = H.make_node('Softmax', ins, [out], axis=int(a.get('axis', -1)))
    elif t == 'Interpolate':
      scales = ir.const_value(ir.src(ir.in_key(lid, 1)))
      axes = ir.const_value(ir.src(ir.in_key(lid, 2))) if len(ir.in_keys(lid)) > 2 else None
      rank = len(ir.in_dims(lid, 0))
      full = np.ones(rank, dtype=np.float32)
      if axes is None:
        full[:] = scales
      else:
        for i, ax in enumerate(np.asarray(axes).astype(int).tolist()):
          full[ax] = scales[i]
      sc = self._init(full)
      node = H.make_node(
        'Resize', [ins[0], '', sc], [out], mode=a.get('mode', 'nearest'),
        coordinate_transformation_mode=a.get('coordinate_transformation_mode', 'asymmetric'),
        nearest_mode=a.get('nearest_mode', 'floor'))
    elif t == 'Transpose':
      perm = ir.const_value(ir.src(ir.in_key(lid, 1))).astype(int).tolist()
      node = H.make_node('Transpose', [ins[0]], [out], perm=perm)
    elif t == 'StridedSlice':
      begin = ir.const_value(ir.src(ir.in_key(lid, 1))).astype(int).tolist()
      end = ir.const_value(ir.src(ir.in_key(lid, 2))).astype(int).tolist()
      strides = ir.const_value(ir.src(ir.in_key(lid, 3))).astype(int).tolist()
      bm = [int(x) for x in a.get('begin_mask', '').split(',') if x != '']
      em = [int(x) for x in a.get('end_mask', '').split(',') if x != '']
      starts, ends, axes, steps = [], [], [], []
      for i in range(len(begin)):
        if i < len(bm) and bm[i] and i < len(em) and em[i] and strides[i] == 1:
          continue
        starts.append(0 if (i < len(bm) and bm[i]) else begin[i])
        ends.append(2**31 - 1 if (i < len(em) and em[i]) else end[i])
        axes.append(i)
        steps.append(strides[i])
      node = H.make_node(
        'Slice', [ins[0], self._init(np.array(starts, np.int64)),
                  self._init(np.array(ends, np.int64)),
                  self._init(np.array(axes, np.int64)),
                  self._init(np.array(steps, np.int64))], [out])
    elif t == 'Slice':
      starts, stops, steps, axes = [ir.const_value(ir.src(ir.in_key(lid, i))).astype(np.int64)
                                    for i in range(1, 5)]
      node = H.make_node(
        'Slice', [ins[0], self._init(starts), self._init(stops),
                  self._init(axes), self._init(steps)], [out])
    elif t == 'Mod':
      node = H.make_node('Mod', ins, [out], fmod=0)
    elif t == 'Greater':
      node = H.make_node('Greater', ins, [out])
    elif t == 'Gather':
      node = H.make_node('Gather', ins, [out], axis=int(a.get('axis', 0)))
    elif t == 'Convert':
      node = H.make_node('Cast', ins, [out], to=self._tensor_proto(a['destination_type']))
    elif t == 'ShapeOf':
      node = H.make_node('Shape', ins, [out])
      if a.get('output_type', 'i64') == 'i32':
        cast = self._name('shape_i32')
        self.nodes.append(node)
        self.nodes.append(H.make_node('Cast', [out], [cast], to=self.TensorProto.INT32))
        self.tensor[ir.out_key(lid)] = cast
        return
    else:
      raise NotImplementedError(t)

    self.nodes.append(node)
    self.tensor[ir.out_key(lid)] = out


def main():
  ap = argparse.ArgumentParser(description='OpenVINO IR -> ONNX（YOLO 检测模型）')
  ap.add_argument('xml', help='输入 .xml')
  ap.add_argument('out', nargs='?', help='输出 .onnx')
  ap.add_argument('--bin', default=None, help='.bin 路径（默认同名）')
  ap.add_argument('--fq', choices=['drop', 'exact'], default='drop',
                  help='FakeQuantize 处理方式：drop=恒等（推荐，利于 OpenCV DNN）')
  ap.add_argument('--opset', type=int, default=13)
  ap.add_argument('--inspect', action='store_true', help='只解析并打印统计，不写 ONNX')
  args = ap.parse_args()

  ir = IRModel(args.xml, args.bin)
  if args.inspect or not args.out:
    inspect(ir)
    return 0

  b = Builder(ir, fq=args.fq, opset=args.opset)
  model = b.build()

  import onnx
  from onnx import checker
  checker.check_model(model)
  onnx.save(model, args.out)
  print('已导出 %s（%d 个节点，%d 个 initializer，%.2f MB）' %
        (args.out, len(b.nodes), len(b.inits),
         os.path.getsize(args.out) / 1024 / 1024))
  return 0


if __name__ == '__main__':
  sys.exit(main())
