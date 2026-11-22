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

  # 同一图片对比 IR / ONNX Runtime / OpenCV DNN CPU，保存数值报告
  python3 tools/ir2onnx.py assets/yolo11.xml assets/yolo11_dnn.onnx --validate \
    --image assets/standard_fanblade.jpg --report out/ir2onnx/yolo11.json

依赖
----
  numpy；转换时额外需要 onnx（pip install onnx）

注意
----
INT8 IR 无法恢复原始 FP32 训练权重。--fq drop 还会移除激活量化，精度变化
必须用实际数据评估，不能仅凭图结构或随机输入保证。
"""

import argparse
import json
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
      if np.issubdtype(vals[0].dtype, np.integer):
        if a.get('m_pythondiv', 'true') == 'true':
          return np.floor_divide(vals[0], vals[1])
        return np.trunc(vals[0] / vals[1]).astype(vals[0].dtype)
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
      axis = int(vals[2].item())
      batch = int(a.get('batch_dims', 0))
      if batch:
        raise NotImplementedError('Gather batch_dims>0')
      return np.take(vals[0], vals[1].astype(int), axis=axis)
    if t == 'ShapeOf':
      dtype = np.int64 if a.get('output_type', 'i64') == 'i64' else np.int32
      return np.array(self.in_dims(lid, 0), dtype=dtype)
    if t == 'Slice':
      start, stop, step = [v.astype(int).reshape(-1).tolist() for v in vals[1:4]]
      axes = vals[4].astype(int).reshape(-1).tolist() if len(vals) > 4 else range(len(start))
      idx = [slice(None)] * vals[0].ndim
      for s, e, st, ax in zip(start, stop, step, axes):
        idx[ax] = slice(s, e, st)
      return vals[0][tuple(idx)]
    if t == 'StridedSlice':
      begin, end, strides = [v.astype(int).reshape(-1).tolist() for v in vals[1:4]]
      bm = [int(x) for x in a.get('begin_mask', '').split(',') if x != '']
      em = [int(x) for x in a.get('end_mask', '').split(',') if x != '']
      idx = []
      for i in range(vals[0].ndim):
        lo = None if i >= len(begin) or (i < len(bm) and bm[i]) else begin[i]
        hi = None if i >= len(end) or (i < len(em) and em[i]) else end[i]
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
  valid = not unknown
  for lid in ir.order:
    if ir.type(lid) in ('Convolution', 'GroupConvolution'):
      n_conv += 1
      wkey = ir.src(ir.in_keys(lid)[1])
      w = ir.const_value(wkey)
      if w is None:
        valid = False
        print('!! Conv %s 的权重无法折叠为常量' % ir.name(lid))
      else:
        bytes_total += w.nbytes
    if ir.type(lid) == 'Const':
      bytes_total += ir.read_const(lid).nbytes
  print('conv   :', n_conv, '个，权重+常量约 %.2f MB' % (bytes_total / 1024 / 1024))
  return valid


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
    # ascontiguousarray 会把标量提升为 [1]，而 ONNX Clip 要求标量上下界。
    self.inits.append(self.numpy_helper.from_array(np.array(arr, copy=True, order='C'), name))
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

    used = {name for node in self.nodes for name in node.input}
    used.update(value.name for value in self.outputs)
    self.inits = [value for value in self.inits if value.name in used]
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
      if levels < 2 or np.any(ih <= il):
        raise ValueError('FakeQuantize 仅支持 levels >= 2 且 input_high > input_low')
      range_in = self._init((ih - il).astype(np.float32))
      range_out = self._init((oh - ol).astype(np.float32))
      count = self._init(np.array(levels - 1, dtype=np.float32))
      lo = self._init(np.array(0, dtype=np.float32))
      hi = self._init(np.array(levels - 1, dtype=np.float32))
      il_n, ol_n = self._init(il.astype(np.float32)), self._init(ol.astype(np.float32))
      n1, n2, n3, n4, n5, n6, n7 = [self._name('fq') for _ in range(7)]
      self.nodes += [
        H.make_node('Sub', [ins[0], il_n], [n1]),
        H.make_node('Div', [n1, range_in], [n2]),
        H.make_node('Mul', [n2, count], [n3]),
        H.make_node('Clip', [n3, lo, hi], [n4]),
        H.make_node('Round', [n4], [n5]),
        H.make_node('Div', [n5, count], [n6]),
        H.make_node('Mul', [n6, range_out], [n7]),
        H.make_node('Add', [n7, ol_n], [out]),
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
        kernel_shape=list(w.shape[2:]),
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
        kernel_shape=list(w.shape[2:]),
        strides=[int(x) for x in a.get('strides', '1, 1').split(',')],
        dilations=[int(x) for x in a.get('dilations', '1, 1').split(',')],
        pads=self._pads(a), group=g)
    elif t in ('Add', 'Multiply', 'Subtract', 'Divide'):
      op = {'Add': 'Add', 'Multiply': 'Mul', 'Subtract': 'Sub', 'Divide': 'Div'}[t]
      node = H.make_node(op, ins, [out])
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
      axis = int(ir.const_value(ir.src(ir.in_key(lid, 1))).item())
      lengths = ir.const_value(ir.src(ir.in_key(lid, 2))).astype(np.int64)
      if np.any(lengths == -1):
        lengths = lengths.copy()
        lengths[lengths == -1] = ir.in_dims(lid, 0)[axis] - lengths[lengths != -1].sum()
      n_out = len(ir.layer(lid).findall('output/port'))
      outs = [self._name('split') for _ in range(n_out)]
      node = H.make_node('Split', [ins[0], self._init(lengths)], outs, axis=axis)
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
      if int(a.get('batch_dims', 0)):
        raise NotImplementedError('Gather batch_dims != 0')
      axis = int(ir.const_value(ir.src(ir.in_key(lid, 2))).item())
      node = H.make_node('Gather', ins[:2], [out], axis=axis)
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


def validate(ir, onnx_path, fq, image_path=None):
  """同一份 RGB/NCHW/0~1 输入，检查 IR、ORT 与 OpenCV CPU 输出。"""
  import cv2
  import onnxruntime as ort
  import openvino as ov

  cv2.setNumThreads(2)
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  session = ort.InferenceSession(
    onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
  shape = session.get_inputs()[0].shape
  if len(shape) != 4 or shape[:2] != [1, 3]:
    raise ValueError('验证仅支持静态 NCHW、batch=1、RGB 三通道输入')
  if image_path:
    image = cv2.imread(image_path)
    if image is None:
      raise ValueError('无法读取验证图片: ' + image_path)
    height, width = shape[2:]
    scale = min(height / image.shape[0], width / image.shape[1])
    h, w = int(image.shape[0] * scale), int(image.shape[1] * scale)
    padded = np.zeros((height, width, 3), dtype=np.uint8)
    padded[:h, :w] = cv2.resize(image, (w, h))
    x = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, swapRB=True)
  else:
    x = np.random.default_rng(0).random(shape, dtype=np.float32)

  core = ov.Core()
  model = core.read_model(ir.xml_path, ir.bin_path)
  config = {'INFERENCE_PRECISION_HINT': 'f32', 'INFERENCE_NUM_THREADS': 2}

  def infer(reference):
    compiled = core.compile_model(reference, 'CPU', config)
    result = compiled([x])
    return [np.array(result[output], copy=True) for output in compiled.outputs]

  def compare(reference, actual):
    if len(reference) != len(actual):
      raise RuntimeError('输出数量不一致')
    stats = []
    for expected, value in zip(reference, actual):
      if expected.shape != value.shape or not (
          np.isfinite(expected).all() and np.isfinite(value).all()):
        raise RuntimeError('输出形状不一致或包含非有限数值')
      delta = np.abs(expected.astype(np.float64) - value.astype(np.float64))
      stats.append({'shape': list(value.shape), 'max_abs': float(delta.max()),
                    'mean_abs': float(delta.mean())})
    return stats

  ir_outputs = infer(model)
  ort_outputs = session.run(None, {'input': x})
  net = cv2.dnn.readNetFromONNX(onnx_path)
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  net.setInput(x)
  cv_outputs = net.forward(net.getUnconnectedOutLayersNames())
  report = {
    'xml': ir.xml_path, 'onnx': onnx_path, 'fq': fq,
    'input': image_path or 'numpy default_rng(0), uniform [0, 1)',
    'input_shape': shape,
    'versions': {'numpy': np.__version__, 'openvino': ov.__version__,
                 'onnxruntime': ort.__version__, 'opencv': cv2.__version__},
    'ir_vs_ort': compare(ir_outputs, ort_outputs),
    'ort_vs_opencv': compare(ort_outputs, cv_outputs),
    'accuracy_validated': False,
  }
  def close(reference, actual):
    return all(np.allclose(a, b, rtol=1e-3, atol=1e-2) for a, b in zip(reference, actual))

  report['checks'] = {'shape_and_finite': True, 'opencv_forward': True,
                      'ir_close_to_ort': close(ir_outputs, ort_outputs),
                      'opencv_close_to_ort': close(ort_outputs, cv_outputs)}
  # drop 的结构正确性要与同样移除 FQ 的 IR 比较；原始 IR 的差异另行保留。
  if fq == 'drop':
    reference = model.clone()
    for op in reference.get_ordered_ops():
      if op.get_type_name() == 'FakeQuantize':
        op.output(0).replace(op.input_value(0))
    reference.validate_nodes_and_infer_types()
    # 与转换器一样先折叠权重反量化；否则 CPU 仍可能将 i8 权重链融合成低精度 FC。
    from openvino.runtime.passes import ConstantFolding, Manager
    passes = Manager()
    passes.register_pass(ConstantFolding())
    passes.run_passes(reference)
    reference_outputs = infer(reference)
    report['ir_without_fq_vs_ort'] = compare(reference_outputs, ort_outputs)
    report['checks']['ir_without_fq_close_to_ort'] = close(reference_outputs, ort_outputs)
  reference_check = 'ir_without_fq_close_to_ort' if fq == 'drop' else 'ir_close_to_ort'
  report['validation_passed'] = (
    report['checks'][reference_check] and report['checks']['opencv_close_to_ort'])
  report['note'] = ('单输入的数值和加载检查不代表检测精度验证；'
                    '原始 INT8 IR 的内核舍入及融合可能与展开的浮点 FQ 不同。')
  return report


def main():
  ap = argparse.ArgumentParser(description='OpenVINO IR -> ONNX（YOLO 检测模型）')
  ap.add_argument('xml', help='输入 .xml')
  ap.add_argument('out', nargs='?', help='输出 .onnx')
  ap.add_argument('--bin', default=None, help='.bin 路径（默认同名）')
  ap.add_argument('--fq', choices=['drop', 'exact'], default='drop',
                  help='FakeQuantize 处理方式：drop=恒等（推荐，利于 OpenCV DNN）')
  ap.add_argument('--opset', type=int, choices=range(13, 18), default=13)
  ap.add_argument('--inspect', action='store_true', help='只解析并打印统计，不写 ONNX')
  ap.add_argument('--validate', action='store_true', help='转换后对比 IR/ORT/OpenCV CPU')
  ap.add_argument('--image', help='验证图片（按项目方式填充、RGB、归一化；默认固定随机输入）')
  ap.add_argument('--report', help='将验证结果写入 JSON 文件（需 --validate）')
  args = ap.parse_args()
  if (args.image or args.report) and not args.validate:
    ap.error('--image/--report 需要 --validate')
  if args.validate and (args.inspect or not args.out):
    ap.error('--validate 需要输出路径，且不能与 --inspect 同用')

  ir = IRModel(args.xml, args.bin)
  if args.inspect or not args.out:
    return 0 if inspect(ir) else 1

  b = Builder(ir, fq=args.fq, opset=args.opset)
  model = b.build()

  import onnx
  from onnx import checker
  checker.check_model(model, full_check=True)
  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  onnx.save(model, args.out)
  print('已导出 %s（%d 个节点，%d 个 initializer，%.2f MB）' %
        (args.out, len(b.nodes), len(b.inits),
         os.path.getsize(args.out) / 1024 / 1024))
  if args.validate:
    report = validate(ir, args.out, args.fq, args.image)
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.report:
      os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
      with open(args.report, 'w', encoding='utf-8') as f:
        f.write(result + '\n')
    if not report['validation_passed']:
      print('数值对齐未通过，详见报告；ONNX 文件已导出。', file=sys.stderr)
      return 1
  return 0


if __name__ == '__main__':
  sys.exit(main())
