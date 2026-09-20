"""Export YOLO model to AEXRT-M flat format for mobile runtime.
Bypasses .aexrt engine; reads ONNX directly, emits a simple binary:
  [header][layer table][weight bytes]. Mobile runtime reads this directly."""
import struct
import sys

import numpy as np
import onnx

sys.path.insert(0, 'src')

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/cs2V8_320.onnx'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'build/tmp/model_flat.bin'

m = onnx.load(MODEL)
graph = m.graph
inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.initializer}

# find conv chains: Conv -> (SiLU pattern: Mul(x, Sigmoid(x)))
# build a simple sequential op list
ops = []
for node in graph.node:
    if node.op_type == 'Conv':
        w = inits[node.input[1]]
        b = inits[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0], np.float32)
        attrs = {a.name: a for a in node.attribute}
        stride = attrs['strides'].ints[0] if 'strides' in attrs else 1
        pads = attrs['pads'].ints if 'pads' in attrs else [0,0,0,0]
        groups = attrs['group'].i if 'group' in attrs else 1
        ops.append({
            'type': 'conv_silu' if node.op_type == 'Conv' else 'conv',  # fix below
            'in': node.input[0], 'out': node.output[0],
            'w': w.astype(np.float32), 'b': b.astype(np.float32),
            'stride': stride, 'pad': pads[0], 'groups': groups,
        })
    elif node.op_type == 'Concat':
        ops.append({'type': 'concat', 'ins': list(node.inputs if hasattr(node, 'inputs') else node.input), 'out': node.output[0]})
    elif node.op_type == 'MaxPool':
        attrs = {a.name: a for a in node.attribute}
        ops.append({'type': 'maxpool', 'in': node.input[0], 'out': node.output[0],
                    'stride': attrs['strides'].ints[0] if 'strides' in attrs else 2})
    elif node.op_type == 'Resize':
        ops.append({'type': 'resize', 'in': node.input[0], 'out': node.output[0]})
    elif node.op_type == 'Add':
        ops.append({'type': 'add', 'in0': node.input[0], 'in1': node.input[1], 'out': node.output[0]})
    elif node.op_type == 'Mul':
        # SiLU: Mul(x, Sigmoid(x)) — detect by checking if one input is a Sigmoid output
        # simplified: assume it's SiLU if next op after Conv
        ops.append({'type': 'silu_marker', 'in': node.input[0], 'in1': node.input[1], 'out': node.output[0]})

# SiLU (Mul with Sigmoid): emit as standalone silu elementwise op
sigmoid_outputs = set()
for node in graph.node:
    if node.op_type == 'Sigmoid':
        sigmoid_outputs.add(node.output[0])

for i, op in enumerate(ops):
    if op['type'] == 'silu_marker':
        # SiLU if one input is a Sigmoid of the other
        if op.get('in1') in sigmoid_outputs:
            op['type'] = 'silu'
        elif op.get('in') in sigmoid_outputs:
            op['type'] = 'silu'
        else:
            op['type'] = 'skip'  # regular mul, not SiLU

ops = [op for op in ops if op['type'] != 'skip' and op['type'] != 'silu_marker']

# assign value names to indices
value_ids = {}
def vid(name):
    if name not in value_ids:
        value_ids[name] = len(value_ids)
    return value_ids[name]

# pre-register all names
weight_blob = bytearray()
layer_table = bytearray()

for op in ops:
    if op['type'] in ('conv', 'conv_silu'):
        w = op['w']
        b = op['b']
        oc, ic, kh, kw = w.shape
        silu = 1 if op['type'] == 'conv_silu' else 0
        w_off = len(weight_blob)
        weight_blob.extend(w.tobytes())
        b_off = len(weight_blob)
        weight_blob.extend(b.tobytes())
        # fixed 28-byte record: type(B) silu(B) stride(B) pad(B) oc(H) ic(H) kh(B) kw(B) in(H) out(H) w_off(I) b_off(I)
        rec = struct.pack('<4B2H2B2H2I', 1, silu, op['stride'], op['pad'],
                          oc, ic, kh, kw, vid(op['in']), vid(op['out']), w_off, b_off)
        layer_table.extend(rec)
    elif op['type'] == 'concat':
        n_in = len(op['ins'])
        # type(B) n_in(B) pad(H) in_vids...  out(H)
        rec = struct.pack('<2BH', 2, n_in, 0)
        for nm in op['ins']:
            rec += struct.pack('<H', vid(nm))
        rec += struct.pack('<H', vid(op['out']))
        layer_table.extend(rec)
    elif op['type'] == 'maxpool':
        rec = struct.pack('<4B2H2B2H2I', 3, 0, op.get('stride',2), 0,
                          0, 0, 0, 0, vid(op['in']), vid(op['out']), 0, 0)
        layer_table.extend(rec)
    elif op['type'] == 'resize':
        rec = struct.pack('<4B2H2B2H2I', 4, 0, 0, 0,
                          0, 0, 0, 0, vid(op['in']), vid(op['out']), 0, 0)
        layer_table.extend(rec)
    elif op['type'] == 'silu':
        rec = struct.pack('<4B2H2B2H2I', 6, 0, 0, 0,
                          0, 0, 0, 0, vid(op['in']), vid(op['out']), 0, 0)
        layer_table.extend(rec)
    elif op['type'] == 'add':
        rec = struct.pack('<4B2H2B2H2I', 5, 0, 0, 0,
                          0, 0, 0, 0, vid(op['in0']), vid(op['out']), vid(op['in1']), 0)
        layer_table.extend(rec)

# value name table (for input/output identification)
graph_input = graph.input[0].name
# find the final output (the one no op consumes)
all_consumed = set()
for op in ops:
    if 'in' in op: all_consumed.add(op['in'])
    if 'in0' in op: all_consumed.add(op['in0'])
graph_output = None
for op in reversed(ops):
    if op['out'] not in all_consumed:
        graph_output = op['out']
        break

MAGIC = b'AXM1'
header = struct.pack('<4sII', MAGIC, len(ops), len(value_ids))
out = bytearray()
out.extend(header)
out.extend(struct.pack('<HH', vid(graph_input), vid(graph_output)))
out.extend(struct.pack('<I', len(layer_table)))
out.extend(layer_table)
out.extend(struct.pack('<I', len(weight_blob)))
out.extend(weight_blob)

open(OUT, 'wb').write(bytes(out))
print(f'flat model: {len(out)} bytes, {len(ops)} layers, {len(value_ids)} values')
print(f'  convs: {sum(1 for o in ops if "conv" in o["type"])}, other: {sum(1 for o in ops if "conv" not in o["type"])}')
print(f'  input={graph_input}({vid(graph_input)}) output={graph_output}({vid(graph_output)})')
