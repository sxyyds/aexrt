"""Create layer-subset test models for binary-search crash isolation."""
import struct, sys
sys.path.insert(0, 'src')
import numpy as np, onnx

m = onnx.load('models/cs2V8_320.onnx')
inits = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
sig_outs = {n.output[0] for n in m.graph.node if n.op_type == 'Sigmoid'}
ops = []
for node in m.graph.node:
    if node.op_type == 'Conv':
        w = inits[node.input[1]]
        b = inits[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0], np.float32)
        a = {x.name: x for x in node.attribute}
        st = a['strides'].ints[0] if 'strides' in a else 1
        pd = a['pads'].ints[0] if 'pads' in a else 0
        ops.append({'t':'conv','w':w.astype(np.float32),'b':b.astype(np.float32),'st':st,'pd':pd,'in':node.input[0],'out':node.output[0]})
    elif node.op_type == 'Mul':
        ops.append({'t':'silu','in':node.input[0],'out':node.output[0]})
    elif node.op_type == 'Concat':
        ops.append({'t':'concat','ins':list(node.input),'out':node.output[0]})
    elif node.op_type == 'MaxPool':
        ops.append({'t':'maxpool','in':node.input[0],'out':node.output[0]})
    elif node.op_type == 'Add':
        ops.append({'t':'add','in0':node.input[0],'in1':node.input[1],'out':node.output[0]})
    elif node.op_type == 'Resize':
        ops.append({'t':'resize','in':node.input[0],'out':node.output[0]})

vid = {}
def V(n):
    if n not in vid: vid[n] = len(vid)
    return vid[n]
for op in ops:
    for k in ('in','in0','in1','out'):
        if k in op: V(op[k])
    if 'ins' in op:
        for n in op['ins']: V(n)

for N in [4, 8, 16, 32, 64, 96, 153]:
    sub = ops[:N]
    in_vals = set()
    for op in sub:
        for k in ('in','in0','in1'):
            if k in op: in_vals.add(op[k])
    last_out = sub[-1].get('out')
    wb = bytearray()
    lt = bytearray()
    for op in sub:
        if op['t'] == 'conv':
            w, b = op['w'], op['b']
            oc, ic, kh, kw = w.shape
            wo = len(wb); wb.extend(w.tobytes())
            bo = len(wb); wb.extend(b.tobytes())
            lt.extend(struct.pack('<4B2H2B2H2I', 1, 0, op['st'], op['pd'], oc, ic, kh, kw, V(op['in']), V(op['out']), wo, bo))
        elif op['t'] == 'silu':
            lt.extend(struct.pack('<4B2H2B2H2I', 6, 0, 0, 0, 0, 0, 0, 0, V(op['in']), V(op['out']), 0, 0))
        elif op['t'] == 'add':
            lt.extend(struct.pack('<4B2H2B2H2I', 5, 0, 0, 0, 0, 0, 0, 0, V(op['in0']), V(op['out']), V(op['in1']), 0))
        elif op['t'] == 'maxpool':
            lt.extend(struct.pack('<4B2H2B2H2I', 3, 0, 2, 0, 0, 0, 0, 0, V(op['in']), V(op['out']), 0, 0))
        elif op['t'] == 'resize':
            lt.extend(struct.pack('<4B2H2B2H2I', 4, 0, 0, 0, 0, 0, 0, 0, V(op['in']), V(op['out']), 0, 0))
        elif op['t'] == 'concat':
            n = len(op['ins'])
            lt.extend(struct.pack('<2BH', 2, n, 0))
            for nm in op['ins']: lt.extend(struct.pack('<H', V(nm)))
            lt.extend(struct.pack('<H', V(op['out'])))
    o = bytearray()
    o.extend(b'AXM1')
    o.extend(struct.pack('<I', len(sub)))
    o.extend(struct.pack('<I', len(vid)))
    o.extend(struct.pack('<H', V('images')))
    o.extend(struct.pack('<H', V(last_out)))
    o.extend(struct.pack('<I', len(lt)))
    o.extend(lt)
    o.extend(struct.pack('<I', len(wb)))
    o.extend(wb)
    fn = f'build/tmp/sub_{N}.bin'
    open(fn, 'wb').write(bytes(o))
    print(f'{N} layers -> {fn} ({len(o)} bytes)')
