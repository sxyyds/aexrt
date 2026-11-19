"""Export a YOLOv8-family ONNX graph to the AEXRT-M flat binary (AXM1 v2).

Header (116 bytes):
  0  'AXM1' | nlayers(I) | nvals(I) | in_vid(H) | out_vid(H)
  16 nc(H) | nlevels(H) | lt_sz(I)
  24 input_size(H) | pad(H)
  28 3 levels x (box_vid, cls_vid, stride, pad) = 24B
  52 dfl weights 16 x f32
Then per-value f32 scale table (nvals), layer table (lt_sz), blob size (I),
fp16 weight blob, 4 KB tail guard.

Ops: conv(1) concat(2) maxpool(3) resize(4) add(5) silu(6) slice(7) sigmoid(8)

Fusion is connectivity-based (not node-order adjacency): Conv->Sigmoid->Mul is
folded into the conv (act=1), Conv->Sigmoid into act=2, and a lone SiLU that
cannot fold remains a unary silu layer. Every emitted layer is executable by
the runtime; anything unsupported raises instead of being silently dropped.
"""
import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import shape_inference

ROOT = Path(__file__).resolve().parents[2]
MAGIC = b"AXM1"
WIN = False
INT8 = False


def die(msg):
    raise SystemExit("export: " + msg)


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return a
    return default


def ints_attr(node, name, default):
    a = attr(node, name)
    return list(a.ints) if a is not None else list(default)


def int_attr(node, name, default):
    a = attr(node, name)
    return int(a.i) if a is not None else default


def parse_graph(model, cut_name):
    graph = model.graph
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in graph.initializer}
    inferred = shape_inference.infer_shapes(model)
    shapes = {}
    for v in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    ops = []
    cut = False
    for node in graph.node:
        if cut_name and node.name == cut_name:
            cut = True
            break
        op = {"node": node, "t": None, "out": node.output[0] if node.output else None}
        if node.op_type == "Conv":
            w = inits[node.input[1]]
            if w.ndim != 4:
                die(f"{node.name}: non-2D conv weight {w.shape}")
            b = inits[node.input[2]] if len(node.input) > 2 else np.zeros(w.shape[0], np.float32)
            op.update(t="conv", w=w.astype(np.float32), b=b.astype(np.float32),
                      stride=ints_attr(node, "strides", [1])[0],
                      pad=ints_attr(node, "pads", [0, 0, 0, 0])[0],
                      group=int_attr(node, "group", 1),
                      in0=node.input[0], oc=int(w.shape[0]), ic=int(w.shape[1]) * int_attr(node, "group", 1),
                      kh=int(w.shape[2]), kw=int(w.shape[3]), act=0)
        elif node.op_type == "Sigmoid":
            op.update(t="sigmoid", in0=node.input[0])
        elif node.op_type == "Mul":
            op.update(t="mul", ins=list(node.input))
        elif node.op_type == "Concat":
            op.update(t="concat", ins=list(node.input))
        elif node.op_type == "Split":
            op.update(t="split", ins=[node.input[0]], outs=list(node.output))
        elif node.op_type == "MaxPool":
            op.update(t="maxpool", in0=node.input[0],
                      kh=ints_attr(node, "kernel_shape", [1, 1])[0],
                      kw=ints_attr(node, "kernel_shape", [1, 1])[1],
                      stride=ints_attr(node, "strides", [1])[0],
                      pad=ints_attr(node, "pads", [0, 0, 0, 0])[0])
        elif node.op_type == "Resize":
            scales = None
            if len(node.input) > 2 and node.input[2] in inits:
                scales = inits[node.input[2]].astype(np.float64)
            if scales is not None and len(scales) == 4:
                fy, fx = float(scales[2]), float(scales[3])
            else:
                si, so = shapes.get(node.input[0]), shapes.get(node.output[0])
                if not si or not so or not si[2] or not si[3]:
                    die(f"{node.name}: Resize size cannot be inferred")
                fy = so[2] / si[2]
                fx = so[3] / si[3]
            if fy != fx or abs(fy - round(fy)) > 1e-6:
                die(f"{node.name}: only isotropic integer resize supported ({fy},{fx})")
            op.update(t="resize", in0=node.input[0], scale=int(round(fy)))
        elif node.op_type == "Add":
            op.update(t="add", in0=node.input[0], ins=list(node.input))
        elif node.op_type == "Reshape":
            # Pure view: AEXRT-M stores position-major, so a flatten/reshape
            # that is not consumed by a layout-sensitive op is free.
            op.update(t="reshape", in0=node.input[0], ins=list(node.input))
        else:
            die(f"{node.name or op['out']}: unsupported op type {node.op_type} before cut")
        ops.append(op)
    if cut_name and not cut:
        die(f"cut node {cut_name} not found")
    return graph, inits, ops


def consumers(graph):
    cons = defaultdict(list)
    for n in graph.node:
        for i in n.input:
            cons[i].append(n)
    return cons


def fuse_activations(ops, cons):
    """Fold Conv->Sigmoid(->Mul) using tensor connectivity; rewrite lone SiLU/Sigmoid.

    Returns the surviving op list and a map from final value name to producer op.
    """
    by_node = {id(op["node"]): op for op in ops}
    dead = set()
    for op in ops:
        if op["t"] != "conv":
            continue
        x = op["out"]
        cx = cons.get(x, [])
        sigs = [n for n in cx if n.op_type == "Sigmoid"]
        if len(sigs) != 1:
            continue
        s = sigs[0]
        if by_node.get(id(s)) is None:
            continue                      # activation lives past the export cut
        sy = s.output[0]
        mul = next((n for n in cx if n.op_type == "Mul" and set(n.input) == {x, sy}), None)
        others = [n for n in cx if id(n) != id(s) and (mul is None or id(n) != id(mul))]
        if mul is not None and not others and len(cons.get(sy, [])) == 1:
            op["act"] = 1                 # conv writes x*sigmoid(x) directly
            op["out"] = mul.output[0]
            dead.add(id(s))
            dead.add(id(mul))
        elif len(cx) == 1:
            op["act"] = 2                 # conv writes sigmoid(x)
            op["out"] = sy
            dead.add(id(s))

    surviving = [op for op in ops if id(op["node"]) not in dead]

    # Rewrite surviving Sigmoid+Mul pairs into a single unary silu layer. This
    # handles interleaved head branches where the conv itself could not fold.
    sig_out = defaultdict(list)
    for op in surviving:
        if op["t"] == "sigmoid":
            sig_out[op["out"]].append(op)
    new_ops = []
    removed = set()
    for op in surviving:
        if op["t"] != "mul":
            new_ops.append(op)
            continue
        a, b = op["ins"]
        cand = None
        for src, other in ((a, b), (b, a)):
            src_ops = sig_out.get(src, [])
            if len(src_ops) == 1 and src_ops[0]["in0"] == other:
                cand = (src_ops[0], other)
                break
        if cand is None:
            die(f"{op['out']}: Mul is not a SiLU pattern; unsupported")
        sig, x = cand
        if len(cons.get(sig["out"], [])) == 1:
            sig["t"] = "silu"             # reuse the node slot as x*sigmoid(x)
            sig["in0"] = x
            sig["out"] = op["out"]
        else:
            # sigmoid output is shared: keep the sigmoid, add an independent
            # x*sigmoid(x) producer for the Mul consumers.
            new_ops.append({"node": op["node"], "t": "silu", "in0": x, "out": op["out"]})
        removed.add(id(op))
    # `removed` entries are Mul ops that were already appended above; drop them.
    final = [op for op in new_ops if id(op) not in removed]
    return final


def prune_ops(ops, keep_values):
    """Dead-code elimination for the exported subgraph (head concat/reshape)."""
    need = set(keep_values)
    kept = []
    for op in reversed(ops):
        outs = op["outs"] if op["t"] == "split" else [op["out"]]
        if any(o in need for o in outs):
            kept.append(op)
            if "in0" in op:
                need.add(op["in0"])
            for i in op.get("ins", []):
                need.add(i)
    return list(reversed(kept))


def expand_and_size(ops):
    """Expand Split into slice views, then compute channel counts in order."""
    out = []
    ch = {"images": 3}
    for op in ops:
        if op["t"] == "split":
            cin = ch.get(op["ins"][0], 0)
            n = len(op["outs"])
            if cin == 0 or cin % n:
                die(f"{op['out']}: cannot split {cin} channels into {n}")
            half = cin // n
            start = 0
            for o in op["outs"]:
                out.append({"t": "slice", "in0": op["ins"][0], "out": o,
                            "c_in": cin, "c_out": half, "start": start})
                ch[o] = half
                start += half
            continue
        out.append(op)
        if op["t"] == "conv":
            ch[op["out"]] = op["oc"]
        elif op["t"] == "concat":
            op["c_ins"] = [ch.get(n, 0) for n in op["ins"]]
            if any(c == 0 for c in op["c_ins"]):
                die(f"{op['out']}: concat input without channel count {op['ins']}")
            ch[op["out"]] = sum(op["c_ins"])
        elif op["t"] == "add":
            ch[op["out"]] = ch[op["in0"]]
        elif op["t"] in ("maxpool", "resize", "sigmoid", "silu", "reshape"):
            ch[op["out"]] = ch[op["in0"]]
        if op["t"] != "slice":
            for k in ("in0",):
                if k in op and op[k] not in ch:
                    die(f"{op['out']}: missing input channels for {op[k]}")
    return out, ch


def pack_layers(ops, ch, vid):
    wb = bytearray()
    lt = bytearray()

    def align(n):
        while len(wb) % n:
            wb.append(0)

    for _op_i, op in enumerate(ops):
        t = op["t"]
        if __import__("os").environ.get("AEXRT_PACK_DEBUG"):
            print("LT", _op_i, t, "at", len(lt))
        if t == "conv":
            if op["group"] != 1:
                die(f"{op['out']}: grouped conv not supported yet (group={op['group']})")
            w, b = op["w"], op["b"]
            oc, ic, kh, kw = w.shape
            if INT8:
                Cpad = 4 if ic < 4 else ic
                R = Cpad * kh * kw
                Rp = (R + 3) & ~3
                wt = w.transpose(0, 2, 3, 1).reshape(oc, ic * kh * kw)
                amax_oc = np.abs(wt).max(axis=1)
                sw = np.where(amax_oc > 0, amax_oc / 127.0, 1.0).astype(np.float32)
                q = np.rint(wt / sw[:, None]).clip(-127, 127).astype(np.int8)
                qw = np.zeros((oc, Rp), np.int8)
                for kk in range(kh * kw):
                    qw[:, kk * Cpad:kk * Cpad + ic] = q[:, kk * ic:(kk + 1) * ic]
                align(64)
                wo = len(wb)
                wb.extend(qw.tobytes())
                align(4)
                bo = len(wb)
                wb.extend(b.astype(np.float32).tobytes())
                align(4)
                so = len(wb)
                wb.extend(sw.tobytes())
                lt.extend(struct.pack("<4B2H2B2H3I", 11, op.get("act", 0), op["stride"], op["pad"],
                                      oc, ic, kh, kw, vid(op["in0"]), vid(op["out"]),
                                      wo, bo // 4, so // 4))
                continue
            if (WIN and kh == 3 and kw == 3 and op["stride"] == 1 and op["pad"] == 1
                    and ic % 4 == 0 and oc % 4 == 0):
                # Winograd F(2x2,3x3): U = G g G^T, layout [p][ic][oc] so the
                # four OC lanes of a thread load one f16vec4 per transform
                # point. fp16 storage, fp32 transforms/accumulation.
                G = np.array([[1.0, 0.0, 0.0],
                              [0.5, 0.5, 0.5],
                              [0.5, -0.5, 0.5],
                              [0.0, 0.0, 1.0]], np.float32)
                U = np.einsum("ij,ocjk,lk->ocil", G, w.astype(np.float32), G)
                wt = U.transpose(2, 3, 1, 0).reshape(16, ic, oc)
                align(64)
                wo = len(wb)
                import os as _os
                if _os.environ.get("AEXRT_PACK_DEBUG"):
                    print("WINO pack", op["out"], "len", len(wb), "wo", wo, "shape", wt.shape)
                wb.extend(wt.astype(np.float16).tobytes())
                align(4)
                bo = len(wb)
                wb.extend(b.astype(np.float32).tobytes())
                lt.extend(struct.pack("<4B2H2B2H3I", 10, op.get("act", 0), 1, 1,
                                      oc, ic, 3, 3, vid(op["in0"]), vid(op["out"]),
                                      wo, bo // 4, 0))
                continue
            if ic < 4:
                # Pad each tap to 4 channels so the stem kernel can issue one
                # f16vec4 weight load per tap per OC row (3 real + 1 zero).
                Cpad = 4
                wt = w.transpose(0, 2, 3, 1).reshape(oc, ic * kh * kw)
                Rp = kh * kw * Cpad
                wp = np.zeros((oc, Rp), np.float16)
                for kk in range(kh * kw):
                    wp[:, kk * Cpad:kk * Cpad + ic] = wt[:, kk * ic:(kk + 1) * ic]
            else:
                R = ic * kh * kw
                Rp = (R + 3) & ~3
                # position-major [oc][kh][kw][ic]; f16vec4-friendly rows
                wt = w.transpose(0, 2, 3, 1).reshape(oc, R)
                wp = np.zeros((oc, Rp), np.float16)
                wp[:, :R] = wt.astype(np.float16)
            align(64)
            wo = len(wb)
            import os as _os
            if _os.environ.get("AEXRT_PACK_DEBUG"):
                print("CONV pack", op["out"], "ic", ic, "oc", oc, "len", len(wb), "wo", wo, "Rp", Rp)
            wb.extend(wp.tobytes())
            align(4)
            bo = len(wb)
            wb.extend(b.astype(np.float32).tobytes())
            lt.extend(struct.pack("<4B2H2B2H3I", 1, op.get("act", 0), op["stride"], op["pad"],
                                  oc, ic, kh, kw, vid(op["in0"]), vid(op["out"]),
                                  wo, bo // 4, 0))
        elif t == "concat":
            n = len(op["ins"])
            lt.extend(struct.pack("<2BH", 2, n, 0))
            for nm, c in zip(op["ins"], op["c_ins"]):
                lt.extend(struct.pack("<2H", vid(nm), c))
            lt.extend(struct.pack("<2H", vid(op["out"]), ch[op["out"]]))
        elif t == "maxpool":
            c = ch[op["out"]]
            lt.extend(struct.pack("<4B2H2B2H2I", 3, 0, op["stride"], op["pad"],
                                  c, c, op["kh"], op["kw"],
                                  vid(op["in0"]), vid(op["out"]), 0, 0))
        elif t == "resize":
            c = ch[op["out"]]
            lt.extend(struct.pack("<4B2H2B2H2I", 4, 0, op["scale"], 0,
                                  c, c, 0, 0, vid(op["in0"]), vid(op["out"]), 0, 0))
        elif t == "add":
            c = ch[op["out"]]
            lt.extend(struct.pack("<4B2H2B2H2I", 5, 0, 0, 0,
                                  c, c, 0, 0, vid(op["in0"]), vid(op["out"]), vid(op["ins"][1]), 0))
        elif t == "silu":
            c = ch[op["out"]]
            lt.extend(struct.pack("<4B2H2B2H2I", 6, 0, 0, 0,
                                  c, c, 0, 0, vid(op["in0"]), vid(op["out"]), 0, 0))
        elif t == "sigmoid":
            c = ch[op["out"]]
            lt.extend(struct.pack("<4B2H2B2H2I", 8, 0, 0, 0,
                                  c, c, 0, 0, vid(op["in0"]), vid(op["out"]), 0, 0))
        elif t in ("slice", "reshape"):
            if t == "slice":
                c_out, c_in, start = op["c_out"], op["c_in"], op["start"]
            else:                          # identity view (output unused in our plan)
                c_out = c_in = ch[op["out"]]
                start = 0
            lt.extend(struct.pack("<4B2H2B2H2I", 7, 0, 0, 0,
                                  c_out, c_in, 0, 0,
                                  vid(op["in0"]), vid(op["out"]), start, 0))
        else:
            die(f"cannot pack op {t}")
    return lt, wb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(ROOT / "models/cs2V8_320.onnx"))
    ap.add_argument("--out-dir", default=str(ROOT / "build/tmp"))
    ap.add_argument("--cut", default="/model.22/Concat_2",
                    help="stop before this ONNX node (head concat); empty exports the whole graph")
    ap.add_argument("--subsets", default="16,32")
    ap.add_argument("--winograd", action="store_true")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    global WIN, INT8
    WIN = args.winograd and not args.int8
    INT8 = args.int8
    model = onnx.load(args.model)
    graph, inits, ops = parse_graph(model, args.cut)
    cons = consumers(graph)
    ops = fuse_activations(ops, cons)

    # Decoder metadata is only meaningful for the full graph with a YOLO head.
    decoder = None
    if args.cut:
        def out_of(node_name):
            for n in graph.node:
                if n.name == node_name:
                    return n.output[0]
            die(f"decoder node {node_name} not found")
        level_names = []
        for L in range(3):
            level_names.append((out_of(f"/model.22/cv2.{L}/cv2.{L}.2/Conv"),
                                out_of(f"/model.22/cv3.{L}/cv3.{L}.2/Conv")))
        dfl = inits["model.22.dfl.conv.weight"].ravel().astype(np.float32)
        assert dfl.size == 16
        nc = 0
        for _, cls in level_names:
            for o in ops:
                if o["out"] == cls:
                    nc = o["oc"]
        if nc == 0:
            die("could not determine class count from head convs")
        decoder = {"nc": nc, "levels": level_names, "dfl": dfl}

    if decoder:
        keep = [v for pair in decoder["levels"] for v in pair]
        before = len(ops)
        ops = prune_ops(ops, keep)
        print(f"prune: {before} -> {len(ops)} layers (head postprocess removed)")
    ops, ch = expand_and_size(ops)
    print(f"ops: {len(ops)} " + str({t: sum(1 for o in ops if o['t'] == t)
                                     for t in sorted({o['t'] for o in ops})}))

    # Value ids cover exactly the emitted values plus decoder references.
    vid = {}

    def V(name):
        if name not in vid:
            vid[name] = len(vid)
        return vid[name]

    for op in ops:
        if "in0" in op:
            V(op["in0"])
        V(op["out"])
        if op["t"] == "concat":
            for nm in op["ins"]:
                V(nm)
        if op["t"] == "add":
            V(op["ins"][1])

    levels = []
    if decoder:
        for b, c in decoder["levels"]:
            levels.append({"box_vid": V(b), "cls_vid": V(c), "stride": 0})
        V(b)

    input_name = graph.input[0].name
    input_shape = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
    input_size = input_shape[2]
    if input_shape[3] != input_size:
        die(f"non-square input {input_shape}")

    # Per-value activation scale table (W8A8 groundwork; fp16 path ignores it).
    amax = {}
    for cand in (outdir / "act_amax.json", outdir.parent / "act_amax.json"):
        try:
            amax = json.load(open(cand))
            break
        except OSError:
            pass
    name_of = {v: k for k, v in vid.items()}

    # W8A8 activation plan: per-value amax, unified across concat/add/view
    # boundaries so the whole graph shares one int8 scale per buffer.
    vscale = [1.0] * len(vid)
    vdtype = [0] * len(vid)
    if INT8:
        a0 = [max(float(amax.get(name_of.get(v, ""), 0.0)), 1e-6)
              for v in range(len(vid))]
        a = list(a0)

        def bump(v, val):
            if val > a[v] + 1e-6:
                a[v] = val
                return True
            return False

        def unify(vs, floor=0.0):
            """Force one scale on an aliasing/in-place group.

            Slice views, zero-copy concat and the byte-wise int8 elementwise
            kernels move raw int8 words without rescaling, so the whole group
            must share a scale.  Add needs a scale that covers a0+a1.  Iterate
            on the *current* a (not a0) so downstream constraints flow back to
            producers.
            """
            m = max(max(a[v] for v in vs), floor)
            ch = False
            for v in vs:
                if a[v] < m - 1e-6:
                    a[v] = m
                    ch = True
            return ch

        for _ in range(16):
            changed = False
            for op in ops:
                t = op["t"]
                if t in ("slice", "reshape", "maxpool", "resize"):
                    iv, ov = V(op["in0"]), V(op["out"])
                    changed |= unify((iv, ov))
                elif t == "concat":
                    vs = [V(n) for n in op["ins"]] + [V(op["out"])]
                    changed |= unify(vs)
                elif t == "add":
                    i0, i1 = V(op["in0"]), V(op["ins"][1])
                    changed |= unify((i0, i1, V(op["out"])), a0[i0] + a0[i1])
            if not changed:
                break
        in_id = V(input_name)
        for v in range(len(vid)):
            vscale[v] = a[v] / 127.0 * 1.15
            vdtype[v] = 0 if v == in_id else 1
        if __import__("os").environ.get("AEXRT_I8_DEBUG"):
            for op in ops:
                if op["t"] not in ("slice", "concat", "add"):
                    continue
                ins = [V(n) for n in op.get("ins", [])]
                iv = V(op["in0"]) if "in0" in op else -1
                ov = V(op["out"])
                print("PLAN", op["t"], iv, ins, "->", ov,
                      "a_io", [round(a0[x], 2) for x in [iv] + ins + [ov]],
                      "a_unified", [round(a[x], 2) for x in [iv] + ins + [ov]])

    def write_model(fn, sub_ops, with_decoder):
        sub_ch = {}
        # channel map is already complete; subset only selects a prefix
        lt, wb = pack_layers(sub_ops, ch, V)
        last_out = sub_ops[-1]["out"] if sub_ops else input_name
        o = bytearray()
        o.extend(b"AXM3" if INT8 else MAGIC)
        o.extend(struct.pack("<I", len(sub_ops)))
        o.extend(struct.pack("<I", len(vid)))
        o.extend(struct.pack("<H", V(input_name)))
        o.extend(struct.pack("<H", 0xFFFF if with_decoder else V(last_out)))
        o.extend(struct.pack("<H", decoder["nc"] if with_decoder else 0))
        o.extend(struct.pack("<H", 3 if with_decoder else 0))
        o.extend(struct.pack("<I", len(lt)))
        o.extend(struct.pack("<H", input_shape[2]))
        o.extend(struct.pack("<H", input_shape[3]))
        if with_decoder:
            strides = (8, 16, 32)
            for L, (b, c) in zip(range(3), decoder["levels"]):
                o.extend(struct.pack("<4H", V(b), V(c), strides[L], 0))
            o.extend(decoder["dfl"].tobytes())
        else:
            o.extend(b"\0" * (24 + 64))
        if INT8:
            for i in range(len(vid)):
                o.extend(struct.pack("<f", vscale[i]))
            o.extend(bytes(vdtype))
            o.extend(bytes((-len(vid)) % 4))
        else:
            for i in range(len(vid)):
                nm = name_of.get(i, "")
                o.extend(struct.pack("<f", amax.get(nm, 0.0) / 127.0 * 1.10 if nm else 1.0))
        o.extend(lt)
        o.extend(struct.pack("<I", len(wb)))
        o.extend(wb)
        o.extend(b"\0" * 4096)
        Path(fn).write_bytes(bytes(o))
        print(f"{fn}: {len(sub_ops)} layers, {len(o)} bytes")
        return o

    # Subset models (debug / host tests) and the full model.
    for n in [int(x) for x in args.subsets.split(",") if x]:
        n = min(n, len(ops))
        write_model(outdir / f"sub_{n}.bin", ops[:n], False)
    write_model(outdir / "cs2v8_full.bin", ops, decoder is not None)
    json.dump(vid, open(outdir / "vid_map.json", "w"))
    if decoder:
        json.dump({"nc": decoder["nc"], "input_size": input_size,
                   "levels": [{"box": b, "cls": c, "stride": s}
                              for (b, c), s in zip(decoder["levels"], (8, 16, 32))],
                   "cut": args.cut},
                  open(outdir / "decoder.json", "w"), indent=1)


if __name__ == "__main__":
    main()
