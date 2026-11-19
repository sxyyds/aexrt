#!/usr/bin/env python3
"""Generate the cooperative-matrix microbenchmark SPIR-V module.

The NDK's glslc/spirv-as predate GL_KHR_cooperative_matrix, so the module is
emitted word-by-word here. Two variants:

  fp32 16x16x16   A/B/C float32 (VK_COMPONENT_TYPE_FLOAT32_KHR = 1)
  fp16 16x32x32   A/B float16 (0), C float32 (1)

Each workgroup runs 32 chained OpCooperativeMatrixMulAddKHR ops; the host
varies the workgroup count to time matrix-unit throughput.

  py mobile/tools/mk_coopmat_spv.py fp32 build/tmp/coopmat_f32.spv
  py mobile/tools/mk_coopmat_spv.py fp16 build/tmp/coopmat_f16.spv
"""
import struct
import sys

SPV_MAGIC = 0x07230203
import os
_ver = os.environ.get("AEXRT_COOP_SPV_VERSION", "1.6")
SPV_VERSION = {"1.3": 0x00010300, "1.4": 0x00010400, "1.5": 0x00010500, "1.6": 0x00010600}[_ver]
_LS = int(os.environ.get("AEXRT_COOP_LS", "32"))
_NOSTRIDE = os.environ.get("AEXRT_COOP_NOSTRIDE") == "1"
_LAYOUT = int(os.environ.get("AEXRT_COOP_LAYOUT", "0"))
_TYPE_ENC = os.environ.get("AEXRT_COOP_TYPE_ENC", "ids")
_VMM = os.environ.get("AEXRT_COOP_VMM") == "1"
_SPEC = os.environ.get("AEXRT_COOP_SPEC") == "1"

OP_CAPABILITY = 17
OP_EXTENSION = 10
OP_MEMORY_MODEL = 14
OP_ENTRY_POINT = 15
OP_EXECUTION_MODE = 16
OP_TYPE_VOID = 19
OP_TYPE_INT = 21
OP_TYPE_FLOAT = 22
OP_TYPE_RUNTIME_ARRAY = 29
OP_TYPE_STRUCT = 30
OP_TYPE_POINTER = 32
OP_TYPE_FUNCTION = 33
OP_CONSTANT = 43
OP_FUNCTION = 54
OP_FUNCTION_END = 56
OP_VARIABLE = 59
OP_ACCESS_CHAIN = 65
OP_DECORATE = 71
OP_MEMBER_DECORATE = 72
OP_LABEL = 248
OP_RETURN = 253
OP_TYPE_COOPMAT = 4456
OP_COOPMAT_LOAD = 4457
OP_COOPMAT_STORE = 4458
OP_COOPMAT_MULADD = 4459

CAP_SHADER = 1
CAP_COOPMAT = 6022
SCOPE_SUBGROUP = 3
USE_A, USE_B, USE_ACC = 0, 1, 2
STORAGE_STORAGE_BUFFER = 12
EXEC_GLCOMPUTE = 5
EXEC_MODE_LOCAL_SIZE = 17
DEC_BLOCK = 2
DEC_ARRAY_STRIDE = 6
DEC_DESCRIPTOR_SET = 34
DEC_BINDING = 33

MULS = 32


def string_words(s):
    b = s.encode("utf-8") + b"\0"
    b += b"\0" * ((4 - len(b) % 4) % 4)
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


class Gen:
    def __init__(self):
        self.words = []

    def op(self, opcode, *operands):
        self.words.append(((1 + len(operands)) << 16) | opcode)
        self.words.extend(operands)

    def op_str(self, opcode, *operands_then_string):
        # string operands are always last in the instructions we emit
        *ops, s = operands_then_string
        sw = string_words(s)
        self.words.append(((1 + len(ops) + len(sw)) << 16) | opcode)
        self.words.extend(ops)
        self.words.extend(sw)


def main():
    fmt = sys.argv[1] if len(sys.argv) > 1 else "fp16"
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    out = sys.argv[5] if len(sys.argv) > 5 else "build/tmp/coopmat.spv"
    mul = int(sys.argv[6]) if len(sys.argv) > 6 else 32
    mode = sys.argv[7] if len(sys.argv) > 7 else "mma"
    if fmt == "fp32":
        comp = 1
    elif fmt == "fp16":
        comp = 0
    else:
        raise SystemExit("format must be fp32 or fp16")

    g = Gen()
    # ids
    I = 1
    def nid():
        nonlocal I
        v = I
        I += 1
        return v
    void, f32, f16, u32, fn = nid(), nid(), nid(), nid(), nid()
    main_id = nid()
    c0, c1, c2, c3, cM, cN, cK, cL, cf0 = (nid() for _ in range(9))
    matA, matB, matC = nid(), nid(), nid()
    arr, blk, ptr_blk, ptr_comp = nid(), nid(), nid(), nid()
    va, vb, vc = nid(), nid(), nid()
    lbl = nid()
    pa, va_id = nid(), None
    A = nid()
    pb, B = nid(), nid()
    pc, C0 = nid(), nid()
    chain = []
    for _ in range(mul):
        chain.append(nid())

    g.op(OP_CAPABILITY, CAP_SHADER)
    g.op(OP_CAPABILITY, CAP_COOPMAT)
    if comp == 0:
        g.op(OP_CAPABILITY, 4423)   # Float16
        g.op(OP_CAPABILITY, 4433)   # StorageBuffer16BitAccess
        g.op_str(OP_EXTENSION, "SPV_KHR_16bit_storage")
    g.op_str(OP_EXTENSION, "SPV_KHR_cooperative_matrix")
    if _VMM:
        g.op(OP_CAPABILITY, 5345)   # VulkanMemoryModel
        g.op_str(OP_EXTENSION, "SPV_KHR_vulkan_memory_model")
        g.op(OP_MEMORY_MODEL, 0, 3)  # Logical, VulkanKHR
    else:
        g.op(OP_MEMORY_MODEL, 0, 1)  # Logical, GLSL450
    g.op(OP_ENTRY_POINT, EXEC_GLCOMPUTE, main_id, *string_words("main"), va, vb, vc)
    g.op(OP_EXECUTION_MODE, main_id, EXEC_MODE_LOCAL_SIZE, _LS, 1, 1)
    g.op(OP_DECORATE, blk, DEC_BLOCK)
    g.op(OP_MEMBER_DECORATE, blk, 0, 35, 0)   # member 0 Offset 0
    g.op(OP_DECORATE, arr, DEC_ARRAY_STRIDE, 4 if comp == 1 else 2)
    for vid, binding in ((va, 0), (vb, 1), (vc, 2)):
        g.op(OP_DECORATE, vid, DEC_DESCRIPTOR_SET, 0)
        g.op(OP_DECORATE, vid, DEC_BINDING, binding)
    g.op(OP_TYPE_VOID, void)
    g.op(OP_TYPE_FLOAT, f32, 32)
    if comp == 0:
        g.op(OP_TYPE_FLOAT, f16, 16)
    g.op(OP_TYPE_INT, u32, 32, 0)
    g.op(OP_TYPE_FUNCTION, fn, void)
    for cid, val in ((c0, 0), (c1, 1), (c2, 2), (c3, SCOPE_SUBGROUP),
                     (cM, m), (cN, n), (cK, k), (cL, _LAYOUT)):
        g.op(50 if _SPEC else OP_CONSTANT, u32, cid, val)   # OpSpecConstant(50) or OpConstant
    g.op(OP_CONSTANT, f32, cf0, 0)
    ctype = f32 if comp == 1 else f16
    if _TYPE_ENC == "lit_all":
        g.op(OP_TYPE_COOPMAT, matA, ctype, SCOPE_SUBGROUP, m, k, USE_A)
        g.op(OP_TYPE_COOPMAT, matB, ctype, SCOPE_SUBGROUP, k, n, USE_B)
        g.op(OP_TYPE_COOPMAT, matC, f32, SCOPE_SUBGROUP, m, n, USE_ACC)
    elif _TYPE_ENC == "lit_rc":
        g.op(OP_TYPE_COOPMAT, matA, ctype, c3, m, k, c0)
        g.op(OP_TYPE_COOPMAT, matB, ctype, c3, k, n, c1)
        g.op(OP_TYPE_COOPMAT, matC, f32, c3, m, n, c2)
    else:
        g.op(OP_TYPE_COOPMAT, matA, ctype, c3, cM, cK, c0)
        g.op(OP_TYPE_COOPMAT, matB, ctype, c3, cK, cN, c1)
        g.op(OP_TYPE_COOPMAT, matC, f32, c3, cM, cN, c2)
    g.op(OP_TYPE_RUNTIME_ARRAY, arr, ctype)
    g.op(OP_TYPE_STRUCT, blk, arr)
    g.op(OP_TYPE_POINTER, ptr_blk, STORAGE_STORAGE_BUFFER, blk)
    g.op(OP_TYPE_POINTER, ptr_comp, STORAGE_STORAGE_BUFFER, ctype)
    for vid in (va, vb, vc):
        g.op(OP_VARIABLE, ptr_blk, vid, STORAGE_STORAGE_BUFFER)
    g.op(OP_FUNCTION, void, main_id, 0, fn)
    g.op(OP_LABEL, lbl)
    if mode == "typeonly":
        ptr_m = nid(); var_m = nid()
        g.op(OP_TYPE_POINTER, ptr_m, 7, matA)   # Function storage
        g.op(OP_VARIABLE, ptr_m, var_m, 7)
        g.op(OP_RETURN)
        g.op(OP_FUNCTION_END)
        header = [SPV_MAGIC, SPV_VERSION, 0, I, 0]
        with open(out, "wb") as fh:
            fh.write(struct.pack("<%dI" % len(header), *header))
            fh.write(struct.pack("<%dI" % len(g.words), *g.words))
        print(f"wrote {out}: typeonly words={len(header)+len(g.words)}")
        return
    g.op(OP_ACCESS_CHAIN, ptr_comp, pa, va, c0, c0)
    if mode == "plain":
        g.op(61, f32, A, pa)          # OpLoad
        g.op(62, pa, A)               # OpStore (dummy, keeps the path alive)
        g.op(OP_RETURN)
        g.op(OP_FUNCTION_END)
        header = [SPV_MAGIC, SPV_VERSION, 0, I, 0]
        with open(out, "wb") as fh:
            fh.write(struct.pack("<%dI" % len(header), *header))
            fh.write(struct.pack("<%dI" % len(g.words), *g.words))
        print(f"wrote {out}: plain skeleton words={len(header)+len(g.words)}")
        return
    g.op(OP_COOPMAT_LOAD, matA, A, pa, cL) if _NOSTRIDE else g.op(OP_COOPMAT_LOAD, matA, A, pa, cL, cK)
    g.op(OP_ACCESS_CHAIN, ptr_comp, pb, vb, c0, c0)
    if mode == "load":
        g.op(OP_COOPMAT_STORE, pa, A, cL, cK)
        g.op(OP_RETURN)
        g.op(OP_FUNCTION_END)
        header = [SPV_MAGIC, SPV_VERSION, 0, I, 0]
        with open(out, "wb") as fh:
            fh.write(struct.pack("<%dI" % len(header), *header))
            fh.write(struct.pack("<%dI" % len(g.words), *g.words))
        print(f"wrote {out}: load/store only words={len(header)+len(g.words)}")
        return
    g.op(OP_COOPMAT_LOAD, matB, B, pb, cL) if _NOSTRIDE else g.op(OP_COOPMAT_LOAD, matB, B, pb, cL, cN)
    g.op(OP_ACCESS_CHAIN, ptr_comp, pc, vc, c0, c0)
    if os.environ.get("AEXRT_COOP_NOCLOAD") == "1":
        g.op(44, matC, C0, cf0)   # OpCompositeConstruct splat
    else:
        g.op(OP_COOPMAT_LOAD, matC, C0, pc, cL) if _NOSTRIDE else g.op(OP_COOPMAT_LOAD, matC, C0, pc, cL, cN)
    prev = C0
    for cid in chain:
        g.op(OP_COOPMAT_MULADD, matC, cid, A, B, prev, 0)
        prev = cid
    g.op(OP_COOPMAT_STORE, pc, prev, cL) if _NOSTRIDE else g.op(OP_COOPMAT_STORE, pc, prev, cL, cN)
    g.op(OP_RETURN)
    g.op(OP_FUNCTION_END)

    header = [SPV_MAGIC, SPV_VERSION, 0, I, 0]
    with open(out, "wb") as fh:
        fh.write(struct.pack("<%dI" % len(header), *header))
        fh.write(struct.pack("<%dI" % len(g.words), *g.words))
    print(f"wrote {out}: fmt={fmt} {m}x{n}x{k} comp={'f32' if comp==1 else 'f16'} "
          f"muls={mul} words={len(header)+len(g.words)}")


if __name__ == "__main__":
    main()
