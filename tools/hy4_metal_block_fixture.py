#!/usr/bin/env python3
"""Stage 3 fixture: a complete-block Hy4-Flash GGUF set (seeded real data, not byte patterns),
packed by the real tools/hy4/pack_gguf.py.  Extends tools/test_hy4_pack_gguf.py's model()
with every tensor a trunk forward reads:

  layer 0: indexed (non-SWA) + dense FFN      layer 1: SWA + MoE + Engram module 0
  layer 2: SWA + MoE                          layer 3: the MTP block (metadata only here)

Geometry overrides vs the packer test's toy (all forced by the C++ loader's own rules or by
wanting the sparse paths exercised at toy positions):
  expert_latent_length 128 -> 256   (hy4_pack.hpp geo(): n_lat % 256 == 0)
  indexer.top_k 2048 -> 8           (top = 8/4 - 1 = 1 pool: selection truncates at pos >= 8)
  sliding_window 512 -> 4           (the SWA window truncates at pos >= 4)

Weights: deterministic seeded gaussians.  Matrices Q8_0 (hand-rolled quantizer below),
norms / router / hyper-connection / Engram small tensors F32, the Engram table F8_E4M3.
The routed experts are Q8_0 here: stage 2 already proved ggml-metal's IQ2_XXS / IQ3_XXS
dequant + matvec bit-true against ggml-cpu, and the block forward treats expert tensors as
opaque mul_mat operands; stage 6 runs the real IQ pack through the same code path.
"""
import importlib.util
import math
import pathlib
import random
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

spec = importlib.util.spec_from_file_location("packtest", HERE / "test_hy4_pack_gguf.py")
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)

ARCH, D, HD, ROWS = pt.ARCH, pt.D, pt.HD, pt.ROWS
NL = 3
E, VOCAB = D, 269
HEADS = [64, 32, 32]
QL, KVL, KMLA, VMLA, ROPE = 128, 64, 96, 64, 32
NFF, NEXP, NUSED, NFFEXP, NLAT = 512, 4, 2, 256, 256
IXH, IXD = 4, 128
HC = 4

rng = random.Random(20261119)


def gauss(n, s):
    return [rng.gauss(0.0, s) for _ in range(n)]


def q8_0(vals, k):
    """vals: row-major [rows][k] floats -> Q8_0 bytes (block 32: fp16 scale + 32 int8)."""
    assert k % 32 == 0 and len(vals) % k == 0
    out = bytearray()
    for r in range(0, len(vals), k):
        row = vals[r:r + k]
        for b in range(0, k, 32):
            blk = row[b:b + 32]
            amax = max(abs(v) for v in blk) or 1e-9
            d = amax / 127.0
            out += struct.pack("<e", d)
            out += bytes((max(-127, min(127, round(v / d))) & 0xFF) for v in blk)
    return bytes(out)


def f8_e4m3(vals):
    """Encode floats as F8_E4M3 bytes (round-to-nearest-even on the mantissa)."""
    out = bytearray()
    for v in vals:
        if v == 0.0:
            out.append(0)
            continue
        sign = 0x80 if v < 0 else 0
        a = abs(v)
        if a < 2.0 ** -6:  # subnormal: step 2^-9, exponent field 0
            m = round(a / 2.0 ** -9)
            out.append(sign | (0x08 if m >= 8 else m))
            continue
        e = math.floor(math.log2(a))
        # normal range: bias 7, mantissa 3 bits; clamp into [2^-6, 448]
        e = max(-6, min(8, e))
        while a / (2.0 ** e) >= 2.0:
            e += 1
        while a / (2.0 ** e) < 1.0:
            e -= 1
        m = round((a / (2.0 ** e) - 1.0) * 8)
        if m == 8:
            m, e = 0, e + 1
        if e + 7 >= 15:
            out.append(sign | 0x7E)  # 448, the largest finite
        else:
            out.append(sign | ((e + 7) << 3) | m)
    return bytes(out)


def gguf_real(path, kv, tensors):
    """GGUF v3 like the packer test's writer, but tensors carry real data bytes.
    tensors: [(name, ggml type id, dims ne0-first, data bytes)]"""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def val(t, v):
        if t == 8:
            return s(v)
        if t == 9:
            at, items = v
            return struct.pack("<IQ", at, len(items)) + b"".join(val(at, x) for x in items)
        return struct.pack("<" + {2: "H", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q"}[t], v)

    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, (t, v) in kv.items():
        head += s(k) + struct.pack("<I", t) + val(t, v)
    off, infos = 0, b""
    for name, typ, dims, data in tensors:
        infos += s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims) \
            + struct.pack("<IQ", typ, off)
        off += (len(data) + 31) // 32 * 32
    head += infos
    head += b"\0" * ((32 - len(head) % 32) % 32)
    with path.open("wb") as f:
        f.write(head)
        for _, _, _, data in tensors:
            f.write(data + b"\0" * ((32 - len(data) % 32) % 32))


F32, Q8_0, F8 = 0, 8, 43


def mat(name, k, n, s=0.05, typ=Q8_0):
    vals = gauss(k * n, s)
    if typ == Q8_0:
        return (name, Q8_0, [k, n], q8_0(vals, k))
    return (name, F32, [k, n], struct.pack("<%df" % len(vals), *vals))


def mat3(name, k, n, b, s=0.05):
    vals = gauss(k * n * b, s)
    return (name, Q8_0, [k, n, b], q8_0(vals, k))


def vec(name, n, s=0.05, base=0.0):
    vals = [base + v for v in gauss(n, s)]
    return (name, F32, [n], struct.pack("<%df" % n, *vals))


def block(il):
    p = "blk.%d." % il
    H = HEADS[il]
    t = [
        vec(p + "attn_norm.weight", E, 0.02, 1.0), vec(p + "ffn_norm.weight", E, 0.02, 1.0),
        vec(p + "post_attention_norm.weight", E, 0.02, 1.0), vec(p + "post_ffw_norm.weight", E, 0.02, 1.0),
        vec(p + "attn_q_a_norm.weight", QL, 0.02, 1.0), vec(p + "attn_kv_a_norm.weight", KVL, 0.02, 1.0),
        vec(p + "attn_sinks.weight", H, 0.5),
        mat(p + "attn_q_a.weight", E, QL), mat(p + "attn_q_b.weight", QL, H * KMLA),
        mat(p + "attn_kv_a_mqa.weight", E, KMLA),
        mat3(p + "attn_k_b.weight", KVL, KVL, H), mat3(p + "attn_v_b.weight", KVL, VMLA, H),
        mat(p + "attn_output.weight", H * VMLA, E), mat(p + "attn_gate.weight", E, H * VMLA),
        mat(p + "hc_attn_fn.weight", HC * E, 2 * HC, 0.02, F32), vec(p + "hc_attn_base.weight", 2 * HC, 0.1),
        vec(p + "hc_attn_scale.weight", 2, 0.1, 1.0),
        mat(p + "hc_ffn_fn.weight", HC * E, 2 * HC, 0.02, F32), vec(p + "hc_ffn_base.weight", 2 * HC, 0.1),
        vec(p + "hc_ffn_scale.weight", 2, 0.1, 1.0),
    ]
    if il == 0:  # the indexed layer
        t += [
            mat(p + "indexer.attn_q_b.weight", QL, IXH * IXD), mat(p + "indexer.attn_k.weight", E, IXD),
            mat(p + "indexer_compressor_value.weight", IXD, IXD), mat(p + "indexer_compressor_gate.weight", IXD, IXD),
            mat(p + "indexer.proj.weight", E, IXH), vec(p + "indexer.k_norm.weight", IXD, 0.02, 1.0),
            mat(p + "indexer_compressor_ape.weight", IXD, 4, 0.1, F32),
            vec(p + "indexer_compressor_norm.weight", IXD, 0.02, 1.0),
            mat(p + "ffn_gate.weight", E, NFF), mat(p + "ffn_up.weight", E, NFF),
            mat(p + "ffn_down.weight", NFF, E),
        ]
    else:
        t += [
            mat(p + "ffn_gate_inp.weight", E, NEXP, 0.3, F32), vec(p + "exp_probs_b.bias", NEXP, 0.2),
            mat(p + "ffn_routed_down.weight", E, NLAT), mat(p + "ffn_routed_up.weight", NLAT, E),
            mat(p + "ffn_gate_shexp.weight", E, NFFEXP), mat(p + "ffn_up_shexp.weight", E, NFFEXP),
            mat(p + "ffn_down_shexp.weight", NFFEXP, E),
            mat3(p + "ffn_gate_exps.weight", NLAT, NFFEXP, NEXP), mat3(p + "ffn_up_exps.weight", NLAT, NFFEXP, NEXP),
            mat3(p + "ffn_down_exps.weight", NFFEXP, NLAT, NEXP),
        ]
    if il == 1:  # the Engram layer (module 0)
        cols = 4
        t += [
            mat(p + "engram_kv.weight", cols * HD, (HC + 1) * E),
            vec(p + "engram_kv_b.weight", (HC + 1) * E, 0.02),
            mat(p + "engram_gate.weight", E, HC, 0.05, F32),
            vec(p + "engram_lookup_norm.weight", cols * HD, 0.02, 1.0),
            vec(p + "engram_pre_gamma.weight", E, 0.02, 1.0),
            mat(p + "engram_hc_scale.weight", E, HC, 0.05, F32),
        ]
    return t


def main():
    args = sys.argv[1:]
    out = pathlib.Path(args[args.index("--out") + 1]) if "--out" in args else \
        pathlib.Path(args[0]) if args else ROOT / "build-hy4-fixture"
    src = out / "gguf"
    src.mkdir(parents=True, exist_ok=True)
    # the packer test's model() writes the metadata set (and its 8 pattern tensors, which the
    # files below then replace wholesale); reuse its kv by rebuilding with the overrides.
    a = lambda k: ARCH + "." + k  # noqa: E731
    over = {
        a("expert_latent_length"): (4, NLAT),
        a("attention.indexer.top_k"): (4, 8),
        a("attention.sliding_window"): (4, 4),
    }
    probe = out / "probe"
    captured = {}
    orig_gguf = pt.gguf

    def cap(path, kv, tensors):
        if "general.architecture" in kv:
            captured.update(kv)
        return orig_gguf(path, kv, tensors)

    pt.gguf = cap
    try:
        pt.model(probe, **over)
    finally:
        pt.gguf = orig_gguf
    kv = dict(captured)
    assert kv, "kv capture failed"

    globals_ = [
        mat("token_embd.weight", E, VOCAB, 0.3, F32),
        mat("output.weight", E, VOCAB, 0.05, Q8_0),
        vec("output_norm.weight", E, 0.02, 1.0),
        mat("output_hc_fn.weight", HC * E, HC, 0.02, F32),
        vec("output_hc_base.weight", HC, 0.1),
        vec("output_hc_scale.weight", 1, 0.1, 1.0),
    ]
    table = [rng.uniform(-1.0, 1.0) for _ in range(HD * ROWS)]
    ple = ("per_layer_token_embd.weight", F8, [HD, ROWS], f8_e4m3(table))

    l0, l1, l2 = block(0), block(1), block(2)
    gguf_real(src / "Hy4-flash-Q4_K_M-00001-of-00002.gguf", kv, globals_ + l0 + l1)
    gguf_real(src / "Hy4-flash-Q4_K_M-00002-of-00002.gguf", {"split.count": (2, 2)}, l2 + [ple])
    r = subprocess.run([sys.executable, str(HERE / "hy4" / "pack_gguf.py"), "--model", str(src)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        return r.returncode
    pack = src / "maya-hy4"
    print("BLOCK_FIXTURE_PACK", pack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
