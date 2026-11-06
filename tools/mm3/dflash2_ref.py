# tools/mm3/dflash2_ref.py - the torch reference for the Metal DFlash2 draft
# (src/metal/mm3_dflash2.cpp), following SGLang's dflash.py semantics and this repo's
# GLM DFlash2 port (src/metal/glm_metal_dflash.cpp).  It is also the fixture generator:
# `fixture(outdir)` writes a tiny random-weights draft in the REAL checkpoint layout
# (config.json + model.safetensors, bare BF16 tensor names), the target emb/head pair
# the test hands the draft, and the expected context/proposal outputs
# (expected.bin / expected.json) that metal_mm3_dflash2_test checks against.
#
# Semantics pinned here (see DFLASH2-SPEC.md): plain Qwen3 RMSNorm (no +1), fc over the
# SIX dflash_config.target_layer_ids taps -> hidden_norm -> per-layer K/V only for
# context; one parallel block pass for proposals, every attention/MLP sublayer wrapped
# in the grouped dynamic conv (prepare on the sublayer input, finish on its output,
# coefficients from one projection of the input); non-causal window attention; the
# candidate selector's greedy lattice walk with lattice-softmax confidences.
import json
import math
from pathlib import Path

import torch


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w


def rope(x, pos, n_rot, theta):
    half = n_rot // 2
    inv = theta ** (-torch.arange(half).float() * 2.0 / n_rot)
    ang = pos.float().unsqueeze(-1) * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    a, b = x[..., :half], x[..., half:n_rot]
    out = x.clone()
    out[..., :half] = a * cos - b * sin
    out[..., half:n_rot] = b * cos + a * sin
    return out


def grouped_conv(x, delta, base, block_size, group):
    # SGLang _grouped_conv: x [B][D], delta [B][taps][G], base = one side [taps][D]
    B, D = x.shape
    taps = base.shape[0]
    G = D // group
    blocks = x.view(B, G, group)
    coef = base.view(1, taps, G, group) + delta.unsqueeze(-1)   # [B][taps][G][group]
    acc = coef[:, 0] * blocks
    pos_in_block = torch.arange(B) % block_size
    for tap in range(1, taps):
        shifted = torch.zeros_like(blocks)
        shifted[tap:] = blocks[:-tap]
        shifted = torch.where((pos_in_block >= tap).view(-1, 1, 1), shifted, torch.zeros_like(shifted))
        acc = acc + coef[:, tap] * shifted
    return acc.reshape(B, D)


class Draft:
    def __init__(self, tensors, cfg):
        self.t = tensors
        self.D = cfg["hidden_size"]
        self.V = cfg["vocab_size"]
        self.n_layers = cfg["num_hidden_layers"]
        self.n_head = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.n_rot = self.hd
        self.eps = cfg["rms_norm_eps"]
        self.theta = float(cfg["rope_theta"])
        self.window = cfg.get("sliding_window", 4096)
        dc = cfg["dflash_config"]
        self.B = dc["block_size"]
        self.mask = dc["mask_token_id"]
        self.taps = dc["target_layer_ids"]
        self.NT = len(self.taps)
        self.group = dc["conv_group_size"]
        self.ctaps = dc["conv_kernel_size"]
        self.rank = dc["selector_rank"]
        self.topk = dc["selector_top_k"]
        self.kc = [None] * self.n_layers
        self.vc = [None] * self.n_layers

    def W(self, name):
        return self.t[name].float()

    def _grow(self, l, upto):
        kv = self.n_kv * self.hd
        for name in ("kc", "vc"):
            cur = getattr(self, name)[l]
            if cur is None or cur.shape[0] < upto:
                n = max(upto, 1024 if cur is None else cur.shape[0] * 2)
                new = torch.zeros(n, kv)
                if cur is not None:
                    new[: cur.shape[0]] = cur
                getattr(self, name)[l] = new

    def context(self, taps, pos0):
        # taps [NT][T][D] -> per-layer K/V at positions pos0..pos0+T-1 (no attention)
        T = taps.shape[1]
        fc = self.W("fc.weight")
        acc = torch.zeros(T, self.D)
        for j in range(self.NT):
            acc = acc + taps[j] @ fc[:, j * self.D:(j + 1) * self.D].t()
        g = rms(acc, self.W("hidden_norm.weight"), self.eps)
        pos = torch.arange(pos0, pos0 + T)
        for l in range(self.n_layers):
            P = f"layers.{l}."
            k = (g @ self.W(P + "self_attn.k_proj.weight").t()).reshape(T, self.n_kv, self.hd)
            v = (g @ self.W(P + "self_attn.v_proj.weight").t()).reshape(T, self.n_kv * self.hd)
            k = rope(rms(k, self.W(P + "self_attn.k_norm.weight"), self.eps), pos.unsqueeze(1), self.n_rot, self.theta)
            self._grow(l, pos0 + T)
            self.kc[l][pos0:pos0 + T] = k.reshape(T, self.n_kv * self.hd)
            self.vc[l][pos0:pos0 + T] = v

    def _conv(self, l, which, x, side):
        # one grouped conv: coefficients from kernel_projection(x); delta side picks
        P = f"layers.{l}.{which}_conv."
        G = self.D // self.group
        dyn = (x @ self.W(P + "kernel_projection.weight").t()).reshape(x.shape[0], 2, self.ctaps, G)
        return grouped_conv(x, dyn[:, side], self.W(P + "base_kernel")[side], self.B, self.group), dyn

    def propose(self, emb, head, anchor, P, n):
        B, D = self.B, self.D
        ids = torch.tensor([anchor] + [self.mask] * (B - 1))
        x = emb.float()[ids].clone()
        pos = torch.arange(P, P + B)
        hi = P + B - 1
        lo = max(0, hi - self.window + 1)
        Twin = hi + 1 - lo
        for l in range(self.n_layers):
            Pr = f"layers.{l}."
            xn = rms(x, self.W(Pr + "input_layernorm.weight"), self.eps)
            xc, dyn = self._conv(l, "attention", xn, 0)
            q = (xc @ self.W(Pr + "self_attn.q_proj.weight").t()).reshape(B, self.n_head, self.hd)
            k = (xc @ self.W(Pr + "self_attn.k_proj.weight").t()).reshape(B, self.n_kv, self.hd)
            v = (xc @ self.W(Pr + "self_attn.v_proj.weight").t()).reshape(B, self.n_kv * self.hd)
            q = rope(rms(q, self.W(Pr + "self_attn.q_norm.weight"), self.eps), pos.unsqueeze(1), self.n_rot, self.theta)
            k = rope(rms(k, self.W(Pr + "self_attn.k_norm.weight"), self.eps), pos.unsqueeze(1), self.n_rot, self.theta)
            self._grow(l, P + B)
            self.kc[l][P:P + B] = k.reshape(B, self.n_kv * self.hd)
            self.vc[l][P:P + B] = v.reshape(B, self.n_kv * self.hd)
            K = self.kc[l][lo:hi + 1].reshape(Twin, self.n_kv, self.hd).repeat_interleave(self.n_head // self.n_kv, dim=1)
            V = self.vc[l][lo:hi + 1].reshape(Twin, self.n_kv, self.hd).repeat_interleave(self.n_head // self.n_kv, dim=1)
            s = torch.einsum("bhd,thd->bht", q, K) / math.sqrt(self.hd)
            a = torch.softmax(s, dim=-1)
            att = torch.einsum("bht,thd->bhd", a, V).reshape(B, self.n_head * self.hd)
            o = att @ self.W(Pr + "self_attn.o_proj.weight").t()
            o = grouped_conv(o, dyn[:, 1], self.W(Pr + "attention_conv.base_kernel")[1], self.B, self.group)
            x = x + o
            xn = rms(x, self.W(Pr + "post_attention_layernorm.weight"), self.eps)
            fc_, dyn2 = self._conv(l, "mlp", xn, 0)
            gt = fc_ @ self.W(Pr + "mlp.gate_proj.weight").t()
            up = fc_ @ self.W(Pr + "mlp.up_proj.weight").t()
            m = (torch.nn.functional.silu(gt) * up) @ self.W(Pr + "mlp.down_proj.weight").t()
            m = grouped_conv(m, dyn2[:, 1], self.W(Pr + "mlp_conv.base_kernel")[1], self.B, self.group)
            x = x + m
        hid = rms(x, self.W("norm.weight"), self.eps)
        logits = hid.to(torch.bfloat16).to(torch.float32) @ head.float().t()

        # the candidate selector: greedy lattice walk over the mask rows' top-K
        S = min(n, B - 1)
        A = self.W("candidate_selector.predecessor_codebook")
        Bc = self.W("candidate_selector.successor_codebook")
        hp = self.W("candidate_selector.hidden_projection.weight")
        cand, unary = [], []
        for s in range(1, B):
            vals, idx = torch.topk(logits[s], self.topk)
            cand.append(idx)
            unary.append(vals)
        drafts, confs = [], []
        pick_prev = -1
        for e in range(S):
            gate = hid[e + 1] @ hp.t()                       # [rank]
            keys = Bc[cand[e]]                               # [K][rank]
            if e == 0:
                pv = A[anchor] * gate
                sc = unary[e] + keys @ pv                    # [K]
            else:
                preds = A[cand[e - 1]] * gate.unsqueeze(0)   # [K][rank] per predecessor
                sc = (unary[e].unsqueeze(0) + preds @ keys.t())[pick_prev]
            pick = int(torch.argmax(sc))
            conf = float(1.0 / torch.exp(sc - sc.max()).sum())
            drafts.append(int(cand[e][pick]))
            confs.append(conf)
            pick_prev = pick
        return drafts, confs, hid, logits


def load_st(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def fixture(outdir):
    g = torch.Generator().manual_seed(20261102)
    rnd = lambda *s: torch.randn(*s, generator=g) * 0.05
    nrm = lambda *s: 1.0 + rnd(*s)
    D, V, L, H, KV, HD, FF = 256, 512, 2, 4, 1, 64, 512
    B, W, NT, R, K, GROUP, CT = 4, 5, 3, 32, 8, 16, 2
    cfg = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3",
        "hidden_size": D, "intermediate_size": FF, "num_hidden_layers": L,
        "num_attention_heads": H, "num_key_value_heads": KV, "head_dim": HD,
        "vocab_size": V, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "sliding_window": W, "max_position_embeddings": 1048576,
        "tie_word_embeddings": False,
        "dflash_config": {"block_size": B, "conv_group_size": GROUP, "conv_kernel_size": CT,
                          "mask_token_id": 509, "selector_rank": R, "selector_top_k": K,
                          "target_layer_ids": [1, 3, 5]},
    }
    t = {}

    def bf16(name, x):
        t[name] = x.to(torch.bfloat16)

    bf16("fc.weight", rnd(D, NT * D) * 4)
    bf16("hidden_norm.weight", nrm(D))
    bf16("norm.weight", nrm(D))
    bf16("candidate_selector.hidden_projection.weight", rnd(R, D))
    bf16("candidate_selector.predecessor_codebook", rnd(V, R))
    bf16("candidate_selector.successor_codebook", rnd(V, R))
    for l in range(L):
        P = f"layers.{l}."
        bf16(P + "input_layernorm.weight", nrm(D))
        bf16(P + "post_attention_layernorm.weight", nrm(D))
        bf16(P + "self_attn.q_norm.weight", nrm(HD))
        bf16(P + "self_attn.k_norm.weight", nrm(HD))
        bf16(P + "self_attn.q_proj.weight", rnd(HD * H, D))
        bf16(P + "self_attn.k_proj.weight", rnd(HD * KV, D))
        bf16(P + "self_attn.v_proj.weight", rnd(HD * KV, D))
        bf16(P + "self_attn.o_proj.weight", rnd(D, HD * H))
        bf16(P + "mlp.gate_proj.weight", rnd(FF, D))
        bf16(P + "mlp.up_proj.weight", rnd(FF, D))
        bf16(P + "mlp.down_proj.weight", rnd(D, FF))
        for which in ("attention_conv", "mlp_conv"):
            base = rnd(2, CT, D)
            base[:, 0, :] += 1.0
            bf16(P + f"{which}.base_kernel", base)
            bf16(P + f"{which}.kernel_projection.weight", rnd(2 * CT * (D // GROUP), D))
    emb = rnd(V, D).to(torch.bfloat16)
    head = rnd(V, D).to(torch.bfloat16)
    out = Path(outdir)
    (out / "dflash2").mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(t, str(out / "dflash2" / "model.safetensors"), metadata={"format": "pt"})
    save_file({"emb": emb, "head": head}, str(out / "emb_head.safetensors"), metadata={"format": "pt"})
    (out / "dflash2" / "config.json").write_text(json.dumps(cfg, indent=1))

    d = Draft(t, cfg)
    taps = torch.randn(NT, 6, D, generator=g) * 0.5          # [NT][T][D], positions 0..5
    d.context(taps, 0)                                       # all 6 in one call; the test splits 4+2
    drafts, confs, hid, logits = d.propose(emb, head, anchor=37, P=6, n=B)
    exp = {"taps": taps.tolist(), "anchor": 37, "pos": 6, "n": B,
           "drafts": drafts, "confs": confs, "hid": hid.tolist(), "logits": logits.tolist()}
    (out / "expected.json").write_text(json.dumps(exp))
    import struct as _st
    S = len(drafts)
    blob = b"".join([
        taps.numpy().astype("float32").tobytes(),
        hid.numpy().astype("float32").tobytes(),
        logits.numpy().astype("float32").tobytes(),
        _st.pack("<%di" % S, *drafts),
        _st.pack("<%df" % S, *confs),
    ])
    (out / "expected.bin").write_bytes(blob)
    print("fixture written:", out, "drafts:", drafts, "confs:", [round(c, 4) for c in confs])


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "fixture":
        fixture(sys.argv[2])
    else:
        print("usage: dflash2_ref.py fixture <dir>")
