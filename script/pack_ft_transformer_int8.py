#!/usr/bin/env python3
"""Quantize a trained FTTransformer to the int8 coefficient streams the
hardware (hw/rtl/ft_transformer_top.v) and the golden C model
(src/models/ft_transformer_cmodel.c) consume.

This is the SINGLE SOURCE OF TRUTH for:
  * quantization scales: all weights/biases at Q1.FRAC (FRAC=7); the cls token
    at Q1.RES_FRAC (Q3.5) because it is injected directly into the residual
    stream;
  * the exact wr_addr order each module's write port expects;
  * the derived constants SCALE = round(2^SCALE_FRAC/sqrt(HD)) and
    EPS_V = round(eps*2^(2*FRAC)*D^2), and N_HEADS, written to a manifest so the
    C model call and the RTL parameters cannot drift apart.

It works directly on a state_dict (no need to import the training module / its
heavy deps). Keys follow FTTransformer in src/models/ft_transformer.py:
    tokenizer.weight/bias, cls_token,
    blocks.{b}.norm1.weight/bias, blocks.{b}.attn.in_proj_weight/in_proj_bias,
    blocks.{b}.attn.out_proj.weight/bias, blocks.{b}.norm2.weight/bias,
    blocks.{b}.ffn.0.weight/bias (Linear1), blocks.{b}.ffn.3.weight/bias (Linear2),
    final_norm.weight/bias, head.weight/bias
Block 0's norm1 is present in the state_dict but UNUSED (is_first), so it is
not emitted (the RTL leaves that LayerNorm bank unloaded).

Outputs (to --out-dir):
  manifest.json            config + derived constants
  ft_weights.txt           write stream: one "layer bank sel addr data" per line
                           (decimal; data is signed int8) for the RTL DPI TB
  arrays.npz               (optional, --emit-npz) per-logical-array int8 tensors
                           in c-model (ft_weights) order, for the C self-test

Usage:
    python3 script/pack_ft_transformer_int8.py --synthetic --out-dir /tmp/ftw
    python3 script/pack_ft_transformer_int8.py --checkpoint model.pt --n-heads 4 \
            --out-dir build/ftw
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

# wr_layer codes (must match ft_transformer_top.v LAYER_*)
LAYER_TOK, LAYER_LN, LAYER_MHA, LAYER_FFN, LAYER_HEAD, LAYER_CLS = 0, 1, 2, 3, 4, 5


# ----------------------------------------------------------------------------
# quantization + derived constants (mirror hw/README.md and the c-models)
# ----------------------------------------------------------------------------
def quantize(arr: np.ndarray, frac: int) -> np.ndarray:
    """float -> signed int8 Q1.frac, round-half-up, saturate to [-128, 127]."""
    q = np.floor(np.asarray(arr, dtype=np.float64) * (1 << frac) + 0.5)
    return np.clip(q, -128, 127).astype(np.int64)


def mha_scale(n_heads: int, d_token: int, scale_frac: int) -> int:
    hd = d_token // n_heads
    return int(math.floor((1 << scale_frac) / math.sqrt(hd) + 0.5))


def eps_v(d_token: int, frac: int, eps: float = 1e-5) -> int:
    return int(math.floor(eps * (1 << (2 * frac)) * d_token * d_token + 0.5))


# ----------------------------------------------------------------------------
# state_dict access
# ----------------------------------------------------------------------------
def _np(t) -> np.ndarray:
    """torch.Tensor / ndarray -> float64 ndarray."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t, dtype=np.float64)


def infer_config(sd: dict, n_heads: int) -> dict:
    tw = _np(sd["tokenizer.weight"])          # (F, d_token)
    F, d_token = tw.shape
    d_ffn = _np(sd["blocks.0.ffn.0.weight"]).shape[0]   # (d_ffn, d_token)
    n_blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    if d_token % n_heads != 0:
        raise ValueError(f"d_token={d_token} not divisible by n_heads={n_heads}")
    return dict(F=int(F), d_token=int(d_token), d_ffn=int(d_ffn),
                n_heads=int(n_heads), n_blocks=int(n_blocks))


# ----------------------------------------------------------------------------
# write-stream builder (the canonical coefficient order)
# ----------------------------------------------------------------------------
def build_write_stream(sd: dict, cfg: dict, frac: int, res_frac: int):
    """Return (writes, arrays). `writes` is a list of (layer,bank,sel,addr,data)
    int tuples in load order; `arrays` is the c-model ft_weights dict (int8)."""
    F, D, DF = cfg["F"], cfg["d_token"], cfg["d_ffn"]
    NB = cfg["n_blocks"]
    writes: list[tuple[int, int, int, int, int]] = []
    arrays: dict[str, np.ndarray] = {}

    def emit(layer, bank, sel, flat_int8):
        for addr, v in enumerate(flat_int8):
            writes.append((layer, bank, sel, addr, int(v)))

    # ---- tokenizer (layer TOK, bank 0): weight then bias, addr j*D+k ----
    tw = quantize(_np(sd["tokenizer.weight"]), frac).reshape(F, D)
    tb = quantize(_np(sd["tokenizer.bias"]), frac).reshape(F, D)
    emit(LAYER_TOK, 0, 0, tw.reshape(-1))      # wr_is_bias=0
    emit(LAYER_TOK, 0, 1, tb.reshape(-1))      # wr_is_bias=1
    arrays["tok_w"] = tw.reshape(-1)
    arrays["tok_b"] = tb.reshape(-1)

    # ---- cls (layer CLS, bank 0) at Q1.res_frac ----
    cls = quantize(_np(sd["cls_token"]).reshape(D), res_frac)
    emit(LAYER_CLS, 0, 0, cls)
    arrays["cls"] = cls

    # ---- LayerNorm banks: norm1[b] (b>=1), norm2[b], final ----
    # bank layout (matches ft_transformer_top.v): norm1->b, norm2->NB+b, final->2*NB
    def emit_ln(bank, gkey, bkey):
        g = quantize(_np(sd[gkey]).reshape(D), frac)
        b = quantize(_np(sd[bkey]).reshape(D), frac)
        emit(LAYER_LN, bank, 0, g)             # wr_is_beta=0 (gamma)
        emit(LAYER_LN, bank, 1, b)             # wr_is_beta=1 (beta)
        return g, b

    n1g = np.zeros((NB, D), np.int64); n1b = np.zeros((NB, D), np.int64)
    n2g = np.zeros((NB, D), np.int64); n2b = np.zeros((NB, D), np.int64)
    for b in range(NB):
        if b >= 1:    # block 0 skips norm1 (is_first)
            n1g[b], n1b[b] = emit_ln(b, f"blocks.{b}.norm1.weight", f"blocks.{b}.norm1.bias")
        n2g[b], n2b[b] = emit_ln(NB + b, f"blocks.{b}.norm2.weight", f"blocks.{b}.norm2.bias")
    fng, fnb = emit_ln(2 * NB, "final_norm.weight", "final_norm.bias")
    arrays.update(norm1_g=n1g.reshape(-1), norm1_b=n1b.reshape(-1),
                  norm2_g=n2g.reshape(-1), norm2_b=n2b.reshape(-1),
                  fnorm_g=fng, fnorm_b=fnb)

    # ---- MHA banks: per block, sel 0=ipw 1=ipb 2=opw 3=opb ----
    ipw = np.zeros((NB, 3 * D * D), np.int64); ipb = np.zeros((NB, 3 * D), np.int64)
    opw = np.zeros((NB, D * D), np.int64);     opb = np.zeros((NB, D), np.int64)
    for b in range(NB):
        w_in = quantize(_np(sd[f"blocks.{b}.attn.in_proj_weight"]), frac).reshape(3 * D, D)
        b_in = quantize(_np(sd[f"blocks.{b}.attn.in_proj_bias"]), frac).reshape(3 * D)
        w_out = quantize(_np(sd[f"blocks.{b}.attn.out_proj.weight"]), frac).reshape(D, D)
        b_out = quantize(_np(sd[f"blocks.{b}.attn.out_proj.bias"]), frac).reshape(D)
        emit(LAYER_MHA, b, 0, w_in.reshape(-1))   # rows Wq|Wk|Wv, addr row*D+col
        emit(LAYER_MHA, b, 1, b_in)
        emit(LAYER_MHA, b, 2, w_out.reshape(-1))  # addr oe*D+k
        emit(LAYER_MHA, b, 3, b_out)
        ipw[b], ipb[b], opw[b], opb[b] = w_in.reshape(-1), b_in, w_out.reshape(-1), b_out
    arrays.update(mha_ipw=ipw.reshape(-1), mha_ipb=ipb.reshape(-1),
                  mha_opw=opw.reshape(-1), mha_opb=opb.reshape(-1))

    # ---- FFN banks: per block, sel 0=W1 1=b1 2=W2 3=b2 ----
    w1 = np.zeros((NB, DF * D), np.int64); b1 = np.zeros((NB, DF), np.int64)
    w2 = np.zeros((NB, D * DF), np.int64); b2 = np.zeros((NB, D), np.int64)
    for b in range(NB):
        W1 = quantize(_np(sd[f"blocks.{b}.ffn.0.weight"]), frac).reshape(DF, D)  # addr o*D+k
        B1 = quantize(_np(sd[f"blocks.{b}.ffn.0.bias"]), frac).reshape(DF)
        W2 = quantize(_np(sd[f"blocks.{b}.ffn.3.weight"]), frac).reshape(D, DF)  # addr o*DF+k
        B2 = quantize(_np(sd[f"blocks.{b}.ffn.3.bias"]), frac).reshape(D)
        emit(LAYER_FFN, b, 0, W1.reshape(-1))
        emit(LAYER_FFN, b, 1, B1)
        emit(LAYER_FFN, b, 2, W2.reshape(-1))
        emit(LAYER_FFN, b, 3, B2)
        w1[b], b1[b], w2[b], b2[b] = W1.reshape(-1), B1, W2.reshape(-1), B2
    arrays.update(ffn_w1=w1.reshape(-1), ffn_b1=b1.reshape(-1),
                  ffn_w2=w2.reshape(-1), ffn_b2=b2.reshape(-1))

    # ---- head (layer HEAD, bank 0): weight then bias ----
    hw = quantize(_np(sd["head.weight"]).reshape(D), frac)
    hb = quantize(_np(sd["head.bias"]).reshape(1), frac)
    emit(LAYER_HEAD, 0, 0, hw)                 # wr_is_bias=0
    emit(LAYER_HEAD, 0, 1, hb)                 # wr_is_bias=1
    arrays["head_w"] = hw
    arrays["head_b"] = hb

    return writes, arrays


# ----------------------------------------------------------------------------
# synthetic model (well-scaled random weights; no checkpoint needed)
# ----------------------------------------------------------------------------
def synthetic_state_dict(F=16, d_token=32, d_ffn=64, n_heads=4, n_blocks=3, seed=0):
    rng = np.random.default_rng(seed)
    we = 1.0 / math.sqrt(d_token)
    wf = 1.0 / math.sqrt(d_ffn)
    sd = {
        "tokenizer.weight": rng.normal(0, 0.5, (F, d_token)),
        "tokenizer.bias":   rng.normal(0, 0.1, (F, d_token)),
        "cls_token":        rng.normal(0, 0.3, (1, 1, d_token)),
        "final_norm.weight": 1.0 + rng.normal(0, 0.05, d_token),
        "final_norm.bias":   rng.normal(0, 0.05, d_token),
        "head.weight":       rng.normal(0, we, (1, d_token)),
        "head.bias":         rng.normal(0, 0.05, 1),
    }
    for b in range(n_blocks):
        sd[f"blocks.{b}.norm1.weight"] = 1.0 + rng.normal(0, 0.05, d_token)
        sd[f"blocks.{b}.norm1.bias"]   = rng.normal(0, 0.05, d_token)
        sd[f"blocks.{b}.norm2.weight"] = 1.0 + rng.normal(0, 0.05, d_token)
        sd[f"blocks.{b}.norm2.bias"]   = rng.normal(0, 0.05, d_token)
        sd[f"blocks.{b}.attn.in_proj_weight"] = rng.normal(0, we, (3 * d_token, d_token))
        sd[f"blocks.{b}.attn.in_proj_bias"]   = rng.normal(0, 0.05, 3 * d_token)
        sd[f"blocks.{b}.attn.out_proj.weight"] = rng.normal(0, we, (d_token, d_token))
        sd[f"blocks.{b}.attn.out_proj.bias"]   = rng.normal(0, 0.05, d_token)
        sd[f"blocks.{b}.ffn.0.weight"] = rng.normal(0, we, (d_ffn, d_token))
        sd[f"blocks.{b}.ffn.0.bias"]   = rng.normal(0, 0.05, d_ffn)
        sd[f"blocks.{b}.ffn.3.weight"] = rng.normal(0, wf, (d_token, d_ffn))
        sd[f"blocks.{b}.ffn.3.bias"]   = rng.normal(0, 0.05, d_token)
    return sd


def load_state_dict(path: str) -> dict:
    import torch
    obj = torch.load(path, map_location="cpu")
    # accept a raw state_dict, or a checkpoint dict that wraps one
    if isinstance(obj, dict) and "tokenizer.weight" not in obj:
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    # strip a common "module." / "model_." prefix if present
    if "tokenizer.weight" not in obj:
        for pre in ("module.", "model_.", "model."):
            stripped = {k[len(pre):]: v for k, v in obj.items() if k.startswith(pre)}
            if "tokenizer.weight" in stripped:
                obj = stripped
                break
    if "tokenizer.weight" not in obj:
        raise KeyError("could not find FTTransformer keys (e.g. 'tokenizer.weight') "
                       f"in checkpoint {path}")
    return obj


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="path to a trained FTTransformer .pt/.pth")
    src.add_argument("--synthetic", action="store_true",
                     help="generate well-scaled random weights (no checkpoint)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-heads", type=int, default=4,
                    help="attention heads (not stored in the state_dict; default 4)")
    ap.add_argument("--frac", type=int, default=7)
    ap.add_argument("--res-frac", type=int, default=5)
    ap.add_argument("--scale-frac", type=int, default=14)
    ap.add_argument("--sm-frac", type=int, default=8)
    ap.add_argument("--recip-frac", type=int, default=24)
    ap.add_argument("--emit-npz", action="store_true",
                    help="also dump arrays.npz (c-model ft_weights order)")
    args = ap.parse_args()

    if args.synthetic:
        sd = synthetic_state_dict(n_heads=args.n_heads)
    else:
        sd = load_state_dict(args.checkpoint)
    cfg = infer_config(sd, args.n_heads)
    cfg["seq_len"] = 1 + cfg["F"]

    writes, arrays = build_write_stream(sd, cfg, args.frac, args.res_frac)

    hd = cfg["d_token"] // cfg["n_heads"]
    manifest = dict(
        **cfg, frac_bits=args.frac, res_frac=args.res_frac,
        scale_frac=args.scale_frac, sm_frac=args.sm_frac, recip_frac=args.recip_frac,
        head_dim=hd,
        scale=mha_scale(cfg["n_heads"], cfg["d_token"], args.scale_frac),
        eps_v=eps_v(cfg["d_token"], args.frac),
        n_ln_bank=2 * cfg["n_blocks"] + 1,
        n_writes=len(writes),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, "ft_weights.txt"), "w") as f:
        for layer, bank, sel, addr, data in writes:
            f.write(f"{layer} {bank} {sel} {addr} {data}\n")
    if args.emit_npz:
        np.savez(os.path.join(args.out_dir, "arrays.npz"),
                 **{k: v.astype(np.int8) for k, v in arrays.items()})

    # expected write count (self-check)
    F, D, DF, NB = cfg["F"], cfg["d_token"], cfg["d_ffn"], cfg["n_blocks"]
    n_ln = (NB - 1) + NB + 1           # norm1(b>=1) + norm2 + final
    exp = (2 * F * D) + D + (n_ln * 2 * D) \
        + NB * (3 * D * D + 3 * D + D * D + D) \
        + NB * (DF * D + DF + D * DF + D) + (D + 1)
    status = "OK" if exp == len(writes) else f"MISMATCH (expected {exp})"
    print(f"packed {manifest['n_writes']} writes  [{status}]")
    print(f"config: {cfg}")
    print(f"derived: SCALE={manifest['scale']} EPS_V={manifest['eps_v']} "
          f"HD={hd} n_ln_bank={manifest['n_ln_bank']}")
    print(f"wrote {args.out_dir}/manifest.json, {args.out_dir}/ft_weights.txt"
          + (", arrays.npz" if args.emit_npz else ""))
    return 0 if exp == len(writes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
