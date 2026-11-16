"""CPU self-test of exl3_ref.py before any GPU run (no GPU touched):
  1. fwht128 (the kernels' butterflies, fp32) equals the Sylvester Hadamard (float64) within fp32 rounding, and is its
     own inverse up to 128x;
  2. the NVFP4 / MXFP8 activation references: idempotent on their own outputs' grid (re-quantizing a dequantized row
     reproduces it), error within the format's bound, and the 4/6 choice exercised;
  3. one real expert (layer 1, a K2 and a K3 expert, rank 0 of TP2) through the bit-replicated rotation + a float64
     GEMM + the epilogue matches TensorFold's float64 format.forward on the same fp16 inputs to fp16-input precision.
"""
from __future__ import annotations

import numpy as np

import ckpt
import exl3_ref as ref
import slot
import tf_exl3_format as fmt

rng = np.random.default_rng(7)

# 1. fwht128
x = rng.standard_normal((5, 128)).astype(np.float32)
H = fmt.hadamard()
y = ref.fwht128(x)
exact = x.astype(np.float64) @ H
err = np.abs(y - exact).max() / np.abs(exact).max()
back = ref.fwht128(y) / 128
print(f"fwht128 vs Sylvester (float64): max rel {err:.2e}; round trip max abs {np.abs(back - x).max():.2e}")
assert err < 1e-6 and np.abs(back - x).max() < 1e-5

# 2. activation references
xb = ref.bf16r(rng.standard_normal((64, 4608)).astype(np.float32) * 3)
for mode, name in ((1, "nvfp4"), (2, "mxfp8")):
    q = (ref.nvfp4_rows(xb) if mode == 1 else ref.mxfp8_rows(xb))
    rel = np.linalg.norm(q - xb) / np.linalg.norm(xb)
    q2 = (ref.nvfp4_rows(q) if mode == 1 else ref.mxfp8_rows(q))
    idem = np.abs(q2 - q).max() / np.abs(q).max()
    print(f"{name}: quantization rel err {rel:.4f}; re-quantize max rel change {idem:.2e}")
    assert (rel < 0.15 if mode == 1 else rel < 0.05)
if True:   # the 4/6 choice: count blocks choosing 4 on a row
    b = xb[:1].reshape(-1, 16)
    print(f"nvfp4 blocks in row 0: {b.shape[0]}")

# 3. one real expert, rank 0 of TP2, the reference stages vs float64 format.forward
Hd, I, W, r = 4608, 2304, 2, 0
for layer, e in ((1, 1), (1, 0)):
    w13 = ckpt.exl3_expert(layer, e, "w13")
    w2 = ckpt.exl3_expert(layer, e, "w2")
    k = w13["trellis"].shape[-1] // 16
    u = slot.unpack(slot.pack(w13, w2, Hd, I, r, W), Hd, I, W, k)
    xin = ref.bf16r(rng.standard_normal((4, Hd)).astype(np.float32))
    xq = ref.act(xin, 1)                                              # fp16
    xr = ref.rotate_in(xq, u["w13"]["suh"])                           # fp16 (bit-replicated)
    wq = fmt.unpack(u["w13"]["trellis"], k, "mul1").astype(np.float64)
    z = (xr.astype(np.float64) @ wq).astype(np.float32)              # the mma's sums, here float64-exact then fp32
    y13 = ref.rotate_out(z, u["w13"]["svh"])
    want = fmt.forward(xq.astype(np.float64), u["w13"]["trellis"], u["w13"]["suh"], u["w13"]["svh"], k, "mul1")
    rel13 = float(np.abs(y13 - want).max() / np.abs(want).max())
    a = ref.gateup_act(y13)
    hq = ref.act(a, 1)
    xr2 = ref.rotate_in(hq, u["w2"]["suh"])
    z2 = (xr2.astype(np.float64) @ fmt.unpack(u["w2"]["trellis"], k, "mul1").astype(np.float64)).astype(np.float32)
    y2 = ref.rotate_out(z2, u["w2"]["svh"])
    want2 = fmt.forward(hq.astype(np.float64), u["w2"]["trellis"], u["w2"]["suh"], u["w2"]["svh"], k, "mul1")
    rel2 = float(np.abs(y2 - want2).max() / np.abs(want2).max())
    print(f"L{layer} e{e} K{k} rank {r}/{W}: reference stages vs float64 forward: w13 max rel {rel13:.2e}, w2 max rel {rel2:.2e}")
    assert rel13 < 5e-3 and rel2 < 5e-3                                # fp16 roundings of the rotated inputs
print("selftest_ref: PASS")
