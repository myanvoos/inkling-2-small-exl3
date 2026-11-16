"""The EXL3 kernel gate: compares tf-inkling2-exl3-check's stage files (OUT_DIR) with the CPU reference.

  make the input:   python3 check_kernels.py make-input X_FILE ROWS [SEED]     (bf16 [ROWS][4608], N(0, 1) * 2)
  run on the GPU:   tf-inkling2-exl3-check VIEW LAYER EXPERT RANK WORLD MODE X_FILE OUT_DIR
  compare:          python3 check_kernels.py compare OUT_DIR X_FILE

Gates (each PASS / FAIL, then OVERALL):
  K1 wq13 / wq2 == TensorFold format.unpack of the rank's slot (bitwise: the decode is ExLlamaV3's codebook exactly)
  K2 xq13 == exl3_ref.act (bitwise)            K3 xr13 == exl3_ref.rotate_in (bitwise)
  K4 z13_g1 == z13_g2 and y_g1 == y_g2 (bitwise: a pair's bits do not depend on the GEMM form)
  K5 z13 vs float64 xr13 @ W_q: max rel <= 1e-5 (fp32 mma accumulation)
  K6 act vs exl3_ref.gateup_act(rotate_out(z13_g1)): <= 1 bf16 ulp where they differ, >= 99% bitwise
  K7 xq2 == exl3_ref.act(act) (bitwise)        K8 xr2 == exl3_ref.rotate_in (bitwise)
  K9 y_g1 vs float64 xr2 @ W_q: max rel <= 1e-5;  K10 y == exl3_ref.rotate_out(y_g1) (bitwise)
  K11 z13_core == z13_g1 and y_core == y_g1 (bitwise: the engine's tf_exl3_grouped_2 is the forms' arithmetic)
  K12 act_p == act and y_p == y (bitwise: the wide-chunk kernels on the engine's prompt hooks, the GEMM + epilogue)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import ckpt
import exl3_ref as ref
import slot
import tf_exl3_format as fmt


def make_input(path: str, rows: int, seed: int = 11, H: int = 4608) -> None:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, H)) * 2).astype(np.float32)
    bits = (x.view(np.uint32) + 0x7FFF + ((x.view(np.uint32) >> 16) & 1)) >> 16     # RTNE to bf16
    Path(path).write_bytes(bits.astype(np.uint16).tobytes())
    print(f"wrote {path}: {rows} x {H} bf16")


def load(out: Path, name: str, dtype, shape):
    return np.fromfile(out / name, dtype=dtype).reshape(shape)


def compare(out_dir: str, x_file: str) -> int:
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text())
    L, e, k, R, r, W, mode = (meta[n] for n in ("layer", "expert", "bits", "rows", "rank", "world", "mode"))
    H, I = meta["hidden"], meta["inter"]
    Il = I // W
    w13 = ckpt.exl3_expert(L, e, "w13")
    w2 = ckpt.exl3_expert(L, e, "w2")
    u = slot.unpack(slot.pack(w13, w2, H, I, r, W), H, I, W, k)
    x = ref.bf16_from_bits(np.fromfile(x_file, dtype=np.uint16).reshape(R, H))
    res = []

    def gate(name, ok, detail):
        res.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")

    wq13 = fmt.unpack(u["w13"]["trellis"], k, "mul1")
    wq2 = fmt.unpack(u["w2"]["trellis"], k, "mul1")
    g13 = load(out, "wq13.f16", np.float16, wq13.shape)
    g2 = load(out, "wq2.f16", np.float16, wq2.shape)
    gate("K1 decode", np.array_equal(g13.view(np.uint16), wq13.view(np.uint16)) and np.array_equal(g2.view(np.uint16), wq2.view(np.uint16)),
         f"w13 {int((g13.view(np.uint16) != wq13.view(np.uint16)).sum())} / {wq13.size} differ, w2 {int((g2.view(np.uint16) != wq2.view(np.uint16)).sum())} / {wq2.size}")
    xq = load(out, "xq13.f16", np.float16, (R, H))
    want = ref.act(x, mode)
    gate("K2 act13", np.array_equal(xq.view(np.uint16), want.view(np.uint16)), f"{int((xq.view(np.uint16) != want.view(np.uint16)).sum())} / {xq.size} differ")
    xr = load(out, "xr13.f16", np.float16, (R, H))
    want = ref.rotate_in(xq, u["w13"]["suh"])
    gate("K3 rot13", np.array_equal(xr.view(np.uint16), want.view(np.uint16)), f"{int((xr.view(np.uint16) != want.view(np.uint16)).sum())} / {xr.size} differ")
    z1 = load(out, "z13_g1.f32", np.float32, (R, 2 * Il))
    z2 = load(out, "z13_g2.f32", np.float32, (R, 2 * Il))
    y1 = load(out, "y_g1.f32", np.float32, (R, H))
    y2 = load(out, "y_g2.f32", np.float32, (R, H))
    gate("K4 forms", np.array_equal(z1.view(np.uint32), z2.view(np.uint32)) and np.array_equal(y1.view(np.uint32), y2.view(np.uint32)),
         f"z13 {int((z1 != z2).sum())} differ, y {int((y1 != y2).sum())} differ")
    zr = xr.astype(np.float64) @ wq13.astype(np.float64)
    rel = float(np.abs(z1 - zr).max() / np.abs(zr).max())
    gate("K5 gemm13", rel <= 1e-5, f"max rel {rel:.2e} vs float64")
    a = load(out, "act.bf16", np.uint16, (R, Il))
    want = ref.gateup_act(ref.rotate_out(z1, u["w13"]["svh"]))
    wb = ref.bf16r(want).view(np.uint32) >> 16
    diff = (a.astype(np.int64) - wb.astype(np.int64))
    same = float((diff == 0).mean())
    gate("K6 gateup epilogue", np.abs(diff).max() <= 1 and same >= 0.99, f"{same * 100:.3f}% bitwise, max {int(np.abs(diff).max())} bf16 ulp")
    av = ref.bf16_from_bits(a)
    xq2 = load(out, "xq2.f16", np.float16, (R, Il))
    want = ref.act(av, mode)
    gate("K7 act2", np.array_equal(xq2.view(np.uint16), want.view(np.uint16)), f"{int((xq2.view(np.uint16) != want.view(np.uint16)).sum())} / {xq2.size} differ (world {W}: own-row amax)")
    xr2 = load(out, "xr2.f16", np.float16, (R, Il))
    want = ref.rotate_in(xq2, u["w2"]["suh"])
    gate("K8 rot2", np.array_equal(xr2.view(np.uint16), want.view(np.uint16)), f"{int((xr2.view(np.uint16) != want.view(np.uint16)).sum())} / {xr2.size} differ")
    yr = xr2.astype(np.float64) @ wq2.astype(np.float64)
    rel = float(np.abs(y1 - yr).max() / np.abs(yr).max())
    gate("K9 gemm2", rel <= 1e-5, f"max rel {rel:.2e} vs float64")
    y = load(out, "y.f32", np.float32, (R, H))
    want = ref.rotate_out(y1, u["w2"]["svh"])
    gate("K10 down epilogue", np.array_equal(y.view(np.uint32), want.view(np.uint32)), f"{int((y.view(np.uint32) != want.view(np.uint32)).sum())} / {y.size} differ")
    if (out / "z13_core.f32").exists():
        zc = load(out, "z13_core.f32", np.float32, (R, 2 * Il))
        yc = load(out, "y_core.f32", np.float32, (R, H))
        gate("K11 engine grouped", np.array_equal(zc.view(np.uint32), z1.view(np.uint32)) and np.array_equal(yc.view(np.uint32), y1.view(np.uint32)),
             f"z13 {int((zc.view(np.uint32) != z1.view(np.uint32)).sum())} differ, y {int((yc.view(np.uint32) != y1.view(np.uint32)).sum())} differ (vs g1)")
        ap = load(out, "act_p.bf16", np.uint16, (R, Il))
        yp = load(out, "y_p.f32", np.float32, (R, H))
        gate("K12 prompt hooks", np.array_equal(ap, a) and np.array_equal(yp.view(np.uint32), y.view(np.uint32)),
             f"act {int((ap != a).sum())} differ, y {int((yp.view(np.uint32) != y.view(np.uint32)).sum())} differ (vs the GEMM + epilogue)")
    ok = all(res)
    print(f"OVERALL {'PASS' if ok else 'FAIL'} ({sum(res)} / {len(res)}): layer {L} expert {e} K{k} rank {r}/{W} mode {mode} rows {R}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "make-input":
        make_input(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 11)
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
