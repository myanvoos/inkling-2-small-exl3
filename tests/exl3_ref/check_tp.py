"""TP=2 split of real EXL3 experts through the slot layout (slot.py), float64 (TensorFold's numpy EXL3 forward):
  w13: each rank's slot gives exactly its output columns of the whole matrix (the rank's 2 Il interleaved channels);
  w2:  the two ranks' partial outputs add to the whole matrix's output.
Also the slot sizes per K, and the whole artifact's per-rank slot bytes."""
from __future__ import annotations

import numpy as np

import ckpt
import slot
import tf_exl3_format as fmt

H, I, W = 4608, 2304, 2
rng = np.random.default_rng(1234)


def run(layer: int, e: int) -> None:
    w13 = ckpt.exl3_expert(layer, e, "w13")
    w2 = ckpt.exl3_expert(layer, e, "w2")
    k = w13["trellis"].shape[-1] // 16
    x = rng.standard_normal((3, H))
    h = rng.standard_normal((3, I))
    y13 = fmt.forward(x, w13["trellis"], w13["suh"], w13["svh"], k, "mul1")
    y2 = fmt.forward(h, w2["trellis"], w2["suh"], w2["svh"], k, "mul1")
    acc = np.zeros_like(y2)
    Il = I // W
    worst13 = 0.0
    for r in range(W):
        s = slot.pack(w13, w2, H, I, r, W)
        assert len(s) == slot.parts(H, I, W, k)["total"][0]
        u = slot.unpack(s, H, I, W, k)
        y13r = fmt.forward(x, u["w13"]["trellis"], u["w13"]["suh"], u["w13"]["svh"], k, "mul1")
        ref = y13[:, 2 * Il * r:2 * Il * (r + 1)]
        worst13 = max(worst13, float(np.abs(y13r - ref).max() / np.abs(ref).max()))
        acc += fmt.forward(h[:, Il * r:Il * (r + 1)], u["w2"]["trellis"], u["w2"]["suh"], u["w2"]["svh"], k, "mul1")
    worst2 = float(np.abs(acc - y2).max() / np.abs(y2).max())
    print(f"L{layer:02d} e{e:03d} K{k}: w13 rank columns vs whole: max rel {worst13:.2e}; "
          f"w2 rank partials summed vs whole: max rel {worst2:.2e}; slot {slot.parts(H, I, W, k)['total'][0]} B")
    assert worst13 < 1e-12 and worst2 < 1e-12


if __name__ == "__main__":
    for layer, e in [(0, 5), (1, 1), (39, 4), (47, 383)]:
        run(layer, e)
    r = ckpt.receipt()
    n = {2: 0, 3: 0}
    for L in range(48):
        for kk in r["layers"][str(L)]["K_by_expert"]:
            n[kk] += 1
    b = {kk: slot.parts(H, I, W, kk)["total"][0] for kk in (2, 3)}
    tot = n[2] * b[2] + n[3] * b[3]
    print(f"expert-halves a rank: K3 {n[3]} x {b[3]} B + K2 {n[2]} x {b[2]} B = {tot} B = {tot / 2**30:.2f} GiB")
