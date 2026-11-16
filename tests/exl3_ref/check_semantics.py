"""Real-bytes semantics check (CPU, float64): the EXL3 experts decode (TensorFold's own numpy decoder) to the original
release's experts (exactly dequantized NVFP4 / MXFP8) up to the 2-3 bit quantization error, with w13's output columns
in the original's interleaved gate/up order and w2 as the original's transpose. Prints one line per matrix."""
from __future__ import annotations

import sys

import numpy as np

import ckpt
import tf_exl3_format as fmt


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def colcorr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return np.sum(a * b, 0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-300)


def check(layer: int, e: int) -> list[str]:
    out = []
    K = ckpt.receipt()["layers"][str(layer)]["K_by_expert"][e]
    for proj in ("w2", "w13"):
        t = ckpt.exl3_expert(layer, e, proj)
        bits = fmt.bits_of(t["trellis"].shape)
        assert bits == K, (layer, e, proj, bits, K)
        W = fmt.dequantize(t["trellis"], t["suh"], t["svh"], bits, "mul1")      # [in, out]
        O = ckpt.orig_expert(layer, e, proj).T                                  # [in, out]
        assert W.shape == O.shape, (W.shape, O.shape)
        r = rel(W, O)
        c = colcorr(W, O)
        line = (f"L{layer:02d} e{e:03d} {proj:3s} K{K} {'nvfp4' if layer < 40 else 'mxfp8'} shape {W.shape}: "
                f"rel {r:.4f}, column corr min {c.min():.4f} median {np.median(c):.4f}")
        if proj == "w13":
            # the alternative layout: EXL3 columns [gate 0..I) | up [I..2I)] against the original's even / odd rows
            n = W.shape[1] // 2
            alt = np.concatenate([O[:, 0::2], O[:, 1::2]], 1)
            line += f" | stacked-order hypothesis: rel {rel(W, alt):.4f}, corr median {np.median(colcorr(W, alt)):.4f}"
            # the first gate and up channels' columns, in the interleaved order
            line += f" | interleaved: gate cols rel {rel(W[:, 0::2], O[:, 0::2]):.4f}, up cols rel {rel(W[:, 1::2], O[:, 1::2]):.4f}"
        out.append(line)
        del W, O
    return out


if __name__ == "__main__":
    cases = [(0, 0), (1, 0), (1, 1), (40, 0)]
    if len(sys.argv) > 1:
        cases = [tuple(int(v) for v in a.split(":")) for a in sys.argv[1:]]
    for layer, e in cases:
        for line in check(layer, e):
            print(line, flush=True)
