"""CPU gate: a view passes the checks exl3.zig's Index.parse and Store.open (tensors()) make, headers only.

    python3 check_index.py VIEW [VIEW ...]

Index: format kortexa-inkling2-exl3-v1, status complete, text_config's shape == config.json's, every layer complete
with K_by_expert in {2, 3}, batches covering [0, E) in order with relative layers/ paths. Files: for every expert of
every layer, w13 / w2 trellis I16 [K/16, N/16, 16K] at its K, suh / svh F16 [K] / [N], mul1 I32 [] == 0x83DCD12D.
Also the per-rank slot bytes at TP2 (exl3.parts).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import slot


def header(p: Path):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        return h, 8 + n


def check(view: Path) -> bool:
    cfg = json.loads((view / "config.json").read_text())
    t = cfg.get("text_config", cfg)
    H, L, E, I = t["hidden_size"], t["num_hidden_layers"], t["n_routed_experts"], t["intermediate_size"]
    r = json.loads((view / "checkpoint.json").read_text())
    errs = []
    if r.get("format") != "kortexa-inkling2-exl3-v1" or r.get("status") != "complete":
        errs.append("format/status")
    tc = r["text_config"]
    if (tc["hidden_size"], tc["num_hidden_layers"], tc["n_routed_experts"], tc["intermediate_size"]) != (H, L, E, I):
        errs.append("text_config shape differs from config.json")
    nbytes = 0
    for l in range(L):
        lo = r["layers"][str(l)]
        kb = lo["K_by_expert"]
        if lo.get("status") != "complete" or len(kb) != E or any(k not in (2, 3) for k in kb):
            errs.append(f"layer {l}: status / K_by_expert")
            continue
        nxt = 0
        for b in lo["batches"]:
            f = b["file"]
            if not f.startswith("layers/") or ".." in f or b["start"] != nxt or not b["start"] < b["end"] <= E:
                errs.append(f"layer {l}: batch {b}")
                break
            nxt = b["end"]
            h, off = header(view / f)
            for e in range(b["start"], b["end"]):
                k = kb[e]
                for proj, (K, N) in (("w13", (H, 2 * I)), ("w2", (I, H))):
                    p = f"experts.{e}.{proj}."
                    want = {"trellis": ("I16", [K // 16, N // 16, 16 * k]), "suh": ("F16", [K]), "svh": ("F16", [N]), "mul1": ("I32", [])}
                    for part, (dt, shape) in want.items():
                        x = h.get(p + part)
                        if x is None or x["dtype"] != dt or x["shape"] != shape:
                            errs.append(f"layer {l} {p}{part}: {x and (x['dtype'], x['shape'])} != {(dt, shape)}")
                    m = h.get(p + "mul1")
                    if m:
                        with (view / f).open("rb") as fh:
                            fh.seek(off + m["data_offsets"][0])
                            if struct.unpack("<I", fh.read(4))[0] != 0x83DCD12D:
                                errs.append(f"layer {l} {p}mul1 marker")
                nbytes += slot.parts(H, I, 2, k)["total"][0] if I % 256 == 0 else 0
        if nxt != E:
            errs.append(f"layer {l}: batches end at {nxt} of {E}")
    ok = not errs
    print(f"{view}: {'PASS' if ok else 'FAIL'} — {L} layers x {E} experts, hidden {H}, inter {I}; TP2 slots a rank "
          f"{nbytes} B ({nbytes / 2**30:.2f} GiB)" + ("" if ok else f"; {len(errs)} problems, first: {errs[:3]}"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([check(Path(v)) for v in sys.argv[1:]]) else 1)
