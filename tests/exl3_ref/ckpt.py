"""Read-only access to the kortexa EXL3 Inkling-2 checkpoint and the original release, header by header.

Every read is a pread of one tensor's byte range followed by POSIX_FADV_DONTNEED on that range, so the page cache
does not keep checkpoint pages (the GB10 hosts share it with the GPU). Nothing here writes to either checkpoint.
"""
from __future__ import annotations

import json
import os
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

EXL3 = Path(os.path.expanduser("~/models/inkling-2-small-exl3-2.59bpw"))
ORIG = Path(os.path.expanduser("~/models/inkling-2-small"))
MUL1 = 0x83DCD12D

_DT = {"F16": np.float16, "F32": np.float32, "I16": np.int16, "I32": np.int32, "U8": np.uint8,
       "BF16": np.uint16, "F8_E4M3": np.uint8, "I64": np.int64}


@lru_cache(maxsize=256)
def header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def raw(path: str | Path, name: str) -> tuple[str, list[int], bytes]:
    """(dtype, shape, bytes) of one tensor, read past the page cache's retention."""
    path = str(path)
    h, off = header(path)
    e = h[name]
    a, b = e["data_offsets"]
    fd = os.open(path, os.O_RDONLY)
    try:
        data = os.pread(fd, b - a, off + a)
        os.posix_fadvise(fd, off + a, b - a, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    if len(data) != b - a:
        raise IOError(f"{path}: short read of {name}")
    return e["dtype"], e["shape"], data


def array(path: str | Path, name: str) -> np.ndarray:
    dt, shape, data = raw(path, name)
    return np.frombuffer(data, dtype=_DT[dt]).reshape(shape).copy()


def raw_rows(path: str | Path, name: str, row0: int, rows: int) -> tuple[str, list[int], bytes]:
    """rows [row0, row0 + rows) of dim 0 of a tensor (one expert of a stacked [E, ...] tensor)."""
    path = str(path)
    h, off = header(path)
    e = h[name]
    a, b = e["data_offsets"]
    shape = e["shape"]
    row = (b - a) // shape[0]
    fd = os.open(path, os.O_RDONLY)
    try:
        data = os.pread(fd, row * rows, off + a + row * row0)
        os.posix_fadvise(fd, off + a + row * row0, row * rows, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return e["dtype"], [rows] + shape[1:], data


@lru_cache(maxsize=1)
def receipt() -> dict:
    return json.loads((EXL3 / "checkpoint.json").read_text())


@lru_cache(maxsize=1)
def orig_index() -> dict:
    return json.loads((ORIG / "model.safetensors.index.json").read_text())["weight_map"]


def expert_file(layer: int, e: int) -> Path:
    for b in receipt()["layers"][str(layer)]["batches"]:
        if b["start"] <= e < b["end"]:
            return EXL3 / b["file"]
    raise KeyError((layer, e))


def exl3_expert(layer: int, e: int, proj: str) -> dict[str, np.ndarray]:
    """{trellis int16 [K/16, N/16, 16K], suh f16 [K], svh f16 [N], mul1 int32} of experts.e.proj."""
    f = expert_file(layer, e)
    out = {k: array(f, f"experts.{e}.{proj}.{k}") for k in ("trellis", "suh", "svh", "mul1")}
    if int(out["mul1"].astype(np.uint32)) != MUL1:
        raise ValueError(f"layer {layer} expert {e} {proj}: mul1 {int(out['mul1']):#x} is not the mul1 codebook")
    return out


# ---- the original release's experts (the verified formats; inkling-tensorfold quant/) -------------------------

def e4m3_table() -> np.ndarray:
    """float64 value of every e4m3fn byte (0x7F / 0xFF NaN)."""
    v = np.arange(256)
    s = np.where(v & 0x80, -1.0, 1.0)
    e = (v >> 3) & 0xF
    m = v & 7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    val = val * s
    val[(v & 0x7F) == 0x7F] = np.nan
    return val


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def orig_expert(layer: int, e: int, proj: str) -> np.ndarray:
    """The original stored matrix of expert e, dequantized exactly (float64): w13 [2I (interleaved g/u), H],
    w2 [H, I]. Layers 0-39 NVFP4 (low nibble first, e4m3 per 16, amax / 2688), 40-47 MXFP8 (e4m3, UE8M0 per 32)."""
    idx = orig_index()
    base = f"model.llm.layers.{layer}.mlp.experts.{proj}_weight"
    f = ORIG / idx[base]
    _, wshape, wb = raw_rows(f, base, e, 1)
    _, sshape, sb = raw_rows(ORIG / idx[base + "_scale"], base + "_scale", e, 1)
    w = np.frombuffer(wb, np.uint8).reshape(wshape[1:])
    s = np.frombuffer(sb, np.uint8).reshape(sshape[1:])
    if layer < 40:
        amax = array(ORIG / idx[base + "_amax"], base + "_amax").astype(np.float32)
        den = array(ORIG / idx[base + "_outer_scale_denominator"], base + "_outer_scale_denominator").astype(np.float32)
        g = np.float32(amax[e]) / np.float32(den.reshape(-1)[0])
        lo, hi = w & 0xF, w >> 4
        codes = np.stack([lo, hi], -1).reshape(w.shape[0], -1)          # low nibble first
        vals = E2M1[codes]
        sc = e4m3_table()[s].repeat(16, axis=1)
        return vals * sc * np.float64(g)
    vals = e4m3_table()[w]
    sc = np.exp2(s.astype(np.float64) - 127).repeat(32, axis=1)
    return vals * sc
