"""CPU reference of the native lane's EXL3 expert arithmetic (zig/kernels/cuda/inkling2_exl3.cu), stage by stage, for
the kernel gate (check_kernels.py) and the tiny-model reference.

Bit-replicated (float32 / float16 IEEE, the kernels' operation order): the activation quantization (ink2_exl3_act:
sglang's NVFP4 4/6 blocks and MXFP8 blocks, kortexa's dequantization), the input rotation (ink2_exl3_rot: fp16(x * suh),
fwht128's butterflies, * 1/sqrt(128), fp16) and the epilogues' rotation and scale. Not bit-replicated: the GEMM (the
mma's fp32 accumulation order is the hardware's: compared within tolerance against float64) and the SwiGLU's
expf / tanhf (libdevice: compared within one bf16 ulp).
"""
from __future__ import annotations

import numpy as np
import torch

F32 = np.float32
HAD_SCALE = F32(0.08838834764831845)
FLT_MAX = F32(3.4028234663852886e38)


# ---- rounding helpers ----------------------------------------------------------------------------------------------

def e4m3_rn(v: np.ndarray) -> np.ndarray:
    """fp32 -> e4m3fn (round to nearest even, values already within +-448) -> fp32 (torch's cast: RTNE)."""
    t = torch.from_numpy(np.ascontiguousarray(v, dtype=F32))
    return t.to(torch.float8_e4m3fn).to(torch.float32).numpy()


def e2m1_rn(v: np.ndarray) -> np.ndarray:
    """e2m1 round to nearest even, saturating at 6 (cvt.rn.satfinite.e2m1x2.f32), sign kept (ink2 e2m1_rn)."""
    a = np.abs(v)
    r = np.select([a <= 0.25, a < 0.75, a <= 1.25, a < 1.75, a <= 2.5, a < 3.5, a <= 5.0],
                  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], 6.0).astype(F32)
    return np.copysign(r, v).astype(F32)


def bf16r(v: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(v, dtype=F32)).to(torch.bfloat16).to(torch.float32).numpy()


def bf16_from_bits(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(F32)


# ---- activation quantization (ink2_exl3_act) ------------------------------------------------------------------------

def nvfp4_rows(x: np.ndarray, amax: np.ndarray | None = None) -> np.ndarray:
    """[R][K] fp32 (bf16 values) -> fp32 dequantized values (before the kernel's fp16 rounding)."""
    x = x.astype(F32)
    R, K = x.shape
    if amax is None:
        amax = np.abs(x).max(1)
    amax = amax.astype(F32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        enc = np.minimum(F32(1536.0) / amax, FLT_MAX).astype(F32)
        dec = (F32(1.0) / enc).astype(F32)
    outer = (amax * F32(1.0 / 1536.0)).astype(F32)
    b = x.reshape(R, K // 16, 16)
    bm = np.abs(b).max(2)
    s6 = e4m3_rn(np.minimum(bm * (enc * F32(1.0 / 6.0))[:, None], F32(448.0)))
    s4 = e4m3_rn(np.minimum(bm * (enc * F32(0.25))[:, None], F32(448.0)))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        i6 = np.minimum(F32(1.0) / (s6 * dec[:, None]), FLT_MAX).astype(F32)
        i4 = np.minimum(F32(1.0) / (s4 * dec[:, None]), FLT_MAX).astype(F32)
    r6 = e2m1_rn((b * i6[..., None]).astype(F32))
    r4 = e2m1_rn((b * i4[..., None]).astype(F32))
    e6 = np.abs(((r6 * s6[..., None]).astype(F32) * outer[:, None, None]).astype(F32) - b).astype(F32)
    e4 = np.abs(((r4 * s4[..., None]).astype(F32) * outer[:, None, None]).astype(F32) - b).astype(F32)
    for w in (8, 4, 2, 1):                                   # adjacent pairs summed, four times
        e6 = (e6[..., 0::2] + e6[..., 1::2]).astype(F32)
        e4 = (e4[..., 0::2] + e4[..., 1::2]).astype(F32)
    four = (e4[..., 0] < e6[..., 0])
    inner = np.where(four[..., None], (r4 * s4[..., None]).astype(F32), (r6 * s6[..., None]).astype(F32))
    return (inner * outer[:, None, None]).astype(F32).reshape(R, K)


def mxfp8_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(F32)
    R, K = x.shape
    b = x.reshape(R, K // 32, 32)
    bm = np.abs(b).max(2).astype(F32)
    raw = (bm * F32(1.0 / 448.0)).astype(F32)
    e = ((raw.view(np.uint32).astype(np.uint64) + 0x007FFFFF) & 0x7F800000) >> 23
    e = np.where(raw <= F32(2.0 ** -127), 0, np.minimum(e, 254)).astype(np.uint32)
    scale = np.where(e > 0, (e << 23).view(F32), np.array(0x00400000, np.uint32).view(F32)).astype(F32)
    inv = np.where(e == 0, F32(2.0 ** 127), (F32(1.0) / scale).astype(F32)).astype(F32)
    q = e4m3_rn(np.clip((b * inv[..., None]).astype(F32), F32(-448.0), F32(448.0)))
    return (q * scale[..., None]).astype(F32).reshape(R, K)


def act(x: np.ndarray, mode: int, amax: np.ndarray | None = None) -> np.ndarray:
    """ink2_exl3_act: [R][K] bf16 values (fp32) -> fp16 (mode 0 none, 1 NVFP4, 2 MXFP8)."""
    if mode == 0:
        return x.astype(np.float16)
    v = nvfp4_rows(x, amax) if mode == 1 else mxfp8_rows(x)
    return v.astype(np.float16)


# ---- the rotation (fwht128, the kernels' butterflies) ----------------------------------------------------------------

def fwht128(v: np.ndarray) -> np.ndarray:
    """experts.cu's fwht128 on [..., 128] fp32: lane L holds elements 4L..4L+3; a 4-point stage in registers, then
    lanes paired by bit m (m = 1..16): low = low + high, high = low - high (the kernel's operand order)."""
    v = v.astype(F32).reshape(*v.shape[:-1], 32, 4).copy()
    v0, v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    a, b, c, d = v0 + v1, v0 - v1, v2 + v3, v2 - v3
    v = np.stack([a + c, b + d, a - c, b - d], -1).astype(F32)
    lane = np.arange(32)
    m = 1
    while m < 32:
        partner = v[..., lane ^ m, :]
        hi = ((lane & m) != 0)[:, None]
        v = np.where(hi, (partner - v).astype(F32), (v + partner).astype(F32)).astype(F32)
        m <<= 1
    return v.reshape(*v.shape[:-2], 128)


def rotate_in(xq: np.ndarray, suh: np.ndarray) -> np.ndarray:
    """ink2_exl3_rot: fp16(H128(fp16(xq * suh)) / sqrt(128)), [R][K] fp16 -> fp16."""
    s = (xq.astype(F32) * suh.astype(F32)).astype(np.float16).astype(F32)
    R, K = s.shape
    h = fwht128(s.reshape(R, K // 128, 128)).reshape(R, K)
    return (h * HAD_SCALE).astype(np.float16)


def rotate_out(z: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """The epilogues' fp32 rotation and scale: H128(z) * 1/sqrt(128) * svh."""
    R, N = z.shape
    h = fwht128(z.astype(F32).reshape(R, N // 128, 128)).reshape(R, N)
    return ((h * HAD_SCALE).astype(F32) * svh.astype(F32)).astype(F32)


def swiglu(g: np.ndarray, u: np.ndarray, cg: float = 4.0, cu: float = 25.0) -> np.ndarray:
    """cg tanh(g / cg) sigmoid(g) cu tanh(u / cu) on the bf16 roundings (fp32; libdevice is not replicated)."""
    g = bf16r(g).astype(np.float64)
    u = bf16r(u).astype(np.float64)
    return (cg * np.tanh(g / cg) * (1 / (1 + np.exp(-g))) * (cu * np.tanh(u / cu))).astype(F32)


def gateup_act(y13: np.ndarray, cg: float = 4.0, cu: float = 25.0) -> np.ndarray:
    """w13's rotated, scaled outputs [R][2 Il] (interleaved gate / up) -> act [R][Il] (bf16 values as fp32)."""
    return bf16r(swiglu(y13[:, 0::2], y13[:, 1::2], cg, cu))
