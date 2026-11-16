"""The EXL3 expert slot of one TP rank: the byte layout the native loader (zig exl3.zig packSlot) writes and the
kernels read. A rank keeps intermediate channels [rank Il, (rank + 1) Il) of every expert (Il = I / world), as the
engine's NVFP4 / MXFP8 lanes do:

  w13 (H -> 2I, output columns interleaved gate/up per channel): the rank's channels are output columns
      [2 Il rank, 2 Il (rank + 1)) = output tiles [nt0, nt0 + 2 Il / 16): for every input tile row, those tiles
      (a strided gather into one contiguous [H/16][2Il/16][16K] int16 block); suh whole [H]; svh [2 Il rank ..) [2 Il].
  w2 (I -> H): the rank's channels are input rows [Il rank, ..) = input tiles [Il rank / 16, ..): one contiguous
      [Il/16][H/16][16K] block; suh [Il rank ..) [Il]; svh whole [H].

Both cuts fall on 128-column / 128-row Hadamard block boundaries (2 Il and Il are multiples of 128 at world <= 2 for
I = 2304: 2304 / 128 = 18, 1152 / 128 = 9), so each rank's rotations are its own and the w2 partials add exactly
(the output rotation is linear). Layout of a slot (every part 16-byte aligned):

  [w13 trellis | w2 trellis | suh13 f16 [H] | svh13 f16 [2Il] | suh2 f16 [Il] | svh2 f16 [H]]
"""
from __future__ import annotations

import numpy as np


def tile_bytes(k: int) -> int:
    return 32 * k                                    # 16 K int16 a 16 x 16 tile


def parts(H: int, I: int, world: int, k: int) -> dict[str, tuple[int, int]]:
    """{part: (offset, bytes)} of one slot."""
    Il = I // world
    sizes = [("w13", (H // 16) * (2 * Il // 16) * tile_bytes(k)), ("w2", (Il // 16) * (H // 16) * tile_bytes(k)),
             ("suh13", 2 * H), ("svh13", 2 * 2 * Il), ("suh2", 2 * Il), ("svh2", 2 * H)]
    out, off = {}, 0
    for name, n in sizes:
        out[name] = (off, n)
        off += (n + 15) // 16 * 16
    out["total"] = (off, off)
    return out


def check_geometry(H: int, I: int, world: int) -> None:
    Il = I // world
    if I % world or (2 * Il) % 128 or Il % 128 or H % 128:
        raise ValueError(f"EXL3 TP split needs 2 Il and Il multiples of 128 (Hadamard blocks): H {H}, I {I}, world {world}")


def pack(w13: dict, w2: dict, H: int, I: int, rank: int, world: int) -> bytes:
    """One rank's slot from the checkpoint tensors (numpy arrays as read: trellis int16, suh/svh f16)."""
    check_geometry(H, I, world)
    Il = I // world
    k = w13["trellis"].shape[-1] // 16
    assert w2["trellis"].shape[-1] == 16 * k, "w13 and w2 of one expert share K"
    assert w13["trellis"].shape == (H // 16, 2 * I // 16, 16 * k) and w2["trellis"].shape == (I // 16, H // 16, 16 * k)
    nt = 2 * Il // 16
    t13 = np.ascontiguousarray(w13["trellis"][:, rank * nt:(rank + 1) * nt, :])
    kt = Il // 16
    t2 = np.ascontiguousarray(w2["trellis"][rank * kt:(rank + 1) * kt])
    p = parts(H, I, world, k)
    buf = bytearray(p["total"][0])

    def put(name, arr):
        off, n = p[name]
        b = np.ascontiguousarray(arr).tobytes()
        assert len(b) == n, (name, len(b), n)
        buf[off:off + n] = b

    put("w13", t13)
    put("w2", t2)
    put("suh13", w13["suh"])
    put("svh13", w13["svh"][2 * Il * rank:2 * Il * (rank + 1)])
    put("suh2", w2["suh"][Il * rank:Il * (rank + 1)])
    put("svh2", w2["svh"])
    return bytes(buf)


def unpack(slot: bytes, H: int, I: int, world: int, k: int) -> dict:
    """The rank's tensors back out of a slot (what the kernels see)."""
    Il = I // world
    p = parts(H, I, world, k)

    def get(name, dtype, shape):
        off, n = p[name]
        return np.frombuffer(slot[off:off + n], dtype=dtype).reshape(shape)

    return {"w13": {"trellis": get("w13", np.int16, (H // 16, 2 * Il // 16, 16 * k)),
                    "suh": get("suh13", np.float16, (H,)), "svh": get("svh13", np.float16, (2 * Il,))},
            "w2": {"trellis": get("w2", np.int16, (Il // 16, H // 16, 16 * k)),
                   "suh": get("suh2", np.float16, (Il,)), "svh": get("svh2", np.float16, (H,))}}
