// Inkling-2's EXL3 routed experts: the kortexa-ai Inkling-2-Small EXL3 export (mul1 codebook, every expert matrix 2 or
// 3 bits, w13 = gate/up fused with its output columns interleaved per channel, w2 = down), one TP rank's slots.
//
// Decode and mma chains: TensorFold 92aa710b533d src/tensorfold/cuda/exl3/experts_grouped.cuh (the universal grouped
// EXL3 expert kernel, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp), its device functions unchanged below
// (cb_pair, Fmt, LaneMap, fetch, decode_tile, mma16816, load_pair, load_words, warp_tiles_rows): the codebook decode
// is ExLlamaV3's bit for bit (that file's tests), and a row's mma sequence never depends on the window it arrived in.
// The torch launchers are dropped; the programs here walk the native engine's grouped plan instead (items of
// (slot, first, count) after ink2_items_to_slots, members = pair ids t * topk + slot), the slot of a pair holding the
// rank's share of one expert (out/exl3/ref/slot.py; zig exl3.zig packSlot):
//   [w13 trellis [H/16][2Il/16][16K] | w2 trellis [Il/16][H/16][16K] | suh13 f16 [H] | svh13 f16 [2Il] | suh2 f16 [Il]
//    | svh2 f16 [H]]     (every part 16-byte aligned; K = 2 or 3; a slot's address and K from slot_ptr / slot_k)
//
// The forward of one projection (kortexa model.Layer.expert, with exllamav3's LinearEXL3 rounding points as kortexa's
// rotated_linear_reference documents them):
//   xq  = fp16(activation(x))                  ink2_exl3_act: NVFP4 / MXFP8 fake quantization or none (bf16 -> fp16)
//   xr  = fp16(H128(fp16(xq * suh)) / sqrt(128))   ink2_exl3_rot (the engine's fwht128 butterflies)
//   z   = xr @ W_q                              ink2_exl3_gemm_g{1,2} (fp32 mma chains, the trellis decoded in place)
//   y   = H128(z) / sqrt(128) * svh             ink2_exl3_gateup_epi (then bf16, the capped SwiGLU, bf16) or
//                                               ink2_exl3_down_epi (fp32 partials: the engine's TP sum and combine)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "exl3_decode.cuh"   // the engine's EXL3 tile decode (tf_exl3) and its cp.async / ldmatrix / mma hooks

namespace tf_exl3x {

// ---- experts_grouped.cuh (TensorFold 92aa710b533d), unchanged -----------------------------------------------------

// Two codebook values (CB 0 3inst, 1 mcg, 2 mul1) as a half2, bit-identical to ExLlamaV3's decode_3inst_2<cb>.
template <int CB>
__device__ __forceinline__ uint32_t cb_pair(uint32_t s0, uint32_t s1) {
    if constexpr (CB == 2) {
        const uint32_t x0 = s0 * 0x83DCD12Du, x1 = s1 * 0x83DCD12Du;
        const uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
        const uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
        const uint32_t hv = __byte_perm(sum0, sum1, 0x5410);
        half2 h = *reinterpret_cast<const half2*>(&hv);
        half2 r = __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
        return *reinterpret_cast<uint32_t*>(&r);
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            x0 = s0 * 0xCBAC1FEDu;
            x1 = s1 * 0xCBAC1FEDu;
        } else {
            x0 = s0 * 89226354u + 64248484u;
            x1 = s1 * 89226354u + 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        uint32_t lo = __byte_perm(x0, x1, 0x5410);
        uint32_t hi = __byte_perm(x0, x1, 0x7632);
        half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
        return *reinterpret_cast<uint32_t*>(&r);
    }
}

// K2 half-bits a value: a tile is 4 * K2 words; a lane's eight windows fall in NG runs of GV within two words.
template <int K2>
struct Fmt {
    static constexpr int TW = 4 * K2;
    static constexpr int LW = (TW + 31) / 32;
    // windows sharing one 64-bit merge (tests/cuda/test_exl3_experts.py checks every K2)
    static constexpr int GV = (K2 >= 13) ? 2 : ((K2 == 7 || (K2 >= 9 && K2 <= 12) || K2 == 16) ? 4 : 8);
    static constexpr int NG = 8 / GV;
    __host__ __device__ static constexpr int end(int p) { return (p >> 1) * K2 + ((p & 1) ? K2 : (K2 >> 1)); }
    // right shift of window j of a run (run starts at an even position) relative to the run's last window
    __host__ __device__ static constexpr int off(int j) { return end(GV - 1) - end(j); }
};

template <int K2>
struct LaneMap {
    int hi[Fmt<K2>::NG], lo[Fmt<K2>::NG], sh[Fmt<K2>::NG];
    __device__ __forceinline__ explicit LaneMap(int lane) {
        constexpr int TW = Fmt<K2>::TW, GV = Fmt<K2>::GV;
#pragma unroll
        for (int g = 0; g < Fmt<K2>::NG; ++g) {
            const int last_end = Fmt<K2>::end(8 * lane + g * GV + GV - 1) + 128 * K2;
            const int hr = (last_end - 1) >> 5;
            hi[g] = hr % TW;
            lo[g] = (hr + TW - 1) % TW;
            sh[g] = (hr + 1) * 32 - last_end;
        }
    }
};

template <int LW>
__device__ __forceinline__ uint32_t fetch(const uint32_t (&w)[LW], int idx) {
    if constexpr (LW == 1) {
        return __shfl_sync(0xffffffffu, w[0], idx);
    } else {
        const uint32_t a = __shfl_sync(0xffffffffu, w[0], idx & 31);
        const uint32_t b = __shfl_sync(0xffffffffu, w[1], idx & 31);
        return idx < 32 ? a : b;
    }
}

// This lane's eight values of a tile as the B fragments of its two n8 halves.
template <int CB, int K2>
__device__ __forceinline__ void decode_tile(const uint32_t (&w)[Fmt<K2>::LW], const LaneMap<K2>& m, int lane,
                                            uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t st[8];
    if constexpr (K2 == 8) {
        // 4 bits: lane L's windows are exactly words L-1 and L (the GLM kernel's decode)
        const uint32_t p = __shfl_sync(0xffffffffu, w[0], (lane + 31) & 31);
        const uint32_t s = __funnelshift_r(w[0], p, 20);
        st[0] = (s >> 8) & 0xffffu;
        st[1] = (s >> 4) & 0xffffu;
        st[2] = s & 0xffffu;
        st[3] = w[0] >> 16;
        st[4] = (w[0] >> 12) & 0xffffu;
        st[5] = (w[0] >> 8) & 0xffffu;
        st[6] = (w[0] >> 4) & 0xffffu;
        st[7] = w[0] & 0xffffu;
    } else {
        constexpr int GV = Fmt<K2>::GV, NG = Fmt<K2>::NG;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const uint32_t whi = fetch<Fmt<K2>::LW>(w, m.hi[g]);
            const uint32_t wlo = fetch<Fmt<K2>::LW>(w, m.lo[g]);
            const uint64_t mm = ((((uint64_t)wlo) << 32) | whi) >> m.sh[g];
#pragma unroll
            for (int j = 0; j < GV; ++j) st[g * GV + j] = (uint32_t)(mm >> Fmt<K2>::off(j)) & 0xffffu;
        }
    }
    b0[0] = cb_pair<CB>(st[0], st[1]);
    b0[1] = cb_pair<CB>(st[2], st[3]);
    b1[0] = cb_pair<CB>(st[4], st[5]);
    b1[1] = cb_pair<CB>(st[6], st[7]);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

template <int K2>
__device__ __forceinline__ void load_words(uint32_t (&dst)[Fmt<K2>::LW], const uint32_t* p, int lane) {
    constexpr int TW = Fmt<K2>::TW;
#pragma unroll
    for (int l = 0; l < Fmt<K2>::LW; ++l) {
        if constexpr ((TW % 32) == 0)
            dst[l] = __ldg(p + l * 32);
        else
            dst[l] = (l * 32 + lane < TW) ? __ldg(p + l * 32) : 0u;
    }
}

// warp_tiles for G member tiles at once: each k tile's weights are decoded once and fed to every live tile's rows (off:
// a row's element offset in X, -1 for none); a row's mma sequence is warp_tiles' (same k tiles, same order, from zero).
template <int CB, int K2, int NT, int PF, int G>
__device__ __forceinline__ void warp_tiles_rows(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                                const half* __restrict__ X, const int (&off)[G][2], int live, int lane,
                                                float (&acc)[G][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    // the rows' A fragments a k tile ahead (the loads' latency behind this tile's decode and mma)
    const half* xr[G][2];
    bool ok[G][2];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            ok[g][j] = off[g][j] >= 0;
            xr[g][j] = X + (ok[g][j] ? off[g][j] : 0) + kt0 * 16;
        }
    uint32_t an[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
        if (g < live) {
            an[g][0] = load_pair(xr[g][0], ok[g][0]);
            an[g][1] = load_pair(xr[g][1], ok[g][1]);
            an[g][2] = load_pair(xr[g][0] + 8, ok[g][0]);
            an[g][3] = load_pair(xr[g][1] + 8, ok[g][1]);
        }

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t a[G][4];
#pragma unroll
                for (int g = 0; g < G; ++g)
#pragma unroll
                    for (int c = 0; c < 4; ++c) a[g][c] = an[g][c];
                if (it + 1 < nkt) {
                    const int k = (it + 1) * 16;
#pragma unroll
                    for (int g = 0; g < G; ++g)
                        if (g < live) {
                            an[g][0] = load_pair(xr[g][0] + k, ok[g][0]);
                            an[g][1] = load_pair(xr[g][1] + k, ok[g][1]);
                            an[g][2] = load_pair(xr[g][0] + k + 8, ok[g][0]);
                            an[g][3] = load_pair(xr[g][1] + k + 8, ok[g][1]);
                        }
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        if (g < live) {
                            mma16816(acc[g][i][0], a[g], b0);
                            mma16816(acc[g][i][1], a[g], b1);
                        }
                    }
                }
            }
        }
    }
}

// experts.cu's fwht128 (TensorFold 92aa710b533d): Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly
// order (strides 1, 2 in registers, 4..64 across lanes). Values 4 lane .. 4 lane + 3 of the 128 are lane's v[0..3].
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

}  // namespace tf_exl3x

namespace {

using namespace tf_exl3x;

constexpr int CB_MUL1 = 2;
constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// ---- the slot (slot.py's parts) ------------------------------------------------------------------------------------

struct Parts {
    size_t w2, suh13, svh13, suh2, svh2;
};

__host__ __device__ __forceinline__ size_t align16(size_t n) { return (n + 15) & ~(size_t)15; }

// Byte offsets of a K-bit slot's parts for hidden H and the rank's Il channels (w13 at 0).
__device__ __forceinline__ Parts slot_parts(int H, int Il, int k) {
    const size_t tile = 32 * (size_t)k;
    Parts p;
    p.w2 = align16((size_t)(H / 16) * (2 * Il / 16) * tile);
    p.suh13 = p.w2 + align16((size_t)(Il / 16) * (H / 16) * tile);
    p.svh13 = p.suh13 + align16(2 * (size_t)H);
    p.suh2 = p.svh13 + align16(4 * (size_t)Il);
    p.svh2 = p.suh2 + align16(2 * (size_t)Il);
    return p;
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// inkling2's SwiGLU, the arithmetic of inkling2_mxfp8.cu's act (the engine's other expert lanes): cg*tanh(g/cg) *
// sigmoid(g) * cu*tanh(u/cu) in fp32, g and u taken at their bf16 roundings (kortexa: the w13 output rounded to bf16).
__device__ __forceinline__ float swiglu(float g, float u, float cg, float cu) {
    g = bf16r(g);
    u = bf16r(u);
    const float sg = 1.0f / (1.0f + expf(-g));
    return cg * tanhf(g / cg) * sg * (cu * tanhf(u / cu));
}

// ---- activation quantization (sglang 237ba6b kernels/ops/inkling2/{nvfp4,moe}.py, kortexa model.Layer.activation) --

// e2m1 round-to-nearest-even, saturating at 6 (cvt.rn.satfinite.e2m1x2.f32's value), sign kept.
__device__ __forceinline__ float e2m1_rn(float v) {
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.0f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.0f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.0f;
    else if (a < 3.5f) r = 3.0f;
    else if (a <= 5.0f) r = 4.0f;
    else r = 6.0f;
    return copysignf(r, v);
}

// fp32 -> e4m3 (round to nearest even, satfinite) -> fp32: Triton's .to(tl.float8e4nv).to(tl.float32).
__device__ __forceinline__ float e4m3_rn(float v) {
    const __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)));
}

// One NVFP4 block of 16 values (sglang _quantize_values: the 4/6 choice by the block's summed error), dequantized as
// kortexa does: (e2m1 * e4m3 scale) * (amax * fp32(1 / 1536)). `x` and `out` are the block's 16 values.
__device__ __forceinline__ void nvfp4_block(const float (&x)[16], float amax, float (&out)[16]) {
    const float enc = fminf(__fdiv_rn(1536.0f, amax), 3.4028234663852886e38f);
    const float dec = __fdiv_rn(1.0f, enc);
    // every product and difference explicitly rounded (__fmul_rn / __fsub_rn never fuse): sglang compiles this with
    // enable_fp_fusion=False, and an FMA in the error terms would move the 4/6 choice
    const float outer = __fmul_rn(amax, 1.0f / 1536.0f);
    float bm = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) bm = fmaxf(bm, fabsf(x[i]));
    const float s6 = e4m3_rn(fminf(__fmul_rn(bm, __fmul_rn(enc, 1.0f / 6.0f)), 448.0f));
    const float s4 = e4m3_rn(fminf(__fmul_rn(bm, __fmul_rn(enc, 0.25f)), 448.0f));
    const float i6 = fminf(__fdiv_rn(1.0f, __fmul_rn(s6, dec)), 3.4028234663852886e38f);
    const float i4 = fminf(__fdiv_rn(1.0f, __fmul_rn(s4, dec)), 3.4028234663852886e38f);
    float r6[16], r4[16], e6[16], e4[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        r6[i] = e2m1_rn(__fmul_rn(x[i], i6));
        r4[i] = e2m1_rn(__fmul_rn(x[i], i4));
        e6[i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r6[i], s6), outer), x[i]));
        e4[i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r4[i], s4), outer), x[i]));
    }
    // _block_error: adjacent pairs summed, four times (the tree's order)
#pragma unroll
    for (int w = 8; w >= 1; w >>= 1)
#pragma unroll
        for (int i = 0; i < w; ++i) {
            e6[i] = e6[2 * i] + e6[2 * i + 1];
            e4[i] = e4[2 * i] + e4[2 * i + 1];
        }
    const bool four = e4[0] < e6[0];
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = __fmul_rn(four ? __fmul_rn(r4[i], s4) : __fmul_rn(r6[i], s6), outer);
}

// One MXFP8 block of 32 values (sglang _mxfp8_block), dequantized as kortexa's mxfp8_dequant: e4m3 * 2^(e - 127).
__device__ __forceinline__ void mxfp8_block(const float (&x)[32], float (&out)[32]) {
    float bm = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; ++i) bm = fmaxf(bm, fabsf(x[i]));
    const float raw = __fmul_rn(bm, 1.0f / 448.0f);
    int e = (int)(((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23);
    e = raw <= 0x1p-127f ? 0 : min(e, 254);
    // 2^(e - 127) exactly from its bits (e 0: the subnormal 2^-127, as _mxfp8_block's max with min_scale), which is
    // also kortexa's dequantization factor torch.exp2(e - 127) (inkling2_mxfp8.cu's pow2e)
    const float scale = e ? __uint_as_float((uint32_t)e << 23) : __uint_as_float(0x00400000u);
    const float inv = e == 0 ? 0x1p127f : __frcp_rn(scale);
#pragma unroll
    for (int i = 0; i < 32; ++i) out[i] = __fmul_rn(e4m3_rn(fminf(fmaxf(__fmul_rn(x[i], inv), -448.0f), 448.0f)), scale);
}

}  // namespace

// ---- entries ---------------------------------------------------------------------------------------------------

// Activation of `rows` rows of K values (bf16, row stride x_stride) into fp16 rows of out (stride out_stride), one
// block of 128 threads a row. mode 0: fp16(x) (no activation quantization); 1: NVFP4 per 16 under the row's amax;
// 2: MXFP8 per 32. amax_all (mode 1): [world][amax_stride] fp32, the row's amax the max of its world entries (a TP
// rank's part of a wider row); null: the row's own max over its K values.
extern "C" __global__ void __launch_bounds__(128) ink2_exl3_act(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                                int K, int mode, const float* __restrict__ amax_all,
                                                                int world, int amax_stride, half* __restrict__ out,
                                                                int out_stride) {
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride;
    half* o = out + (size_t)row * out_stride;
    if (mode == 0) {
        for (int i = threadIdx.x; i < K; i += blockDim.x) o[i] = __float2half_rn(__bfloat162float(xr[i]));
        return;
    }
    if (mode == 2) {
        for (int b = threadIdx.x; b < K / 32; b += blockDim.x) {
            float v[32], q[32];
#pragma unroll
            for (int i = 0; i < 32; ++i) v[i] = __bfloat162float(xr[32 * b + i]);
            mxfp8_block(v, q);
#pragma unroll
            for (int i = 0; i < 32; ++i) o[32 * b + i] = __float2half_rn(q[i]);
        }
        return;
    }
    __shared__ float red[128];
    float amax = 0.0f;
    if (amax_all != nullptr) {
        for (int r = 0; r < world; ++r) amax = fmaxf(amax, amax_all[(size_t)r * amax_stride + row]);
    } else {
        float m = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(xr[i])));
        red[threadIdx.x] = m;
        __syncthreads();
        for (int s = 64; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            __syncthreads();
        }
        amax = red[0];
    }
    for (int b = threadIdx.x; b < K / 16; b += blockDim.x) {
        float v[16], q[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = __bfloat162float(xr[16 * b + i]);
        nvfp4_block(v, amax, q);
#pragma unroll
        for (int i = 0; i < 16; ++i) o[16 * b + i] = __float2half_rn(q[i]);
    }
}

// Each row's max |x| over its K bf16 values (a TP rank's part of the row: the ranks' maxima are gathered after).
extern "C" __global__ void __launch_bounds__(128) ink2_exl3_row_amax(const __nv_bfloat16* __restrict__ x,
                                                                     int x_stride, int K, float* __restrict__ amax) {
    __shared__ float red[128];
    const __nv_bfloat16* xr = x + (size_t)blockIdx.x * x_stride;
    float m = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(xr[i])));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) amax[blockIdx.x] = red[0];
}

// The input rotation of every routed pair: xr[p] = fp16(H128(fp16(xq[row] * suh)) / sqrt(128)), row = p / row_slots
// (row_slots 0: p), suh the pair's slot's (which 0: w13's [H], 1: w2's [Il]). A program a (plan item, 128 inputs),
// one warp; it walks the item's members.
extern "C" __global__ void __launch_bounds__(32) ink2_exl3_rot(const half* __restrict__ xq, int xq_stride,
                                                               int row_slots, const int* __restrict__ items,
                                                               const int* __restrict__ counts,
                                                               const int* __restrict__ members,
                                                               const unsigned long long* __restrict__ slot_ptr,
                                                               const int* __restrict__ slot_k, int nslots, int which,
                                                               int H, int Il, half* __restrict__ out, int K) {
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots) return;
    const Parts pt = slot_parts(H, Il, __ldg(slot_k + slot));
    const half* suh = reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) +
                                                    (which ? pt.suh2 : pt.suh13)) + blk * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(suh[j]);
    for (int m = 0; m < cnt; ++m) {
        const int p = __ldg(members + first + m);
        const int row = row_slots ? p / row_slots : p;
        const half* xr = xq + (size_t)row * xq_stride + blk * 128 + 4 * lane;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = __half2float(__float2half_rn(__half2float(xr[j]) * s[j]));
        fwht128(v, lane);
        half* o = out + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
    }
}

namespace {

// Program (plan item, NT n tiles of 16 columns): the item's members, 16 G at a time, times W_q's columns over all of K
// (which 0: w13, K = H, N = 2 Il; 1: w2, K = Il, N = H), the K range in `kparts` equal parts: in each part W warps
// each a contiguous (K / kparts) / W, their sums added in warp order through shared memory, the part's raw fp32 sum
// written to its own plane, out + part * plane, at [p][col - c0] (row stride ldo), for columns [c0 + NT 16 y, ...).
// A row's bits do not depend on G, the item or the window (warp_tiles_rows: the same mma chain from zero, the same
// warp order): decode and prompt plans give every pair the same output. kparts makes the down projection's sums
// independent of the TP width: a world-1 rank computes w2 in the two parts a world-2 rank pair computes (each part
// the same 4 warp chains over its Il channels), so the epilogue then adds them in rank order exactly as the combine
// adds two ranks' partials (canonical TP, Model.canon).
template <int NT, int W, int PF, int G>
__device__ __forceinline__ void gemm_body(const half* __restrict__ X, const int* __restrict__ items,
                                          const int* __restrict__ counts, const int* __restrict__ members,
                                          const unsigned long long* __restrict__ slot_ptr,
                                          const int* __restrict__ slot_k, int nslots, int which, int H, int Il,
                                          float* __restrict__ out, int ldo, int c0, int kparts, long long plane) {
    const int it = blockIdx.x;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots || cnt <= 0) return;
    const int k = __ldg(slot_k + slot);
    const Parts pt = slot_parts(H, Il, k);
    const uint32_t* T = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) +
                                                          (which ? pt.w2 : 0));
    const int K = which ? Il : H, N = which ? H : 2 * Il;
    const int KT = K >> 4, NTILES = N >> 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g8 = lane >> 2, t = lane & 3;
    const int per_part = KT / kparts;
    const int per_warp = per_part / W;
    const int nt0 = (c0 >> 4) + blockIdx.y * NT;

    __shared__ int rows_sh[16 * G];
    __shared__ float red[W][16][NT * 16];
    for (int part = 0; part < kparts; ++part)
    for (int m0 = 0; m0 < cnt; m0 += 16 * G) {
        const int kt0 = part * per_part + warp * per_warp;
        float* const op = out + (size_t)part * (size_t)plane;
        __syncthreads();                                   // the last pass's readers of rows_sh / red are done
        if (threadIdx.x < 16 * G) {
            const int m = m0 + threadIdx.x;
            rows_sh[threadIdx.x] = m < cnt ? __ldg(members + first + m) : -1;
        }
        __syncthreads();
        const int live = min(G, (cnt - m0 + 15) / 16);
        int off[G][2];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const int r0 = rows_sh[16 * g + g8], r1 = rows_sh[16 * g + g8 + 8];
            off[g][0] = r0 < 0 ? -1 : r0 * K + 2 * t;
            off[g][1] = r1 < 0 ? -1 : r1 * K + 2 * t;
        }
        float acc[G][NT][2][4];
#pragma unroll
        for (int g = 0; g < G; ++g)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[g][i][h][c] = 0.f;
        switch (2 * k) {
            case 4: warp_tiles_rows<CB_MUL1, 4, NT, PF, G>(T, NTILES, kt0, per_warp, nt0, X, off, live, lane, acc); break;
            case 6: warp_tiles_rows<CB_MUL1, 6, NT, PF, G>(T, NTILES, kt0, per_warp, nt0, X, off, live, lane, acc); break;
            default: __trap();
        }
        for (int g = 0; g < live; ++g) {
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = i * 16 + h * 8 + 2 * t;
                    red[warp][g8][col] = acc[g][i][h][0];
                    red[warp][g8][col + 1] = acc[g][i][h][1];
                    red[warp][g8 + 8][col] = acc[g][i][h][2];
                    red[warp][g8 + 8][col + 1] = acc[g][i][h][3];
                }
            __syncthreads();
            for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
                const int row = idx / (NT * 16), col = idx % (NT * 16);
                const int p = rows_sh[16 * g + row];
                if (p >= 0) {
                    float s = red[0][row][col];
#pragma unroll
                    for (int w = 1; w < W; ++w) s += red[w][row][col];
                    op[(size_t)p * ldo + (size_t)(nt0 * 16 - c0) + col] = s;
                }
            }
            __syncthreads();
        }
    }
}

constexpr int GEMM_NT = 8, GEMM_W = 4;

}  // namespace

// G 1: decode / verify plans (items of up to 16 pairs); G 2: prompt plans (two member tiles a pass, the weights decoded
// once for both). Grid (plan items, columns / 128), 128 threads. The host checks K / 16 divides by 4 (the warps).
extern "C" __global__ void __launch_bounds__(GEMM_W * 32) ink2_exl3_gemm_g1(
    const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k,
    int nslots, int which, int H, int Il, float* __restrict__ out, int ldo, int c0, int kparts, long long plane) {
    gemm_body<GEMM_NT, GEMM_W, 2, 1>(X, items, counts, members, slot_ptr, slot_k, nslots, which, H, Il, out, ldo, c0,
                                    kparts, plane);
}

extern "C" __global__ void __launch_bounds__(GEMM_W * 32) ink2_exl3_gemm_g2(
    const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k,
    int nslots, int which, int H, int Il, float* __restrict__ out, int ldo, int c0, int kparts, long long plane) {
    gemm_body<GEMM_NT, GEMM_W, 1, 2>(X, items, counts, members, slot_ptr, slot_k, nslots, which, H, Il, out, ldo, c0,
                                    kparts, plane);
}


// w13's epilogue: for each routed pair and 128 of its 2 Il raw outputs (z, row stride ldz), y = H128(z) / sqrt(128) *
// svh13, then each channel's (gate, up) = (y[2c], y[2c + 1]) (interleaved) into act[p][c] = bf16(swiglu(g, u))
// (act row stride Il). A program a (plan item, 128 columns), one warp: lane L holds columns 4L .. 4L + 3, channels
// 2L and 2L + 1 of the block's 64.
extern "C" __global__ void __launch_bounds__(32) ink2_exl3_gateup_epi(
    const float* __restrict__ z, int ldz, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k,
    int nslots, int H, int Il, float cg, float cu, __nv_bfloat16* __restrict__ act) {
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots) return;
    const Parts pt = slot_parts(H, Il, __ldg(slot_k + slot));
    const half* svh = reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) +
                                                    pt.svh13) + blk * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    for (int m = 0; m < cnt; ++m) {
        const int p = __ldg(members + first + m);
        const float* zr = z + (size_t)p * ldz + blk * 128 + 4 * lane;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = zr[j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * s[j];
        const int c = blk * 64 + 2 * lane;
        act[(size_t)p * Il + c] = __float2bfloat16_rn(swiglu(v[0], v[1], cg, cu));
        act[(size_t)p * Il + c + 1] = __float2bfloat16_rn(swiglu(v[2], v[3], cg, cu));
    }
}

// w2's epilogue: for each routed pair, columns [c0 + 128 y, ...) of its raw outputs become, per part, H128(z_part) /
// sqrt(128) * svh2 (fp32), the parts added in order from the first, v = t0; v = v + t1 (combine_parts' rank-order add:
// a world-1 rank's two canonical parts give the bits a world-2 pair's combine gives); y = v (row stride ldy, the
// buffer's column 0 = model column c0). z holds the parts' raw sums (z + part * zplane, the same strides; z == y with
// one part: in place). A program a (plan item, 128 columns), one warp.
extern "C" __global__ void __launch_bounds__(32) ink2_exl3_down_epi(
    float* __restrict__ y, int ldy, int c0, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k,
    int nslots, int H, int Il, const float* z, int parts, long long zplane) {
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots) return;
    const Parts pt = slot_parts(H, Il, __ldg(slot_k + slot));
    const half* svh = reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) +
                                                    pt.svh2) + c0 + blk * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    for (int m = 0; m < cnt; ++m) {
        const int p = __ldg(members + first + m);
        float acc[4];
        for (int part = 0; part < parts; ++part) {
            const float* zr = z + (size_t)part * (size_t)zplane + (size_t)p * ldy + blk * 128 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = zr[j];
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                // explicit roundings: a world-2 rank stores each term in fp32 and the combine adds them later, so no
                // product may fuse into the add here (an FMA would round once where the pair rounds twice)
                const float term = __fmul_rn(__fmul_rn(v[j], HAD_SCALE), s[j]);
                acc[j] = part == 0 ? term : __fadd_rn(acc[j], term);
            }
        }
        float* yr = y + (size_t)p * ldy + blk * 128 + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) yr[j] = acc[j];
    }
}

// The check's decode (not on the forward path): W_q [K][N] fp16 of one slot's matrix (which 0: w13, 1: w2) through
// the forward's own lane decode, for the gate against ExLlamaV3's reconstruct (format.py unpack). Grid (K/16, N/16),
// one warp a tile.
extern "C" __global__ void __launch_bounds__(32) ink2_exl3_dequant(const unsigned long long slot_addr, int k,
                                                                   int which, int H, int Il,
                                                                   half* __restrict__ out) {
    const Parts pt = slot_parts(H, Il, k);
    const uint32_t* T = reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(slot_addr) +
                                                          (which ? pt.w2 : 0));
    const int N = which ? H : 2 * Il;
    const int kt = blockIdx.x, nt = blockIdx.y, lane = threadIdx.x;
    const int NTILES = N >> 4;
    uint32_t b0[2], b1[2];
    if (2 * k == 4) {
        uint32_t w[Fmt<4>::LW];
        load_words<4>(w, T + ((size_t)kt * NTILES + nt) * Fmt<4>::TW + lane, lane);
        const LaneMap<4> map(lane);
        decode_tile<CB_MUL1, 4>(w, map, lane, b0, b1);
    } else {
        uint32_t w[Fmt<6>::LW];
        load_words<6>(w, T + ((size_t)kt * NTILES + nt) * Fmt<6>::TW + lane, lane);
        const LaneMap<6> map(lane);
        decode_tile<CB_MUL1, 6>(w, map, lane, b0, b1);
    }
    const int g = lane >> 2, t = lane & 3;
    uint32_t v[4] = {b0[0], b0[1], b1[0], b1[1]};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        half2 h = *reinterpret_cast<half2*>(&v[q]);
        const int col = nt * 16 + g + 8 * (q >> 1);
        const int row = kt * 16 + 2 * t + 8 * (q & 1);
        out[(size_t)row * N + col] = __low2half(h);
        out[(size_t)(row + 1) * N + col] = __high2half(h);
    }
}

// ---- the prompt path on the engine's EXL3 prompt hooks -----------------------------------------------------------
// A wide prompt chunk's experts as the engine's EXL3 prompt kernels run theirs (exl3_experts.cu tf_exl3_prompt_gu /
// tf_exl3_prompt_down, exl3_prompt.cuh tmain): grid (128-column block, plan item), 8 warps, the item's 64 members a
// pass, its rows and the block's weight tiles staged together through cp.async (tf_common::cp16z, ST stages of KC k
// tiles), each tile decoded from shared memory by the engine's decoder (tf_exl3::load_lane_words + decode_lane, mul1,
// 2 or 3 bits), the rows read with qmm_frag::ldmatrix4, the products by tf_exl3::mma16816 (qmm_frag::mma_f16_v), and
// the epilogue in the same program (the 128 columns are one Hadamard block): w13's rotation, svh13 and interleaved
// capped SwiGLU straight to act; w2's rotation and svh2 straight to y. Each warp owns one 16-column tile over all of K
// with the activations as the mma's A operand, and keeps gemm_body's W chains apart (each chain's sum from zero over
// its k tiles in order, added to the running total when it ends: ((c0 + c1) + c2) + c3): every pair's bits are the
// decode form's (g1 then the epilogues; tf-inkling2-exl3-bench compares them over a whole layer).
namespace ink2_prompt {

constexpr int PW = 8;           // warps, one 16-column tile each: 128 columns
constexpr int PT = PW * 32;
constexpr int ES = 128 + 4;     // the epilogue's row stride (floats)
template <int KC>
__host__ __device__ constexpr int xrow() { return KC * 16 + 8; }   // halves a staged row (8 of padding: ldmatrix rows in distinct banks)
// G 16-row tiles a program (PM = 16 G of an item's members: one pass; the item's other passes are other programs)
template <int G, int K2, int KC>
__host__ __device__ constexpr int stage_bytes() { return 16 * G * xrow<KC>() * 2 + PW * KC * tf_exl3::tile_words<K2>() * 4; }
template <int G, int KC, int ST>
__host__ __device__ constexpr int stages_bytes() { return ST * stage_bytes<G, 6, KC>(); }
// the stages, then E: the chains' running totals [16 G rows][ES], which the epilogue then reads
template <int G, int KC, int ST>
__host__ __device__ constexpr int smem_bytes() { return stages_bytes<G, KC, ST>() + 16 * G * ES * 4; }

// One part of one pass: k tiles [kt_base, kt_base + P) in W chains of P / W; rows rows_sh[0 .. 16 nblocks); this warp's
// column tile nt0 + warp; each chain's sums added into E (the running totals, the epilogue's layout).
template <int G, int NB, int K2, int KC, int ST, int W>
__device__ __forceinline__ void pmain_nb(const half* __restrict__ X, int K, const uint32_t* __restrict__ T, int NTILES,
                                      int kt_base, int P, int nt0, const int* rows_sh, int nblocks,
                                      unsigned char* smem, float* E) {
    constexpr int PM = 16 * G, TW = tf_exl3::tile_words<K2>(), XR = xrow<KC>(), SB = stage_bytes<G, K2, KC>();
    const int g8 = (threadIdx.x & 31) >> 2, t = threadIdx.x & 3, warp_c = (threadIdx.x >> 5) * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nchunks = (P + KC - 1) / KC, per_chain = P / W;
    auto load = [&](int ch) {
        unsigned char* st = smem + (size_t)(ch % ST) * SB;
        half* xs = reinterpret_cast<half*>(st);
        uint32_t* ws = reinterpret_cast<uint32_t*>(st + PM * XR * 2);
        const int k0 = ch * KC;
        for (int idx = threadIdx.x; idx < PM * 2 * KC; idx += PT) {
            const int r = idx / (2 * KC), j = idx % (2 * KC);
            if (r >= nblocks * 16) continue;   // a 16-row block without a member: never read
            const int p = rows_sh[r], kt = k0 + (j >> 1);
            const bool ok = p >= 0 && kt < P;
            tf_common::cp16z(xs + (size_t)r * XR + j * 8, ok ? X + (size_t)p * K + (size_t)(kt_base + kt) * 16 + (j & 1) * 8 : X, ok);
        }
        for (int idx = threadIdx.x; idx < KC * PW * (TW / 4); idx += PT) {   // the block's tiles, 16-byte pieces
            const int kk = idx / (PW * (TW / 4)), rem = idx % (PW * (TW / 4)), q = rem / (TW / 4), piece = rem % (TW / 4);
            const int kt = k0 + kk;
            const bool ok = kt < P;
            tf_common::cp16z(ws + ((size_t)kk * PW + q) * TW + piece * 4,
                             ok ? T + ((size_t)(kt_base + kt) * NTILES + nt0 + q) * TW + piece * 4 : T, ok);
        }
    };
    float acc[NB][2][4];
#pragma unroll
    for (int g = 0; g < NB; ++g)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[g][h][c] = 0.f;
#pragma unroll
    for (int c = 0; c < ST - 1; ++c) {
        if (c < nchunks) load(c);
        tf_common::cp_commit();
    }
    int left = per_chain;   // k tiles left in the current chain
    bool first = true;      // the current chain is the part's first
    for (int ch = 0; ch < nchunks; ++ch) {
        tf_common::cp_wait<ST - 2>();
        __syncthreads();   // chunk ch landed for every thread; the stage the next load overwrites is read by none
        if (ch + ST - 1 < nchunks) load(ch + ST - 1);
        tf_common::cp_commit();
        const unsigned char* st = smem + (size_t)(ch % ST) * SB;
        const half* xs = reinterpret_cast<const half*>(st);
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(st + PM * XR * 2);
#pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            const int kt = ch * KC + kk;
            if (kt < P) {
                uint32_t wl[tf_exl3::lane_words<K2>()];
                tf_exl3::load_lane_words<K2>(ws + ((size_t)kk * PW + warp) * TW, lane, wl);
                uint32_t b0[2], b1[2];
                tf_exl3::decode_lane<K2, tf_exl3::CB_MUL1>(wl, lane, b0, b1);
                // the live row tiles only (NB, a compile-time count): every mma unpredicated, accumulating in place
#pragma unroll
                for (int g = 0; g < NB; ++g) {
                    uint32_t a[4];
                    qmm_frag::ldmatrix4(a, xs + (size_t)(16 * g + (lane & 7) + ((lane >> 3) & 1) * 8) * XR + kk * 16 + (lane >> 4) * 8);
                    tf_exl3::mma16816(acc[g][0], a, b0);
                    tf_exl3::mma16816(acc[g][1], a, b1);
                }
                if (--left == 0) {   // a chain ends: total = total + chain (the first: total = chain), in E
#pragma unroll
                    for (int g = 0; g < NB; ++g)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            float2* e0 = reinterpret_cast<float2*>(E + (size_t)(16 * g + g8) * ES + warp_c + h * 8 + 2 * t);
                            float2* e1 = reinterpret_cast<float2*>(E + (size_t)(16 * g + g8 + 8) * ES + warp_c + h * 8 + 2 * t);
                            if (first) {
                                *e0 = make_float2(acc[g][h][0], acc[g][h][1]);
                                *e1 = make_float2(acc[g][h][2], acc[g][h][3]);
                            } else {
                                const float2 u0 = *e0, u1 = *e1;
                                *e0 = make_float2(u0.x + acc[g][h][0], u0.y + acc[g][h][1]);
                                *e1 = make_float2(u1.x + acc[g][h][2], u1.y + acc[g][h][3]);
                            }
#pragma unroll
                            for (int c = 0; c < 4; ++c) acc[g][h][c] = 0.f;
                        }
                    first = false;
                    left = per_chain;
                }
            }
        }
    }
    tf_common::cp_wait<0>();
    __syncthreads();   // every warp's totals are in E (and the stages are free)
}

// pmain: pmain_nb at the pass's live 16-row tiles (1 .. G), so the k loop has no per-tile predicate (a predicated mma
// cannot accumulate in place: the compiler copies the accumulators around each one, about 11 MOVs an HMMA under ncu)
template <int G, int K2, int KC, int ST, int W>
__device__ __forceinline__ void pmain(const half* __restrict__ X, int K, const uint32_t* __restrict__ T, int NTILES,
                                      int kt_base, int P, int nt0, const int* rows_sh, int nblocks,
                                      unsigned char* smem, float* E) {
    if constexpr (G >= 4)
        if (nblocks == 4) return pmain_nb<G, 4, K2, KC, ST, W>(X, K, T, NTILES, kt_base, P, nt0, rows_sh, nblocks, smem, E);
    if constexpr (G >= 3)
        if (nblocks == 3) return pmain_nb<G, 3, K2, KC, ST, W>(X, K, T, NTILES, kt_base, P, nt0, rows_sh, nblocks, smem, E);
    if constexpr (G >= 2)
        if (nblocks == 2) return pmain_nb<G, 2, K2, KC, ST, W>(X, K, T, NTILES, kt_base, P, nt0, rows_sh, nblocks, smem, E);
    pmain_nb<G, 1, K2, KC, ST, W>(X, K, T, NTILES, kt_base, P, nt0, rows_sh, nblocks, smem, E);
}

template <int G, int KC, int ST>
__device__ __forceinline__ void prompt_gu(const half* __restrict__ X, const int* __restrict__ items,
                                          const int* __restrict__ counts, const int* __restrict__ members,
                                          const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k,
                                          int nslots, int H, int Il, float cg, float cu, __nv_bfloat16* __restrict__ act) {
    const int cb = blockIdx.x, it = blockIdx.y;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots || cnt <= 0) return;
    const int k = __ldg(slot_k + slot);
    const Parts pt = slot_parts(H, Il, k);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot));
    const uint32_t* T = reinterpret_cast<const uint32_t*>(base);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int PM = 16 * G;
    __shared__ __align__(16) unsigned char smem[smem_bytes<G, KC, ST>()];
    __shared__ int rows_sh[PM];
    const half* svh = reinterpret_cast<const half*>(base + pt.svh13) + cb * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    for (int m0 = blockIdx.z * PM; m0 < cnt; m0 += PM * gridDim.z) {
        __syncthreads();   // the last pass's readers of rows_sh and smem are done
        if (threadIdx.x < PM) rows_sh[threadIdx.x] = m0 + (int)threadIdx.x < cnt ? __ldg(members + first + m0 + threadIdx.x) : -1;
        __syncthreads();
        const int rows = min(PM, cnt - m0), nblocks = (rows + 15) >> 4;
        float* E = reinterpret_cast<float*>(smem + stages_bytes<G, KC, ST>());
        switch (k) {
            case 2: pmain<G, 4, KC, ST, GEMM_W>(X, H, T, 2 * Il / 16, 0, H / 16, cb * 8, rows_sh, nblocks, smem, E); break;
            case 3: pmain<G, 6, KC, ST, GEMM_W>(X, H, T, 2 * Il / 16, 0, H / 16, cb * 8, rows_sh, nblocks, smem, E); break;
            default: __trap();
        }
        for (int r = warp; r < rows; r += PW) {   // ink2_exl3_gateup_epi's arithmetic on the pair's 128 sums
            const int p = rows_sh[r];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = E[(size_t)r * ES + 4 * lane + j];
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * s[j];
            const int c = cb * 64 + 2 * lane;
            act[(size_t)p * Il + c] = __float2bfloat16_rn(swiglu(v[0], v[1], cg, cu));
            act[(size_t)p * Il + c + 1] = __float2bfloat16_rn(swiglu(v[2], v[3], cg, cu));
        }
    }
}

// y's columns [c0 + 128 blockIdx.x, + 128) (y's column 0 = model column c0, row stride ldy), from K in kparts parts,
// each part's terms added in rank order (ink2_exl3_down_epi's arithmetic; part 0 stores, later parts add).
template <int G, int KC, int ST>
__device__ __forceinline__ void prompt_down(const half* __restrict__ X, const int* __restrict__ items,
                                            const int* __restrict__ counts, const int* __restrict__ members,
                                            const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k,
                                            int nslots, int H, int Il, float* __restrict__ y, int ldy, int c0, int kparts) {
    const int cb = blockIdx.x, it = blockIdx.y;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots || cnt <= 0) return;
    const int k = __ldg(slot_k + slot);
    const Parts pt = slot_parts(H, Il, k);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot));
    const uint32_t* T = reinterpret_cast<const uint32_t*>(base + pt.w2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int P = (Il / 16) / kparts, nt0 = (c0 >> 4) + cb * 8;
    constexpr int PM = 16 * G;
    __shared__ __align__(16) unsigned char smem[smem_bytes<G, KC, ST>()];
    __shared__ int rows_sh[PM];
    const half* svh = reinterpret_cast<const half*>(base + pt.svh2) + c0 + cb * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    for (int m0 = blockIdx.z * PM; m0 < cnt; m0 += PM * gridDim.z) {
        __syncthreads();
        if (threadIdx.x < PM) rows_sh[threadIdx.x] = m0 + (int)threadIdx.x < cnt ? __ldg(members + first + m0 + threadIdx.x) : -1;
        __syncthreads();
        const int rows = min(PM, cnt - m0), nblocks = (rows + 15) >> 4;
        for (int part = 0; part < kparts; ++part) {
            float* E = reinterpret_cast<float*>(smem + stages_bytes<G, KC, ST>());
            switch (k) {
                case 2: pmain<G, 4, KC, ST, GEMM_W>(X, Il, T, H / 16, part * P, P, nt0, rows_sh, nblocks, smem, E); break;
                case 3: pmain<G, 6, KC, ST, GEMM_W>(X, Il, T, H / 16, part * P, P, nt0, rows_sh, nblocks, smem, E); break;
                default: __trap();
            }
            for (int r = warp; r < rows; r += PW) {
                const int p = rows_sh[r];
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = E[(size_t)r * ES + 4 * lane + j];
                fwht128(v, lane);
                float* yr = y + (size_t)p * ldy + cb * 128 + 4 * lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // explicit roundings, as ink2_exl3_down_epi: no product may fuse into the parts' add
                    const float term = __fmul_rn(__fmul_rn(v[j], HAD_SCALE), s[j]);
                    yr[j] = part == 0 ? term : __fadd_rn(yr[j], term);
                }
            }
            __syncthreads();   // E (the stages) is read before the next part's loads
        }
    }
}


}  // namespace ink2_prompt

// Entries (G 16-row tiles a program, KC k tiles a stage, ST stages, MB programs an SM at least): grid (columns / 128,
// plan items, the item's passes: 64 / (16 G)), 256 threads.
#define INK2_EXL3_PROMPT(SUFFIX, G, KC, ST, MB)                                                                     \
    extern "C" __global__ void __launch_bounds__(ink2_prompt::PT, MB) ink2_exl3_prompt_gu##SUFFIX(               \
        const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,                   \
        const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,                            \
        const int* __restrict__ slot_k, int nslots, int H, int Il, float cg, float cu,                               \
        __nv_bfloat16* __restrict__ act) {                                                                           \
        ink2_prompt::prompt_gu<G, KC, ST>(X, items, counts, members, slot_ptr, slot_k, nslots, H, Il, cg, cu, act);  \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(ink2_prompt::PT, MB) ink2_exl3_prompt_down##SUFFIX(             \
        const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,                   \
        const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,                            \
        const int* __restrict__ slot_k, int nslots, int H, int Il, float* __restrict__ y, int ldy, int c0,           \
        int kparts) {                                                                                                \
        ink2_prompt::prompt_down<G, KC, ST>(X, items, counts, members, slot_ptr, slot_k, nslots, H, Il, y, ldy, c0,  \
                                            kparts);                                                                 \
    }
INK2_EXL3_PROMPT(_g4k2s2, 4, 2, 2, 2)   // the forward's default (kern.exl3_prompt_configs[0])
INK2_EXL3_PROMPT(_g2k8s2, 2, 8, 2, 2)
INK2_EXL3_PROMPT(_g2k4s3, 2, 4, 3, 2)
INK2_EXL3_PROMPT(_g2k4s2, 2, 4, 2, 3)

// ==== e-load (prefill lane phase 5): the store's load on the device (exl3.zig loadDirect) ===========================
// One copy job: `rows` rows of `bytes` bytes from src (a mapped pinned stage the reads landed in) to dst (a slot), the
// rows `spitch` / `dpitch` bytes apart. exl3.zig packSlot's byte permutation is six such jobs an expert (w13's rank
// column tiles of every row, w2's rank rows, the four scales); nothing is decoded or rounded. Grid (row blocks, jobs),
// 256 threads; the widest access every address and length allow (a stage keeps its file's offsets mod 4096).
struct Ink2CutJob {
    unsigned long long src, dst, spitch, dpitch;
    unsigned int rows, bytes;
};

template <typename V>
__device__ __forceinline__ void ink2_cut_row(const unsigned char* s, unsigned char* d, unsigned int bytes) {
    const V* sv = reinterpret_cast<const V*>(s);
    V* dv = reinterpret_cast<V*>(d);
    for (unsigned int i = threadIdx.x; i < bytes / sizeof(V); i += blockDim.x) dv[i] = sv[i];
}

extern "C" __global__ void __launch_bounds__(256) ink2_exl3_cut(const Ink2CutJob* __restrict__ jobs) {
    const Ink2CutJob j = jobs[blockIdx.y];
    const unsigned long long a = j.src | j.dst | j.spitch | j.dpitch | j.bytes;
    for (unsigned int r = blockIdx.x; r < j.rows; r += gridDim.x) {
        const unsigned char* s = reinterpret_cast<const unsigned char*>(j.src + (unsigned long long)r * j.spitch);
        unsigned char* d = reinterpret_cast<unsigned char*>(j.dst + (unsigned long long)r * j.dpitch);
        if ((a & 15) == 0) ink2_cut_row<uint4>(s, d, j.bytes);
        else if ((a & 7) == 0) ink2_cut_row<uint2>(s, d, j.bytes);
        else if ((a & 3) == 0) ink2_cut_row<unsigned int>(s, d, j.bytes);
        else ink2_cut_row<unsigned char>(s, d, j.bytes);
    }
}

// Each slot's 64-bit hash (exl3.zig slotHash is the host's same function): 256 threads each fold the slot's 8-byte
// words t, t + 256, ... with their index (fmix64), then thread 0 folds the 256 partial hashes in thread order. A check
// (TF_INKLING2_EXL3_HASH, tf-inkling2-exl3-load): two loads' slots compared without copying 89 GiB to the host.
__device__ __forceinline__ unsigned long long ink2_fmix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

extern "C" __global__ void __launch_bounds__(256) ink2_exl3_hash(const unsigned long long* __restrict__ slot_ptr,
                                                                 const int* __restrict__ slot_k, unsigned long long bytes2,
                                                                 unsigned long long bytes3,
                                                                 unsigned long long* __restrict__ out) {
    __shared__ unsigned long long hs[256];
    const unsigned long long* p = reinterpret_cast<const unsigned long long*>(slot_ptr[blockIdx.x]);
    const unsigned long long n = (slot_k[blockIdx.x] == 2 ? bytes2 : bytes3) / 8;
    unsigned long long h = 0x9E3779B97F4A7C15ull ^ (unsigned long long)threadIdx.x;
    for (unsigned long long i = threadIdx.x; i < n; i += 256) h = ink2_fmix64(h ^ p[i] ^ (i * 0xD6E8FEB86659FD93ull));
    hs[threadIdx.x] = h;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long H = n;
        for (int t = 0; t < 256; ++t) H = ink2_fmix64(H ^ hs[t]);
        out[blockIdx.x] = H;
    }
}

// ==== e-prefill (prefill lane, phase 5): the wide-chunk EXL3 kernels, v3 (TF_INKLING2_EXL3_PROMPT=v3g4d) ===============
// ink2_prompt's arithmetic bit for bit: each warp one 16-column tile over K with the activations as the mma's A operand,
// gemm_body's W = 4 K chains each summed from zero and added to the running total when it ends (((c0 + c1) + c2) + c3,
// the first chain as it is), then the same epilogues. What differs is where values live and when work runs:
//  - the running totals stay in registers, started at -0 (-0 + x == x for every x: the first chain's bits, no select),
//    and step once a chunk where KC divides the K tiles and the chains (every shape of this model); E (the epilogue's
//    sums) only aliases the stages after the K loop, which leaves the shared memory to deeper cp.async pipelines at two
//    programs an SM (ink2_prompt keeps the totals in E: 33.8 KB at 64 rows);
//  - an item's passes (members past the program's 16 G rows) are programs next to each other in grid x, beside the
//    column block (ink2_prompt: grid z, the slowest dimension, so each pass streamed the item's weights from DRAM again
//    a whole grid later); the items of one expert follow each other in y: programs reading the same weight tiles run
//    together and share them through L2;
//  - each thread's copy sources are computed once and advanced a chunk at a time, the lane's decode words located once
//    a program, the mul1 pairs packed with one PRMT (cb_pair's form of decode2: the same half2 bits);
//  - the plan's item size is the host's (grid x = column blocks x npass), so 64- and 128-pair plans both run.
namespace ink2_p2 {

using ink2_prompt::ES;
using ink2_prompt::PT;
using ink2_prompt::PW;
using ink2_prompt::xrow;

template <int G, int K2, int KC>
__host__ __device__ constexpr int stage_bytes() { return 16 * G * xrow<KC>() * 2 + PW * KC * tf_exl3::tile_words<K2>() * 4; }
// the stages (K3's, the larger) or E [16 G rows][ES] fp32, whichever is larger: E aliases the stages
template <int G, int KC, int ST>
__host__ __device__ constexpr int smem_bytes() {
    return ST * stage_bytes<G, 6, KC>() > 16 * G * ES * 4 ? ST * stage_bytes<G, 6, KC>() : 16 * G * ES * 4;
}

__device__ __forceinline__ uint32_t dyn_smem() {
    uint32_t n;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(n));
    return n;
}

// This lane's two words of a tile (lane_words<K2>() == 2 for 2 and 3 bits: tf_exl3::lane_start's word and the next,
// modulo the tile) and the bit offset of its first window, once a program; decode_pre is tf_exl3::decode_lane on them.
template <int K2>
struct LanePre {
    uint32_t w0, w1;   // byte offsets in the tile
    int offset;
    __device__ __forceinline__ explicit LanePre(int lane) {
        static_assert(tf_exl3::lane_words<K2>() == 2, "the 2- and 3-bit tiles: two words a lane");
        int word;
        tf_exl3::lane_start<K2>(lane, word, offset);
        w0 = (uint32_t)(word % tf_exl3::tile_words<K2>()) * 4;
        w1 = (uint32_t)((word + 1) % tf_exl3::tile_words<K2>()) * 4;
    }
};

// tf_exl3::decode_lane<K2, mul1> of the tile at shared address `tile`: lane_states' windows, then each pair's mul1 values
// as tf_exl3x::cb_pair packs them (one PRMT where decode2 masks, shifts and ors: the same half2 bits)
template <int K2>
__device__ __forceinline__ void decode_pre(uint32_t tile, const LanePre<K2>& lp, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t w0, w1;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w0) : "r"(tile + lp.w0));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w1) : "r"(tile + lp.w1));
    const uint32_t hi = __funnelshift_l(w1, w0, lp.offset), mid = __funnelshift_l(0u, w1, lp.offset), lo = 0u;
    uint32_t st[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) st[j] = tf_exl3::window16(hi, mid, lo, tf_exl3::stream_end<K2>(j) - tf_exl3::stream_end<K2>(0));
    b0[0] = cb_pair<CB_MUL1>(st[0], st[1]);
    b0[1] = cb_pair<CB_MUL1>(st[2], st[3]);
    b1[0] = cb_pair<CB_MUL1>(st[4], st[5]);
    b1[1] = cb_pair<CB_MUL1>(st[6], st[7]);
}

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// One part of one pass: k tiles [kt_base, kt_base + P) in W chains of P / W, rows rows_sh[0 .. 16 NB) (NB live 16-row
// tiles), this warp's column tile nt0 + warp. CE: KC divides P and the chains (every chain ends with a chunk: the chain
// step once a chunk, outside the unrolled k tiles); else every k tile checks (a shape no deployment has: the tiny's).
// The running totals start at -0 (-0 + chain = the chain, bit for bit: the first chain's sum as it is). Each thread's
// copies are fixed pieces of every stage, their sources advanced a chunk at a time. Ends with the totals in E
// (smem[0 ..), [16 NB rows][ES]) and the program synchronized: the caller's epilogue reads E, and syncs again before
// the next part's loads.
template <int G, int NB, int K2, int KC, int ST, int UK, bool CE>
__device__ __forceinline__ void pmain(const half* __restrict__ X, int K, const uint32_t* __restrict__ T, int NTILES,
                                      int kt_base, int P, int nt0, const int* rows_sh, unsigned char* smem) {
    constexpr int TW = tf_exl3::tile_words<K2>(), XR = xrow<KC>(), SB = stage_bytes<G, K2, KC>();
    constexpr int XB = 16 * G * XR * 2;                // a stage's row bytes (the weight tiles follow)
    constexpr int RP = NB * 16 * 2 * KC, RPT = (RP + PT - 1) / PT;   // row pieces (16 bytes) a stage, a thread
    constexpr int WP = KC * PW * (TW / 4);            // weight pieces a stage
    constexpr int WPT = (WP + PT - 1) / PT;
    constexpr int W = GEMM_W;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t = lane & 3, warp_c = warp * 16;
    const int nchunks = (P + KC - 1) / KC, per_chain = P / W;
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem);
    // this thread's row pieces: idx = tid + i PT: row idx / (2 KC), piece idx % (2 KC) (k tile / 2, half % 2)
    const half* rsrc[RPT];
    uint32_t rdst[RPT];
    int rkk[RPT];
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
        const int idx = tid + i * PT, r = idx / (2 * KC), j = idx % (2 * KC);
        const int p = idx < RP ? rows_sh[r] : -1;
        rsrc[i] = p >= 0 ? X + (size_t)p * K + (size_t)kt_base * 16 + j * 8 : nullptr;
        rdst[i] = idx < RP ? (uint32_t)(r * XR + j * 8) * 2 : 0xffffffffu;   // none: past the live rows
        rkk[i] = j >> 1;
    }
    // this thread's weight pieces: k tile kk, column tile q, 16 bytes `piece` of its TW words
    const uint32_t* wsrc[WPT];
    uint32_t wdst[WPT];
    int wkk[WPT];
#pragma unroll
    for (int i = 0; i < WPT; ++i) {
        const int idx = tid + i * PT;
        const int kk = idx / (PW * (TW / 4)), rem = idx % (PW * (TW / 4)), q = rem / (TW / 4), piece = rem % (TW / 4);
        wsrc[i] = T + ((size_t)(kt_base + kk) * NTILES + nt0 + q) * TW + piece * 4;
        wdst[i] = idx < WP ? XB + (uint32_t)((kk * PW + q) * TW + piece * 4) * 4 : 0xffffffffu;
        wkk[i] = kk;
    }
    const size_t wstep = (size_t)KC * NTILES * TW;   // words a chunk
    auto load = [&](int ch, uint32_t stage) {
#pragma unroll
        for (int i = 0; i < RPT; ++i)
            if (rdst[i] != 0xffffffffu) {
                const bool ok = rsrc[i] != nullptr && (CE || ch * KC + rkk[i] < P);
                tf_common::cp16z(stage + rdst[i], ok ? rsrc[i] + (size_t)ch * KC * 16 : X, ok);
            }
#pragma unroll
        for (int i = 0; i < WPT; ++i)
            if (wdst[i] != 0xffffffffu) {
                const bool ok = CE || ch * KC + wkk[i] < P;
                tf_common::cp16z(stage + wdst[i], ok ? wsrc[i] + (size_t)ch * wstep : T, ok);
            }
    };
    const LanePre<K2> lp(lane);
    const uint32_t a_lane = (uint32_t)(((lane & 7) + ((lane >> 3) & 1) * 8) * XR + (lane >> 4) * 8) * 2;
    const uint32_t w_warp = XB + (uint32_t)warp * TW * 4;
    float acc[NB][2][4], tot[NB][2][4];
#pragma unroll
    for (int g = 0; g < NB; ++g)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                acc[g][h][c] = 0.f;
                tot[g][h][c] = -0.f;
            }
    auto chain_end = [&]() {   // total = total + chain, the chain restarted from zero
#pragma unroll
        for (int g = 0; g < NB; ++g)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    tot[g][h][c] = tot[g][h][c] + acc[g][h][c];
                    acc[g][h][c] = 0.f;
                }
    };
#pragma unroll
    for (int c = 0; c < ST - 1; ++c) {
        if (c < nchunks) load(c, s0 + (uint32_t)c * SB);
        tf_common::cp_commit();
    }
    const int chain_chunks = CE ? per_chain / KC : 0;
    int left = CE ? chain_chunks : per_chain;   // chunks (CE) or k tiles left in the current chain
    int cs = 0;                                 // chunk ch's stage (ch % ST)
    for (int ch = 0; ch < nchunks; ++ch) {
        tf_common::cp_wait<ST - 2>();
        __syncthreads();   // chunk ch landed for every thread; the stage the next load overwrites is read by none
        {
            const int ls = cs == 0 ? ST - 1 : cs - 1;   // (ch + ST - 1) % ST
            if (ch + ST - 1 < nchunks) load(ch + ST - 1, s0 + (uint32_t)ls * SB);
        }
        tf_common::cp_commit();
        const uint32_t st = s0 + (uint32_t)cs * SB;
#pragma unroll UK
        for (int kk = 0; kk < KC; ++kk) {
            if (!CE && ch * KC + kk >= P) break;
            uint32_t b0[2], b1[2];
            decode_pre<K2>(st + w_warp + (uint32_t)kk * PW * TW * 4, lp, b0, b1);
            uint32_t a[4];
#pragma unroll
            for (int g = 0; g < NB; ++g) {
                ldsm4(a, st + a_lane + (uint32_t)(16 * g * XR + kk * 16) * 2);
                tf_exl3::mma16816(acc[g][0], a, b0);
                tf_exl3::mma16816(acc[g][1], a, b1);
            }
            if constexpr (!CE) {
                if (--left == 0) {
                    chain_end();
                    left = per_chain;
                }
            }
        }
        if constexpr (CE) {
            if (--left == 0) {
                chain_end();
                left = chain_chunks;
            }
        }
        cs = cs == ST - 1 ? 0 : cs + 1;
    }
    tf_common::cp_wait<0>();
    __syncthreads();   // every warp is done with the stages: E takes their place
    float* E = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int g = 0; g < NB; ++g)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            *reinterpret_cast<float2*>(E + (size_t)(16 * g + g8) * ES + warp_c + h * 8 + 2 * t) = make_float2(tot[g][h][0], tot[g][h][1]);
            *reinterpret_cast<float2*>(E + (size_t)(16 * g + g8 + 8) * ES + warp_c + h * 8 + 2 * t) = make_float2(tot[g][h][2], tot[g][h][3]);
        }
    __syncthreads();   // E holds every warp's totals
}

// pmain at the pass's live 16-row tiles (1 .. G, a compile-time count: every mma unpredicated, accumulating in place)
template <int G, int K2, int KC, int ST, int UK, bool CE>
__device__ __forceinline__ void pmain_nb_ce(int nblocks, const half* __restrict__ X, int K, const uint32_t* __restrict__ T,
                                            int NTILES, int kt_base, int P, int nt0, const int* rows_sh, unsigned char* smem) {
#define INK2_P2_NB(N) \
    if constexpr (G >= N) \
        if (nblocks == N) return pmain<G, N, K2, KC, ST, UK, CE>(X, K, T, NTILES, kt_base, P, nt0, rows_sh, smem);
    INK2_P2_NB(8) INK2_P2_NB(7) INK2_P2_NB(6) INK2_P2_NB(5) INK2_P2_NB(4) INK2_P2_NB(3) INK2_P2_NB(2)
#undef INK2_P2_NB
    pmain<G, 1, K2, KC, ST, UK, CE>(X, K, T, NTILES, kt_base, P, nt0, rows_sh, smem);
}

// pmain_nb_ce with CE when KC divides P and its W chains (every deployment shape: P 288, 72 or 36 with KC 4 or 2 / 3)
template <int G, int K2, int KC, int ST, int UK>
__device__ __forceinline__ void pmain_nb(int nblocks, const half* __restrict__ X, int K, const uint32_t* __restrict__ T,
                                         int NTILES, int kt_base, int P, int nt0, const int* rows_sh, unsigned char* smem) {
    if (P % KC == 0 && (P / GEMM_W) % KC == 0)
        pmain_nb_ce<G, K2, KC, ST, UK, true>(nblocks, X, K, T, NTILES, kt_base, P, nt0, rows_sh, smem);
    else
        pmain_nb_ce<G, K2, KC, ST, UK, false>(nblocks, X, K, T, NTILES, kt_base, P, nt0, rows_sh, smem);
}

// w13 and its epilogue (ink2_prompt::prompt_gu's arithmetic): grid (2 Il / 128 column blocks x npass, plan items); the
// program (column block cb, pass q) takes the item's members [16 G (q + npass i), + 16 G) for i = 0, 1, ...
template <int G, int KC, int ST>
__device__ __forceinline__ void gu(const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,
                                   const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,
                                   const int* __restrict__ slot_k, int nslots, int H, int Il, float cg, float cu,
                                   __nv_bfloat16* __restrict__ act, int npass) {
    constexpr int PM = 16 * G;
    const int ncb = gridDim.x / npass, cb = blockIdx.x % ncb, pass = blockIdx.x / ncb, it = blockIdx.y;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots || pass * PM >= cnt) return;
    if (dyn_smem() < (uint32_t)smem_bytes<G, KC, ST>()) __trap();   // the host's launch must give the stages
    const int k = __ldg(slot_k + slot);
    const Parts pt = slot_parts(H, Il, k);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot));
    const uint32_t* T = reinterpret_cast<const uint32_t*>(base);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    extern __shared__ __align__(16) unsigned char p2_smem[];
    __shared__ int rows_sh[PM];
    const half* svh = reinterpret_cast<const half*>(base + pt.svh13) + cb * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    for (int m0 = pass * PM; m0 < cnt; m0 += PM * npass) {
        __syncthreads();   // the last pass's readers of rows_sh and E are done
        if (threadIdx.x < PM) rows_sh[threadIdx.x] = m0 + (int)threadIdx.x < cnt ? __ldg(members + first + m0 + threadIdx.x) : -1;
        __syncthreads();
        const int rows = min(PM, cnt - m0), nblocks = (rows + 15) >> 4;
        switch (k) {
            case 2: pmain_nb<G, 4, KC, ST, KC>(nblocks, X, H, T, 2 * Il / 16, 0, H / 16, cb * 8, rows_sh, p2_smem); break;
            case 3: pmain_nb<G, 6, KC, ST, KC>(nblocks, X, H, T, 2 * Il / 16, 0, H / 16, cb * 8, rows_sh, p2_smem); break;
            default: __trap();
        }
        const float* E = reinterpret_cast<const float*>(p2_smem);
        for (int r = warp; r < rows; r += PW) {   // ink2_exl3_gateup_epi's arithmetic on the pair's 128 sums
            const int p = rows_sh[r];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = E[(size_t)r * ES + 4 * lane + j];
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * s[j];
            const int c = cb * 64 + 2 * lane;
            act[(size_t)p * Il + c] = __float2bfloat16_rn(swiglu(v[0], v[1], cg, cu));
            act[(size_t)p * Il + c + 1] = __float2bfloat16_rn(swiglu(v[2], v[3], cg, cu));
        }
    }
}

// w2 and its epilogue (ink2_prompt::prompt_down's arithmetic) on y's columns [c0 + 128 cb, + 128) (y's column 0 = model
// column c0, row stride ldy), K in kparts parts, each part's terms added in rank order (part 0 stores, later parts add).
template <int G, int KC, int ST>
__device__ __forceinline__ void down(const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,
                                     const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,
                                     const int* __restrict__ slot_k, int nslots, int H, int Il, float* __restrict__ y,
                                     int ldy, int c0, int kparts, int npass) {
    constexpr int PM = 16 * G;
    const int ncb = gridDim.x / npass, cb = blockIdx.x % ncb, pass = blockIdx.x / ncb, it = blockIdx.y;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots || pass * PM >= cnt) return;
    if (dyn_smem() < (uint32_t)smem_bytes<G, KC, ST>()) __trap();
    const int k = __ldg(slot_k + slot);
    const Parts pt = slot_parts(H, Il, k);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot));
    const uint32_t* T = reinterpret_cast<const uint32_t*>(base + pt.w2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int P = (Il / 16) / kparts, nt0 = (c0 >> 4) + cb * 8;
    extern __shared__ __align__(16) unsigned char p2_smem[];
    __shared__ int rows_sh[PM];
    const half* svh = reinterpret_cast<const half*>(base + pt.svh2) + c0 + cb * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    for (int m0 = pass * PM; m0 < cnt; m0 += PM * npass) {
        __syncthreads();
        if (threadIdx.x < PM) rows_sh[threadIdx.x] = m0 + (int)threadIdx.x < cnt ? __ldg(members + first + m0 + threadIdx.x) : -1;
        __syncthreads();
        const int rows = min(PM, cnt - m0), nblocks = (rows + 15) >> 4;
        for (int part = 0; part < kparts; ++part) {
            switch (k) {
                case 2: pmain_nb<G, 4, KC, ST, KC>(nblocks, X, Il, T, H / 16, part * P, P, nt0, rows_sh, p2_smem); break;
                case 3: pmain_nb<G, 6, KC, ST, KC>(nblocks, X, Il, T, H / 16, part * P, P, nt0, rows_sh, p2_smem); break;
                default: __trap();
            }
            const float* E = reinterpret_cast<const float*>(p2_smem);
            for (int r = warp; r < rows; r += PW) {
                const int p = rows_sh[r];
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = E[(size_t)r * ES + 4 * lane + j];
                fwht128(v, lane);
                float* yr = y + (size_t)p * ldy + cb * 128 + 4 * lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // explicit roundings, as ink2_exl3_down_epi: no product may fuse into the parts' add
                    const float term = __fmul_rn(__fmul_rn(v[j], HAD_SCALE), s[j]);
                    yr[j] = part == 0 ? term : __fadd_rn(yr[j], term);
                }
            }
            __syncthreads();   // E (the stages) is read before the next part's loads
        }
    }
}

}  // namespace ink2_p2

// Entries (G 16-row tiles a program; gate-up KCG k tiles a stage, STG stages; down KCD, STD; MB programs an SM at least):
// grid (column blocks x npass, plan items), 256 threads, ink2_p2::smem_bytes<G, KC, ST>() of dynamic shared memory each
// (kern.zig's table). KCG 4 and KCD 3 divide every deployment shape's K tiles and chains (288 / 72; 72 / 18, 36 / 9).
#define INK2_EXL3_P2(SUFFIX, G, KCG, STG, KCD, STD, MB)                                                             \
    extern "C" __global__ void __launch_bounds__(ink2_prompt::PT, MB) ink2_exl3_prompt_gu##SUFFIX(               \
        const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,                   \
        const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,                            \
        const int* __restrict__ slot_k, int nslots, int H, int Il, float cg, float cu,                               \
        __nv_bfloat16* __restrict__ act, int npass) {                                                                \
        ink2_p2::gu<G, KCG, STG>(X, items, counts, members, slot_ptr, slot_k, nslots, H, Il, cg, cu, act, npass);    \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(ink2_prompt::PT, MB) ink2_exl3_prompt_down##SUFFIX(             \
        const half* __restrict__ X, const int* __restrict__ items, const int* __restrict__ counts,                   \
        const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,                            \
        const int* __restrict__ slot_k, int nslots, int H, int Il, float* __restrict__ y, int ldy, int c0,           \
        int kparts, int npass) {                                                                                     \
        ink2_p2::down<G, KCD, STD>(X, items, counts, members, slot_ptr, slot_k, nslots, H, Il, y, ldy, c0, kparts,   \
                                   npass);                                                                           \
    }
// v3g4d: 64 rows a weight tile, two programs an SM; gate-up 4 stages of 4 k tiles, down 5 of 3 (the hill-climb's best:
// out/prefill/p5/e-prefill/REPORT.md, where the other builds tried are recorded)
INK2_EXL3_P2(_v3g4d, 4, 4, 4, 3, 5, 2)

// ---- e-prefill: the activation and the rotation in one pass (TF_INKLING2_EXL3_ACTROT=1) -----------------------------
// ink2_exl3_act then ink2_exl3_rot, the same bits, without the fp16 xq rows between them: a warp takes 128 values of a
// row, 4 a lane (the rotation's layout), quantizes them as ink2_exl3_act does (NVFP4's 16-value blocks over 4 lanes,
// MXFP8's 32 over 8: the block max and the 4/6 choice's error tree by shuffles in the tree's order), rounds them to fp16
// as act's output, and rotates them as rot does (fp16(xq * suh), fwht128, fp16(v / sqrt(128))).
namespace ink2_ar {

constexpr unsigned FULL = 0xffffffffu;

// nvfp4_block's arithmetic on a 16-value block held by 4 lanes (lane 4b + i: values 4i .. 4i + 3 of block b)
__device__ __forceinline__ void nvfp4_q4(const float (&x)[4], float amax, float (&out)[4]) {
    const float enc = fminf(__fdiv_rn(1536.0f, amax), 3.4028234663852886e38f);
    const float dec = __fdiv_rn(1.0f, enc);
    const float outer = __fmul_rn(amax, 1.0f / 1536.0f);
    float bm = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) bm = fmaxf(bm, fabsf(x[i]));
    bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, 1));   // a max: the order of its operands does not matter
    bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, 2));
    const float s6 = e4m3_rn(fminf(__fmul_rn(bm, __fmul_rn(enc, 1.0f / 6.0f)), 448.0f));
    const float s4 = e4m3_rn(fminf(__fmul_rn(bm, __fmul_rn(enc, 0.25f)), 448.0f));
    const float i6 = fminf(__fdiv_rn(1.0f, __fmul_rn(s6, dec)), 3.4028234663852886e38f);
    const float i4 = fminf(__fdiv_rn(1.0f, __fmul_rn(s4, dec)), 3.4028234663852886e38f);
    float r6[4], r4[4], e6[4], e4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r6[i] = e2m1_rn(__fmul_rn(x[i], i6));
        r4[i] = e2m1_rn(__fmul_rn(x[i], i4));
        e6[i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r6[i], s6), outer), x[i]));
        e4[i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r4[i], s4), outer), x[i]));
    }
    // _block_error's tree (nvfp4_block): pairs, then pairs of pairs, in this lane (its 4 values are a level-4 node), then
    // the lanes' nodes in pairs (lanes 4b, 4b + 1 and 4b + 2, 4b + 3), then the two halves: the same adds (an add's two
    // operands commute exactly)
    float a6 = __fadd_rn(__fadd_rn(e6[0], e6[1]), __fadd_rn(e6[2], e6[3]));
    float a4 = __fadd_rn(__fadd_rn(e4[0], e4[1]), __fadd_rn(e4[2], e4[3]));
    a6 = __fadd_rn(a6, __shfl_xor_sync(FULL, a6, 1));
    a4 = __fadd_rn(a4, __shfl_xor_sync(FULL, a4, 1));
    a6 = __fadd_rn(a6, __shfl_xor_sync(FULL, a6, 2));
    a4 = __fadd_rn(a4, __shfl_xor_sync(FULL, a4, 2));
    const bool four = a4 < a6;
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __fmul_rn(four ? __fmul_rn(r4[i], s4) : __fmul_rn(r6[i], s6), outer);
}

// mxfp8_block's arithmetic on a 32-value block held by 8 lanes
__device__ __forceinline__ void mxfp8_q8(const float (&x)[4], float (&out)[4]) {
    float bm = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) bm = fmaxf(bm, fabsf(x[i]));
    bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, 1));
    bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, 2));
    bm = fmaxf(bm, __shfl_xor_sync(FULL, bm, 4));
    const float raw = __fmul_rn(bm, 1.0f / 448.0f);
    int e = (int)(((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23);
    e = raw <= 0x1p-127f ? 0 : min(e, 254);
    const float scale = e ? __uint_as_float((uint32_t)e << 23) : __uint_as_float(0x00400000u);
    const float inv = e == 0 ? 0x1p127f : __frcp_rn(scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __fmul_rn(e4m3_rn(fminf(fmaxf(__fmul_rn(x[i], inv), -448.0f), 448.0f)), scale);
}

// This lane's 4 values of a 128-block of a bf16 row, quantized (mode 0 none, 1 NVFP4 under amax, 2 MXFP8) and rounded to
// fp16 as ink2_exl3_act writes them (returned as floats)
__device__ __forceinline__ void act4(const __nv_bfloat16* __restrict__ src, int mode, float amax, float (&xq)[4]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(src);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
    float v[4], q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(h[j]);
    if (mode == 1) nvfp4_q4(v, amax, q);
    else if (mode == 2) mxfp8_q8(v, q);
    else {
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = v[j];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) xq[j] = __half2float(__float2half_rn(q[j]));
}

// ink2_exl3_rot's arithmetic on this lane's 4 values: fp16(xq * suh), fwht128, fp16(v / sqrt(128)) into o (8 bytes)
__device__ __forceinline__ void rot4(const float (&xq)[4], const half* __restrict__ suh, int lane, half* __restrict__ o) {
    const uint2 sr = *reinterpret_cast<const uint2*>(suh);
    const half* sh = reinterpret_cast<const half*>(&sr);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __half2float(__float2half_rn(xq[j] * __half2float(sh[j])));
    fwht128(v, lane);
    half2 r0 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
    half2 r1 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
    uint2 w;
    w.x = *reinterpret_cast<uint32_t*>(&r0);
    w.y = *reinterpret_cast<uint32_t*>(&r1);
    *reinterpret_cast<uint2*>(o) = w;
}

}  // namespace ink2_ar

// w13's input: every token's activation (ink2_exl3_act on its K values, mode 1 under the row's own amax) and each of its
// topk pairs' rotation by its expert's suh13 (slot = lut[picks[t topk + j]]), out [t topk + j][K] fp16 (ink2_exl3_rot
// with row_slots topk). A program a token, 8 warps over its 128-value blocks.
extern "C" __global__ void __launch_bounds__(256) ink2_exl3_actrot13(
    const __nv_bfloat16* __restrict__ x, int x_stride, int K, int mode, const int* __restrict__ picks, int topk,
    const int* __restrict__ lut, const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k,
    int nslots, int H, int Il, half* __restrict__ out) {
    const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const __nv_bfloat16* xr = x + (size_t)t * x_stride;
    __shared__ float red[256];
    __shared__ const half* suh_sh[16];
    if (threadIdx.x < topk) {
        const int e = __ldg(picks + (size_t)t * topk + threadIdx.x);
        const int slot = e < 0 ? -1 : __ldg(lut + e);
        suh_sh[threadIdx.x] = slot < 0 || slot >= nslots ? nullptr
            : reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) + slot_parts(H, Il, __ldg(slot_k + slot)).suh13);
    }
    float amax = 0.0f;
    if (mode == 1) {   // the row's max |x| (ink2_exl3_act's: a max, exact in any order)
        float m = 0.0f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(xr[i])));
        red[threadIdx.x] = m;
        __syncthreads();
        for (int s2 = 128; s2 > 0; s2 >>= 1) {
            if (threadIdx.x < s2) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s2]);
            __syncthreads();
        }
        amax = red[0];
    } else __syncthreads();   // suh_sh
    for (int blk = warp; blk < K / 128; blk += 8) {
        float xq[4];
        ink2_ar::act4(xr + blk * 128 + 4 * lane, mode, amax, xq);
        for (int j = 0; j < topk; ++j) {
            const half* suh = suh_sh[j];
            if (suh == nullptr) continue;   // a pick without a slot: ink2_exl3_rot never visits it either
            ink2_ar::rot4(xq, suh + blk * 128 + 4 * lane, lane, out + ((size_t)t * topk + j) * K + blk * 128 + 4 * lane);
        }
    }
}

// w2's input: each routed pair's activation (ink2_exl3_act on its K = Il values: mode 1 under amax_all's max over the
// world's entries of the pair, which the caller gathers (exl3RowAmax first at world 1), mode 2 MXFP8) and its rotation
// by its slot's suh2 (ink2_exl3_rot, row_slots 0), out [p][K] fp16. A program a (plan item, 128 inputs), one warp.
extern "C" __global__ void __launch_bounds__(32) ink2_exl3_actrot2(
    const __nv_bfloat16* __restrict__ x, int x_stride, int mode, const float* __restrict__ amax_all, int world,
    int amax_stride, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k, int nslots, int H, int Il,
    half* __restrict__ out, int K) {
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots) return;
    const Parts pt = slot_parts(H, Il, __ldg(slot_k + slot));
    const half* suh = reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) + pt.suh2) +
                      blk * 128 + 4 * lane;
    for (int m = 0; m < cnt; ++m) {
        const int p = __ldg(members + first + m);
        float amax = 0.0f;
        if (mode == 1)
            for (int r = 0; r < world; ++r) amax = fmaxf(amax, __ldg(amax_all + (size_t)r * amax_stride + p));
        float xq[4];
        ink2_ar::act4(x + (size_t)p * x_stride + blk * 128 + 4 * lane, mode, amax, xq);
        ink2_ar::rot4(xq, suh, lane, out + (size_t)p * K + blk * 128 + 4 * lane);
    }
}
// ==== end e-prefill =================================================================================================
// ==== e-dense (prefill lane phase 5): a dense EXL3 linear's wide prompt chunk, its output side ====
// y = (yr @ H128) / sqrt(128) * svh for the fp32 products yr = xh @ W_q of a wide chunk (dense_exl3.zig's wide path:
// core's input rotation and candidate unpack, then one cuBLASLt fp16 GEMM with fp32 sums): the decode linear's finish
// (exl3_linear.cu finish, its fwht128 and scale order) on every row; one warp a (row, 128 columns), 4 values a lane;
// rounded once to bf16, or kept fp32 (o_proj's TP partials).
extern "C" __global__ void __launch_bounds__(128) ink2_dx_finish(const float* __restrict__ yr, const __half* __restrict__ svh,
                                                                void* __restrict__ y, int n, int y_f32) {
  const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
  if (blk * 128 >= n) return;
  const int col = blk * 128 + 4 * lane;
  const size_t at = (size_t)row * n + col;
  const float4 u = *reinterpret_cast<const float4*>(yr + at);
  float v[4] = {u.x, u.y, u.z, u.w};
  tf_exl3::fwht128(v, lane);
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = v[j] * tf_exl3::HAD_SCALE * __half2float(svh[col + j]);
  if (y_f32) {
    *reinterpret_cast<float4*>(static_cast<float*>(y) + at) = make_float4(v[0], v[1], v[2], v[3]);
  } else {
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&a);
    o.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(y) + at) = o;
  }
}
// ==== end e-dense ====

// ==== e-step (prefill lane phase 5): a projection group of EXL3 dense linears reading the same rows ===================
// (TF_INKLING2_DENSE_GROUP, on e-dense's dense EXL3 sets): up to five 6-bit mul1 linears of one K (a layer's q, k, v, r
// and gate on the attention input) in two launches instead of ten: each member's input rotation (core's rot_in body,
// blockIdx.z the member), then each member's decode-row linear on its own blocks of one grid (core's linear_body<12, 2,
// 4>, member i's blocks ends[i - 1] .. ends[i]: the linear its own launch would run, its plan's K split and warps).
// Copied from core's exl3_linear.cu (the same arithmetic, so a member's bits are its own launch's; core's group kernel
// takes 4-bit layers and three members). PDL as core's: the rotation lets the linear launch, the linear prefetches its
// first k step's weights before waiting for the rotation's rows.
namespace es_x3 {
using namespace tf_exl3;   // fwht128, HAD_SCALE and mma16816 named tf_exl3:: (the file's anonymous namespace has its own)

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

struct LinearArgs {
    const void* xh;
    const uint32_t* t;
    long long stride_k;
    long long stride_nb;
    const half* svh;
    const half* bias;
    void* y;
    float* z;
    int* counters;
    int y_dtype;
    int n, sk, m, k;
};

struct RotGroup5 {
    const void* x;
    const half* suh[5];
    half* xh[5];
    int x_dtype;
    int m, k;
};

struct Group5 {
    LinearArgs l[5];
    int ends[5];
    int members;
};

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    tf_exl3::fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * tf_exl3::HAD_SCALE * __half2float(__ldg(svh + col + j));
        const half bv = __ldg((bias ? bias : svh) + col + j);
        if (bias) v[j] += __half2float(bv);
    }
}

template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[8 * lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
}

template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[8 * lane_words<K2>()], int j,
                                                uint32_t (&w)[lane_words<K2>()]) {
#pragma unroll
    for (int q = 0; q < lane_words<K2>(); ++q) w[q] = raw[j * lane_words<K2>() + q];
}

__device__ __forceinline__ void pdl_wait() { qmm_frag::grid_wait(); }
__device__ __forceinline__ void pdl_trigger() { qmm_frag::grid_launch(); }

__device__ __forceinline__ void rot_in_body(const void* x, int x_dtype, const half* suh, half* xh, int kdim) {
    pdl_trigger();
    pdl_wait();
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= kdim) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * kdim + k, v);
    load4(suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    tf_exl3::fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= tf_exl3::HAD_SCALE;
    store4(xh, F16, (size_t)row * kdim + k, v);
}

// core's linear_body (exl3_linear.cu), unchanged: one program of (N / 128) * SK blocks, split-major
template <int K2, int CB, int WK>
__device__ __forceinline__ void linear_body(const LinearArgs& a, int bid) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    constexpr bool PF = K2 <= 12;
    constexpr int SR = 8 * LW;
    extern __shared__ __align__(16) float red[];
    __shared__ int last;
    const int M = a.m, K = a.k, N = a.n, SK = a.sk, NB = N >> 7;
    const int RH = imin(M, 8);
    const int nb = bid % NB, split = bid / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = a.t + nb * a.stride_nb;
    const int col0 = nb * 128;
    pdl_wait();
    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = imin(16, M - m0);
        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = static_cast<const half*>(a.xh) + (size_t)r0 * K;
        const half* x1 = static_cast<const half*>(a.xh) + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * a.stride_k;
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);
        auto load_x = [&](int kt, uint32_t (&u)[4]) {
            u[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            u[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            u[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            u[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
        };
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            uint32_t u[4];
            load_x(kt0 + i, u);
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * a.stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, w);
                else ldg_lane_words<K2>(tile + (size_t)i * a.stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                tf_exl3::mma16816(acc[j][0], u, b0);
                tf_exl3::mma16816(acc[j][1], u, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }
        if (m0 == 0) pdl_trigger();
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = imin(R - rlo, 8);
            __syncthreads();
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3])
                                : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();
            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
                    store4(a.y, a.y_dtype, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(a.z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(a.counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(a.z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, a.svh, a.bias, col0 + 4 * lane);
                    store4(a.y, a.y_dtype, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) a.counters[pass * NB + nb] = 0;
            }
        }
        __syncthreads();
    }
}
}  // namespace es_x3

extern "C" __global__ void __launch_bounds__(128) ink2_es_rot_group5(es_x3::RotGroup5 a) {
    es_x3::rot_in_body(a.x, a.x_dtype, a.suh[blockIdx.z], a.xh[blockIdx.z], a.k);
}

extern "C" __global__ void __launch_bounds__(128) ink2_es_group5_12_2_4(es_x3::Group5 g) {
    const int b = blockIdx.x;
    int i = 0;
    while (i + 1 < g.members && b >= g.ends[i]) ++i;
    es_x3::linear_body<12, 2, 4>(g.l[i], b - (i ? g.ends[i - 1] : 0));
}
// ==== end e-step
// ---- e-decode: the decode / verify windows' experts in fewer launches (TF_INKLING2_EXL3_DEC=1) --------------------
// The forms above, regrouped for chunks of up to max_rows rows (decode, verify windows, short prompt chunks), same
// arithmetic, same bits a pair (e-decode's tf-inkling2-exl3-dbench, not in this tree, compared every plan's act and y):
//   ink2_exl3_drot13  ink2_exl3_act + ink2_exl3_rot for w13's input in one launch: a program (item, 128 inputs), one
//                     warp, lane L values 4L .. 4L + 3; the activation quantization spread over the lanes of each block
//                     (NVFP4: 4 lanes a 16-value block, its max and its error tree by shuffles in the tree's own pairing;
//                     MXFP8: 8 lanes a 32-value block), under the row's amax (ink2_exl3_row_amax, NVFP4 only); it also
//                     zeroes the pair's w2 amax (accumulated by ink2_exl3_dgu).
//   ink2_exl3_dgu     ink2_exl3_gemm_g1 + ink2_exl3_gateup_epi (+ ink2_exl3_row_amax of the SwiGLU rows): the 4 K chains
//                     (gemm_body's warp_tiles_rows, G 1) summed in warp order through shared memory, then each member's
//                     128 sums rotated, scaled by svh13, the capped SwiGLU, bf16; each pair's max |act| over its rank's
//                     channels by atomicMax on the float's bits (non-negative: the int order is the float order; exact).
//   ink2_exl3_drot2   ink2_exl3_act + ink2_exl3_rot for w2's input (the pair's amax over every rank's entries).
//   ink2_exl3_ddown   ink2_exl3_gemm_g1 + ink2_exl3_down_epi (one K part): y written whole for every member (so the
//                     forward's zeroing of y is not needed).
// GRP column groups of 4 chain warps a program (NTW = 8 / GRP tiles a warp, 128 columns a program: one Hadamard block),
// PF k steps of words in flight a warp: launch shapes only, the per-column chains and their order are g1's.
namespace ink2_dec {

constexpr int ZS = 128 + 8;   // the chains' sum's row stride (floats): float2 rows 8 floats apart in banks

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }

// The ordered sum of the 4 K chains of this warp's NTW tiles (columns col0 + 16 i + ..) into Z[16][ZS]: chain 0 stores,
// chains 1, 2, 3 add in turn (gemm_body's s = red[0]; s += red[1]; ...: ((c0 + c1) + c2) + c3). Every thread calls it.
template <int NTW>
__device__ __forceinline__ void chain_sum(const float (&acc)[NTW][2][4], int chain, int col0, float* Z, int lane) {
    const int g8 = lane >> 2, t = lane & 3;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (chain == w) {
#pragma unroll
            for (int i = 0; i < NTW; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = col0 + i * 16 + h * 8 + 2 * t;
                    float2* z0 = reinterpret_cast<float2*>(Z + g8 * ZS + col);
                    float2* z1 = reinterpret_cast<float2*>(Z + (g8 + 8) * ZS + col);
                    if (w == 0) {
                        *z0 = make_float2(acc[i][h][0], acc[i][h][1]);
                        *z1 = make_float2(acc[i][h][2], acc[i][h][3]);
                    } else {
                        const float2 a0 = *z0, a1 = *z1;
                        *z0 = make_float2(a0.x + acc[i][h][0], a0.y + acc[i][h][1]);
                        *z1 = make_float2(a1.x + acc[i][h][2], a1.y + acc[i][h][3]);
                    }
                }
        }
        __syncthreads();
    }
}

// Programmatic dependent launch (TF_INKLING2_EXL3_DEC_PDL=1 launches the GEMMs with it): the rotations let their
// dependents launch at once; a GEMM reads the previous launch's output only after the wait. Both are no-ops for a
// kernel launched without the attribute (and its dependents).
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// warp_tiles_rows<CB_MUL1, K2, NT, PF, 1> (G 1: the same loads, decode, mma chain and order, so the same bits), with
// the dependent-launch wait between the first PF k steps of weights (issued first: the slots are static and the plan
// was final before the previous launch began) and the first read of X (the previous launch's output). X is not
// __restrict__ on this path: nvcc reads a const __restrict__ kernel argument as invariant (ld.global.nc) and may hoist
// those loads above the wait (measured: wrong bits with the dependent launch, tf-inkling2-exl3-dbench v2).
template <int K2, int NT, int PF>
__device__ __forceinline__ void dec_tiles(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                          const half* X, const int (&off)[2], int lane,
                                          float (&acc)[NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;
    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);
    pdl_wait();
    const bool ok0 = off[0] >= 0, ok1 = off[1] >= 0;
    const half* xr0 = X + (ok0 ? off[0] : 0) + kt0 * 16;
    const half* xr1 = X + (ok1 ? off[1] : 0) + kt0 * 16;
    uint32_t an[4];
    an[0] = load_pair(xr0, ok0);
    an[1] = load_pair(xr1, ok1);
    an[2] = load_pair(xr0 + 8, ok0);
    an[3] = load_pair(xr1 + 8, ok1);
    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t a[4];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = an[c];
                if (it + 1 < nkt) {
                    const int k = (it + 1) * 16;
                    an[0] = load_pair(xr0 + k, ok0);
                    an[1] = load_pair(xr1 + k, ok1);
                    an[2] = load_pair(xr0 + k + 8, ok0);
                    an[3] = load_pair(xr1 + k + 8, ok1);
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB_MUL1, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
    }
}

// One pass of up to 16 members (rows_sh[0..16), -1 none): this warp's chain over k tiles [chain KT / 4, ..) of its NTW
// column tiles from nt0 (dec_tiles: g1's mma chains), then the ordered sum into Z.
template <int K2, int NTW, int PF>
__device__ __forceinline__ void pass(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0, const half* X,
                                     int K, const int* rows_sh, int chain, int col0, float* Z, int lane) {
    const int g8 = lane >> 2, t = lane & 3;
    int off[2];
    const int r0 = rows_sh[g8], r1 = rows_sh[g8 + 8];
    off[0] = r0 < 0 ? -1 : r0 * K + 2 * t;
    off[1] = r1 < 0 ? -1 : r1 * K + 2 * t;
    float acc[NTW][2][4];
#pragma unroll
    for (int i = 0; i < NTW; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;
    const int per = KT / 4;
    dec_tiles<K2, NTW, PF>(T, NTILES, chain * per, per, nt0, X, off, lane, acc);
    chain_sum<NTW>(acc, chain, col0, Z, lane);
}

// dec_tiles with the operands' roles swapped: D (16 columns x 8 members) = W^T (the decoded 16 x 16 tile as the A
// operand) x X^T (8 members' inputs as the B operand), one m16n8k16 a tile and member group of 8 (MG groups a pass)
// where the X-as-A form takes two a 16-row tile (its rows 8-15 are never live at <= 8 members), and 4 accumulators a
// tile and group where it holds 8. The lane's decoded values are its A fragment as they stand: R0 = (n g, k 2t..) =
// b0[0], R1 = (n g + 8, k 2t..) = b1[0], R2 = (n g, k 2t + 8..) = b0[1], R3 = (n g + 8, k 2t + 8..) = b1[1]; the B
// fragment is member g's two input pairs. A column's chain is the same 16 products a k tile, accumulated in the same k
// order from zero; whether the instruction rounds a (column, member) sum the same with the operands swapped is checked
// per plan by tf-inkling2-exl3-dbench (act and y against the forms').
template <int K2, int NT, int PF, int MG>
__device__ __forceinline__ void dec_tiles_t(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                            const half* X, const int (&off)[MG], int lane, float (&acc)[NT][MG][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;
    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);
    pdl_wait();
    bool ok[MG];
    const half* xr[MG];
    uint32_t bn[MG][2];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
        ok[g] = off[g] >= 0;
        xr[g] = X + (ok[g] ? off[g] : 0) + kt0 * 16;
        bn[g][0] = load_pair(xr[g], ok[g]);
        bn[g][1] = load_pair(xr[g] + 8, ok[g]);
    }
    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                uint32_t b[MG][2];
#pragma unroll
                for (int g = 0; g < MG; ++g) {
                    b[g][0] = bn[g][0];
                    b[g][1] = bn[g][1];
                }
                if (it + 1 < nkt) {
                    const int k = (it + 1) * 16;
#pragma unroll
                    for (int g = 0; g < MG; ++g) {
                        bn[g][0] = load_pair(xr[g] + k, ok[g]);
                        bn[g][1] = load_pair(xr[g] + k + 8, ok[g]);
                    }
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB_MUL1, K2>(w[i], map, lane, b0, b1);
                    const uint32_t a[4] = {b0[0], b1[0], b0[1], b1[1]};
#pragma unroll
                    for (int g = 0; g < MG; ++g) mma16816(acc[i][g], a, b[g]);
                }
            }
        }
    }
}

// The ordered sum of the 4 chains for the swapped form: acc[i][g] = D of tile i, member group g (rows: columns
// col0 + 16 i + l / 4, + 8; columns: members 8 g + 2 (l % 4), + 1) into Z[member][column]; chain 0 stores, chains 1-3
// add in turn (gemm_body's s = red[0]; s += red[1]; ...).
template <int NTW, int MG>
__device__ __forceinline__ void chain_sum_t(const float (&acc)[NTW][MG][4], int chain, int col0, float* Z, int lane) {
    const int q = lane >> 2, t = lane & 3;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (chain == w) {
#pragma unroll
            for (int i = 0; i < NTW; ++i)
#pragma unroll
                for (int g = 0; g < MG; ++g) {
                    const int c = col0 + i * 16 + q;
                    float* z0 = Z + (8 * g + 2 * t) * ZS + c;
                    float* z1 = z0 + ZS;
                    if (w == 0) {
                        z0[0] = acc[i][g][0];
                        z1[0] = acc[i][g][1];
                        z0[8] = acc[i][g][2];
                        z1[8] = acc[i][g][3];
                    } else {
                        z0[0] = z0[0] + acc[i][g][0];
                        z1[0] = z1[0] + acc[i][g][1];
                        z0[8] = z0[8] + acc[i][g][2];
                        z1[8] = z1[8] + acc[i][g][3];
                    }
                }
        }
        __syncthreads();
    }
}

// pass for the swapped form: members rows_sh[0 .. 8 MG) (-1 none), member 8 g + l / 4's inputs as group g's B fragment.
template <int K2, int NTW, int PF, int MG>
__device__ __forceinline__ void pass_t(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0, const half* X, int K,
                                       const int* rows_sh, int chain, int col0, float* Z, int lane) {
    const int q = lane >> 2, t = lane & 3;
    int off[MG];
#pragma unroll
    for (int g = 0; g < MG; ++g) {
        const int r = rows_sh[8 * g + q];
        off[g] = r < 0 ? -1 : r * K + 2 * t;
    }
    float acc[NTW][MG][4];
#pragma unroll
    for (int i = 0; i < NTW; ++i)
#pragma unroll
        for (int g = 0; g < MG; ++g)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][g][c] = 0.f;
    const int per = KT / 4;
    dec_tiles_t<K2, NTW, PF, MG>(T, NTILES, chain * per, per, nt0, X, off, lane, acc);
    chain_sum_t<NTW, MG>(acc, chain, col0, Z, lane);
}

// The GEMM of a pass in the build's form: TR the swapped operands (passes of up to 8 members: one member group, so the
// kernel's registers are the one-group path's; an item of 9-16 members, a short prompt chunk's, takes two passes),
// else the X-as-A form (passes of up to 16).
template <bool TR, int K2, int NTW, int PF>
__device__ __forceinline__ void pass_any(const uint32_t* __restrict__ T, int NTILES, int KT, int nt0, const half* X,
                                         int K, const int* rows_sh, int chain, int col0, float* Z, int lane) {
    if constexpr (TR) {
        pass_t<K2, NTW, PF, 1>(T, NTILES, KT, nt0, X, K, rows_sh, chain, col0, Z, lane);
    } else {
        pass<K2, NTW, PF>(T, NTILES, KT, nt0, X, K, rows_sh, chain, col0, Z, lane);
    }
}

// w13 + its epilogue. Grid (items, 2 Il / 128), 128 GRP threads.
template <int GRP, int PF, bool TR>
__device__ __forceinline__ void dgu(const half* X, const int* __restrict__ items, const int* __restrict__ counts,
                                    const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,
                                    const int* __restrict__ slot_k, int nslots, int H, int Il, float cg, float cu,
                                    __nv_bfloat16* __restrict__ act, int act_stride, float* __restrict__ amax, int swap) {
    constexpr int NTW = 8 / GRP;
    // swap: grid (column blocks, items), so the scheduler deals each SM CTAs of several experts (2- and 3-bit mixed)
    const int it = swap ? blockIdx.y : blockIdx.x, blk = swap ? blockIdx.x : blockIdx.y;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots || cnt <= 0) return;
    const int k = __ldg(slot_k + slot);
    const Parts pt = slot_parts(H, Il, k);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot));
    const uint32_t* T = reinterpret_cast<const uint32_t*>(base);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, chain = warp & 3, gi = warp >> 2;
    const int nt0 = blk * 8 + gi * NTW;
    __shared__ int rows_sh[16];
    __shared__ __align__(16) float Z[16 * ZS];
    const half* svh = reinterpret_cast<const half*>(base + pt.svh13) + blk * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    constexpr int PM = TR ? 8 : 16;   // members a pass
    for (int m0 = 0; m0 < cnt; m0 += PM) {
        __syncthreads();   // the last pass's readers of rows_sh and Z are done
        if (threadIdx.x < 16)
            rows_sh[threadIdx.x] = (int)threadIdx.x < PM && m0 + (int)threadIdx.x < cnt ? __ldg(members + first + m0 + threadIdx.x) : -1;
        __syncthreads();
        const int rows = min(PM, cnt - m0);
        switch (2 * k) {
            case 4: pass_any<TR, 4, NTW, PF>(T, 2 * Il / 16, H / 16, nt0, X, H, rows_sh, chain, gi * NTW * 16, Z, lane); break;
            case 6: pass_any<TR, 6, NTW, PF>(T, 2 * Il / 16, H / 16, nt0, X, H, rows_sh, chain, gi * NTW * 16, Z, lane); break;
            default: __trap();
        }
        for (int r = warp; r < rows; r += 4 * GRP) {   // ink2_exl3_gateup_epi's arithmetic on the member's 128 sums
            const int p = rows_sh[r];
            const float4 z4 = *reinterpret_cast<const float4*>(Z + r * ZS + 4 * lane);
            float v[4] = {z4.x, z4.y, z4.z, z4.w};
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * s[j];
            const int c = blk * 64 + 2 * lane;
            const __nv_bfloat16 a0 = __float2bfloat16_rn(swiglu(v[0], v[1], cg, cu));
            const __nv_bfloat16 a1 = __float2bfloat16_rn(swiglu(v[2], v[3], cg, cu));
            act[(size_t)p * act_stride + c] = a0;
            act[(size_t)p * act_stride + c + 1] = a1;
            if (amax != nullptr) {
                // ink2_exl3_row_amax's max |x| over this block's 64 channels, then over the blocks by atomicMax
                float mx = fmaxf(fmaxf(0.0f, fabsf(bf(a0))), fabsf(bf(a1)));
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                if (lane == 0) atomicMax(reinterpret_cast<int*>(amax) + p, __float_as_int(mx));
            }
        }
    }
}

// w2 + its epilogue (one K part: c0 0, the whole row). Grid (items, H / 128), 128 GRP threads.
template <int GRP, int PF, bool TR>
__device__ __forceinline__ void ddown(const half* X, const int* __restrict__ items, const int* __restrict__ counts,
                                      const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,
                                      const int* __restrict__ slot_k, int nslots, int H, int Il, float* __restrict__ y,
                                      int ldy, int swap) {
    constexpr int NTW = 8 / GRP;
    const int it = swap ? blockIdx.y : blockIdx.x, blk = swap ? blockIdx.x : blockIdx.y;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots || cnt <= 0) return;
    const int k = __ldg(slot_k + slot);
    const Parts pt = slot_parts(H, Il, k);
    const uint8_t* base = reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot));
    const uint32_t* T = reinterpret_cast<const uint32_t*>(base + pt.w2);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, chain = warp & 3, gi = warp >> 2;
    const int nt0 = blk * 8 + gi * NTW;
    __shared__ int rows_sh[16];
    __shared__ __align__(16) float Z[16 * ZS];
    const half* svh = reinterpret_cast<const half*>(base + pt.svh2) + blk * 128 + 4 * lane;
    float s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] = __half2float(svh[j]);
    constexpr int PM = TR ? 8 : 16;   // members a pass
    for (int m0 = 0; m0 < cnt; m0 += PM) {
        __syncthreads();
        if (threadIdx.x < 16)
            rows_sh[threadIdx.x] = (int)threadIdx.x < PM && m0 + (int)threadIdx.x < cnt ? __ldg(members + first + m0 + threadIdx.x) : -1;
        __syncthreads();
        const int rows = min(PM, cnt - m0);
        switch (2 * k) {
            case 4: pass_any<TR, 4, NTW, PF>(T, H / 16, Il / 16, nt0, X, Il, rows_sh, chain, gi * NTW * 16, Z, lane); break;
            case 6: pass_any<TR, 6, NTW, PF>(T, H / 16, Il / 16, nt0, X, Il, rows_sh, chain, gi * NTW * 16, Z, lane); break;
            default: __trap();
        }
        for (int r = warp; r < rows; r += 4 * GRP) {   // ink2_exl3_down_epi's arithmetic (one part)
            const int p = rows_sh[r];
            const float4 z4 = *reinterpret_cast<const float4*>(Z + r * ZS + 4 * lane);
            float v[4] = {z4.x, z4.y, z4.z, z4.w};
            fwht128(v, lane);
            float o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = __fmul_rn(__fmul_rn(v[j], HAD_SCALE), s[j]);
            *reinterpret_cast<float4*>(y + (size_t)p * ldy + blk * 128 + 4 * lane) = make_float4(o[0], o[1], o[2], o[3]);
        }
    }
}

// nvfp4_block over 4 lanes (lane L: values 4 (L % 4) .. of the 16-value block L / 4): the same products, max and
// rounding points; the error tree's levels 1-2 in the lane, levels 3-4 by xor shuffles 1, 2 (the tree's pairing; an
// fp32 add is commutative, so each lane of a pair gets the pair's sum).
__device__ __forceinline__ void nvfp4_quad(const float (&x)[4], float amax, float (&out)[4]) {
    const float enc = fminf(__fdiv_rn(1536.0f, amax), 3.4028234663852886e38f);
    const float dec = __fdiv_rn(1.0f, enc);
    const float outer = __fmul_rn(amax, 1.0f / 1536.0f);
    float bm = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) bm = fmaxf(bm, fabsf(x[i]));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
    const float s6 = e4m3_rn(fminf(__fmul_rn(bm, __fmul_rn(enc, 1.0f / 6.0f)), 448.0f));
    const float s4 = e4m3_rn(fminf(__fmul_rn(bm, __fmul_rn(enc, 0.25f)), 448.0f));
    const float i6 = fminf(__fdiv_rn(1.0f, __fmul_rn(s6, dec)), 3.4028234663852886e38f);
    const float i4 = fminf(__fdiv_rn(1.0f, __fmul_rn(s4, dec)), 3.4028234663852886e38f);
    float r6[4], r4[4], e6[4], e4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r6[i] = e2m1_rn(__fmul_rn(x[i], i6));
        r4[i] = e2m1_rn(__fmul_rn(x[i], i4));
        e6[i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r6[i], s6), outer), x[i]));
        e4[i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r4[i], s4), outer), x[i]));
    }
    float l6 = __fadd_rn(__fadd_rn(e6[0], e6[1]), __fadd_rn(e6[2], e6[3]));
    float l4 = __fadd_rn(__fadd_rn(e4[0], e4[1]), __fadd_rn(e4[2], e4[3]));
    l6 = __fadd_rn(l6, __shfl_xor_sync(0xffffffffu, l6, 1));
    l4 = __fadd_rn(l4, __shfl_xor_sync(0xffffffffu, l4, 1));
    l6 = __fadd_rn(l6, __shfl_xor_sync(0xffffffffu, l6, 2));
    l4 = __fadd_rn(l4, __shfl_xor_sync(0xffffffffu, l4, 2));
    const bool four = l4 < l6;
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __fmul_rn(four ? __fmul_rn(r4[i], s4) : __fmul_rn(r6[i], s6), outer);
}

// mxfp8_block over 8 lanes (lane L: values 4 (L % 8) .. of the 32-value block L / 8): the block max by xor shuffles
// 1, 2, 4, then mxfp8_block's per-value arithmetic.
__device__ __forceinline__ void mxfp8_oct(const float (&x)[4], float (&out)[4]) {
    float bm = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) bm = fmaxf(bm, fabsf(x[i]));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 4));
    const float raw = __fmul_rn(bm, 1.0f / 448.0f);
    int e = (int)(((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23);
    e = raw <= 0x1p-127f ? 0 : min(e, 254);
    const float scale = e ? __uint_as_float((uint32_t)e << 23) : __uint_as_float(0x00400000u);
    const float inv = e == 0 ? 0x1p127f : __frcp_rn(scale);
#pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = __fmul_rn(e4m3_rn(fminf(fmaxf(__fmul_rn(x[i], inv), -448.0f), 448.0f)), scale);
}

// One member's 128-input block through ink2_exl3_act (mode 0 none, 1 NVFP4 under amax, 2 MXFP8) and ink2_exl3_rot
// (fp16, times suh, fp16, H128, / sqrt(128), fp16): lane L's 4 inputs f, its 4 outputs into o.
__device__ __forceinline__ void act_rot(const float (&f)[4], int mode, float amax, const float (&su)[4], half* o, int lane) {
    float q[4];
    if (mode == 1) nvfp4_quad(f, amax, q);
    else if (mode == 2) mxfp8_oct(f, q);
    else {
#pragma unroll
        for (int j = 0; j < 4; ++j) q[j] = f[j];
    }
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __half2float(__float2half_rn(__half2float(__float2half_rn(q[j])) * su[j]));
    fwht128(v, lane);
    __align__(8) half h[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) h[j] = __float2half_rn(v[j] * HAD_SCALE);
    *reinterpret_cast<uint2*>(o) = *reinterpret_cast<const uint2*>(h);
}

}  // namespace ink2_dec

// w13's input: every member pair p of each item (row p / row_slots of x, bf16, stride xs): the activation (mode; NVFP4
// under row_amax[row], ink2_exl3_row_amax's) then the rotation by the slot's suh13, into out [p][H] fp16; with w2_amax,
// the pair's entry zeroed (blk 0) for ink2_exl3_dgu's maxima. Grid (items, H / 128), 32 threads.
extern "C" __global__ void __launch_bounds__(32) ink2_exl3_drot13(
    const __nv_bfloat16* __restrict__ x, int xs, int row_slots, int mode, const float* __restrict__ row_amax,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k, int nslots, int H, int Il,
    half* __restrict__ out, float* __restrict__ w2_amax) {
    ink2_dec::pdl_trigger();   // the GEMM after it may launch now (it waits before reading out)
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots) return;
    const Parts pt = slot_parts(H, Il, __ldg(slot_k + slot));
    const half* suh = reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) + pt.suh13) +
                      blk * 128 + 4 * lane;
    float su[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) su[j] = __half2float(suh[j]);
    for (int m = 0; m < cnt; ++m) {
        const int p = __ldg(members + first + m);
        const int row = p / row_slots;
        const uint2 raw = *reinterpret_cast<const uint2*>(x + (size_t)row * xs + blk * 128 + 4 * lane);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
        const float f[4] = {ink2_dec::bf(b[0]), ink2_dec::bf(b[1]), ink2_dec::bf(b[2]), ink2_dec::bf(b[3])};
        ink2_dec::act_rot(f, mode, mode == 1 ? __ldg(row_amax + row) : 0.0f, su, out + (size_t)p * H + blk * 128 + 4 * lane, lane);
        if (w2_amax != nullptr && blk == 0 && lane == 0) w2_amax[p] = 0.0f;
    }
}

// w2's input: every member pair p's SwiGLU row (act [p][Il] bf16, stride as): the activation (NVFP4: under the pair's
// amax, the max of amax_all[r][p] over the world's ranks) then the rotation by suh2, into out [p][Il] fp16. Grid
// (items, Il / 128), 32 threads.
extern "C" __global__ void __launch_bounds__(32) ink2_exl3_drot2(
    const __nv_bfloat16* __restrict__ act, int as, int mode, const float* __restrict__ amax_all, int world,
    int amax_stride, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    const unsigned long long* __restrict__ slot_ptr, const int* __restrict__ slot_k, int nslots, int H, int Il,
    half* __restrict__ out) {
    ink2_dec::pdl_trigger();
    const int it = blockIdx.x, blk = blockIdx.y, lane = threadIdx.x;
    if (it >= __ldg(counts)) return;
    const int slot = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (slot < 0 || slot >= nslots) return;
    const Parts pt = slot_parts(H, Il, __ldg(slot_k + slot));
    const half* suh = reinterpret_cast<const half*>(reinterpret_cast<const uint8_t*>(__ldg(slot_ptr + slot)) + pt.suh2) +
                      blk * 128 + 4 * lane;
    float su[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) su[j] = __half2float(suh[j]);
    for (int m = 0; m < cnt; ++m) {
        const int p = __ldg(members + first + m);
        float amax = 0.0f;
        if (mode == 1)
            for (int r = 0; r < world; ++r) amax = fmaxf(amax, __ldg(amax_all + (size_t)r * amax_stride + p));
        const uint2 raw = *reinterpret_cast<const uint2*>(act + (size_t)p * as + blk * 128 + 4 * lane);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
        const float f[4] = {ink2_dec::bf(b[0]), ink2_dec::bf(b[1]), ink2_dec::bf(b[2]), ink2_dec::bf(b[3])};
        ink2_dec::act_rot(f, mode, amax, su, out + (size_t)p * Il + blk * 128 + 4 * lane, lane);
    }
}

// ink2_exl3_row_amax for rows of bf16 with K % 8 == 0 and 16-byte aligned rows (the w13 input): each row's max |x| by
// 16-byte loads and warp shuffles, the max of the same values (exact in any order; NaN skipped as fmaxf does). One
// block of 256 a row.
extern "C" __global__ void __launch_bounds__(256) ink2_exl3_drow_amax(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                                      int K, float* __restrict__ amax) {
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)blockIdx.x * x_stride);
    float m = 0.0f;
    for (int i = threadIdx.x; i < K / 8; i += 256) {
        const uint4 v = xr[i];
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) m = fmaxf(m, fabsf(__bfloat162float(b[j])));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < 8 ? red[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (threadIdx.x == 0) amax[blockIdx.x] = m;
    }
}

// The GEMM entries by build (GRP column groups of 4 chain warps, PF k steps in flight): grid (items, columns / 128).
#define INK2_EXL3_DEC(SUFFIX, GRP, PF, MB, TR)                                                                          \
    extern "C" __global__ void __launch_bounds__(128 * GRP, MB) ink2_exl3_dgu##SUFFIX(                               \
        const half* X, const int* __restrict__ items, const int* __restrict__ counts,                   \
        const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,                            \
        const int* __restrict__ slot_k, int nslots, int H, int Il, float cg, float cu,                               \
        __nv_bfloat16* __restrict__ act, int act_stride, float* __restrict__ amax, int swap) {                       \
        ink2_dec::dgu<GRP, PF, TR>(X, items, counts, members, slot_ptr, slot_k, nslots, H, Il, cg, cu, act,          \
                                   act_stride,                                                                       \
                               amax, swap);                                                                          \
    }                                                                                                                \
    extern "C" __global__ void __launch_bounds__(128 * GRP, MB) ink2_exl3_ddown##SUFFIX(                             \
        const half* X, const int* __restrict__ items, const int* __restrict__ counts,                   \
        const int* __restrict__ members, const unsigned long long* __restrict__ slot_ptr,                            \
        const int* __restrict__ slot_k, int nslots, int H, int Il, float* __restrict__ y, int ldy, int swap) {       \
        ink2_dec::ddown<GRP, PF, TR>(X, items, counts, members, slot_ptr, slot_k, nslots, H, Il, y, ldy, swap);      \
    }
INK2_EXL3_DEC(_t1p2, 1, 2, 3, true)   // the swapped operands (dec_tiles_t): one mma a tile and 8 members (the only build here:
                                      // e-decode's fastest bit-identical one; the other builds are in out/prefill/p5/e-decode.patch)
// ==== end e-decode ====
