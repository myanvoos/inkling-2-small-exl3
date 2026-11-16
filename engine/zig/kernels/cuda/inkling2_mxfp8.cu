// Inkling-2's MXFP8 routed experts (layers 40-47): the device code of the inkling-tensorfold overlay's
// kernels/mxfp8_grouped.cu (verified bit-exact on real layer-40 bytes and TP2 slices, docs/mxfp8-kernel.md there),
// unchanged; its torch host launchers and PYBIND module dropped, its header's constants inlined, and C entries added
// for the engine: the softcapped SwiGLU gate-up to bf16 (inputs rounded to bf16 first, HF's points) and the fp32 down,
// at 16, 32 and 64 rows a pass (RT 1, 2, 4: a pair's bits do not depend on it), and the decode check.
// Grouped MXFP8 routed experts (W8A16): the inkling2 family's Experts8 lane for layers 40-47.
//
// Structure from TensorFold 92aa710b533d src/tensorfold/cuda/nvfp4/experts.cu (Experts4): one warp a (plan item,
// 32-column block) unit, an item's pairs 16 at a time as the rows of bf16 m16n8k16 MMAs, two-stage register
// pipeline of non-caching 16-byte weight loads, the tensorfold.cuda.experts plan (items / counts / members).
// Arithmetic from the MXFP8 decode lane (qmmf.cu, vendored as K3's dense_mx8f.cu): an e4m3 byte pair becomes the
// exact bf16 pair by moving its fields into bf16's and multiplying by 2^120 (``fp8pair``); a 32-input group's
// product is two k16 MMAs from zero, then acc = fma(product, 2^(e - 127), acc) in group order. New here: the block
// layout below, UE8M0 byte 0 decoded exactly (2^-127, not 0), and inkling2's softcapped SwiGLU epilogue.
//
// Block (one (expert, matrix), 32 output columns x 32 inputs), 1056 bytes:
//   codes  [lane L = 4 * gq + t][j][h][4]: W[8j + gq][16h + 4t + i], i = 0..3 (j: n8 tile, h: k16 half)
//   scales [t][j][c]: exponent byte of column 8j + 2t + c (the accumulator's columns)
// A lane's activations are inputs 16h + 4t .. 16h + 4t + 3 of its rows (one 8-byte load); the MMA's k slots
// (2t, 2t + 1, 2t + 8, 2t + 9) carry them, and the weight bytes are stored in the same order.

#include <cuda_bf16.h>
#include <stdint.h>

namespace mxfp8g {

constexpr int COLS = 32;                 // output columns a block (a warp's unit: four n8 MMA tiles)
constexpr int GS = 32;                   // inputs an MX block (one UE8M0 exponent)
constexpr int CODE_BYTES = COLS * GS;    // 1024: lane L's 32 code bytes at [32L, 32L + 32)
constexpr int SCALE_BYTES = COLS;        // 32: [t][j][c] = exponent of column 8j + 2t + c
constexpr int BLOCK_BYTES = CODE_BYTES + SCALE_BYTES;   // 1056
constexpr int BLOCK_I32 = BLOCK_BYTES / 4;               // 264

// Epilogues. Pairs are rows of ``out``; slots > 0 means X holds tokens (pair p reads row p / slots).
enum Epi : int {
  RAW_F32 = 0,      // M = 1: out [pairs, N] fp32
  RAW_BF16 = 1,     // M = 1: out [pairs, N] bf16
  ACT_F32 = 2,      // M = 2 (gate, up): out [pairs, N] fp32 = cg*tanh(g/cg) * sigmoid(g) * cu*tanh(u/cu)
  ACT_BF16 = 3,     // M = 2: the same activation rounded once to bf16 (the down projection's input)
  GATE_UP_F32 = 4,  // M = 2: out [pairs, 2N] fp32 = [gate | up] (HF's gate_up before chunk)
};

}  // namespace mxfp8g

namespace {

using namespace mxfp8g;

constexpr int NTW = 4;                         // n8 tiles a warp
constexpr int BLOCK_U4 = BLOCK_BYTES / 16;     // 66
constexpr int CODE_U4 = CODE_BYTES / 16;       // 64
static_assert(NTW * 8 == COLS, "a warp covers a block's columns");

__device__ __forceinline__ uint4 ld_nc4(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_nc2(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// bf16x2 of the e4m3 bytes at bits [0, 8) (low half) and [8, 16) (high half): exponent and mantissa moved into
// bf16's fields (value x 2^-120, subnormals included), times 2^120 -- an exact product. (mx8f.cu's fp8pair.)
__device__ __forceinline__ uint32_t fp8pair(uint32_t w) {
  const uint32_t x = (w & 0xFFu) | ((w & 0xFF00u) << 8);
  const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
}

// UE8M0 byte -> 2^(b - 127) in fp32, exactly: byte 0 is the subnormal 2^-127 (not 0). Byte 255 (NaN) is refused
// before packing, so it never reaches here.
__device__ __forceinline__ float pow2e(uint32_t b) { return __uint_as_float(b ? (b << 23) : 0x00400000u); }

__device__ __forceinline__ float lo_f(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_f(uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// inkling2's SwiGLU (modeling_inkling2.py; reference/torch_ref/model.py): cg*tanh(g/cg) * sigmoid(g) * cu*tanh(u/cu),
// in fp32 on the fp32 accumulators (``round_in``: on their bf16 roundings, as a bf16 gate_up would give).
__device__ __forceinline__ float act(float g, float u, float cg, float cu, bool round_in) {
  if (round_in) {
    g = bf(g);
    u = bf(u);
  }
  const float sg = 1.0f / (1.0f + expf(-g));
  return cg * tanhf(g / cg) * sg * (cu * tanhf(u / cu));
}

// RT row tiles (16 pairs each) share every weight load and every e4m3 -> bf16 conversion; a pair's arithmetic is the
// same for any RT (its own MMA row, the same k order), so RT changes speed, never bits.
template <int M, int RT>
struct Stage {
  uint4 w[M][2];           // the lane's 32 code bytes: words (j, h) = 2j + h
  uint2 s[M];              // its quad's 8 exponent bytes: [j][c]
  uint2 xa[RT][2], xb[RT][2];  // per row tile, rows gq and gq + 8: inputs 16h + 4t .. 16h + 4t + 3
};

struct Rows {              // a row tile's two rows of this lane: pairs pr0 (row gq) and pr1 (row gq + 8)
  const __nv_bfloat16* x0;
  const __nv_bfloat16* x1;
  int pr0, pr1;
  bool v0, v1;
};

template <int M, int RT>
__device__ __forceinline__ void load_stage(Stage<M, RT>& st, const uint4* blk, int g, int lane, int t,
                                           const Rows (&rw)[RT]) {
  const uint4* b = blk + (size_t)g * (M * BLOCK_U4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m][0] = ld_nc4(b + m * BLOCK_U4 + 2 * lane);
    st.w[m][1] = ld_nc4(b + m * BLOCK_U4 + 2 * lane + 1);
    st.s[m] = ld_nc2(reinterpret_cast<const uint2*>(b + m * BLOCK_U4 + CODE_U4) + t);
  }
  const uint2 zero = make_uint2(0u, 0u);
  const int k0 = g * GS + 4 * t;
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      st.xa[r][h] = rw[r].v0 ? __ldg(reinterpret_cast<const uint2*>(rw[r].x0 + k0 + 16 * h)) : zero;
      st.xb[r][h] = rw[r].v1 ? __ldg(reinterpret_cast<const uint2*>(rw[r].x1 + k0 + 16 * h)) : zero;
    }
}

template <int M, int RT>
__device__ __forceinline__ void compute_stage(float (&acc)[M][RT][NTW][4], const Stage<M, RT>& st) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      float p[RT][4];                          // this group's 32 products a row, from zero
#pragma unroll
      for (int r = 0; r < RT; ++r) p[r][0] = p[r][1] = p[r][2] = p[r][3] = 0.f;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t word = comp(st.w[m][j >> 1], ((j & 1) << 1) | h);
        const uint32_t b0 = fp8pair(word & 0xFFFFu), b1 = fp8pair(word >> 16);
#pragma unroll
        for (int r = 0; r < RT; ++r) mma(p[r], st.xa[r][h].x, st.xb[r][h].x, st.xa[r][h].y, st.xb[r][h].y, b0, b1);
      }
      const uint32_t sw = ((j >> 1) ? st.s[m].y : st.s[m].x) >> (16 * (j & 1));
      const float s0 = pow2e(sw & 0xFFu), s1 = pow2e((sw >> 8) & 0xFFu);
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        float(&a)[4] = acc[m][r][j];
        a[0] = fmaf(p[r][0], s0, a[0]);
        a[1] = fmaf(p[r][1], s1, a[1]);
        a[2] = fmaf(p[r][2], s0, a[2]);
        a[3] = fmaf(p[r][3], s1, a[3]);
      }
    }
}

template <int M, int RT>
__device__ __forceinline__ void k_loop(float (&acc)[M][RT][NTW][4], const uint4* blk, int KG, int lane, int t,
                                       const Rows (&rw)[RT]) {
  constexpr int D = 2;
  Stage<M, RT> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M, RT>(st[d], blk, d, lane, t, rw);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M, RT>(acc, st[d]);
        if (g + D < KG) load_stage<M, RT>(st[d], blk, g + D, lane, t, rw);
      }
    }
  }
}

template <int EPI, int M, int RT>
__device__ __forceinline__ void epilogue(const float (&acc)[M][RT][NTW][4], int r, void* out, int N, int col0,
                                         const Rows& rw, float cg, float cu, bool round_in) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? rw.v1 : rw.v0)) continue;
    const size_t pr = (size_t)(h ? rw.pr1 : rw.pr0);
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][r][j][2 * h], a1 = acc[0][r][j][2 * h + 1];
      if constexpr (EPI == RAW_F32) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + pr * N + col) = make_float2(a0, a1);
      } else if constexpr (EPI == RAW_BF16) {
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + pr * N + col) =
            __floats2bfloat162_rn(a0, a1);
      } else if constexpr (EPI == GATE_UP_F32) {
        float* row = reinterpret_cast<float*>(out) + pr * (2 * (size_t)N);
        *reinterpret_cast<float2*>(row + col) = make_float2(a0, a1);
        *reinterpret_cast<float2*>(row + N + col) =
            make_float2(acc[M - 1][r][j][2 * h], acc[M - 1][r][j][2 * h + 1]);
      } else {
        const float o0 = act(a0, acc[M - 1][r][j][2 * h], cg, cu, round_in);
        const float o1 = act(a1, acc[M - 1][r][j][2 * h + 1], cg, cu, round_in);
        if constexpr (EPI == ACT_F32)
          *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + pr * N + col) = make_float2(o0, o1);
        else
          *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + pr * N + col) =
              __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

template <int M, int EPI, int WARPS, int RT>
__device__ __forceinline__ void
    mxfp8_expert_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        int E, int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,
                        const int* __restrict__ members, void* __restrict__ out, int N, float cg, float cu,
                        bool round_in, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip || e < 0 || e >= E) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK_U4);
    for (int r0 = 0; r0 < cnt; r0 += 16 * RT) {
      Rows rw[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const int a = r0 + 16 * r + gq, b = a + 8;
        rw[r].v0 = a < cnt;
        rw[r].v1 = b < cnt;
        rw[r].pr0 = rw[r].v0 ? __ldg(members + first + a) : 0;
        rw[r].pr1 = rw[r].v1 ? __ldg(members + first + b) : 0;
        rw[r].x0 = X + (size_t)(slots ? rw[r].pr0 / slots : rw[r].pr0) * x_stride;
        rw[r].x1 = X + (size_t)(slots ? rw[r].pr1 / slots : rw[r].pr1) * x_stride;
      }
      float acc[M][RT][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
      k_loop<M, RT>(acc, blk, KG, lane, t, rw);
#pragma unroll
      for (int r = 0; r < RT; ++r) epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, rw[r], cg, cu, round_in);
    }
  }
}

// One warp a block: the lane decodes its own 32 code bytes with the GEMM's fp8pair and the exponent of each byte's
// column, so the decoded matrix is exactly what the MMAs are fed (times the scale the accumulator applies).
__device__ __forceinline__ void mxfp8_decode_body(const uint4* __restrict__ W, int E, int NB, int KG, int M, float* __restrict__ out) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const long long blocks = (long long)E * NB * KG * M;
  const long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (b >= blocks) return;
  long long r = b;
  const int m = (int)(r % M);
  r /= M;
  const int g = (int)(r % KG);
  r /= KG;
  const int cb = (int)(r % NB);
  const int e = (int)(r / NB);
  const int N = NB * COLS, K = KG * GS;
  const uint4* base = W + b * BLOCK_U4;
  const uint4 w[2] = {ld_nc4(base + 2 * lane), ld_nc4(base + 2 * lane + 1)};
  const uint2 sc = ld_nc2(reinterpret_cast<const uint2*>(base + CODE_U4) + (gq >> 1));   // columns 8j + gq
#pragma unroll
  for (int j = 0; j < NTW; ++j) {
    const uint32_t sw = ((j >> 1) ? sc.y : sc.x) >> (16 * (j & 1));
    const float s = pow2e((sw >> (8 * (gq & 1))) & 0xFFu);
    const size_t row = ((size_t)(e * M + m) * N + cb * COLS + 8 * j + gq) * K;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t word = comp(w[j >> 1], ((j & 1) << 1) | h);
      const uint32_t b0 = fp8pair(word & 0xFFFFu), b1 = fp8pair(word >> 16);
      float4 v = make_float4(lo_f(b0) * s, hi_f(b0) * s, lo_f(b1) * s, hi_f(b1) * s);
      *reinterpret_cast<float4*>(out + row + g * GS + 16 * h + 4 * t) = v;
    }
  }
}
}  // namespace

#define INK2_MX8(NAME, M, EPI, RT)                                                                                     \
  extern "C" __global__ void __launch_bounds__(4 * 32) NAME(                                                         \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int E, int KG, int NB, \
      const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,                \
      void* __restrict__ out, int N, float cg, float cu, int round_in, int skip) {                                    \
    mxfp8_expert_body<M, EPI, 4, RT>(X, x_stride, slots, W, E, KG, NB, items, counts, members, out, N, cg, cu,       \
                                     round_in != 0, skip);                                                           \
  }
INK2_MX8(ink2_mx8_gate_up_r1, 2, mxfp8g::ACT_BF16, 1)
INK2_MX8(ink2_mx8_gate_up_r2, 2, mxfp8g::ACT_BF16, 2)
INK2_MX8(ink2_mx8_gate_up_r4, 2, mxfp8g::ACT_BF16, 4)
INK2_MX8(ink2_mx8_down_f32_r1, 1, mxfp8g::RAW_F32, 1)
INK2_MX8(ink2_mx8_down_f32_r2, 1, mxfp8g::RAW_F32, 2)
INK2_MX8(ink2_mx8_down_f32_r4, 1, mxfp8g::RAW_F32, 4)

extern "C" __global__ void ink2_mx8_decode(const uint4* __restrict__ W, int E, int NB, int KG, int M, float* __restrict__ out) {
  mxfp8_decode_body(W, E, NB, KG, M, out);
}

// ==== p3-expert: wide prompt chunks' MXFP8 form over plan items of up to 128 pairs (TF_INKLING2_PREFILL_EXPERTS) ====
// The r1/r2/r4 forms stream a warp's own weights through registers, once per 16 RT pairs of an item. Here a CTA of
// 2 WN warps stages each k32 group pair of the item's rows and of its weight blocks in shared memory (cp.async, ST
// stages), and the warps share them: an item of more than 64 pairs as two warp rows of WN warps (128 rows x WN
// 32-column blocks, the weights read once a 128 pairs), one of at most 64 as one row of 2 WN warps (64 rows x 2 WN
// blocks). Each warp's sums are compute_stage's for RT = 4, instruction for instruction: per k32 group two k16 MMAs
// from zero, then acc = fma(product, 2^(e - 127), acc) in group order, then the epilogue; a pair's bits do not
// depend on the form (a row past the item is never read into a valid row's MMA row). Served at WN 2 (4 warps, two CTAs
// an SM); WN 4 (8 warps, 3 stages) measured the same at 2048-row chunks (p3-expert's report).
#include "common/cp_async.cuh"

namespace {

template <int M, int WN>
struct WideMx {
  static constexpr int SG = 2, XC = 8, WB = M * BLOCK_U4;
  static constexpr int SU_A = 128 * XC + WN * SG * WB, SU_B = 64 * XC + 2 * WN * SG * WB;
  static constexpr int SU = SU_A > SU_B ? SU_A : SU_B;
  // 8 warps (WN 4): one CTA an SM either way (registers), so 3 stages when they fit; 4 warps (WN 2): 2 stages, two CTAs
  static constexpr int ST = WN == 2 ? 2 : SU * 16 * 3 <= 99 * 1024 ? 3 : 2;
  static constexpr int SMEM = ST * SU * 16;
};

// row r's 16-byte chunk c of a stage's inputs: rows r .. r + 3 put a 16-input block's lanes on distinct banks
__device__ __forceinline__ int mx_chunk(int r, int c) { return r * 8 + (c ^ ((r & 3) << 1)); }

template <int M, int EPI, int WM, int WNL, int NT, int SU, int ST>
__device__ __forceinline__ void mx_wide_tiles(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                              const uint4* __restrict__ W, int KG, int NB,
                                              const int* __restrict__ members, void* __restrict__ out, int N, float cg,
                                              float cu, bool round_in, int cbt, int e, int first, int cnt, int tb) {
  constexpr int RT = 4, THREADS = WM * WNL * 32, BM = 16 * RT * WM, XC = 8, SG = 2, WB = M * BLOCK_U4;
  constexpr int XU = BM * XC, XPT = (XU + THREADS - 1) / THREADS;
  static_assert(XU + WNL * SG * WB <= SU, "a stage of this layout fits the shared stage");
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WNL, wn = warp - wm * WNL;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WNL + wn, cbs = min(WNL, NB - cbt * WNL);
  int xsrc[XPT];  // the input chunk's element offset in X (-1: a row past the item)
  int xdst[XPT];
#pragma unroll
  for (int i = 0; i < XPT; ++i) {
    const int q = tid + i * THREADS, r = q / XC, c = q - r * XC;
    xdst[i] = mx_chunk(r, c);
    xsrc[i] = -1;
    if (q < XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = (slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WNL) * (size_t)KG * WB;
  auto stage = [&](int s, int g0) {  // k32 groups g0 .. g0 + ng - 1 (ng < SG only at an odd end)
    const int ng = min(SG, KG - g0);
    uint4* xs = sm + s * SU;
#pragma unroll
    for (int i = 0; i < XPT; ++i)
      if (xsrc[i] >= 0 && ((tid + i * THREADS) % XC) < ng * 4) tf_common::cp16(xs + xdst[i], X + xsrc[i] + g0 * GS);
    uint4* ws = xs + XU;
    for (int q = tid; q < cbs * SG * WB; q += THREADS) {
      const int j = q / (SG * WB), o = q - j * SG * WB;
      if (o < ng * WB) tf_common::cp16(ws + q, wsrc + ((size_t)j * KG + g0) * WB + o);
    }
  };
  constexpr int TA = NT > 0 ? NT : 1;  // array extent
  const int base = 16 * tb;  // the warp row's first row (tb its first 16-row tile)
  const bool active = NT > 0 && cb < NB;
  float acc[M][TA][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < NT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = (KG + SG - 1) / SG;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < steps) stage(s, s * SG);
    tf_common::cp_commit();
  }
  for (int k = 0; k < steps; ++k) {
    tf_common::cp_wait<ST - 2>();
    __syncthreads();
    if (k + ST - 1 < steps) stage((k + ST - 1) % ST, (k + ST - 1) * SG);
    tf_common::cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % ST) * SU;
#pragma unroll
    for (int gi = 0; gi < SG; ++gi) {
      if (k * SG + gi >= KG) break;
      const uint4* ws = xs + XU + (wn * SG + gi) * WB;
      uint4 w[M][2];  // the lane's 32 code bytes: words (j, h) = 2j + h
      uint2 sv[M];    // its quad's 8 exponent bytes: [j][c]
#pragma unroll
      for (int m = 0; m < M; ++m) {
        w[m][0] = ws[m * BLOCK_U4 + 2 * lane];
        w[m][1] = ws[m * BLOCK_U4 + 2 * lane + 1];
        sv[m] = reinterpret_cast<const uint2*>(ws + m * BLOCK_U4 + CODE_U4)[t];
      }
      uint2 xa[TA][2], xb[TA][2];  // per row tile, rows gq and gq + 8: inputs 16h + 4t .. 16h + 4t + 3
#pragma unroll
      for (int r = 0; r < NT; ++r)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int u = gi * 8 + h * 4 + t, r0 = base + 16 * r + gq;
          xa[r][h] = reinterpret_cast<const uint2*>(xs + mx_chunk(r0, u >> 1))[u & 1];
          xb[r][h] = reinterpret_cast<const uint2*>(xs + mx_chunk(r0 + 8, u >> 1))[u & 1];
        }
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          float p[TA][4];  // this group's 32 products a row, from zero
#pragma unroll
          for (int r = 0; r < NT; ++r) p[r][0] = p[r][1] = p[r][2] = p[r][3] = 0.f;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t word = comp(w[m][j >> 1], ((j & 1) << 1) | h);
            const uint32_t b0 = fp8pair(word & 0xFFFFu), b1 = fp8pair(word >> 16);
#pragma unroll
            for (int r = 0; r < NT; ++r) mma(p[r], xa[r][h].x, xb[r][h].x, xa[r][h].y, xb[r][h].y, b0, b1);
          }
          const uint32_t sw = ((j >> 1) ? sv[m].y : sv[m].x) >> (16 * (j & 1));
          const float s0 = pow2e(sw & 0xFFu), s1 = pow2e((sw >> 8) & 0xFFu);
#pragma unroll
          for (int r = 0; r < NT; ++r) {
            float(&a)[4] = acc[m][r][j];
            a[0] = fmaf(p[r][0], s0, a[0]);
            a[1] = fmaf(p[r][1], s1, a[1]);
            a[2] = fmaf(p[r][2], s0, a[2]);
            a[3] = fmaf(p[r][3], s1, a[3]);
          }
        }
    }
  }
  if (!active) return;
  if constexpr (NT > 0) {
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      Rows rw;
      const int m0 = base + 16 * r + gq, m1 = m0 + 8;
      rw.v0 = m0 < cnt;
      rw.v1 = m1 < cnt;
      rw.pr0 = rw.v0 ? __ldg(members + first + m0) : 0;
      rw.pr1 = rw.v1 ? __ldg(members + first + m1) : 0;
      rw.x0 = rw.x1 = X;  // not read by the epilogue
      epilogue<EPI, M, TA>(acc, r, out, N, cb * COLS + 2 * t, rw, cg, cu, round_in);
    }
  }
}

// One CTA a (plan item, WN-block group): grid items x ceil(NB / WN). An item of at most 64 pairs takes its first
// ceil(NB / 2 WN) CTAs, 2 WN blocks each; the rest leave at once. Warps of one CTA run different row-tile counts
// (mx_wide_tiles<NT>), each the same `steps` barriers, as inkling2_nvfp4.cu's prompt_body does.
template <int M, int EPI, int WN>
__device__ __forceinline__ void mx_wide_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                             const uint4* __restrict__ W, int E, int KG, int NB,
                                             const int* __restrict__ items, const int* __restrict__ counts,
                                             const int* __restrict__ members, void* __restrict__ out, int N, float cg,
                                             float cu, bool round_in, int skip) {
  using P = WideMx<M, WN>;
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip || e < 0 || e >= E) return;
  const int warp = threadIdx.x >> 5;
  if (cnt > 64) {
    // the item's 16-row tiles split evenly between the warp rows (5 .. 8 tiles: row 0 takes ceil(tiles / 2))
    const int tiles = (cnt + 15) >> 4, t0 = (tiles + 1) >> 1, wm = warp / WN;
    const int nt = wm ? tiles - t0 : t0, tb = wm ? t0 : 0;
#define TF_WIDE(NT)                                                                                                 \
  mx_wide_tiles<M, EPI, 2, WN, NT, P::SU, P::ST>(X, x_stride, slots, W, KG, NB, members, out, N, cg, cu, round_in, \
                                                 cbt, e, first, cnt, tb)
    switch (nt) {
      case 4: TF_WIDE(4); break;
      case 3: TF_WIDE(3); break;
      case 2: TF_WIDE(2); break;
      case 1: TF_WIDE(1); break;
      default: TF_WIDE(0);
    }
#undef TF_WIDE
  } else {
    if (cbt * 2 * WN >= NB) return;  // the whole CTA: no barrier is entered
    const int nt = (cnt + 15) >> 4, tb = 0;
#define TF_WIDE(NT)                                                                                                     \
  mx_wide_tiles<M, EPI, 1, 2 * WN, NT, P::SU, P::ST>(X, x_stride, slots, W, KG, NB, members, out, N, cg, cu, round_in, \
                                                     cbt, e, first, cnt, tb)
    switch (nt) {
      case 4: TF_WIDE(4); break;
      case 3: TF_WIDE(3); break;
      case 2: TF_WIDE(2); break;
      case 1: TF_WIDE(1); break;
      default: TF_WIDE(0);
    }
#undef TF_WIDE
  }
}

}  // namespace

// The wide entries: 2 WN warps a CTA, dynamic shared memory WideMx<M, WN>::SMEM (kern.zig wideSmem).
#define INK2_MX8_WIDE(NAME, M, EPI, WN)                                                                                 \
  extern "C" __global__ void __launch_bounds__(2 * WN * 32) NAME(                                                     \
      const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int E, int KG, int NB, \
      const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,                \
      void* __restrict__ out, int N, float cg, float cu, int round_in, int skip) {                                    \
    mx_wide_body<M, EPI, WN>(X, x_stride, slots, W, E, KG, NB, items, counts, members, out, N, cg, cu, round_in != 0, \
                            skip);                                                                                   \
  }
INK2_MX8_WIDE(ink2_mx8_wide_gate_up, 2, mxfp8g::ACT_BF16, 2)
INK2_MX8_WIDE(ink2_mx8_wide_down_f32, 1, mxfp8g::RAW_F32, 2)
static_assert(WideMx<2, 2>::SMEM == 2 * 25088, "kern.zig wideSmem(gate-up)");
static_assert(WideMx<1, 2>::SMEM == 2 * 20608, "kern.zig wideSmem(down)");
// ==== end p3-expert ====

// ---------------------------------------------------------------------------- p3-comm: the down form in column parts
// ink2_mx8_down_f32_r4 over the column blocks [cb0, cb0 + nbp) only (units = items x nbp), written as rows of
// N = nbp * 32 floats (pair p's part at out + p N): each unit is mxfp8_expert_body's (the same block's weights, the
// same k_loop and epilogue), so the parts together are that kernel's bits, each weight read once. A wide chunk's TP
// exchange sends one part's rows while the next part computes (forward.moeOverlap).
namespace {
__device__ __forceinline__ void mxfp8_down_cols_body(const __nv_bfloat16* __restrict__ X, int x_stride, int slots,
                                                     const uint4* __restrict__ W, int E, int KG, int NB,
                                                     const int* __restrict__ items, const int* __restrict__ counts,
                                                     const int* __restrict__ members, void* __restrict__ out, int N,
                                                     int skip, int cb0, int nbp) {
  constexpr int M = 1, RT = 4, WARPS = 4;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * nbp;
  // the epilogue writes column cb * 32 + c at out + p N + col: the part's rows start at its first column
  void* o = reinterpret_cast<float*>(out) - (ptrdiff_t)cb0 * COLS;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / nbp, cb = cb0 + (unit - it * nbp);
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip || e < 0 || e >= E) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK_U4);
    for (int r0 = 0; r0 < cnt; r0 += 16 * RT) {
      Rows rw[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const int a = r0 + 16 * r + gq, b = a + 8;
        rw[r].v0 = a < cnt;
        rw[r].v1 = b < cnt;
        rw[r].pr0 = rw[r].v0 ? __ldg(members + first + a) : 0;
        rw[r].pr1 = rw[r].v1 ? __ldg(members + first + b) : 0;
        rw[r].x0 = X + (size_t)(slots ? rw[r].pr0 / slots : rw[r].pr0) * x_stride;
        rw[r].x1 = X + (size_t)(slots ? rw[r].pr1 / slots : rw[r].pr1) * x_stride;
      }
      float acc[M][RT][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
      k_loop<M, RT>(acc, blk, KG, lane, t, rw);
#pragma unroll
      for (int r = 0; r < RT; ++r) epilogue<RAW_F32, M, RT>(acc, r, o, N, cb * COLS + 2 * t, rw[r], 0.f, 0.f, false);
    }
  }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(4 * 32) ink2_mx8_down_f32_r4_cols(
    const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int E, int KG, int NB,
    const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
    void* __restrict__ out, int N, float cg, float cu, int round_in, int skip, int cb0, int nbp) {
  (void)cg;
  (void)cu;
  (void)round_in;  // the raw fp32 epilogue reads none of them (ink2_mx8_down_f32_r4's arguments, kept in its order)
  mxfp8_down_cols_body(X, x_stride, slots, W, E, KG, NB, items, counts, members, out, N, skip, cb0, nbp);
}

// ================================== p4-mx: W8A8, the checkpoint's declared MXFP8 numerics (TF_INKLING2_MX_ACT=1) ====
// hf_quant_config.json group 3 declares compute mxfp8 with required_activation_quantization {block_size 32,
// mxfp8_e4m3, rounding nearest, ue8m0}: both expert products take MXFP8 inputs. ink2_mx8_quant makes a token row's
// codes and exponents with the EXL3 lane's arithmetic (inkling2_exl3.cu mxfp8_block, sglang / vLLM PR 67325
// _mxfp8_block: amax / 448 rounded up to a power of two, e4m3_rn of the clamped x / scale), so a native and an EXL3
// expert see a token's inputs identically (theirs dequantized, these codes as the mma takes them); the gate-up epilogue
// quantizes each pair's softcapped SwiGLU row (rounded to bf16 first, the W8A16 kernels' rounding point) the same way
// for the down. The products run on the block-scaled e4m3 mma (mma.sync ... kind::mxf8f6f4.block_scale.scale_vec::1X
// .m16n8k32, sm_121a): codes times codes, both sides' ue8m0 scales applied by the tensor core, fp32 accumulate through
// the mma's C in k32 group order (the probe: 7.8e-7 rms relative error at K 4608, vs 1.5e-7 for fp32 adds of exact
// group sums). The weights stay the arena's packed blocks: lane (gq, t)'s code word (j, h) is column 8j + gq, inputs
// 16h + 4t .. + 3, the e4m3 mma's B fragment as stored; column 8j + gq's exponent is byte [gq >> 1][j][gq & 1].
//
// A quantized row ("q8 row", q8_stride(K) bytes): K codes, then K / 32 exponents, padded to 16 bytes. Inside each
// 32-input group the codes are in fragment order, byte 8t + 4h + i holding input 16h + 4t + i: a lane's two A words of
// a row (inputs 4t .. and 16 + 4t ..) are one 8-byte load.
#include <cuda_fp8.h>

namespace {

__host__ __device__ constexpr int q8_stride(int k) { return (k + k / 32 + 15) / 16 * 16; }

// mxfp8_block's scale (inkling2_exl3.cu): amax (fmaxf from 0 over |x|) * (1 / 448), its exponent rounded up (bits +
// 0x7FFFFF), 0 at or below 2^-127, at most 254; the inverse 2^127 at exponent 0, else rcp_rn(2^(e - 127)) (exact)
__device__ __forceinline__ uint32_t q8_exponent(float amax) {
  const float raw = __fmul_rn(amax, 1.0f / 448.0f);
  const uint32_t e = ((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23;
  return raw <= 0x1p-127f ? 0u : min(e, 254u);
}

__device__ __forceinline__ float q8_inverse(uint32_t e) {
  return e == 0u ? 0x1p127f : __frcp_rn(__uint_as_float(e << 23));
}

// two values' e4m3 codes, a in the low byte: e4m3_rn(clamp(v x inverse, -448, 448)), cvt.rn.satfinite (mxfp8_block's
// e4m3_rn, kept as the code instead of dequantized)
__device__ __forceinline__ uint32_t q8_codes2(float a, float b, float inv) {
  const float2 v = make_float2(fminf(fmaxf(__fmul_rn(a, inv), -448.0f), 448.0f),
                               fminf(fmaxf(__fmul_rn(b, inv), -448.0f), 448.0f));
  return __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
}

// D += (A x 2^(sa - 127)) (B x 2^(sb - 127)): a0 / a1 rows gq / gq + 8 inputs 4t .. 4t + 3, a2 / a3 inputs 16 + 4t ..;
// b0 / b1 column gq, the same inputs. Scales (scale_vec::1X, thread-id 0): row gq's byte from lane 4gq, row gq + 8's
// from lane 4gq + 1 (sa: this lane's row gq + 8 (t & 1)), column gq's from lane 4gq, byte BB of sb.
template <int BB>
__device__ __forceinline__ void mma_q8(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1, uint32_t sa, uint32_t sb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
  const uint16_t bid_b = BB, zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "h"(zero), "h"(zero), "r"(sb), "h"(bid_b),
        "h"(zero));
#else
  __trap();
#endif
}

// one k32 group of a warp's 32-column block for its RT row tiles: per matrix m and n8 tile j one mma a row tile; the
// lane's exponent word sw[m] holds its column 8j + gq at byte 2 (j & 1) of word j >> 1 (already shifted by gq & 1)
template <int M, int RT>
__device__ __forceinline__ void q8_group(float (&acc)[M][RT][NTW][4], const uint4 (&w)[M][2],
                                         const uint32_t (&sw)[M][2], const uint2 (&xa)[RT], const uint2 (&xb)[RT],
                                         const uint32_t (&sa)[RT]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int j = 0; j < NTW; ++j) {
      const uint32_t b0 = comp(w[m][j >> 1], (j & 1) << 1), b1 = comp(w[m][j >> 1], ((j & 1) << 1) | 1);
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if (j & 1)
          mma_q8<2>(acc[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1, sa[r], sw[m][j >> 1]);
        else
          mma_q8<0>(acc[m][r][j], xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1, sa[r], sw[m][j >> 1]);
      }
    }
}

// The quantized-input epilogues. Q8_F32 (M 1, down): out [pairs][N] fp32 at column col0 + 8j (+ 2t), raw_f32's.
// Q8_ACT (M 2, gate-up): the softcapped SwiGLU on the bf16-rounded sums (act(..., true)), rounded to bf16, then the
// row's 32 columns of this block quantized (amax over the quad's lanes) into the pair's q8 row: code bytes at
// fragment order positions, the exponent at N + cb. Every lane runs the shuffles; invalid rows store nothing.
enum Q8Epi : int { Q8_F32 = 0, Q8_ACT = 1 };

template <int EPI, int M, int RT>
__device__ __forceinline__ void q8_epilogue(const float (&acc)[M][RT][NTW][4], int r, void* out, int N, int cb,
                                            int col0, bool v0, bool v1, int pr0, int pr1, float cg, float cu, int t) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const bool valid = h ? v1 : v0;
    const size_t pr = (size_t)(h ? pr1 : pr0);
    if constexpr (EPI == Q8_F32) {
      if (!valid) continue;
#pragma unroll
      for (int j = 0; j < NTW; ++j)
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + pr * N + col0 + 8 * j) =
            make_float2(acc[0][r][j][2 * h], acc[0][r][j][2 * h + 1]);
    } else {
      float o[NTW][2];
      float amax = 0.0f;  // mxfp8_block's: fmaxf from 0
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          o[j][c] = bf(act(acc[0][r][j][2 * h + c], acc[M - 1][r][j][2 * h + c], cg, cu, true));
          amax = fmaxf(amax, fabsf(o[j][c]));
        }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      if (!valid) continue;
      const uint32_t e = q8_exponent(amax);
      const float inv = q8_inverse(e);
      uint8_t* row = reinterpret_cast<uint8_t*>(out) + pr * (size_t)q8_stride(N);
      // column 8j + 2t + c of the block is input k = that: fragment position 8 (k & 15) / 4 + 4 (k >> 4) + (k & 3)
#pragma unroll
      for (int j = 0; j < NTW; ++j) {
        const int pos = (2 * (j & 1) + (t >> 1)) * 8 + (j >> 1) * 4 + 2 * (t & 1);
        *reinterpret_cast<uint16_t*>(row + cb * 32 + pos) = (uint16_t)q8_codes2(o[j][0], o[j][1], inv);
      }
      if (t == 0) row[N + cb] = (uint8_t)e;
    }
  }
}

// a row tile's two rows of this lane: pairs pr0 (row gq) and pr1 (row gq + 8), their q8 rows, and the exponent row
// this lane hands the mma (row gq + 8 (t & 1))
struct Q8Rows {
  const uint8_t* x0;
  const uint8_t* x1;
  const uint8_t* xs;
  int pr0, pr1;
  bool v0, v1, vs;
};

template <int M, int RT>
struct Q8Stage {
  uint4 w[M][2];       // the lane's 32 code bytes: words (j, h) = 2j + h
  uint2 s[M];          // exponents [gq >> 1][j][c]: columns 8j + 2 (gq >> 1) + c
  uint2 xa[RT], xb[RT];  // rows gq / gq + 8: the group's code bytes 8t .. 8t + 7 (inputs 4t .., 16 + 4t ..)
  uint32_t sa[RT];
};

template <int M, int RT>
__device__ __forceinline__ void q8_load(Q8Stage<M, RT>& st, const uint4* blk, int g, int lane, int gq, int t,
                                        const Q8Rows (&rw)[RT]) {
  const uint4* b = blk + (size_t)g * (M * BLOCK_U4);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    st.w[m][0] = ld_nc4(b + m * BLOCK_U4 + 2 * lane);
    st.w[m][1] = ld_nc4(b + m * BLOCK_U4 + 2 * lane + 1);
    st.s[m] = ld_nc2(reinterpret_cast<const uint2*>(b + m * BLOCK_U4 + CODE_U4) + (gq >> 1));
  }
  const uint2 zero = make_uint2(0u, 0u);
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    st.xa[r] = rw[r].v0 ? __ldg(reinterpret_cast<const uint2*>(rw[r].x0 + g * 32 + 8 * t)) : zero;
    st.xb[r] = rw[r].v1 ? __ldg(reinterpret_cast<const uint2*>(rw[r].x1 + g * 32 + 8 * t)) : zero;
    st.sa[r] = rw[r].vs ? (uint32_t)__ldg(rw[r].xs + g) : 0u;
  }
}

template <int M, int RT>
__device__ __forceinline__ void q8_compute(float (&acc)[M][RT][NTW][4], const Q8Stage<M, RT>& st, int gq) {
  uint32_t sw[M][2];
#pragma unroll
  for (int m = 0; m < M; ++m) {
    sw[m][0] = st.s[m].x >> (8 * (gq & 1));
    sw[m][1] = st.s[m].y >> (8 * (gq & 1));
  }
  q8_group<M, RT>(acc, st.w, sw, st.xa, st.xb, st.sa);
}

// mxfp8_expert_body with quantized inputs: one warp a (plan item, 32-column block) unit over the column blocks
// [cb0, cb0 + nbp) (nbp = NB: all of them), the item's pairs 16 RT at a time, a two-stage register pipeline of
// k32 groups. X: q8 rows x_stride bytes apart (slots > 0: token rows, pair p reads row p / slots). Q8_F32 writes
// rows of N floats from column cb0 32 (a part's own rows), Q8_ACT q8 rows of N inputs.
template <int M, int EPI, int RT>
__device__ __forceinline__ void q8_body(const uint8_t* __restrict__ X, int x_stride, int slots,
                                        const uint4* __restrict__ W, int E, int KG, int NB,
                                        const int* __restrict__ items, const int* __restrict__ counts,
                                        const int* __restrict__ members, void* __restrict__ out, int N, float cg,
                                        float cu, int skip, int cb0, int nbp) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int K = KG * GS;
  const int units = __ldg(counts) * nbp;
  for (int unit = blockIdx.x * 4 + (threadIdx.x >> 5); unit < units; unit += gridDim.x * 4) {
    const int it = unit / nbp, cb = cb0 + (unit - it * nbp);
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip || e < 0 || e >= E) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK_U4);
    for (int r0 = 0; r0 < cnt; r0 += 16 * RT) {
      Q8Rows rw[RT];
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        const int a = r0 + 16 * r + gq, b = a + 8;
        rw[r].v0 = a < cnt;
        rw[r].v1 = b < cnt;
        rw[r].pr0 = rw[r].v0 ? __ldg(members + first + a) : 0;
        rw[r].pr1 = rw[r].v1 ? __ldg(members + first + b) : 0;
        rw[r].x0 = X + (size_t)(slots ? rw[r].pr0 / slots : rw[r].pr0) * x_stride;
        rw[r].x1 = X + (size_t)(slots ? rw[r].pr1 / slots : rw[r].pr1) * x_stride;
        rw[r].vs = (t & 1) ? rw[r].v1 : rw[r].v0;
        rw[r].xs = ((t & 1) ? rw[r].x1 : rw[r].x0) + K;
      }
      float acc[M][RT][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
          for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
      constexpr int D = 2;
      Q8Stage<M, RT> st[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < KG) q8_load<M, RT>(st[d], blk, d, lane, gq, t, rw);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            q8_compute<M, RT>(acc, st[d], gq);
            if (g + D < KG) q8_load<M, RT>(st[d], blk, g + D, lane, gq, t, rw);
          }
        }
      }
      // Q8_F32: a part's rows hold its own columns (N wide), so the column is relative to cb0
      const int col0 = (EPI == Q8_F32 ? (cb - cb0) : cb) * COLS + 2 * t;
#pragma unroll
      for (int r = 0; r < RT; ++r)
        q8_epilogue<EPI, M, RT>(acc, r, out, N, cb, col0, rw[r].v0, rw[r].v1, rw[r].pr0, rw[r].pr1, cg, cu, t);
    }
  }
}

}  // namespace

// Rows [rows][K] bf16 (x_stride elements apart) into q8 rows (q_stride bytes apart): a thread a 32-input group.
extern "C" __global__ void __launch_bounds__(256) ink2_mx8_quant(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                                 int rows, int K, uint8_t* __restrict__ q,
                                                                 int q_stride) {
  const int G = K >> 5;
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= rows * G) return;
  const int r = u / G, g = u - r * G;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)r * x_stride + g * 32);
  float v[32];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 w = __ldg(src + c);
    v[8 * c + 0] = lo_f(w.x);
    v[8 * c + 1] = hi_f(w.x);
    v[8 * c + 2] = lo_f(w.y);
    v[8 * c + 3] = hi_f(w.y);
    v[8 * c + 4] = lo_f(w.z);
    v[8 * c + 5] = hi_f(w.z);
    v[8 * c + 6] = lo_f(w.w);
    v[8 * c + 7] = hi_f(w.w);
  }
  float amax = 0.0f;  // mxfp8_block's: fmaxf from 0 over |x|
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const uint32_t e = q8_exponent(amax);
  const float inv = q8_inverse(e);
  uint32_t w[8];  // fragment order: word 2t + h holds inputs 16h + 4t .. 16h + 4t + 3
#pragma unroll
  for (int tt = 0; tt < 4; ++tt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int k0 = 16 * h + 4 * tt;
      w[2 * tt + h] = q8_codes2(v[k0], v[k0 + 1], inv) | q8_codes2(v[k0 + 2], v[k0 + 3], inv) << 16;
    }
  uint4* dst = reinterpret_cast<uint4*>(q + (size_t)r * q_stride + g * 32);
  dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
  dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
  q[(size_t)r * q_stride + K + g] = (uint8_t)e;
}

// The register forms on q8 inputs: gate-up (Q8_ACT into q8 rows of N) and down (Q8_F32, columns [cb0, cb0 + nbp)).
#define INK2_MX8A(NAME, M, EPI, RT)                                                                                   \
  extern "C" __global__ void __launch_bounds__(4 * 32) NAME(                                                       \
      const uint8_t* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int E, int KG, int NB,  \
      const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,              \
      void* __restrict__ out, int N, float cg, float cu, int skip, int cb0, int nbp) {                              \
    q8_body<M, EPI, RT>(X, x_stride, slots, W, E, KG, NB, items, counts, members, out, N, cg, cu, skip, cb0, nbp); \
  }
INK2_MX8A(ink2_mx8a_gate_up_r1, 2, Q8_ACT, 1)
INK2_MX8A(ink2_mx8a_gate_up_r2, 2, Q8_ACT, 2)
INK2_MX8A(ink2_mx8a_gate_up_r4, 2, Q8_ACT, 4)
INK2_MX8A(ink2_mx8a_down_f32_r1, 1, Q8_F32, 1)
INK2_MX8A(ink2_mx8a_down_f32_r2, 1, Q8_F32, 2)
INK2_MX8A(ink2_mx8a_down_f32_r4, 1, Q8_F32, 4)

// The wide prompt chunks' W8A8 form (128-pair plan items; mx_wide_tiles' structure on q8 rows): a CTA of 2 WN warps
// stages each pair of k32 groups of the item's rows (their codes: 64 bytes a row, half of bf16's) and of its weight
// blocks in shared memory (cp.async, ST stages), and the warps share them: an item of more than 64 pairs as two warp
// rows of WN warps (128 rows x WN 32-column blocks), one of at most 64 as one row of 2 WN warps (64 rows x 2 WN
// blocks). Each stage also stages the rows' exponents (a 4-byte word of 4 groups a row, cp.async 4), so every
// main-loop operand arrives asynchronously as in the bf16 form. The weight chunks are staged in place: permuting their
// shared-memory slots for conflict-free reads made the cp.async writes conflict and measured 37% slower at 2048 rows
// (out/prefill/p4/p4-mx/bench/bench4.log). Column blocks [cb0, cb0 + nbp) only (the down's column parts,
// forward.moeOverlap; nbp = NB: all), Q8_F32 rows nbp 32 wide.
namespace {

template <int M, int WN, int ST_>
struct WideQ8 {
  static constexpr int SG = 2, XC = 4, WB = M * BLOCK_U4, ST = ST_;  // + the rows' exponent words: 128 / 4 uint4
  static constexpr int SU_A = 128 * XC + 32 + WN * SG * WB, SU_B = 64 * XC + 32 + 2 * WN * SG * WB;
  static constexpr int SU = SU_A > SU_B ? SU_A : SU_B;
  static constexpr int SMEM = ST * SU * 16;
};

// row r's 16-byte code chunk c of a stage (4 a row): rows r .. r + 3 put a group's 8-byte lane reads on distinct banks
__device__ __forceinline__ int q8_chunk(int r, int c) { return r * 4 + (c ^ (((r >> 1) & 1) << 1)); }

template <int M, int EPI, int WM, int WNL, int NT, int SU, int ST>
__device__ __forceinline__ void q8_wide_tiles(const uint8_t* __restrict__ X, int x_stride, int slots,
                                              const uint4* __restrict__ W, int KG, int NB,
                                              const int* __restrict__ members, void* __restrict__ out, int N, float cg,
                                              float cu, int cbt, int e, int first, int cnt, int tb, int cb0, int nbp) {
  constexpr int THREADS = WM * WNL * 32, BM = 64 * WM, XC = 4, SG = 2, WB = M * BLOCK_U4, SCU = 32;
  constexpr int XU = BM * XC, XPT = (XU + THREADS - 1) / THREADS;
  static_assert(XU + SCU + WNL * SG * WB <= SU, "a stage of this layout fits the shared stage");
  static_assert(BM <= THREADS, "a thread stages a row's exponent word");
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wn = warp % WNL;
  const int t = lane & 3, gq = lane >> 2, cbl = cbt * WNL + wn, cb = cb0 + cbl, cbs = min(WNL, nbp - cbt * WNL);
  const int K = KG * GS;
  int xsrc[XPT];  // the row's q8 offset in X plus the chunk's (-1: a row past the item)
  int xdst[XPT];
#pragma unroll
  for (int i = 0; i < XPT; ++i) {
    const int q = tid + i * THREADS, r = q / XC, c = q - r * XC;
    xdst[i] = q8_chunk(r, c);
    xsrc[i] = -1;
    if (q < XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = (slots ? p / slots : p) * x_stride + 16 * c;
    }
  }
  int srow = -1;  // this thread's row (tid < BM): its exponents' offset in X
  if (tid < BM && tid < cnt) {
    const int p = __ldg(members + first + tid);
    srow = (slots ? p / slots : p) * x_stride + K;
  }
  const uint4* wsrc = W + ((size_t)e * NB + cb0 + cbt * WNL) * (size_t)KG * WB;
  auto stage = [&](int s, int g0) {  // k32 groups g0 .. g0 + ng - 1 (ng < SG only at an odd end)
    const int ng = min(SG, KG - g0);
    uint4* xs = sm + s * SU;
#pragma unroll
    for (int i = 0; i < XPT; ++i)
      if (xsrc[i] >= 0 && ((tid + i * THREADS) % XC) < ng * 2)
        tf_common::cp16(xs + xdst[i], reinterpret_cast<const uint4*>(X + xsrc[i] + g0 * GS));
    // the word of groups 4 (g0 / 4) .. + 3 (this stage's two at bytes g0 & 3, (g0 & 3) + 1)
    if (srow >= 0) tf_common::cp4(reinterpret_cast<uint32_t*>(xs + XU) + tid, X + srow + 4 * (g0 >> 2));
    uint4* ws = xs + XU + SCU;
    for (int q = tid; q < cbs * SG * WB; q += THREADS) {
      const int j = q / (SG * WB), o = q - j * SG * WB;
      if (o < ng * WB) tf_common::cp16(ws + q, wsrc + ((size_t)j * KG + g0) * WB + o);
    }
  };
  constexpr int TA = NT > 0 ? NT : 1;  // array extent
  const int base = 16 * tb;            // the warp row's first row (tb its first 16-row tile)
  const bool active = NT > 0 && cbl < nbp;
  float acc[M][TA][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < TA; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = (KG + SG - 1) / SG;
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < steps) stage(s, s * SG);
    tf_common::cp_commit();
  }
  for (int k = 0; k < steps; ++k) {
    tf_common::cp_wait<ST - 2>();
    __syncthreads();
    if (k + ST - 1 < steps) stage((k + ST - 1) % ST, (k + ST - 1) * SG);
    tf_common::cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % ST) * SU;
#pragma unroll
    for (int gi = 0; gi < SG; ++gi) {
      if (k * SG + gi >= KG) break;
      const uint4* ws = xs + XU + SCU + (wn * SG + gi) * WB;
      uint4 w[M][2];
      uint32_t sw[M][2];
#pragma unroll
      for (int m = 0; m < M; ++m) {
        w[m][0] = ws[m * BLOCK_U4 + 2 * lane];
        w[m][1] = ws[m * BLOCK_U4 + 2 * lane + 1];
        const uint2 sv = reinterpret_cast<const uint2*>(ws + m * BLOCK_U4 + CODE_U4)[gq >> 1];
        sw[m][0] = sv.x >> (8 * (gq & 1));
        sw[m][1] = sv.y >> (8 * (gq & 1));
      }
      uint2 xa[TA], xb[TA];
      uint32_t sa[TA];
#pragma unroll
      for (int r = 0; r < NT; ++r) {
        const int r0 = base + 16 * r + gq, c = 2 * gi + (t >> 1);
        xa[r] = reinterpret_cast<const uint2*>(xs + q8_chunk(r0, c))[t & 1];
        xb[r] = reinterpret_cast<const uint2*>(xs + q8_chunk(r0 + 8, c))[t & 1];
        // the exponent this lane hands the mma: row gq + 8 (t & 1)'s, byte (2k + gi) & 3 of its staged word
        sa[r] = reinterpret_cast<const uint32_t*>(xs + XU)[r0 + 8 * (t & 1)] >> (8 * (2 * (k & 1) + gi));
      }
      if constexpr (NT > 0) q8_group<M, TA>(acc, w, sw, xa, xb, sa);
    }
  }
  if (!active) return;
  if constexpr (NT > 0) {
    const int col0 = (EPI == Q8_F32 ? cbl : cb) * COLS + 2 * t;
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int m0 = base + 16 * r + gq, m1 = m0 + 8;
      const bool v0 = m0 < cnt, v1 = m1 < cnt;
      const int pr0 = v0 ? __ldg(members + first + m0) : 0, pr1 = v1 ? __ldg(members + first + m1) : 0;
      q8_epilogue<EPI, M, TA>(acc, r, out, N, cb, col0, v0, v1, pr0, pr1, cg, cu, t);
    }
  }
}

// One CTA a (plan item, WN-block group of the part): grid items x ceil(nbp / WN). An item of more than 64 pairs: its
// 16-row tiles split between the two warp rows (row 0 takes ceil(tiles / 2)); of at most 64: its first
// ceil(nbp / 2 WN) CTAs, 2 WN blocks each, the rest leave at once. Every warp of a CTA runs the same `steps` barriers.
template <int M, int EPI, int WN, int ST>
__device__ __forceinline__ void q8_wide_body(const uint8_t* __restrict__ X, int x_stride, int slots,
                                             const uint4* __restrict__ W, int E, int KG, int NB,
                                             const int* __restrict__ items, const int* __restrict__ counts,
                                             const int* __restrict__ members, void* __restrict__ out, int N, float cg,
                                             float cu, int skip, int cb0, int nbp) {
  using P = WideQ8<M, WN, ST>;
  const int nbt = (nbp + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip || e < 0 || e >= E) return;
  if (cnt > 64) {
    const int tiles = (min(cnt, 128) + 15) >> 4, t0 = (tiles + 1) >> 1, wm = (threadIdx.x >> 5) / WN;
    const int nt = wm ? tiles - t0 : t0, tb = wm ? t0 : 0;
#define TF_WIDE(NT)                                                                                              \
  q8_wide_tiles<M, EPI, 2, WN, NT, P::SU, P::ST>(X, x_stride, slots, W, KG, NB, members, out, N, cg, cu, cbt, e, \
                                                 first, cnt, tb, cb0, nbp)
    switch (nt) {
      case 4: TF_WIDE(4); break;
      case 3: TF_WIDE(3); break;
      case 2: TF_WIDE(2); break;
      case 1: TF_WIDE(1); break;
      default: TF_WIDE(0);
    }
#undef TF_WIDE
  } else {
    if (cbt * 2 * WN >= nbp) return;  // the whole CTA: no barrier is entered
    const int nt = (cnt + 15) >> 4;
#define TF_WIDE(NT)                                                                                                  \
  q8_wide_tiles<M, EPI, 1, 2 * WN, NT, P::SU, P::ST>(X, x_stride, slots, W, KG, NB, members, out, N, cg, cu, cbt, e, \
                                                     first, cnt, 0, cb0, nbp)
    switch (nt) {
      case 4: TF_WIDE(4); break;
      case 3: TF_WIDE(3); break;
      case 2: TF_WIDE(2); break;
      case 1: TF_WIDE(1); break;
      default: TF_WIDE(0);
    }
#undef TF_WIDE
  }
}

}  // namespace

// The wide W8A8 entries (kern.zig q8_wide: WN and ST of each, wideQ8Smem = WideQ8<M, WN, ST>::SMEM): 2 WN warps a CTA.
#define INK2_MX8A_WIDE(NAME, M, EPI, WN, ST)                                                                          \
  extern "C" __global__ void __launch_bounds__(2 * WN * 32) NAME(                                                   \
      const uint8_t* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W, int E, int KG, int NB,   \
      const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,               \
      void* __restrict__ out, int N, float cg, float cu, int skip, int cb0, int nbp) {                               \
    q8_wide_body<M, EPI, WN, ST>(X, x_stride, slots, W, E, KG, NB, items, counts, members, out, N, cg, cu, skip, cb0, \
                                 nbp);                                                                              \
  }
#ifndef INK2_MX8A_UP_WN
#define INK2_MX8A_UP_WN 2
#define INK2_MX8A_UP_ST 2
#define INK2_MX8A_DN_WN 2
#define INK2_MX8A_DN_ST 3
#endif
INK2_MX8A_WIDE(ink2_mx8a_wide_gate_up, 2, Q8_ACT, INK2_MX8A_UP_WN, INK2_MX8A_UP_ST)
INK2_MX8A_WIDE(ink2_mx8a_wide_down_f32, 1, Q8_F32, INK2_MX8A_DN_WN, INK2_MX8A_DN_ST)
// kern.zig wideQ8Smem: ST x max(128 rows' 4 chunks + WN blocks, 64 rows' + 2 WN blocks) of 2 groups, + 32 uint4 of
// exponent words, x 16 bytes
constexpr int wide_q8_smem(int parts, int wn, int st) {
  const int a = 128 * 4 + wn * 2 * parts * 66, b = 64 * 4 + 2 * wn * 2 * parts * 66;
  return st * 16 * (32 + (a > b ? a : b));
}
static_assert(WideQ8<2, INK2_MX8A_UP_WN, INK2_MX8A_UP_ST>::SMEM == wide_q8_smem(2, INK2_MX8A_UP_WN, INK2_MX8A_UP_ST));
static_assert(WideQ8<1, INK2_MX8A_DN_WN, INK2_MX8A_DN_ST>::SMEM == wide_q8_smem(1, INK2_MX8A_DN_WN, INK2_MX8A_DN_ST));
// ==== end p4-mx ====
