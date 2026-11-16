// p4-dattn: split-K flash decoding on tensor cores for a few query rows against a long key range, shared by
// inkling2.cu (the target's decode / verify / short prompt chunks, TF_INKLING2_DECODE_ATTN) and inkling2_dspark.cu (the
// DSpark draft's block, TF_INKLING2_DRAFT_ATTN). NOT bitwise with the scalar kernels it replaces (ink2_attention,
// dsp_attention): the products are the same bf16 x bf16 ones, but the dot and value sums run in the mma's order, the
// softmax is the online one (a running max and sum a key split, rescaled; splits merged in split order), and the
// probabilities enter the value product as bf16(exp(s - running max)) instead of the reference's bf16(p) (or fp32 p).
//
// One block = (key split, kv head): every query of the kv group (rows x G heads, 16 a warp) against the split's keys,
// so each key and value row is read once a launch (the scalar kernels read them once a (row, head)). Keys stream
// through shared memory in tiles of KT (two cp.async stages, [key][D + 8] bf16 rows: ldmatrix rows on distinct
// banks); a tile's scores are q k^T on mma.sync m16n8k16 (fp32 sums), the policy's epilogue makes the score (and -inf
// where the query does not see the key), then the online softmax update and o += p v (p bf16 as the A operand, the
// values by ldmatrix.trans). The block writes each query's (m, l, o[D]) partial: m its running max, l = sum exp(s - m)
// (fp32, before the bf16 rounding of p), o = sum bf16(exp(s - m)) v (fp32). The caller's merge kernel combines the
// splits in split order. The grid is fixed (gridDim.x splits, each a contiguous run of tiles of the range the policy
// computes from its arguments), so a CUDA graph replays it at any position with only the position argument patched.
#pragma once
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace fdec {
constexpr int KT = 32;  // keys a staged tile

__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// 16 bytes global -> shared, asynchronously; `bytes` 0 zero-fills them (the source is not read)
__device__ __forceinline__ void cp16(uint32_t dst, const void* src, int bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
#endif
}
__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// four 8x8 bf16 matrices from shared memory (lane l gives row l % 8 of matrix l / 8), plain or transposed
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

// c += a b, one m16n8k16 product: bf16 inputs (exact products), fp32 accumulation
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  __trap();
#endif
}

// two floats rounded to bf16 into an mma operand register, the lower column in the low half
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

// Dynamic shared memory of a block: NS stages of a key tile and a value tile of KTILE keys.
__host__ __device__ constexpr int smem_bytes(int D, int KTILE = KT, int NS = 2) { return NS * 2 * KTILE * (D + 8) * 2; }

// The partial of split `blockIdx.x` for the block's queries. P (the policy) gives:
//   int nq, nkeys                     the block's queries (rows x G) and the key range's length (keys 0 .. nkeys - 1)
//   typename P::Query query(int qi)   a query's own data (its position, window, bias row, partial slot ...)
//   const __nv_bfloat16* qrow(int qi) its D bf16 values
//   bool rows(int j, k&, v&)          key j's key and value rows (false: zero-filled, never read)
//   float score(float dot, Query, j)  the score from the fp32 dot, -INFINITY where the query does not see key j
//   float* slot(Query)                where its partial goes: [splits][2 + D] floats (m, l, o[D]) a query
// blockDim.x = 32 x ceil(nq / 16) (warp w: queries 16 w .. 16 w + 15; queries past nq repeat the last, unwritten).
// KTILE keys a tile (a multiple of 16), NS cp.async stages (NS - 1 tiles in flight while one is used).
template <int D, int KTILE, int NS, class P>
__device__ __forceinline__ void partial_t(const P& p) {
  constexpr int KT = KTILE;
  constexpr int KS = D / 16;   // q k^T k-steps
  constexpr int RW = D + 8;    // bf16 a staged key or value row
  constexpr int ST = KT * RW;  // bf16 a stage of keys (or values)
  constexpr int NT = KT / 8;   // the tile's 8-key n-tiles
  extern __shared__ uint4 fd_smem[];
  const uint32_t kt = saddr(fd_smem);    // [NS][KT][RW] keys
  const uint32_t vt = kt + NS * ST * 2;  // [NS][KT][RW] values
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int S = gridDim.x, s = blockIdx.x;
  const int ntile = (p.nkeys + KT - 1) / KT;
  const int ta = (int)((long long)ntile * s / S), tb = (int)((long long)ntile * (s + 1) / S);
  // the lane's two queries: fragment rows lane / 4 and lane / 4 + 8 of the warp's 16
  int qi[2];
  typename P::Query qy[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    qi[i] = min(warp * 16 + (lane >> 2) + 8 * i, p.nq - 1);
    qy[i] = p.query(qi[i]);
  }
  // the queries as the A operand, once: register j of k-step s holds row (j & 1 ? +8 : +0), dims 16 s + 2 (lane % 4)
  // + 8 (j / 2) and the next
  uint32_t qa[KS][4];
  {
    const uint32_t* q0r = reinterpret_cast<const uint32_t*>(p.qrow(qi[0]));
    const uint32_t* q1r = reinterpret_cast<const uint32_t*>(p.qrow(qi[1]));
#pragma unroll
    for (int k = 0; k < KS; ++k) {
      const int c = k * 8 + (lane & 3);
      qa[k][0] = q0r[c];
      qa[k][1] = q1r[c];
      qa[k][2] = q0r[c + 4];
      qa[k][3] = q1r[c + 4];
    }
  }
  // tile t (keys KT t ..) into buffer t % NS: every thread a run of 16-byte pieces, a key's rows from the policy
  auto stage = [&](int t) {
    const int j0 = t * KT;
    const uint32_t b = (uint32_t)((t % NS) * ST * 2);
    for (int c = threadIdx.x; c < KT * (D / 8); c += blockDim.x) {
      const int kk = c / (D / 8), part = c - kk * (D / 8);
      const __nv_bfloat16* kr;
      const __nv_bfloat16* vr;
      const bool in = p.rows(j0 + kk, kr, vr);
      const uint32_t dst = b + (uint32_t)((kk * RW + part * 8) * 2);
      cp16(kt + dst, in ? (const void*)(kr + part * 8) : (const void*)p.qrow(0), in ? 16 : 0);
      cp16(vt + dst, in ? (const void*)(vr + part * 8) : (const void*)p.qrow(0), in ? 16 : 0);
    }
    commit();
  };
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  float oa[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) oa[j][0] = oa[j][1] = oa[j][2] = oa[j][3] = 0.f;
  // the pipeline: tiles ta .. ta + NS - 2 in flight first; each step stages tile t + NS - 1 (an empty group past the
  // split's end, so the count of pending groups stays NS - 1), waits for tile t's, uses it
#pragma unroll
  for (int i = 0; i < NS - 1; ++i) {
    if (ta + i < tb) {
      stage(ta + i);
    } else {
      commit();
    }
  }
  for (int t = ta; t < tb; ++t) {
    if (t + NS - 1 < tb) {
      stage(t + NS - 1);  // into the buffer tile t - 1 used: every warp passed the barrier after it
    } else {
      commit();
    }
    wait<NS - 1>();
    __syncthreads();
    // the tile's dots: fragment (n, e) is row (e < 2 ? +0 : +8), key 8 n + 2 (lane % 4) + (e & 1) of the tile
    float sa[NT][4];
    {
      const uint32_t b = kt + (uint32_t)((t % NS) * ST * 2);
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.f;
#pragma unroll
        for (int k = 0; k < KS; k += 2) {
          uint32_t r[4];  // keys 8 n .. 8 n + 7 by dims 16 k + {0, 8, 16, 24}: k-steps k and k + 1
          ldsm4(r, b + (uint32_t)(((n * 8 + (lane & 7)) * RW + k * 16 + (lane >> 3) * 8) * 2));
          mma(sa[n], qa[k], r[0], r[1]);
          mma(sa[n], qa[k + 1], r[2], r[3]);
        }
      }
    }
    // the scores, the tile's row maxima (a row's 32 scores are its quad's: lanes 4 (lane / 4) ..)
    float tmx[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = t * KT + n * 8 + 2 * (lane & 3) + (e & 1);
        sa[n][e] = p.score(sa[n][e], qy[e >> 1], j);
        tmx[e >> 1] = fmaxf(tmx[e >> 1], sa[n][e]);
      }
    float sc[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      tmx[i] = fmaxf(tmx[i], __shfl_xor_sync(0xffffffffu, tmx[i], 1));
      tmx[i] = fmaxf(tmx[i], __shfl_xor_sync(0xffffffffu, tmx[i], 2));
      const float mn = fmaxf(m[i], tmx[i]);
      sc[i] = m[i] == -INFINITY ? 0.f : __expf(m[i] - mn);  // nothing seen yet: o and l are 0
      m[i] = mn;
      l[i] *= sc[i];
    }
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
      oa[j][0] *= sc[0];
      oa[j][1] *= sc[0];
      oa[j][2] *= sc[1];
      oa[j][3] *= sc[1];
    }
    // p = exp(s - m) (0 where masked) as the A operand: n-tiles 2 k and 2 k + 1 are k-step k's two key halves (the
    // score fragment's layout is the operand's)
    uint32_t pa[NT / 2][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      float pv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        pv[e] = sa[n][e] == -INFINITY ? 0.f : __expf(sa[n][e] - m[e >> 1]);
        l[e >> 1] += pv[e];
      }
      pa[n >> 1][(n & 1) * 2] = pack(pv[0], pv[1]);
      pa[n >> 1][(n & 1) * 2 + 1] = pack(pv[2], pv[3]);
    }
    // o += p v: B = the values (ldmatrix.trans of [key][d]); matrix mm = lane / 8 is keys 16 k + 8 (mm % 2) ..,
    // dims 8 j + 8 (mm / 2) ..
    {
      const uint32_t b = vt + (uint32_t)((t % NS) * ST * 2);
      const int mm = lane >> 3;
#pragma unroll
      for (int k = 0; k < NT / 2; ++k)
#pragma unroll
        for (int j = 0; j < D / 8; j += 2) {
          uint32_t r[4];
          ldsm4t(r, b + (uint32_t)(((k * 16 + (mm & 1) * 8 + (lane & 7)) * RW + j * 8 + (mm >> 1) * 8) * 2));
          mma(oa[j], pa[k], r[0], r[1]);
          mma(oa[j + 1], pa[k], r[2], r[3]);
        }
    }
    __syncthreads();  // the buffer is read before the next stage refills it
  }
  // the row sums over the quad, then the partials: lane 4 r + c holds dims 8 j + 2 c, + 1 of its two rows
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
    l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (warp * 16 + (lane >> 2) + 8 * i >= p.nq) continue;  // a padding row
    float* o = p.slot(qy[i]) + (size_t)s * (2 + D);
    if ((lane & 3) == 0) {
      o[0] = m[i];
      o[1] = l[i];
    }
#pragma unroll
    for (int j = 0; j < D / 8; ++j)
      *reinterpret_cast<float2*>(o + 2 + j * 8 + 2 * (lane & 3)) = make_float2(oa[j][2 * i], oa[j][2 * i + 1]);
  }
}

// The production configuration: 32-key tiles, two stages.
template <int D, class P>
__device__ __forceinline__ void partial(const P& p) {
  partial_t<D, KT, 2>(p);
}

// The merge of a query's `S` partials ([S][2 + D] at `part`) for output dim d, in split order:
// M = max(m_s..., x), L = sum l_s exp(m_s - M) (+ exp(x - M) with has_extra: the target's sink x),
// o = sum o_s exp(m_s - M); returns o / L (every thread of the query computes the same M and L).
__device__ __forceinline__ float merge(const float* __restrict__ part, int S, int D, int d, float extra_max,
                                       bool has_extra) {
  float M = has_extra ? extra_max : -INFINITY;
  for (int s = 0; s < S; ++s) M = fmaxf(M, part[(size_t)s * (2 + D)]);
  float L = has_extra ? expf(extra_max - M) : 0.f;
  float o = 0.f;
  for (int s = 0; s < S; ++s) {
    const float* x = part + (size_t)s * (2 + D);
    if (x[0] == -INFINITY) continue;  // a split with no key this query sees
    const float w = expf(x[0] - M);
    L += x[1] * w;
    o += x[2 + d] * w;
  }
  return o / L;
}
}  // namespace fdec
