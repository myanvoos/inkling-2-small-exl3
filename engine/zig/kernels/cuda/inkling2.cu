// Inkling-2's own ops on CUDA, each at the reference's (HF transformers' modeling_inkling2, eager) dtypes and
// rounding points; the M:line comments name the reference lines (as the inkling-tensorfold overlay's model.py does).
// Bring-up kernels: one block a row (or a row's head), plain loops, fp32 sums. The routed experts are
// inkling2_nvfp4.cu / inkling2_mxfp8.cu; the dense projections are cuBLASLt GEMMs (fp32 compute).
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p, size_t i) { return __bfloat162float(p[i]); }

// sum of v over the block (blockDim.x a multiple of 32, at most 1024), every thread gets it
__device__ float block_sum(float v) {
  __shared__ float part[32];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  float s = l < nw ? part[l] : 0.f;
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}

__device__ float block_max(float v) {
  __shared__ float part[32];
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  const int nw = blockDim.x >> 5;
  float s = l < nw ? part[l] : -INFINITY;
  for (int o = 16; o > 0; o >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, o));
  return s;
}

}  // namespace

// ------------------------------------------------------------------------------------------------- norms
// Inkling2RMSNorm (M:107-121): x fp32 (from bf16 or fp32 rows), x * rsqrt(mean(x^2) + eps), times the weight (bf16
// -> fp32), rounded to bf16 (the callers' .to(dtype)). One block a row. in_f32: rows are fp32, else bf16. Each
// thread's strided sum is taken in its order with six loads in flight (the bits of the one-load loop).
namespace {
template <bool F32>
__device__ __forceinline__ float rn_x(const void* in, size_t i) {
  if constexpr (F32) return reinterpret_cast<const float*>(in)[i];
  return ld(reinterpret_cast<const __nv_bfloat16*>(in), i);
}

template <bool F32>
__device__ __forceinline__ void rmsnorm_body(const void* __restrict__ in, const __nv_bfloat16* __restrict__ w,
                                             __nv_bfloat16* __restrict__ out, int H, float eps) {
  const size_t r = blockIdx.x;
  const int B = blockDim.x;
  float ss = 0.f;
  int i = threadIdx.x;
  for (; i + 5 * B < H; i += 6 * B) {
    float x[6];
#pragma unroll
    for (int j = 0; j < 6; ++j) x[j] = rn_x<F32>(in, r * H + i + j * B);
#pragma unroll
    for (int j = 0; j < 6; ++j) ss += x[j] * x[j];
  }
  for (; i < H; i += B) {
    const float x = rn_x<F32>(in, r * H + i);
    ss += x * x;
  }
  const float inv = rsqrtf(block_sum(ss) / (float)H + eps);
  for (int i = threadIdx.x; i < H; i += B) {
    const float x = rn_x<F32>(in, r * H + i);
    out[r * H + i] = __float2bfloat16_rn(ld(w, i) * (x * inv));
  }
}
}  // namespace

extern "C" __global__ void ink2_rmsnorm(const void* __restrict__ in, int in_f32, const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ out, int H, float eps) {
  if (in_f32)
    rmsnorm_body<true>(in, w, out, H, eps);
  else
    rmsnorm_body<false>(in, w, out, H, eps);
}

// ------------------------------------------------------------------------------------------------- short convolutions
// Inkling2ShortConvolution (M:247-305) on a chunk of T rows, per channel: y[t] = (sum_k w[c][k] x[t - (K-1) + k]
// + x[t]) / sqrt(K + 1), fp32 (the conv weight is kept fp32: KEEP_IN_FP32), rounded to bf16. `ring` [K-1][C] fp32
// holds the inputs of the K-1 positions before the chunk (zeros at the sequence start) and is advanced past it.
// x / y [T][C] bf16 (y may alias x: each thread reads its own channel's column before it writes it).
// hist (speculation's rollback; null: none): [K - 1 + T][C] fp32 -- the ring as it was, then every row's input, so
// the ring after keeping j rows is hist rows j .. j + K - 2 (Model.rewind).
extern "C" __global__ void ink2_sconv_static(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* y,
                                             const __nv_bfloat16* __restrict__ w, float* __restrict__ ring, int T,
                                             int C, int K, float* __restrict__ hist) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float win[8];  // the last K inputs, oldest first (K <= 8)
  for (int k = 0; k < K - 1; ++k) win[k] = ring[(size_t)k * C + c];
  if (hist != nullptr) {
    for (int k = 0; k < K - 1; ++k) hist[(size_t)k * C + c] = win[k];
    for (int t = 0; t < T; ++t) hist[(size_t)(K - 1 + t) * C + c] = ld(x, (size_t)t * C + c);
  }
  float wk[8];
  for (int k = 0; k < K; ++k) wk[k] = ld(w, (size_t)c * K + k);
  const float div = sqrtf((float)(K + 1));
  for (int t = 0; t < T; ++t) {
    const float xt = ld(x, (size_t)t * C + c);
    win[K - 1] = xt;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s = fmaf(wk[k], win[k], s);
    y[(size_t)t * C + c] = __float2bfloat16_rn((s + xt) / div);
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
  for (int k = 0; k < K - 1; ++k) ring[(size_t)k * C + c] = win[k];
}

// Inkling2DynamicShortConvolution (M:308-353): taps static_w[lag][c] (bf16 -> fp32) + dyn[t][head(c)][lag] (fp32,
// from weight_proj); y[t] = (sum_lag tap * x[t - lag] + x[t]) / sqrt(2K + 1); lag 0 is the current token. `head(c)`
// = c / D (D = channels a kv head). dyn [T][Hkv][K] fp32; ring as ink2_sconv_static's.
extern "C" __global__ void ink2_sconv_dynamic(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* y,
                                              const __nv_bfloat16* __restrict__ static_w, const float* __restrict__ dyn,
                                              float* __restrict__ ring, int T, int C, int K, int D,
                                              float* __restrict__ hist) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int head = c / D, heads = C / D;
  float win[8];
  for (int k = 0; k < K - 1; ++k) win[k] = ring[(size_t)k * C + c];
  if (hist != nullptr) {
    for (int k = 0; k < K - 1; ++k) hist[(size_t)k * C + c] = win[k];
    for (int t = 0; t < T; ++t) hist[(size_t)(K - 1 + t) * C + c] = ld(x, (size_t)t * C + c);
  }
  const float div = sqrtf((float)(2 * K + 1));
  for (int t = 0; t < T; ++t) {
    const float xt = ld(x, (size_t)t * C + c);
    win[K - 1] = xt;
    float s = 0.f;
    for (int lag = 0; lag < K; ++lag) {
      const float tap = ld(static_w, (size_t)lag * C + c) + dyn[((size_t)t * heads + head) * K + lag];
      s = fmaf(tap, win[K - 1 - lag], s);
    }
    y[(size_t)t * C + c] = __float2bfloat16_rn((s + xt) / div);
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
  for (int k = 0; k < K - 1; ++k) ring[(size_t)k * C + c] = win[k];
}

// The two short convolutions on a prompt chunk's T rows in parallel, in place, without a history: the rows in segments
// of SC_SEG. ink2_sconv_edges first saves, for every segment, the K - 1 inputs before it (from the ring for the first
// segment) into `edges` [segments][K - 1][C] fp32, and segment 0's thread then advances the ring to the chunk's last
// K - 1 inputs (no other thread reads the ring). Then ink2_sconv_static_seg / ink2_sconv_dynamic_seg run the one-thread
// loop of ink2_sconv_static / ink2_sconv_dynamic over each (segment, channel), its window started from `edges`: every
// output is that loop's arithmetic on the same inputs (each row read before its thread writes it), the same bits.
namespace {
constexpr int SC_SEG = 16;
}

extern "C" __global__ void ink2_sconv_edges(const __nv_bfloat16* __restrict__ x, float* __restrict__ ring,
                                            float* __restrict__ edges, int T, int C, int K) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int segs = (T + SC_SEG - 1) / SC_SEG;
  if (i >= (size_t)segs * C) return;
  const int sg = (int)(i / C), c = (int)(i - (size_t)sg * C);
  for (int k = 0; k < K - 1; ++k) {
    const int src = sg * SC_SEG - (K - 1) + k;
    edges[((size_t)sg * (K - 1) + k) * C + c] = src >= 0 ? ld(x, (size_t)src * C + c) : ring[(size_t)(K - 1 + src) * C + c];
  }
  if (sg != 0) return;
  float win[8];  // the chunk's last K - 1 inputs, oldest first (the ring's own entries when T < K - 1)
  for (int k = 0; k < K - 1; ++k) {
    const int src = T - (K - 1) + k;
    win[k] = src >= 0 ? ld(x, (size_t)src * C + c) : ring[(size_t)(K - 1 + src) * C + c];
  }
  for (int k = 0; k < K - 1; ++k) ring[(size_t)k * C + c] = win[k];
}

extern "C" __global__ void ink2_sconv_static_seg(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ w,
                                                 const float* __restrict__ edges, int T, int C, int K) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int segs = (T + SC_SEG - 1) / SC_SEG;
  if (i >= (size_t)segs * C) return;
  const int sg = (int)(i / C), c = (int)(i - (size_t)sg * C);
  float win[8];
  for (int k = 0; k < K - 1; ++k) win[k] = edges[((size_t)sg * (K - 1) + k) * C + c];
  float wk[8];
  for (int k = 0; k < K; ++k) wk[k] = ld(w, (size_t)c * K + k);
  const float div = sqrtf((float)(K + 1));
  const int t1 = min(T, (sg + 1) * SC_SEG);
  for (int t = sg * SC_SEG; t < t1; ++t) {
    const float xt = ld(x, (size_t)t * C + c);
    win[K - 1] = xt;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s = fmaf(wk[k], win[k], s);
    x[(size_t)t * C + c] = __float2bfloat16_rn((s + xt) / div);
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
}

extern "C" __global__ void ink2_sconv_dynamic_seg(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ static_w,
                                                  const float* __restrict__ dyn, const float* __restrict__ edges, int T,
                                                  int C, int K, int D) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int segs = (T + SC_SEG - 1) / SC_SEG;
  if (i >= (size_t)segs * C) return;
  const int sg = (int)(i / C), c = (int)(i - (size_t)sg * C);
  const int head = c / D, heads = C / D;
  float win[8];
  for (int k = 0; k < K - 1; ++k) win[k] = edges[((size_t)sg * (K - 1) + k) * C + c];
  const float div = sqrtf((float)(2 * K + 1));
  const int t1 = min(T, (sg + 1) * SC_SEG);
  for (int t = sg * SC_SEG; t < t1; ++t) {
    const float xt = ld(x, (size_t)t * C + c);
    win[K - 1] = xt;
    float s = 0.f;
    for (int lag = 0; lag < K; ++lag) {
      const float tap = ld(static_w, (size_t)lag * C + c) + dyn[((size_t)t * heads + head) * K + lag];
      s = fmaf(tap, win[K - 1 - lag], s);
    }
    x[(size_t)t * C + c] = __float2bfloat16_rn((s + xt) / div);
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
}

// ------------------------------------------------------------------------------------------------- q / k / v / rel
// One block a (row, head) of `v` [T][heads][D] bf16, in place:
//   mode 0  Inkling2ScaledL2Clip (M:150-162): x fp32 times scale[head] (bf16 -> fp32), scaled down to L2 norm thr
//           when longer (thr / (norm + 1e-6), clamped at 1), to bf16; then, with tau_on, times tau(pos) in fp32 and
//           to bf16 again (global layers' log scaling, M:589-596 / M:601-602)
//   mode 1  Inkling2L2Clip (M:136-147): the same without a scale (the keys)
//   mode 2  Inkling2UnweightedRMSNorm (M:127-133): x * rsqrt(mean(x^2) + eps), fp32, to bf16 (the values)
// tau(pos) = 1 + alpha log(max((pos + 1) / n_floor, 1)), pos = pos0 + row.
extern "C" __global__ void ink2_head_prep(__nv_bfloat16* v, const __nv_bfloat16* __restrict__ scale, int heads, int D,
                                          int mode, float thr, float eps, int tau_on, int pos0, float alpha,
                                          float n_floor) {
  const int row = blockIdx.x / heads, h = blockIdx.x - row * heads;
  __nv_bfloat16* p = v + ((size_t)row * heads + h) * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float x = __bfloat162float(p[i]);
    if (mode == 0) x *= ld(scale, (size_t)h * D + i);
    ss += x * x;
  }
  const float tot = block_sum(ss);
  float f;
  if (mode == 2) {
    f = rsqrtf(tot / (float)D + eps);
  } else {
    f = fminf(thr / (sqrtf(tot) + 1e-6f), 1.0f);
  }
  float tau = 1.f;
  if (tau_on) tau = 1.f + alpha * logf(fmaxf((float)(pos0 + row + 1) / n_floor, 1.0f));
  __syncthreads();
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float x = __bfloat162float(p[i]);
    if (mode == 0) x *= ld(scale, (size_t)h * D + i);
    float o = bf(x * f);
    if (tau_on) o = bf(o * tau);
    p[i] = __float2bfloat16_rn(o);
  }
}

// ------------------------------------------------------------------------------------------------- attention
// eager_attention_forward (M:368-399) with the relative bias, the sink and the output gate (M:475-497), one block a
// (query row, query head). Keys and values live in a cache of `slots` positions per kv head, position p at slot
// p % slots (a sliding layer's ring of window + chunk rows; a global layer's whole window), [slot][kv heads][D]
// bf16, the chunk's rows already written. Row i (absolute position q0 + i) sees positions p <= q0 + i, and on a
// sliding layer (window > 0) p > q0 + i - window. Scores at the reference's points: s = bf16(q . k) (the bf16
// matmul), bf16(s * scaling), bf16(s + bias) with bias = rel[i][h][q0 + i - p] (bf16; 0 at or past the extent);
// softmax over [s..., sink] in fp32; p = bf16(prob); o = bf16(sum p v) (the bf16 matmul); then
// bf16(o * sigmoid(gate)) in fp32 (M:496). scores: a caller scratch of `cap` floats a block.
// Launched with 128 threads (the exp sum's partials are per thread: its bits depend on it). A thread's key dot is
// taken in d order from 16-byte loads (q staged in shared memory as fp32); each position's probability is made once
// into the scratch, then every output dim sums them in position order.
extern "C" __global__ void __launch_bounds__(128) ink2_attention(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
    const __nv_bfloat16* __restrict__ rel, const __nv_bfloat16* __restrict__ sinks,
    const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, float* __restrict__ scores, int heads,
    int kv_heads, int D, int q0, int window, int slots, int extent, float scaling, int cap) {
  const int row = blockIdx.x / heads, h = blockIdx.x - row * heads;
  const int g = h / (heads / kv_heads);
  const int pos = q0 + row;
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  const int n = pos - lo + 1;
  float* sc = scores + (size_t)blockIdx.x * cap;
  const __nv_bfloat16* qr = q + ((size_t)row * heads + h) * D;
  __shared__ float qs[256];
  const bool vec = D <= 256 && (D & 7) == 0;
  if (vec)
    for (int d = threadIdx.x; d < D; d += blockDim.x) qs[d] = __bfloat162float(qr[d]);
  __syncthreads();
  // pass 1: the scores, their max (the sink included)
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const int p = lo + j;
    const __nv_bfloat16* kr = kc + ((size_t)(p % slots) * kv_heads + g) * D;
    float dot = 0.f;
    if (vec) {
      const uint4* k4 = reinterpret_cast<const uint4*>(kr);
      for (int d8 = 0; d8 < D / 8; d8 += 4) {
        uint4 kv[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) kv[u] = d8 + u < D / 8 ? __ldg(k4 + d8 + u) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          if (d8 + u >= D / 8) break;
          const uint32_t wv[4] = {kv[u].x, kv[u].y, kv[u].z, kv[u].w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int d = 8 * (d8 + u) + 2 * c;
            dot = fmaf(qs[d], __uint_as_float(wv[c] << 16), dot);
            dot = fmaf(qs[d + 1], __uint_as_float(wv[c] & 0xFFFF0000u), dot);
          }
        }
      }
    } else {
      for (int d = 0; d < D; ++d) dot = fmaf(__bfloat162float(qr[d]), __bfloat162float(kr[d]), dot);
    }
    float s = bf(bf(dot) * scaling);
    const int dist = pos - p;
    const float bias = dist < extent ? ld(rel, ((size_t)row * heads + h) * extent + dist) : 0.f;
    s = bf(s + bias);
    sc[j] = s;
    mx = fmaxf(mx, s);
  }
  const float sink = ld(sinks, (size_t)row * heads + h);
  mx = fmaxf(block_max(mx), sink);
  float se = 0.f;
  for (int j = threadIdx.x; j < n; j += blockDim.x) se += expf(sc[j] - mx);
  const float sum = block_sum(se) + expf(sink - mx);
  // the probabilities, once (each thread rewrites the scores it wrote)
  for (int j = threadIdx.x; j < n; j += blockDim.x) sc[j] = bf(expf(sc[j] - mx) / sum);
  __syncthreads();
  // pass 2: each thread owns output dims, the positions summed in order
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    int j = 0;
    for (; j + 7 < n; j += 8) {
      float pv[8], vv[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        pv[u] = sc[j + u];
        vv[u] = ld(vc, ((size_t)((lo + j + u) % slots) * kv_heads + g) * D + d);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u) acc = fmaf(pv[u], vv[u], acc);
    }
    for (; j < n; ++j) acc = fmaf(sc[j], ld(vc, ((size_t)((lo + j) % slots) * kv_heads + g) * D + d), acc);
    const float o = bf(acc);
    const size_t at = ((size_t)row * heads + h) * D + d;
    const float gt = ld(gate, at);
    out[at] = __float2bfloat16_rn(o * (1.f / (1.f + expf(-gt))));
  }
}

// ink2_attention for a prompt chunk's rows, one block a (query row, kv head): the kv group's heads / kv_heads query
// heads, 128 threads each, share every tile of 128 keys (then of their values) staged once in shared memory, where
// ink2_attention's blocks each read every key and value of the row from memory themselves. A head's arithmetic is
// ink2_attention's thread for thread: its thread u takes keys j = u, u + 128, ... (each dot in d order from the same
// 16-byte groups, the scores at the same rounding points, its exp partial sum in j order), the max and the sum reduce
// over its own four warps exactly as block_max / block_sum do over a 128-thread block, and output dim d = u (and
// u + 128) sums the positions in order. So every output is ink2_attention's, bit for bit. D a multiple of 8, at most
// 128; heads / kv_heads at most 8. Dynamic shared memory: attn_gqa_smem(D, groups) bytes. scores: as ink2_attention's
// (the (row, head)'s `cap` floats at ((row * heads) + head) * cap).
namespace {
constexpr int AG_KEYS = 128;  // keys a tile: one a thread of a head

// block_max / block_sum on one head's 128 threads (u its thread, `part` its four floats): the same shuffles and
// the same second stage (lanes 0 .. 3 the warps' totals, the rest -inf / 0), so the same bits. Every thread of the
// block calls them together (they hold block barriers).
__device__ float group_max(float v, float* part, int u) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  const int w = u >> 5, l = u & 31;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  float s = l < 4 ? part[l] : -INFINITY;
  for (int o = 16; o > 0; o >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, o));
  return s;
}

__device__ float group_sum(float v, float* part, int u) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = u >> 5, l = u & 31;
  __syncthreads();
  if (l == 0) part[w] = v;
  __syncthreads();
  float s = l < 4 ? part[l] : 0.f;
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  return s;
}
}  // namespace

extern "C" __global__ void __launch_bounds__(1024) ink2_attention_gqa(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ vc,
    const __nv_bfloat16* __restrict__ rel, const __nv_bfloat16* __restrict__ sinks,
    const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, float* __restrict__ scores, int heads,
    int kv_heads, int D, int q0, int window, int slots, int extent, float scaling, int cap) {
  extern __shared__ uint4 ag_smem[];
  const int G = heads / kv_heads;
  const int row = blockIdx.x / kv_heads, g = blockIdx.x - row * kv_heads;
  const int hg = threadIdx.x >> 7, u = threadIdx.x & 127;
  const int h = g * G + hg;
  const int pos = q0 + row;
  const int lo = window > 0 ? max(0, pos - window + 1) : 0;
  const int n = pos - lo + 1;
  const int D8 = D / 8, RW = D8 + 1;  // a tile row: the key's D8 16-byte groups and one of padding (bank spread)
  uint4* tile = ag_smem;                                          // [AG_KEYS][RW]
  float* qs = reinterpret_cast<float*>(tile + AG_KEYS * RW);      // [G][D] the heads' queries in fp32
  float* pt = qs + G * D;                                         // [G][AG_KEYS] a tile's probabilities
  float* part = pt + G * AG_KEYS;                                 // [G][4] the reductions' second stage
  float* sc = scores + ((size_t)row * heads + h) * cap;
  const __nv_bfloat16* qr = q + ((size_t)row * heads + h) * D;
  for (int d = u; d < D; d += 128) qs[hg * D + d] = __bfloat162float(qr[d]);
  const float* qh = qs + hg * D;
  // pass 1: the scores, their max (the sink included)
  float mx = -INFINITY;
  for (int t0 = 0; t0 < n; t0 += AG_KEYS) {
    const int nt = min(AG_KEYS, n - t0);
    __syncthreads();  // the previous tile is read (and, the first time, the queries are staged)
    for (int i = threadIdx.x; i < nt * D8; i += blockDim.x) {
      const int kk = i / D8, c = i - kk * D8;
      tile[kk * RW + c] = __ldg(reinterpret_cast<const uint4*>(kc + ((size_t)((lo + t0 + kk) % slots) * kv_heads + g) * D) + c);
    }
    __syncthreads();
    if (u < nt) {
      const int j = t0 + u, p = lo + j;
      const uint4* k4 = tile + u * RW;
      float dot = 0.f;
      for (int d8 = 0; d8 < D8; d8 += 4) {
        uint4 kv[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) kv[w] = d8 + w < D8 ? k4[d8 + w] : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          if (d8 + w >= D8) break;
          const uint32_t wv[4] = {kv[w].x, kv[w].y, kv[w].z, kv[w].w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int d = 8 * (d8 + w) + 2 * c;
            dot = fmaf(qh[d], __uint_as_float(wv[c] << 16), dot);
            dot = fmaf(qh[d + 1], __uint_as_float(wv[c] & 0xFFFF0000u), dot);
          }
        }
      }
      float s = bf(bf(dot) * scaling);
      const int dist = pos - p;
      const float bias = dist < extent ? ld(rel, ((size_t)row * heads + h) * extent + dist) : 0.f;
      s = bf(s + bias);
      sc[j] = s;
      mx = fmaxf(mx, s);
    }
  }
  const float sink = ld(sinks, (size_t)row * heads + h);
  mx = fmaxf(group_max(mx, part + hg * 4, u), sink);
  float se = 0.f;
  for (int j = u; j < n; j += 128) se += expf(sc[j] - mx);
  const float sum = group_sum(se, part + hg * 4, u) + expf(sink - mx);
  // the probabilities, once (each thread rewrites the scores it wrote; it alone reads them back below)
  for (int j = u; j < n; j += 128) sc[j] = bf(expf(sc[j] - mx) / sum);
  // pass 2: output dims u and u + 128, the positions summed in order
  float acc0 = 0.f, acc1 = 0.f;
  const __nv_bfloat16* vt = reinterpret_cast<const __nv_bfloat16*>(tile);
  const float* ph = pt + hg * AG_KEYS;
  for (int t0 = 0; t0 < n; t0 += AG_KEYS) {
    const int nt = min(AG_KEYS, n - t0);
    __syncthreads();  // the previous tile is read
    for (int i = threadIdx.x; i < nt * D8; i += blockDim.x) {
      const int kk = i / D8, c = i - kk * D8;
      tile[kk * RW + c] = __ldg(reinterpret_cast<const uint4*>(vc + ((size_t)((lo + t0 + kk) % slots) * kv_heads + g) * D) + c);
    }
    if (u < nt) pt[hg * AG_KEYS + u] = sc[t0 + u];
    __syncthreads();
    if (u < D)
      for (int kk = 0; kk < nt; ++kk) acc0 = fmaf(ph[kk], __bfloat162float(vt[kk * RW * 8 + u]), acc0);
    if (u + 128 < D)
      for (int kk = 0; kk < nt; ++kk) acc1 = fmaf(ph[kk], __bfloat162float(vt[kk * RW * 8 + u + 128]), acc1);
  }
  for (int d = u, r = 0; d < D; d += 128, ++r) {
    const float o = bf(r == 0 ? acc0 : acc1);
    const size_t at = ((size_t)row * heads + h) * D + d;
    const float gt = ld(gate, at);
    out[at] = __float2bfloat16_rn(o * (1.f / (1.f + expf(-gt))));
  }
}

// Writes a chunk's keys and values into a cache of `slots` positions (position p at slot p % slots).
extern "C" __global__ void ink2_cache_write(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                                            __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc, int T,
                                            int width, int q0, int slots) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * width) return;
  const int t = (int)(i / width), c = (int)(i - (size_t)t * width);
  const size_t at = (size_t)((q0 + t) % slots) * width + c;
  kc[at] = k[i];
  vc[at] = v[i];
}

// ------------------------------------------------------------------------------------------------- attn_res
// Inkling2BlockAttentionResidual (M:733-751), one block a (row, query): sources = the row's S block states (bf16
// [T][Smax][H]) then its partial state (fp32 or bf16 [T][H]), each RMS-normed without a gamma (eps 1e-6, fp32), as
// keys and values; scores = query . source * H^-0.5 (fp32), softmax, out [nq][T][H] fp32 = sum p source.
// The sums are those of one 256-thread block (thread u's strided partials over i = u, u + 256, ..., then the block
// reduction ink2's block_sum does); the block runs 1024 threads, four a partial thread, each taking every fourth
// source, so all sources' partials are in flight at once and reduced together. S < 64.
namespace {
constexpr int AR_U = 256;  // the partial threads whose sums define the bits

template <bool F32>
__device__ __forceinline__ float ar_x(const __nv_bfloat16* blocks, const void* partial, int S, int Smax, int H,
                                      size_t row, int s, int i) {
  if (s < S) return ld(blocks, (row * Smax + s) * H + i);
  if constexpr (F32) return reinterpret_cast<const float*>(partial)[row * H + i];
  return ld(reinterpret_cast<const __nv_bfloat16*>(partial), row * H + i);
}

template <bool F32>
__device__ __forceinline__ void attn_res_body(const __nv_bfloat16* __restrict__ blocks, int S, int Smax,
                                              const void* __restrict__ partial, const float* __restrict__ query,
                                              int nq, float* __restrict__ out, int H) {
  const size_t row = blockIdx.x / nq;
  const int qi = blockIdx.x - (int)row * nq;
  const int u = threadIdx.x & (AR_U - 1), sg = threadIdx.x / AR_U, groups = blockDim.x / AR_U;
  const int lane = threadIdx.x & 31, wu = u >> 5;  // wu: u's warp among the 8 of a 256-thread block
  __shared__ float red[64][2][AR_U / 32];
  __shared__ float inv[64];
  __shared__ float prob[64];
  const float* qv = query + (size_t)qi * H;
  for (int s = sg; s <= S; s += groups) {
    float ss = 0.f, dot = 0.f;
    int i = u;
    for (; i + 3 * AR_U < H; i += 4 * AR_U) {  // four loads in flight, summed in u's order
      float x[4];
      float q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        x[j] = ar_x<F32>(blocks, partial, S, Smax, H, row, s, i + j * AR_U);
        q[j] = qv[i + j * AR_U];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ss += x[j] * x[j];
        dot += q[j] * x[j];
      }
    }
    for (; i < H; i += AR_U) {
      const float x = ar_x<F32>(blocks, partial, S, Smax, H, row, s, i);
      ss += x * x;
      dot += qv[i] * x;
    }
    for (int o = 16; o > 0; o >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, o);
      dot += __shfl_xor_sync(0xffffffffu, dot, o);
    }
    if (lane == 0) {
      red[s][0][wu] = ss;
      red[s][1][wu] = dot;
    }
  }
  __syncthreads();
  // block_sum's second stage: lanes l < 8 hold the warps' sums, the rest zero, reduced over the warp
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int v = w; v < 2 * (S + 1); v += nw) {
    const int s = v >> 1, which = v & 1;
    float a = lane < AR_U / 32 ? red[s][which][lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    __syncwarp();
    if (lane == 0) red[s][which][0] = a;
  }
  __syncthreads();
  if (threadIdx.x <= S) {
    const int s = threadIdx.x;
    inv[s] = rsqrtf(red[s][0][0] / (float)H + 1e-6f);
    prob[s] = red[s][1][0] * inv[s] * rsqrtf((float)H);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = 0; s <= S; ++s) m = fmaxf(m, prob[s]);
    float z = 0.f;
    for (int s = 0; s <= S; ++s) z += expf(prob[s] - m);
    for (int s = 0; s <= S; ++s) prob[s] = expf(prob[s] - m) / z;
  }
  __syncthreads();
  const int rows = gridDim.x / nq;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float acc = 0.f;
    for (int s = 0; s <= S; ++s) acc = fmaf(prob[s], ar_x<F32>(blocks, partial, S, Smax, H, row, s, i) * inv[s], acc);
    out[((size_t)qi * rows + row) * H + i] = acc;  // [nq][rows][H]: each query's rows contiguous
  }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(1024) ink2_attn_res(const __nv_bfloat16* __restrict__ blocks, int S,
                                                                 int Smax, const void* __restrict__ partial,
                                                                 int partial_f32, const float* __restrict__ query,
                                                                 int nq, float* __restrict__ out, int H) {
  if (partial_f32)
    attn_res_body<true>(blocks, S, Smax, partial, query, nq, out, H);
  else
    attn_res_body<false>(blocks, S, Smax, partial, query, nq, out, H);
}

// Appends the rows' layer input (bf16 [T][H]) as block state `s` of [T][Smax][H].
extern "C" __global__ void ink2_block_push(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ blocks,
                                           int T, int H, int s, int Smax) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * H) return;
  const size_t t = i / H, c = i - t * H;
  blocks[(t * Smax + s) * H + c] = x[i];
}

// ------------------------------------------------------------------------------------------------- residual stream
// After the attention's post-convolution (M:798-802): residual = h.float(), plus the layer input when the layer is not
// a block boundary. h [n] bf16, input [n] bf16, residual [n] fp32.
extern "C" __global__ void ink2_residual(const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ input,
                                         float* __restrict__ residual, int boundary, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = __bfloat162float(h[i]);
  residual[i] = boundary ? a : __bfloat162float(input[i]) + a;
}

// The layer's output (M:807): (residual + mlp.float()).to(bf16).
extern "C" __global__ void ink2_layer_out(const float* __restrict__ residual, const __nv_bfloat16* __restrict__ mlp,
                                          __nv_bfloat16* __restrict__ out, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = __float2bfloat16_rn(residual[i] + __bfloat162float(mlp[i]));
}

extern "C" __global__ void ink2_bf16_to_f32(const __nv_bfloat16* __restrict__ x, float* __restrict__ y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __bfloat162float(x[i]);
}

extern "C" __global__ void ink2_f32_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

// TP: parts [world][n] (an all-gather's rows, rank order) summed in rank order on every rank. f32: fp32 parts and
// fp32 out (an fp32 partial's all-reduce); else bf16 parts summed in fp32 and rounded once (exact for the embedding,
// where one part is non-zero).
extern "C" __global__ void ink2_rank_sum(const void* __restrict__ parts, void* __restrict__ out, int world, size_t n,
                                         int f32) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (f32) {
    const float* p = reinterpret_cast<const float*>(parts);
    float s = p[i];
    for (int r = 1; r < world; ++r) s = s + p[(size_t)r * n + i];
    reinterpret_cast<float*>(out)[i] = s;
  } else {
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(parts);
    float s = __bfloat162float(p[i]);
    for (int r = 1; r < world; ++r) s = s + __bfloat162float(p[(size_t)r * n + i]);
    reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(s);
  }
}

// TP, o_proj: fp32 parts [world][n] summed in rank order (ink2_rank_sum's f32 form) and rounded once to bf16
// (ink2_f32_to_bf16): the two launches in one, the same bits.
extern "C" __global__ void ink2_rank_sum_bf16(const float* __restrict__ parts, __nv_bfloat16* __restrict__ out,
                                              int world, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = parts[i];
  for (int r = 1; r < world; ++r) s = s + parts[(size_t)r * n + i];
  out[i] = __float2bfloat16_rn(s);
}

// ink2_rank_sum_bf16 at two ranks, each rank's fp32 partials in a buffer of its own (a wide prompt chunk's row half):
// out = bf16(y0 + y1), rank 0's first -- that kernel's add and rounding, the same bits.
extern "C" __global__ void ink2_rank_sum_pair_bf16(const float* __restrict__ y0, const float* __restrict__ y1,
                                                   __nv_bfloat16* __restrict__ out, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float s = y0[i];
  s = s + y1[i];
  out[i] = __float2bfloat16_rn(s);
}

// ------------------------------------------------------------------------------------------------- embedding, head
// The vocabulary-split embedding rows: ids in [lo, hi) take their row of this rank's table, the rest zeros.
extern "C" __global__ void ink2_embed(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ table,
                                      __nv_bfloat16* __restrict__ out, int T, int H, int lo, int hi) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * H) return;
  const int t = (int)(i / H), c = (int)(i - (size_t)t * H);
  const int id = ids[t];
  out[i] = (id >= lo && id < hi) ? table[(size_t)(id - lo) * H + c] : __float2bfloat16_rn(0.f);
}

// The logits (M:1388-1390): bf16 head rows .float(), softcap c tanh(l / c), the first `unpadded` columns; per row the
// first index of the maximum (torch.argmax). logits [T][V] bf16 in global vocabulary order; out_f32 [T][unpadded]
// (may be null); argmax [T] int32. One block a row (1024 threads); a thread's columns i = tid + 1024 m are read eight
// at a time and taken in order.
extern "C" __global__ void ink2_logits(const __nv_bfloat16* __restrict__ logits, int V, int unpadded, float cap,
                                       float* __restrict__ out_f32, int* __restrict__ argmax) {
  const size_t r = blockIdx.x;
  const int B = blockDim.x;
  float best = -INFINITY;
  int bi = 0x7fffffff;
  for (int i0 = threadIdx.x; i0 < unpadded; i0 += 8 * B) {
    float lv[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = i0 + u * B;
      lv[u] = i < unpadded ? ld(logits, r * V + i) : 0.f;
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = i0 + u * B;
      if (i >= unpadded) break;
      const float l = tanhf(lv[u] / cap) * cap;
      if (out_f32) out_f32[r * unpadded + i] = l;
      if (l > best || (l == best && i < bi)) {
        best = l;
        bi = i;
      }
    }
  }
  // (value, index) under "greater value, else lower index": a total order (best is never NaN), so the tree below
  // finds the one-thread scan's pick
  auto take = [](float& b, int& x, float ob, int ox) {
    if (ob > b || (ob == b && ox < x)) {
      b = ob;
      x = ox;
    }
  };
  for (int o = 16; o > 0; o >>= 1) take(best, bi, __shfl_xor_sync(0xffffffffu, best, o), __shfl_xor_sync(0xffffffffu, bi, o));
  __shared__ float bv[32];
  __shared__ int bx[32];
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) {
    bv[w] = best;
    bx[w] = bi;
  }
  __syncthreads();
  if (w == 0) {
    float b = l < (B >> 5) ? bv[l] : -INFINITY;
    int x = l < (B >> 5) ? bx[l] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) take(b, x, __shfl_xor_sync(0xffffffffu, b, o), __shfl_xor_sync(0xffffffffu, x, o));
    if (l == 0) argmax[r] = x;
  }
}

// A batched head's gathered parts [world][T][Vl] into rows [T][world * Vl] (rank r's vocabulary rows at r * Vl).
extern "C" __global__ void ink2_unsplit(const __nv_bfloat16* __restrict__ parts, __nv_bfloat16* __restrict__ rows, int T,
                                        int Vl, int world) {
  const size_t t = blockIdx.y, r = blockIdx.z;
  const __nv_bfloat16* src = parts + (r * T + t) * Vl;
  __nv_bfloat16* dst = rows + t * Vl * world + r * Vl;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Vl; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

// ------------------------------------------------------------------------------------------------- MoE
// Inkling2TopKRouter (M:696-716) with residency (TideLM's masked router): logits fp32 [T][E] (fp32 GEMM),
// scores = sigmoid(logits); the pick is the top-k of scores + bias over the ROUTABLE experts (routable[l][e] != 0;
// a missing expert's place goes to the next-best routable one: full substitution); weights = the picked scores
// renormalized (+1e-20) times route_scale. Demand: the unmasked top-k ids and scores, what the router wanted.
// picks / weights [T][k] in pick order; want_ids / want_scores [T][k] (may be null). One block of 64 a row: warp 0
// the masked pick, warp 1 the unmasked demand; each takes k rounds of a warp argmax over the experts (lane l holds
// experts l, l + 32, ...), descending key, the lower expert id first on a tie -- the order the one-thread insertion
// it replaces kept (equal keys never displace an earlier expert), so picks, weights and wants are its bits (for every
// finite logit). E <= 1024, k <= 16.
extern "C" __global__ void ink2_route(const float* __restrict__ logits, const float* __restrict__ bias,
                                      const uint8_t* __restrict__ routable, int T, int E, int k, float route_scale,
                                      int* __restrict__ picks, float* __restrict__ weights, int* __restrict__ want_ids,
                                      float* __restrict__ want_scores) {
  constexpr int PER = 32;  // experts a lane at most
  constexpr int NONE = 0x7fffffff;
  const int t = blockIdx.x, pass = threadIdx.x >> 5, lane = threadIdx.x & 31;  // 0: masked pick, 1: unmasked demand
  if (t >= T || pass > 1) return;
  if (pass == 1 && want_ids == nullptr) return;
  const float* lg = logits + (size_t)t * E;
  float key[PER];
  uint32_t live = 0;  // bit m: expert lane + 32 m is a candidate not yet picked
#pragma unroll
  for (int m = 0; m < PER; ++m) {
    const int e = lane + 32 * m;
    key[m] = 0.f;
    if (e < E && (pass == 1 || routable == nullptr || routable[e] != 0)) {
      const float s = 1.f / (1.f + expf(-lg[e]));
      key[m] = s + bias[e];
      live |= 1u << m;
    }
  }
  int ids[16];
  int n = 0;
  for (int r = 0; r < k; ++r) {
    float bv = 0.f;
    int be = NONE, bm = -1;
#pragma unroll
    for (int m = 0; m < PER; ++m)  // ascending e: a later expert takes the lead only on a strictly greater key
      if (((live >> m) & 1u) && (be == NONE || key[m] > bv)) {
        bv = key[m];
        be = lane + 32 * m;
        bm = m;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (oe != NONE && (be == NONE || ov > bv || (ov == bv && oe < be))) {
        bv = ov;
        be = oe;
      }
    }
    if (be == NONE) break;  // fewer candidates than k
    if (be == lane + 32 * bm) live &= ~(1u << bm);
    ids[r] = be;
    n = r + 1;
  }
  float sum = 0.f;
  float sc[16];
  for (int j = 0; j < n; ++j) {
    sc[j] = 1.f / (1.f + expf(-lg[ids[j]]));
    sum += sc[j];
  }
  if (lane != 0) return;
  for (int j = 0; j < k; ++j) {
    if (pass == 0) {
      picks[(size_t)t * k + j] = j < n ? ids[j] : -1;
      weights[(size_t)t * k + j] = j < n ? sc[j] / (sum + 1e-20f) * route_scale : 0.f;
    } else {
      want_ids[(size_t)t * k + j] = ids[j];
      want_scores[(size_t)t * k + j] = sc[j];
    }
  }
}

// Inkling2Experts's sum (M:672-689) over (row, slot) pairs whose down outputs y [T][k][H] are fp32 (after the TP
// all-reduce): each pair rounded to bf16 (the reference's down output), times its routing weight in fp32, added in
// ascending expert id from zero (the reference's index_add_ order), the row rounded to bf16. Block (row, chunk):
// gridDim.y chunks of the row's H outputs (each output's sum is its own; the split never moves a bit); each block
// reads the row's k picks and weights once and sorts the slot order (k <= 16).
extern "C" __global__ void ink2_combine(const float* __restrict__ y, const int* __restrict__ picks,
                                        const float* __restrict__ weights, __nv_bfloat16* __restrict__ out, int k,
                                        int H) {
  const size_t t = blockIdx.x;
  __shared__ int s_order[16];
  __shared__ float s_w[16];
  __shared__ int s_n;
  if (threadIdx.x == 0) {
    int pk[16], order[16];
    for (int j = 0; j < k; ++j) {
      pk[j] = picks[t * k + j];
      order[j] = j;
    }
    for (int a = 1; a < k; ++a) {  // ascending expert id, stable
      const int v = order[a];
      int b = a;
      while (b > 0 && pk[order[b - 1]] > pk[v]) {
        order[b] = order[b - 1];
        --b;
      }
      order[b] = v;
    }
    int n = 0;
    for (int j = 0; j < k; ++j)
      if (pk[order[j]] >= 0) {
        s_order[n] = order[j];
        s_w[n] = weights[t * k + order[j]];
        ++n;
      }
    s_n = n;
  }
  __syncthreads();
  const int n = s_n;
  for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < H; i += gridDim.y * blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc = acc + bf(y[(t * k + s_order[j]) * H + i]) * s_w[j];
    out[t * H + i] = __float2bfloat16_rn(acc);
  }
}

// ink2_combine on the TP all-gather's rows y [world][T][k][H] (n = T k H a rank): each pair's fp32 parts summed in rank
// order first (ink2_rank_sum's f32 form), then ink2_combine's arithmetic per element: the rank-sum launch folded in,
// the same bits. A block a (row, blockDim columns) rather than a row (one block a decode row starves the GPU).
extern "C" __global__ void ink2_combine_parts(const float* __restrict__ y, const int* __restrict__ picks,
                                              const float* __restrict__ weights, __nv_bfloat16* __restrict__ out,
                                              int k, int H, int world, size_t n) {
  const int chunks = (H + blockDim.x - 1) / blockDim.x;
  const size_t t = blockIdx.x / chunks;
  const int i = (blockIdx.x % chunks) * blockDim.x + threadIdx.x;
  if (i >= H) return;
  int order[16];
  for (int j = 0; j < k; ++j) order[j] = j;
  for (int a = 1; a < k; ++a) {  // ascending expert id, stable
    const int v = order[a];
    int b = a;
    while (b > 0 && picks[t * k + order[b - 1]] > picks[t * k + v]) {
      order[b] = order[b - 1];
      --b;
    }
    order[b] = v;
  }
  float acc = 0.f;
  for (int j = 0; j < k; ++j) {
    const int s = order[j];
    if (picks[t * k + s] < 0) continue;
    const size_t at = (t * k + s) * H + i;
    float v = y[at];
    for (int r = 1; r < world; ++r) v = v + y[(size_t)r * n + at];
    acc = acc + bf(v) * weights[t * k + s];
  }
  out[t * H + i] = __float2bfloat16_rn(acc);
}

// ink2_combine_parts at two ranks, each rank's partials in a buffer of its own: a wide prompt chunk's rows split
// between the ranks, y0 rank 0's fp32 partials [t][k][H] of this rank's rows and y1 rank 1's; each pair's v = y0 + y1
// (rank order: the add ink2_combine_parts's world-2 loop makes), then ink2_combine's arithmetic, written the same way.
// picks / weights / out start at the rows' first. A block a (row, blockDim columns), as ink2_combine_parts.
extern "C" __global__ void ink2_combine_pair(const float* __restrict__ y0, const float* __restrict__ y1,
                                             const int* __restrict__ picks, const float* __restrict__ weights,
                                             __nv_bfloat16* __restrict__ out, int k, int H) {
  const int chunks = (H + blockDim.x - 1) / blockDim.x;
  const size_t t = blockIdx.x / chunks;
  const int i = (blockIdx.x % chunks) * blockDim.x + threadIdx.x;
  if (i >= H) return;
  int order[16];
  for (int j = 0; j < k; ++j) order[j] = j;
  for (int a = 1; a < k; ++a) {  // ascending expert id, stable
    const int v = order[a];
    int b = a;
    while (b > 0 && picks[t * k + order[b - 1]] > picks[t * k + v]) {
      order[b] = order[b - 1];
      --b;
    }
    order[b] = v;
  }
  float acc = 0.f;
  for (int j = 0; j < k; ++j) {
    const int s = order[j];
    if (picks[t * k + s] < 0) continue;
    const size_t at = (t * k + s) * H + i;
    float v = y0[at];
    v = v + y1[at];
    acc = acc + bf(v) * weights[t * k + s];
  }
  out[t * H + i] = __float2bfloat16_rn(acc);
}

// Expert ids to arena slots through the layer's LUT (-1 never routed: the masked pick only names routable experts).
extern "C" __global__ void ink2_slot_map(const int* __restrict__ picks, const int* __restrict__ lut, int* __restrict__ slots,
                                         int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) slots[i] = picks[i] < 0 ? -1 : lut[picks[i]];
}

// Residency (TideLM's slot LUT): the grouped plan is made over expert ids (at most 1,024, the plan's limit); each
// item's expert id is then replaced by its arena slot, lut[e], which the expert kernels index their weights and
// global scales by. counts[0]: the plan's item count. A swap rewrites lut entries in place: graphs survive it.
extern "C" __global__ void ink2_items_to_slots(int* __restrict__ items, const int* __restrict__ counts,
                                               const int* __restrict__ lut) {
  const int n = counts[0];
  for (int it = threadIdx.x; it < n; it += blockDim.x) {
    const int e = items[3 * it];
    items[3 * it] = e < 0 ? e : lut[e];
  }
}

// TideLM demand (adaptive-experts.md, score metric): every row's unmasked top-k wants, per expert of the layer:
// demand[e][0] += 1 (picks), [1] += its sigmoid score, [2] += 1 when the masked router could not take it (the
// expert was not routable: a substituted pick). fp32 [E][3]; counts stay exact below 2^24.
extern "C" __global__ void ink2_demand(const int* __restrict__ want_ids, const float* __restrict__ want_scores,
                                       const uint8_t* __restrict__ routable, int n, float* __restrict__ demand) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int e = want_ids[i];
  if (e < 0) return;
  atomicAdd(demand + 3 * e, 1.f);
  atomicAdd(demand + 3 * e + 1, want_scores[i]);
  if (routable != nullptr && routable[e] == 0) atomicAdd(demand + 3 * e + 2, 1.f);
}

// ------------------------------------------------------------------------------------------------- expert packing
// The fill's packers on the device (fill.zig's GPU path): one expert's stored byte ranges, read past the page cache
// into mapped pinned memory, permuted into its arena slot. Nothing is decoded or rounded: each output word is the
// byte (or nibble) permutation of experts.zig / core Nvfp4Experts.packPart (NVFP4) and experts.zig packMx (MXFP8),
// and every byte of this rank's cut is read exactly once, so the host packers' refusals are kept as bits of *bad:
// 1 an NVFP4 e4m3 block scale NaN or negative, 2 an MXFP8 e4m3 code NaN, 4 an MXFP8 UE8M0 exponent 0xFF.
// w13 / s13: this rank's 2 Il stored rows (gate even, up odd); w2 / s2: the expert's whole rows (the rank's columns
// [rank Il, (rank + 1) Il) are cut here). Grid: 3 matrices (gate, up, down) x (Il / 32) (H / 32) blocks.

// NVFP4: a block of 144 words, 128 code words (gq, t, j) then 16 words of e4m3 scale bytes (t, h, j, c).
extern "C" __global__ void ink2_pack_nv(uint32_t* __restrict__ up, uint32_t* __restrict__ down,
                                        const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13,
                                        const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2, int H, int Il,
                                        int I, int rank, float* __restrict__ up_scale, float* __restrict__ down_scale,
                                        float g13, float g2, unsigned* __restrict__ bad) {
  const int nb = (Il / 32) * (H / 32);
  const int mat = blockIdx.x / nb, b = blockIdx.x % nb;
  const int tid = threadIdx.x;
  if (tid >= 144) return;
  const bool dn = mat == 2;
  const int kg = dn ? Il / 32 : H / 32;
  const int cb = b / kg, g = b % kg;
  const size_t crow = dn ? (size_t)(I / 2) : (size_t)2 * (H / 2);  // a stored row's code bytes (w13: 2 rows a channel)
  const size_t srow = dn ? (size_t)(I / 16) : (size_t)2 * (H / 16);
  const uint8_t* codes = dn ? w2 + (size_t)rank * (Il / 2) : w13 + (size_t)mat * (H / 2);
  const uint8_t* scales = dn ? s2 + (size_t)rank * (Il / 16) : s13 + (size_t)mat * (H / 16);
  uint32_t* out = dn ? down + (size_t)b * 144 : up + ((size_t)b * 2 + mat) * 144;
  if (blockIdx.x == 0 && tid == 0) {
    up_scale[0] = g13;
    up_scale[1] = g13;
    down_scale[0] = g2;
  }
  if (tid < 128) {
    const int j = tid & 3, t = (tid >> 2) & 3, gq = tid >> 4;
    const uint8_t* r = codes + (size_t)(cb * 32 + j * 8 + gq) * crow;
    uint32_t w = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int in = g * 32 + h * 16 + t * 4;  // inputs in .. in + 3: bytes in / 2, in / 2 + 1
      const uint32_t b0 = r[in / 2], b1 = r[in / 2 + 1];
      // slot(h, q) = 4 (2 h + q / 2 + 4 (q % 2)): q 0 low nibble of b0, q 1 its high, q 2 / 3 b1's
      w |= (b0 & 0xF) << (4 * (2 * h)) | (b0 >> 4) << (4 * (2 * h + 4)) | (b1 & 0xF) << (4 * (2 * h + 1)) |
           (b1 >> 4) << (4 * (2 * h + 5));
    }
    out[tid] = w;
  } else {
    uint32_t w = 0;
    unsigned flag = 0;
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const int bi = 4 * (tid - 128) + x;
      const int c = bi & 1, j = (bi >> 1) & 3, h = (bi >> 3) & 1, t = bi >> 4;
      const uint32_t v = scales[(size_t)(cb * 32 + j * 8 + t * 2 + c) * srow + g * 2 + h];
      if ((v & 0x7F) == 0x7F || ((v & 0x80) && v != 0x80)) flag = 1;
      w |= v << (8 * x);
    }
    out[tid] = w;
    if (flag) atomicOr(bad, 1u);
  }
}

// MXFP8: a block of 264 words, 256 words of e4m3 codes (gq, t, j, h; 4 inputs a word) then 8 of UE8M0 exponents
// (t, j, c).
extern "C" __global__ void ink2_pack_mx(uint32_t* __restrict__ up, uint32_t* __restrict__ down,
                                        const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13,
                                        const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2, int H, int Il,
                                        int I, int rank, unsigned* __restrict__ bad) {
  const int nb = (Il / 32) * (H / 32);
  const int mat = blockIdx.x / nb, b = blockIdx.x % nb;
  const int tid = threadIdx.x;
  if (tid >= 264) return;
  const bool dn = mat == 2;
  const int kg = dn ? Il / 32 : H / 32;
  const int cb = b / kg, g = b % kg;
  const size_t crow = dn ? (size_t)I : (size_t)2 * H;
  const size_t srow = dn ? (size_t)(I / 32) : (size_t)2 * (H / 32);
  const uint8_t* codes = dn ? w2 + (size_t)rank * Il : w13 + (size_t)mat * H;
  const uint8_t* exps = dn ? s2 + (size_t)rank * (Il / 32) : s13 + (size_t)mat * (H / 32);
  uint32_t* out = dn ? down + (size_t)b * 264 : up + ((size_t)b * 2 + mat) * 264;
  uint32_t w = 0;
  unsigned flag = 0;
  if (tid < 256) {
    const int h = tid & 1, j = (tid >> 1) & 3, t = (tid >> 3) & 3, gq = tid >> 5;
    const uint8_t* r = codes + (size_t)(cb * 32 + j * 8 + gq) * crow + g * 32 + h * 16 + t * 4;
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const uint32_t v = r[x];
      if ((v & 0x7F) == 0x7F) flag = 2;
      w |= v << (8 * x);
    }
  } else {
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const int bi = 4 * (tid - 256) + x;
      const int c = bi & 1, j = (bi >> 1) & 3, t = bi >> 3;
      const uint32_t v = exps[(size_t)(cb * 32 + j * 8 + t * 2 + c) * srow + g];
      if (v == 0xFF) flag = 4;
      w |= v << (8 * x);
    }
  }
  out[tid] = w;
  if (flag) atomicOr(bad, flag);
}

// ================================================================================================= p3-tc (opt-in)
// ink2_attention_tc_d64 / _d128: a prompt chunk's attention on tensor cores, chosen only by
// TF_INKLING2_PREFILL_ATTN=tc (Model.prefill_attn) for wide chunks. NOT ink2_attention's bits: the same
// eager_attention_forward (M:368-399) rounding points, but the reference's two bf16 matmuls (q k^T and p v) are
// mma.sync m16n8k16 bf16 products with fp32 accumulation (exact products summed in the tensor core's order, as
// cuBLAS's bf16 GEMMs sum them), and the softmax's exp sum is taken in another order. Per (row, head):
//   s = bf16(q . k)  (fp32 mma sum)      s = bf16(s * scaling)      s = bf16(s + bias), bias = rel[row][head][pos - p]
//   (0 at or past `extent`); mx = max(s..., sink) (exact); sum = sum of expf(s - mx) (quad partials, then the four
//   lanes' shuffle sum) + expf(sink - mx); p = bf16(expf(s - mx) / sum); o = bf16(fp32 mma sum of p v);
//   out = bf16(o * (1 / (1 + expf(-gate)))).
// p is rounded with the final max and sum, so the scores are made three times (no online softmax, no score
// scratch): pass 1 the max, pass 2 the exp sum, pass 3 p and the p v product. One block a (16-row tile, kv head):
// warp w is query head kv_head * G + w (G = heads / kv_heads <= 8 warps) on the tile's 16 rows, the block's warps
// share each staged tile of 32 keys (and, in pass 3, their values): two cp.async stages, [key][D + 8] bf16 rows
// (ldmatrix rows on distinct banks). Row i (position q0 + i) sees keys p in [lo_i, q0 + i] (lo_i = q0 + i - window
// + 1 clamped at 0 on a sliding layer, else 0); a tile stages [the first row's lo, the last row's position] from a
// multiple of 16 and masks per row (so a row's bits do not depend on the tiling or the chunking); staged keys outside
// that range are zero-filled (never read from the cache). Tiles of the launch's last rows first (on a global layer
// they read the most keys). D 64 or 128, sm_80 and up.
namespace {
constexpr int TC_ROWS = 16;  // query rows a block (each warp's mma rows)
constexpr int TC_KEYS = 32;  // keys a staged tile

__device__ __forceinline__ uint32_t tc_saddr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes global -> shared, asynchronously; `bytes` 0 zero-fills the 16 (the source is then not read)
__device__ __forceinline__ void tc_cp16(uint32_t dst, const void* src, int bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
#endif
}
__device__ __forceinline__ void tc_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void tc_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// four 8x8 bf16 matrices from shared memory (lane l gives row l % 8 of matrix l / 8), plain or transposed
__device__ __forceinline__ void tc_ldsm4(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}
__device__ __forceinline__ void tc_ldsm4t(uint32_t (&r)[4], uint32_t addr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

// c += a b, one m16n8k16 product: bf16 inputs (exact products), fp32 accumulation
__device__ __forceinline__ void tc_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
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

// two bf16 values (already bf16, so exact) into an mma operand register, the lower column in the low half
__device__ __forceinline__ uint32_t tc_pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

template <int D>
__device__ __forceinline__ void attn_tc_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                             const __nv_bfloat16* __restrict__ vc,
                                             const __nv_bfloat16* __restrict__ rel,
                                             const __nv_bfloat16* __restrict__ sinks,
                                             const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out,
                                             int rows, int heads, int kv_heads, int q0, int window, int slots,
                                             int extent, float scaling) {
  constexpr int KS = D / 16;        // the head dim's k-steps
  constexpr int RW = D + 8;         // bf16 a staged key: D and 16 bytes of padding
  constexpr int NT = TC_KEYS / 8;   // a tile's 8-key n-tiles
  constexpr int ST = TC_KEYS * RW;  // bf16 a stage
  extern __shared__ uint4 tc_smem[];
  const uint32_t kt = tc_saddr(tc_smem);  // [2][TC_KEYS][RW] keys
  const uint32_t vt = kt + 2 * ST * 2;    // [2][TC_KEYS][RW] values
  const int G = heads / kv_heads;
  const int tiles = (rows + TC_ROWS - 1) / TC_ROWS;
  const int bt = (int)blockIdx.x / kv_heads;
  const int g = (int)blockIdx.x - bt * kv_heads;
  const int tile = tiles - 1 - bt;  // the last rows first
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h = g * G + warp;
  const int r0 = tile * TC_ROWS;
  const int rl = min(r0 + TC_ROWS, rows) - 1;  // the tile's last row
  // the keys the tile reads, [klo, khi], staged from kb = klo rounded down to 16: every key sits at the same lane and
  // k-step slot whatever the tiling (its position mod 16), so a row's bits depend on its keys alone (keys a row does
  // not see add exact zeros); staged keys outside [klo, khi] are zero-filled, never read from the cache
  const int khi = q0 + rl;
  const int klo = window > 0 ? max(0, q0 + r0 - window + 1) : 0;
  const int kb = klo & ~15;
  const int ntile = (khi - kb + TC_KEYS) / TC_KEYS;
  // the lane's two fragment rows, r0 + lane / 4 and that + 8 (rows past the launch's: copies of its last, unwritten)
  int pos[2], lo[2];
  const __nv_bfloat16* relr[2];
  float sink[2];
  const uint32_t* qr[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = min(r0 + (lane >> 2) + 8 * i, rl);
    pos[i] = q0 + r;
    lo[i] = window > 0 ? max(0, pos[i] - window + 1) : 0;
    relr[i] = rel + ((size_t)r * heads + h) * extent;
    sink[i] = ld(sinks, (size_t)r * heads + h);
    qr[i] = reinterpret_cast<const uint32_t*>(q + ((size_t)r * heads + h) * D);
  }
  // the query rows as the A operand (rows by dims), once: register j of k-step s holds row (j & 1 ? +8 : +0),
  // dims 16 s + 2 (lane % 4) + 8 (j / 2) and the next
  uint32_t qa[KS][4];
#pragma unroll
  for (int s = 0; s < KS; ++s) {
    const int c = s * 8 + (lane & 3);
    qa[s][0] = qr[0][c];
    qa[s][1] = qr[1][c];
    qa[s][2] = qr[0][c + 4];
    qa[s][3] = qr[1][c + 4];
  }
  // stage tile t (keys kb + 32 t ..) into buffer t % 2, with the values in pass 3
  auto stage = [&](int t, bool values) {
    const int k0 = kb + t * TC_KEYS;
    const uint32_t b = (uint32_t)((t & 1) * ST * 2);
    for (int c = threadIdx.x; c < TC_KEYS * (D / 8); c += blockDim.x) {
      const int kk = c / (D / 8), part = c - kk * (D / 8);
      const int p = k0 + kk;
      const bool in = p >= klo && p <= khi;
      const size_t off = in ? ((size_t)(p % slots) * kv_heads + g) * D + part * 8 : 0;
      const uint32_t dst = b + (uint32_t)((kk * RW + part * 8) * 2);
      tc_cp16(kt + dst, kc + off, in ? 16 : 0);
      if (values) tc_cp16(vt + dst, vc + off, in ? 16 : 0);
    }
    tc_commit();
  };
  // a staged tile's raw scores q . k (fp32 mma sums): fragment (n, e) is row (e < 2 ? +0 : +8), key 8 n + 2 (lane %
  // 4) + (e & 1) of the tile
  auto scores = [&](int t, float (&sa)[NT][4]) {
    const uint32_t b = kt + (uint32_t)((t & 1) * ST * 2);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.f;
#pragma unroll
      for (int s = 0; s < KS; s += 2) {
        uint32_t r[4];  // keys 8 n .. 8 n + 7 by dims 16 s + {0, 8, 16, 24}: k-steps s and s + 1
        tc_ldsm4(r, b + (uint32_t)(((n * 8 + (lane & 7)) * RW + s * 16 + (lane >> 3) * 8) * 2));
        tc_mma(sa[n], qa[s], r[0], r[1]);
        tc_mma(sa[n], qa[s + 1], r[2], r[3]);
      }
    }
  };
  // fragment (n, e)'s score at the reference's points, -inf where its row does not see the key
  auto finish = [&](float acc, int t, int n, int e) -> float {
    const int i = e >> 1;
    const int p = kb + t * TC_KEYS + n * 8 + (lane & 3) * 2 + (e & 1);
    const int dist = pos[i] - p;
    if (p < lo[i] || dist < 0) return -INFINITY;
    const float s = bf(bf(acc) * scaling);
    const float bias = dist < extent ? ld(relr[i], dist) : 0.f;
    return bf(s + bias);
  };
  // one pass over the tiles: the next tile's copies in flight while this one is used
  auto sweep = [&](bool values, auto&& use) {
    stage(0, values);
    for (int t = 0; t < ntile; ++t) {
      if (t + 1 < ntile) {
        stage(t + 1, values);
        tc_wait<1>();
      } else {
        tc_wait<0>();
      }
      __syncthreads();
      float sa[NT][4];
      scores(t, sa);
      use(t, sa);
      __syncthreads();  // the buffer is read before the next stage refills it
    }
  };
  // pass 1: the max (the sink included)
  float mx[2] = {-INFINITY, -INFINITY};
  sweep(false, [&](int t, float (&sa)[NT][4]) {
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) mx[e >> 1] = fmaxf(mx[e >> 1], finish(sa[n][e], t, n, e));
  });
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
    mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
    mx[i] = fmaxf(mx[i], sink[i]);
  }
  // pass 2: the exp sum (expf(-inf) = 0 for the keys a row does not see)
  float se[2] = {0.f, 0.f};
  sweep(false, [&](int t, float (&sa)[NT][4]) {
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) se[e >> 1] += expf(finish(sa[n][e], t, n, e) - mx[e >> 1]);
  });
  float sum[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    se[i] += __shfl_xor_sync(0xffffffffu, se[i], 1);
    se[i] += __shfl_xor_sync(0xffffffffu, se[i], 2);
    sum[i] = se[i] + expf(sink[i] - mx[i]);
  }
  // pass 3: p = bf16(expf(s - mx) / sum) as the A operand (rows by keys: n-tiles 2 k and 2 k + 1 are k-step k's
  // two key halves, the score fragment's layout is the operand's), times the values (B: ldmatrix.trans of [key][d])
  float oa[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) oa[j][0] = oa[j][1] = oa[j][2] = oa[j][3] = 0.f;
  sweep(true, [&](int t, float (&sa)[NT][4]) {
    uint32_t pa[NT / 2][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      float pv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) pv[e] = bf(expf(finish(sa[n][e], t, n, e) - mx[e >> 1]) / sum[e >> 1]);
      pa[n >> 1][(n & 1) * 2] = tc_pack(pv[0], pv[1]);
      pa[n >> 1][(n & 1) * 2 + 1] = tc_pack(pv[2], pv[3]);
    }
    const uint32_t b = vt + (uint32_t)((t & 1) * ST * 2);
    const int m = lane >> 3;  // ldmatrix: matrix m is keys 16 k + 8 (m % 2) .., dims 8 j + 8 (m / 2) ..
#pragma unroll
    for (int k = 0; k < NT / 2; ++k)
#pragma unroll
      for (int j = 0; j < D / 8; j += 2) {
        uint32_t r[4];
        tc_ldsm4t(r, b + (uint32_t)(((k * 16 + (m & 1) * 8 + (lane & 7)) * RW + j * 8 + (m >> 1) * 8) * 2));
        tc_mma(oa[j], pa[k], r[0], r[1]);
        tc_mma(oa[j + 1], pa[k], r[2], r[3]);
      }
  });
  // o = bf16(sum), then the gate (M:496): fragment (j, e) is row (e < 2 ? +0 : +8), dim 8 j + 2 (lane % 4) + (e & 1)
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = r0 + (lane >> 2) + 8 * i;
    if (r > rl) continue;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
      const size_t at = ((size_t)r * heads + h) * D + j * 8 + (lane & 3) * 2;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float o = bf(oa[j][2 * i + e]);
        const float gt = ld(gate, at + e);
        out[at + e] = __float2bfloat16_rn(o * (1.f / (1.f + expf(-gt))));
      }
    }
  }
}
}  // namespace

#define INK2_ATTN_TC(D)                                                                                              \
  extern "C" __global__ void __launch_bounds__(256) ink2_attention_tc_d##D(                                          \
      const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,                                     \
      const __nv_bfloat16* __restrict__ vc, const __nv_bfloat16* __restrict__ rel,                                   \
      const __nv_bfloat16* __restrict__ sinks, const __nv_bfloat16* __restrict__ gate,                               \
      __nv_bfloat16* __restrict__ out, int rows, int heads, int kv_heads, int q0, int window, int slots, int extent, \
      float scaling) {                                                                                               \
    attn_tc_body<D>(q, kc, vc, rel, sinks, gate, out, rows, heads, kv_heads, q0, window, slots, extent, scaling);    \
  }
INK2_ATTN_TC(64)
INK2_ATTN_TC(128)
#undef INK2_ATTN_TC
// ================================================================================================= end p3-tc

// ================================================================================= p3-attn: the tiled prompt attention
// ink2_attention_gqa's outputs, bit for bit, for a prompt chunk's rows in three launches that tile both products over
// rows, heads and keys (TF_INKLING2_PREFILL_ATTN=tiled, Model.prefill_attn). Every number is made by the arithmetic
// ink2_attention makes it with, only by another thread:
//   ink2_attn_tiled_scores  a block 16 rows x the kv group's G heads x 128 keys; a thread one row's G heads x 8 keys,
//                           register-tiled: each score's dot is its own sequential fmaf chain over d = 0 .. D-1 from
//                           0.f (q and k bf16 -> fp32, staged in shared memory 32 dims a stage), then
//                           s = bf(bf(dot) * scaling), s = bf(s + bias) (bias 0.f at or past the extent) as before.
//                           The scores go to the scratch as bf16 (they are bf16 values): (row, head)'s at
//                           ((row * heads) + head) * stride, position p at p - (lo & ~7) (16-byte aligned runs).
//   ink2_attn_tiled_softmax a warp a (row, head): the max over the scores and the sink (exact, any order); the exp sum
//                           as ink2_attention's 128 threads take it: lane l keeps the partials of u = l, l + 32,
//                           l + 64, l + 96 (u = j mod 128, j = p - lo, each summed in j order as `se += expf(..)`),
//                           each partial set goes through the warp butterfly a 128-thread block_sum's warp w does, and
//                           (W0 + W2) + (W1 + W3) is block_sum's second stage; then + expf(sink - mx); the
//                           probabilities bf(expf(s - mx) / sum) over the scores in place (bf16).
//   ink2_attn_tiled_values  a block 8 rows x G heads x all D dims; a thread one row's G heads x 8 dims, each
//                           output's sequential fmaf chain over the row's keys in position order from 0.f (keys
//                           outside the row's range are skipped, never multiplied by 0), staged 32 keys a stage;
//                           then bf(o) * (1 / (1 + expf(-gate))) to bf16, as before.
// The expressions are written as ink2_attention_gqa's (the compiler fuses expf's last product into the exp sum's add,
// the sink's add and the gate's 1 + ..., but not into the probability's division, in both). G (heads / kv_heads)
// 4 or 6, D a multiple of 32 up to 128 (Kernels.tiledFits). Keys past a launch's last row are never read; keys of
// the tile before a row's window are read and computed but never used.
namespace {
constexpr int AT_QR = 16;   // scores: rows a block
constexpr int AT_QN = 128;  // scores: keys a block (thread kg: keys 4kg .. 4kg+3 and 64+4kg .. 64+4kg+3)
constexpr int AT_QD = 32;   // scores: dims a stage
constexpr int AT_VR = 8;    // values: rows a block
constexpr int AT_VK = 32;   // values: keys a stage

__device__ __forceinline__ int at_lo(int pos, int window) { return window > 0 ? max(0, pos - window + 1) : 0; }

// 8 bf16 (a 16-byte group, element 0 in the low half of .x) to fp32, exactly
__device__ __forceinline__ void at_bf8(const uint4 v, float* f) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    f[2 * c] = __uint_as_float(w[c] << 16);
    f[2 * c + 1] = __uint_as_float(w[c] & 0xFFFF0000u);
  }
}

// G consecutive floats from shared memory (8-byte aligned when G is even)
template <int G>
__device__ __forceinline__ void at_ldg(const float* p, float* x) {
  if constexpr (G % 4 == 0) {
#pragma unroll
    for (int i = 0; i < G; i += 4) {
      const float4 t = *reinterpret_cast<const float4*>(p + i);
      x[i] = t.x, x[i + 1] = t.y, x[i + 2] = t.z, x[i + 3] = t.w;
    }
  } else if constexpr (G % 2 == 0) {
#pragma unroll
    for (int i = 0; i < G; i += 2) {
      const float2 t = *reinterpret_cast<const float2*>(p + i);
      x[i] = t.x, x[i + 1] = t.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < G; ++i) x[i] = p[i];
  }
}

// scores: block (row tile x, key tile y, kv group z), AT_QR * AT_QN / 8 = 256 threads, thread (r = tid / 16, kg)
template <int G>
__device__ __forceinline__ void at_scores(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                          const __nv_bfloat16* __restrict__ rel, __nv_bfloat16* __restrict__ sc,
                                          int rows, int heads, int kv_heads, int D, int q0, int window, int slots,
                                          int extent, float scaling, int stride, float* smem) {
  float* qs = smem;                    // [AT_QD][AT_QR * G] the stage's query dims, (row, head) contiguous
  float* ks = qs + AT_QD * AT_QR * G;  // [AT_QD][AT_QN] the stage's key dims, key k at k ^ (8 * (d / 8 % 4))
  const int g = blockIdx.z;
  const int r0 = blockIdx.x * AT_QR;
  const int plast = q0 + min(rows - 1, r0 + AT_QR - 1);  // the tile's last row's position: no key past it is read
  const int p0 = (at_lo(q0 + r0, window) & ~7) + blockIdx.y * AT_QN;
  if (p0 > plast) return;
  const int tid = threadIdx.x, r = tid >> 4, kg = tid & 15;
  // the thread's staging groups (8 dims each, the same every stage but for d0): query groups v = tid + 256 i of
  // AT_QR * G * 4 ((row, head) v / 4, dims 8 (v % 4)), key groups v = tid + 256 i of AT_QN * 4 (key v / 4)
  constexpr int NQ = (AT_QR * G * (AT_QD / 8) + 255) / 256;
  const uint4* qsrc[NQ];
  int qdst[NQ];
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    const int v = tid + 256 * i, dv = v & 3, rh = v >> 2, rr = rh / G, hg = rh - rr * G;
    const bool in = v < AT_QR * G * (AT_QD / 8);
    qsrc[i] = in && r0 + rr < rows ? reinterpret_cast<const uint4*>(q + ((size_t)(r0 + rr) * heads + g * G + hg) * D) + dv
                                   : nullptr;
    qdst[i] = in ? dv * 8 * (AT_QR * G) + rh : -1;
  }
  const uint4* ksrc[2];
  int kdst[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v = tid + 256 * i, dv = v & 3, kk = v >> 2, p = p0 + kk;
    ksrc[i] = p <= plast ? reinterpret_cast<const uint4*>(kc + ((size_t)(p % slots) * kv_heads + g) * D) + dv : nullptr;
    kdst[i] = dv * 8 * AT_QN + (kk ^ (dv << 3));
  }
  uint4 qx[NQ], kx[2];
  auto fetch = [&](int d0) {
#pragma unroll
    for (int i = 0; i < NQ; ++i) qx[i] = qsrc[i] ? __ldg(qsrc[i] + d0 / 8) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < 2; ++i) kx[i] = ksrc[i] ? __ldg(ksrc[i] + d0 / 8) : make_uint4(0, 0, 0, 0);
  };
  fetch(0);
  float acc[G][8];
#pragma unroll
  for (int i = 0; i < G; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for (int d0 = 0; d0 < D; d0 += AT_QD) {
    __syncthreads();  // the previous stage is read
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
      if (qdst[i] < 0) continue;
      float f[8];
      at_bf8(qx[i], f);
#pragma unroll
      for (int e = 0; e < 8; ++e) qs[qdst[i] + e * (AT_QR * G)] = f[e];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float f[8];
      at_bf8(kx[i], f);
#pragma unroll
      for (int e = 0; e < 8; ++e) ks[kdst[i] + e * AT_QN] = f[e];
    }
    __syncthreads();
    if (d0 + AT_QD < D) fetch(d0 + AT_QD);  // the next stage's loads fly during this stage's products
#pragma unroll 8
    for (int d = 0; d < AT_QD; ++d) {
      const int sw = ((d >> 3) & 3) << 3;
      const float4 ka = *reinterpret_cast<const float4*>(ks + d * AT_QN + ((4 * kg) ^ sw));
      const float4 kb = *reinterpret_cast<const float4*>(ks + d * AT_QN + ((64 + 4 * kg) ^ sw));
      float qv[G];
      at_ldg<G>(qs + d * (AT_QR * G) + r * G, qv);
      const float kv[8] = {ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w};
#pragma unroll
      for (int i = 0; i < G; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(qv[i], kv[j], acc[i][j]);
    }
  }
  const int row = r0 + r;
  if (row >= rows) return;
  const int pos = q0 + row, lo8 = at_lo(pos, window) & ~7;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int pr = p0 + half * 64 + 4 * kg;  // the run's first key
    if (pr < lo8 || pr > pos) continue;      // no key of the run in [lo8, pos]
#pragma unroll
    for (int i = 0; i < G; ++i) {
      const int h = g * G + i;
      uint32_t w[2];
#pragma unroll
      for (int e = 0; e < 4; e += 2) {
        float o[2];
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const int p = pr + e + f;
          float s = bf(bf(acc[i][half * 4 + e + f]) * scaling);
          const int dist = pos - p;
          const float bias = dist >= 0 && dist < extent ? ld(rel, ((size_t)row * heads + h) * extent + dist) : 0.f;
          s = bf(s + bias);
          o[f] = s;
        }
        w[e >> 1] = (__float_as_uint(o[0]) >> 16) | (__float_as_uint(o[1]) & 0xFFFF0000u);
      }
      *reinterpret_cast<uint2*>(sc + ((size_t)row * heads + h) * stride + (pr - lo8)) = make_uint2(w[0], w[1]);
    }
  }
}

// values: block (row tile x, kv group y, dim slice z), AT_VR * Dh / 8 threads (Dh = D / gridDim.z: the dims split
// over 1, 2 or 4 blocks, the same bits; Kernels.attentionTiled launches 1), thread (r = tid / (Dh / 8), dg): dims
// z Dh + 4dg .. +3 and z Dh + Dh/2 + 4dg .. +3
template <int G>
__device__ __forceinline__ void at_values(const __nv_bfloat16* __restrict__ sc, const __nv_bfloat16* __restrict__ vc,
                                          const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out,
                                          int rows, int heads, int kv_heads, int D, int q0, int window, int slots,
                                          int stride, float* smem) {
  const int Dh = D / gridDim.z, d0 = blockIdx.z * Dh, DG = Dh >> 3, nt = AT_VR * DG;
  float* vs = smem;             // [AT_VK][Dh] the stage's values (the block's dims)
  float* ps = vs + AT_VK * Dh;  // [AT_VK][AT_VR * G] the stage's probabilities, (row, head) contiguous
  const int g = blockIdx.y, r0 = blockIdx.x * AT_VR;
  const int tid = threadIdx.x, r = tid / DG, dg = tid - r * DG;
  const int row = r0 + r, pos = q0 + row, lo = at_lo(pos, window);
  const int tlast = q0 + min(rows - 1, r0 + AT_VR - 1);  // the tile's keys: [its first row's lo & ~7, tlast]
  float acc[G][8];
#pragma unroll
  for (int i = 0; i < G; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for (int t0 = at_lo(q0 + r0, window) & ~7; t0 <= tlast; t0 += AT_VK) {
    __syncthreads();  // the previous stage is read
    for (int v = tid; v < AT_VK * DG; v += nt) {
      const int kk = v / DG, dv = v - kk * DG, p = t0 + kk;
      if (p > tlast) continue;
      float f[8];
      at_bf8(__ldg(reinterpret_cast<const uint4*>(vc + ((size_t)(p % slots) * kv_heads + g) * D + d0) + dv), f);
      *reinterpret_cast<float4*>(vs + kk * Dh + dv * 8) = make_float4(f[0], f[1], f[2], f[3]);
      *reinterpret_cast<float4*>(vs + kk * Dh + dv * 8 + 4) = make_float4(f[4], f[5], f[6], f[7]);
    }
    for (int v = tid; v < (AT_VK / 8) * AT_VR * G; v += nt) {
      const int rh = v % (AT_VR * G), gi = v / (AT_VR * G);
      const int rr = rh / G, hg = rh - rr * G, prow = r0 + rr;
      if (prow >= rows) continue;
      const int ppos = q0 + prow, plo8 = at_lo(ppos, window) & ~7, gp = t0 + gi * 8;
      if (gp < plo8 || gp > ppos) continue;
      float f[8];
      at_bf8(__ldg(reinterpret_cast<const uint4*>(sc + ((size_t)prow * heads + g * G + hg) * stride + (gp - plo8))), f);
#pragma unroll
      for (int i = 0; i < 8; ++i) ps[(gi * 8 + i) * (AT_VR * G) + rh] = f[i];
    }
    __syncthreads();
    if (row < rows) {
      const int kb = max(lo, t0) - t0, ke = min(pos + 1, t0 + AT_VK) - t0;  // the row's keys in the stage
#pragma unroll 4
      for (int k = kb; k < ke; ++k) {
        const float4 va = *reinterpret_cast<const float4*>(vs + k * Dh + 4 * dg);
        const float4 vb = *reinterpret_cast<const float4*>(vs + k * Dh + (Dh >> 1) + 4 * dg);
        float pv[G];
        at_ldg<G>(ps + k * (AT_VR * G) + r * G, pv);
        const float vv[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
#pragma unroll
        for (int i = 0; i < G; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(pv[i], vv[j], acc[i][j]);
      }
    }
  }
  if (row >= rows) return;
#pragma unroll
  for (int i = 0; i < G; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = d0 + (j < 4 ? 4 * dg + j : (Dh >> 1) + 4 * dg + j - 4);
      const float o = bf(acc[i][j]);
      const size_t at = ((size_t)row * heads + g * G + i) * D + d;
      const float gt = ld(gate, at);
      out[at] = __float2bfloat16_rn(o * (1.f / (1.f + expf(-gt))));
    }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(AT_QR * AT_QN / 8) ink2_attn_tiled_scores(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ rel,
    __nv_bfloat16* __restrict__ sc, int rows, int heads, int kv_heads, int D, int q0, int window, int slots, int extent,
    float scaling, int stride) {
  extern __shared__ float4 at_smem[];
  float* smem = reinterpret_cast<float*>(at_smem);
  if (heads / kv_heads == 6) {
    at_scores<6>(q, kc, rel, sc, rows, heads, kv_heads, D, q0, window, slots, extent, scaling, stride, smem);
  } else {
    at_scores<4>(q, kc, rel, sc, rows, heads, kv_heads, D, q0, window, slots, extent, scaling, stride, smem);
  }
}

// a warp a (row, head) of the launch's rows * heads, 256 threads a block
extern "C" __global__ void __launch_bounds__(256) ink2_attn_tiled_softmax(__nv_bfloat16* __restrict__ sc,
                                                                           const __nv_bfloat16* __restrict__ sinks,
                                                                           int rows, int heads, int q0, int window,
                                                                           int stride) {
  const int w = blockIdx.x * 8 + (threadIdx.x >> 5), l = threadIdx.x & 31;
  if (w >= rows * heads) return;  // (no block barriers below)
  const int row = w / heads, pos = q0 + row, lo = at_lo(pos, window), n = pos - lo + 1;
  __nv_bfloat16* s = sc + (size_t)w * stride + (lo & 7);  // s[j]: position lo + j
  float mx = -INFINITY;
  for (int j0 = 0; j0 < n; j0 += 128) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int j = j0 + 32 * c + l;
      if (j < n) mx = fmaxf(mx, __bfloat162float(s[j]));
    }
  }
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const float sink = ld(sinks, (size_t)w);
  mx = fmaxf(mx, sink);
  // partial u = 32 c + l (ink2_attention's thread u, its warp c) in e[c]
  float e[4] = {0.f, 0.f, 0.f, 0.f};
  for (int j0 = 0; j0 < n; j0 += 128) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int j = j0 + 32 * c + l;
      if (j < n) e[c] += expf(__bfloat162float(s[j]) - mx);
    }
  }
#pragma unroll
  for (int c = 0; c < 4; ++c)
    for (int o = 16; o > 0; o >>= 1) e[c] += __shfl_xor_sync(0xffffffffu, e[c], o);
  const float sum = ((e[0] + e[2]) + (e[1] + e[3])) + expf(sink - mx);
  for (int j0 = 0; j0 < n; j0 += 128) {
    float x[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int j = j0 + 32 * c + l;
      x[c] = j < n ? __bfloat162float(s[j]) : 0.f;
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int j = j0 + 32 * c + l;
      if (j < n) s[j] = __float2bfloat16_rn(expf(x[c] - mx) / sum);
    }
  }
}

extern "C" __global__ void __launch_bounds__(AT_VR * 16) ink2_attn_tiled_values(
    const __nv_bfloat16* __restrict__ sc, const __nv_bfloat16* __restrict__ vc, const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out, int rows, int heads, int kv_heads, int D, int q0, int window, int slots,
    int stride) {
  extern __shared__ float4 at_smem[];
  float* smem = reinterpret_cast<float*>(at_smem);
  if (heads / kv_heads == 6) {
    at_values<6>(sc, vc, gate, out, rows, heads, kv_heads, D, q0, window, slots, stride, smem);
  } else {
    at_values<4>(sc, vc, gate, out, rows, heads, kv_heads, D, q0, window, slots, stride, smem);
  }
}

// ------------------------------------------------------------------- p3-comm: the split combine on a column part
// ink2_combine_pair on the output columns [c0, c0 + n) of the rows (out at the rows' first row and column c0, rows H
// apart), each rank's partials of them in a part of its own: y0 / y1 [t][k][n] (the down form's column parts,
// forward.moeOverlap). Every output element is ink2_combine_pair's: the same picks order, v = y0 + y1 in rank order,
// the same bf16 rounding of v, product and sum order, the same final rounding. A block a (row, blockDim columns).
extern "C" __global__ void ink2_combine_pair_cols(const float* __restrict__ y0, const float* __restrict__ y1,
                                                  const int* __restrict__ picks, const float* __restrict__ weights,
                                                  __nv_bfloat16* __restrict__ out, int k, int n, int H) {
  const int chunks = (n + blockDim.x - 1) / blockDim.x;
  const size_t t = blockIdx.x / chunks;
  const int i = (blockIdx.x % chunks) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int order[16];
  for (int j = 0; j < k; ++j) order[j] = j;
  for (int a = 1; a < k; ++a) {  // ascending expert id, stable
    const int v = order[a];
    int b = a;
    while (b > 0 && picks[t * k + order[b - 1]] > picks[t * k + v]) {
      order[b] = order[b - 1];
      --b;
    }
    order[b] = v;
  }
  float acc = 0.f;
  for (int j = 0; j < k; ++j) {
    const int s = order[j];
    if (picks[t * k + s] < 0) continue;
    const size_t at = (t * k + s) * n + i;
    float v = y0[at];
    v = v + y1[at];
    acc = acc + bf(v) * weights[t * k + s];
  }
  out[t * H + i] = __float2bfloat16_rn(acc);
}

// ======================================================= p4-dattn: decode / verify attention, split-K flash decoding
// TF_INKLING2_DECODE_ATTN=1 (Model.decode_attn): a chunk of at most max_rows rows (decode, verify windows, a short
// prompt chunk) runs ink2_fdec_d* then ink2_fdec_merge instead of ink2_attention (inkling2_fdec.cuh: one block a (key
// split, kv head) with every row x query head of the kv group, so each key and value is read once). NOT bitwise:
//   s = bf16(q . k) (fp32 mma sum)   s = bf16(s * scaling)   s = bf16(s + bias), bias = rel[row][head][pos - p] (0 at
//   or past `extent`), -inf where the row does not see p (p > its position, or before its window); per split the online
//   softmax (m = running max, l = sum exp(s - m) in fp32, o = sum bf16(exp(s - m)) v on the mma); the merge (a block a
//   (row, head), the splits in order): M = max(m_s..., sink), L = sum l_s exp(m_s - M) + exp(sink - M),
//   o = bf16(sum o_s exp(m_s - M) / L), out = bf16(o * (1 / (1 + exp(-gate)))). The reference's rounding points but
//   p's (bf16(exp(s - m)) here, bf16(exp(s - mx) / sum) there) and the sums' order. Keys of the launch: row 0's first
//   (its window's) through the last row's position; the grid is (splits, kv heads) whatever the position.
// part: [rows][heads][splits][2 + D] fp32 (m, l, o).
#include "inkling2_fdec.cuh"

namespace {
template <int D>
struct Ink2Fd {
  struct Query {
    int pos, lo;
    const __nv_bfloat16* rel;
    float* slot;
  };
  const __nv_bfloat16* q;
  const __nv_bfloat16* kc;
  const __nv_bfloat16* vc;
  const __nv_bfloat16* rel;
  float* part;
  int heads, kvh, G, g, q0, window, slots, extent, kb, nkeys, nq, sbase;
  float scaling;

  __device__ __forceinline__ Query query(int qi) const {
    const int r = qi / G, h = g * G + (qi - r * G);
    Query x;
    x.pos = q0 + r;
    x.lo = window > 0 ? max(0, x.pos - window + 1) : 0;
    x.rel = rel + ((size_t)r * heads + h) * extent;
    x.slot = part + ((size_t)r * heads + h) * gridDim.x * (2 + D);
    return x;
  }
  __device__ __forceinline__ const __nv_bfloat16* qrow(int qi) const {
    const int r = qi / G, h = g * G + (qi - r * G);
    return q + ((size_t)r * heads + h) * D;
  }
  // key j of the launch is position kb + j, at slot (kb + j) % slots: the range is at most `slots` long (a sliding
  // ring holds window + rows, a global cache the context), so one wrap at most
  __device__ __forceinline__ bool rows(int j, const __nv_bfloat16*& k, const __nv_bfloat16*& v) const {
    if (j >= nkeys) return false;
    int sl = sbase + j;
    if (sl >= slots) sl -= slots;
    const size_t off = ((size_t)sl * kvh + g) * D;
    k = kc + off;
    v = vc + off;
    return true;
  }
  __device__ __forceinline__ float score(float dot, const Query& x, int j) const {
    const int p = kb + j, dist = x.pos - p;
    if (p < x.lo || dist < 0) return -INFINITY;
    const float s = bf(bf(dot) * scaling);
    const float bias = dist < extent ? ld(x.rel, dist) : 0.f;
    return bf(s + bias);
  }
  __device__ __forceinline__ float* slot(const Query& x) const { return x.slot; }
};

template <int D>
__device__ __forceinline__ void ink2_fdec_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                               const __nv_bfloat16* __restrict__ vc,
                                               const __nv_bfloat16* __restrict__ rel, float* __restrict__ part,
                                               int rows, int heads, int kv_heads, int q0, int window, int slots,
                                               int extent, float scaling) {
  Ink2Fd<D> p;
  p.q = q;
  p.kc = kc;
  p.vc = vc;
  p.rel = rel;
  p.part = part;
  p.heads = heads;
  p.kvh = kv_heads;
  p.G = heads / kv_heads;
  p.g = blockIdx.y;
  p.q0 = q0;
  p.window = window;
  p.slots = slots;
  p.extent = extent;
  p.kb = window > 0 ? max(0, q0 - window + 1) : 0;  // row 0's first key
  p.nkeys = q0 + rows - p.kb;                       // through the last row's position
  p.nq = rows * p.G;
  p.sbase = p.kb % slots;
  p.scaling = scaling;
  fdec::partial<D>(p);
}
}  // namespace

#define INK2_FDEC(D)                                                                                                 \
  extern "C" __global__ void __launch_bounds__(256) ink2_fdec_d##D(                                                  \
      const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,                                     \
      const __nv_bfloat16* __restrict__ vc, const __nv_bfloat16* __restrict__ rel, float* __restrict__ part,         \
      int rows, int heads, int kv_heads, int q0, int window, int slots, int extent, float scaling) {                 \
    ink2_fdec_body<D>(q, kc, vc, rel, part, rows, heads, kv_heads, q0, window, slots, extent, scaling);              \
  }
INK2_FDEC(64)
INK2_FDEC(128)
#undef INK2_FDEC

// The splits of every (row, head) of a launch merged (a block a (row, head) = blockIdx.x = row * heads + head, its
// threads the dims), with the sink and the output gate (the formula above).
extern "C" __global__ void ink2_fdec_merge(const float* __restrict__ part, const __nv_bfloat16* __restrict__ sinks,
                                           const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out,
                                           int D, int splits) {
  const size_t qg = blockIdx.x;
  const float sink = ld(sinks, qg);
  const float* pq = part + qg * splits * (2 + D);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float o = bf(fdec::merge(pq, splits, D, d, sink, true));
    const size_t at = qg * D + d;
    const float gt = ld(gate, at);
    out[at] = __float2bfloat16_rn(o * (1.f / (1.f + expf(-gt))));
  }
}
// ======================================================= end p4-dattn
// ================================================================================================= p4-small
// The small prefill items (out/prefill/p4/p4-small/REPORT.md); every kernel here runs only for a prompt chunk wider
// than max_rows and only with its switch on (Model.rel_wide, Model.fuse): off, phase 3's launches run unchanged.
//
// ink2_attn_tiled_scores_rel (TF_INKLING2_REL_WIDE=2): ink2_attn_tiled_scores with each bias made where it is read,
// from the launch's r rows (bf16 [rows][heads][16], after their scaled clip) and the layer's projection (bf16
// [16][extent]): bias = bf16(sum_k r[k] proj[k][dist]) as a sequential fmaf chain over k = 0 .. 15 from 0.f. The
// reference's rounding point (the bf16 product's output) is kept; the fp32 sum's order is the chain's, not
// cuBLASLt's tensor-core sum (so a bias can differ by one bf16 step from phase 3's: NOT bitwise, measured). The bias
// rows [rows][heads][extent] are never written or read. Everything else is at_scores' arithmetic, unchanged.
namespace {
constexpr int RL_DR = 16;                // d_rel (Kernels.attentionTiledRel checks it)
constexpr int RL_SPAN = AT_QR + AT_QN;   // a block's dists: row r0 .. r0 + 15 against keys p0 .. p0 + 127 (143, padded)

template <int G>
__device__ __forceinline__ void at_scores_rel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                              const __nv_bfloat16* __restrict__ rr,
                                              const __nv_bfloat16* __restrict__ proj, __nv_bfloat16* __restrict__ sc,
                                              int rows, int heads, int kv_heads, int D, int q0, int window, int slots,
                                              int extent, float scaling, int stride, float* smem) {
  float* qs = smem;                    // [AT_QD][AT_QR * G] the stage's query dims, (row, head) contiguous
  float* ks = qs + AT_QD * AT_QR * G;  // [AT_QD][AT_QN] the stage's key dims, key k at k ^ (8 * (d / 8 % 4))
  const int g = blockIdx.z;
  const int r0 = blockIdx.x * AT_QR;
  const int plast = q0 + min(rows - 1, r0 + AT_QR - 1);  // the tile's last row's position: no key past it is read
  const int p0 = (at_lo(q0 + r0, window) & ~7) + blockIdx.y * AT_QN;
  if (p0 > plast) return;
  const int tid = threadIdx.x, r = tid >> 4, kg = tid & 15;
  constexpr int NQ = (AT_QR * G * (AT_QD / 8) + 255) / 256;
  const uint4* qsrc[NQ];
  int qdst[NQ];
#pragma unroll
  for (int i = 0; i < NQ; ++i) {
    const int v = tid + 256 * i, dv = v & 3, rh = v >> 2, rr_ = rh / G, hg = rh - rr_ * G;
    const bool in = v < AT_QR * G * (AT_QD / 8);
    qsrc[i] = in && r0 + rr_ < rows ? reinterpret_cast<const uint4*>(q + ((size_t)(r0 + rr_) * heads + g * G + hg) * D) + dv
                                    : nullptr;
    qdst[i] = in ? dv * 8 * (AT_QR * G) + rh : -1;
  }
  const uint4* ksrc[2];
  int kdst[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int v = tid + 256 * i, dv = v & 3, kk = v >> 2, p = p0 + kk;
    ksrc[i] = p <= plast ? reinterpret_cast<const uint4*>(kc + ((size_t)(p % slots) * kv_heads + g) * D) + dv : nullptr;
    kdst[i] = dv * 8 * AT_QN + (kk ^ (dv << 3));
  }
  uint4 qx[NQ], kx[2];
  auto fetch = [&](int d0) {
#pragma unroll
    for (int i = 0; i < NQ; ++i) qx[i] = qsrc[i] ? __ldg(qsrc[i] + d0 / 8) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < 2; ++i) kx[i] = ksrc[i] ? __ldg(ksrc[i] + d0 / 8) : make_uint4(0, 0, 0, 0);
  };
  fetch(0);
  float acc[G][8];
#pragma unroll
  for (int i = 0; i < G; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for (int d0 = 0; d0 < D; d0 += AT_QD) {
    __syncthreads();  // the previous stage is read
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
      if (qdst[i] < 0) continue;
      float f[8];
      at_bf8(qx[i], f);
#pragma unroll
      for (int e = 0; e < 8; ++e) qs[qdst[i] + e * (AT_QR * G)] = f[e];
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float f[8];
      at_bf8(kx[i], f);
#pragma unroll
      for (int e = 0; e < 8; ++e) ks[kdst[i] + e * AT_QN] = f[e];
    }
    __syncthreads();
    if (d0 + AT_QD < D) fetch(d0 + AT_QD);  // the next stage's loads fly during this stage's products
#pragma unroll 8
    for (int d = 0; d < AT_QD; ++d) {
      const int sw = ((d >> 3) & 3) << 3;
      const float4 ka = *reinterpret_cast<const float4*>(ks + d * AT_QN + ((4 * kg) ^ sw));
      const float4 kb = *reinterpret_cast<const float4*>(ks + d * AT_QN + ((64 + 4 * kg) ^ sw));
      float qv[G];
      at_ldg<G>(qs + d * (AT_QR * G) + r * G, qv);
      const float kv[8] = {ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w};
#pragma unroll
      for (int i = 0; i < G; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(qv[i], kv[j], acc[i][j]);
    }
  }
  // the bias inputs, staged over the (now read) query and key stages: the block's rows' r [AT_QR * G][16] and the
  // projection's columns dmin .. dmin + RL_SPAN - 1 (dist = pos - p of row r0 + rr against key p0 + kk is dmin + rr +
  // 127 - kk); zeros outside [0, extent) (never read). Blocks whose dists all lie at or past the extent stage nothing.
  float* rs = smem;                     // [AT_QR * G][RL_DR]
  float* ps = rs + AT_QR * G * RL_DR;   // [RL_DR][RL_SPAN]
  const int dmin = q0 + r0 - p0 - (AT_QN - 1);
  const bool need = dmin < extent && q0 + r0 + AT_QR - 1 - p0 >= 0;
  __syncthreads();  // every thread's last stage is read
  if (need) {
    for (int v = tid; v < AT_QR * G * RL_DR; v += 256) {
      const int rh = v / RL_DR, kk = v - rh * RL_DR, rr_ = rh / G, hg = rh - rr_ * G;
      rs[v] = r0 + rr_ < rows ? ld(rr, ((size_t)(r0 + rr_) * heads + g * G + hg) * RL_DR + kk) : 0.f;
    }
    for (int v = tid; v < RL_DR * RL_SPAN; v += 256) {
      const int kk = v / RL_SPAN, j = v - kk * RL_SPAN, dist = dmin + j;
      ps[v] = dist >= 0 && dist < extent ? ld(proj, (size_t)kk * extent + dist) : 0.f;
    }
  }
  __syncthreads();
  const int row = r0 + r;
  if (row >= rows) return;
  const int pos = q0 + row, lo8 = at_lo(pos, window) & ~7;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int pr = p0 + half * 64 + 4 * kg;  // the run's first key
    if (pr < lo8 || pr > pos) continue;      // no key of the run in [lo8, pos]
#pragma unroll
    for (int i = 0; i < G; ++i) {
      const int h = g * G + i;
      const float* rv = rs + (r * G + i) * RL_DR;
      uint32_t w[2];
#pragma unroll
      for (int e = 0; e < 4; e += 2) {
        float o[2];
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const int p = pr + e + f;
          float s = bf(bf(acc[i][half * 4 + e + f]) * scaling);
          const int dist = pos - p;
          float bias = 0.f;
          if (dist >= 0 && dist < extent) {
            const float* pc = ps + (dist - dmin);
            float b = 0.f;
#pragma unroll
            for (int kk = 0; kk < RL_DR; ++kk) b = fmaf(rv[kk], pc[kk * RL_SPAN], b);
            bias = bf(b);
          }
          s = bf(s + bias);
          o[f] = s;
        }
        w[e >> 1] = (__float_as_uint(o[0]) >> 16) | (__float_as_uint(o[1]) & 0xFFFF0000u);
      }
      *reinterpret_cast<uint2*>(sc + ((size_t)row * heads + h) * stride + (pr - lo8)) = make_uint2(w[0], w[1]);
    }
  }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(AT_QR * AT_QN / 8) ink2_attn_tiled_scores_rel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc, const __nv_bfloat16* __restrict__ rr,
    const __nv_bfloat16* __restrict__ proj, __nv_bfloat16* __restrict__ sc, int rows, int heads, int kv_heads, int D,
    int q0, int window, int slots, int extent, float scaling, int stride) {
  extern __shared__ float4 at_smem[];
  float* smem = reinterpret_cast<float*>(at_smem);
  if (heads / kv_heads == 6) {
    at_scores_rel<6>(q, kc, rr, proj, sc, rows, heads, kv_heads, D, q0, window, slots, extent, scaling, stride, smem);
  } else {
    at_scores_rel<4>(q, kc, rr, proj, sc, rows, heads, kv_heads, D, q0, window, slots, extent, scaling, stride, smem);
  }
}

// ink2_attn_res_norm (TF_INKLING2_FUSE): ink2_attn_res and the ink2_rmsnorm that reads its output, in one block a
// (row, query): the query's output row is attn_res_body's arithmetic (every partial sum, the softmax and the fmaf sum
// over the sources in the same order), kept in shared memory as the fp32 values ink2_attn_res would write; then
// rmsnorm_body<true>'s arithmetic on it (thread u < 256 sums x^2 over i = u, u + 256, ... in order, six loads in
// flight as there; block_sum's two stages over those 8 warps; x * inv times the weight, one rounding to bf16). The
// same bits as the two launches; the fp32 rows [nq][T][H] are never written. The partial source (pm): 0 bf16 rows
// (the layer input), 1 fp32 rows, 2 ink2_residual's sum made on the fly from the attention's output `o` and the layer
// input `h` (o alone at a block boundary): float(h) + float(o), the add ink2_residual makes (so x.res is never
// written). Query qi's norm weight is w[qi] and its output out[qi] (bf16 [T][H]). Each source row is read from memory
// once: pass 1 keeps what it loads in shared memory (the S block states as bf16, exact; the partial as fp32) and the
// output sum reads them there (the same values, so the same bits); the output row then overwrites the staged partial
// in place (each element by the thread that read it). Dynamic shared memory: H floats + S x H bf16.
namespace {
struct ArnArgs {
  const __nv_bfloat16* w[3];
  __nv_bfloat16* out[3];
};

template <int PM>
__device__ __forceinline__ float arn_x(const __nv_bfloat16* blocks, const void* partial, const __nv_bfloat16* hh,
                                       int boundary, int S, int Smax, int H, size_t row, int s, int i) {
  if (s < S) return ld(blocks, (row * Smax + s) * H + i);
  if constexpr (PM == 1) return reinterpret_cast<const float*>(partial)[row * H + i];
  if constexpr (PM == 2) {
    const float a = ld(reinterpret_cast<const __nv_bfloat16*>(partial), row * H + i);
    return boundary ? a : ld(hh, row * H + i) + a;
  }
  return ld(reinterpret_cast<const __nv_bfloat16*>(partial), row * H + i);
}

// the reductions' shared arrays, declared once in the kernel (a template's own __shared__ arrays would be one set an
// instantiation: three sets, 13,920 bytes, past the 99 KiB a block beside the staged rows)
struct ArnShared {
  float red[64][2][AR_U / 32];
  float inv[64];
  float prob[64];
  float npart[AR_U / 32];
};

template <int PM>
__device__ __forceinline__ void attn_res_norm_body(const __nv_bfloat16* __restrict__ blocks, int S, int Smax,
                                                   const void* __restrict__ partial,
                                                   const __nv_bfloat16* __restrict__ hh, int boundary,
                                                   const float* __restrict__ query, int nq, const ArnArgs& o, int H,
                                                   float eps, ArnShared& sh) {
  extern __shared__ float arn_xs[];  // [H] the staged partial (fp32), then the query's fp32 output row
  __nv_bfloat16* arn_bs = reinterpret_cast<__nv_bfloat16*>(arn_xs + H);  // [S][H] the staged block states
  const size_t row = blockIdx.x / nq;
  const int qi = blockIdx.x - (int)row * nq;
  const int u = threadIdx.x & (AR_U - 1), sg = threadIdx.x / AR_U, groups = blockDim.x / AR_U;
  const int lane = threadIdx.x & 31, wu = u >> 5;  // wu: u's warp among the 8 of a 256-thread block
  float (&red)[64][2][AR_U / 32] = sh.red;
  float* inv = sh.inv;
  float* prob = sh.prob;
  float* npart = sh.npart;
  const float* qv = query + (size_t)qi * H;
  for (int s = sg; s <= S; s += groups) {
    float ss = 0.f, dot = 0.f;
    int i = u;
    for (; i + 3 * AR_U < H; i += 4 * AR_U) {  // four loads in flight, summed in u's order
      float x[4];
      float q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        x[j] = arn_x<PM>(blocks, partial, hh, boundary, S, Smax, H, row, s, i + j * AR_U);
        q[j] = qv[i + j * AR_U];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ss += x[j] * x[j];
        dot += q[j] * x[j];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {  // staged for the output sum (a block state's value is a bf16: exact)
        if (s < S)
          arn_bs[(size_t)s * H + i + j * AR_U] = __float2bfloat16_rn(x[j]);
        else
          arn_xs[i + j * AR_U] = x[j];
      }
    }
    for (; i < H; i += AR_U) {
      const float x = arn_x<PM>(blocks, partial, hh, boundary, S, Smax, H, row, s, i);
      ss += x * x;
      dot += qv[i] * x;
      if (s < S)
        arn_bs[(size_t)s * H + i] = __float2bfloat16_rn(x);
      else
        arn_xs[i] = x;
    }
    for (int o2 = 16; o2 > 0; o2 >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, o2);
      dot += __shfl_xor_sync(0xffffffffu, dot, o2);
    }
    if (lane == 0) {
      red[s][0][wu] = ss;
      red[s][1][wu] = dot;
    }
  }
  __syncthreads();
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int v = w; v < 2 * (S + 1); v += nw) {
    const int s = v >> 1, which = v & 1;
    float a = lane < AR_U / 32 ? red[s][which][lane] : 0.f;
    for (int o2 = 16; o2 > 0; o2 >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o2);
    __syncwarp();
    if (lane == 0) red[s][which][0] = a;
  }
  __syncthreads();
  if (threadIdx.x <= S) {
    const int s = threadIdx.x;
    inv[s] = rsqrtf(red[s][0][0] / (float)H + 1e-6f);
    prob[s] = red[s][1][0] * inv[s] * rsqrtf((float)H);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = -INFINITY;
    for (int s = 0; s <= S; ++s) m = fmaxf(m, prob[s]);
    float z = 0.f;
    for (int s = 0; s <= S; ++s) z += expf(prob[s] - m);
    for (int s = 0; s <= S; ++s) prob[s] = expf(prob[s] - m) / z;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float acc = 0.f;
    for (int s = 0; s <= S; ++s) {
      const float x = s < S ? __bfloat162float(arn_bs[(size_t)s * H + i]) : arn_xs[i];
      acc = fmaf(prob[s], x * inv[s], acc);
    }
    arn_xs[i] = acc;  // what ink2_attn_res stores at out[qi][row][i] (over the staged partial: this thread's element)
  }
  __syncthreads();
  // rmsnorm_body<true> on arn_xs: its 256 threads' sums (the others idle), then its block_sum over their 8 warps
  if (threadIdx.x < AR_U) {
    const int B = AR_U;
    float ss = 0.f;
    int i = threadIdx.x;
    for (; i + 5 * B < H; i += 6 * B) {
      float x[6];
#pragma unroll
      for (int j = 0; j < 6; ++j) x[j] = arn_xs[i + j * B];
#pragma unroll
      for (int j = 0; j < 6; ++j) ss += x[j] * x[j];
    }
    for (; i < H; i += B) {
      const float x = arn_xs[i];
      ss += x * x;
    }
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o2);
    if (lane == 0) npart[w] = ss;
  }
  __syncthreads();
  float tot = lane < AR_U / 32 ? npart[lane] : 0.f;
  for (int o2 = 16; o2 > 0; o2 >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o2);
  const float invn = rsqrtf(tot / (float)H + eps);
  const __nv_bfloat16* wq = o.w[qi];
  __nv_bfloat16* outq = o.out[qi] + row * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) outq[i] = __float2bfloat16_rn(ld(wq, i) * (arn_xs[i] * invn));
}
}  // namespace

extern "C" __global__ void __launch_bounds__(1024) ink2_attn_res_norm(const __nv_bfloat16* __restrict__ blocks, int S,
                                                                      int Smax, const void* __restrict__ partial,
                                                                      int pm, const __nv_bfloat16* __restrict__ hh,
                                                                      int boundary, const float* __restrict__ query,
                                                                      int nq, ArnArgs o, int H, float eps) {
  __shared__ ArnShared sh;
  if (pm == 2)
    attn_res_norm_body<2>(blocks, S, Smax, partial, hh, boundary, query, nq, o, H, eps, sh);
  else if (pm == 1)
    attn_res_norm_body<1>(blocks, S, Smax, partial, hh, boundary, query, nq, o, H, eps, sh);
  else
    attn_res_norm_body<0>(blocks, S, Smax, partial, hh, boundary, query, nq, o, H, eps, sh);
}

// The short convolutions of a wide chunk under TF_INKLING2_FUSE: ink2_sconv_static_seg's loop with the segment's
// inputs loaded before any output is stored (the in-place loop's stores held each next load back) and, for the
// release's K = 4, the window in registers (a compile-time K). Every output is that loop's arithmetic in its order.
//   ink2_sconv_static_seg_ld   in place, as ink2_sconv_static_seg (the attention's and the sliding k / v ones)
//   ink2_sconv_static_seg_out  the MLP's: x (the MLP output) read only, and for each row y = bf16((s + x) / div) (the
//                              value the in-place loop stores), res = float(h) + float(o) (o alone at a block
//                              boundary: ink2_residual's add), h = bf16(res + y) (ink2_layer_out's): the layer input h
//                              overwritten with the layer output, each element by the thread that read it.
// The same bits as ink2_sconv_static_seg (+ ink2_residual + ink2_layer_out).
namespace {
template <int KK, bool OUT>
__device__ __forceinline__ void sconv_seg_ld(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ w,
                                             const float* __restrict__ edges, const __nv_bfloat16* __restrict__ o,
                                             __nv_bfloat16* h, int boundary, int T, int C, int Kr) {
  const int K = KK > 0 ? KK : Kr;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int segs = (T + SC_SEG - 1) / SC_SEG;
  if (i >= (size_t)segs * C) return;
  const int sg = (int)(i / C), c = (int)(i - (size_t)sg * C);
  const int t0 = sg * SC_SEG, n = min(T, t0 + SC_SEG) - t0;
  float xs[SC_SEG], os[OUT ? SC_SEG : 1], hs[OUT ? SC_SEG : 1];
#pragma unroll
  for (int j = 0; j < SC_SEG; ++j) {
    const size_t at = (size_t)(t0 + j) * C + c;
    xs[j] = j < n ? ld(x, at) : 0.f;
    if constexpr (OUT) {
      os[j] = j < n ? ld(o, at) : 0.f;
      hs[j] = j < n && !boundary ? ld(h, at) : 0.f;
    }
  }
  float win[8];
  for (int k = 0; k < K - 1; ++k) win[k] = edges[((size_t)sg * (K - 1) + k) * C + c];
  float wk[8];
  for (int k = 0; k < K; ++k) wk[k] = ld(w, (size_t)c * K + k);
  const float div = sqrtf((float)(K + 1));
#pragma unroll
  for (int j = 0; j < SC_SEG; ++j) {
    if (j >= n) break;
    const float xt = xs[j];
    win[K - 1] = xt;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s = fmaf(wk[k], win[k], s);
    const size_t at = (size_t)(t0 + j) * C + c;
    if constexpr (OUT) {
      const float y = __bfloat162float(__float2bfloat16_rn((s + xt) / div));
      const float res = boundary ? os[j] : hs[j] + os[j];
      h[at] = __float2bfloat16_rn(res + y);
    } else {
      x[at] = __float2bfloat16_rn((s + xt) / div);
    }
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
}
}  // namespace

extern "C" __global__ void ink2_sconv_static_seg_ld(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ w,
                                                    const float* __restrict__ edges, int T, int C, int K) {
  if (K == 4)
    sconv_seg_ld<4, false>(x, w, edges, nullptr, nullptr, 0, T, C, K);
  else
    sconv_seg_ld<0, false>(x, w, edges, nullptr, nullptr, 0, T, C, K);
}

extern "C" __global__ void ink2_sconv_static_seg_out(const __nv_bfloat16* __restrict__ x,
                                                     const __nv_bfloat16* __restrict__ w,
                                                     const float* __restrict__ edges,
                                                     const __nv_bfloat16* __restrict__ o, __nv_bfloat16* h,
                                                     int boundary, int T, int C, int K) {
  __nv_bfloat16* xm = const_cast<__nv_bfloat16*>(x);  // read only on this path
  if (K == 4)
    sconv_seg_ld<4, true>(xm, w, edges, o, h, boundary, T, C, K);
  else
    sconv_seg_ld<0, true>(xm, w, edges, o, h, boundary, T, C, K);
}
// ink2_attention_fa_d64 / _d128 (TF_INKLING2_PREFILL_ATTN=fa): ink2_attention_tc's blocks, tiles, staging and mma
// with the softmax online (FlashAttention 2): ONE sweep over the key tiles, each staging its keys and values; per row
// the running max m (started at the sink) and the lanes' partial exp sums, both rescaled by expf(m_old - m_new) when a
// tile raises the max, as is the p v accumulator; p = bf16(expf(s - m)) feeds the mma unnormalized. At the end
// sum = (the quad's partial sums) + expf(sink - m), o = bf16(acc / sum), then the gate. The scores are made at the
// reference's rounding points (bf16(q . k), * scaling, + bias, as tc's `finish`); p is rounded before its
// normalization (FlashAttention's rounding, not the reference's bf16(p / sum)): NOT bitwise, measured. tc sweeps the
// keys three times (the max, the sum, then p and p v); this sweeps them once.
// ink2_attention_far_d64 / _d128 (with TF_INKLING2_REL_WIDE=2): the same with each tile's relative bias made in the
// kernel from the launch's r rows (bf16 [rows][heads][16]) and the projection (bf16 [16][extent]) instead of read
// from bias rows: a warp's tile of 16 rows x 32 keys needs the bias at the 47 dists (row - key) the tile spans; one
// m16n8k16 product a block of 8 dists (r rows as A, the staged projection columns as B: the same tensor-core sum of
// 16 exact products as cuBLASLt's product), rounded to bf16 (the reference's point) into the warp's [16][48] tile in
// shared memory, read back by each score. Tiles whose dists all lie at or past the extent skip it (bias 0).
namespace {
// Launched with 6 warps (one a query head of the kv group: Kernels.faFits) and bounded for two blocks an SM: at most
// 65536 / (2 x 192) registers a thread (at d 128 the unbounded build took 170, which rounds to 176: one block an SM)
constexpr int FA_THREADS = 192;
constexpr int FA_PW = 72;  // the staged projection's row: 64 dists from an 8-aligned start, + 8 (ldmatrix banks)
constexpr int FA_BW = 48;  // a warp's bias tile row: the 47 (+1) dists of a tile's 16 rows x 32 keys

template <int D, bool REL>
__device__ __forceinline__ void attn_fa_body(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                             const __nv_bfloat16* __restrict__ vc,
                                             const __nv_bfloat16* __restrict__ rel,
                                             const __nv_bfloat16* __restrict__ proj,
                                             const __nv_bfloat16* __restrict__ sinks,
                                             const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out,
                                             int rows, int heads, int kv_heads, int q0, int window, int slots,
                                             int extent, float scaling) {
  constexpr int KS = D / 16;
  constexpr int RW = D + 8;
  constexpr int NT = TC_KEYS / 8;
  constexpr int ST = TC_KEYS * RW;
  extern __shared__ uint4 tc_smem[];
  const uint32_t kt = tc_saddr(tc_smem);  // [2][TC_KEYS][RW] keys
  const uint32_t vt = kt + 2 * ST * 2;    // [2][TC_KEYS][RW] values
  // REL: the tile's projection columns [16][FA_PW] bf16, then each warp's bias tile [16][FA_BW] bf16
  __nv_bfloat16* ps = reinterpret_cast<__nv_bfloat16*>(tc_smem) + 4 * ST;
  const int G = heads / kv_heads;
  const int tiles = (rows + TC_ROWS - 1) / TC_ROWS;
  const int bt = (int)blockIdx.x / kv_heads;
  const int g = (int)blockIdx.x - bt * kv_heads;
  const int tile = tiles - 1 - bt;  // the last rows first
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h = g * G + warp;
  __nv_bfloat16* bw = ps + 16 * FA_PW + warp * 16 * FA_BW;
  const int r0 = tile * TC_ROWS;
  const int rl = min(r0 + TC_ROWS, rows) - 1;
  const int khi = q0 + rl;
  const int klo = window > 0 ? max(0, q0 + r0 - window + 1) : 0;
  const int kb = klo & ~15;
  const int ntile = (khi - kb + TC_KEYS) / TC_KEYS;
  int pos[2], lo[2];
  const __nv_bfloat16* relr[2];
  float sink[2];
  const uint32_t* qr[2];
  uint32_t ra[4];  // REL: the rows' r as the A operand (rows by d_rel), as qa's k-step 0
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = min(r0 + (lane >> 2) + 8 * i, rl);
    pos[i] = q0 + r;
    lo[i] = window > 0 ? max(0, pos[i] - window + 1) : 0;
    relr[i] = REL ? nullptr : rel + ((size_t)r * heads + h) * extent;
    sink[i] = ld(sinks, (size_t)r * heads + h);
    qr[i] = reinterpret_cast<const uint32_t*>(q + ((size_t)r * heads + h) * D);
    if constexpr (REL) {
      const uint32_t* rw = reinterpret_cast<const uint32_t*>(rel + ((size_t)r * heads + h) * 16);
      ra[i] = rw[lane & 3];
      ra[2 + i] = rw[(lane & 3) + 4];
    }
  }
  uint32_t qa[KS][4];
#pragma unroll
  for (int s = 0; s < KS; ++s) {
    const int c = s * 8 + (lane & 3);
    qa[s][0] = qr[0][c];
    qa[s][1] = qr[1][c];
    qa[s][2] = qr[0][c + 4];
    qa[s][3] = qr[1][c + 4];
  }
  auto stage = [&](int t) {
    const int k0 = kb + t * TC_KEYS;
    const uint32_t b = (uint32_t)((t & 1) * ST * 2);
    for (int c = threadIdx.x; c < TC_KEYS * (D / 8); c += blockDim.x) {
      const int kk = c / (D / 8), part = c - kk * (D / 8);
      const int p = k0 + kk;
      const bool in = p >= klo && p <= khi;
      const size_t off = in ? ((size_t)(p % slots) * kv_heads + g) * D + part * 8 : 0;
      const uint32_t dst = b + (uint32_t)((kk * RW + part * 8) * 2);
      tc_cp16(kt + dst, kc + off, in ? 16 : 0);
      tc_cp16(vt + dst, vc + off, in ? 16 : 0);
    }
    tc_commit();
  };
  // tile t's smallest dist (row r0 against its last key) and the 8-aligned start of its staged projection columns
  auto dmin_of = [&](int t) { return q0 + r0 - (kb + t * TC_KEYS + TC_KEYS - 1); };
  auto finish = [&](float acc, int t, int n, int e) -> float {
    const int i = e >> 1;
    const int p = kb + t * TC_KEYS + n * 8 + (lane & 3) * 2 + (e & 1);
    const int dist = pos[i] - p;
    if (p < lo[i] || dist < 0) return -INFINITY;
    const float s = bf(bf(acc) * scaling);
    float bias = 0.f;
    if (dist < extent) {
      if constexpr (REL) {
        // row (lane / 4 + 8 i) of the tile; its dist's column in the warp's bias tile
        bias = __bfloat162float(bw[((lane >> 2) + 8 * i) * FA_BW + (dist - dmin_of(t))]);
      } else {
        bias = ld(relr[i], dist);
      }
    }
    return bf(s + bias);
  };
  float m[2] = {sink[0], sink[1]};  // the running max, the sink included from the start
  float l[2] = {0.f, 0.f};          // the lane's partial exp sums of its row's scores at m
  float oa[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) oa[j][0] = oa[j][1] = oa[j][2] = oa[j][3] = 0.f;
  stage(0);
  for (int t = 0; t < ntile; ++t) {
    if (t + 1 < ntile) {
      stage(t + 1);
      tc_wait<1>();
    } else {
      tc_wait<0>();
    }
    const int dmin = dmin_of(t);
    const bool biased = REL && dmin < extent;  // block-uniform: some dist of the tile below the extent
    if constexpr (REL) {
      if (biased) {
        // the projection's columns d0 .. d0 + 63 (d0 = dmin rounded down to 8; zeros outside [0, extent): whole
        // 8-dist pieces, since 0 and extent are multiples of 8), 16 rows of 8 pieces
        const int d0 = dmin & ~7;
        for (int c = threadIdx.x; c < 16 * 8; c += blockDim.x) {
          const int kk = c >> 3, part = c & 7, d = d0 + part * 8;
          uint4 v = make_uint4(0, 0, 0, 0);
          if (d >= 0 && d < extent) v = __ldg(reinterpret_cast<const uint4*>(proj + (size_t)kk * extent + d));
          *reinterpret_cast<uint4*>(ps + kk * FA_PW + part * 8) = v;
        }
      }
    }
    __syncthreads();
    float sa[NT][4];
    {
      const uint32_t b = kt + (uint32_t)((t & 1) * ST * 2);
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        sa[n][0] = sa[n][1] = sa[n][2] = sa[n][3] = 0.f;
#pragma unroll
        for (int s = 0; s < KS; s += 2) {
          uint32_t r[4];
          tc_ldsm4(r, b + (uint32_t)(((n * 8 + (lane & 7)) * RW + s * 16 + (lane >> 3) * 8) * 2));
          tc_mma(sa[n], qa[s], r[0], r[1]);
          tc_mma(sa[n], qa[s + 1], r[2], r[3]);
        }
      }
    }
    if constexpr (REL) {
      if (biased) {
        // bias[row][d0 + col] for the tile's 16 rows and 64 columns: 8 products of 8 columns, each one k-step
        const int d0 = dmin & ~7, sh = dmin - d0;  // tile column j = staged column j + sh
        const uint32_t pb = tc_saddr(ps);
        const int mm = lane >> 3;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          uint32_t r[4];
          tc_ldsm4t(r, pb + (uint32_t)((((mm & 1) * 8 + (lane & 7)) * FA_PW + j * 8 + (mm >> 1) * 8) * 2));
          float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
          tc_mma(c0, ra, r[0], r[1]);
          tc_mma(c1, ra, r[2], r[3]);
          // fragment (jj, e): row lane / 4 + 8 (e / 2), staged column 8 jj + 2 (lane % 4) + (e % 2)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int row = (lane >> 2) + 8 * (e >> 1);
            const int ca = (j * 8 + (lane & 3) * 2 + (e & 1)) - sh;
            const int cb = ca + 8;
            if (ca >= 0 && ca < FA_BW) bw[row * FA_BW + ca] = __float2bfloat16_rn(c0[e]);
            if (cb >= 0 && cb < FA_BW) bw[row * FA_BW + cb] = __float2bfloat16_rn(c1[e]);
          }
        }
        __syncwarp();
      }
    }
    float tmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        sa[n][e] = finish(sa[n][e], t, n, e);
        tmax[e >> 1] = fmaxf(tmax[e >> 1], sa[n][e]);
      }
    float scale[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      tmax[i] = fmaxf(tmax[i], __shfl_xor_sync(0xffffffffu, tmax[i], 1));
      tmax[i] = fmaxf(tmax[i], __shfl_xor_sync(0xffffffffu, tmax[i], 2));
      const float mn = fmaxf(m[i], tmax[i]);
      scale[i] = expf(m[i] - mn);
      m[i] = mn;
      l[i] *= scale[i];
    }
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
      oa[j][0] *= scale[0];
      oa[j][1] *= scale[0];
      oa[j][2] *= scale[1];
      oa[j][3] *= scale[1];
    }
    uint32_t pa[NT / 2][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      float pv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        pv[e] = expf(sa[n][e] - m[e >> 1]);
        l[e >> 1] += pv[e];
      }
      pa[n >> 1][(n & 1) * 2] = tc_pack(pv[0], pv[1]);
      pa[n >> 1][(n & 1) * 2 + 1] = tc_pack(pv[2], pv[3]);
    }
    const uint32_t b = vt + (uint32_t)((t & 1) * ST * 2);
    const int mm = lane >> 3;
#pragma unroll
    for (int k = 0; k < NT / 2; ++k)
#pragma unroll
      for (int j = 0; j < D / 8; j += 2) {
        uint32_t r[4];
        tc_ldsm4t(r, b + (uint32_t)(((k * 16 + (mm & 1) * 8 + (lane & 7)) * RW + j * 8 + (mm >> 1) * 8) * 2));
        tc_mma(oa[j], pa[k], r[0], r[1]);
        tc_mma(oa[j + 1], pa[k], r[2], r[3]);
      }
    __syncthreads();  // the buffers (and the staged projection) are read before the next stage refills them
  }
  float inv[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float sum = l[i];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    sum += expf(sink[i] - m[i]);
    inv[i] = 1.f / sum;
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int r = r0 + (lane >> 2) + 8 * i;
    if (r > rl) continue;
#pragma unroll
    for (int j = 0; j < D / 8; ++j) {
      const size_t at = ((size_t)r * heads + h) * D + j * 8 + (lane & 3) * 2;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float o = bf(oa[j][2 * i + e] * inv[i]);
        const float gt = ld(gate, at + e);
        out[at + e] = __float2bfloat16_rn(o * (1.f / (1.f + expf(-gt))));
      }
    }
  }
}
}  // namespace

#define INK2_ATTN_FA(D)                                                                                              \
  extern "C" __global__ void __launch_bounds__(FA_THREADS, 2) ink2_attention_fa_d##D(                               \
      const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,                                     \
      const __nv_bfloat16* __restrict__ vc, const __nv_bfloat16* __restrict__ rel,                                   \
      const __nv_bfloat16* __restrict__ sinks, const __nv_bfloat16* __restrict__ gate,                               \
      __nv_bfloat16* __restrict__ out, int rows, int heads, int kv_heads, int q0, int window, int slots, int extent, \
      float scaling) {                                                                                               \
    attn_fa_body<D, false>(q, kc, vc, rel, nullptr, sinks, gate, out, rows, heads, kv_heads, q0, window, slots,     \
                           extent, scaling);                                                                         \
  }                                                                                                                  \
  extern "C" __global__ void __launch_bounds__(FA_THREADS, 2) ink2_attention_far_d##D(                              \
      const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kc,                                     \
      const __nv_bfloat16* __restrict__ vc, const __nv_bfloat16* __restrict__ rr,                                    \
      const __nv_bfloat16* __restrict__ proj, const __nv_bfloat16* __restrict__ sinks,                               \
      const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ out, int rows, int heads, int kv_heads,    \
      int q0, int window, int slots, int extent, float scaling) {                                                    \
    attn_fa_body<D, true>(q, kc, vc, rr, proj, sinks, gate, out, rows, heads, kv_heads, q0, window, slots, extent,   \
                          scaling);                                                                                  \
  }
INK2_ATTN_FA(64)
INK2_ATTN_FA(128)
#undef INK2_ATTN_FA
// ================================================================================================= end p4-small

// ------------------------------------------------------------------- p4-xchg: the MoE's TP sum by token rows
// TF_INKLING2_MOE_XCHG=token (forward.moeTok): each rank first combines its own fp32 down partials (its share of the
// intermediate channels) of a token's routed pairs into one row, and the ranks exchange those rows: 1 / top-k of the
// bytes of the pairs' partials (ink2_combine_parts / _pair / _pair_cols). Per output (row t, column i):
//   r_rank[t][i] = the sum over t's picks in ascending expert id (ink2_combine's order), from zero, of
//                  fmaf(weight, y_rank, sum)  (fp32, one rounding a pick)
//   out[t][i]   = bf16(r_0 + r_1)  (the ranks' rows added in rank order, rounded once: ink2_rank_sum_bf16 or
//                  ink2_rank_sum_pair_bf16, forward.moeTok)
// with the routing weights as ink2_combine takes them (renormalized, times route_scale). What this drops against the
// pair kernels: each pair's bf16 rounding of its rank-summed down output (the reference's bf16 expert output; a rank
// holds only its part of that sum, so the rounding cannot be made before the exchange), and their order of adds (the
// pair's two parts first, then the weighted sum; here the weighted sums first, then the two ranks). The row's one
// bf16 rounding stays the last step, where ink2_combine has it.

// A rank's token rows: y [t][k][n] fp32 (n = H; or a column part's width, pair p's part at y + p n, as the down
// form's column parts write them) into out [t][n] fp32. A block a (row, 4 blockDim columns), four columns a thread
// (n % 4 == 0, y and out 16-byte aligned); the block's first thread puts the row's routed picks in ascending expert id
// (stable; unrouted -1 picks dropped) and their weights into shared memory.
extern "C" __global__ void ink2_xchg_tok_rows(const float* __restrict__ y, const int* __restrict__ picks,
                                              const float* __restrict__ weights, float* __restrict__ out, int k,
                                              int n) {
  const int per = blockDim.x * 4;
  const int chunks = (n + per - 1) / per;
  const size_t t = blockIdx.x / chunks;
  const int i = (blockIdx.x % chunks) * per + threadIdx.x * 4;
  __shared__ int s_slot[16];
  __shared__ float s_w[16];
  __shared__ int s_n;
  if (threadIdx.x == 0) {
    int order[16];
    for (int j = 0; j < k; ++j) order[j] = j;
    for (int a = 1; a < k; ++a) {  // ascending expert id, stable
      const int v = order[a];
      int b = a;
      while (b > 0 && picks[t * k + order[b - 1]] > picks[t * k + v]) {
        order[b] = order[b - 1];
        --b;
      }
      order[b] = v;
    }
    int m = 0;
    for (int j = 0; j < k; ++j) {
      const int s = order[j];
      if (picks[t * k + s] < 0) continue;
      s_slot[m] = s;
      s_w[m] = weights[t * k + s];
      ++m;
    }
    s_n = m;
  }
  __syncthreads();
  if (i >= n) return;
  const int m = s_n;
  const float* yr = y + t * k * (size_t)n + i;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int j = 0; j < m; ++j) {
    const float4 v = *reinterpret_cast<const float4*>(yr + (size_t)s_slot[j] * n);
    const float w = s_w[j];
    acc.x = fmaf(w, v.x, acc.x);
    acc.y = fmaf(w, v.y, acc.y);
    acc.z = fmaf(w, v.z, acc.z);
    acc.w = fmaf(w, v.w, acc.w);
  }
  *reinterpret_cast<float4*>(out + t * (size_t)n + i) = acc;
}

// ------------------------------------------------------------------- end p4-xchg

// ------------------------------------------------------------------- x-mixed: the mixed store's item split
// TF_INKLING2_MIX=1 (forward.moeMix): every expert resident as its EXL3 trellis, and a native bank (the arena) of the
// release's own NVFP4 / MXFP8 bytes. A layer's grouped plan over expert ids (items (expert, first, count), counts[0] of
// them; members = pair ids) is split by the bank's map nmap (the arena LUT's row of the layer: the bank slot of expert
// e, or -1): an item whose expert the bank holds goes to items_n with its bank slot, any other to items_x with the EXL3
// store's slot xbase + e (layer E + expert: exl3.Store's slot_ptr / slot_k). A stable compaction: each list keeps the
// plan's order, first and count are copied, so both lists walk the same members and every pair is in exactly one list.
// counts_n / counts_x: [0] the list's items, [1] its distinct experts (the plan's counts[1] split the same way: an
// expert's items are adjacent in the plan). One block (blockDim a multiple of 32, at most 1024); any item count, in
// passes of blockDim with carried totals. Ids and counts only: nothing is computed on values.
extern "C" __global__ void ink2_mix_split(const int* __restrict__ items, const int* __restrict__ counts,
                                          const int* __restrict__ nmap, int xbase, int* __restrict__ items_n,
                                          int* __restrict__ counts_n, int* __restrict__ items_x,
                                          int* __restrict__ counts_x) {
  __shared__ int wn[32], wh[32];  // a pass's native items and native expert heads by warp, then their inclusive scans
  __shared__ int tot_n, tot_hn, tot_hx;
  const int n = counts[0];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = blockDim.x >> 5;
  if (tid == 0) {
    tot_n = 0;
    tot_hn = 0;
    tot_hx = 0;
  }
  __syncthreads();
  for (int at = 0; at < n; at += blockDim.x) {
    const int it = at + tid;
    const bool live = it < n;
    const int e = live ? items[3 * it] : -1;
    const int slot = (live && e >= 0) ? nmap[e] : -1;
    const bool nat = live && slot >= 0;
    const bool head = live && (it == 0 || items[3 * (it - 1)] != e);  // its expert's first item
    const unsigned bn = __ballot_sync(0xffffffffu, nat);
    const unsigned bh = __ballot_sync(0xffffffffu, nat && head);
    const unsigned bx = __ballot_sync(0xffffffffu, live && !nat && head);
    if (lane == 0) {
      wn[warp] = __popc(bn);
      wh[warp] = __popc(bh);
      if (bx) atomicAdd(&tot_hx, __popc(bx));  // a count: its order does not matter
    }
    __syncthreads();
    if (warp == 0) {
      int v = lane < warps ? wn[lane] : 0;
      int h = lane < warps ? wh[lane] : 0;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, o);
        const int g = __shfl_up_sync(0xffffffffu, h, o);
        if (lane >= o) {
          v += u;
          h += g;
        }
      }
      wn[lane] = v;
      wh[lane] = h;
    }
    __syncthreads();
    // the pass's native items before this one: the warps before it, then the lanes before it in its warp
    const int before = (warp ? wn[warp - 1] : 0) + __popc(bn & ((1u << lane) - 1u));
    if (live) {
      const int first = items[3 * it + 1], cnt = items[3 * it + 2];
      int* o = nat ? items_n + 3 * (tot_n + before) : items_x + 3 * (it - tot_n - before);
      o[0] = nat ? slot : xbase + e;
      o[1] = first;
      o[2] = cnt;
    }
    __syncthreads();  // every thread has read tot_n before it moves on
    if (tid == 0) {
      tot_n += wn[warps - 1];
      tot_hn += wh[warps - 1];
    }
    __syncthreads();
  }
  if (tid == 0) {
    counts_n[0] = tot_n;
    counts_n[1] = tot_hn;
    counts_x[0] = n - tot_n;
    counts_x[1] = tot_hx;
  }
}
// ------------------------------------------------------------------- end x-mixed

// ==== e-step (prefill lane phase 5): a decode / verify chunk's memory-bound launches fused (TF_INKLING2_STEP_FUSE) ====
// Each kernel below is the unfused launches' arithmetic, element for element and in the same order: the same bits,
// fewer launches (forward.layers, chunks of at most max_rows rows). The rewind's histories are kept as the narrow
// convolution keeps them (hist: the K - 1 ring rows, then the T inputs).
namespace {
// ink2_sconv_static's body for one channel (in place: y may be x), its history when `hist`
__device__ __forceinline__ void es_sconv_static(const __nv_bfloat16* x, __nv_bfloat16* y,
                                                const __nv_bfloat16* __restrict__ w, float* __restrict__ ring, int T,
                                                int C, int K, float* __restrict__ hist, int c) {
  float win[8];
  for (int k = 0; k < K - 1; ++k) win[k] = ring[(size_t)k * C + c];
  if (hist != nullptr) {
    for (int k = 0; k < K - 1; ++k) hist[(size_t)k * C + c] = win[k];
    for (int t = 0; t < T; ++t) hist[(size_t)(K - 1 + t) * C + c] = ld(x, (size_t)t * C + c);
  }
  float wk[8];
  for (int k = 0; k < K; ++k) wk[k] = ld(w, (size_t)c * K + k);
  const float div = sqrtf((float)(K + 1));
  for (int t = 0; t < T; ++t) {
    const float xt = ld(x, (size_t)t * C + c);
    win[K - 1] = xt;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s = fmaf(wk[k], win[k], s);
    y[(size_t)t * C + c] = __float2bfloat16_rn((s + xt) / div);
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
  for (int k = 0; k < K - 1; ++k) ring[(size_t)k * C + c] = win[k];
}
}  // namespace

// The sliding layers' k and v short convolutions in one launch: blockIdx.y 0 = (x0, w0, ring0, hist0), 1 = the other
// set; each ink2_sconv_static's (in place).
extern "C" __global__ void ink2_es_sconv_static2(__nv_bfloat16* x0, const __nv_bfloat16* __restrict__ w0,
                                                 float* __restrict__ ring0, float* __restrict__ hist0,
                                                 __nv_bfloat16* x1, const __nv_bfloat16* __restrict__ w1,
                                                 float* __restrict__ ring1, float* __restrict__ hist1, int T, int C,
                                                 int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  if (blockIdx.y == 0)
    es_sconv_static(x0, x0, w0, ring0, T, C, K, hist0, c);
  else
    es_sconv_static(x1, x1, w1, ring1, T, C, K, hist1, c);
}

// o_proj's TP sum and the attention's short convolution: x = bf16(the parts [world][T][C] summed in rank order)
// (ink2_rank_sum_bf16's adds and rounding), the conv input; its history and ring as ink2_sconv_static's; the conv's
// output into out [T][C] (bf16). ink2_rank_sum_bf16 + ink2_sconv_static (in place on out), the same bits.
extern "C" __global__ void ink2_es_rank_sum_sconv(const float* __restrict__ parts, int world,
                                                  const __nv_bfloat16* __restrict__ w, float* __restrict__ ring,
                                                  float* __restrict__ hist, __nv_bfloat16* __restrict__ out, int T,
                                                  int C, int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const size_t n = (size_t)T * C;
  float win[8];
  for (int k = 0; k < K - 1; ++k) win[k] = ring[(size_t)k * C + c];
  if (hist != nullptr)
    for (int k = 0; k < K - 1; ++k) hist[(size_t)k * C + c] = win[k];
  float wk[8];
  for (int k = 0; k < K; ++k) wk[k] = ld(w, (size_t)c * K + k);
  const float div = sqrtf((float)(K + 1));
  for (int t = 0; t < T; ++t) {
    const size_t i = (size_t)t * C + c;
    float sum = parts[i];
    for (int r = 1; r < world; ++r) sum = sum + parts[(size_t)r * n + i];
    const float xt = __bfloat162float(__float2bfloat16_rn(sum));  // ink2_rank_sum_bf16's output: the conv's input
    if (hist != nullptr) hist[(size_t)(K - 1 + t) * C + c] = xt;
    win[K - 1] = xt;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s = fmaf(wk[k], win[k], s);
    out[i] = __float2bfloat16_rn((s + xt) / div);
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
  for (int k = 0; k < K - 1; ++k) ring[(size_t)k * C + c] = win[k];
}

// The MLP's short convolution with the layer output add: x (the MLP output, read only) through ink2_sconv_static's
// loop (ring, history), each row's conv output y = bf16((s + x) / div) (not stored), res = float(h) + float(o) (o
// alone at a block boundary: ink2_residual's add), h = bf16(res + y) (ink2_layer_out's), the layer input h
// overwritten with the layer output, each element by the thread that read it. ink2_sconv_static + ink2_residual +
// ink2_layer_out, the same bits (x.res never written).
extern "C" __global__ void ink2_es_sconv_static_out(const __nv_bfloat16* __restrict__ x,
                                                    const __nv_bfloat16* __restrict__ w, float* __restrict__ ring,
                                                    const __nv_bfloat16* __restrict__ o, __nv_bfloat16* h, int boundary,
                                                    int T, int C, int K, float* __restrict__ hist) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float win[8];
  for (int k = 0; k < K - 1; ++k) win[k] = ring[(size_t)k * C + c];
  if (hist != nullptr) {
    for (int k = 0; k < K - 1; ++k) hist[(size_t)k * C + c] = win[k];
    for (int t = 0; t < T; ++t) hist[(size_t)(K - 1 + t) * C + c] = ld(x, (size_t)t * C + c);
  }
  float wk[8];
  for (int k = 0; k < K; ++k) wk[k] = ld(w, (size_t)c * K + k);
  const float div = sqrtf((float)(K + 1));
  for (int t = 0; t < T; ++t) {
    const size_t i = (size_t)t * C + c;
    const float xt = ld(x, i);
    win[K - 1] = xt;
    float s = 0.f;
    for (int k = 0; k < K; ++k) s = fmaf(wk[k], win[k], s);
    const float y = __bfloat162float(__float2bfloat16_rn((s + xt) / div));
    const float a = ld(o, i);
    const float res = boundary ? a : ld(h, i) + a;
    h[i] = __float2bfloat16_rn(res + y);
    for (int k = 0; k < K - 1; ++k) win[k] = win[k + 1];
  }
  for (int k = 0; k < K - 1; ++k) ring[(size_t)k * C + c] = win[k];
}

// The layer's four head preparations (q, k, v, r: ink2_head_prep's calls) in one launch: block b belongs to call j
// with first[j] <= b < first[j + 1] (first[j] the blocks before it: rows x heads of each earlier call), and runs that
// call's ink2_head_prep block (its row and head, 128 threads, the same arithmetic and block_sum). pos0 and eps are
// the calls' common values; pos0 stays one kernel argument (graphs.zig rewrites it on replay).
struct EsPrep {
  __nv_bfloat16* v[4];
  const __nv_bfloat16* scale[4];
  int heads[4], D[4], mode[4], tau_on[4];
  float thr[4], alpha[4], n_floor[4];
  int first[5];
};

extern "C" __global__ void ink2_es_head_prep4(EsPrep a, float eps, int pos0) {
  int j = 0;
  while (j < 3 && (int)blockIdx.x >= a.first[j + 1]) ++j;
  const int b = (int)blockIdx.x - a.first[j];
  const int heads = a.heads[j], D = a.D[j], mode = a.mode[j], tau_on = a.tau_on[j];
  const float thr = a.thr[j], alpha = a.alpha[j], n_floor = a.n_floor[j];
  const __nv_bfloat16* scale = a.scale[j];
  const int row = b / heads, h = b - row * heads;
  __nv_bfloat16* p = a.v[j] + ((size_t)row * heads + h) * D;
  float ss = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float x = __bfloat162float(p[i]);
    if (mode == 0) x *= ld(scale, (size_t)h * D + i);
    ss += x * x;
  }
  const float tot = block_sum(ss);
  float f;
  if (mode == 2) {
    f = rsqrtf(tot / (float)D + eps);
  } else {
    f = fminf(thr / (sqrtf(tot) + 1e-6f), 1.0f);
  }
  float tau = 1.f;
  if (tau_on) tau = 1.f + alpha * logf(fmaxf((float)(pos0 + row + 1) / n_floor, 1.0f));
  __syncthreads();
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float x = __bfloat162float(p[i]);
    if (mode == 0) x *= ld(scale, (size_t)h * D + i);
    float o = bf(x * f);
    if (tau_on) o = bf(o * tau);
    p[i] = __float2bfloat16_rn(o);
  }
}
// ==== end e-step

// ==== e-step: the router's top-k pick, compact (TF_INKLING2_STEP_FUSE) ==============================================
// ink2_route's outputs, bit for bit, from a small kernel: ink2_route is 32-way unrolled over a lane's experts with a
// branch, a sigmoid and a division each (about 4,800 SASS instructions, one warp a pass), and takes ~16 us a launch
// whatever its selection (e-decode measured a rank selection at the same time). Here a block a (row, pass), 256
// threads, each expert's key made by the same expression (sigmoid(logit) + bias in fp32, the masked pass skipping
// what `routable` excludes), then k rounds of a block argmax under ink2_route's order (the greater key first, the
// lower id on a tie), the picked expert taken out of the next rounds; the pick's weight from the same sigmoid, summed
// over the picks in pick order and normalized as there. Pass 0 (blockIdx.y 0) writes picks / weights, pass 1 (no
// mask) want_ids / want_scores (when want_ids is set).
extern "C" __global__ void __launch_bounds__(256) ink2_es_route(const float* __restrict__ logits,
                                                                const float* __restrict__ bias,
                                                                const uint8_t* __restrict__ routable, int T, int E,
                                                                int k, float route_scale, int* __restrict__ picks,
                                                                float* __restrict__ weights,
                                                                int* __restrict__ want_ids,
                                                                float* __restrict__ want_scores) {
  constexpr int NONE = 0x7fffffff;
  constexpr int PER = 4;  // experts a thread at most (E <= 1024)
  const int t = blockIdx.x, pass = blockIdx.y;
  if (t >= T) return;
  if (pass == 1 && want_ids == nullptr) return;
  const float* lg = logits + (size_t)t * E;
  __shared__ float wv[8];
  __shared__ int we[8];
  __shared__ int ids[16];
  float key[PER];
  bool live[PER];
#pragma unroll
  for (int m = 0; m < PER; ++m) {
    const int e = threadIdx.x + 256 * m;
    key[m] = 0.f;
    live[m] = false;
    if (e < E && (pass == 1 || routable == nullptr || routable[e] != 0)) {
      const float s = 1.f / (1.f + expf(-lg[e]));
      key[m] = s + bias[e];
      live[m] = true;
    }
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int n = 0;
  for (int r = 0; r < k; ++r) {
    float bv = 0.f;
    int be = NONE;
#pragma unroll
    for (int m = 0; m < PER; ++m)  // ascending e within the thread: a later one leads only on a strictly greater key
      if (live[m] && (be == NONE || key[m] > bv)) {
        bv = key[m];
        be = threadIdx.x + 256 * m;
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (oe != NONE && (be == NONE || ov > bv || (ov == bv && oe < be))) {
        bv = ov;
        be = oe;
      }
    }
    if (lane == 0) {
      wv[warp] = bv;
      we[warp] = be;
    }
    __syncthreads();
    if (warp == 0) {
      bv = lane < 8 ? wv[lane] : 0.f;
      be = lane < 8 ? we[lane] : NONE;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oe = __shfl_xor_sync(0xffffffffu, be, o);
        if (oe != NONE && (be == NONE || ov > bv || (ov == bv && oe < be))) {
          bv = ov;
          be = oe;
        }
      }
      if (lane == 0) we[0] = be;
    }
    __syncthreads();
    const int pick = we[0];
    __syncthreads();  // we[0] read by every thread before the next round writes it
    if (pick == NONE) break;  // fewer candidates than k
#pragma unroll
    for (int m = 0; m < PER; ++m)
      if (pick == threadIdx.x + 256 * m) live[m] = false;
    if (threadIdx.x == 0) ids[r] = pick;
    n = r + 1;
  }
  if (threadIdx.x != 0) return;
  float sum = 0.f;
  float sc[16];
  for (int j = 0; j < n; ++j) {
    sc[j] = 1.f / (1.f + expf(-lg[ids[j]]));
    sum += sc[j];
  }
  for (int j = 0; j < k; ++j) {
    if (pass == 0) {
      picks[(size_t)t * k + j] = j < n ? ids[j] : -1;
      weights[(size_t)t * k + j] = j < n ? sc[j] / (sum + 1e-20f) * route_scale : 0.f;
    } else {
      want_ids[(size_t)t * k + j] = ids[j];
      want_scores[(size_t)t * k + j] = sc[j];
    }
  }
}
// ==== end e-step

// ==== e-step: the batched head's argmax without its logits' exchange (TF_INKLING2_STEP_FUSE, argmax-only windows) ====
// ink2_logits' argmax over a row's every real token (the softcap tanh(x / cap) * cap of each, the greater value first,
// the lower index on a tie: a total order) taken in two steps: each rank over its own vocabulary columns (global index
// lo + i, i < its real columns) into a (value, index) pair a row (ink2_es_head_best: ink2_logits' scan and tree), the
// pairs exchanged (16 bytes a row instead of a row's whole logits), then every row's pick over the ranks
// (ink2_es_head_pick). Under a total order the pick is the one-pass argmax's: the same argmaxes.
extern "C" __global__ void ink2_es_head_best(const __nv_bfloat16* __restrict__ rows, int vl, int lo, int unpadded,
                                             float cap, float2* __restrict__ best) {
  const size_t r = blockIdx.x;
  const int B = blockDim.x;
  const int n = min(vl, max(0, unpadded - lo));
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i0 = threadIdx.x; i0 < n; i0 += 8 * B) {
    float lv[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = i0 + u * B;
      lv[u] = i < n ? ld(rows, r * vl + i) : 0.f;
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int i = i0 + u * B;
      if (i >= n) break;
      const float l = tanhf(lv[u] / cap) * cap;
      const int g = lo + i;
      if (l > bv || (l == bv && g < bi)) {
        bv = l;
        bi = g;
      }
    }
  }
  auto take = [](float& b, int& x, float ob, int ox) {
    if (ob > b || (ob == b && ox < x)) {
      b = ob;
      x = ox;
    }
  };
  for (int o = 16; o > 0; o >>= 1) take(bv, bi, __shfl_xor_sync(0xffffffffu, bv, o), __shfl_xor_sync(0xffffffffu, bi, o));
  __shared__ float sv[32];
  __shared__ int sx[32];
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) {
    sv[w] = bv;
    sx[w] = bi;
  }
  __syncthreads();
  if (w == 0) {
    float b = l < (B >> 5) ? sv[l] : -INFINITY;
    int x = l < (B >> 5) ? sx[l] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) take(b, x, __shfl_xor_sync(0xffffffffu, b, o), __shfl_xor_sync(0xffffffffu, x, o));
    if (l == 0) best[r] = make_float2(b, __int_as_float(x));
  }
}

// Each row's pick over the ranks' pairs [world][stride] (rank order), the same order: argmax[r].
extern "C" __global__ void ink2_es_head_pick(const float2* __restrict__ all, int world, int T, int stride,
                                             int* __restrict__ argmax) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= T) return;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int w = 0; w < world; ++w) {
    const float2 p = all[(size_t)w * stride + r];
    const int g = __float_as_int(p.y);
    if (g == 0x7fffffff) continue;
    if (p.x > bv || (p.x == bv && g < bi)) {
      bv = p.x;
      bi = g;
    }
  }
  argmax[r] = bi;
}
// ==== end e-step
