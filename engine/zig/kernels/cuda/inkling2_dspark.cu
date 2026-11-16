// Inkling-2's DSpark draft head (mtp.safetensors): the block drafter and its Markov walk. Ported from the verified
// overlay port overlay/families/inkling2/dspark.py (itself vLLM PR #67325's math; docs/dspark-spec.md, gates G1-G10),
// at its rounding points: r(x) there is __float2bfloat16_rn here. Dims are the release's: hidden 4608, 48 query
// heads, 8 kv heads, head dim 128, Markov rank 256. Dense products are cuBLASLt's (gemm.zig); these are the rest.

#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int HD = 128;
constexpr int HALF = 64;
__device__ __forceinline__ float ld(const __nv_bfloat16* p, size_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float rb(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Block-wide sum of one float (blockDim.x a multiple of 32, <= 1024).
__device__ float block_sum(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  float s = 0.f;
  if (threadIdx.x < 32) {
    s = threadIdx.x < nw ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (threadIdx.x == 0) red[0] = s;
  }
  __syncthreads();
  s = red[0];
  __syncthreads();
  return s;
}

// The pick rule on every rank's (value, local index) [world][2]: the largest value, the lowest global id on ties.
__device__ __forceinline__ int pick_of(const float* all, int world, int per_rank) {
  float v = -INFINITY;
  int id = 0x7fffffff;
  for (int r = 0; r < world; ++r) {
    const float x = all[2 * r];
    const int li = __float_as_int(all[2 * r + 1]);
    if (li == 0x7fffffff) continue;
    const int g = r * per_rank + li;
    if (x > v || (x == v && g < id)) {
      v = x;
      id = g;
    }
  }
  return id;
}

__device__ __forceinline__ float dot8(uint4 wv, const float* x) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&wv);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 f = __bfloat1622float2(p[k]);
    s = fmaf(f.x, x[2 * k], s);
    s = fmaf(f.y, x[2 * k + 1], s);
  }
  return s;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
}  // namespace

// mix_rmsnorm (K:13-67): one block a row. taps [rows][8][H] bf16; mixed = sum_t taps[t] * alpha[t] in fp32, tap
// order; out = r(mixed * rsqrt(sum(mixed^2) / H + eps) * w).
extern "C" __global__ void dsp_mix_norm(const __nv_bfloat16* __restrict__ taps, const float* __restrict__ alpha,
                                       const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int H,
                                       int ntaps, float eps) {
  __shared__ float red[32];
  const size_t row = blockIdx.x;
  const __nv_bfloat16* t = taps + row * ntaps * H;
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float m = 0.f;
    for (int k = 0; k < ntaps; ++k) m = m + ld(t, (size_t)k * H + i) * alpha[k];
    ss += m * m;
  }
  const float inv = rsqrtf(block_sum(ss, red) / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float m = 0.f;
    for (int k = 0; k < ntaps; ++k) m = m + ld(t, (size_t)k * H + i) * alpha[k];
    out[row * H + i] = __float2bfloat16_rn(m * inv * ld(w, i));
  }
}

// flashinfer rmsnorm (M:66) and fused_add_rmsnorm (M:68): one block a row. add 0: out = r(x * inv * w), res = x;
// add 1: s = x + res (fp32), out = r(s * inv * w), res = r(s).
extern "C" __global__ void dsp_rmsnorm(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* res,
                                      const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out, int H,
                                      float eps, int add) {
  __shared__ float red[32];
  const size_t row = blockIdx.x;
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float s = ld(x, row * H + i);
    if (add) s = s + ld(res, row * H + i);
    ss += s * s;
  }
  const float inv = rsqrtf(block_sum(ss, red) / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float s = ld(x, row * H + i);
    if (add) s = s + ld(res, row * H + i);
    out[row * H + i] = __float2bfloat16_rn(s * inv * ld(w, i));
    res[row * H + i] = __float2bfloat16_rn(s);  // add 0: the residual is x itself
  }
}

// Per-head norm and neox RoPE, in place on x [rows][heads][128] bf16; one block (64 threads) a (row, head).
// mode 0 (qk_norm_rope, K:70-145): inv = 1 / sqrt(ms + eps) (IEEE), halves r(x * inv * w), roped in fp32, stored.
// mode 1 (rms_norm then rope_only, M:260-262): r(x * rsqrt(ms + eps) * w) over the head, then roped, stored.
// mode 2 (rope_only, K:148-165). cs [positions][128] fp32 (cos 64 | sin 64); a row's position pos[row].
extern "C" __global__ void dsp_norm_rope(__nv_bfloat16* x, const __nv_bfloat16* __restrict__ w,
                                        const float* __restrict__ cs, const int* __restrict__ pos, int heads,
                                        float eps, int mode) {
  __shared__ float red[32];
  const int row = blockIdx.x / heads, head = blockIdx.x % heads;
  __nv_bfloat16* h = x + ((size_t)row * heads + head) * HD;
  const int d = threadIdx.x;  // 0..63
  float a = __bfloat162float(h[d]), b = __bfloat162float(h[d + HALF]);
  if (mode != 2) {
    const float ss = block_sum(a * a + b * b, red);
    const float ms = ss / HD;
    if (mode == 0) {
      const float inv = 1.0f / sqrtf(ms + eps);
      a = rb(a * inv * ld(w, d));
      b = rb(b * inv * ld(w, d + HALF));
    } else {
      const float inv = rsqrtf(ms + eps);
      a = rb(a * inv * ld(w, d));
      b = rb(b * inv * ld(w, d + HALF));
    }
  }
  const float* c = cs + (size_t)pos[row] * HD;
  const float cv = c[d], sv = c[HALF + d];
  h[d] = __float2bfloat16_rn(a * cv - b * sv);
  h[d + HALF] = __float2bfloat16_rn(b * cv + a * sv);
}

// The draft's attention (M:195-248): one block a (block row, query head), 128 threads. Keys: the context cache's
// visible positions [lo, anchor) (a ring of `slots` [slot][kvh][128], slot = pos % slots), then the 8 block rows
// (bidirectional). scores fp32 * d^-0.5, softmax fp32 (P unrounded), out = r(sum p v). scr [rows][heads][keys] fp32.
extern "C" __global__ void dsp_attention(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kb,
                                        const __nv_bfloat16* __restrict__ vb, const __nv_bfloat16* __restrict__ kc,
                                        const __nv_bfloat16* __restrict__ vc, int slots, int lo, int anchor, int L,
                                        int heads, int kvh, float* scr, int max_keys, __nv_bfloat16* __restrict__ out) {
  __shared__ float red[32];
  __shared__ float qs[HD];
  const int row = blockIdx.x / heads, head = blockIdx.x % heads, kh = head / (heads / kvh);
  const int nctx = anchor - lo, nk = nctx + L;
  float* s = scr + (size_t)blockIdx.x * max_keys;
  qs[threadIdx.x] = ld(q, ((size_t)row * heads + head) * HD + threadIdx.x);
  __syncthreads();
  const float scale = rsqrtf((float)HD);
  float mx = -INFINITY;
  for (int j = threadIdx.x; j < nk; j += blockDim.x) {
    const __nv_bfloat16* k = j < nctx ? kc + ((size_t)((lo + j) % slots) * kvh + kh) * HD
                                      : kb + ((size_t)(j - nctx) * kvh + kh) * HD;
    float dot = 0.f;
    for (int i = 0; i < HD; ++i) dot = fmaf(qs[i], __bfloat162float(k[i]), dot);
    dot *= scale;
    s[j] = dot;
    mx = fmaxf(mx, dot);
  }
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = -INFINITY;
  for (int i = 0; i < (int)(blockDim.x >> 5); ++i) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float sum = 0.f;
  for (int j = threadIdx.x; j < nk; j += blockDim.x) {
    const float e = expf(s[j] - mx);
    s[j] = e;
    sum += e;
  }
  sum = block_sum(sum, red);
  __syncthreads();
  const int d = threadIdx.x;  // one output dim a thread
  float acc = 0.f;
  for (int j = 0; j < nk; ++j) {
    const __nv_bfloat16* v = j < nctx ? vc + ((size_t)((lo + j) % slots) * kvh + kh) * HD
                                      : vb + ((size_t)(j - nctx) * kvh + kh) * HD;
    acc = fmaf(s[j], __bfloat162float(v[d]), acc);
  }
  out[((size_t)row * heads + head) * HD + d] = __float2bfloat16_rn(acc / sum);
}

// Rows [n][kvw] of k / v into the context ring at positions pos0 .. pos0 + n - 1.
extern "C" __global__ void dsp_ctx_put(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                                      __nv_bfloat16* kc, __nv_bfloat16* vc, int slots, int pos0, int kvw) {
  const int r = blockIdx.x;
  const size_t at = (size_t)((pos0 + r) % slots) * kvw;
  for (int i = threadIdx.x; i < kvw; i += blockDim.x) {
    kc[at + i] = k[(size_t)r * kvw + i];
    vc[at + i] = v[(size_t)r * kvw + i];
  }
}

// silu_and_mul (K:168-188): r(g / (1 + exp(-g)) * u), fp32, IEEE divide.
extern "C" __global__ void dsp_silu_mul(const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
                                       __nv_bfloat16* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gv = __bfloat162float(g[i]);
  out[i] = __float2bfloat16_rn(gv / (1.0f + expf(-gv)) * __bfloat162float(u[i]));
}

// Rows 1..L-1 of the block: mask_embed (M:428). Row 0 is written by the target's embedding and norm.
extern "C" __global__ void dsp_mask_rows(const __nv_bfloat16* __restrict__ mask, __nv_bfloat16* x, int H, int L) {
  const int r = blockIdx.x + 1;
  for (int i = threadIdx.x; i < H; i += blockDim.x) x[(size_t)r * H + i] = mask[i];
}

// The Markov head's inputs: rows [n][2H] = [hidden_i | f_rnn] (M:369).
extern "C" __global__ void dsp_cat_rnn(const __nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ frnn,
                                      __nv_bfloat16* out, int H) {
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    out[(size_t)r * 2 * H + i] = hidden[(size_t)r * H + i];
    out[(size_t)r * 2 * H + H + i] = frnn[i];
  }
}

// The global pick of every rank's (value, local index) [world][2] (rank order = ascending vocabulary): the largest
// value, the lowest id on ties; written as the next step's prev and into toks[step].
extern "C" __global__ void dsp_pick(const float* __restrict__ all, int world, int per_rank, int* prev, int* toks,
                                   int step) {
  const int id = pick_of(all, world, per_rank);
  prev[0] = id;
  toks[step] = id;
}

// fold (M:348-360) in fp64: fw[v][j] = sum_h head[v][h] * proj[h][j] (proj = hidden_out.weight [H][r]), fb[v] = sum_h
// head[v][h] * pb[h]; stored bf16. One block a vocabulary row, r threads (+ the bias by thread 0's loop).
extern "C" __global__ void dsp_fold(const __nv_bfloat16* __restrict__ head, const __nv_bfloat16* __restrict__ proj,
                                   const __nv_bfloat16* __restrict__ pb, int H, int r, __nv_bfloat16* fw,
                                   __nv_bfloat16* fb) {
  extern __shared__ double hrow[];
  const size_t v = blockIdx.x;
  for (int h = threadIdx.x; h < H; h += blockDim.x) hrow[h] = (double)__bfloat162float(head[v * H + h]);
  __syncthreads();
  const int j = threadIdx.x;
  double acc = 0.0;
  for (int h = 0; h < H; ++h) acc += hrow[h] * (double)__bfloat162float(proj[(size_t)h * r + j]);
  fw[v * r + j] = __double2bfloat16(acc);
  if (j == 0) {
    double b = 0.0;
    for (int h = 0; h < H; ++h) b += hrow[h] * (double)__bfloat162float(pb[h]);
    fb[v] = __double2bfloat16(b);
  }
}

// The target's taps: rows [T][H] of `src` into taps [T][ntaps][H] at tap `which`.
extern "C" __global__ void dsp_tap(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* taps, int H, int ntaps,
                                  int which) {
  const size_t t = blockIdx.x;
  for (int i = threadIdx.x; i < H; i += blockDim.x) taps[(t * ntaps + which) * H + i] = src[t * H + i];
}

// ---- TP: the draft's layers split like the target's (o_proj / down_proj fp32 partials over this rank's heads /
// columns, gathered in rank order).

// The rank-order sum of fp32 partials parts [world][rows][H] (s = p0; s = s + p_r, as ink2_rank_sum), rounded once
// to bf16 (the reference's bf16 GEMM output), then dsp_rmsnorm on it: add 0: out = r(x * inv * w), res = x; add 1:
// s = x + res (fp32), out = r(s * inv * w), res = r(s). One block a row; dynamic shared H floats.
extern "C" __global__ void dsp_sum_rmsnorm(const float* __restrict__ parts, int world, long long stride,
                                          __nv_bfloat16* res, const __nv_bfloat16* __restrict__ w,
                                          __nv_bfloat16* __restrict__ out, int H, float eps, int add) {
  __shared__ float red[32];
  extern __shared__ float sx[];
  const size_t row = blockIdx.x;
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float s = parts[row * H + i];
    for (int r = 1; r < world; ++r) s = s + parts[(size_t)r * stride + row * H + i];
    float x = rb(s);
    if (add) x = x + ld(res, row * H + i);
    sx[i] = x;
    ss += x * x;
  }
  const float inv = rsqrtf(block_sum(ss, red) / H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    out[row * H + i] = __float2bfloat16_rn(sx[i] * inv * ld(w, i));
    res[row * H + i] = __float2bfloat16_rn(sx[i]);
  }
}

// ---- the Markov walk, two kernels a step (and one gather of 8 bytes a rank between them at TP).


// One Markov step's recurrence (M:375-383): prev = the anchor (step 0) or the previous step's global pick (written to
// toks[step - 1]); raw = [state | markov_w1[prev]] . rnn_proj[:, :2r]^T (fp32) + (ctx + bias); markov_gate -> the
// next state (state_out: ping-pong, every block reads state_in whole) and latent. r = 256, 2r = 512: a warp an index
// j (its rows j, r + j, 2r + j), 8 warps a block, r / 8 blocks.
extern "C" __global__ void dsp_walk_step(const float* __restrict__ best_all, int world, int per_rank, int anchor,
                                        int step, int* __restrict__ toks, const __nv_bfloat16* __restrict__ state_in,
                                        __nv_bfloat16* __restrict__ state_out, const __nv_bfloat16* __restrict__ w1,
                                        const __nv_bfloat16* __restrict__ w_step, const float* __restrict__ ctx,
                                        const float* __restrict__ bias, __nv_bfloat16* __restrict__ latent, int r) {
  __shared__ __align__(16) float inp[512];
  const int prev = step == 0 ? anchor : pick_of(best_all, world, per_rank);
  if (step > 0 && blockIdx.x == 0 && threadIdx.x == 0) toks[step - 1] = prev;
  for (int i = threadIdx.x; i < 2 * r; i += blockDim.x)
    inp[i] = i < r ? __bfloat162float(state_in[i]) : __bfloat162float(w1[(size_t)prev * r + (i - r)]);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j = blockIdx.x * (blockDim.x >> 5) + warp;
  if (j >= r) return;
  float raw[3];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const uint4* row = reinterpret_cast<const uint4*>(w_step + (size_t)(g * r + j) * 2 * r);
    float s = 0.f;
    for (int c = lane; c < 2 * r / 8; c += 32) s += dot8(row[c], inp + 8 * c);
    raw[g] = warp_sum(s);
  }
  if (lane != 0) return;
  const float g0 = rb(raw[0] + (ctx[j] + bias[j]));
  const float c0 = rb(raw[1] + (ctx[r + j] + bias[r + j]));
  const float o0 = rb(raw[2] + (ctx[2 * r + j] + bias[2 * r + j]));
  const float s = __bfloat162float(state_in[j]);
  const float g = rb(1.0f / (1.0f + expf(-g0)));
  const float c = rb(tanhf(c0));
  const float keep = rb(g * s);
  const float upd = rb(rb(1.0f - g) * c);
  state_out[j] = __float2bfloat16_rn(keep + upd);
  latent[j] = __float2bfloat16_rn(tanhf(o0));
}

// One step's logits on this rank's vocabulary rows and their argmax, in one launch: mm = latent . fold_w[j] (fp32),
// bias = r(mm + fold_bias), x = r(base + bias), logit = r(tanh(x / cap) * cap); the first maximum (NaN as -inf)
// among ids < valid. A warp a row (fold rows are r = 256 bf16: 8 a lane), grid-stride; each block's best into
// part[block], the last block to finish reduces them (block order, lowest id on ties) into best (value, local index
// as float bits) and re-arms the counter.
extern "C" __global__ void dsp_walk_fold(const __nv_bfloat16* __restrict__ latent,
                                        const __nv_bfloat16* __restrict__ fold_w, const __nv_bfloat16* __restrict__ fb,
                                        const __nv_bfloat16* __restrict__ base, int valid, float cap, float* part,
                                        unsigned* counter, float* best) {
  __shared__ __align__(16) float lat[256];
  __shared__ float wv[32];
  __shared__ int wi[32];
  __shared__ bool last;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lat[i] = __bfloat162float(latent[i]);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  float v = -INFINITY;
  int id = 0x7fffffff;
  for (int j = blockIdx.x * nw + warp; j < valid; j += gridDim.x * nw) {
    const uint4 w = reinterpret_cast<const uint4*>(fold_w + (size_t)j * 256)[lane];
    const float mm = warp_sum(dot8(w, lat + 8 * lane));
    const float bias = rb(mm + __bfloat162float(fb[j]));
    const float x = rb(__bfloat162float(base[j]) + bias);
    float l = rb(tanhf(x / cap) * cap);
    if (l != l) l = -INFINITY;
    if (l > v || (l == v && j < id)) {
      v = l;
      id = j;
    }
  }
  if (lane == 0) {
    wv[warp] = v;
    wi[warp] = id;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < nw; ++k)
      if (wv[k] > v || (wv[k] == v && wi[k] < id)) {
        v = wv[k];
        id = wi[k];
      }
    part[2 * blockIdx.x] = v;
    part[2 * blockIdx.x + 1] = __int_as_float(id);
    __threadfence();
    last = atomicAdd(counter, 1u) == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  // the last block: every block's best, block order
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int b = threadIdx.x; b < (int)gridDim.x; b += blockDim.x) {
    const float x = ((volatile float*)part)[2 * b];
    const int i = __float_as_int(((volatile float*)part)[2 * b + 1]);
    if (x > bv || (x == bv && i < bi)) {
      bv = x;
      bi = i;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float x = __shfl_xor_sync(0xffffffffu, bv, o);
    const int i = __shfl_xor_sync(0xffffffffu, bi, o);
    if (x > bv || (x == bv && i < bi)) {
      bv = x;
      bi = i;
    }
  }
  __syncthreads();
  if (lane == 0) {
    wv[warp] = bv;
    wi[warp] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int k = 1; k < nw; ++k)
      if (wv[k] > bv || (wv[k] == bv && wi[k] < bi)) {
        bv = wv[k];
        bi = wi[k];
      }
    best[0] = bv;
    best[1] = __int_as_float(bi);
    *counter = 0u;
  }
}

// A call's positions pos[i] = pos0 + i (i < n <= blockDim.x) and, anchor >= 0, ids[0] = anchor: no host upload.
extern "C" __global__ void dsp_positions(int* pos, int pos0, int n, int* ids, int anchor) {
  const int i = threadIdx.x;
  if (i < n) pos[i] = pos0 + i;
  if (i == 0 && anchor >= 0) ids[0] = anchor;
}

// ======================================================= p4-dattn: the draft's attention, split-K flash decoding
// TF_INKLING2_DRAFT_ATTN=1 (Model.draft_attn): dsp_fdec then dsp_fdec_merge instead of dsp_attention
// (inkling2_fdec.cuh: one block a (key split, kv head) with the block's 8 rows x the kv group's query heads, so each
// context key and value is read once, not once a (row, head)). The same keys (the context's [lo, anchor) then the 8
// block rows, every row sees every key) and the same math up to the sums' order and p's precision: s = (q . k) fp32
// (mma sums) * d^-0.5; per split the online softmax (m, l = sum exp(s - m) fp32, o = sum bf16(exp(s - m)) v on the
// mma: the reference keeps p in fp32); the merge in split order, out = r(o / L). part: [rows][heads][splits][2 + 128].
#include "inkling2_fdec.cuh"

namespace {
struct DspFd {
  struct Query {
    float* slot;
  };
  const __nv_bfloat16* q;
  const __nv_bfloat16* kb;
  const __nv_bfloat16* vb;
  const __nv_bfloat16* kc;
  const __nv_bfloat16* vc;
  float* part;
  int slots, nctx, heads, kvh, G, g, nkeys, nq, sbase;
  float scale;

  __device__ __forceinline__ Query query(int qi) const {
    const int r = qi / G, h = g * G + (qi - r * G);
    return {part + ((size_t)r * heads + h) * gridDim.x * (2 + HD)};
  }
  __device__ __forceinline__ const __nv_bfloat16* qrow(int qi) const {
    const int r = qi / G, h = g * G + (qi - r * G);
    return q + ((size_t)r * heads + h) * HD;
  }
  // key j: context position lo + j (ring slot (lo + j) % slots, nctx <= slots: one wrap at most), then block row
  // j - nctx
  __device__ __forceinline__ bool rows(int j, const __nv_bfloat16*& k, const __nv_bfloat16*& v) const {
    if (j >= nkeys) return false;
    if (j < nctx) {
      int sl = sbase + j;
      if (sl >= slots) sl -= slots;
      const size_t off = ((size_t)sl * kvh + g) * HD;
      k = kc + off;
      v = vc + off;
    } else {
      const size_t off = ((size_t)(j - nctx) * kvh + g) * HD;
      k = kb + off;
      v = vb + off;
    }
    return true;
  }
  __device__ __forceinline__ float score(float dot, const Query&, int j) const {
    return j < nkeys ? dot * scale : -INFINITY;
  }
  __device__ __forceinline__ float* slot(const Query& x) const { return x.slot; }
};
}  // namespace

// grid (splits, kvh), 32 x ceil(L x heads / kvh / 16) threads, fdec::smem_bytes(128) of dynamic shared memory
extern "C" __global__ void __launch_bounds__(256) dsp_fdec(const __nv_bfloat16* __restrict__ q,
                                                           const __nv_bfloat16* __restrict__ kb,
                                                           const __nv_bfloat16* __restrict__ vb,
                                                           const __nv_bfloat16* __restrict__ kc,
                                                           const __nv_bfloat16* __restrict__ vc,
                                                           float* __restrict__ part, int slots, int lo, int anchor,
                                                           int L, int heads, int kvh) {
  DspFd p;
  p.q = q;
  p.kb = kb;
  p.vb = vb;
  p.kc = kc;
  p.vc = vc;
  p.part = part;
  p.slots = slots;
  p.nctx = anchor - lo;
  p.heads = heads;
  p.kvh = kvh;
  p.G = heads / kvh;
  p.g = blockIdx.y;
  p.nkeys = p.nctx + L;
  p.nq = L * p.G;
  p.sbase = lo % slots;
  p.scale = rsqrtf((float)HD);
  fdec::partial<HD>(p);
}

// The splits of every (row, head) merged in split order (a block a (row, head), 128 threads: a dim each).
extern "C" __global__ void dsp_fdec_merge(const float* __restrict__ part, __nv_bfloat16* __restrict__ out,
                                          int splits) {
  const size_t qg = blockIdx.x;
  const int d = threadIdx.x;
  out[qg * HD + d] = __float2bfloat16_rn(fdec::merge(part + qg * splits * (2 + HD), splits, HD, d, 0.f, false));
}
// ======================================================= end p4-dattn
