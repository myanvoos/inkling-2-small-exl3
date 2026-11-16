//! One TP rank of Inkling-2 on its GPU: the dense weights by role (dense.zig), the routed experts' arena (TideLM's
//! residency: slots of packed stored bytes, a device LUT [layer][expert] -> slot and a routable mask), one sequence's
//! state (K/V caches: a sliding layer's ring of window + chunk rows, a global layer's whole context; the convolutions'
//! fp32 rings) and the forward's scratch for up to `max_rows` rows. The forward is forward.zig.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const plan_mod = @import("plan.zig");
const dense_mod = @import("dense.zig");
const experts_mod = @import("experts.zig");
const kern = @import("kern.zig");
const defaults = @import("defaults.zig"); // the user, 2026-11-15: everything on by default
const page_cache = @import("cache.zig"); // e-load (x5): the page cache a load leaves, dropped
const gemm = @import("gemm.zig");
const fill = @import("fill.zig");
const graphs_mod = @import("graphs.zig");
const lazy_mod = @import("lazy.zig"); // p4-mem
const exl3_mod = @import("exl3.zig");
const dense_exl3 = @import("dense_exl3.zig"); // e-dense
const hcap_mod = @import("hcap.zig"); // e-dense

const Config = config.Config;
const Buf = cuda.DeviceBuffer;

pub const Layer = struct {
    local: bool,
    boundary: bool,
    fmt: config.Format,
    wq: u64,
    wk: u64,
    wv: u64,
    wr: u64,
    wo: u64,
    wgate: u64,
    wsink: u64,
    qscale: u64,
    relscale: u64,
    relproj: u64,
    extent: usize,
    kconv: u64 = 0, // sliding: static k / v convolutions [C][1][K]
    vconv: u64 = 0,
    kstatic: u64 = 0, // global: static taps [K][C] and input-generated taps' projection [Hkv K][H]
    kproj: u64 = 0,
    vstatic: u64 = 0,
    vproj: u64 = 0,
    norm_attn: u64 = 0, // sliding: input_layernorm
    norm_q: u64 = 0, // global: q / k / v input norms
    norm_k: u64 = 0,
    norm_v: u64 = 0,
    norm_mlp: u64,
    conv_attn: u64,
    conv_mlp: u64,
    router: u64,
    bias: u64,
    attn_query: u64 = 0, // fp32 [nq][H] (none for layer 0)
    nq: usize = 0,
    mlp_query: u64,
    /// Canonical TP (Model.canon): o_proj's two column halves [H][qw / 2], contiguous (a world-2 rank's own wo).
    wo_half: [2]u64 = .{ 0, 0 },
};

/// The routed experts' slots on this rank. A slot holds one expert's packed share (experts.zig); `lut` [L][E] i32 maps
/// (layer, expert) to its slot in the layer format's pool (-1: not resident), `routable` [L][E] u8 is the masked
/// router's view of the same. The host copies are what a swap edits before it uploads them.
pub const Arena = struct {
    nv_up: Buf,
    nv_down: Buf,
    nv_up_scale: Buf, // [slots][2] f32: g13, g13
    nv_down_scale: Buf, // [slots][1] f32: g2
    nv_slots: usize,
    mx_up: Buf,
    mx_down: Buf,
    mx_slots: usize,
    lut: Buf,
    routable: Buf,
    host_lut: []i32,
    host_routable: []u8,
    words: [2][2]usize, // [format][up, down] words a slot
    /// Slots [0, resident[f]) of format f's pool hold the resident set; [resident[f], slots) are exact mode's
    /// transient ring (none when 0). owner[f][slot]: layer * E + expert in that slot, or -1.
    resident: [2]usize,
    owner: [2][]i32,

    pub fn deinit(a: *Arena, gpa: std.mem.Allocator) void {
        for ([_]*Buf{ &a.nv_up, &a.nv_down, &a.nv_up_scale, &a.nv_down_scale, &a.mx_up, &a.mx_down, &a.lut, &a.routable }) |b| b.free();
        gpa.free(a.host_lut);
        gpa.free(a.host_routable);
        gpa.free(a.owner[0]);
        gpa.free(a.owner[1]);
    }

    pub fn slots(a: *const Arena, f: usize) usize {
        return if (f == 0) a.nv_slots else a.mx_slots;
    }

    /// Bytes one slot of format `f` takes on the device (packed words and, NVFP4, its global scales).
    pub fn slotBytes(words: [2][2]usize, f: usize) usize {
        return (words[f][0] + words[f][1]) * 4 + (if (f == 0) @as(usize, 12) else 0);
    }
};

/// Speculation's target side (dspark.zig drafts): every forwarded row's 8 taps for the draft's context, and each
/// convolution ring's history so a verify window can be cut back to the rows it keeps (rewind).
pub const Spec = struct {
    /// The draft's taps: the inputs of these target layers, then (tap 7) the final-norm output.
    tap_layers: []const usize = &.{ 5, 11, 17, 23, 29, 35, 41 },
    taps: Buf, // [max_rows][8][H] bf16: the last chunk's rows
    hist: [][4]Buf, // per layer: k, v, attn, mlp rings' [(K - 1) + max_rows][C] fp32
    widths: [4]usize, // C of each ring kind
    hist_all: [4]Buf, // each kind's histories, one block: [L][(K - 1) + max_rows][C] (hist[li][kind] its slices)

    /// Rows [T][H] of `src` into tap `which` of the chunk's taps: one strided copy (rows 8 taps apart).
    pub fn tap(sp: *Spec, m: *Model, src: u64, T: usize, which: usize) !void {
        const H = m.c.hidden;
        try copy2d(m, sp.taps.ptr + which * H * 2, 8 * H * 2, src, H * 2, H * 2, T);
    }
};

/// `rows` rows of `width` bytes, device to device, each side's rows its pitch apart: one copy on the model's stream.
pub fn copy2d(m: *const Model, dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, rows: usize) !void {
    if (rows == 0 or width == 0) return;
    const c: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = src_pitch, .dst = dst, .dst_pitch = dst_pitch, .width = width, .height = rows };
    try m.d.check(m.d.api.cuMemcpy2DAsync_v2(&c, m.s.handle), "cuMemcpy2DAsync");
}

pub const State = struct {
    kc: []Buf,
    vc: []Buf,
    slots: []usize,
    ring_k: []Buf,
    ring_v: []Buf,
    ring_attn: []Buf,
    ring_mlp: []Buf,
    rings_all: [4]Buf = undefined, // each ring kind's layers, one block [L][K - 1][C] (ring_k[li] ... its slices)
    len: usize = 0,
};

pub const Scratch = struct {
    ids: Buf,
    emb: Buf,
    emb_all: Buf,
    h: Buf,
    blocks: Buf,
    ar: Buf,
    xq: Buf,
    xk: Buf,
    xv: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    r: Buf,
    gate: Buf,
    sinks: Buf,
    dyn: Buf,
    dyn32: Buf,
    rel: Buf,
    scores: Buf,
    attn: Buf,
    o32: Buf,
    o_all: Buf,
    o: Buf,
    res: Buf,
    xm: Buf,
    lg: Buf,
    picks: Buf,
    wts: Buf,
    want_ids: Buf,
    want_sc: Buf,
    members: Buf,
    items: Buf,
    counts: Buf,
    plan_rank: Buf,
    plan_hist: Buf,
    act: Buf,
    y: Buf,
    y_all: Buf,
    ysum: Buf,
    mo: Buf,
    fin: Buf,
    head: Buf,
    head_all: Buf,
    head_rows: Buf, // [max_rows][Vl] bf16: a batched head (argmax-only chunks)
    head_rows_all: Buf, // [world][max_rows][Vl]
    head_cat: Buf, // [max_rows][V]
    logits: Buf,
    rows_f32: Buf, // xs: [graphs.max_rows][unpadded] fp32: a graphed sampled verify window's rows (graphs.Head.rows)
    argmax: Buf,
    demand: Buf, // fp32 [L][E][3] (residency.demand_fields), accumulated by every forward until zeroed
    edges: Buf, // a wide prompt chunk's parallel convolutions' saved inputs (kern.sconvEdgeBytes); made by growPrompt
    mxq: Buf, // p4-mx: [R] q8 rows of x.xm (Kernels.q8Stride(H) bytes each) for the W8A8 MXFP8 experts; none when off
    // the EXL3 export's experts (forward.moeExl3; 16 bytes each for the release): the activations as fp16
    // ([R][H] for w13, [pairs][Il] for w2), each pair's rotated rows ([pairs][H] / [pairs][Il]), w13's raw sums
    // [pairs][2 Il] fp32, and the w2 activations' per-pair amax ([pairs], every rank's [W][pairs]: NVFP4 under TP)
    e_xq: Buf,
    e_xr: Buf,
    e_z: Buf,
    e_amax: Buf,
    e_amax_all: Buf,
    // canonical TP (Model.canon; 16 bytes each otherwise): a half-width product's rows before they move into place,
    // and the EXL3 down projection's two K parts [2][pairs][H] fp32
    canon_tmp: Buf,
    e_zd: Buf,
    // x-mixed (Model.mix; none otherwise): the plan's items split by the native bank (forward.moeMix, 3 ints an item:
    // the bank's with their bank slots, the rest with the EXL3 store's slots layer E + expert) and their counts ([0]
    // items, [1] experts), and the EXL3 SwiGLU rows bf16 [pairs][Il] (x.act holds the native MXFP8 gate-up's q8 rows at
    // another stride under TF_INKLING2_MX_ACT; a bank pair's row here is never read)
    items_n: Buf,
    counts_n: Buf,
    items_x: Buf,
    counts_x: Buf,
    e_act: Buf,
    // x-mixed (Model.mix; none otherwise): the EXL3 store's own (layer, expert) -> slot map, layer E + expert, i32
    // [L][E]. The arena's LUT is the bank's in the mixed store (a bank slot or -1), but e-prefill's
    // TF_INKLING2_EXL3_ACTROT kernel (exl3ActRot13, forward.exl3Up) looks each pick's EXL3 slot up in a LUT; given the
    // bank's, it skipped every pick the bank does not hold (-1: its rotated row never written) and rotated the bank's with
    // another expert's vectors.
    xlut: Buf,
};

/// TF_INKLING2_EXL3_ACT: the EXL3 experts' input quantization. `native` (the default): the release's NVFP4 (layers
/// whose release format is NVFP4, per 16 under the row's amax) or MXFP8 (per 32) activation quantization the export was
/// calibrated and measured with (kortexa model.Layer.activation; K3's DENSE_ACT=a8 analog); `bf16`: none (the bf16
/// activations as fp16).
pub const Exl3Act = enum { native, bf16 };

/// TF_ROCE_MAX_KB in KiB (GLM's and MiniMax's switch), bounded by the RDMA scatter/gather length.
fn roceBytes(text: ?[]const u8) !usize {
    const v = text orelse return 4096 << 10;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadRoceCap;
    if (n == 0 or n > std.math.maxInt(u32) / 1024) return error.BadRoceCap;
    return n * 1024;
}

/// A wide prompt chunk's attention kernel (Model.prefill_attn, TF_INKLING2_PREFILL_ATTN).
/// p4-small: `fa` the tensor cores with an online softmax (one sweep over the keys; kern.attentionFa; not bitwise).
pub const PrefillAttn = enum { gqa, tiled, tc, fa };

/// p4-xchg: the MoE's TP sum (Model.moe_xchg, TF_INKLING2_MOE_XCHG): `pair` (phase 3, the default) exchanges every
/// routed pair's fp32 down partials and rounds each pair's sum to bf16 before the combine; `token` combines each rank's
/// own partials of a token's picks first and exchanges one fp32 row a token (forward.moeTok; not phase 3's bits).
pub const MoeXchg = enum { pair, token };

/// TF_INKLING2_PREFILL_OVERLAP: the most column parts of a wide chunk's overlapped expert exchange (0: off), default 6
/// (forward.overlapPartsFor takes fewer for a narrower chunk).
fn overlapParts(text: ?[]const u8) !usize {
    const v = text orelse return 6;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadPrefillOverlap;
    if (n > Overlap.max_parts) return error.BadPrefillOverlap;
    return n;
}

/// p4-small: an opt-in switch (`1`, or anything but `0` and empty, turns it on; unset is off).
fn envOn(name: [*:0]const u8) bool {
    const v = std.mem.span(std.c.getenv(name) orelse return false);
    return v.len > 0 and !std.mem.eql(u8, v, "0");
}

/// The user, 2026-11-15 ("Turn everything on by default"): envOn's reading when set; unset is on (off under
/// TF_INKLING2_LEGACY=1, the old default).
fn envOnDefault(name: [*:0]const u8) bool {
    return if (std.c.getenv(name) != null) envOn(name) else !defaults.legacy();
}

/// forward.moeOverlap's second stream and its events: `down[q]` marks part q of the down projection written (the
/// model stream), `recvd[q]` part q's exchange done (the comm stream), so each side waits only for what it reads.
pub const Overlap = struct {
    pub const max_parts = 18; // the hidden size's part_cols tiles at most (4608 / 256)
    comm: cuda.Stream,
    down: [max_parts]cuda.Event,
    recvd: [max_parts]cuda.Event,

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Overlap {
        const o = try gpa.create(Overlap);
        errdefer gpa.destroy(o);
        // the highest priority: the block scheduler starts NCCL's few CTAs before more of the down form's queued ones
        o.comm = try cuda.Stream.initPriority(d, .highest);
        errdefer o.comm.deinit();
        var made: usize = 0;
        errdefer for (0..made) |q| {
            o.down[q].deinit();
            o.recvd[q].deinit();
        };
        while (made < max_parts) : (made += 1) {
            o.down[made] = try cuda.Event.init(d, false);
            o.recvd[made] = cuda.Event.init(d, false) catch |err| {
                o.down[made].deinit();
                return err;
            };
        }
        return o;
    }

    fn deinit(o: *Overlap, gpa: std.mem.Allocator) void {
        o.comm.deinit(); // synchronizes first
        for (&o.down, &o.recvd) |*a, *b| {
            a.deinit();
            b.deinit();
        }
        gpa.destroy(o);
    }
};

// ---- p4-dattn: the decode attention switches (Model.decode_attn, Model.draft_attn)

/// An on/off switch: 1 / on / true on, 0 / off / false off, unset `default`.
fn envFlag(name: [*:0]const u8, default: bool) bool {
    const v = std.mem.span(std.c.getenv(name) orelse return default);
    if (std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "on") or std.ascii.eqlIgnoreCase(v, "true")) return true;
    if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false")) return false;
    std.log.warn("inkling2: {s}={s} is not 0 or 1; {s}", .{ std.mem.span(name), v, if (default) "1" else "0" });
    return default;
}

/// The decode attention's key splits [global layers, sliding layers]: TF_INKLING2_DECODE_SPLITS=G,S, else
/// `global` (two blocks an SM over the kv heads, kern.zig) and 8 (a sliding layer's ~1K keys in runs of ~4 tiles).
fn fdecSplits(global: usize) ![2]usize {
    const v = std.mem.span(std.c.getenv("TF_INKLING2_DECODE_SPLITS") orelse return .{ global, 8 });
    var it = std.mem.tokenizeScalar(u8, v, ',');
    var out: [2]usize = undefined;
    for (&out) |*x| {
        x.* = std.fmt.parseInt(usize, it.next() orelse return error.BadDecodeSplits, 10) catch return error.BadDecodeSplits;
        if (x.* == 0 or x.* > kern.Kernels.fdec_max_splits) return error.BadDecodeSplits;
    }
    return out;
}
// ---- end p4-dattn

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: kern.Kernels,
    g: gemm.Gemms,
    c: Config,
    rank: usize,
    world: usize,
    group: ?cuda.tp_group.Group,
    dense: dense_mod.Dense,
    layers: []Layer,
    embed: u64,
    embed_norm: u64,
    norm: u64,
    unembed: u64,
    final_query: u64,
    vocab_lo: usize,
    arena: Arena,
    arena_on: bool = false,
    /// Exact streaming (the quality baseline): every unmasked pick not resident is loaded into the transient ring
    /// before its layer's experts run, so nothing is substituted. Off: TideLM's masked router (full substitution).
    exact: bool = false,
    loader: ?*fill.Loader = null,
    transient_used: [2]usize = .{ 0, 0 },
    exact_loads: usize = 0,
    spec: ?*Spec = null,
    state: State,
    x: Scratch,
    max_rows: usize,
    /// Rows a prompt chunk may take (prefill), at least max_rows: growPrompt re-makes the row-scaled scratch, the
    /// sliding layers' K / V rings and the draft taps for it; verify, decode and graphs keep max_rows.
    prompt_rows: usize,
    /// A prompt chunk wider than max_rows at two ranks splits its expert sum by rows (forward.moeSplit): each rank
    /// receives the other's fp32 partials of its own rows only and the bf16 halves are gathered after. Off
    /// (TF_INKLING2_PREFILL_SPLIT=0): every pair's partials gathered, as a verify window's are.
    split_moe: bool,
    /// A prompt chunk wider than max_rows runs its own kernels (TF_INKLING2_PREFILL_KERNELS=0: the verify path's):
    /// the attention with each kv head's query heads in one block (ink2_attention_gqa), the short convolutions in
    /// parallel over rows (ink2_sconv_*_seg), o_proj's sum split by rows at two ranks; each the same bits.
    prefill_kernels: bool,
    /// p3-expert: a prompt chunk wider than max_rows plans an MXFP8 layer's routed pairs in items of up to 128
    /// (kern.wide_tile) on the shared-memory wide form (kern.zig mx_wide_*: the CTA's warps share every staged weight
    /// stage), r1 / r4's sums pair for pair; TF_INKLING2_PREFILL_EXPERTS=0: r4 on 64-pair items. NVFP4 layers keep the
    /// fact forms (a 128-pair NVFP4 form measured slower, REPORT.md).
    prefill_experts: bool,
    /// A wide prompt chunk's attention (with prefill_kernels; TF_INKLING2_PREFILL_ATTN): `tiled` (the default) the
    /// three tiled launches (Kernels.attentionTiled), `gqa` phase 2's ink2_attention_gqa — both ink2_attention's bits —
    /// or `tc` (opt-in: ink2_attention_tc on tensor cores, the reference's rounding points but its bf16 matmuls summed
    /// in the mma's order: NOT bitwise; divergence measured in out/prefill/p3/p3-tc/REPORT.md).
    prefill_attn: PrefillAttn,
    /// A wide prompt chunk's expert exchange at two ranks overlapped with its down projection (forward.moeOverlap):
    /// the down form in up to this many column parts, each part's rows sent on `ov`'s stream while the next computes
    /// (TF_INKLING2_PREFILL_OVERLAP: 0 = moeSplit's serialized exchange, N = up to N parts; default 6). Same bits.
    overlap: usize,
    ov: ?*Overlap = null,
    /// p4-cache: TF_INKLING2_PROMPT_REUSE=1 asks the server to resume a request from the state the last one left
    /// (serve/reuse.zig; snapshots by saveState below). Default 0: every request prefills from position 0.
    prompt_reuse: bool,
    // ---- p4-small: the small prefill items (out/prefill/p4/p4-small/REPORT.md). Each switch is OFF by default:
    // off, a chunk runs phase 3's launches (the same bits). On, only prompt chunks wider than max_rows change.
    /// TF_INKLING2_REL_WIDE: a wide chunk's relative bias rows (rel . rel_logits_proj.proj): 0 in max_rows-row pieces
    /// (phase 3), 1 one product per chunk, 2 made inside the tiled scores kernel (the bias rows never written; only
    /// with the tiled attention, else 1).
    rel_wide: u8,
    /// TF_INKLING2_ABSORB_WIDE=1: the draft head absorbs a wide chunk's taps in one piece (dspark.absorbWide).
    absorb_wide: bool,
    /// TF_INKLING2_FUSE=1: a wide chunk's memory-bound kernels fused (forward.layers' p4-small blocks): the block
    /// attention residuals with their rmsnorm, the residual add folded into its two readers, the MLP's short
    /// convolution with the layer's output add.
    fuse: bool,
    // ---- end p4-small
    /// e-step: TF_INKLING2_STEP_FUSE=1: a decode / verify chunk (at most max_rows rows) takes the fused memory-bound
    /// launches too (forward.stepFused; FUSE's attn_res + norm forms, the MLP conv with the layer output, the k / v
    /// convs, the head preps, o_proj's sum with its conv): the same bits as the unfused launches. Off by default.
    step_fuse: bool = false,
    /// e-step: TF_INKLING2_DENSE_GROUP=1: a layer's 6-bit EXL3 projections of the attention input as one projection
    /// group (forward.projGroup; e-dense's sets): two launches instead of up to ten, the same bits. Off by default.
    dense_group: bool = false,
    /// p4-mx: the MXFP8 layers' routed experts in the checkpoint's declared W8A8 numerics (TF_INKLING2_MX_ACT=1; off
    /// by default): both products take MXFP8 activations on the block-scaled e4m3 mma (forward.moe, kern.expertsQ8).
    mx_act: bool = false,
    /// p4-xchg: the MoE's TP sum (TF_INKLING2_MOE_XCHG=pair|token, default pair; every rank the same, agreed at load):
    /// `token` exchanges each token's combined fp32 row instead of its picks' partials (forward.moeTok).
    moe_xchg: MoeXchg = .pair,
    context: usize,
    bufs: std.ArrayList(Buf) = .empty,
    /// The decode / verify forwards captured as CUDA graphs (forward.zig, graphs.zig); dropped when what they read moves.
    graphs: graphs_mod.Graphs = .{},
    /// xs: TF_INKLING2_SAMPLED_GRAPH (on by default; 0 or TF_INKLING2_LEGACY=1: eager): a sampled verify window (every
    /// row's logits read back) as a CUDA graph whose head is the eager window's per-row head, each row into its slot
    /// of x.rows_f32 (graphs.Head.rows): the same launches as the eager window, one readback instead of one a row.
    sampled_graph: bool = false,
    /// p4-dattn: TF_INKLING2_DECODE_ATTN=1: a chunk of at most max_rows rows (decode, verify windows, a short prompt
    /// chunk) runs its attention on the split-K flash-decoding kernels (Kernels.attentionFdec: each kv head's keys and
    /// values read once a launch; NOT bitwise, inkling2.cu says where it rounds); 0 / unset: ink2_attention (phase 3).
    decode_attn: bool,
    /// p4-dattn: TF_INKLING2_DRAFT_ATTN (unset: decode_attn's value): the DSpark draft's attention the same way
    /// (dsp_fdec, dspark.zig); 0: dsp_attention.
    draft_attn: bool,
    /// p4-dattn: the decode attention's key splits [global, sliding] (fdecSplits) and its partials
    /// (Kernels.fdecPartBytes at max_rows and the larger split count; allocated only with decode_attn).
    fd_splits: [2]usize,
    fd_part: Buf,
    /// p4-mem (TF_INKLING2_LAZY_CONTEXT=1, default 0): the context-sized buffers (global layers' K / V caches, the
    /// score scratch, the draft head's) reserved whole and mapped as the sequence grows (lazy.zig); null: made whole
    /// at open (phase 3).
    lazy: ?*lazy_mod.Lazy = null,
    /// The EXL3 export's slots (exl3.Store.open, every expert resident); null for the release.
    ex: ?*exl3_mod.Store = null,
    exl3_act: Exl3Act = .native,
    /// e-decode: TF_INKLING2_EXL3_DEC=1: a decode / verify window's EXL3 experts at two ranks on kern.exl3Dec* (the
    /// same bits a pair, five launches a layer instead of ten and a zeroing: forward.exl3DecExperts); =0: off; unset:
    /// the default (Model.open's one expression).
    exl3_dec: bool = false,
    /// Canonical TP (the EXL3 export at world 1; TF_INKLING2_CANON_TP=0: off): every product a TP=2 pair splits runs
    /// here as the pair runs it (the head-split and vocab-split products as two half-width GEMMs of the pair's shapes,
    /// o_proj as two half-K GEMMs added in rank order, the embedding's two vocabulary halves summed, the EXL3 down
    /// projection's two K parts), so one GPU's logits are the bits two GPUs give (forward.zig).
    canon: bool = false,
    // ---- x-mixed: the mixed expert store (TF_INKLING2_MIX=1; out/prefill/p5/x-mixed/DESIGN.md). MODEL is the EXL3 view
    // (every expert's trellis resident, `ex`) and the arena is a native bank of the release's own NVFP4 / MXFP8 bytes,
    // read from TF_INKLING2_MIX_NATIVE (the release checkpoint): a routed pick runs its native expert when the bank holds
    // it, else that expert's EXL3 trellis (forward.moeMix); the router is never masked. Off: x1 exactly.
    mix: bool = false,
    /// The release checkpoint's directory and its Config (the bank's loader reads the release's own formats there).
    mix_native: []const u8 = "",
    mix_cfg: ?*Config = null,
    /// x-mixed (TF_INKLING2_SCORES): the decode score scratch (x.scores) not made: no kernel of this configuration
    /// reads it (split-K decode attention for every chunk of at most max_rows rows, fa or tc for wider ones). Unset:
    /// automatic (Model.open); 1 made; 0 none (refused when a kernel can read it).
    scores_off: bool = false,
    /// x-mixed (TF_INKLING2_MIX_CHECK=1, a gate): every layer's split checked on the host (forward.mixCheck; eager
    /// forwards only, so the graphs are off), and the count of splits checked.
    mix_check: bool = false,
    mix_checks: usize = 0,
    /// x-mixed: the bank hidden from the device (publishBank(false): every pick on the EXL3 store; the host keeps it).
    bank_hidden: bool = false,
    /// x-mixed: a partial view of the bank on the device ([L][E], 1 = shown; publishBankMask: the run tool's bankfrac /
    /// bankmx arms, frozen banks only), or null (the host's bank as it is).
    bank_mask: ?[]u8 = null,
    // ---- end x-mixed
    /// e-dense: the non-expert linears' EXL3 sets (dense_exl3.zig; TF_INKLING2_DENSE_EXL3, default off: null) and the
    /// calibration Hessians' capture (hcap.zig; the run tool's --hcap only).
    dx: ?*dense_exl3.DenseExl3 = null,
    hcap: ?*hcap_mod.Hcap = null,

    // ---- this rank's widths
    pub fn hq(m: *const Model) usize {
        return m.c.heads / m.world;
    }
    pub fn hkv(m: *const Model) usize {
        return m.c.kv_heads / m.world;
    }
    pub fn inter(m: *const Model) usize {
        return m.c.inter / m.world;
    }
    pub fn vocabLocal(m: *const Model) usize {
        return m.c.vocab / m.world;
    }
    pub fn smax(m: *const Model) usize {
        return (m.c.layers + m.c.block - 1) / m.c.block + 1;
    }

    /// The EXL3 activation mode of layer li for ink2_exl3_act: 0 none, 1 NVFP4, 2 MXFP8.
    pub fn exl3Mode(m: *const Model, li: usize) u8 {
        if (m.exl3_act == .bf16) return 0;
        return if (m.c.act[li] == .nvfp4) 1 else 2;
    }

    /// The EXL3 kernels' view of the store.
    pub fn exl3View(m: *const Model) @import("kern.zig").Kernels.Exl3 {
        const st = m.ex.?;
        return .{ .slot_ptr = st.slot_ptr.ptr, .slot_k = st.slot_k.ptr, .nslots = st.nslots, .hidden = m.c.hidden, .inter = m.inter(), .cg = m.c.gate_cap, .cu = m.c.up_cap, .tp13 = st.tp13.ptr, .tp2 = st.tp2.ptr, .k2 = st.k2.ptr };
    }

    fn alloc(m: *Model, bytes: usize) !Buf {
        const b = try Buf.alloc(m.d, @max(bytes, 16));
        try m.bufs.append(m.gpa, b);
        return b;
    }

    fn ptr(m: *const Model, name: []const u8) !u64 {
        const t = m.dense.get(name) orelse {
            std.log.err("inkling2: no loaded tensor {s}", .{name});
            return error.MissingTensor;
        };
        return t.buf.ptr;
    }

    /// e-dense: a tensor's pointer, or 0 when the load left it out (dropReplaced: an EXL3 set runs it instead).
    fn ptrOr0(m: *const Model, name: []const u8) u64 {
        const t = m.dense.get(name) orelse return 0;
        return t.buf.ptr;
    }

    fn lname(m: *const Model, buf: []u8, i: usize, comptime rest: []const u8) ![]const u8 {
        _ = m;
        return std.fmt.bufPrint(buf, "model.llm.layers.{d}." ++ rest, .{i});
    }

    /// Loads rank `rank` of `world` (its TP group already open in `group` when world > 1): the dense weights, every
    /// layer's roles, the arena (empty: `fill` places experts), the state for `context` positions and the scratch.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, rank: usize, world: usize, group: ?cuda.tp_group.Group, context: usize, max_rows: usize) !*Model {
        const d = ctx.d;
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        m.d = d;
        m.bufs = .empty;
        m.c = try config.read(gpa, io, dir);
        try m.c.checkSplit(@intCast(world));
        m.rank = rank;
        m.world = world;
        m.group = group;
        m.context = context;
        m.max_rows = max_rows;
        m.prompt_rows = max_rows;
        if (defaults.legacy()) std.log.info("inkling2 rank {d}: TF_INKLING2_LEGACY=1: the old defaults (every speed switch off unless set)", .{rank});
        m.split_moe = if (std.c.getenv("TF_INKLING2_PREFILL_SPLIT")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        m.prefill_kernels = if (std.c.getenv("TF_INKLING2_PREFILL_KERNELS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        m.prefill_experts = if (std.c.getenv("TF_INKLING2_PREFILL_EXPERTS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        m.prefill_attn = if (std.c.getenv("TF_INKLING2_PREFILL_ATTN")) |v| std.meta.stringToEnum(PrefillAttn, std.mem.span(v)) orelse blk: {
            std.log.warn("inkling2: TF_INKLING2_PREFILL_ATTN={s} is not gqa, tiled, tc or fa; tiled", .{std.mem.span(v)});
            break :blk .tiled;
        } else if (defaults.legacy()) .tiled else .fa;
        if (m.prefill_attn == .fa) std.log.info("inkling2 rank {d}: wide prompt chunks' attention on tensor cores, online softmax (TF_INKLING2_PREFILL_ATTN=fa; not bitwise)", .{rank});
        if (m.prefill_attn == .tc) std.log.info("inkling2 rank {d}: wide prompt chunks' attention on tensor cores (TF_INKLING2_PREFILL_ATTN=tc; not bitwise)", .{rank});
        m.overlap = try overlapParts(if (std.c.getenv("TF_INKLING2_PREFILL_OVERLAP")) |v| std.mem.span(v) else null);
        // p4-xchg: TF_INKLING2_MOE_XCHG=pair|token (default token; pair under TF_INKLING2_LEGACY=1: phase 3)
        m.moe_xchg = if (std.c.getenv("TF_INKLING2_MOE_XCHG")) |v| std.meta.stringToEnum(MoeXchg, std.mem.span(v)) orelse blk: {
            std.log.warn("inkling2: TF_INKLING2_MOE_XCHG={s} is not pair or token; pair", .{std.mem.span(v)});
            break :blk .pair;
        } else if (defaults.legacy()) .pair else .token;
        m.ov = null;
        m.ex = null;
        m.canon = m.c.exl3 and world == 1 and !(if (std.c.getenv("TF_INKLING2_CANON_TP")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false);
        if (m.canon) std.log.info("inkling2 rank 0: canonical TP (a TP=2 pair's arithmetic at world 1)", .{});
        m.exl3_act = if (std.c.getenv("TF_INKLING2_EXL3_ACT")) |v| std.meta.stringToEnum(Exl3Act, std.mem.span(v)) orelse blk: {
            std.log.warn("inkling2: TF_INKLING2_EXL3_ACT={s} is not native or bf16; native", .{std.mem.span(v)});
            break :blk .native;
        } else .native;
        if (m.c.exl3) std.log.info("inkling2 rank {d}: the EXL3 export's experts, activations {t}", .{ rank, m.exl3_act });
        // ---- x-mixed: TF_INKLING2_MIX=1 (forward.moeMix): the EXL3 view's experts plus a native bank of the release's
        // own bytes (TF_INKLING2_MIX_NATIVE: its directory; its geometry and formats must be the view's)
        m.mix = false;
        m.mix_native = "";
        m.mix_cfg = null;
        m.scores_off = false;
        m.mix_check = false;
        m.mix_checks = 0;
        m.bank_hidden = false;
        m.bank_mask = null;
        errdefer if (m.mix_cfg) |nc| gpa.destroy(nc);
        if (envOn("TF_INKLING2_MIX")) {
            if (!m.c.exl3) {
                std.log.err("inkling2: TF_INKLING2_MIX=1 needs the EXL3 view as the model (its checkpoint.json): {s} is not one", .{dir});
                return error.MixNeedsExl3;
            }
            const nat = std.mem.span(std.c.getenv("TF_INKLING2_MIX_NATIVE") orelse {
                std.log.err("inkling2: TF_INKLING2_MIX=1 needs TF_INKLING2_MIX_NATIVE, the release checkpoint the native bank reads", .{});
                return error.MixNoNative;
            });
            const nc = try gpa.create(Config);
            m.mix_cfg = nc;
            nc.* = try config.read(gpa, io, nat);
            var same = !nc.exl3 and nc.hidden == m.c.hidden and nc.layers == m.c.layers and nc.experts == m.c.experts and nc.inter == m.c.inter and nc.topk == m.c.topk;
            for (0..m.c.layers) |l| same = same and nc.formats[l] == m.c.act[l];
            if (!same) {
                std.log.err("inkling2: TF_INKLING2_MIX_NATIVE={s} is not this view's release (geometry or expert formats differ)", .{nat});
                return error.MixGeometry;
            }
            m.mix = true;
            m.mix_native = nat;
            m.mix_check = envOn("TF_INKLING2_MIX_CHECK");
            std.log.info("inkling2 rank {d}: the mixed expert store (TF_INKLING2_MIX=1): every expert's EXL3 trellis plus a native bank of {s}'s NVFP4 / MXFP8 experts; a pick native when the bank holds it, else EXL3", .{ rank, nat });
        }
        // ---- end x-mixed
        // e-decode: TF_INKLING2_EXL3_DEC=1 on, any other value off; unset on (off under TF_INKLING2_LEGACY=1); EXL3 store
        // only (forward.exl3DecOn)
        m.exl3_dec = if (std.c.getenv("TF_INKLING2_EXL3_DEC")) |v| std.mem.eql(u8, std.mem.span(v), "1") else !defaults.legacy();
        if (m.c.exl3 and m.exl3_dec) std.log.info("inkling2 rank {d}: EXL3 decode / verify windows on the fused kernels (TF_INKLING2_EXL3_DEC)", .{rank});
        // p4-cache; on by default (the user, 2026-11-14: "Just turn it on"); TF_INKLING2_PROMPT_REUSE=0 turns it off
        m.prompt_reuse = if (std.c.getenv("TF_INKLING2_PROMPT_REUSE")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
        // p4-dattn: the decode attention switches; on by default (the user, 2026-11-14: "Just turn it on"); =0 gives phase 3's kernels
        m.decode_attn = envFlag("TF_INKLING2_DECODE_ATTN", true);
        m.draft_attn = envFlag("TF_INKLING2_DRAFT_ATTN", m.decode_attn);
        m.fd_part = .{ .d = d, .ptr = 0, .len = 0 };
        const sms: usize = @intCast(@max(try ctx.attribute(.multiprocessor_count), 1));
        // x-merge: under canonical TP a pair rank's split count (its kv heads), so a head's splits are the pair's
        const split_world: usize = if (m.canon) 2 else world;
        m.fd_splits = try fdecSplits(std.math.clamp(2 * sms / @max(m.c.kv_heads / split_world, 1), 1, kern.Kernels.fdec_max_splits));
        if (m.decode_attn or m.draft_attn) std.log.info("inkling2 rank {d}: decode attention {s}, draft attention {s} (split-K flash decoding: not bitwise), splits {d} global / {d} sliding", .{ rank, if (m.decode_attn) "on" else "off", if (m.draft_attn) "on" else "off", m.fd_splits[0], m.fd_splits[1] });
        // p4-small: its switches, on unless set to 0 (REL_WIDE 2); TF_INKLING2_LEGACY=1: each 0 / off (phase 3)
        m.rel_wide = if (std.c.getenv("TF_INKLING2_REL_WIDE")) |v| std.fmt.parseInt(u8, std.mem.span(v), 10) catch return error.BadRelWide else if (defaults.legacy()) 0 else 2;
        if (m.rel_wide > 2) return error.BadRelWide;
        m.absorb_wide = envOnDefault("TF_INKLING2_ABSORB_WIDE");
        m.fuse = envOnDefault("TF_INKLING2_FUSE");
        if (m.rel_wide != 0 or m.absorb_wide or m.fuse) std.log.info("inkling2 rank {d}: p4-small switches REL_WIDE={d} ABSORB_WIDE={} FUSE={}", .{ rank, m.rel_wide, m.absorb_wide, m.fuse });
        // p4-mx: W8A8 MXFP8 experts (TF_INKLING2_MX_ACT=1; unset on, off under TF_INKLING2_LEGACY=1); read before
        // makeScratch (x.mxq)
        m.mx_act = if (std.c.getenv("TF_INKLING2_MX_ACT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else !defaults.legacy();
        if (m.mx_act) std.log.info("inkling2 rank {d}: MXFP8 experts in W8A8, the checkpoint's activation quantization (TF_INKLING2_MX_ACT=1)", .{rank});
        // p4-mem: the lazy context maps the context-sized buffers as the sequence grows (lazy.zig):
        // TF_INKLING2_LAZY_CONTEXT=1 on, any other value off; unset on (off under TF_INKLING2_LEGACY=1)
        m.lazy = null;
        const lazy_on = if (std.c.getenv("TF_INKLING2_LAZY_CONTEXT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else !defaults.legacy();
        if (lazy_on) m.lazy = try lazy_mod.Lazy.open(gpa, d, ctx.device, rank, context);
        // ---- x-mixed: the decode score scratch (x.scores: context-sized, 0.72 GiB at a 500K window under the lazy
        // context's floor) is made only when a kernel this configuration can run reads it (forward.layers): a chunk of at
        // most max_rows rows off the split-K decode attention (TF_INKLING2_DECODE_ATTN=0, or heads it does not fit:
        // fdecFits is monotone in the rows), a wider chunk off fa / tc (TF_INKLING2_PREFILL_ATTN=tiled or gqa, as
        // TF_INKLING2_LEGACY=1 selects, or TF_INKLING2_PREFILL_KERNELS=0). TF_INKLING2_SCORES=1 makes it; =0 makes none
        // (refused when a kernel can read it); unset (the defaults layer, P5-BRIEF ~11:40): none when nothing reads it,
        // nothing can switch a reader on at run time (no TF_INKLING2_ARM_SWITCHES: the run tool's arms may select tiled
        // attention or dattn=0) and not TF_INKLING2_LEGACY=1. No numerics change: the kernels that run never read it.
        {
            const shq = m.c.heads / world;
            const shkv = m.c.kv_heads / world;
            const shd: usize = m.c.head_dim;
            const wide_ok = m.prefill_kernels and ((m.prefill_attn == .fa and kern.Kernels.faFits(shq, shkv, shd)) or (m.prefill_attn == .tc and kern.Kernels.tcFits(shq, shkv, shd)));
            const reads_none = m.decode_attn and kern.Kernels.fdecFits(max_rows, shq, shkv, shd) and wide_ok;
            const sv: ?[]const u8 = if (std.c.getenv("TF_INKLING2_SCORES")) |v| std.mem.span(v) else null;
            if (sv) |v| {
                if (std.mem.eql(u8, v, "0")) {
                    if (!reads_none) {
                        std.log.err("inkling2: TF_INKLING2_SCORES=0 needs the split-K decode attention (TF_INKLING2_DECODE_ATTN) and TF_INKLING2_PREFILL_ATTN=fa or tc: another kernel reads the score scratch", .{});
                        return error.ScoresRead;
                    }
                    m.scores_off = true;
                } else if (!std.mem.eql(u8, v, "1")) std.log.warn("inkling2: TF_INKLING2_SCORES={s} is not 0 or 1; the score scratch is made", .{v});
            } else m.scores_off = reads_none and !defaults.legacy() and std.c.getenv("TF_INKLING2_ARM_SWITCHES") == null;
            if (m.scores_off) std.log.info("inkling2 rank {d}: no decode score scratch ({s}: the split-K decode attention and {t} read none)", .{ rank, if (sv != null) "TF_INKLING2_SCORES=0" else "unset: nothing in this configuration reads it", m.prefill_attn });
        }
        // ---- end x-mixed
        // xa: under TF_INKLING2_MIX the decode stream runs at the context's highest priority (TF_INKLING2_MIX_PRIO, default
        // on; 0 off), the bank loader's side stream (fill.zig) at the default, the lowest: its pack kernels (15,552 blocks
        // an expert at the release's shape) then take only the SMs the forwards leave, where at equal priorities every
        // verify kernel queued behind them (a scheduling choice: no bit changes)
        const prio = m.mix and !(if (std.c.getenv("TF_INKLING2_MIX_PRIO")) |v| std.mem.eql(u8, std.mem.span(v), "0") else false);
        m.s = if (prio) try cuda.Stream.initPriority(d, .highest) else try cuda.Stream.init(d, true);
        if (prio) std.log.info("inkling2 rank {d}: the decode stream at the highest priority (TF_INKLING2_MIX_PRIO): the native bank's loads take the SMs its forwards leave", .{rank});
        m.k = try kern.Kernels.load(d, @intCast(@max(try ctx.attribute(.multiprocessor_count), 1)));
        if (m.fuse and !m.k.arn_big) { // p4-small: FUSE needs ink2_attn_res_norm's shared memory opt-in
            std.log.warn("inkling2 rank {d}: TF_INKLING2_FUSE off: the device refused ink2_attn_res_norm's shared memory", .{rank});
            m.fuse = false;
        }
        // e-step: TF_INKLING2_STEP_FUSE=1 (the same opt-in as FUSE's)
        // (x5 port: on by default, the same bits; TF_INKLING2_STEP_FUSE=0 or TF_INKLING2_LEGACY=1 turns it off)
        m.step_fuse = envOnDefault("TF_INKLING2_STEP_FUSE") and m.k.arn_big;
        if (m.step_fuse) std.log.info("inkling2 rank {d}: e-step STEP_FUSE: decode / verify chunks take the fused launches (the same bits)", .{rank});
        m.g = try gemm.Gemms.init(gpa, d);
        var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
        var ck_drop = page_cache.Files.of(gpa, &ck); // e-load (x5): its shards out of the page cache after ck.close() below
        defer ck_drop.drop();
        defer ck.close();
        var arena_alloc = std.heap.ArenaAllocator.init(gpa);
        defer arena_alloc.deinit();
        var entries = try plan_mod.plan(arena_alloc.allocator(), &m.c, @intCast(rank), @intCast(world));
        // e-dense: TF_INKLING2_DENSE_EXL3=<classes> takes those linears from a dense EXL3 pack (the view itself, or
        // TF_INKLING2_DENSE_EXL3_PACK); the bf16 copies it replaces are not loaded (TF_INKLING2_DENSE_EXL3_KEEP_BF16=1,
        // or canonical TP whose o_proj halves are made from the bf16 wo, keeps them; the head's always stays: the
        // draft's). Loaded here, before the state, the scratch and (the caller's) arena / EXL3 slots / bank are sized:
        // what the set frees and takes is in MemAvailable when they are.
        m.dx = null;
        m.hcap = null;
        // the user, 2026-11-15: unset, the classes this checkpoint carries as EXL3 (a dense-pack view: all three)
        const dx_classes = if (std.c.getenv("TF_INKLING2_DENSE_EXL3") != null or defaults.legacy()) try dense_exl3.envClasses() else dense_exl3.packClasses(&ck);
        const keep_bf16 = m.canon or envOn("TF_INKLING2_DENSE_EXL3_KEEP_BF16");
        if (dx_classes.count() > 0 and !keep_bf16) entries = try dropReplaced(arena_alloc.allocator(), entries, dx_classes);
        m.dense = try dense_mod.load(gpa, d, &ck, entries);
        for (entries) |e| if (std.mem.eql(u8, e.name, "model.llm.embed.weight")) {
            m.vocab_lo = e.lo;
        };
        try m.roles();
        if (dx_classes.count() > 0) {
            // e-dense: the switch's set, active from the first forward (prewarm, graphs) on
            m.dx = try dense_exl3.DenseExl3.init(gpa, d, &m.g, rank, world, m.canon);
            const pack = if (std.c.getenv("TF_INKLING2_DENSE_EXL3_PACK")) |v| std.mem.span(v) else dir;
            const wide: dense_exl3.Wide = if (std.c.getenv("TF_INKLING2_DENSE_EXL3_WIDE")) |v| std.meta.stringToEnum(dense_exl3.Wide, std.mem.span(v)) orelse .gemm else .gemm;
            _ = try m.dx.?.addSet(io, "env", dx_classes, pack, m.c.layers, wide, false);
            try m.dx.?.activate(m.s, 0);
            try m.s.synchronize();
        }
        // e-step: TF_INKLING2_DENSE_GROUP=1 (with a dense EXL3 set): the groups' scratch now, before any capture
        // (x5 port: on by default, the same bits; TF_INKLING2_DENSE_GROUP=0 or TF_INKLING2_LEGACY=1 turns it off)
        m.dense_group = envOnDefault("TF_INKLING2_DENSE_GROUP") and m.dx != null;
        if (m.dense_group) try @import("forward.zig").reserveGroups(m);
        try m.makeState();
        try m.makeScratch();
        // p4-dattn: the decode attention's split partials (max_rows rows; small: 4.6 MiB a rank at 24 splits)
        if (m.decode_attn) m.fd_part = try m.alloc(kern.Kernels.fdecPartBytes(max_rows, m.hq(), m.c.head_dim, @max(m.fd_splits[0], m.fd_splits[1])));
        // the TP exchanges by one-shot RoCE writes (cuda.tp_group's fast path, as MiniMax and GLM open it at two
        // ranks): every rank or none; TF_ROCE=0 keeps NCCL, TF_ROCE_MAX_KB caps a rank's shard (default 4 MiB: a
        // 16-row chunk's expert partials are 2.25 MiB a rank, its batched head 3.07 MiB). The size alone picks the path.
        if (m.group) |*g| try g.openRoce(gpa, io, d, m.s, try roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |v| std.mem.span(v) else null));
        if (world == 2) {
            // the ranks run the same exchanges (a part's NCCL group each): the fewest parts any rank asks for
            var all: [2]usize = undefined;
            try m.group.?.gather(usize, d, m.s, m.overlap, &all);
            m.overlap = @min(all[0], all[1]);
            if (m.overlap > 0) m.ov = try Overlap.init(gpa, d);
        }
        if (world == 2) {
            // p4-xchg: the ranks' MoE exchanges must match in form and size: both ranks' setting, else pair on both
            var all: [2]usize = undefined;
            try m.group.?.gather(usize, d, m.s, @intFromEnum(m.moe_xchg), &all);
            if (all[0] != all[1]) {
                std.log.warn("inkling2 rank {d}: the ranks' TF_INKLING2_MOE_XCHG settings differ; pair", .{rank});
                m.moe_xchg = .pair;
            }
        }
        if (m.moe_xchg == .token and world > 1) std.log.info("inkling2 rank {d}: the MoE's TP sum by token rows (TF_INKLING2_MOE_XCHG=token; not phase 3's bits)", .{rank});
        m.arena_on = false;
        m.exact = false;
        m.loader = null;
        m.transient_used = .{ 0, 0 };
        m.exact_loads = 0;
        m.spec = null;
        m.graphs = graphs_mod.Graphs.fromEnv();
        if (m.mix_check) m.graphs.on = false; // x-mixed: the split's check reads it back (a host sync a layer)
        m.sampled_graph = envOnDefault("TF_INKLING2_SAMPLED_GRAPH") and m.x.rows_f32.len > 0; // xs
        if (m.graphs.on and m.sampled_graph) std.log.info("inkling2 rank {d}: sampled verify windows graphed (their per-row head in the graph; TF_INKLING2_SAMPLED_GRAPH=0: eager)", .{m.rank});
        return m;
    }

    pub fn close(m: *Model) void {
        const gr = &m.graphs;
        if (gr.replays > 0) std.debug.print("inkling2 rank {d}: CUDA graphs: {d} captured in {d:.1} ms each, {d} replays, positions rewritten in {d:.1} us a replay\n", .{ m.rank, gr.captures, @as(f64, @floatFromInt(gr.capture_ns)) / 1e6 / @as(f64, @floatFromInt(@max(gr.captures, 1))), gr.replays, @as(f64, @floatFromInt(gr.patch_ns)) / 1e3 / @as(f64, @floatFromInt(gr.replays)) });
        gr.drop(m.gpa);
        for (m.bufs.items) |*b| b.free();
        m.bufs.deinit(m.gpa);
        if (m.arena_on) m.arena.deinit(m.gpa);
        if (m.ex) |st| st.close();
        if (m.mix_cfg) |nc| m.gpa.destroy(nc); // x-mixed
        if (m.bank_mask) |mk| m.gpa.free(mk); // x-mixed
        if (m.mix_check) std.debug.print("inkling2 rank {d}: x-mixed: {d} layer splits checked against the plan and the bank's LUT: every one equal\n", .{ m.rank, m.mix_checks });
        if (m.lazy) |lz| lz.deinit(); // p4-mem: the context's and the arena's ranges (after the views above)
        if (m.spec) |sp| {
            m.gpa.free(sp.hist);
            m.gpa.destroy(sp);
        }
        for ([_][]Buf{ m.state.kc, m.state.vc, m.state.ring_k, m.state.ring_v, m.state.ring_attn, m.state.ring_mlp }) |l| m.gpa.free(l);
        m.gpa.free(m.state.slots);
        m.gpa.free(m.layers);
        if (m.dx) |dx| dx.deinit(); // e-dense (its wide plans on m.g's cuBLASLt handle: before m.g)
        if (m.hcap) |h| h.deinit();
        m.dense.deinit();
        m.g.deinit();
        m.k.unload();
        if (m.ov) |ov| ov.deinit(m.gpa);
        m.s.deinit();
        if (m.group) |*g| g.close();
        m.gpa.destroy(m);
    }

    fn roles(m: *Model) !void {
        const c = &m.c;
        m.embed = try m.ptr("model.llm.embed.weight");
        m.embed_norm = try m.ptr("model.llm.embed_norm.weight");
        m.norm = try m.ptr("model.llm.norm.weight");
        m.unembed = try m.ptr("model.llm.unembed.weight");
        m.final_query = try m.ptr("model.llm.attn_res.output_query");
        m.layers = try m.gpa.alloc(Layer, c.layers);
        var b: [160]u8 = undefined;
        for (m.layers, 0..) |*l, i| {
            const local = c.isLocal(@intCast(i));
            l.* = .{
                .local = local,
                .boundary = c.isBoundary(@intCast(i)),
                .fmt = c.formats[i],
                .wq = m.ptrOr0(try m.lname(&b, i, "attn.wq_du.weight")), // e-dense: 0 when an EXL3 set replaces it
                .wk = m.ptrOr0(try m.lname(&b, i, "attn.wk_dv.weight")), // e-dense: 0 when an EXL3 set replaces it
                .wv = m.ptrOr0(try m.lname(&b, i, "attn.wv_dv.weight")), // e-dense: 0 when an EXL3 set replaces it
                .wr = m.ptrOr0(try m.lname(&b, i, "attn.wr_du.weight")), // e-dense: 0 when an EXL3 set replaces it
                .wo = m.ptrOr0(try m.lname(&b, i, "attn.wo_ud.weight")), // e-dense: 0 when an EXL3 set replaces it
                .wgate = m.ptrOr0(try m.lname(&b, i, "attn.gate_attn.weight")), // e-dense: 0 when an EXL3 set replaces it
                .wsink = try m.ptr(try m.lname(&b, i, "attn.attn_sink_proj.weight")),
                .qscale = try m.ptr(try m.lname(&b, i, "attn.q_l2_clip.scale")),
                .relscale = try m.ptr(try m.lname(&b, i, "attn.rel_l2_clip.scale")),
                .relproj = try m.ptr(try m.lname(&b, i, "attn.rel_logits_proj.proj")),
                .extent = if (local) c.window else c.rel_extent,
                .norm_mlp = try m.ptr(try m.lname(&b, i, "mlp_norm.weight")),
                .conv_attn = try m.ptr(try m.lname(&b, i, "attn_sconv.weight")),
                .conv_mlp = try m.ptr(try m.lname(&b, i, "mlp_sconv.weight")),
                .router = m.ptrOr0(try m.lname(&b, i, "mlp.gate.weight")), // e-dense: 0 when an EXL3 set replaces it
                .bias = try m.ptr(try m.lname(&b, i, "mlp.gate.bias")),
                .mlp_query = try m.ptr(try std.fmt.bufPrint(&b, "model.llm.attn_res.mlp_queries.{d}", .{i})),
            };
            if (local) {
                l.kconv = try m.ptr(try m.lname(&b, i, "attn.k_sconv.weight"));
                l.vconv = try m.ptr(try m.lname(&b, i, "attn.v_sconv.weight"));
                l.norm_attn = try m.ptr(try m.lname(&b, i, "attn_norm.weight"));
            } else {
                l.kstatic = try m.ptr(try m.lname(&b, i, "attn.k_sconv.static_w"));
                l.kproj = try m.ptr(try m.lname(&b, i, "attn.k_sconv.weight_proj"));
                l.vstatic = try m.ptr(try m.lname(&b, i, "attn.v_sconv.static_w"));
                l.vproj = try m.ptr(try m.lname(&b, i, "attn.v_sconv.weight_proj"));
                l.norm_q = try m.ptr(try m.lname(&b, i, "q_input_norm.weight"));
                l.norm_k = try m.ptr(try m.lname(&b, i, "k_input_norm.weight"));
                l.norm_v = try m.ptr(try m.lname(&b, i, "v_input_norm.weight"));
            }
            if (i > 0) {
                l.attn_query = try m.ptr(try std.fmt.bufPrint(&b, "model.llm.attn_res.attn_queries.{d}", .{i - 1}));
                l.nq = if (local) 1 else 3;
            }
            if (m.canon) {
                // wo [H][qw]: each half of its columns as the pair's ranks hold it, [H][qw / 2] contiguous
                const qw = m.hq() * c.head_dim;
                const half = qw / 2;
                for (0..2) |r| {
                    const bh = try m.alloc(c.hidden * half * 2);
                    try copy2d(m, bh.ptr, half * 2, l.wo + r * half * 2, qw * 2, half * 2, c.hidden);
                    l.wo_half[r] = bh.ptr;
                }
            }
        }
    }

    fn makeState(m: *Model) !void {
        const L = m.c.layers;
        const st = &m.state;
        st.* = .{ .kc = try m.gpa.alloc(Buf, L), .vc = try m.gpa.alloc(Buf, L), .slots = try m.gpa.alloc(usize, L), .ring_k = try m.gpa.alloc(Buf, L), .ring_v = try m.gpa.alloc(Buf, L), .ring_attn = try m.gpa.alloc(Buf, L), .ring_mlp = try m.gpa.alloc(Buf, L) };
        const kvw = m.hkv() * m.c.head_dim;
        const kk = m.c.conv - 1;
        // each ring kind's layers in one block, so a rewind restores a kind in one strided copy
        const widths = [4]usize{ kvw, kvw, m.c.hidden, m.c.hidden };
        for (&st.rings_all, widths) |*b, w| b.* = try m.alloc(L * kk * w * 4);
        for (m.layers, 0..) |l, i| {
            st.slots[i] = if (l.local) m.c.window + m.max_rows else m.context;
            if (m.lazy != null and !l.local) {
                // p4-mem: a global layer's cache (position p at slot p) reserved whole, mapped for the mapped positions
                try m.lazy.?.register(m, &st.kc[i], kvw * 2, 0, 0, "k");
                try m.lazy.?.register(m, &st.vc[i], kvw * 2, 0, 0, "v");
            } else {
                st.kc[i] = try m.alloc(st.slots[i] * kvw * 2);
                st.vc[i] = try m.alloc(st.slots[i] * kvw * 2);
            }
            for ([_][]Buf{ st.ring_k, st.ring_v, st.ring_attn, st.ring_mlp }, st.rings_all, widths) |r, all, w| {
                r[i] = .{ .d = m.d, .ptr = all.ptr + i * kk * w * 4, .len = kk * w * 4, .borrowed = true };
            }
        }
        try m.reset();
    }

    /// A new sequence: position 0, every convolution ring zero (the reference's left padding). Caches need no clearing:
    /// attention reads positions below the sequence's length only.
    pub fn reset(m: *Model) !void {
        const st = &m.state;
        for (st.rings_all) |x| try x.fill8(0, m.s.handle);
        st.len = 0;
    }

    fn makeScratch(m: *Model) !void {
        const R = m.max_rows;
        const H = m.c.hidden;
        const W = m.world;
        const k = m.c.topk;
        const hd = m.c.head_dim;
        const qw = m.hq() * hd;
        const kvw = m.hkv() * hd;
        const ext = @max(m.c.window, m.c.rel_extent);
        const pairs = R * k;
        const items = cuda.grouped.maxItems(pairs, m.c.experts, kern.decode_tile);
        m.x = .{
            .ids = try m.alloc(R * 4),
            .emb = try m.alloc(R * H * 2),
            .emb_all = try m.alloc(m.planes() * R * H * 2),
            .h = try m.alloc(R * H * 2),
            .blocks = try m.alloc(R * m.smax() * H * 2),
            .ar = try m.alloc(3 * R * H * 4),
            .xq = try m.alloc(R * H * 2),
            .xk = try m.alloc(R * H * 2),
            .xv = try m.alloc(R * H * 2),
            .q = try m.alloc(R * qw * 2),
            .k = try m.alloc(R * kvw * 2),
            .v = try m.alloc(R * kvw * 2),
            .r = try m.alloc(R * m.hq() * m.c.d_rel * 2),
            .gate = try m.alloc(R * qw * 2),
            .sinks = try m.alloc(R * m.hq() * 2),
            .dyn = try m.alloc(R * m.hkv() * m.c.conv * 2),
            .dyn32 = try m.alloc(R * m.hkv() * m.c.conv * 4),
            .rel = try m.alloc(R * m.hq() * ext * 2),
            .scores = if (m.lazy == null and !m.scores_off) try m.alloc(R * m.hq() * @max(m.context, m.c.window + R) * 4) else undefined, // x-mixed: scores_off
            .attn = try m.alloc(R * qw * 2),
            .o32 = try m.alloc(R * H * 4),
            .o_all = try m.alloc(m.planes() * R * H * 4),
            .o = try m.alloc(R * H * 2),
            .res = try m.alloc(R * H * 4),
            .xm = try m.alloc(R * H * 2),
            .lg = try m.alloc(R * m.c.experts * 4),
            .picks = try m.alloc(pairs * 4),
            .wts = try m.alloc(pairs * 4),
            .want_ids = try m.alloc(pairs * 4),
            .want_sc = try m.alloc(pairs * 4),
            .members = try m.alloc(pairs * 4),
            .items = try m.alloc(3 * items * 4),
            .counts = try m.alloc(16),
            .plan_rank = try m.alloc(pairs * 4),
            .plan_hist = try m.alloc(1024 * 64 * 4),
            .act = try m.alloc(pairs * m.inter() * 2),
            .y = try m.alloc(pairs * H * 4),
            .y_all = try m.alloc(W * pairs * H * 4),
            .ysum = try m.alloc(pairs * H * 4),
            .mo = try m.alloc(R * H * 2),
            .fin = try m.alloc(R * H * 4),
            .head = try m.alloc(m.vocabLocal() * 2),
            .head_all = try m.alloc(m.c.vocab * 2),
            .head_rows = try m.alloc(R * m.vocabLocal() * 2),
            .head_rows_all = try m.alloc(m.planes() * R * m.vocabLocal() * 2),
            .head_cat = try m.alloc(R * m.c.vocab * 2),
            .logits = try m.alloc(m.c.unpadded_vocab * 4),
            // xs: a graphed sampled window's rows (6.4 MB a rank; none with TF_INKLING2_SAMPLED_GRAPH off: xk's memory)
            .rows_f32 = if (envOnDefault("TF_INKLING2_SAMPLED_GRAPH")) try m.alloc(@min(R, graphs_mod.max_rows) * m.c.unpadded_vocab * 4) else .{ .d = m.d, .ptr = 0, .len = 0 },
            .argmax = try m.alloc(@max(R, 4) * 4),
            .demand = try m.alloc(m.c.layers * m.c.experts * 3 * 4),
            .edges = .{ .d = m.d, .ptr = 0, .len = 0 }, // none until growPrompt: the arena's room is the one before
            .mxq = if (m.mx_act) try m.alloc(R * kern.Kernels.q8Stride(H)) else .{ .d = m.d, .ptr = 0, .len = 0 }, // p4-mx
            .e_xq = try m.alloc(m.exl3Bytes(R, .xq)),
            .e_xr = try m.alloc(m.exl3Bytes(R, .xr)),
            .e_z = try m.alloc(m.exl3Bytes(R, .z)),
            .e_amax = try m.alloc(m.exl3Bytes(R, .amax)),
            .e_amax_all = try m.alloc(m.exl3Bytes(R, .amax_all)),
            .canon_tmp = try m.alloc(m.canonBytes(R, .tmp)),
            .e_zd = try m.alloc(m.canonBytes(R, .zd)),
            // x-mixed (Model.mix; none otherwise): moeMix's split items and counts, the EXL3 SwiGLU rows
            .items_n = if (m.mix) try m.alloc(3 * items * 4) else .{ .d = m.d, .ptr = 0, .len = 0 },
            .counts_n = if (m.mix) try m.alloc(16) else .{ .d = m.d, .ptr = 0, .len = 0 },
            .items_x = if (m.mix) try m.alloc(3 * items * 4) else .{ .d = m.d, .ptr = 0, .len = 0 },
            .counts_x = if (m.mix) try m.alloc(16) else .{ .d = m.d, .ptr = 0, .len = 0 },
            .e_act = if (m.mix) try m.alloc(pairs * m.inter() * 2) else .{ .d = m.d, .ptr = 0, .len = 0 },
            .xlut = if (m.mix) try m.alloc(m.c.layers * m.c.experts * 4) else .{ .d = m.d, .ptr = 0, .len = 0 },
        };
        if (m.mix) { // x-mixed: the EXL3 slots' identity map (exl3.Store's slot_ptr / slot_k are layer E + expert)
            const ids = try m.gpa.alloc(i32, m.c.layers * m.c.experts);
            defer m.gpa.free(ids);
            for (ids, 0..) |*v, i| v.* = @intCast(i);
            try m.x.xlut.upload(0, std.mem.sliceAsBytes(ids));
        }
        // p4-mem: the score scratch at scoreCap() floats a (row, head), mapped for the mapped positions and at least
        // TF_INKLING2_LAZY_SCORES_MIB, at most the context's (the reservation; prompt launches take its whole length)
        // x-mixed: none with TF_INKLING2_SCORES=0 (Model.scores_off: no kernel that runs reads it; a 16-byte stand-in)
        if (m.scores_off) {
            m.x.scores = try m.alloc(16);
        } else if (m.lazy) |lz| {
            const floor_rows = @min(lz.scores_min / (R * m.hq() * 4), m.context);
            try lz.register(m, &m.x.scores, R * m.hq() * 4, 0, @max(m.c.window + R, floor_rows), "scores");
        }
        try m.x.demand.fill8(0, m.s.handle);
    }

    const Exl3Buf = enum { xq, xr, z, amax, amax_all };
    const CanonBuf = enum { tmp, zd };

    /// Planes of the gathered-partials buffers (emb_all, o_all, head_rows_all): the ranks', or a canonical pair's 2.
    fn planes(m: *const Model) usize {
        return if (m.canon) 2 else m.world;
    }

    /// Bytes of a canonical-TP buffer for chunks of R rows (0 when off). tmp holds the widest half product's rows: a
    /// half of q / gate (bf16 [R][qw / 2]), of o_proj's input, of the relative bias' rows and products.
    fn canonBytes(m: *const Model, R: usize, b: CanonBuf) usize {
        if (!m.canon) return 0;
        const nh = m.hq();
        const ext = @max(m.c.window, m.c.rel_extent);
        return switch (b) {
            .tmp => R * (nh * m.c.head_dim + nh * ext + nh * m.c.d_rel) * 2,
            .zd => 2 * R * m.c.topk * m.c.hidden * 4,
        };
    }

    /// Bytes of an EXL3 scratch buffer for chunks of R rows (0 for the release: the allocation's 16 bytes).
    fn exl3Bytes(m: *const Model, R: usize, b: Exl3Buf) usize {
        if (!m.c.exl3) return 0;
        const H = m.c.hidden;
        const pairs = R * m.c.topk;
        const Il = m.inter();
        return switch (b) {
            .xq => @max(R * H, pairs * Il) * 2,
            .xr => pairs * @max(H, Il) * 2,
            .z => pairs * 2 * Il * 4,
            .amax => pairs * 4,
            .amax_all => m.world * pairs * 4,
        };
    }

    const Grown = struct { buf: *Buf, bytes: usize };
    const max_grown = 56 + 2 * config.max_layers;

    /// The buffers whose size a prompt chunk of `R` rows sets, each with its bytes at R: the forward's row-scaled
    /// scratch, the sliding layers' K / V rings (window + R slots) and the draft taps. What verify and decode alone
    /// read keeps max_rows: the heads (head*, logits, argmax), the rings' histories, the demand and the plan's
    /// counts; and the attention's score scratch (a prompt chunk's launches take as many rows as it holds at the
    /// chunk's longest row, forward.layers).
    fn promptBuffers(m: *Model, R: usize, out: *[max_grown]Grown) []Grown {
        const H = m.c.hidden;
        const W = m.world;
        const k = m.c.topk;
        const hd = m.c.head_dim;
        const qw = m.hq() * hd;
        const kvw = m.hkv() * hd;
        const pairs = R * k;
        const items = cuda.grouped.maxItems(pairs, m.c.experts, kern.decode_tile);
        const x = &m.x;
        var n: usize = 0;
        const list = [_]Grown{
            .{ .buf = &x.ids, .bytes = R * 4 },
            .{ .buf = &x.emb, .bytes = R * H * 2 },
            .{ .buf = &x.h, .bytes = R * H * 2 },
            .{ .buf = &x.blocks, .bytes = R * m.smax() * H * 2 },
            .{ .buf = &x.ar, .bytes = 3 * R * H * 4 },
            .{ .buf = &x.xq, .bytes = R * H * 2 },
            .{ .buf = &x.xk, .bytes = R * H * 2 },
            .{ .buf = &x.xv, .bytes = R * H * 2 },
            .{ .buf = &x.q, .bytes = R * qw * 2 },
            .{ .buf = &x.k, .bytes = R * kvw * 2 },
            .{ .buf = &x.v, .bytes = R * kvw * 2 },
            .{ .buf = &x.r, .bytes = R * m.hq() * m.c.d_rel * 2 },
            .{ .buf = &x.rel, .bytes = R * m.hq() * @max(m.c.window, m.c.rel_extent) * 2 },
            .{ .buf = &x.gate, .bytes = R * qw * 2 },
            .{ .buf = &x.sinks, .bytes = R * m.hq() * 2 },
            .{ .buf = &x.dyn, .bytes = R * m.hkv() * m.c.conv * 2 },
            .{ .buf = &x.dyn32, .bytes = R * m.hkv() * m.c.conv * 4 },
            .{ .buf = &x.attn, .bytes = R * qw * 2 },
            .{ .buf = &x.o32, .bytes = R * H * 4 },
            .{ .buf = &x.o, .bytes = R * H * 2 },
            .{ .buf = &x.res, .bytes = R * H * 4 },
            .{ .buf = &x.xm, .bytes = R * H * 2 },
            .{ .buf = &x.lg, .bytes = R * m.c.experts * 4 },
            .{ .buf = &x.picks, .bytes = pairs * 4 },
            .{ .buf = &x.wts, .bytes = pairs * 4 },
            .{ .buf = &x.want_ids, .bytes = pairs * 4 },
            .{ .buf = &x.want_sc, .bytes = pairs * 4 },
            .{ .buf = &x.members, .bytes = pairs * 4 },
            .{ .buf = &x.items, .bytes = 3 * items * 4 },
            .{ .buf = &x.plan_rank, .bytes = pairs * 4 },
            .{ .buf = &x.act, .bytes = pairs * m.inter() * 2 },
            .{ .buf = &x.y, .bytes = pairs * H * 4 },
            .{ .buf = &x.mo, .bytes = (R + 1) * H * 2 }, // moeSplit: two halves of (R + 1) / 2 rows
            .{ .buf = &x.fin, .bytes = R * H * 4 },
            .{ .buf = &x.edges, .bytes = kern.Kernels.sconvEdgeBytes(R, H, m.c.conv) }, // H: the widest ring
            .{ .buf = &x.e_xq, .bytes = m.exl3Bytes(R, .xq) },
            .{ .buf = &x.e_xr, .bytes = m.exl3Bytes(R, .xr) },
            // e_z / e_zd: only chunks under kern.exl3_fused_rows rows write them (from there the GEMMs take their
            // epilogues: forward.moeExl3)
            .{ .buf = &x.e_z, .bytes = m.exl3Bytes(@min(R, kern.exl3_fused_rows - 1), .z) },
            .{ .buf = &x.e_zd, .bytes = m.canonBytes(@min(R, kern.exl3_fused_rows - 1), .zd) },
            .{ .buf = &x.e_amax, .bytes = m.exl3Bytes(R, .amax) },
            .{ .buf = &x.e_amax_all, .bytes = m.exl3Bytes(R, .amax_all) },
            .{ .buf = &x.canon_tmp, .bytes = m.canonBytes(R, .tmp) },
        };
        for (list) |g| {
            out[n] = g;
            n += 1;
        }
        if (m.mix) { // x-mixed: moeMix's split items and the EXL3 SwiGLU rows, only in the mixed store
            for ([_]Grown{ .{ .buf = &x.items_n, .bytes = 3 * items * 4 }, .{ .buf = &x.items_x, .bytes = 3 * items * 4 }, .{ .buf = &x.e_act, .bytes = pairs * m.inter() * 2 } }) |g| {
                out[n] = g;
                n += 1;
            }
        }
        if (m.mx_act) { // p4-mx: the W8A8 MXFP8 experts' q8 rows of x.xm, only when on (off: phase 3's buffers exactly)
            out[n] = .{ .buf = &x.mxq, .bytes = R * kern.Kernels.q8Stride(H) };
            n += 1;
        }
        // the TP exchanges' gathered rows (world > 1), or the one-rank sum's (world 1): only the form this rank runs
        if (W > 1) {
            // y_all / o_all: the split sums (moeSplit, oprojSplit) receive the other rank's partials of half the rows;
            // else every rank's partials of every row are gathered
            const half = W == 2 and m.split_moe;
            // p4-xchg: the MoE's sum by token rows (forward.moeTok) gathers / receives its rows in o_all (its sizes
            // here hold them) and never reads y_all: y_all keeps makeScratch's size
            const y_all = if (m.moe_xchg == .token) 0 else if (half) (R + 1) / 2 * k * H * 4 else W * pairs * H * 4;
            const o_all = if (half and m.prefill_kernels) (R + 1) / 2 * H * 4 else W * R * H * 4;
            for ([_]Grown{ .{ .buf = &x.emb_all, .bytes = W * R * H * 2 }, .{ .buf = &x.o_all, .bytes = o_all }, .{ .buf = &x.y_all, .bytes = y_all } }) |g| {
                out[n] = g;
                n += 1;
            }
        } else {
            out[n] = .{ .buf = &x.ysum, .bytes = pairs * H * 4 };
            n += 1;
            if (m.canon) {
                // canonical TP at world 1: the pair's two planes of the embedding's and o_proj's partials
                for ([_]Grown{ .{ .buf = &x.emb_all, .bytes = 2 * R * H * 2 }, .{ .buf = &x.o_all, .bytes = 2 * R * H * 4 } }) |g| {
                    out[n] = g;
                    n += 1;
                }
            }
        }
        if (m.spec) |sp| {
            out[n] = .{ .buf = &sp.taps, .bytes = R * 8 * H * 2 };
            n += 1;
        }
        for (m.layers, 0..) |l, i| if (l.local) {
            const ring = (m.c.window + R) * kvw * 2;
            out[n] = .{ .buf = &m.state.kc[i], .bytes = ring };
            out[n + 1] = .{ .buf = &m.state.vc[i], .bytes = ring };
            n += 2;
        };
        return out[0..n];
    }

    /// Device bytes growPrompt(rows) adds (0 at or below the current prompt rows).
    pub fn promptGrowth(m: *Model, rows: usize) usize {
        if (rows <= m.prompt_rows) return 0;
        var list: [max_grown]Grown = undefined;
        var n: usize = 0;
        for (m.promptBuffers(rows, &list)) |g| n += @max(g.bytes, 16) -| g.buf.len;
        return n;
    }

    /// Prompt chunks of up to `rows` rows from here on (promptBuffers re-made at that size). Called once the arena is
    /// filled (its size is the floor's, computed before: the residency does not move) and before a request, with the
    /// sequence empty (the sliding rings' slot count changes; reset after). Graphs are dropped (none should exist).
    /// A row's arithmetic does not depend on the chunk's width but for the dense products' summation order (cuBLASLt
    /// plans a width of its own): the attention, the relative bias pieces and the experts' per-pair sums are the same.
    pub fn growPrompt(m: *Model, rows: usize) !void {
        if (rows <= m.prompt_rows) return;
        if (m.state.len != 0) return error.SequenceNotEmpty;
        m.graphs.drop(m.gpa);
        try m.s.synchronize();
        var list: [max_grown]Grown = undefined;
        for (m.promptBuffers(rows, &list)) |g| {
            const bytes = @max(g.bytes, 16);
            if (bytes <= g.buf.len) continue;
            if (g.buf.ptr == 0) {
                g.buf.* = try m.alloc(bytes); // one only a wide chunk uses: made here, the first time
                continue;
            }
            var nb = try Buf.alloc(m.d, bytes);
            // every grown buffer is one of the model's own (alloc): its entry freed and replaced
            const at = for (m.bufs.items, 0..) |b, j| {
                if (b.ptr == g.buf.ptr) break j;
            } else {
                nb.free();
                return error.NotModelBuffer;
            };
            m.bufs.items[at].free();
            m.bufs.items[at] = nb;
            g.buf.* = nb;
        }
        for (m.layers, 0..) |l, i| if (l.local) {
            m.state.slots[i] = m.c.window + rows;
        };
        m.prompt_rows = rows;
        try m.reset();
    }

    /// Words of one slot of format index `f` (0 NVFP4, 1 MXFP8) for this rank.
    pub fn slotWords(m: *const Model) [2][2]usize {
        return .{ experts_mod.slotWords(.nvfp4, m.c.hidden, m.inter()), experts_mod.slotWords(.mxfp8, m.c.hidden, m.inter()) };
    }

    /// The arena: `resident[f]` slots of each format's pool plus `transient[f]` (exact mode's ring), every LUT entry -1.
    pub fn allocArena(m: *Model, resident: [2]usize, transient: [2]usize) !void {
        if (m.arena_on) return error.ArenaAllocated;
        m.graphs.drop(m.gpa);
        const E = m.c.experts;
        const w = m.slotWords();
        const n = [2]usize{ resident[0] + transient[0], resident[1] + transient[1] };
        const L = m.c.layers;
        var a: Arena = undefined;
        // p4-mem: with the lazy context the four slot arrays are ranges of 64 MiB chunks (views of lazy.zig's), so a
        // growing context can take whole chunks of slots back (adapt.zig giveBack)
        const lz = m.lazy;
        a.nv_up = if (lz) |z| try z.arenaPart(0, n[0] * w[0][0] * 4) else try Buf.alloc(m.d, @max(16, n[0] * w[0][0] * 4));
        errdefer a.nv_up.free();
        a.nv_down = if (lz) |z| try z.arenaPart(1, n[0] * w[0][1] * 4) else try Buf.alloc(m.d, @max(16, n[0] * w[0][1] * 4));
        errdefer a.nv_down.free();
        a.nv_up_scale = try Buf.alloc(m.d, @max(16, n[0] * 2 * 4));
        errdefer a.nv_up_scale.free();
        a.nv_down_scale = try Buf.alloc(m.d, @max(16, n[0] * 4));
        errdefer a.nv_down_scale.free();
        a.mx_up = if (lz) |z| try z.arenaPart(2, n[1] * w[1][0] * 4) else try Buf.alloc(m.d, @max(16, n[1] * w[1][0] * 4));
        errdefer a.mx_up.free();
        a.mx_down = if (lz) |z| try z.arenaPart(3, n[1] * w[1][1] * 4) else try Buf.alloc(m.d, @max(16, n[1] * w[1][1] * 4));
        errdefer a.mx_down.free();
        a.lut = try Buf.alloc(m.d, L * E * 4);
        errdefer a.lut.free();
        a.routable = try Buf.alloc(m.d, L * E);
        errdefer a.routable.free();
        a.nv_slots = n[0];
        a.mx_slots = n[1];
        a.host_lut = try m.gpa.alloc(i32, L * E);
        a.host_routable = try m.gpa.alloc(u8, L * E);
        a.words = w;
        a.resident = resident;
        a.owner = .{ try m.gpa.alloc(i32, n[0]), try m.gpa.alloc(i32, n[1]) };
        @memset(a.host_lut, -1);
        @memset(a.host_routable, 0);
        @memset(a.owner[0], -1);
        @memset(a.owner[1], -1);
        m.arena = a;
        m.arena_on = true;
        if (lz) |z| z.arena_boot = n; // p4-mem: what a reclaim maps back to at most
        try m.publish();
    }

    /// A packed slot `sl` (layer, expert's) uploaded into its format's slot `slot`; the host LUT maps it there and
    /// marks it routable (the caller publishes). The slot's previous owner, if any, is unmapped first. The arena's
    /// format of a layer is its release format (Config.act; x-merge: the same as formats for the release).
    pub fn put(m: *Model, layer: usize, expert: usize, slot: usize, sl: *const experts_mod.Slot) !void {
        const a = &m.arena;
        const fi: usize = if (sl.fmt == .nvfp4) 0 else 1;
        if (sl.fmt != m.c.act[layer] or slot >= a.slots(fi)) return error.BadSlot;
        m.evict(fi, slot);
        switch (sl.fmt) {
            .nvfp4 => {
                try a.nv_up.upload(slot * sl.up.len * 4, std.mem.sliceAsBytes(sl.up));
                try a.nv_down.upload(slot * sl.down.len * 4, std.mem.sliceAsBytes(sl.down));
                const g = [2]f32{ sl.g13, sl.g13 };
                try a.nv_up_scale.upload(slot * 8, std.mem.sliceAsBytes(&g));
                try a.nv_down_scale.upload(slot * 4, std.mem.asBytes(&sl.g2));
            },
            .mxfp8 => {
                try a.mx_up.upload(slot * sl.up.len * 4, std.mem.sliceAsBytes(sl.up));
                try a.mx_down.upload(slot * sl.down.len * 4, std.mem.sliceAsBytes(sl.down));
            },
            .exl3 => return error.BadSlot,
        }
        m.place(layer, expert, fi, slot);
    }

    /// Host bookkeeping of a slot's new owner: (layer, expert) mapped to slot `slot` of format index `fi`'s pool and
    /// marked routable, the slot's previous owner unmapped first (the caller publishes).
    pub fn place(m: *Model, layer: usize, expert: usize, fi: usize, slot: usize) void {
        const a = &m.arena;
        const E = m.c.experts;
        m.evict(fi, slot);
        a.owner[fi][slot] = @intCast(layer * E + expert);
        a.host_lut[layer * E + expert] = @intCast(slot);
        a.host_routable[layer * E + expert] = 1;
    }

    /// Unmaps whatever slot `slot` of format index `fi` holds (the caller publishes).
    pub fn evict(m: *Model, fi: usize, slot: usize) void {
        const a = &m.arena;
        if (a.owner[fi][slot] < 0) return;
        const old: usize = @intCast(a.owner[fi][slot]);
        a.host_lut[old] = -1;
        a.host_routable[old] = 0;
        a.owner[fi][slot] = -1;
    }

    /// Speculation on: the taps buffer and the rings' histories (every chunk records them from now on).
    pub fn enableSpec(m: *Model) !void {
        if (m.spec != null) return;
        m.graphs.drop(m.gpa);
        const sp = try m.gpa.create(Spec);
        const kk = m.c.conv - 1;
        const kvw = m.hkv() * m.c.head_dim;
        sp.* = .{ .taps = try m.alloc(m.max_rows * 8 * m.c.hidden * 2), .hist = try m.gpa.alloc([4]Buf, m.c.layers), .widths = .{ kvw, kvw, m.c.hidden, m.c.hidden }, .hist_all = undefined };
        for (&sp.hist_all, sp.widths) |*b, w| b.* = try m.alloc(m.c.layers * (kk + m.max_rows) * w * 4);
        for (sp.hist, 0..) |*h, li| for (h, sp.hist_all, sp.widths) |*b, all, w| {
            const each = (kk + m.max_rows) * w * 4;
            b.* = .{ .d = m.d, .ptr = all.ptr + li * each, .len = each, .borrowed = true };
        };
        m.spec = sp;
    }

    /// The history buffer of layer `li`'s ring `which` (0 k, 1 v, 2 attn, 3 mlp); 0 without speculation.
    pub fn hist(m: *const Model, li: usize, which: usize) u64 {
        const sp = m.spec orelse return 0;
        return sp.hist[li][which].ptr;
    }

    /// After a chunk of rows starting at position `base`, keep its first `keep` rows: the length goes back to
    /// base + keep and every ring to its state after those rows (hist rows keep .. keep + K - 2). The caches are
    /// positional: rows past the kept ones are overwritten by the next chunk before anything reads them.
    pub fn rewind(m: *Model, base: usize, keep: usize) !void {
        const sp = m.spec orelse return error.NoSpeculation;
        const kk = m.c.conv - 1;
        const st = &m.state;
        if (keep > m.max_rows) return error.BadKeep;
        // one strided copy a ring kind: every layer's ring <- its history rows keep .. keep + K - 2
        for (st.rings_all, sp.hist_all, sp.widths) |ring, h, w| {
            try copy2d(m, ring.ptr, kk * w * 4, h.ptr + keep * w * 4, (kk + m.max_rows) * w * 4, kk * w * 4, m.c.layers);
        }
        st.len = base + keep;
    }

    /// Exact mode: every transient slot unmapped (after the stream is idle; the caller publishes).
    pub fn dropTransient(m: *Model) void {
        for (0..2) |fi| {
            for (m.arena.resident[fi]..m.arena.resident[fi] + m.transient_used[fi]) |slot| m.evict(fi, slot);
            m.transient_used[fi] = 0;
        }
    }

    /// p4-mem: floats a (row, head) of a decode / verify chunk's score scratch (forward.layers): the context's, or with
    /// the lazy context the mapped positions' (the scratch is mapped for them; ink2_attention reads it as a stride, and
    /// lazy.zig drops the graphs, which hold it, when it changes).
    pub fn scoreCap(m: *const Model) usize {
        const positions = if (m.lazy) |lz| lz.mapped else m.context;
        return @max(positions, m.c.window + m.max_rows);
    }

    /// Uploads the host LUT and routable mask (after placements or a swap's edits).
    pub fn publish(m: *Model) !void {
        try m.arena.lut.upload(0, std.mem.sliceAsBytes(m.arena.host_lut));
        try m.arena.routable.upload(0, m.arena.host_routable);
    }

    /// x-mixed: the bank published (`show`: the host's LUT and mask, publish) or hidden (every LUT entry -1 and the
    /// mask 0: each pick on the EXL3 store) without touching the host's maps; graphs survive either (the run tool's arm
    /// switch bank=0|1: one load measures the EXL3 store alone and the mixed store).
    pub fn publishBank(m: *Model, show: bool) !void {
        if (m.bank_mask) |mk| m.gpa.free(mk);
        m.bank_mask = null;
        m.bank_hidden = !show;
        if (show) return m.publish();
        const n = m.c.layers * m.c.experts;
        const none = try m.gpa.alloc(i32, n);
        defer m.gpa.free(none);
        @memset(none, -1);
        const zero = try m.gpa.alloc(u8, n);
        defer m.gpa.free(zero);
        @memset(zero, 0);
        try m.arena.lut.upload(0, std.mem.sliceAsBytes(none));
        try m.arena.routable.upload(0, zero);
    }

    /// x-mixed: a partial view of the bank published: (layer, expert) on the device as the host's bank has it where
    /// `mask` is 1, else on the EXL3 store (LUT -1, mask 0); the host's bank is untouched and the view is kept for
    /// forward.mixCheck. The run tool's bankfrac / bankmx arms (frozen banks: an admission's publish shows the whole bank
    /// again, so the run tool refuses admission with a partial view).
    pub fn publishBankMask(m: *Model, mask: []const u8) !void {
        const n = m.c.layers * m.c.experts;
        if (mask.len != n) return error.BadBankMask;
        const mk = m.bank_mask orelse try m.gpa.alloc(u8, n);
        @memcpy(mk, mask);
        m.bank_mask = mk;
        m.bank_hidden = false;
        const lut = try m.gpa.alloc(i32, n);
        defer m.gpa.free(lut);
        const rt = try m.gpa.alloc(u8, n);
        defer m.gpa.free(rt);
        for (0..n) |i| {
            lut[i] = if (mask[i] != 0) m.arena.host_lut[i] else -1;
            rt[i] = if (mask[i] != 0) m.arena.host_routable[i] else 0;
        }
        try m.arena.lut.upload(0, std.mem.sliceAsBytes(lut));
        try m.arena.routable.upload(0, rt);
    }

    // ---- p4-cache: the sequence state at a position, saved and restored (serve/reuse.zig, TF_INKLING2_PROMPT_REUSE)
    // A pass resuming at position `at` needs, beyond what stays in place: every convolution ring (the last K - 1
    // inputs before `at`, overwritten by every later row) and each sliding layer's last `window` positions of K / V
    // (ring slots p % slots, overwritten window + prompt_rows positions on, and by a verify window's rejected rows).
    // A global layer's K / V is positional: its positions below `at` hold until a pass starts below `at`, and a pass
    // from `at` writes each chunk's rows before its attention reads them. Plain copies on the model's stream: the
    // bits of the state, nothing recomputed.

    /// Bytes of a saved state's rings part (the four rings_all blocks back to back).
    pub fn snapRingBytes(m: *const Model) usize {
        var n: usize = 0;
        for (m.state.rings_all) |b| n += b.len;
        return n;
    }

    /// Bytes of a saved state's window part: [sliding layers][k, v][window][kv width] bf16.
    pub fn snapWindowBytes(m: *const Model) usize {
        var n: usize = 0;
        for (m.layers) |l| n += @intFromBool(l.local);
        return n * 2 * m.c.window * m.hkv() * m.c.head_dim * 2;
    }

    /// The state at the current length into `rings` (snapRingBytes) and `window` (snapWindowBytes).
    pub fn saveState(m: *Model, rings: Buf, window: Buf) !void {
        var off: usize = 0;
        for (m.state.rings_all) |b| {
            try rings.copyFrom(off, b.ptr, b.len, m.s.handle);
            off += b.len;
        }
        try m.windowCopy(window, m.state.len, true);
    }

    /// The state saved at position `at` by saveState back in place; the sequence's length becomes `at`.
    pub fn restoreState(m: *Model, at: usize, rings: Buf, window: Buf) !void {
        if (at > m.context) return error.ContextFull;
        var off: usize = 0;
        for (m.state.rings_all) |b| {
            try b.copyFrom(0, rings.ptr + off, b.len, m.s.handle);
            off += b.len;
        }
        try m.windowCopy(window, at, false);
        m.state.len = at;
    }

    /// Each sliding layer's positions max(0, at - window) .. at - 1 between its K / V rings and `window`'s rows 0 ..
    /// (`save`: ring to window). Their slots are contiguous but for one wrap: at most two copies a ring.
    fn windowCopy(m: *Model, window: Buf, at: usize, save: bool) !void {
        const row = m.hkv() * m.c.head_dim * 2;
        const n = @min(at, m.c.window);
        const each = m.c.window * row;
        var j: usize = 0;
        for (m.layers, 0..) |l, li| if (l.local) {
            const slots = m.state.slots[li];
            const s0 = (at - n) % slots;
            const first = @min(n, slots - s0);
            for ([_]Buf{ m.state.kc[li], m.state.vc[li] }, 0..) |ring, kv| {
                const base = (2 * j + kv) * each;
                if (save) {
                    try window.copyFrom(base, ring.ptr + s0 * row, first * row, m.s.handle);
                    try window.copyFrom(base + first * row, ring.ptr, (n - first) * row, m.s.handle);
                } else {
                    try ring.copyFrom(s0 * row, window.ptr + base, first * row, m.s.handle);
                    try ring.copyFrom(0, window.ptr + base + first * row, (n - first) * row, m.s.handle);
                }
            }
            j += 1;
        };
    }
    // ---- end p4-cache
};

// ---- e-dense (prefill lane phase 5) ----

/// The plan without the bf16 tensors a dense EXL3 set of `classes` replaces (attn: the six projections; router: the
/// gate weight, not its bias); the head stays (the draft reads it).
fn dropReplaced(a: std.mem.Allocator, entries: []plan_mod.Entry, classes: dense_exl3.Classes) ![]plan_mod.Entry {
    var out: std.ArrayList(plan_mod.Entry) = .empty;
    for (entries) |e| {
        const attn = std.mem.indexOf(u8, e.name, ".attn.") != null and std.mem.endsWith(u8, e.name, ".weight") and
            (std.mem.indexOf(u8, e.name, ".wq_du.") != null or std.mem.indexOf(u8, e.name, ".wk_dv.") != null or
                std.mem.indexOf(u8, e.name, ".wv_dv.") != null or std.mem.indexOf(u8, e.name, ".wr_du.") != null or
                std.mem.indexOf(u8, e.name, ".wo_ud.") != null or std.mem.indexOf(u8, e.name, ".gate_attn.") != null);
        const router = std.mem.endsWith(u8, e.name, ".mlp.gate.weight");
        if ((attn and classes.contains(.attn)) or (router and classes.contains(.router))) continue;
        try out.append(a, e);
    }
    return out.toOwnedSlice(a);
}

test "inkling2 dense exl3: the replaced bf16 tensors leave the plan, the rest stays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = Config{};
    const all = try plan_mod.plan(arena.allocator(), &c, 0, 2);
    const kept = try dropReplaced(arena.allocator(), all, try dense_exl3.parseClasses("attn,router"));
    try std.testing.expectEqual(all.len - c.layers * 7, kept.len);
    for (kept) |e| {
        try std.testing.expect(std.mem.indexOf(u8, e.name, ".wq_du.") == null);
        try std.testing.expect(!std.mem.endsWith(u8, e.name, ".mlp.gate.weight"));
    }
    const head_only = try dropReplaced(arena.allocator(), all, try dense_exl3.parseClasses("head"));
    try std.testing.expectEqual(all.len, head_only.len);
}
