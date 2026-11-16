//! Inkling-2's DSpark draft head (mtp.safetensors, config.json `mtp_config`): 5 dense bf16 layers over an 8-row block
//! (the anchor, then 7 mask rows), each layer's context the target's 8 tapped features mixed (fuse_weight) and
//! normed into its own K / V; then the Markov RNN walk over the block's base logits, one token a step. The math and
//! its rounding points are the overlay's verified port (overlay/families/inkling2/dspark.py, docs/dspark-spec.md);
//! the kernels are inkling2_dspark.cu, the dense products cuBLASLt's (gemm.zig).
//!
//! At TP the draft's layers are split like the target's: each rank holds its query / kv heads (q, k, v, k_ctx, v_ctx
//! rows; its share of the context rings) and its MLP columns (gate / up rows), and o_proj / down_proj produce fp32
//! partials over them, gathered and summed in rank order, then rounded once to bf16 (the reference's bf16 GEMM
//! output; only the summation order differs from one rank's GEMM) before the fused add + rmsnorm. What is the target's
//! -- the embedding, the LM head, so the Markov fold -- stays split by vocabulary, and each step's pick is the largest
//! of every rank's (value, id) (the lower id on ties, rank order = ascending vocabulary). The Markov walk is two
//! kernels a step (the recurrence, then the fold + softcap + argmax) and one 8-byte gather a rank between steps.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const page_cache = @import("cache.zig"); // e-load (x5): mtp.safetensors out of the page cache once on the device
const model_mod = @import("model.zig");
const forward = @import("forward.zig");
const kern = @import("kern.zig"); // p4-dattn
const draft_graph = @import("draft_graph.zig"); // e-step
const draft_exl3 = @import("draft_exl3.zig"); // e-step
const dense_exl3 = @import("dense_exl3.zig"); // e-step: the target's EXL3 head (TF_INKLING2_DRAFT_HEAD=exl3)
const defaults = @import("defaults.zig"); // e-step (x5 port): measured wins on by default

const Model = model_mod.Model;
const Buf = cuda.DeviceBuffer;
const Args = cuda.Args;
const Function = cuda.Function;

pub const H = 4608;
pub const heads = 48;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const inter = 18432;
pub const layers = 5;
pub const block = 8; // the anchor and 7 mask rows
pub const spec = block - 1; // tokens a draft proposes
pub const rank = 256; // the Markov head's width
pub const ntaps = 8;
pub const taps = [ntaps]u32{ 5, 11, 17, 23, 29, 35, 41, 48 }; // target layer inputs (48: the final-norm output)
pub const window = 1024; // local layers' context window
const local = [layers]bool{ true, true, false, true, true };
pub const local_layers = local; // xk-rows: the state's image names the sliding layers (serve/rows_layout.zig)
const theta: f32 = 1e6;
const eps: f32 = 1e-6;
const softcap: f32 = 30;
const qw = heads * head_dim; // 6144
const kvw = kv_heads * head_dim; // 1024

const Layer = struct {
    in_norm: Buf,
    post_norm: Buf,
    dflare_norm: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    k_ctx: Buf,
    v_ctx: Buf,
    o: Buf,
    q_norm: Buf,
    k_norm: Buf,
    k_norm_ctx: Buf,
    gate: Buf,
    up: Buf,
    down: Buf,
    kc: Buf, // context ring K [slots][1024] bf16
    vc: Buf,
    slots: usize,
};

pub const Draft = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    mod: cuda.Module,
    f: struct { mix_norm: Function, rmsnorm: Function, sum_rmsnorm: Function, norm_rope: Function, attention: Function, ctx_put: Function, silu_mul: Function, mask_rows: Function, cat_rnn: Function, walk_step: Function, walk_fold: Function, pick: Function, positions: Function, fold: Function, tap: Function, fdec: Function, fdec_merge: Function }, // fdec*: p4-dattn
    layer: [layers]Layer,
    norm: Buf,
    mask_embed: Buf,
    markov_norm: Buf,
    w1: Buf, // markov_w1 [padded vocab][256]
    w_step: Buf, // rnn_proj[:, :512] [768][512]
    w_ctx: Buf, // rnn_proj[:, 512:] [768][9216]
    rnn_bias: Buf, // fp32 [768]
    fold_w: Buf, // this rank's vocabulary rows of lm_head @ hidden_out.weight [Vl][256] bf16
    fold_b: Buf, // ... and of lm_head @ hidden_out.bias [Vl] bf16
    alpha: [layers + 1][ntaps]f32, // softmax(fuse_weight) rows
    alpha_dev: Buf, // the same, fp32 [6][8]
    cs: Buf, // rope table [positions][128] fp32
    positions: usize,
    vl: usize, // this rank's vocabulary rows
    valid: usize, // of them, ids below the unpadded vocabulary
    world: usize,
    hq: usize, // this rank's query heads, kv heads, q width, kv width, MLP width
    hkv: usize,
    qwl: usize,
    kvwl: usize,
    interl: usize,
    // scratch
    x: Buf,
    res: Buf,
    xn: Buf,
    q: Buf,
    kb: Buf,
    vb: Buf,
    attn: Buf,
    o: Buf,
    g: Buf,
    u: Buf,
    act: Buf,
    hidden: Buf,
    scr: Buf,
    max_keys: usize,
    pos: Buf, // i32 positions of a call's rows
    mix: Buf, // [max_rows][H]
    kctx: Buf,
    vctx: Buf,
    frnn: Buf,
    cat: Buf,
    ctxmm: Buf,
    base: Buf,
    state: Buf, // the Markov state, ping-pong [2][r] bf16
    latent: Buf,
    part: Buf, // walk_fold's per-block bests [fold_blocks][2] fp32
    counter: Buf, // its last-block counter (u32, zero between launches)
    o32: Buf, // fp32 partials [block][H] (o_proj, down_proj) at TP
    o_all: Buf, // [world][block][H]
    best: Buf,
    best_all: Buf,
    prev: Buf,
    toks: Buf,
    ids: Buf,
    emb: Buf,
    emb_all: Buf,
    max_rows: usize,
    fd_part: Buf, // p4-dattn: dsp_fdec's split partials [block][hq][splits][2 + 128] fp32 (Model.draft_attn only)
    // ---- p4-small (TF_INKLING2_ABSORB_WIDE): absorbWide's buffers for `wide_rows` rows (none until the first wide
    // chunk: mix [rows][H], k / v context rows [rows][kv width], positions [rows])
    wide_rows: usize,
    wmix: Buf,
    wkctx: Buf,
    wvctx: Buf,
    wpos: Buf,
    // ---- end p4-small
    // ---- e-step: the pass as a CUDA graph (TF_INKLING2_DRAFT_GRAPH=1, with the draft attention and graphs on): the
    // graph, whether it is on, and the target graphs' drop count it was captured at (a drop drops it too)
    dgraph: ?*draft_graph.Entry = null,
    dgraph_on: bool = false,
    dgraph_gen: u64 = 0,
    // e-step: the seven projections a layer as EXL3 (TF_INKLING2_DRAFT_EXL3=<pack>; draft_exl3.zig) and whether the
    // pass runs them (else the bf16 ones, kept only with TF_INKLING2_DRAFT_EXL3_KEEP_BF16=1)
    dx: ?*draft_exl3.DraftExl3 = null,
    dx_on: bool = false,
    dx_bf16: bool = true,
    // e-step: TF_INKLING2_DRAFT_HEAD=exl3: the base logits on the target's EXL3 head (e-dense's active set): its
    // weight (this rank's head_n / world columns), so its vocabulary split (vl, the rank's rows at rank * vl)
    head_x: ?dense_exl3.Weight = null,
    // ---- end e-step
    bufs: std.ArrayList(Buf) = .empty,

    fn alloc(d: *Draft, bytes: usize) !Buf {
        const b = try Buf.alloc(d.m.d, @max(bytes, 16));
        try d.bufs.append(d.gpa, b);
        return b;
    }

    fn put(d: *Draft, bytes: []const u8) !Buf {
        const b = try Buf.fromHost(d.m.d, bytes);
        try d.bufs.append(d.gpa, b);
        return b;
    }

    const fold_blocks = 384;

    /// Device bytes the draft holds on this rank of `world` at `context` positions (the arena is sized after it).
    pub fn deviceBytes(context: usize, vl: usize, world: usize) usize {
        const per_layer = (qw * H + 4 * kvw * H + H * qw + 3 * inter * H) / world * 2 + 3 * H * 2 + 3 * head_dim * 2;
        const rings = 2 * kvw / world * 2 * ((layers - 1) * (window + 64) + context + 64);
        return layers * per_layer + rings + 201024 * rank * 2 + 768 * 9728 * 2 + vl * (rank + 1) * 2 + (context + 64) * head_dim * 4 + (64 << 20);
    }

    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *Model, dir: []const u8, max_rows: usize) !*Draft {
        const d = try gpa.create(Draft);
        errdefer gpa.destroy(d);
        d.* = undefined;
        d.gpa = gpa;
        d.m = m;
        d.bufs = .empty;
        d.max_rows = max_rows;
        d.wide_rows = 0; // p4-small: absorbWide's buffers made at its first wide chunk
        // e-step: TF_INKLING2_DRAFT_GRAPH=1 (the split-K draft attention and the target's graphs on: its nodes are the
        // ones finish patches)
        d.dgraph = null;
        d.dgraph_gen = 0;
        d.dgraph_on = if (std.c.getenv("TF_INKLING2_DRAFT_GRAPH")) |v| std.mem.eql(u8, std.mem.span(v), "1") and m.draft_attn and m.graphs.on else false;
        if (d.dgraph_on) std.log.info("inkling2 rank {d}: e-step TF_INKLING2_DRAFT_GRAPH=1: the DSpark pass replayed as a CUDA graph", .{m.rank});
        d.mod = try cuda.Module.load(m.d, cuda.kernels.inkling2_dspark);
        inline for (comptime std.meta.fieldNames(@TypeOf(d.f))) |fl| @field(d.f, fl) = try d.mod.function("dsp_" ++ fl);
        var ck: core.checkpoint.Checkpoint = .{ .gpa = gpa, .io = io };
        var ck_drop: page_cache.Files = .{ .gpa = gpa }; // e-load (x5): mtp.safetensors out of the page cache after ck.close()
        defer ck_drop.drop();
        defer ck.close();
        try ck.add(dir, "mtp.safetensors");
        ck_drop.take(&ck);
        var nb: [96]u8 = undefined;
        const W = m.world;
        if (heads % W != 0 or kv_heads % W != 0 or inter % W != 0) return error.DraftSplit;
        d.world = W;
        d.hq = heads / W;
        d.hkv = kv_heads / W;
        d.qwl = d.hq * head_dim;
        d.kvwl = d.hkv * head_dim;
        d.interl = inter / W;
        const rk = m.rank;
        // e-step: TF_INKLING2_DRAFT_EXL3=<pack>: the seven projections from the EXL3 pack; their bf16 copies are not
        // uploaded unless TF_INKLING2_DRAFT_EXL3_KEEP_BF16=1 (the run tool's A/B in one process)
        d.dx = null;
        d.dx_on = false;
        d.dx_bf16 = true;
        // (x5 port: unset and not TF_INKLING2_LEGACY=1, the model folder's draft-q6 pack when it has one; 0 or none: the
        // bf16 projections)
        const dx_set: ?[]const u8 = if (std.c.getenv("TF_INKLING2_DRAFT_EXL3")) |v| std.mem.span(v) else null;
        var dx_nb: [std.fs.max_path_bytes]u8 = undefined;
        const dx_dir: ?[]const u8 = if (dx_set) |v| (if (v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "none")) null else v) else if (defaults.legacy()) null else try std.fmt.bufPrint(&dx_nb, "{s}/draft-q6", .{dir});
        if (dx_dir) |dd| {
            if (draft_exl3.DraftExl3.open(gpa, io, m.d, m.s, dd, m.rank, m.world)) |x| {
                d.dx = x;
                d.dx_on = true;
                d.dx_bf16 = if (std.c.getenv("TF_INKLING2_DRAFT_EXL3_KEEP_BF16")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
            } else |err| {
                // the default's pack may be absent (a view without draft-q6): the bf16 projections; a named pack must open
                if (dx_set != null or err != error.FileNotFound) return err;
                std.log.info("inkling2 rank {d}: e-step: no draft-q6 pack in {s}: the draft's bf16 projections", .{ m.rank, dir });
            }
        }
        errdefer if (d.dx) |x| x.close();
        // a rank's columns of o_proj / down_proj, gathered on the host
        const colbuf = try gpa.alloc(u8, H * @max(d.qwl, d.interl) * 2);
        defer gpa.free(colbuf);
        for (&d.layer, 0..) |*l, i| {
            const T = struct {
                fn t(c: *core.checkpoint.Checkpoint, b: *[96]u8, li: usize, comptime n: []const u8, shape: []const usize) ![]const u8 {
                    return (try c.expect(try std.fmt.bufPrint(b, "layers.{d}." ++ n, .{li}), .bf16, shape)).bytes;
                }
                // rows [r * n, (r + 1) * n) of a row-major [n * world][k] bf16 matrix (q / k / v heads, gate / up)
                fn rows(x: []const u8, r: usize, n: usize, k: usize) []const u8 {
                    return x[r * n * k * 2 ..][0 .. n * k * 2];
                }
                // columns [r * n, (r + 1) * n) of a row-major [nrows][n * world] bf16 matrix (o_proj, down_proj)
                fn cols(x: []const u8, out: []u8, nrows: usize, r: usize, n: usize, w: usize) []const u8 {
                    for (0..nrows) |row| @memcpy(out[row * n * 2 ..][0 .. n * 2], x[(row * n * w + r * n) * 2 ..][0 .. n * 2]);
                    return out[0 .. nrows * n * 2];
                }
            };
            l.in_norm = try d.put(try T.t(&ck, &nb, i, "input_layernorm.weight", &.{H}));
            l.post_norm = try d.put(try T.t(&ck, &nb, i, "post_attention_layernorm.weight", &.{H}));
            l.dflare_norm = try d.put(try T.t(&ck, &nb, i, "dflare_norm.weight", &.{H}));
            const none: Buf = .{ .d = m.d, .ptr = 0, .len = 0 }; // e-step: a projection the EXL3 pack replaces
            l.q = if (d.dx_bf16) try d.put(T.rows(try T.t(&ck, &nb, i, "self_attn.q_proj.weight", &.{ qw, H }), rk, d.qwl, H)) else none;
            l.k = if (d.dx_bf16) try d.put(T.rows(try T.t(&ck, &nb, i, "self_attn.k_proj.weight", &.{ kvw, H }), rk, d.kvwl, H)) else none;
            l.v = if (d.dx_bf16) try d.put(T.rows(try T.t(&ck, &nb, i, "self_attn.v_proj.weight", &.{ kvw, H }), rk, d.kvwl, H)) else none;
            l.k_ctx = try d.put(T.rows(try T.t(&ck, &nb, i, "self_attn.k_proj_ctx.weight", &.{ kvw, H }), rk, d.kvwl, H));
            l.v_ctx = try d.put(T.rows(try T.t(&ck, &nb, i, "self_attn.v_proj_ctx.weight", &.{ kvw, H }), rk, d.kvwl, H));
            l.o = if (d.dx_bf16) try d.put(T.cols(try T.t(&ck, &nb, i, "self_attn.o_proj.weight", &.{ H, qw }), colbuf, H, rk, d.qwl, W)) else none;
            l.q_norm = try d.put(try T.t(&ck, &nb, i, "self_attn.q_norm.weight", &.{head_dim}));
            l.k_norm = try d.put(try T.t(&ck, &nb, i, "self_attn.k_norm.weight", &.{head_dim}));
            l.k_norm_ctx = try d.put(try T.t(&ck, &nb, i, "self_attn.k_norm_ctx.weight", &.{head_dim}));
            l.gate = if (d.dx_bf16) try d.put(T.rows(try T.t(&ck, &nb, i, "mlp.gate_proj.weight", &.{ inter, H }), rk, d.interl, H)) else none;
            l.up = if (d.dx_bf16) try d.put(T.rows(try T.t(&ck, &nb, i, "mlp.up_proj.weight", &.{ inter, H }), rk, d.interl, H)) else none;
            l.down = if (d.dx_bf16) try d.put(T.cols(try T.t(&ck, &nb, i, "mlp.down_proj.weight", &.{ H, inter }), colbuf, H, rk, d.interl, W)) else none;
            l.slots = if (local[i]) window + 64 else m.context + 64;
            if (m.lazy != null and !local[i]) {
                // p4-mem: the global layer's ring reserved whole, mapped for the target's mapped positions + 64
                try m.lazy.?.register(d, &l.kc, d.kvwl * 2, 64, 0, "draft k");
                try m.lazy.?.register(d, &l.vc, d.kvwl * 2, 64, 0, "draft v");
            } else {
                l.kc = try d.alloc(l.slots * d.kvwl * 2);
                l.vc = try d.alloc(l.slots * d.kvwl * 2);
            }
        }
        d.norm = try d.put((try ck.expect("norm.weight", .bf16, &.{H})).bytes);
        d.mask_embed = try d.put((try ck.expect("mask_embed", .bf16, &.{H})).bytes);
        d.markov_norm = try d.put((try ck.expect("markov_head.dflare_norm.weight", .bf16, &.{H})).bytes);
        const padded = m.c.vocab;
        d.w1 = try d.put((try ck.expect("markov_head.markov_w1.weight", .bf16, &.{ padded, rank })).bytes);
        // rnn_proj [768][9728]: columns [state | w1(prev) | hidden | f_rnn] -> the step part and the context part
        {
            const rp = try ck.expect("markov_head.rnn_proj.weight", .bf16, &.{ 3 * rank, 2 * rank + 2 * H });
            const cols = 2 * rank + 2 * H;
            const step = try gpa.alloc(u8, 3 * rank * 2 * rank * 2);
            defer gpa.free(step);
            const ctxp = try gpa.alloc(u8, 3 * rank * 2 * H * 2);
            defer gpa.free(ctxp);
            for (0..3 * rank) |r| {
                @memcpy(step[r * 2 * rank * 2 ..][0 .. 2 * rank * 2], rp.bytes[r * cols * 2 ..][0 .. 2 * rank * 2]);
                @memcpy(ctxp[r * 2 * H * 2 ..][0 .. 2 * H * 2], rp.bytes[(r * cols + 2 * rank) * 2 ..][0 .. 2 * H * 2]);
            }
            d.w_step = try d.put(step);
            d.w_ctx = try d.put(ctxp);
            const rb = try ck.expect("markov_head.rnn_proj.bias", .bf16, &.{3 * rank});
            var bias: [3 * rank]f32 = undefined;
            for (&bias, 0..) |*b, j| b.* = bf16(std.mem.bytesToValue(u16, rb.bytes[j * 2 ..][0..2]));
            d.rnn_bias = try d.put(std.mem.sliceAsBytes(&bias));
        }
        // fuse_weight [6][8] -> softmax rows in fp32 (M:502-504)
        {
            const fw = try ck.expect("fuse_weight", .bf16, &.{ layers + 1, ntaps });
            for (0..layers + 1) |r| {
                var row: [ntaps]f32 = undefined;
                var mx: f32 = -std.math.inf(f32);
                for (&row, 0..) |*x, t| {
                    x.* = bf16(std.mem.bytesToValue(u16, fw.bytes[(r * ntaps + t) * 2 ..][0..2]));
                    mx = @max(mx, x.*);
                }
                var sum: f32 = 0;
                for (&row) |*x| {
                    x.* = @exp(x.* - mx);
                    sum += x.*;
                }
                for (&row, 0..) |x, t| d.alpha[r][t] = x / sum;
            }
            d.alpha_dev = try d.put(std.mem.sliceAsBytes(&d.alpha));
        }
        // rope table (M:409-414): inv_freq = 1 / theta^(2i / 128), freqs = pos * inv_freq, [cos | sin] fp32
        d.positions = m.context + 64;
        if (m.lazy) |lz| {
            // p4-mem: the table reserved for every position, written for the mapped ones (grown: as they grow)
            try lz.register(d, &d.cs, head_dim * 4, 64, 0, "draft rope");
            try d.ropeRows(0, lz.mapped + 64);
        } else {
            const tab = try gpa.alloc(f32, d.positions * head_dim);
            defer gpa.free(tab);
            for (0..d.positions) |p| for (0..head_dim / 2) |j| {
                const inv: f32 = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * j)) / @as(f32, head_dim));
                const fr: f32 = @as(f32, @floatFromInt(p)) * inv;
                tab[p * head_dim + j] = @cos(fr);
                tab[p * head_dim + head_dim / 2 + j] = @sin(fr);
            };
            d.cs = try d.put(std.mem.sliceAsBytes(tab));
        }
        // the Markov fold on this rank's LM head rows, in fp64 (M:348-360)
        d.vl = m.vocabLocal();
        d.valid = if (m.c.unpadded_vocab > m.vocab_lo) @min(d.vl, m.c.unpadded_vocab - m.vocab_lo) else 0;
        // ---- e-step: TF_INKLING2_DRAFT_HEAD=exl3 (two ranks, e-dense's set with a head): the draft takes the EXL3
        // head's vocabulary split, rank r's rows [r vl, (r + 1) vl) with vl = its columns (whole 128-blocks: 100096 of
        // 100512), and its fold is made for those rows below (the rows a rank's bf16 half lacks read from the checkpoint)
        d.head_x = null;
        // (x5 port: on by default when the active set has a head; TF_INKLING2_DRAFT_HEAD=bf16 or TF_INKLING2_LEGACY=1: off)
        const want_xhead = if (std.c.getenv("TF_INKLING2_DRAFT_HEAD")) |v| std.mem.eql(u8, std.mem.span(v), "exl3") else !defaults.legacy();
        if (want_xhead and m.world == 2 and !m.canon) if (m.dx) |dxx| if (dxx.active()) |set| if (set.head) |h| {
            d.head_x = h;
        };
        const head_lo: usize = if (d.head_x) |h| m.rank * h.n else m.vocab_lo;
        if (d.head_x) |h| {
            d.vl = h.n;
            d.valid = if (m.c.unpadded_vocab > head_lo) @min(d.vl, m.c.unpadded_vocab - head_lo) else 0;
            std.log.info("inkling2 rank {d}: e-step TF_INKLING2_DRAFT_HEAD=exl3: the draft's base logits on the EXL3 head, vocabulary rows [{d}, {d}) of this rank ({d} real)", .{ m.rank, head_lo, head_lo + d.vl, d.valid });
        }
        // ---- end e-step
        {
            const how = try ck.expect("markov_head.hidden_out.weight", .bf16, &.{ H, rank });
            const hob = try ck.expect("markov_head.hidden_out.bias", .bf16, &.{H});
            var proj = try Buf.fromHost(m.d, how.bytes);
            defer proj.free();
            var pb = try Buf.fromHost(m.d, hob.bytes);
            defer pb.free();
            d.fold_w = try d.alloc(d.vl * rank * 2);
            d.fold_b = try d.alloc(d.vl * 2);
            try d.f.fold.allowDynamicShared(H * 8);
            if (d.head_x == null) {
                var a: Args = .{};
                for ([_]u64{ m.unembed, proj.ptr, pb.ptr }) |v| a.add(v);
                a.add(@as(c_int, H));
                a.add(@as(c_int, rank));
                for ([_]u64{ d.fold_w.ptr, d.fold_b.ptr }) |v| a.add(v);
                try go(d.f.fold, m.s, d.vl, rank, H * 8, &a);
            } else {
                // e-step: rows [head_lo, head_lo + vl): those in this rank's bf16 half from m.unembed, the others from the
                // checkpoint's unembed (a few hundred rows), each range folded into its place
                const own_lo = m.vocab_lo;
                const own_hi = m.vocab_lo + m.vocabLocal();
                var hk = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
                defer hk.close();
                const un = try hk.get("model.llm.unembed.weight");
                var at = head_lo;
                const end = head_lo + d.vl;
                while (at < end) {
                    const here = at >= own_lo and at < own_hi;
                    const stop = if (here) @min(end, own_hi) else if (at < own_lo) @min(end, own_lo) else end;
                    var tmp: ?Buf = null;
                    defer if (tmp) |*t| t.free();
                    const src: u64 = if (here) m.unembed + (at - own_lo) * H * 2 else blk: {
                        tmp = try Buf.fromHost(m.d, un.bytes[at * H * 2 .. stop * H * 2]);
                        break :blk tmp.?.ptr;
                    };
                    var a: Args = .{};
                    for ([_]u64{ src, proj.ptr, pb.ptr }) |v| a.add(v);
                    a.add(@as(c_int, H));
                    a.add(@as(c_int, rank));
                    for ([_]u64{ d.fold_w.ptr + (at - head_lo) * rank * 2, d.fold_b.ptr + (at - head_lo) * 2 }) |v| a.add(v);
                    try go(d.f.fold, m.s, stop - at, rank, H * 8, &a);
                    try m.s.synchronize(); // the range's temporary rows are freed at the loop's end
                    at = stop;
                }
            }
            try m.s.synchronize();
        }
        // scratch
        const R = @max(max_rows, block);
        d.max_keys = if (m.lazy) |lz| lz.mapped + 64 + block else m.context + 64 + block; // p4-mem: grown
        d.x = try d.alloc(block * H * 2);
        d.res = try d.alloc(block * H * 2);
        d.xn = try d.alloc(block * H * 2);
        d.q = try d.alloc(block * d.qwl * 2);
        d.kb = try d.alloc(block * d.kvwl * 2);
        d.vb = try d.alloc(block * d.kvwl * 2);
        d.attn = try d.alloc(block * d.qwl * 2);
        d.o = try d.alloc(block * H * 2);
        d.g = try d.alloc(block * d.interl * 2);
        d.u = try d.alloc(block * d.interl * 2);
        d.act = try d.alloc(block * d.interl * 2);
        d.hidden = try d.alloc(block * H * 2);
        if (m.lazy) |lz| {
            // p4-mem: max_keys floats a (row, head), mapped for the mapped positions; grown keeps the stride with them
            // (and writes the rope rows). With Model.draft_attn (p4-dattn) dsp_attention never runs: no scratch made
            if (m.draft_attn) {
                d.scr = .{ .d = m.d, .ptr = 0, .len = 0 };
            } else try lz.register(d, &d.scr, block * d.hq * 4, 64 + block, 0, "draft scores");
            try lz.hook(d, d, grown);
        } else d.scr = try d.alloc(block * d.hq * d.max_keys * 4);
        // p4-dattn: the split kernels' partials (with Model.draft_attn: d.scr is then never read)
        d.fd_part = if (m.draft_attn) try d.alloc(block * d.hq * kern.Kernels.fdec_max_splits * (2 + head_dim) * 4) else .{ .d = m.d, .ptr = 0, .len = 0 };
        d.pos = try d.alloc(R * 4);
        d.mix = try d.alloc(R * H * 2);
        d.kctx = try d.alloc(R * d.kvwl * 2);
        d.vctx = try d.alloc(R * d.kvwl * 2);
        d.frnn = try d.alloc(H * 2);
        d.cat = try d.alloc(spec * 2 * H * 2);
        d.ctxmm = try d.alloc(spec * 3 * rank * 4);
        d.base = try d.alloc(spec * d.vl * 2);
        d.state = try d.alloc(2 * rank * 2);
        d.latent = try d.alloc(rank * 2);
        d.part = try d.alloc(fold_blocks * 8);
        d.counter = try d.alloc(16);
        try d.counter.fill8(0, m.s.handle);
        d.o32 = try d.alloc(block * H * 4);
        d.o_all = try d.alloc(m.world * block * H * 4);
        try d.f.sum_rmsnorm.allowDynamicShared(H * 4);
        d.best = try d.alloc(8);
        d.best_all = try d.alloc(8 * m.world);
        d.prev = try d.alloc(4);
        d.toks = try d.alloc(spec * 4);
        d.ids = try d.alloc(4);
        d.emb = try d.alloc(H * 2);
        d.emb_all = try d.alloc(m.world * H * 2);
        return d;
    }

    /// Device bytes this rank's draft head holds (weights, context rings, fold, scratch).
    pub fn heldBytes(d: *const Draft) usize {
        var n: usize = 0;
        for (d.bufs.items) |b| n += b.len;
        return n;
    }

    pub fn close(d: *Draft) void {
        if (d.dgraph) |e| e.deinit(d.gpa); // e-step
        if (d.dx) |x| x.close(); // e-step
        if (d.m.lazy) |lz| lz.drop(d); // p4-mem: its context-sized ranges
        for (d.bufs.items) |*b| b.free();
        d.bufs.deinit(d.gpa);
        d.mod.unload();
        d.gpa.destroy(d);
    }

    // ---- p4-mem: the draft head's context-sized buffers with the lazy context (lazy.zig)

    /// The lazy context's mapped positions went from `old` to `new`: the score scratch's stride follows them, and the
    /// rope table's rows of the positions mapped now are written.
    fn grown(ctx: *anyopaque, old: usize, new: usize) anyerror!void {
        const d: *Draft = @ptrCast(@alignCast(ctx));
        d.max_keys = new + 64 + block;
        if (new > old) try d.ropeRows(old + 64, new + 64);
    }

    /// Rows [p0, p1) of the rope table (the whole table's arithmetic at open, row for row: the same values).
    /// xr: each inverse frequency computed once (the same expression: the same f32), rows on up to 8 threads past 8192
    /// of them: a 138K restore's growth wrote 131,072 rows in 170 ms on one thread with pow in the loop, 19 ms so
    /// (the bits equal: the bench in out/prefill/p5/p4-cache/xr/).
    fn ropeRows(d: *Draft, p0: usize, p1: usize) !void {
        if (p1 <= p0) return;
        const tab = try d.gpa.alloc(f32, (p1 - p0) * head_dim);
        defer d.gpa.free(tab);
        var inv: [head_dim / 2]f32 = undefined;
        for (&inv, 0..) |*x, j| x.* = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * j)) / @as(f32, head_dim));
        const n = p1 - p0;
        const threads: usize = if (n >= 8192) 8 else 1;
        var ths: [8]?std.Thread = @splat(null);
        const per = (n + threads - 1) / threads;
        for (1..threads) |t| {
            const lo = p0 + t * per;
            if (lo >= p1) break;
            ths[t] = std.Thread.spawn(.{}, ropeSpan, .{ tab, p0, lo, @min(p1, lo + per), &inv }) catch null;
            if (ths[t] == null) ropeSpan(tab, p0, lo, @min(p1, lo + per), &inv); // a thread that would not start: here
        }
        ropeSpan(tab, p0, p0, @min(p1, p0 + per), &inv);
        for (ths) |t| if (t) |th| th.join();
        try d.cs.upload(p0 * head_dim * 4, std.mem.sliceAsBytes(tab));
    }

    /// Rope rows [lo, hi) into `tab` (whose row 0 is position `base`).
    fn ropeSpan(tab: []f32, base: usize, lo: usize, hi: usize, inv: *const [head_dim / 2]f32) void {
        for (lo..hi) |p| for (0..head_dim / 2) |j| {
            const fr: f32 = @as(f32, @floatFromInt(p)) * inv[j];
            tab[(p - base) * head_dim + j] = @cos(fr);
            tab[(p - base) * head_dim + head_dim / 2 + j] = @sin(fr);
        };
    }

    fn bf16(bits: u16) f32 {
        return @bitCast(@as(u32, bits) << 16);
    }

    fn go(f: Function, s: cuda.Stream, grid: usize, threads: usize, shared: u32, a: *Args) !void {
        if (grid == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = @intCast(threads) }, .shared = shared }, s, a);
    }

    fn ci(v: usize) c_int {
        return @intCast(v);
    }

    /// A call's positions pos0 .. pos0 + n - 1 into d.pos (and, with `anchor`, the embedding's id into d.ids), by a
    /// kernel: a pageable host upload waits for the stream (measured 2.4 ms a call in the draft check's profile).
    fn positionsFrom(d: *Draft, pos0: usize, n: usize, anchor: ?i32) !void {
        if (n > 64) return error.TooManyRows;
        var a: Args = .{};
        a.add(d.pos.ptr);
        a.add(ci(pos0));
        a.add(ci(n));
        a.add(d.ids.ptr);
        a.add(@as(c_int, anchor orelse -1));
        try go(d.f.positions, d.m.s, 1, 64, 0, &a);
    }

    fn mixNorm(d: *Draft, taps_rows: u64, n: usize, row: usize, w: u64, out: u64) !void {
        var a: Args = .{};
        for ([_]u64{ taps_rows, d.alpha_dev.ptr + row * ntaps * 4, w, out }) |v| a.add(v);
        a.add(ci(H));
        a.add(ci(ntaps));
        a.add(eps);
        try go(d.f.mix_norm, d.m.s, n, 512, 0, &a);
    }

    fn normRope(d: *Draft, x: u64, w: u64, n: usize, nh: usize, mode: c_int) !void {
        var a: Args = .{};
        for ([_]u64{ x, w, d.cs.ptr, d.pos.ptr }) |v| a.add(v);
        a.add(ci(nh));
        a.add(eps);
        a.add(mode);
        try go(d.f.norm_rope, d.m.s, n * nh, 64, 0, &a);
    }

    fn rmsnorm(d: *Draft, x: u64, res: u64, w: u64, out: u64, n: usize, add: bool) !void {
        var a: Args = .{};
        for ([_]u64{ x, res, w, out }) |v| a.add(v);
        a.add(ci(H));
        a.add(eps);
        a.add(@as(c_int, @intFromBool(add)));
        try go(d.f.rmsnorm, d.m.s, n, 512, 0, &a);
    }

    /// The block's rows x [block][k] . w [H][k]^T (k: this rank's heads / MLP columns), then the fused add + rmsnorm
    /// with `nw` into `out` (residual d.res). One rank: the bf16 GEMM (one rounding) and dsp_rmsnorm. TP: fp32
    /// partials, gathered, summed in rank order and rounded once to bf16 inside dsp_sum_rmsnorm.
    fn projNorm(d: *Draft, x: u64, w: u64, k: usize, nw: u64, out: u64) !void {
        return d.projNormX(x, w, null, k, nw, out);
    }

    /// projNorm with the product on `xw` (an EXL3 weight, e-step) when given, else on the bf16 `w`.
    fn projNormX(d: *Draft, x: u64, w: u64, xw: ?draft_exl3.Weight, k: usize, nw: u64, out: u64) !void {
        const m = d.m;
        if (d.world == 1) {
            if (xw) |e| try d.dx.?.run(m.s, e, x, block, d.o.ptr, false) else try m.g.run(m.s, x, w, d.o.ptr, block, H, k, false);
            return d.rmsnorm(d.o.ptr, d.res.ptr, nw, out, block, true);
        }
        if (xw) |e| try d.dx.?.run(m.s, e, x, block, d.o32.ptr, true) else try m.g.run(m.s, x, w, d.o32.ptr, block, H, k, true);
        try m.group.?.allGather(m.s, d.o32.ptr, d.o_all.ptr, block * H * 4);
        var a: Args = .{};
        a.add(d.o_all.ptr);
        a.add(ci(d.world));
        a.add(@as(u64, block * H));
        for ([_]u64{ d.res.ptr, nw, out }) |v| a.add(v);
        a.add(ci(H));
        a.add(eps);
        a.add(@as(c_int, 1));
        try go(d.f.sum_rmsnorm, m.s, block, 512, H * 4, &a);
    }

    /// The target's tapped rows `n` [n][8][H] of positions pos0 .. pos0 + n - 1 into every layer's context
    /// (M:513-526), and the last one's f_rnn (M:528-530): each layer mixes with its fuse row, dflare_norm, then
    /// k_proj_ctx / v_proj_ctx, k_norm_ctx and RoPE on K.
    pub fn absorb(d: *Draft, taps_rows: u64, pos0: usize, n: usize) !void {
        // p4-small (TF_INKLING2_ABSORB_WIDE): a chunk wider than max_rows in one piece
        if (d.m.absorb_wide and n > d.max_rows) return d.absorbWide(taps_rows, pos0, n);
        // a prompt chunk wider than the draft's rows (Model.prompt_rows) in pieces of max_rows rows from its first:
        // the products and launches a chunking by max_rows makes, one piece at a time
        var at: usize = 0;
        while (at < n) : (at += d.max_rows) {
            const piece = @min(d.max_rows, n - at);
            try d.absorbRows(taps_rows + at * ntaps * H * 2, pos0 + at, piece);
        }
    }

    fn absorbRows(d: *Draft, taps_rows: u64, pos0: usize, n: usize) !void {
        if (n == 0) return;
        if (n > d.max_rows) return error.TooManyRows;
        const m = d.m;
        try d.positionsFrom(pos0, n, null);
        for (&d.layer, 0..) |*l, i| {
            try d.mixNorm(taps_rows, n, i, l.dflare_norm.ptr, d.mix.ptr);
            try m.g.run(m.s, d.mix.ptr, l.k_ctx.ptr, d.kctx.ptr, n, d.kvwl, H, false);
            try m.g.run(m.s, d.mix.ptr, l.v_ctx.ptr, d.vctx.ptr, n, d.kvwl, H, false);
            try d.normRope(d.kctx.ptr, l.k_norm_ctx.ptr, n, d.hkv, 1);
            var a: Args = .{};
            for ([_]u64{ d.kctx.ptr, d.vctx.ptr, l.kc.ptr, l.vc.ptr }) |v| a.add(v);
            a.add(ci(l.slots));
            a.add(ci(pos0));
            a.add(ci(d.kvwl));
            try go(d.f.ctx_put, m.s, n, 256, 0, &a);
        }
        try d.setFrnn(taps_rows + (n - 1) * ntaps * H * 2);
    }

    // ---- p4-small (TF_INKLING2_ABSORB_WIDE)

    /// absorbRows over a wide chunk's `n` rows in one piece: each layer's mix + dflare_norm, k_ctx / v_ctx products
    /// (one GEMM of the rows: cuBLASLt plans the width, so the sums' order is not the 16-row products'), k_norm_ctx +
    /// RoPE and the ring writes, as absorbRows does them. Only the rows a layer's context ring keeps are made: a
    /// ring of `slots` positions holds the chunk's last `slots` rows once it is absorbed (a draft reads positions
    /// below its anchor only, at most a window back on a sliding layer), so a sliding layer skips the rows before
    /// them; and one launch never writes two rows into one slot (their order would be undefined).
    fn absorbWide(d: *Draft, taps_rows: u64, pos0: usize, n: usize) !void {
        const m = d.m;
        if (n > d.wide_rows) try d.growWide(@max(n, m.prompt_rows));
        // the positions, dsp_positions' block (up to 1024 rows) a launch
        var at: usize = 0;
        while (at < n) : (at += 1024) {
            var a: Args = .{};
            a.add(d.wpos.ptr + at * 4);
            a.add(ci(pos0 + at));
            a.add(ci(@min(1024, n - at)));
            a.add(d.ids.ptr);
            a.add(@as(c_int, -1));
            try go(d.f.positions, m.s, 1, 1024, 0, &a);
        }
        for (&d.layer, 0..) |*l, i| {
            const r0 = n - @min(n, l.slots); // the first row the ring keeps
            const k = n - r0;
            const rows = taps_rows + r0 * ntaps * H * 2;
            try d.mixNorm(rows, k, i, l.dflare_norm.ptr, d.wmix.ptr);
            try m.g.run(m.s, d.wmix.ptr, l.k_ctx.ptr, d.wkctx.ptr, k, d.kvwl, H, false);
            try m.g.run(m.s, d.wmix.ptr, l.v_ctx.ptr, d.wvctx.ptr, k, d.kvwl, H, false);
            var a: Args = .{};
            for ([_]u64{ d.wkctx.ptr, l.k_norm_ctx.ptr, d.cs.ptr, d.wpos.ptr + r0 * 4 }) |v| a.add(v);
            a.add(ci(d.hkv));
            a.add(eps);
            a.add(@as(c_int, 1));
            try go(d.f.norm_rope, m.s, k * d.hkv, 64, 0, &a);
            var b: Args = .{};
            for ([_]u64{ d.wkctx.ptr, d.wvctx.ptr, l.kc.ptr, l.vc.ptr }) |v| b.add(v);
            b.add(ci(l.slots));
            b.add(ci(pos0 + r0));
            b.add(ci(d.kvwl));
            try go(d.f.ctx_put, m.s, k, 256, 0, &b);
        }
        try d.setFrnn(taps_rows + (n - 1) * ntaps * H * 2);
    }

    /// absorbWide's buffers for `rows` rows (made after the arena, like Model.growPrompt's: wideBytes).
    fn growWide(d: *Draft, rows: usize) !void {
        d.wmix = try d.alloc(rows * H * 2);
        d.wkctx = try d.alloc(rows * d.kvwl * 2);
        d.wvctx = try d.alloc(rows * d.kvwl * 2);
        d.wpos = try d.alloc(rows * 4);
        d.wide_rows = rows;
        std.log.info("inkling2 rank {d}: draft absorbs wide chunks in one piece ({d} rows, {d:.1} MB)", .{ d.m.rank, rows, @as(f64, @floatFromInt(wideBytes(rows, d.kvwl))) / 1e6 });
    }

    /// Device bytes absorbWide's buffers take for `rows` rows at `kvwl` context columns a row.
    pub fn wideBytes(rows: usize, kvwl: usize) usize {
        return rows * H * 2 + 2 * rows * kvwl * 2 + rows * 4;
    }
    // ---- end p4-small

    /// f_rnn (M:528-530) from one row's 8 taps: the last committed row before the anchor.
    pub fn setFrnn(d: *Draft, taps_row: u64) !void {
        try d.mixNorm(taps_row, 1, layers, d.markov_norm.ptr, d.frnn.ptr);
    }

    /// 7 tokens drafted from `anchor` at position `anchor_pos` (the context holds every position below it), into
    /// `out`. Every rank runs it (the embedding and the per-step picks are collectives).
    pub fn draft(d: *Draft, anchor: i32, anchor_pos: usize, out: *[spec]i32) !void {
        const m = d.m;
        const s = m.s;
        if (anchor_pos + block > d.positions) return error.PastContext;
        // ---- e-step: TF_INKLING2_DRAFT_GRAPH=1: the pass replayed (its three integers rewritten), or run eagerly and
        // captured after (a pass's first run makes its plans outside a capture); a capture failure turns it off
        if (d.dgraph_on) {
            if (d.dgraph != null and d.dgraph_gen != m.graphs.gen) {
                d.dgraph.?.deinit(d.gpa);
                d.dgraph = null;
            }
            if (d.dgraph) |e| {
                try draft_graph.replay(m.d, e, s, anchor, @intCast(anchor_pos), window);
            } else {
                try d.pass(anchor, anchor_pos);
                d.dgraph = d.capturePass(anchor, anchor_pos) catch |err| blk: {
                    std.log.warn("inkling2 rank {d}: the draft pass's graph capture failed ({s}); drafts run eagerly from here", .{ m.rank, @errorName(err) });
                    d.dgraph_on = false;
                    break :blk null;
                };
                d.dgraph_gen = m.graphs.gen;
            }
        } else try d.pass(anchor, anchor_pos);
        // ---- end e-step
        try d.toks.downloadAsync(0, std.mem.sliceAsBytes(out), s.handle);
        try s.synchronize();
    }

    // ---- e-step: the pass's capture (draft_graph.zig): 1 dsp_positions, a dsp_fdec a layer, a dsp_walk_step a step
    fn capturePass(d: *Draft, anchor: i32, anchor_pos: usize) !*draft_graph.Entry {
        const s = d.m.s;
        try s.synchronize();
        try cuda.graph.beginCapture(s, .thread_local);
        d.pass(anchor, anchor_pos) catch |err| {
            if (cuda.graph.endCapture(s)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var locals: [layers]u64 = undefined;
        var nl: usize = 0;
        for (&d.layer, 0..) |*l, i| if (local[i]) {
            locals[nl] = l.kc.ptr;
            nl += 1;
        };
        const wanted = [_]draft_graph.Wanted{
            .{ .f = d.f.positions, .args = &.{ .{ .index = 1, .rule = .pos }, .{ .index = 4, .rule = .anchor } } },
            .{ .f = d.f.fdec, .args = &.{ .{ .index = 7, .rule = .lo_local }, .{ .index = 8, .rule = .pos } }, .local_kc = locals[0..nl], .kc_index = 3 },
            .{ .f = d.f.walk_step, .args = &.{.{ .index = 3, .rule = .anchor }} },
        };
        return draft_graph.finish(d.gpa, d.m.d, s, &wanted, 1 + layers + spec, anchor, @intCast(anchor_pos), window);
    }

    /// The pass's launches from `anchor` at `anchor_pos` through the pick (d.toks); the caller reads them back.
    fn pass(d: *Draft, anchor: i32, anchor_pos: usize) !void {
        const m = d.m;
        const s = m.s;
        // ---- end e-step (the body below is draft()'s, unchanged)
        try d.positionsFrom(anchor_pos, block, anchor);
        // block row 0: the target's embedding (vocabulary-split, summed) and embed_norm; rows 1..7: mask_embed
        try m.k.embed(s, d.ids.ptr, m.embed, d.emb.ptr, 1, H, m.vocab_lo, m.vocab_lo + m.vocabLocal());
        try forward.reduce(m, d.emb.ptr, d.emb_all.ptr, d.emb.ptr, H, false);
        try m.k.rmsnorm(s, d.emb.ptr, false, m.embed_norm, d.x.ptr, 1, H, m.c.eps);
        {
            var a: Args = .{};
            for ([_]u64{ d.mask_embed.ptr, d.x.ptr }) |v| a.add(v);
            a.add(ci(H));
            a.add(ci(block));
            try go(d.f.mask_rows, s, block - 1, 256, 0, &a);
        }
        // M:312-314: the first layer's residual is its input; every later norm is the fused add (M:316, M:318, M:438)
        try d.rmsnorm(d.x.ptr, d.res.ptr, d.layer[0].in_norm.ptr, d.xn.ptr, block, false);
        for (&d.layer, 0..) |*l, i| {
            // e-step: the seven projections on their EXL3 weights (TF_INKLING2_DRAFT_EXL3) when on
            const xl: ?draft_exl3.LayerW = if (d.dx_on) d.dx.?.layer[i] else null;
            if (xl) |e| {
                try d.dx.?.run(s, e.q, d.xn.ptr, block, d.q.ptr, false);
                try d.dx.?.run(s, e.k, d.xn.ptr, block, d.kb.ptr, false);
                try d.dx.?.run(s, e.v, d.xn.ptr, block, d.vb.ptr, false);
            } else {
                try m.g.run(s, d.xn.ptr, l.q.ptr, d.q.ptr, block, d.qwl, H, false);
                try m.g.run(s, d.xn.ptr, l.k.ptr, d.kb.ptr, block, d.kvwl, H, false);
                try m.g.run(s, d.xn.ptr, l.v.ptr, d.vb.ptr, block, d.kvwl, H, false);
            }
            try d.normRope(d.q.ptr, l.q_norm.ptr, block, d.hq, 0);
            try d.normRope(d.kb.ptr, l.k_norm.ptr, block, d.hkv, 0);
            const lo: usize = if (local[i] and anchor_pos > window) anchor_pos - window else 0;
            if (m.draft_attn) {
                try d.attentionFdec(l, lo, anchor_pos, m.fd_splits[@intFromBool(local[i])]); // p4-dattn
            } else {
                var a: Args = .{};
                for ([_]u64{ d.q.ptr, d.kb.ptr, d.vb.ptr, l.kc.ptr, l.vc.ptr }) |v| a.add(v);
                for ([_]usize{ l.slots, lo, anchor_pos, block, d.hq, d.hkv }) |v| a.add(ci(v));
                a.add(d.scr.ptr);
                a.add(ci(d.max_keys));
                a.add(d.attn.ptr);
                try go(d.f.attention, s, block * d.hq, head_dim, 0, &a);
            }
            try d.projNormX(d.attn.ptr, l.o.ptr, if (xl) |e| e.o else null, d.qwl, l.post_norm.ptr, d.xn.ptr);
            if (xl) |e| {
                try d.dx.?.run(s, e.gate, d.xn.ptr, block, d.g.ptr, false);
                try d.dx.?.run(s, e.up, d.xn.ptr, block, d.u.ptr, false);
            } else {
                try m.g.run(s, d.xn.ptr, l.gate.ptr, d.g.ptr, block, d.interl, H, false);
                try m.g.run(s, d.xn.ptr, l.up.ptr, d.u.ptr, block, d.interl, H, false);
            }
            {
                var a: Args = .{};
                for ([_]u64{ d.g.ptr, d.u.ptr, d.act.ptr }) |v| a.add(v);
                a.add(ci(block * d.interl));
                try go(d.f.silu_mul, s, (block * d.interl + 255) / 256, 256, 0, &a);
            }
            const last = i + 1 == layers;
            try d.projNormX(d.act.ptr, l.down.ptr, if (xl) |e| e.down else null, d.interl, if (last) d.norm.ptr else d.layer[i + 1].in_norm.ptr, if (last) d.hidden.ptr else d.xn.ptr);
        }
        // base logits of rows 1..7 on this rank's vocabulary rows; the Markov context of each row
        const rows = d.hidden.ptr + H * 2;
        if (d.head_x) |h| { // e-step: TF_INKLING2_DRAFT_HEAD=exl3
            try m.dx.?.run(s, h, rows, spec, d.base.ptr, false);
        } else try m.g.run(s, rows, m.unembed, d.base.ptr, spec, d.vl, H, false);
        {
            var a: Args = .{};
            for ([_]u64{ rows, d.frnn.ptr, d.cat.ptr }) |v| a.add(v);
            a.add(ci(H));
            try go(d.f.cat_rnn, s, spec, 256, 0, &a);
        }
        try m.g.run(s, d.cat.ptr, d.w_ctx.ptr, d.ctxmm.ptr, spec, 3 * rank, 2 * H, true);
        try d.state.fill8(0, s.handle);
        // the walk: a step's recurrence (with the previous step's global pick), its fold + softcap + argmax on this
        // rank's rows, and every rank's (value, id) gathered for the next step's pick
        const gathered = if (m.world > 1) d.best_all.ptr else d.best.ptr;
        const fold_grid = @max(1, @min(fold_blocks, (d.valid + 7) / 8));
        for (0..spec) |i| {
            {
                var a: Args = .{};
                a.add(gathered);
                a.add(ci(m.world));
                a.add(ci(d.vl));
                a.add(@as(c_int, anchor));
                a.add(ci(i));
                a.add(d.toks.ptr);
                a.add(d.state.ptr + (i % 2) * rank * 2);
                a.add(d.state.ptr + ((i + 1) % 2) * rank * 2);
                for ([_]u64{ d.w1.ptr, d.w_step.ptr, d.ctxmm.ptr + i * 3 * rank * 4, d.rnn_bias.ptr, d.latent.ptr }) |v| a.add(v);
                a.add(ci(rank));
                try go(d.f.walk_step, s, rank / 8, 256, 0, &a);
            }
            {
                var a: Args = .{};
                for ([_]u64{ d.latent.ptr, d.fold_w.ptr, d.fold_b.ptr, d.base.ptr + i * d.vl * 2 }) |v| a.add(v);
                a.add(ci(d.valid));
                a.add(softcap);
                for ([_]u64{ d.part.ptr, d.counter.ptr, d.best.ptr }) |v| a.add(v);
                try go(d.f.walk_fold, s, fold_grid, 256, 0, &a);
            }
            if (m.world > 1) try m.group.?.allGather(s, d.best.ptr, d.best_all.ptr, 8);
        }
        {
            var a: Args = .{};
            a.add(gathered);
            a.add(ci(m.world));
            a.add(ci(d.vl));
            for ([_]u64{ d.prev.ptr, d.toks.ptr }) |v| a.add(v);
            a.add(ci(spec - 1));
            try go(d.f.pick, s, 1, 1, 0, &a);
        }
    }

    // ---- p4-cache: the draft's context at a position, saved and restored (serve/reuse.zig, TF_INKLING2_PROMPT_REUSE)
    // A draft from an anchor at or past `at` reads each sliding layer's positions anchor - window .. anchor - 1 and
    // f_rnn (set from the last absorbed row). A sliding layer's ring of window + 64 slots holds a position only 64
    // positions on, so a resume further back needs those rows saved; the global layer's context is positional (its
    // positions below `at` hold until a pass absorbs below `at`). Plain copies on the model's stream.

    /// Bytes of a saved context: [sliding layers][k, v][window][this rank's kv width] bf16, then f_rnn [H] bf16.
    pub fn snapBytes(d: *const Draft) usize {
        var n: usize = 0;
        for (local) |lc| n += @intFromBool(lc);
        return n * 2 * window * d.kvwl * 2 + H * 2;
    }

    /// The context at position `at` (every position below it absorbed) into `buf` (snapBytes).
    pub fn saveContext(d: *Draft, at: usize, buf: Buf) !void {
        try d.contextCopy(buf, at, true);
    }

    /// A context saved at `at` by saveContext back in place.
    pub fn restoreContext(d: *Draft, at: usize, buf: Buf) !void {
        try d.contextCopy(buf, at, false);
    }

    fn contextCopy(d: *Draft, buf: Buf, at: usize, save: bool) !void {
        const s = d.m.s.handle;
        const row = d.kvwl * 2;
        const n = @min(at, window);
        const each = window * row;
        var j: usize = 0;
        for (&d.layer, 0..) |*l, i| if (local[i]) {
            const s0 = (at - n) % l.slots;
            const first = @min(n, l.slots - s0);
            for ([_]Buf{ l.kc, l.vc }, 0..) |ring, kv| {
                const base = (2 * j + kv) * each;
                if (save) {
                    try buf.copyFrom(base, ring.ptr + s0 * row, first * row, s);
                    try buf.copyFrom(base + first * row, ring.ptr, (n - first) * row, s);
                } else {
                    try ring.copyFrom(s0 * row, buf.ptr + base, first * row, s);
                    try ring.copyFrom(0, buf.ptr + base + first * row, (n - first) * row, s);
                }
            }
            j += 1;
        };
        const fo = 2 * j * each;
        if (save) try buf.copyFrom(fo, d.frnn.ptr, H * 2, s) else try d.frnn.copyFrom(0, buf.ptr + fo, H * 2, s);
    }
    // ---- end p4-cache

    // ---- p4-dattn: the draft's attention as split-K flash decoding (Model.draft_attn, TF_INKLING2_DRAFT_ATTN)

    /// The block's attention on layer `l` (its context [lo, anchor_pos), then the 8 block rows) into d.attn:
    /// dsp_fdec over (`splits` key splits, kv heads), each block the 8 rows x the kv group's query heads (each key and
    /// value read once), then dsp_fdec_merge a block a (row, head), the splits in order. Not dsp_attention's bits
    /// (inkling2_dspark.cu's end says where they differ).
    fn attentionFdec(d: *Draft, l: *const Layer, lo: usize, anchor_pos: usize, splits: usize) !void {
        const s = d.m.s;
        if (splits == 0 or splits > kern.Kernels.fdec_max_splits) return error.FdecShape;
        var a: Args = .{};
        for ([_]u64{ d.q.ptr, d.kb.ptr, d.vb.ptr, l.kc.ptr, l.vc.ptr, d.fd_part.ptr }) |v| a.add(v);
        for ([_]usize{ l.slots, lo, anchor_pos, block, d.hq, d.hkv }) |v| a.add(ci(v));
        const warps = (block * (d.hq / d.hkv) + 15) / 16;
        try cuda.launch.launch(d.f.fdec, .{ .grid = .{ .x = @intCast(splits), .y = @intCast(d.hkv) }, .block = .{ .x = @intCast(32 * warps) }, .shared = kern.Kernels.fdecSmem(head_dim) }, s, &a);
        var b: Args = .{};
        for ([_]u64{ d.fd_part.ptr, d.attn.ptr }) |v| b.add(v);
        b.add(ci(splits));
        try go(d.f.fdec_merge, s, block * d.hq, head_dim, 0, &b);
    }
    // ---- end p4-dattn
};
