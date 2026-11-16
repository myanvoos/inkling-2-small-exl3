//! e-dense (prefill lane phase 5): Inkling-2's non-expert linears as EXL3 trellises from a dense pack
//! (the dense pack: ExLlamaV3's `{name}.trellis` int16 [K/16][N/16][16 bits], `.suh` fp16
//! [K], `.svh` fp16 [N] and the `.mul1` codebook marker, made by tools/quantize_dense.py with the fork's
//! quantize_exl3), run through core's CUDA EXL3 projection (cuda.exl3.Dense: the input rotation to fp16, then the
//! decode-row linear with fp32 sums, 1-128 rows a call; docs/recipes/exl3.md); wider prompt chunks: Wide.
//!
//! Classes (TF_INKLING2_DENSE_EXL3=attn,router,head | all; unset or 0: off, the EXL3 lane's forward untouched):
//!   attn    wq_du, wk_dv, wv_dv, wr_du, gate_attn (head-split: a rank's 128-column blocks, svh with them, suh
//!           whole) and wo_ud (K-split: a rank's 128-row blocks of K, suh with them, svh whole; the ranks' fp32
//!           partials summed as today's o_proj partials);
//!   router  mlp.gate, whole on every rank (one plan from the shape alone, so the routing agrees bit for bit);
//!   head    unembed rows [0, N) with N = 2 x 782 128-column blocks = 200192 (the 200058 real tokens and padding):
//!           a TP2 rank's vocabulary half is 100096 columns, whole blocks (the bf16 head's halves, 100512, are not).
//!           The DSpark draft keeps the bf16 head (its Markov fold and base logits read it).
//! The bf16 copies of the attention and router tensors a set replaces are not loaded when the switch is on
//! (TF_INKLING2_DENSE_EXL3_KEEP_BF16=1 keeps them); the head's always is (the draft). Several sets may be loaded at
//! once (the run tool's --dense-sets, its A/B in one process); one or none is active.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const page_cache = @import("cache.zig"); // e-load (x5): the pack out of the page cache once on the device
const gemm = @import("gemm.zig");

pub const Weight = cuda.exl3.Weight;

/// How a set runs rows past 128 (wide prompt chunks): `gemm` (the default) decodes each weight once a chunk into
/// fp16 (core's candidate unpack), multiplies the rotated rows on cuBLASLt (fp16 operands, fp32 sums) and finishes
/// them as the decode linear does (ink2_dx_finish: H / sqrt(128), svh, one rounding); `tiles` core's candidate prompt
/// kernels (256-row chunks, 64 rows a block); `blocks` 128-row blocks of the decode linear (a row's decode bits; each
/// block decodes the whole weight again). TF_INKLING2_DENSE_EXL3_WIDE for the switch's set, a set suffix in the run
/// tool.
pub const Wide = enum { gemm, tiles, blocks };

/// The least rows the gemm form takes (narrower prompt chunks: 128-row blocks, at most two).
pub const wide_min_rows = 256;

pub const Class = enum { attn, router, head };
pub const Classes = std.EnumSet(Class);

/// `attn,router,head`, `attn+head`, `all`; empty, `0`, `off` or `none`: none.
pub fn parseClasses(text: []const u8) !Classes {
    var c = Classes.empty;
    if (text.len == 0 or std.mem.eql(u8, text, "0") or std.mem.eql(u8, text, "off") or std.mem.eql(u8, text, "none")) return c;
    if (std.mem.eql(u8, text, "1") or std.mem.eql(u8, text, "all")) return Classes.full;
    var it = std.mem.tokenizeAny(u8, text, ",+");
    while (it.next()) |t| c.insert(std.meta.stringToEnum(Class, t) orelse return error.BadDenseExl3Class);
    return c;
}

/// The switch's classes (TF_INKLING2_DENSE_EXL3), none when unset.
pub fn envClasses() !Classes {
    const v = std.c.getenv("TF_INKLING2_DENSE_EXL3") orelse return Classes.empty;
    return parseClasses(std.mem.span(v));
}

/// The user, 2026-11-15 ("Turn everything on by default"): with TF_INKLING2_DENSE_EXL3 unset (and no
/// TF_INKLING2_LEGACY=1), the classes `ck` carries as EXL3 trellises (layer 0's tensors stand for every layer's; e-dense's
/// make_pack.py view: all three), else none.
pub fn packClasses(ck: *const core.checkpoint.Checkpoint) Classes {
    var c = Classes.empty;
    if (ck.has("model.llm.layers.0.attn.wq_du.trellis")) c.insert(.attn);
    if (ck.has("model.llm.layers.0.mlp.gate.trellis")) c.insert(.router);
    if (ck.has("model.llm.unembed.trellis")) c.insert(.head);
    return c;
}

/// A layer's EXL3 weights in one set (null: the bf16 weight).
pub const LayerW = struct {
    wq: ?Weight = null,
    wk: ?Weight = null,
    wv: ?Weight = null,
    wr: ?Weight = null,
    wgate: ?Weight = null,
    wo: ?Weight = null,
    router: ?Weight = null,
};

pub const Set = struct {
    name: []const u8,
    classes: Classes,
    layers: []LayerW,
    head: ?Weight = null,
    /// The head's columns over every rank (the pack's N); a rank holds head_n / world (both halves at canon).
    head_n: usize = 0,
    /// Rows past 128 (wide prompt chunks): see Wide.
    wide: Wide = .gemm,
    /// Decode / verify rows (1-128) on core's candidate kernels (exl3_candidate rows: 16 rows a window on the tensor
    /// units) instead of the decode linear: the run tool's set suffix `:dtiles` (an A/B; not the decode linear's bits).
    decode_tiles: bool = false,
};

/// How a tensor is cut for a rank: by 128-column blocks of N (head-split), by 128-row blocks of K (o_proj), or whole.
const Cut = enum { cols, rows, none };

/// The attention tensors (checkpoint names after `model.llm.layers.N.attn.`) and their cuts.
const attn_names = [_]struct { []const u8, Cut }{
    .{ "wq_du", .cols }, .{ "wk_dv", .cols }, .{ "wv_dv", .cols }, .{ "wr_du", .cols }, .{ "gate_attn", .cols }, .{ "wo_ud", .rows },
};

pub const DenseExl3 = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    /// The projections' kernels and their one-stream scratch (reserve()d per set before any graph capture).
    op: cuda.exl3.Dense,
    /// The candidate implementation for a set's wide chunks (Wide.tiles), loaded on first use.
    op_wide: ?cuda.exl3.Dense = null,
    /// The gemm form (Wide.gemm), loaded on first use: core's candidate module (tf_exl3_unpack), the family's
    /// inkling2_exl3 module (ink2_dx_finish), the fp16 GEMMs' plans by shape on the model's cuBLASLt handle, the
    /// scratch.
    g: *gemm.Gemms,
    wide_mods: ?[2]cuda.Module = null,
    f_unpack: cuda.Function = undefined,
    f_finish: cuda.Function = undefined,
    wide_plans: std.AutoHashMapUnmanaged([3]u64, cuda.cublaslt.Gemm) = .empty,
    w_xh: ?cuda.DeviceBuffer = null, // the rotated rows, fp16 [rows][k]
    w_q: ?cuda.DeviceBuffer = null, // W_q, fp16 [k][n]
    w_yr: ?cuda.DeviceBuffer = null, // their products, fp32 [rows][n]
    sets: std.ArrayList(Set) = .empty,
    cur: ?usize = null,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Weights already uploaded, by tensor prefix, width and a hash of its suh (the same quantization from any pack:
    /// a recipe pack's symlinks share the q6 / q8 uploads), so sets share them.
    cache: std.StringHashMapUnmanaged(Weight) = .empty,
    names: std.heap.ArenaAllocator,
    /// This rank's device bytes of every upload.
    bytes: u64 = 0,
    rank: usize,
    world: usize,
    /// Canonical TP (Model.canon, world 1): every tensor whole; the forward takes a TP=2 rank's share as views.
    canon: bool,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: *gemm.Gemms, rank: usize, world: usize, canon: bool) !*DenseExl3 {
        const dx = try gpa.create(DenseExl3);
        errdefer gpa.destroy(dx);
        dx.* = .{ .gpa = gpa, .d = d, .op = try cuda.exl3.Dense.load(d), .g = g, .names = std.heap.ArenaAllocator.init(gpa), .rank = rank, .world = world, .canon = canon };
        return dx;
    }

    pub fn deinit(dx: *DenseExl3) void {
        dx.op.unload();
        if (dx.op_wide) |*w| w.unload();
        var pit = dx.wide_plans.valueIterator();
        while (pit.next()) |v| v.deinit();
        dx.wide_plans.deinit(dx.gpa);
        if (dx.wide_mods) |*mods| for (mods) |*m| m.unload();
        for ([_]*?cuda.DeviceBuffer{ &dx.w_xh, &dx.w_q, &dx.w_yr }) |slot| if (slot.*) |*b| b.free();
        for (dx.bufs.items) |*b| b.free();
        dx.bufs.deinit(dx.gpa);
        for (dx.sets.items) |s| dx.gpa.free(s.layers);
        dx.sets.deinit(dx.gpa);
        dx.cache.deinit(dx.gpa);
        dx.names.deinit();
        dx.gpa.destroy(dx);
    }

    pub fn active(dx: *const DenseExl3) ?*const Set {
        const i = dx.cur orelse return null;
        return &dx.sets.items[i];
    }

    /// Set `idx` active (null: none, the bf16 weights): its weights' scratch reserved on `s`, so a graph capture of a
    /// decode / verify forward allocates nothing. The caller drops captured graphs (they hold the old weights).
    pub fn activate(dx: *DenseExl3, s: cuda.Stream, idx: ?usize) !void {
        dx.cur = idx;
        const set = dx.active() orelse return;
        for (set.layers) |l| for ([_]?Weight{ l.wq, l.wk, l.wv, l.wr, l.wgate, l.wo, l.router }) |w| if (w) |x| try dx.reserve(s, x);
        if (set.head) |h| try dx.reserve(s, h);
    }

    /// A weight's scratch, and at canon its TP=2 halves' (the shapes the forward runs).
    fn reserve(dx: *DenseExl3, s: cuda.Stream, w: Weight) !void {
        try dx.op.reserve(s, w);
        if (!dx.canon) return;
        for (0..2) |r| {
            if (w.n % 256 == 0) try dx.op.reserve(s, try half(w, .cols, r));
            if (w.k % 256 == 0) try dx.op.reserve(s, try half(w, .rows, r));
        }
    }

    /// y [rows][n] = x [rows][k] through `w` (x bf16; y bf16 or fp32): decode rows in one call (1-128); wider chunks
    /// as the active set's Wide says (a canonical half's column view keeps its parent's strides, which the candidate's
    /// unpack and prompt kernels do not take: blocks).
    pub fn run(dx: *DenseExl3, s: cuda.Stream, w: Weight, x: u64, rows: usize, y: u64, y_f32: bool) !void {
        const yd: cuda.exl3.DType = if (y_f32) .f32 else .bf16;
        const stored = w.halves % 2 == 0 and w.stride_k == cuda.exl3.strides(w.k, w.n, w.halves, false).k;
        if (rows <= 128) {
            if (dx.active()) |set| if (set.decode_tiles and stored) return (try dx.tilesOp()).decode(s, w, x, .bf16, rows, y, yd);
            return dx.op.decode(s, w, x, .bf16, rows, y, yd);
        }
        const mode: Wide = if (dx.active()) |set| set.wide else .blocks;
        switch (mode) {
            .gemm => if (rows >= wide_min_rows and stored) return dx.wideGemm(s, w, x, rows, y, y_f32),
            .tiles => if (stored) return (try dx.tilesOp()).prompt(s, w, x, .bf16, rows, y, yd),
            .blocks => {},
        }
        return dx.op.prompt(s, w, x, .bf16, rows, y, yd);
    }

    /// The candidate implementation (core's exl3_candidate kernels), loaded on first use.
    fn tilesOp(dx: *DenseExl3) !*cuda.exl3.Dense {
        if (dx.op_wide == null) {
            dx.op_wide = try cuda.exl3.Dense.load(dx.d);
            dx.op_wide.?.implementation = .tiles;
        }
        return &dx.op_wide.?;
    }

    /// Wide.gemm: xh = fp16(((x * suh) H) / sqrt(128)) (core's tf_exl3_rot_in, every row in one launch), W_q [k][n]
    /// fp16 (core's tf_exl3_unpack: the trellis decoded once), yr = xh W_q on cuBLASLt (fp32 sums), y = finish(yr)
    /// (ink2_dx_finish). The decode linear's arithmetic up to the order of the k sums.
    fn wideGemm(dx: *DenseExl3, s: cuda.Stream, w: Weight, x: u64, rows: usize, y: u64, y_f32: bool) !void {
        if (dx.wide_mods == null) {
            var cand = try cuda.Module.load(dx.d, cuda.kernels.exl3_candidate);
            errdefer cand.unload();
            var fam = try cuda.Module.load(dx.d, cuda.kernels.inkling2_exl3);
            errdefer fam.unload();
            dx.f_unpack = try cand.function("tf_exl3_unpack");
            dx.f_finish = try fam.function("ink2_dx_finish");
            dx.wide_mods = .{ cand, fam };
        }
        const k: usize = w.k;
        const n: usize = w.n;
        const xh = try dx.scratch(s, &dx.w_xh, rows * k * 2);
        const wq = try dx.scratch(s, &dx.w_q, k * n * 2);
        const yr = try dx.scratch(s, &dx.w_yr, rows * n * 4);
        var r: cuda.Args = .{};
        r.add(x);
        r.add(w.suh);
        r.add(xh);
        r.add(@as(i32, @backingInt(cuda.exl3.DType.bf16)));
        r.add(@as(i32, @intCast(rows)));
        r.add(@as(i32, @intCast(k)));
        try cuda.launch.launch(dx.op.lin.rot_in, .{ .grid = .{ .x = @intCast((k / 128 + 3) / 4), .y = @intCast(rows), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &r);
        var u: cuda.Args = .{};
        u.add(w.trellis);
        u.add(cuda.exl3_candidate.MatrixArgs{ .k = @intCast(k), .n = @intCast(n), .bits = w.halves / 2, .tiles_n = @intCast(n / 16), .split = @intCast(cuda.exl3_candidate.split(k, n)) });
        u.add(@as(u32, @backingInt(w.codebook)));
        u.add(wq);
        try cuda.launch.launch(dx.f_unpack, .{ .grid = .{ .x = @intCast(n / 16), .y = @intCast(k / 16), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &u);
        const g = dx.g;
        const key = [3]u64{ n, rows, k };
        const got = try dx.wide_plans.getOrPut(dx.gpa, key);
        if (!got.found_existing) {
            // column-major: yr^T [n][rows] = W_q^T [n][k] . xh^T [k][rows] (row-major [r][c] is column-major [c][r])
            got.value_ptr.* = cuda.cublaslt.Gemm.init(&g.lt, g.handle, .{ .m = n, .n = rows, .k = k, .a = .f16, .b = .f16, .c = .f32, .lda = n, .ldb = k, .ldc = n }, g.ws.len) catch |err| {
                _ = dx.wide_plans.remove(key);
                return err;
            };
        }
        try got.value_ptr.run(1, wq, xh, 0, yr, g.ws.ptr, g.ws.len, s.handle);
        var f: cuda.Args = .{};
        f.add(yr);
        f.add(w.svh);
        f.add(y);
        f.add(@as(i32, @intCast(n)));
        f.add(@as(i32, @intFromBool(y_f32)));
        try cuda.launch.launch(dx.f_finish, .{ .grid = .{ .x = @intCast((n / 128 + 3) / 4), .y = @intCast(rows), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &f);
    }

    /// A scratch buffer of at least `bytes` (the old one freed once the stream is idle). Only prompt chunks use them,
    /// never a graph capture.
    fn scratch(dx: *DenseExl3, s: cuda.Stream, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.*) |b| if (b.len >= bytes) return b.ptr;
        try s.synchronize();
        if (slot.*) |*b| b.free();
        slot.* = null;
        slot.* = try cuda.DeviceBuffer.alloc(dx.d, bytes);
        return slot.*.?.ptr;
    }

    /// Loads set `name` from the pack in `dir` (its classes' tensors of every layer, cut for this rank) and returns its
    /// index. The pack is any folder core.checkpoint opens (a model.safetensors.index.json, or model*.safetensors).
    pub fn addSet(dx: *DenseExl3, io: std.Io, name: []const u8, classes: Classes, dir: []const u8, layers: usize, wide: Wide, decode_tiles: bool) !usize {
        var ck = try core.checkpoint.Checkpoint.openModel(dx.gpa, io, dir);
        var ck_drop = page_cache.Files.of(dx.gpa, &ck); // e-load (x5): its files out of the page cache after ck.close()
        defer ck_drop.drop();
        defer ck.close();
        const set_layers = try dx.gpa.alloc(LayerW, layers);
        errdefer dx.gpa.free(set_layers);
        @memset(set_layers, .{});
        var set: Set = .{ .name = try dx.names.allocator().dupe(u8, name), .classes = classes, .layers = set_layers, .wide = wide, .decode_tiles = decode_tiles };
        var nb: [160]u8 = undefined;
        const before = dx.bytes;
        for (0..layers) |li| {
            const l = &set_layers[li];
            if (classes.contains(.attn)) {
                const slots = [_]*?Weight{ &l.wq, &l.wk, &l.wv, &l.wr, &l.wgate, &l.wo };
                for (attn_names, slots) |an, slot| {
                    const prefix = try std.fmt.bufPrint(&nb, "model.llm.layers.{d}.attn.{s}", .{ li, an[0] });
                    slot.* = try dx.weight(&ck, dir, prefix, an[1]);
                }
            }
            if (classes.contains(.router)) {
                l.router = try dx.weight(&ck, dir, try std.fmt.bufPrint(&nb, "model.llm.layers.{d}.mlp.gate", .{li}), .none);
            }
        }
        if (classes.contains(.head)) {
            const h = try dx.weight(&ck, dir, "model.llm.unembed", .cols);
            set.head = h;
            set.head_n = if (dx.canon) h.n else h.n * dx.world;
        }
        try dx.sets.append(dx.gpa, set);
        var cb: [32]u8 = undefined;
        std.log.info("inkling2 rank {d}: dense EXL3 set '{s}' ({s}) from {s}: {d:.3} GiB of new trellis on the device", .{ dx.rank, name, classesText(classes, &cb), dir, @as(f64, @floatFromInt(dx.bytes - before)) / (1024.0 * 1024.0 * 1024.0) });
        return dx.sets.items.len - 1;
    }

    /// One tensor group's rank share on the device (cached by pack and prefix).
    fn weight(dx: *DenseExl3, ck: *core.checkpoint.Checkpoint, dir: []const u8, prefix: []const u8, cut: Cut) !Weight {
        _ = dir;
        var nb: [192]u8 = undefined;
        const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix}));
        if (t.dtype != .i16 or t.rank != 3 or t.shape[2] % 8 != 0) return error.BadDenseExl3Trellis;
        const K = t.shape[0] * 16;
        const N = t.shape[1] * 16;
        const halves: u32 = @intCast(t.shape[2] / 8); // 16 bits int16 words a tile = 8 x (2 bits)
        const suh = try ck.expect(try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), .f16, &.{K});
        const key = try std.fmt.allocPrint(dx.names.allocator(), "{s}|{d}|{x}", .{ prefix, halves, std.hash.Wyhash.hash(0, suh.bytes) });
        if (dx.cache.get(key)) |w| return w;
        const svh = try ck.expect(try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), .f16, &.{N});
        const cb: cuda.exl3.Codebook = if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mul1", .{prefix}))) .mul1 else if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mcg", .{prefix}))) .mcg else .inst3;
        if (K % 128 != 0 or N % 128 != 0) return error.BadDenseExl3Shape;
        // this rank's [lo, hi) of the cut dimension, whole 128-blocks (canon: whole, the forward takes views)
        const ways: usize = if (dx.canon or cut == .none) 1 else dx.world;
        const r: usize = if (ways == 1) 0 else dx.rank;
        const dim = if (cut == .rows) K else N;
        if (dim % (128 * ways) != 0) {
            std.log.err("{s}: {d} is not {d} whole 128-blocks a rank", .{ prefix, dim, ways });
            return error.BadDenseExl3Split;
        }
        const lo = r * dim / ways;
        const hi = (r + 1) * dim / ways;
        const k = if (cut == .rows) hi - lo else K;
        const n = if (cut == .rows or cut == .none) N else hi - lo;
        const tile: usize = t.shape[2] * 2; // bytes a 16x16 tile
        const host = try dx.gpa.alloc(u8, (k / 16) * (n / 16) * tile);
        defer dx.gpa.free(host);
        if (cut == .cols and ways > 1) {
            const row = (n / 16) * tile;
            for (0..K / 16) |kr| @memcpy(host[kr * row ..][0..row], t.bytes[(kr * (N / 16) + lo / 16) * tile ..][0..row]);
        } else if (cut == .rows and ways > 1) {
            @memcpy(host, t.bytes[(lo / 16) * (N / 16) * tile ..][0..host.len]);
        } else @memcpy(host, t.bytes[0..host.len]);
        const tr = try dx.keep(host);
        const su = try dx.keep(if (cut == .rows) suh.bytes[lo * 2 .. hi * 2] else suh.bytes);
        const sv = try dx.keep(if (cut == .cols) svh.bytes[lo * 2 .. hi * 2] else svh.bytes);
        const st = cuda.exl3.strides(k, n, halves, false);
        const w: Weight = .{ .trellis = tr, .suh = su, .svh = sv, .k = @intCast(k), .n = @intCast(n), .halves = halves, .codebook = cb, .stride_k = st.k, .stride_nb = st.nb };
        if (cuda.exl3.pick(w, 1, false, false, 1) == null) return error.UnsupportedExl3Width;
        try dx.cache.put(dx.gpa, key, w);
        return w;
    }

    fn keep(dx: *DenseExl3, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(dx.d, bytes);
        errdefer b.free();
        try dx.bufs.append(dx.gpa, b);
        dx.bytes += bytes.len;
        return b.ptr;
    }
};

/// Half `r` of a whole (canon) weight as a TP=2 rank holds it: columns [r n/2, (r+1) n/2) (a view, core's
/// Weight.columns), or the 128-row blocks [r k/2, (r+1) k/2) of K (the trellis' tile rows and suh from there).
pub fn half(w: Weight, cut: Cut, r: usize) !Weight {
    if (cut == .cols) return w.columns(@intCast(r * w.n / 2), w.n / 2);
    if (cut != .rows or w.k % 256 != 0) return error.Exl3Shape;
    var v = w;
    v.k = w.k / 2;
    v.sk = 0;
    v.wk = 0;
    v.trellis = w.trellis + @as(u64, r) * (w.k / 2 / 16) * w.stride_k * 4;
    v.suh = w.suh + @as(u64, r) * (w.k / 2) * 2;
    return v;
}

pub const HalfCut = Cut;

/// `attn+router+head` style text of a class set (`none` when empty) in `buf`.
pub fn classesText(c: Classes, buf: *[32]u8) []const u8 {
    var n: usize = 0;
    var it = c.iterator();
    while (it.next()) |k| {
        if (n > 0) {
            buf[n] = '+';
            n += 1;
        }
        const t = @tagName(k);
        @memcpy(buf[n..][0..t.len], t);
        n += t.len;
    }
    if (n == 0) return "none";
    return buf[0..n];
}

test "inkling2 dense exl3: classes parse, the off forms are empty" {
    try std.testing.expect((try parseClasses("")).count() == 0);
    try std.testing.expect((try parseClasses("0")).count() == 0);
    try std.testing.expect((try parseClasses("all")).count() == 3);
    const c = try parseClasses("attn,head");
    try std.testing.expect(c.contains(.attn) and c.contains(.head) and !c.contains(.router));
    try std.testing.expect((try parseClasses("router+attn")).contains(.router));
    try std.testing.expectError(error.BadDenseExl3Class, parseClasses("lm"));
}

test "inkling2 dense exl3: a canon half is a TP=2 rank's cut (columns view; K-half tile rows and suh)" {
    const st = cuda.exl3.strides(6144, 4608, 12, false);
    const w: Weight = .{ .trellis = 0x100000, .suh = 0x2000, .svh = 0x3000, .k = 6144, .n = 4608, .halves = 12, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
    const h1 = try half(w, .rows, 1);
    try std.testing.expectEqual(@as(u32, 3072), h1.k);
    try std.testing.expectEqual(w.trellis + 192 * st.k * 4, h1.trellis);
    try std.testing.expectEqual(w.suh + 3072 * 2, h1.suh);
    try std.testing.expectEqual(w.svh, h1.svh);
    // a TP=2 rank's own cut of the same rows: the same strides (N is whole), so the same addressing from its base
    const cut = cuda.exl3.strides(3072, 4608, 12, false);
    try std.testing.expectEqual(cut.k, h1.stride_k);
    const c1 = try half(w, .cols, 1);
    try std.testing.expectEqual(@as(u32, 2304), c1.n);
    try std.testing.expectEqual(w.svh + 2304 * 2, c1.svh);
}
