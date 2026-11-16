//! A TP rank's share of Inkling-2 by the checkpoint's own tensor names (the reference's weights.plan): query heads,
//! kv heads, the vocabulary and each expert's intermediate channels are split; the rest is whole on every rank.
//! Every entry carries the dtype and shape the family's contract expects, so a loader refuses any other.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;

const DType = core.safetensors.DType;
pub const max_dims = 3;

/// Dense weights are loaded at open; expert tensors are read per (layer, expert) into the residency arena.
pub const Kind = enum { dense, experts };

pub const Entry = struct {
    name: []const u8,
    dtype: DType,
    dims: u8,
    shape: [max_dims]usize, // the whole tensor, as the checkpoint stores it
    split: ?u8 = null, // the dimension this rank takes a part of; null: whole
    lo: usize = 0, // this rank's [lo, hi) along `split`
    hi: usize = 0,
    kind: Kind = .dense,

    pub fn wholeShape(e: *const Entry) []const usize {
        return e.shape[0..e.dims];
    }

    /// The rank's shape: the whole shape with `split` cut to [lo, hi).
    pub fn rankShape(e: Entry) [max_dims]usize {
        var s = e.shape;
        if (e.split) |d| s[d] = e.hi - e.lo;
        return s;
    }

    pub fn rankBytes(e: Entry) usize {
        const s = e.rankShape();
        var n: usize = e.dtype.size();
        for (s[0..e.dims]) |d| n *= d;
        return n;
    }
};

/// The rank's [lo, hi) of `n` elements of a dimension, cut evenly in whole `unit`s (Config.checkSplit made it even).
fn span(n: usize, unit: usize, rank: u32, world: u32) [2]usize {
    const per = n / world;
    std.debug.assert(per % unit == 0);
    return .{ rank * per, (rank + 1) * per };
}

const Builder = struct {
    a: std.mem.Allocator,
    c: *const Config,
    rank: u32,
    world: u32,
    out: std.ArrayList(Entry) = .empty,

    fn add(b: *Builder, name: []const u8, dtype: DType, shape: []const usize, split: ?u8, unit: usize, kind: Kind) !void {
        var e: Entry = .{ .name = name, .dtype = dtype, .dims = @intCast(shape.len), .shape = @splat(1), .kind = kind };
        @memcpy(e.shape[0..shape.len], shape);
        if (split) |d| {
            const s = span(shape[d], unit, b.rank, b.world);
            e.split = d;
            e.lo = s[0];
            e.hi = s[1];
        }
        try b.out.append(b.a, e);
    }

    fn fmtName(b: *Builder, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.allocPrint(b.a, fmt, args);
    }

    /// One decoder layer's tensors, attention first, in the checkpoint's own names.
    fn layer(b: *Builder, i: u32) !void {
        const c = b.c;
        const H: usize = c.hidden;
        const hd: usize = c.head_dim;
        const K: usize = c.conv;
        const p = "model.llm.layers.";
        // attention: query heads split by head, kv heads by kv head, o_proj by the query heads' columns
        try b.add(try b.fmtName("{s}{d}.attn.wq_du.weight", .{ p, i }), .bf16, &.{ c.heads * hd, H }, 0, hd, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.gate_attn.weight", .{ p, i }), .bf16, &.{ c.heads * hd, H }, 0, hd, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.wr_du.weight", .{ p, i }), .bf16, &.{ c.heads * c.d_rel, H }, 0, c.d_rel, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.attn_sink_proj.weight", .{ p, i }), .bf16, &.{ c.heads, H }, 0, 1, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.q_l2_clip.scale", .{ p, i }), .bf16, &.{ c.heads, hd }, 0, 1, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.rel_l2_clip.scale", .{ p, i }), .bf16, &.{ c.heads, c.d_rel }, 0, 1, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.wk_dv.weight", .{ p, i }), .bf16, &.{ c.kv_heads * hd, H }, 0, hd, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.wv_dv.weight", .{ p, i }), .bf16, &.{ c.kv_heads * hd, H }, 0, hd, .dense);
        try b.add(try b.fmtName("{s}{d}.attn.wo_ud.weight", .{ p, i }), .bf16, &.{ H, c.heads * hd }, 1, hd, .dense);
        const extent: usize = if (c.isLocal(i)) c.window else c.rel_extent;
        try b.add(try b.fmtName("{s}{d}.attn.rel_logits_proj.proj", .{ p, i }), .bf16, &.{ c.d_rel, extent }, null, 1, .dense);
        if (c.isLocal(i)) {
            // sliding layers: static depthwise k / v convolutions, kv-head split
            for ([_][]const u8{ "k", "v" }) |s| {
                try b.add(try b.fmtName("{s}{d}.attn.{s}_sconv.weight", .{ p, i, s }), .bf16, &.{ c.kv_heads * hd, 1, K }, 0, hd, .dense);
            }
            try b.add(try b.fmtName("{s}{d}.attn_norm.weight", .{ p, i }), .bf16, &.{H}, null, 1, .dense);
        } else {
            // global layers: static taps [K, Hkv hd] by columns, input-generated taps [Hkv K, H] kv-head-major rows
            for ([_][]const u8{ "k", "v" }) |s| {
                try b.add(try b.fmtName("{s}{d}.attn.{s}_sconv.static_w", .{ p, i, s }), .bf16, &.{ K, c.kv_heads * hd }, 1, hd, .dense);
                try b.add(try b.fmtName("{s}{d}.attn.{s}_sconv.weight_proj", .{ p, i, s }), .bf16, &.{ c.kv_heads * K, H }, 0, K, .dense);
            }
            for ([_][]const u8{ "q", "k", "v" }) |s| {
                try b.add(try b.fmtName("{s}{d}.{s}_input_norm.weight", .{ p, i, s }), .bf16, &.{H}, null, 1, .dense);
            }
        }
        try b.add(try b.fmtName("{s}{d}.mlp_norm.weight", .{ p, i }), .bf16, &.{H}, null, 1, .dense);
        try b.add(try b.fmtName("{s}{d}.attn_sconv.weight", .{ p, i }), .bf16, &.{ H, 1, K }, null, 1, .dense);
        try b.add(try b.fmtName("{s}{d}.mlp_sconv.weight", .{ p, i }), .bf16, &.{ H, 1, K }, null, 1, .dense);
        // router: replicated (its fp32 logits and sigmoid scores are every rank's, so routing agrees bit for bit)
        try b.add(try b.fmtName("{s}{d}.mlp.gate.weight", .{ p, i }), .bf16, &.{ c.experts, H }, null, 1, .dense);
        try b.add(try b.fmtName("{s}{d}.mlp.gate.bias", .{ p, i }), .f32, &.{c.experts}, null, 1, .dense);
        try b.experts(i);
    }

    /// The routed experts of layer i: every expert on every rank, each rank its intermediate channels
    /// [rank I / world, ...). w13's stored rows interleave gate and up per channel, so a rank's rows are one range.
    fn experts(b: *Builder, i: u32) !void {
        const c = b.c;
        const E: usize = c.experts;
        const H: usize = c.hidden;
        const I: usize = c.inter;
        const p = "model.llm.layers.";
        switch (c.formats[i]) {
            .nvfp4 => {
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w13_weight", .{ p, i }), .u8, &.{ E, 2 * I, H / 2 }, 1, 64, .experts);
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w13_weight_scale", .{ p, i }), .f8_e4m3, &.{ E, 2 * I, H / 16 }, 1, 64, .experts);
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w2_weight", .{ p, i }), .u8, &.{ E, H, I / 2 }, 2, 16, .experts);
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w2_weight_scale", .{ p, i }), .f8_e4m3, &.{ E, H, I / 16 }, 2, 2, .experts);
                for ([_][]const u8{ "w13", "w2" }) |w| {
                    try b.add(try b.fmtName("{s}{d}.mlp.experts.{s}_weight_amax", .{ p, i, w }), .f32, &.{E}, null, 1, .experts);
                    try b.add(try b.fmtName("{s}{d}.mlp.experts.{s}_weight_outer_scale_denominator", .{ p, i, w }), .f32, &.{1}, null, 1, .experts);
                }
            },
            .mxfp8 => {
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w13_weight", .{ p, i }), .f8_e4m3, &.{ E, 2 * I, H }, 1, 64, .experts);
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w13_weight_scale", .{ p, i }), .u8, &.{ E, 2 * I, H / 32 }, 1, 64, .experts);
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w2_weight", .{ p, i }), .f8_e4m3, &.{ E, H, I }, 2, 32, .experts);
                try b.add(try b.fmtName("{s}{d}.mlp.experts.w2_weight_scale", .{ p, i }), .u8, &.{ E, H, I / 32 }, 2, 1, .experts);
            },
            // the EXL3 export's experts live in its own batch files, not in the view's index (exl3.zig loads them)
            .exl3 => {},
        }
    }
};

/// Every tensor rank `rank` of `world` keeps, in load order: the top-level tensors, then each layer's. The vision
/// tower (model.visual.*) and the MTP drafter (mtp.safetensors) are not in the plan: the loader skips them by name.
pub fn plan(a: std.mem.Allocator, c: *const Config, rank: u32, world: u32) ![]Entry {
    try c.checkSplit(world);
    if (rank >= world) return error.BadRank;
    var b: Builder = .{ .a = a, .c = c, .rank = rank, .world = world };
    const H: usize = c.hidden;
    const V: usize = c.vocab;
    try b.add("model.llm.embed.weight", .bf16, &.{ V, H }, 0, 1, .dense);
    try b.add("model.llm.embed_norm.weight", .bf16, &.{H}, null, 1, .dense);
    try b.add("model.llm.norm.weight", .bf16, &.{H}, null, 1, .dense);
    try b.add("model.llm.unembed.weight", .bf16, &.{ V, H }, 0, 1, .dense);
    try b.add("model.llm.attn_res.output_query", .f32, &.{H}, null, 1, .dense);
    for (0..c.layers) |l| {
        // attn_res: layer l's mlp_input query is mlp_queries.l; its attention_input query is attn_queries.(l - 1)
        // (layer 0 has none), [3, H] when layer l is global (one query each for q, k and v), else [H]
        try b.add(try b.fmtName("model.llm.attn_res.mlp_queries.{d}", .{l}), .f32, &.{H}, null, 1, .dense);
        if (l > 0) {
            const n = try b.fmtName("model.llm.attn_res.attn_queries.{d}", .{l - 1});
            if (c.isLocal(@intCast(l))) try b.add(n, .f32, &.{H}, null, 1, .dense) else try b.add(n, .f32, &.{ 3, H }, null, 1, .dense);
        }
    }
    for (0..c.layers) |l| try b.layer(@intCast(l));
    return b.out.toOwnedSlice(a);
}

/// Bytes a rank holds of each kind (the dense load and the experts' on-disk share).
pub fn bytesOf(entries: []const Entry, kind: Kind) u64 {
    var n: u64 = 0;
    for (entries) |e| if (e.kind == kind) {
        n += e.rankBytes();
    };
    return n;
}

test "inkling2 plan: release TP2 rank 0 takes half of every split axis at the reference's spans" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const c = Config{};
    const r0 = try plan(arena.allocator(), &c, 0, 2);
    const r1 = try plan(arena.allocator(), &c, 1, 2);
    try std.testing.expectEqual(r0.len, r1.len);
    // 4 + 1 + 48 mlp queries + 47 attn queries, then per layer: 10 attention + 2 or 7 conv / norms + 3 + 2 router
    var by: std.StringHashMapUnmanaged(Entry) = .empty;
    for (r1) |e| try by.put(arena.allocator(), e.name, e);
    const q = by.get("model.llm.layers.0.attn.wq_du.weight").?;
    try std.testing.expectEqual([2]usize{ 3072, 6144 }, .{ q.lo, q.hi });
    const o = by.get("model.llm.layers.0.attn.wo_ud.weight").?;
    try std.testing.expectEqual(@as(?u8, 1), o.split);
    const wp = by.get("model.llm.layers.5.attn.k_sconv.weight_proj").?;
    try std.testing.expectEqual([2]usize{ 16, 32 }, .{ wp.lo, wp.hi });
    const sw = by.get("model.llm.layers.5.attn.k_sconv.static_w").?;
    try std.testing.expectEqual([2]usize{ 512, 1024 }, .{ sw.lo, sw.hi });
    const w13 = by.get("model.llm.layers.0.mlp.experts.w13_weight").?;
    try std.testing.expectEqual([2]usize{ 2304, 4608 }, .{ w13.lo, w13.hi });
    const w2 = by.get("model.llm.layers.40.mlp.experts.w2_weight_scale").?;
    try std.testing.expectEqual([2]usize{ 36, 72 }, .{ w2.lo, w2.hi });
    const emb = by.get("model.llm.embed.weight").?;
    try std.testing.expectEqual([2]usize{ 100512, 201024 }, .{ emb.lo, emb.hi });
    try std.testing.expect(by.get("model.llm.attn_res.attn_queries.4").?.dims == 2); // feeds global layer 5
    try std.testing.expect(by.get("model.llm.attn_res.attn_queries.47") == null);
    // the dense share is the reference's 6.74 GB a rank (weights.envelope, GB = 1e9)
    const dense = bytesOf(r0, .dense);
    try std.testing.expect(dense > 6_700_000_000 and dense < 6_800_000_000);
}
