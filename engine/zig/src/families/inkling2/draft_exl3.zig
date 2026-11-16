//! e-step (prefill lane phase 5): the DSpark draft head's seven projections a layer (q, k, v, o_proj, gate, up,
//! down_proj of mtp.safetensors) as EXL3 trellises (TF_INKLING2_DRAFT_EXL3=<pack folder>: tools/quantize_draft.py,
//! the ExLlamaV3 fork's quantize_exl3 at 6 bits, `{prefix}.trellis` int16 [K/16][N/16][16 bits] / `.suh` / `.svh` /
//! `.mul1`, prefix = `layers.<i>.<module>`), run through core's CUDA EXL3 projection (cuda.exl3.Dense: the input
//! rotation, the decode-row linear with fp32 sums; the draft block's 8 rows a call). A rank holds its share as the
//! bf16 path splits them (dspark.zig open): q / k / v / gate / up by 128-column blocks of N (svh with them), o_proj /
//! down_proj by 128-row blocks of K (suh with them; the ranks' fp32 partials summed as today's). The context
//! projections (k_proj_ctx, v_proj_ctx: the prompt's absorb), the norms, the Markov head and the LM head stay bf16.
//! The drafter's numerics never change a served token, only how many drafts land: acceptance and draft time are the
//! measures (P5 brief). The cut is e-dense's dense_exl3.zig `weight` (a copy, so the two files merge apart).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

pub const Weight = cuda.exl3.Weight;
const Cut = enum { cols, rows };

pub const layers = 5;

/// One layer's EXL3 projections (this rank's share).
pub const LayerW = struct { q: Weight, k: Weight, v: Weight, o: Weight, gate: Weight, up: Weight, down: Weight };

const names = [_]struct { []const u8, Cut }{
    .{ "self_attn.q_proj", .cols }, .{ "self_attn.k_proj", .cols }, .{ "self_attn.v_proj", .cols }, .{ "self_attn.o_proj", .rows },
    .{ "mlp.gate_proj", .cols },    .{ "mlp.up_proj", .cols },      .{ "mlp.down_proj", .rows },
};

pub const DraftExl3 = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    op: cuda.exl3.Dense,
    layer: [layers]LayerW = undefined,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0, // this rank's device bytes of every upload
    rank: usize,
    world: usize,

    /// The pack in `dir`, cut for `rank` of `world`; every weight's scratch reserved on `s` (a graph capture of the
    /// pass allocates nothing).
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8, rank: usize, world: usize) !*DraftExl3 {
        const dx = try gpa.create(DraftExl3);
        errdefer gpa.destroy(dx);
        dx.* = .{ .gpa = gpa, .d = d, .op = try cuda.exl3.Dense.load(d), .rank = rank, .world = world };
        errdefer dx.free();
        var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        var nb: [96]u8 = undefined;
        for (&dx.layer, 0..) |*l, li| {
            var ws: [names.len]Weight = undefined;
            for (names, &ws) |nm, *w| w.* = try dx.weight(&ck, try std.fmt.bufPrint(&nb, "layers.{d}.{s}", .{ li, nm[0] }), nm[1]);
            l.* = .{ .q = ws[0], .k = ws[1], .v = ws[2], .o = ws[3], .gate = ws[4], .up = ws[5], .down = ws[6] };
            for (ws) |w| try dx.op.reserve(s, w);
        }
        std.log.info("inkling2 rank {d}: e-step draft head projections as EXL3 from {s}: {d:.3} GiB of trellis on the device", .{ rank, dir, @as(f64, @floatFromInt(dx.bytes)) / (1024.0 * 1024.0 * 1024.0) });
        return dx;
    }

    fn free(dx: *DraftExl3) void {
        dx.op.unload();
        for (dx.bufs.items) |*b| b.free();
        dx.bufs.deinit(dx.gpa);
    }

    pub fn close(dx: *DraftExl3) void {
        dx.free();
        dx.gpa.destroy(dx);
    }

    /// y [rows][n] = x [rows][k] (bf16) through `w`, y bf16 or fp32 (rows <= 128: one decode-row call).
    pub fn run(dx: *DraftExl3, s: cuda.Stream, w: Weight, x: u64, rows: usize, y: u64, y_f32: bool) !void {
        return dx.op.decode(s, w, x, .bf16, rows, y, if (y_f32) .f32 else .bf16);
    }

    /// One tensor's rank share on the device (e-dense's dense_exl3.zig `weight`, without its cache or canon).
    fn weight(dx: *DraftExl3, ck: *core.checkpoint.Checkpoint, prefix: []const u8, cut: Cut) !Weight {
        var nb: [160]u8 = undefined;
        const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix}));
        if (t.dtype != .i16 or t.rank != 3 or t.shape[2] % 8 != 0) return error.BadDraftExl3Trellis;
        const K = t.shape[0] * 16;
        const N = t.shape[1] * 16;
        const halves: u32 = @intCast(t.shape[2] / 8);
        const suh = try ck.expect(try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), .f16, &.{K});
        const svh = try ck.expect(try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), .f16, &.{N});
        const cb: cuda.exl3.Codebook = if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mul1", .{prefix}))) .mul1 else if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mcg", .{prefix}))) .mcg else .inst3;
        if (K % 128 != 0 or N % 128 != 0) return error.BadDraftExl3Shape;
        const ways = dx.world;
        const r = dx.rank;
        const dim = if (cut == .rows) K else N;
        if (dim % (128 * ways) != 0) return error.BadDraftExl3Split;
        const lo = r * dim / ways;
        const hi = (r + 1) * dim / ways;
        const k = if (cut == .rows) hi - lo else K;
        const n = if (cut == .rows) N else hi - lo;
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
        return w;
    }

    fn keep(dx: *DraftExl3, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(dx.d, bytes);
        errdefer b.free();
        try dx.bufs.append(dx.gpa, b);
        dx.bytes += bytes.len;
        return b.ptr;
    }
};
