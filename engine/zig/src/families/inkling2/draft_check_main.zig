//! tf-inkling2-draft-check MODEL_DIR [RANK WORLD] PREFIX: the DSpark draft head on captured target taps, against the
//! verified overlay port's drafts (tests/acceptance: probe.py, tap convention A). PREFIX.taps [rows][8][4608] bf16,
//! PREFIX.ids the sequence (int32), PREFIX.drafts [anchors][1 + 7] int32 (the anchor position, then the 7 drafted
//! tokens). TP 1 (no RANK WORLD) or rank RANK of WORLD (the tp2 helper's argument order; NCCL_COMM_ID set): the
//! target's dense weights for its embedding, embed_norm and LM head (this rank's share); no experts. Every rank reads
//! the same taps (the target's taps are replicated). Every row's taps go into the context once (a draft at anchor p
//! sees positions < p only); f_rnn is row p - 1's.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 5) {
        std.debug.print("usage: tf-inkling2-draft-check MODEL_DIR [RANK WORLD] PREFIX\n", .{});
        return 2;
    }
    const rank: usize = if (args.len == 5) try std.fmt.parseInt(usize, args[2], 10) else 0;
    const world: usize = if (args.len == 5) try std.fmt.parseInt(usize, args[3], 10) else 1;
    const prefix = args[args.len - 1];
    var buf: [512]u8 = undefined;
    const taps = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&buf, "{s}.taps", .{prefix}), a, .unlimited);
    const ids_b = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&buf, "{s}.ids", .{prefix}), a, .unlimited);
    const dr_b = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&buf, "{s}.drafts", .{prefix}), a, .unlimited);
    const H = ink.dspark.H;
    const row_bytes = 8 * H * 2;
    const rows = taps.len / row_bytes;
    const ids: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, ids_b));
    const refs: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, dr_b));
    const anchors = refs.len / 8;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    const group: ?cuda.tp_group.Group = if (world > 1) try cuda.tp_group.Group.open(&driver, rank, world) else null;
    const m = try ink.model.Model.open(init.gpa, io, &ctx, args[1], rank, world, group, 4096, 16);
    defer m.close();
    const t0 = std.Io.Clock.awake.now(io);
    const d = try ink.dspark.Draft.open(init.gpa, io, m, args[1], 16);
    defer d.close();
    const load_ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    std.debug.print("rank {d}/{d}: draft head holds {d:.3} GB of device memory\n", .{ rank, world, @as(f64, @floatFromInt(d.heldBytes())) / 1e9 });
    var dev = try cuda.DeviceBuffer.fromHost(&driver, taps);
    defer dev.free();
    var p: usize = 0;
    // p4-small: TF_INKLING2_DRAFT_CHECK_ROWS rows an absorb (default 16); with TF_INKLING2_ABSORB_WIDE=1 a piece
    // wider than 16 rows takes the draft's wide absorb
    const piece: usize = if (std.c.getenv("TF_INKLING2_DRAFT_CHECK_ROWS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 16;
    if (piece == 0) return error.BadPiece;
    while (p < rows) : (p += piece) {
        const n = @min(piece, rows - p);
        // p4-mem: with the lazy context the positions a piece and its drafts reach mapped first (lazy.zig)
        if (m.lazy) |lz| try lz.ensure(m, p + n + ink.dspark.block);
        try d.absorb(dev.ptr + p * row_bytes, p, n);
    }
    try m.s.synchronize();
    var equal: usize = 0;
    var total: usize = 0;
    var same_k: usize = 0;
    var ms_sum: f64 = 0;
    var ms_first: f64 = 0;
    for (0..anchors) |i| {
        const r = refs[i * 8 ..][0..8];
        const pos: usize = @intCast(r[0]);
        try d.setFrnn(dev.ptr + (pos - 1) * row_bytes);
        var out: [7]i32 = undefined;
        const td = std.Io.Clock.awake.now(io);
        try d.draft(ids[pos], pos, &out);
        const one = @as(f64, @floatFromInt(td.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
        if (i == 0) ms_first = one;
        ms_sum += one;
        var k_zig: usize = 0;
        while (k_zig < 7 and pos + 1 + k_zig < ids.len and out[k_zig] == ids[pos + 1 + k_zig]) k_zig += 1;
        var k_ref: usize = 0;
        while (k_ref < 7 and pos + 1 + k_ref < ids.len and r[1 + k_ref] == ids[pos + 1 + k_ref]) k_ref += 1;
        for (0..7) |j| {
            if (out[j] == r[1 + j]) equal += 1;
        }
        total += 7;
        if (k_zig == k_ref) same_k += 1;
        if (i < 4 or out[0] != r[1]) std.debug.print("anchor {d}: zig {any} | overlay {any} | accepted zig {d} overlay {d}\n", .{ pos, out, r[1..8], k_zig, k_ref });
    }
    std.debug.print("rank {d}/{d}: draft head loaded in {d:.0} ms; {d} anchors: drafted tokens equal to the overlay port's {d}/{d} ({d:.3}); accepted length equal on {d}/{d}; {d:.1} ms a draft ({d:.1} past the first)\n", .{ rank, world, load_ms, anchors, equal, total, @as(f64, @floatFromInt(equal)) / @as(f64, @floatFromInt(total)), same_k, anchors, ms_sum / @as(f64, @floatFromInt(anchors)), (ms_sum - ms_first) / @as(f64, @floatFromInt(@max(anchors, 2) - 1)) });
    return 0;
}
