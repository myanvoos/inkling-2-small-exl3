//! tf-inkling2-check MODEL_DIR [WORLD] [--plan RANK]: the family's reading of a checkpoint (config.json,
//! hf_quant_config.json), every planned tensor checked against the shard headers (name, dtype, shape) with nothing
//! left over but the skipped parts, and each rank's TP share. --plan RANK prints that rank's entries. No GPU.
const std = @import("std");
const core = @import("core");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-check MODEL_DIR [WORLD] [--plan RANK]\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var world: u32 = 2;
    var show: ?u32 = null;
    var ai: usize = 2;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--plan") and ai + 1 < args.len) {
            show = try std.fmt.parseInt(u32, args[ai + 1], 10);
            ai += 1;
        } else world = try std.fmt.parseInt(u32, args[ai], 10);
    }
    var out_buf: [1 << 16]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &out.interface;
    defer w.flush() catch {};
    const c = ink.config.read(a, io, args[1]) catch |err| {
        try w.print("{s}: refused: {s}\n", .{ args[1], @errorName(err) });
        return 1;
    };
    try w.print("{s}: inkling2, release geometry: {}\n", .{ args[1], c.isRelease() });
    try w.print("  layers {d} hidden {d} vocab {d} (unpadded {d}) heads {d} kv {d} head_dim {d} d_rel {d}\n", .{ c.layers, c.hidden, c.vocab, c.unpadded_vocab, c.heads, c.kv_heads, c.head_dim, c.d_rel });
    try w.print("  window {d} rel_extent {d} attn_res block {d} conv {d} eps {e}\n", .{ c.window, c.rel_extent, c.block, c.conv, c.eps });
    try w.print("  experts {d} top-{d} inter {d} caps gate {d} up {d} logits {d} route_scale {d} eos {d}\n", .{ c.experts, c.topk, c.inter, c.gate_cap, c.up_cap, c.logit_cap, c.route_scale, c.eos });
    try w.print("  formats: {d} nvfp4, {d} mxfp8; global layers:", .{ c.countFormat(.nvfp4), c.countFormat(.mxfp8) });
    for (0..c.layers) |i| if (!c.isLocal(@intCast(i))) try w.print(" {d}", .{i});
    try w.print("\n", .{});
    c.checkSplit(world) catch |err| {
        try w.print("  TP{d}: refused: {s}\n", .{ world, @errorName(err) });
        return 1;
    };
    // every planned tensor against the headers; the rest must be the parts skipped by name
    var ck = try core.checkpoint.Checkpoint.openModel(init.gpa, io, args[1]);
    defer ck.close();
    const entries = try ink.plan.plan(a, &c, 0, world);
    var bad: usize = 0;
    for (entries) |*e| {
        _ = ck.expect(e.name, e.dtype, e.wholeShape()) catch {
            bad += 1;
            continue;
        };
    }
    try ck.skip("model.visual.");
    const left = ck.unused();
    try w.print("  headers: {d} planned tensors, {d} refused, {d} left over beyond model.visual.*\n", .{ entries.len, bad, left });
    for (0..world) |r| {
        const p = try ink.plan.plan(a, &c, @intCast(r), world);
        try w.print("  TP{d} rank {d}: {d} entries, dense {d} B, experts {d} B (on disk, this rank's channels)\n", .{ world, r, p.len, ink.plan.bytesOf(p, .dense), ink.plan.bytesOf(p, .experts) });
    }
    if (show) |r| {
        const p = try ink.plan.plan(a, &c, r, world);
        try w.print("# rank {d} of {d}: name | dtype | whole shape | split dim | lo:hi | kind\n", .{ r, world });
        for (p) |*e| {
            try w.print("{s} | {s} | {any}", .{ e.name, @tagName(e.dtype), e.wholeShape() });
            if (e.split) |d| try w.print(" | d{d} | {d}:{d}", .{ d, e.lo, e.hi }) else try w.print(" | whole | -", .{});
            try w.print(" | {s}\n", .{@tagName(e.kind)});
        }
    }
    return if (bad == 0 and left == 0) 0 else 1;
}
