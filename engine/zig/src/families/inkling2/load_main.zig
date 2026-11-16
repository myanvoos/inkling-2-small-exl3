//! tf-inkling2-load MODEL_DIR RANK WORLD [--digests]: one rank's dense weights onto this GPU (dense.zig), the time and
//! bytes it took; --digests reads every tensor back from the device and prints `sha256 name bytes`, for the gate
//! against the reference loader's tensors (tests/zig_port/compare_dense.py). Experts are not loaded here.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-load MODEL_DIR RANK WORLD [--digests]\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[2], 10);
    const world = try std.fmt.parseInt(u32, args[3], 10);
    const digests = args.len > 4 and std.mem.eql(u8, args[4], "--digests");
    var out_buf: [1 << 16]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const w = &out.interface;
    defer w.flush() catch {};
    const c = try ink.config.read(a, io, args[1]);
    const entries = try ink.plan.plan(a, &c, rank, world);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    var ck = try core.checkpoint.Checkpoint.openModel(init.gpa, io, args[1]);
    defer ck.close();
    const t0 = std.Io.Clock.awake.now(io);
    var dense = try ink.dense.load(init.gpa, &driver, &ck, entries);
    defer dense.deinit();
    const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    try w.print("# rank {d}/{d}: {d} dense tensors, {d} bytes on the device in {d:.0} ms\n", .{ rank, world, dense.tensors.items.len, dense.bytes, ms });
    if (!digests) return 0;
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(init.gpa);
    for (dense.tensors.items) |*t| {
        const n = t.entry.rankBytes();
        try host.resize(init.gpa, n);
        try t.buf.download(0, host.items);
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host.items, &h, .{});
        try w.print("{x} {s} {d}\n", .{ &h, t.name, n });
    }
    return 0;
}
