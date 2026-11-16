//! tf-inkling2-route-check OUT_PREFIX [ROWS]: ink2_route (the masked router + unmasked demand) on seeded random router
//! logits, a bias and three masks (all routable, a random half, a mask that removes most of each row's unmasked
//! top-8), at the release's E 384, top-8, route_scale 8. Writes OUT_PREFIX.{logits,bias,masks,picks,weights,want_ids,
//! want_scores} (little-endian; [masks][rows][...]) for tests/zig_port/check_router.py, which runs HF's
//! Inkling2TopKRouter on the same logits (bias, and bias with -inf at absent experts for the masked rule).
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

const E = 384;
const K = 8;
const scale: f32 = 8.0;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 2) {
        std.debug.print("usage: tf-inkling2-route-check OUT_PREFIX [ROWS]\n", .{});
        return 2;
    }
    const prefix = args[1];
    const T: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 256;
    var prng = std.Random.DefaultPrng.init(0x1a2b3c);
    const rng = prng.random();
    const logits = try a.alloc(f32, T * E);
    for (logits) |*v| v.* = (rng.float(f32) - 0.5) * 8; // sigmoid's working range, both tails
    const bias = try a.alloc(f32, E);
    for (bias) |*v| v.* = (rng.float(f32) - 0.5) * 0.2;
    const M = 3;
    const masks = try a.alloc(u8, M * E);
    @memset(masks[0..E], 1);
    for (masks[E .. 2 * E]) |*m| m.* = @intFromBool(rng.boolean());
    // the third: every expert routable but the 48 with the largest mean (score + bias) over the rows
    {
        var key: [E]f64 = @splat(0);
        for (0..T) |t| for (0..E) |e| {
            key[e] += 1.0 / (1.0 + @exp(-@as(f64, logits[t * E + e]))) + bias[e];
        };
        var order: [E]u32 = undefined;
        ink.residency.topIndices(&key, &order);
        @memset(masks[2 * E .. 3 * E], 1);
        for (order[0..48]) |e| masks[2 * E + e] = 0;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    var k = try ink.kern.Kernels.load(&driver, @intCast(try ctx.attribute(.multiprocessor_count)));
    defer k.unload();
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var lg = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(logits));
    defer lg.free();
    var bs = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(bias));
    defer bs.free();
    var mk = try cuda.DeviceBuffer.fromHost(&driver, masks);
    defer mk.free();
    var bufs: [4]cuda.DeviceBuffer = undefined;
    for (&bufs) |*b| b.* = try cuda.DeviceBuffer.alloc(&driver, T * K * 4);
    defer for (&bufs) |*b| b.free();
    const out = [4][]u8{ try a.alloc(u8, M * T * K * 4), try a.alloc(u8, M * T * K * 4), try a.alloc(u8, M * T * K * 4), try a.alloc(u8, M * T * K * 4) };
    for (0..M) |mi| {
        for (&bufs) |*b| try b.fill8(0xff, s.handle);
        try k.route(s, lg.ptr, bs.ptr, mk.ptr + mi * E, T, E, K, scale, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr);
        try s.synchronize();
        for (bufs, out) |b, o| try b.download(0, o[mi * T * K * 4 ..][0 .. T * K * 4]);
    }
    const names = [_][]const u8{ "logits", "bias", "masks", "picks", "weights", "want_ids", "want_scores" };
    const data = [_][]const u8{ std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(bias), masks, out[0], out[1], out[2], out[3] };
    var buf: [512]u8 = undefined;
    for (names, data) |n, d| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, n }), .data = d });
    std.debug.print("ink2_route: {d} rows x {d} experts, top-{d}, route_scale {d}, {d} masks -> {s}.*\n", .{ T, E, K, scale, M, prefix });
    return 0;
}
