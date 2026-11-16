//! tf-inkling2-pack MODEL_DIR RANK WORLD OUT_DIR LAYER:E,E,...[;LAYER:...]: one rank's packed slots of the named
//! experts (experts.zig), each written as OUT_DIR/L{layer}_E{expert}.{up,down}.bin (int32 words, little-endian) and,
//! for NVFP4, .g.bin (g13 then g2, f32). For the gate against the reference packers (tests/zig_port/compare_pack.py).
//! TF_INKLING2_PACK_DIRECT=1: the fill's path instead of the mapping -- each expert's ranges read past the page cache
//! (fill.Loader's direct_gather) and packed from those copies with the precomputed global scales.
//! TF_INKLING2_PACK_GPU=1: the fill's device path -- each expert's ranges read past the page cache into the loader's
//! mapped stages and packed by the device packers (ink2_pack_nv / ink2_pack_mx) into device slots, downloaded here.
//! This host tool embeds no kernels: TF_INKLING2_PACK_FATBIN names the inkling2.fatbin the run tool embeds
//! (`zig build fatbins` installs it as zig-out/fatbin/inkling2.fatbin).
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-pack MODEL_DIR RANK WORLD OUT_DIR LAYER:E,E,...[;LAYER:...]\n";

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[2], 10);
    const world = try std.fmt.parseInt(u32, args[3], 10);
    const c = try ink.config.read(a, io, args[1]);
    try c.checkSplit(world);
    var ck = try core.checkpoint.Checkpoint.openModel(init.gpa, io, args[1]);
    defer ck.close();
    const store: ink.experts.Store = .{ .c = &c, .ck = &ck, .rank = rank, .world = world };
    if (if (std.c.getenv("TF_INKLING2_PACK_GPU")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false) return packGpu(init, a, io, &c, args, rank, world);
    const direct = if (std.c.getenv("TF_INKLING2_PACK_DIRECT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    const loader: ?*ink.fill.Loader = if (direct) try ink.fill.Loader.open(init.gpa, io, &c, args[1], rank, world, 1, 8) else null;
    defer if (loader) |l| l.close();
    if (loader) |l| std.debug.print("packing from O_DIRECT copies (direct={})\n", .{l.direct});
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[4], .{});
    defer dir.close(io);
    var n: usize = 0;
    var layers = std.mem.tokenizeScalar(u8, args[5], ';');
    while (layers.next()) |spec| {
        const colon = std.mem.indexOfScalar(u8, spec, ':') orelse return error.BadSpec;
        const layer = try std.fmt.parseInt(u32, spec[0..colon], 10);
        if (layer >= c.layers) return error.BadLayer;
        const words = ink.experts.slotWords(c.formats[layer], c.hidden, store.inter());
        var es = std.mem.tokenizeScalar(u8, spec[colon + 1 ..], ',');
        while (es.next()) |t| {
            const e = try std.fmt.parseInt(usize, t, 10);
            if (e >= c.experts) return error.BadExpert;
            var slot: ink.experts.Slot = .{ .fmt = c.formats[layer], .up = try a.alloc(u32, words[0]), .down = try a.alloc(u32, words[1]) };
            if (loader) |l| {
                var v = try l.store.views(layer, e);
                var ptrs = [4]*[]const u8{ &v.w13, &v.s13, &v.w2, &v.s2 };
                try l.gather.read(l.sources, &ptrs);
                try l.store.packViews(init.gpa, layer, e, v, &slot);
            } else try store.pack(init.gpa, layer, e, &slot);
            var name_buf: [64]u8 = undefined;
            try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name_buf, "L{d}_E{d}.up.bin", .{ layer, e }), .data = std.mem.sliceAsBytes(slot.up) });
            try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name_buf, "L{d}_E{d}.down.bin", .{ layer, e }), .data = std.mem.sliceAsBytes(slot.down) });
            if (slot.fmt == .nvfp4) {
                const g = [2]f32{ slot.g13, slot.g2 };
                try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name_buf, "L{d}_E{d}.g.bin", .{ layer, e }), .data = std.mem.sliceAsBytes(&g) });
            }
            n += 1;
        }
    }
    std.debug.print("rank {d}/{d}: {d} slots packed into {s}\n", .{ rank, world, n, args[4] });
    return 0;
}

/// The device path: every named expert into a slot of its format's device pool (in spec order), then each slot
/// downloaded and written as the host paths write theirs.
fn packGpu(init: std.process.Init, a: std.mem.Allocator, io: std.Io, c: *const ink.config.Config, args: []const [:0]const u8, rank: u32, world: u32) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    const loader = try ink.fill.Loader.open(init.gpa, io, c, args[1], rank, world, 1, 8);
    defer loader.close();
    if (std.c.getenv("TF_INKLING2_PACK_FATBIN")) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(path), a, .limited(256 << 20));
        const image = try a.alignedAlloc(u8, .@"16", bytes.len);
        @memcpy(image, bytes);
        loader.image = image;
        std.debug.print("device packers from {s} ({d} bytes)\n", .{ std.mem.span(path), image.len });
    }
    std.debug.print("packing on the device from O_DIRECT reads into mapped stages (direct={})\n", .{loader.direct});
    var jobs: std.ArrayList(ink.fill.Job) = .empty;
    var count: [2]usize = .{ 0, 0 };
    var layers = std.mem.tokenizeScalar(u8, args[5], ';');
    while (layers.next()) |spec| {
        const colon = std.mem.indexOfScalar(u8, spec, ':') orelse return error.BadSpec;
        const layer = try std.fmt.parseInt(u32, spec[0..colon], 10);
        if (layer >= c.layers) return error.BadLayer;
        const fi: usize = if (c.formats[layer] == .nvfp4) 0 else 1;
        var es = std.mem.tokenizeScalar(u8, spec[colon + 1 ..], ',');
        while (es.next()) |t| {
            const e = try std.fmt.parseInt(u32, t, 10);
            if (e >= c.experts) return error.BadExpert;
            try jobs.append(a, .{ .layer = layer, .expert = e, .slot = @intCast(count[fi]) });
            count[fi] += 1;
        }
    }
    const Il = c.inter / world;
    const words = [2][2]usize{ ink.experts.slotWords(.nvfp4, c.hidden, Il), ink.experts.slotWords(.mxfp8, c.hidden, Il) };
    var bufs: [6]cuda.DeviceBuffer = undefined;
    for (0..2) |fi| {
        bufs[2 * fi] = try cuda.DeviceBuffer.alloc(&driver, @max(16, count[fi] * words[fi][0] * 4));
        bufs[2 * fi + 1] = try cuda.DeviceBuffer.alloc(&driver, @max(16, count[fi] * words[fi][1] * 4));
    }
    bufs[4] = try cuda.DeviceBuffer.alloc(&driver, @max(16, count[0] * 8));
    bufs[5] = try cuda.DeviceBuffer.alloc(&driver, @max(16, count[0] * 4));
    defer for (&bufs) |*b| b.free();
    for (bufs) |b| try b.fill8(0xA5, null); // a slot the packers missed cannot pass
    const t: ink.fill.Target = .{ .up = .{ bufs[0].ptr, bufs[2].ptr }, .down = .{ bufs[1].ptr, bufs[3].ptr }, .up_scale = bufs[4].ptr, .down_scale = bufs[5].ptr, .words = words };
    try loader.loadInto(&driver, t, jobs.items, null);
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[4], .{});
    defer dir.close(io);
    for (jobs.items) |j| {
        const fi: usize = if (c.formats[j.layer] == .nvfp4) 0 else 1;
        const up = try a.alloc(u32, words[fi][0]);
        const down = try a.alloc(u32, words[fi][1]);
        try bufs[2 * fi].download(j.slot * words[fi][0] * 4, std.mem.sliceAsBytes(up));
        try bufs[2 * fi + 1].download(j.slot * words[fi][1] * 4, std.mem.sliceAsBytes(down));
        var name_buf: [64]u8 = undefined;
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name_buf, "L{d}_E{d}.up.bin", .{ j.layer, j.expert }), .data = std.mem.sliceAsBytes(up) });
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name_buf, "L{d}_E{d}.down.bin", .{ j.layer, j.expert }), .data = std.mem.sliceAsBytes(down) });
        if (fi == 0) {
            var g2: [2]f32 = undefined;
            var g: [3]f32 = undefined;
            try bufs[4].download(j.slot * 8, std.mem.sliceAsBytes(g[0..2]));
            try bufs[5].download(j.slot * 4, std.mem.sliceAsBytes(g[2..3]));
            if (g[0] != g[1]) return error.ScalePair;
            g2 = .{ g[0], g[2] };
            try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&name_buf, "L{d}_E{d}.g.bin", .{ j.layer, j.expert }), .data = std.mem.sliceAsBytes(&g2) });
        }
    }
    const st = loader.stats;
    std.debug.print("rank {d}/{d}: {d} slots packed on the device into {s} ({d:.1} MB read, {d:.1} ms read, {d:.1} ms waiting on packs)\n", .{ rank, world, jobs.items.len, args[4], @as(f64, @floatFromInt(st.bytes)) / 1e6, @as(f64, @floatFromInt(st.read_ns)) / 1e6, @as(f64, @floatFromInt(st.pack_ns)) / 1e6 });
    return 0;
}
