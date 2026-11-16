//! tf-inkling2-exl3-load VIEW RANK WORLD [FIRST LAST]: the EXL3 store's slots of (RANK, WORLD) by the direct loader
//! (e-load, exl3.loadDirect: O_DIRECT reads into pinned stages, slots cut on the device) on one GPU, timed, and every
//! slot's hash on the device (ink2_exl3_hash; TF_INKLING2_EXL3_HASH=PATH writes them to PATH.r<RANK>, the run tool's
//! format). With FIRST LAST, every slot of layers FIRST..LAST is compared byte for byte with the lane's packing
//! (exl3.packLayer: whole batch files read, packSlot on the host), its host hash (exl3.slotHash) with the device's,
//! and two planted controls must be caught: one byte flipped on the device (the compare and the hash both see it).
//! VIEW "env": the view from TF_INKLING2_VIEW (the launcher copies absolute paths named on its command line to the worker).
//! No model, no TP group, no dense weights: the store alone; MemAvailable before and after is printed.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-exl3-load VIEW|env RANK WORLD [FIRST LAST]\n";
const gib: f64 = 1024.0 * 1024.0 * 1024.0;

fn sec(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn monoNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4 and args.len != 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const dir: []const u8 = if (std.mem.eql(u8, args[1], "env")) std.mem.span(std.c.getenv("TF_INKLING2_VIEW") orelse {
        std.debug.print("TF_INKLING2_VIEW is not set\n", .{});
        return 2;
    }) else args[1];
    const rank = try std.fmt.parseInt(usize, args[2], 10);
    const world = try std.fmt.parseInt(usize, args[3], 10);
    const first: ?usize = if (args.len == 6) try std.fmt.parseInt(usize, args[4], 10) else null;
    const last: usize = if (args.len == 6) try std.fmt.parseInt(usize, args[5], 10) else 0;
    const c = try ink.config.read(a, io, dir);
    if (!c.exl3 or rank >= world) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const H: usize = c.hidden;
    const I: usize = c.inter;
    const Il = I / world;
    const L = c.layers;
    const E = c.experts;
    if (first) |f| if (f > last or last >= L) {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    var ix = try ink.exl3.Index.open(a, io, dir, &c);
    defer ix.deinit();

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    const d = &driver;
    var mod = try cuda.Module.load(d, cuda.kernels.inkling2_exl3);
    defer mod.unload();
    const cut = try mod.function("ink2_exl3_cut");
    const hash = try mod.function("ink2_exl3_hash");
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();

    // ---- the load
    const avail0 = ink.fill.readAvail() orelse 0;
    const bufs = try a.alloc(cuda.DeviceBuffer, L);
    @memset(bufs, .{ .d = d, .ptr = 0, .len = 0 });
    defer for (bufs) |*b| if (b.ptr != 0) b.free();
    const host_ptr = try a.alloc(u64, L * E);
    const host_k = try a.alloc(i32, L * E);
    var stats: ink.exl3.Store.Stats = .{};
    const floor_gib: f64 = if (std.c.getenv("TF_INKLING2_FLOOR_GIB")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 2 else 2;
    const t0 = monoNs();
    const bytes = try ink.exl3.loadDirect(init.gpa, io, d, cut, &ix, dir, H, I, rank, world, @intFromFloat(floor_gib * gib), bufs, host_ptr, host_k, &stats, null);
    const load_ns = monoNs() - t0;
    const avail1 = ink.fill.readAvail() orelse 0;
    const x = stats.direct;
    std.debug.print("exl3-load rank {d}/{d}: {d} slots, {d:.2} GiB, loaded in {d:.2} s: headers {d:.2} s ({d} files, {d} read through the page cache), stream {d:.2} s: {d:.1} GB read in {d} pieces ({d:.2} GB/s), {d} readers busy {d:.0}%; main thread: reads wait {d:.2} s, checks {d:.2} s, buffers {d:.2} s, cuts {d:.2} s ({d:.1} GB of slots), pace {d:.2} s (paused {d} ms); {d} stages of {d} MiB, ring {d}; MemAvailable {d:.2} -> {d:.2} GiB (min seen {d:.2})\n", .{ rank, world, L * E, @as(f64, @floatFromInt(bytes)) / gib, sec(load_ns), sec(x.heads_ns), stats.files, x.cached, sec(x.stream_ns), @as(f64, @floatFromInt(stats.read_bytes)) / 1e9, x.pieces, @as(f64, @floatFromInt(stats.read_bytes)) / 1e9 / @max(sec(x.stream_ns), 1e-9), x.readers, 100 * sec(x.busy_ns) / @max(sec(x.stream_ns) * @as(f64, @floatFromInt(x.readers)), 1e-9), sec(x.wait_ns), sec(x.check_ns), sec(x.alloc_ns), sec(x.cut_ns), @as(f64, @floatFromInt(x.cut_bytes)) / 1e9, sec(x.pace_ns), stats.paused_ms, x.stages, x.stage_bytes >> 20, x.ring, @as(f64, @floatFromInt(avail0)) / gib, @as(f64, @floatFromInt(avail1)) / gib, @as(f64, @floatFromInt(stats.min_avail)) / gib });

    // ---- every slot's hash on the device
    var tptr = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host_ptr));
    defer tptr.free();
    var tk = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host_k));
    defer tk.free();
    const hashes = try a.alloc(u64, L * E);
    const th = monoNs();
    try ink.exl3.hashSlots(d, hash, s, tptr.ptr, tk.ptr, L * E, H, Il, hashes);
    var all: u64 = L * E;
    for (hashes) |h| all = ink.exl3.fmix64(all ^ h);
    std.debug.print("exl3-load rank {d}/{d}: {d} slot hashes in {d:.2} s; all {x:0>16}\n", .{ rank, world, L * E, sec(monoNs() - th), all });
    if (std.c.getenv("TF_INKLING2_EXL3_HASH")) |p| {
        const name = try std.fmt.allocPrint(a, "{s}.r{d}", .{ std.mem.span(p), rank });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(hashes) });
        std.debug.print("exl3-load: hashes in {s}\n", .{name});
    }
    const f = first orelse return 0;

    // ---- layers first..last byte for byte against the lane's packing, and the host hash of every such slot
    // (packLayer's per-file buffers from the general allocator: the arena would keep every one of them)
    const per = std.mem.alignForward(usize, ink.exl3.parts(H, Il, 3).total, 256);
    const host = try a.alloc(u8, E * per);
    const offs = try a.alloc(usize, E);
    const ks = try a.alloc(u8, E);
    var most: usize = 0;
    for (bufs) |b| most = @max(most, b.len);
    const got = try a.alloc(u8, most);
    var slots_bad: usize = 0;
    var bytes_bad: usize = 0;
    var hash_bad: usize = 0;
    var compared: u64 = 0;
    const tc = monoNs();
    for (f..last + 1) |l| {
        _ = try ink.exl3.packLayer(init.gpa, dir, &ix, l, H, I, rank, world, host, offs, ks);
        try bufs[l].download(0, got[0..bufs[l].len]);
        var lb: usize = 0;
        var ls: usize = 0;
        var lh: usize = 0;
        for (0..E) |e| {
            const n = ink.exl3.parts(H, Il, ks[e]).total;
            const at = host_ptr[l * E + e] - bufs[l].ptr;
            if (ks[e] != host_k[l * E + e]) return error.BitsDiffer;
            const new = got[at..][0..n];
            const old = host[offs[e]..][0..n];
            var nb: usize = 0;
            if (!std.mem.eql(u8, new, old)) for (new, old) |p, q| {
                nb += @intFromBool(p != q);
            };
            if (nb > 0) {
                ls += 1;
                lb += nb;
                if (ls <= 3) std.debug.print("  layer {d} expert {d}: {d} of {d} bytes differ\n", .{ l, e, nb, n });
            }
            if (ink.exl3.slotHash(new) != hashes[l * E + e]) lh += 1;
            compared += n;
        }
        std.debug.print("exl3-load cmp layer {d}: {d} slots, {d} bytes: {d} slots differ ({d} bytes), {d} device hashes differ from the host's\n", .{ l, E, bufs[l].len, ls, lb, lh });
        slots_bad += ls;
        bytes_bad += lb;
        hash_bad += lh;
    }
    std.debug.print("exl3-load cmp rank {d}/{d}: layers {d}..{d}, {d} slots, {d:.2} GB compared in {d:.1} s: {d} slots differ ({d} bytes), {d} hashes differ: {s}\n", .{ rank, world, f, last, (last + 1 - f) * E, @as(f64, @floatFromInt(compared)) / 1e9, sec(monoNs() - tc), slots_bad, bytes_bad, hash_bad, if (slots_bad == 0 and hash_bad == 0) "IDENTICAL" else "DIFFER" });

    // ---- the controls: one byte of layer `f` expert 0's w2 part flipped on the device, then restored
    const p0 = ink.exl3.parts(H, Il, @intCast(host_k[f * E]));
    const at0 = host_ptr[f * E] - bufs[f].ptr + p0.w2 + 123;
    var one: [1]u8 = undefined;
    try bufs[f].download(at0, &one);
    const keep = one[0];
    one[0] ^= 0x01;
    try bufs[f].upload(at0, &one);
    const h1 = try a.alloc(u64, L * E);
    try ink.exl3.hashSlots(d, hash, s, tptr.ptr, tk.ptr, L * E, H, Il, h1);
    var hdiff: usize = 0;
    for (hashes, h1) |p, q| hdiff += @intFromBool(p != q);
    _ = try ink.exl3.packLayer(init.gpa, dir, &ix, f, H, I, rank, world, host, offs, ks);
    try bufs[f].download(0, got[0..bufs[f].len]);
    var cdiff: usize = 0;
    for (0..E) |e| {
        const n = ink.exl3.parts(H, Il, ks[e]).total;
        const at = host_ptr[f * E + e] - bufs[f].ptr;
        if (!std.mem.eql(u8, got[at..][0..n], host[offs[e]..][0..n])) for (got[at..][0..n], host[offs[e]..][0..n]) |p, q| {
            cdiff += @intFromBool(p != q);
        };
    }
    one[0] = keep;
    try bufs[f].upload(at0, &one);
    std.debug.print("exl3-load control (layer {d} expert 0, one byte flipped): compare sees {d} byte(s) differ, the hash {d} slot(s): {s}\n", .{ f, cdiff, hdiff, if (cdiff == 1 and hdiff == 1) "RED as planted" else "MISSED" });
    return if (slots_bad == 0 and hash_bad == 0 and cdiff == 1 and hdiff == 1) 0 else 1;
}
