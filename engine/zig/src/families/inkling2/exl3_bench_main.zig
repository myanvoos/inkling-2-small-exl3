//! tf-inkling2-exl3-bench VIEW LAYER RANK WORLD ROWS REPS [SKEW]: the EXL3 lane's prompt kernels on one layer's real
//! slots (every expert of LAYER, this rank's share, packed once a batch file: exl3.packLayer), timed on one GPU, each
//! GEMM form's outputs compared bit for bit with the decode form's (g1: a pair's bits must not depend on the form).
//! The plan: ROWS rows, each routed to topk distinct experts (uniform, or SKEW s > 0: expert weights 1 / (r + 1)^s
//! over a fixed shuffle), pairs grouped by expert into items of up to kern.prompt_tile, as a prompt chunk's plan.
//! Inputs are seeded random (x bf16 ~ N(0, 1); rotated activations fp16 ~ N(0, 1)). No TP group, no model.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-exl3-bench VIEW LAYER RANK WORLD ROWS REPS [SKEW]\n";

fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

fn f16bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}

/// fp32 outputs `got` against `want`: values that differ, poison left (0xffffffff: never written), the largest |diff|
/// over values both finite, the first differing index.
fn diff(a: std.mem.Allocator, name: []const u8, got_b: []const u8, want_b: []const u8) ![]const u8 {
    const got = std.mem.bytesAsSlice(u32, got_b);
    const want = std.mem.bytesAsSlice(u32, want_b);
    var n: usize = 0;
    var poison: usize = 0;
    var first: ?usize = null;
    var worst: f32 = 0;
    for (got, want, 0..) |g, w, i| {
        if (g == 0xffffffff) poison += 1;
        if (g == w) continue;
        n += 1;
        if (first == null) first = i;
        const gf: f32 = @bitCast(g);
        const wf: f32 = @bitCast(w);
        if (std.math.isFinite(gf) and std.math.isFinite(wf)) worst = @max(worst, @abs(gf - wf));
    }
    return std.fmt.allocPrint(a, "{s}: {d} of {d} differ, {d} unwritten, max |diff| {e}, first at {?d}", .{ name, n, got.len, poison, worst, first });
}

/// Bytes that differ between `got` and `want` (bf16 / any element type), and the first differing index.
fn diffBytes(a: std.mem.Allocator, got: []const u8, want: []const u8) ![]const u8 {
    var n: usize = 0;
    var first: ?usize = null;
    for (got, want, 0..) |g, w, i| {
        if (g == w) continue;
        n += 1;
        if (first == null) first = i;
    }
    return std.fmt.allocPrint(a, "{d} of {d} bytes differ, first at byte {?d}", .{ n, got.len, first });
}

const Timer = struct {
    t0: cuda.Event,
    t1: cuda.Event,
    fn ms(t: *Timer, s: cuda.Stream, reps: usize) !f64 {
        try t.t1.record(s);
        try t.t1.synchronize();
        return @as(f64, try cuda.Event.elapsedMs(t.t0, t.t1)) / @as(f64, @floatFromInt(reps));
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 7 and args.len != 8) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const dir = args[1];
    const layer = try std.fmt.parseInt(usize, args[2], 10);
    const rank = try std.fmt.parseInt(usize, args[3], 10);
    const world = try std.fmt.parseInt(usize, args[4], 10);
    const T = try std.fmt.parseInt(usize, args[5], 10);
    const reps = try std.fmt.parseInt(usize, args[6], 10);
    const skew: f64 = if (args.len == 8) try std.fmt.parseFloat(f64, args[7]) else 0;
    const c = try ink.config.read(a, io, dir);
    if (!c.exl3 or rank >= world or T == 0 or reps == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const H: usize = c.hidden;
    const I: usize = c.inter;
    const Il = I / world;
    const E: usize = c.experts;
    const topk: usize = c.topk;
    const pairs = T * topk;
    try ink.exl3.checkSplit(H, I, world);
    var ix = try ink.exl3.Index.open(a, io, dir, &c);
    defer ix.deinit();

    // every expert's slot of this layer and rank, packed on the host, one device buffer
    const t_load = std.Io.Clock.awake.now(io);
    const per = std.mem.alignForward(usize, ink.exl3.parts(H, Il, 3).total, 256);
    const host = try a.alloc(u8, E * per);
    const offs = try a.alloc(usize, E);
    const ks = try a.alloc(u8, E);
    const used = try ink.exl3.packLayer(a, dir, &ix, layer, H, I, rank, world, host, offs, ks);

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    const d = &driver;
    var k = try ink.kern.Kernels.load(d, @intCast(@max(try ctx.attribute(.multiprocessor_count), 1)));
    defer k.unload();
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();

    var slots = try cuda.DeviceBuffer.fromHost(d, host[0..used]);
    defer slots.free();
    const ptr_h = try a.alloc(u64, E);
    const k_h = try a.alloc(i32, E);
    var k2: usize = 0;
    for (0..E) |e| {
        ptr_h[e] = slots.ptr + offs[e];
        k_h[e] = ks[e];
        if (ks[e] == 2) k2 += 1;
    }
    var tptr = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ptr_h));
    defer tptr.free();
    var tk = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(k_h));
    defer tk.free();
    // the engine's grouped kernel's tables (exl3.Store's): w13 and w2 trellis addresses, half-bits
    const t13_h = try a.alloc(u64, E);
    const t2_h = try a.alloc(u64, E);
    const k2_h = try a.alloc(i32, E);
    for (0..E) |e| {
        t13_h[e] = ptr_h[e];
        t2_h[e] = ptr_h[e] + ink.exl3.parts(H, Il, ks[e]).w2;
        k2_h[e] = 2 * @as(i32, ks[e]);
    }
    var t13 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(t13_h));
    defer t13.free();
    var t2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(t2_h));
    defer t2.free();
    var tk2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(k2_h));
    defer tk2.free();
    const ex: ink.kern.Kernels.Exl3 = .{ .slot_ptr = tptr.ptr, .slot_k = tk.ptr, .nslots = E, .hidden = H, .inter = Il, .cg = c.gate_cap, .cu = c.up_cap, .tp13 = t13.ptr, .tp2 = t2.ptr, .k2 = tk2.ptr };
    std.debug.print("exl3-bench: layer {d} ({d} K2 + {d} K3 experts), rank {d}/{d}: {d:.2} GiB of slots packed and uploaded in {d:.1} s\n", .{ layer, k2, E - k2, rank, world, @as(f64, @floatFromInt(used)) / (1 << 30), @as(f64, @floatFromInt(t_load.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9 });

    // the plan: each row's topk distinct experts, pairs p = t topk + j, grouped by expert into items of <= prompt_tile
    var prng = std.Random.DefaultPrng.init(0x1d2c3b4a);
    const r = prng.random();
    const perm = try a.alloc(usize, E);
    for (perm, 0..) |*v, i| v.* = i;
    r.shuffle(usize, perm);
    const cum = try a.alloc(f64, E);
    var total: f64 = 0;
    // e-prefill: EXL3_BENCH_PRIOR=FILE (fp32 [L][E], e.g. the residency prior plan-gamma-scores.f32): expert e drawn with
    // weight prior[LAYER][e] (perm = identity), the real model's skew instead of SKEW's power law
    const prior_w: ?[]f32 = if (std.c.getenv("EXL3_BENCH_PRIOR")) |pp| blk: {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(pp), a, .limited(c.layers * E * 4 + 1));
        if (bytes.len != c.layers * E * 4) return error.PriorShape;
        const all = try a.alloc(f32, c.layers * E);
        @memcpy(std.mem.sliceAsBytes(all), bytes);
        break :blk all[layer * E .. (layer + 1) * E];
    } else null;
    if (prior_w != null) for (perm, 0..) |*v, i| {
        v.* = i;
    };
    for (0..E) |i| {
        total += if (prior_w) |pw| @max(@as(f64, pw[i]), 0) else if (skew > 0) 1.0 / std.math.pow(f64, @floatFromInt(i + 1), skew) else 1.0;
        cum[i] = total;
    }
    const picks = try a.alloc(usize, pairs);
    for (0..T) |t| {
        var j: usize = 0;
        while (j < topk) {
            const u = r.float(f64) * total;
            var lo: usize = 0;
            var hi: usize = E - 1;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (cum[mid] < u) lo = mid + 1 else hi = mid;
            }
            const e = perm[lo];
            const dup = for (picks[t * topk .. t * topk + j]) |q| {
                if (q == e) break true;
            } else false;
            if (dup) continue;
            picks[t * topk + j] = e;
            j += 1;
        }
    }
    const per_e = try a.alloc(std.ArrayList(i32), E);
    for (per_e) |*l| l.* = .empty;
    for (picks, 0..) |e, p| try per_e[e].append(a, @intCast(p));
    var items_l: std.ArrayList(i32) = .empty;
    var members_l: std.ArrayList(i32) = .empty;
    var nitems: usize = 0;
    var maxc: usize = 0;
    for (per_e, 0..) |l, e| {
        maxc = @max(maxc, l.items.len);
        var at: usize = 0;
        while (at < l.items.len) : (at += ink.kern.prompt_tile) {
            const n = @min(ink.kern.prompt_tile, l.items.len - at);
            try items_l.appendSlice(a, &.{ @intCast(e), @intCast(members_l.items.len), @intCast(n) });
            try members_l.appendSlice(a, l.items[at .. at + n]);
            nitems += 1;
        }
    }
    // e-prefill: the same pairs in items of up to 128 (the v2 builds of 128 rows a weight tile)
    var items_l128: std.ArrayList(i32) = .empty;
    var members_l128: std.ArrayList(i32) = .empty;
    var nitems128: usize = 0;
    var hist: [6]usize = @splat(0); // items of the 64-pair plan by members: 1-16, 17-32, 33-48, 49-63, 64; experts unused
    for (per_e, 0..) |l, e| {
        if (l.items.len == 0) hist[5] += 1;
        var at: usize = 0;
        while (at < l.items.len) : (at += 128) {
            const n = @min(128, l.items.len - at);
            try items_l128.appendSlice(a, &.{ @intCast(e), @intCast(members_l128.items.len), @intCast(n) });
            try members_l128.appendSlice(a, l.items[at .. at + n]);
            nitems128 += 1;
        }
        at = 0;
        while (at < l.items.len) : (at += ink.kern.prompt_tile) {
            const n = @min(ink.kern.prompt_tile, l.items.len - at);
            hist[if (n == 64) 4 else (n - 1) / 16] += 1;
        }
    }
    var items128 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(items_l128.items));
    defer items128.free();
    const counts128_h = [4]i32{ @intCast(nitems128), 0, 0, 0 };
    var counts128 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&counts128_h));
    defer counts128.free();
    var members128 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(members_l128.items));
    defer members128.free();
    const plan128: cuda.grouped.Plan = .{ .members = members128.ptr, .items = items128.ptr, .counts = counts128.ptr, .rank = 0, .hist = 0 };
    // the same pairs as a decode plan: items of up to decode_tile (the forward's one-row form)
    var items_dl: std.ArrayList(i32) = .empty;
    var members_dl: std.ArrayList(i32) = .empty;
    var nitems_d: usize = 0;
    for (per_e, 0..) |l, e| {
        var at: usize = 0;
        while (at < l.items.len) : (at += ink.kern.decode_tile) {
            const n = @min(ink.kern.decode_tile, l.items.len - at);
            try items_dl.appendSlice(a, &.{ @intCast(e), @intCast(members_dl.items.len), @intCast(n) });
            try members_dl.appendSlice(a, l.items[at .. at + n]);
            nitems_d += 1;
        }
    }
    var items_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(items_dl.items));
    defer items_d.free();
    const counts_dh = [4]i32{ @intCast(nitems_d), 0, 0, 0 };
    var counts_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&counts_dh));
    defer counts_d.free();
    var members_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(members_dl.items));
    defer members_d.free();
    const plan_d: cuda.grouped.Plan = .{ .members = members_d.ptr, .items = items_d.ptr, .counts = counts_d.ptr, .rank = 0, .hist = 0 };
    var items = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(items_l.items));
    defer items.free();
    const counts_h = [4]i32{ @intCast(nitems), 0, 0, 0 };
    var counts = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&counts_h));
    defer counts.free();
    var members = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(members_l.items));
    defer members.free();
    const plan: cuda.grouped.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = 0, .hist = 0 };
    std.debug.print("exl3-bench: {d} rows x {d} = {d} pairs, skew {d}{s}: {d} items (tile {d}), most pairs an expert {d}; items by members 1-16 {d}, 17-32 {d}, 33-48 {d}, 49-63 {d}, 64 {d}; experts unused {d}; {d} items of up to 128\n", .{ T, topk, pairs, skew, if (prior_w != null) " (EXL3_BENCH_PRIOR)" else "", nitems, ink.kern.prompt_tile, maxc, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], nitems128 });
    // e-prefill: EXL3_BENCH_FAST=1 skips the decode forms and the engine's grouped kernel (the prompt builds only)
    const fast = std.c.getenv("EXL3_BENCH_FAST") != null;
    // EXL3_BENCH_CFGS=a,b,...: only these prompt builds (kern.exl3PromptConfig names)
    const cfg_filter: ?[]const u8 = if (std.c.getenv("EXL3_BENCH_CFGS")) |v| std.mem.span(v) else null;

    // inputs
    const xh = try a.alloc(u16, T * H);
    for (xh) |*v| v.* = bf16(r.floatNorm(f32));
    const x13h = try a.alloc(u16, pairs * H);
    for (x13h) |*v| v.* = f16bits(r.floatNorm(f32));
    const x2h = try a.alloc(u16, pairs * Il);
    for (x2h) |*v| v.* = f16bits(r.floatNorm(f32));
    var x = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(xh));
    defer x.free();
    var xr13 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x13h));
    defer xr13.free();
    var xr2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x2h));
    defer xr2.free();
    var xq = try cuda.DeviceBuffer.alloc(d, pairs * H * 2);
    defer xq.free();
    var xrot = try cuda.DeviceBuffer.alloc(d, pairs * H * 2);
    defer xrot.free();
    var act = try cuda.DeviceBuffer.alloc(d, pairs * Il * 2);
    defer act.free();
    var z_ref = try cuda.DeviceBuffer.alloc(d, pairs * 2 * Il * 4);
    defer z_ref.free();
    var z = try cuda.DeviceBuffer.alloc(d, pairs * 2 * Il * 4);
    defer z.free();
    var y_ref = try cuda.DeviceBuffer.alloc(d, pairs * H * 4);
    defer y_ref.free();
    var y = try cuda.DeviceBuffer.alloc(d, pairs * H * 4);
    defer y.free();
    const zh_ref = try a.alloc(u8, pairs * 2 * Il * 4);
    const zh = try a.alloc(u8, pairs * 2 * Il * 4);
    const yh_ref = try a.alloc(u8, pairs * H * 4);
    const yh = try a.alloc(u8, pairs * H * 4);

    var tm: Timer = .{ .t0 = try cuda.Event.init(d, true), .t1 = try cuda.Event.init(d, true) };
    // the reference: the decode form (g1), once, into poisoned buffers (every output must be written)
    for ([_]*cuda.DeviceBuffer{ &z_ref, &y_ref, &z, &y }) |buf| try buf.fill8(0xff, s.handle);
    try k.exl3Gemm(s, ex, false, false, xr13.ptr, plan, nitems, z_ref.ptr, 2 * Il, 0, 2 * Il, 1, 0);
    try k.exl3Gemm(s, ex, true, false, xr2.ptr, plan, nitems, y_ref.ptr, H, 0, H, 1, 0);
    try s.synchronize();
    try z_ref.download(0, zh_ref);
    try y_ref.download(0, yh_ref);
    std.debug.print("exl3-bench: g1 reference: {s}; {s}\n", .{ try diff(a, "w13", zh_ref, zh_ref), try diff(a, "w2", yh_ref, yh_ref) });

    // the GEMM forms: time, then bits against g1's
    const forms = [_]struct { name: []const u8, prompt: bool }{ .{ .name = "g1 (decode form)", .prompt = false }, .{ .name = "g2 (prompt form)", .prompt = true } };
    var w13_flops: f64 = @floatFromInt(pairs);
    w13_flops *= 2.0 * @as(f64, @floatFromInt(H * 2 * Il));
    var w2_flops: f64 = @floatFromInt(pairs);
    w2_flops *= 2.0 * @as(f64, @floatFromInt(Il * H));
    for (forms) |f| {
        if (fast) break;
        try z.fill8(0xff, s.handle);
        try y.fill8(0xff, s.handle);
        try k.exl3Gemm(s, ex, false, f.prompt, xr13.ptr, plan, nitems, z.ptr, 2 * Il, 0, 2 * Il, 1, 0); // warm
        try tm.t0.record(s);
        for (0..reps) |_| try k.exl3Gemm(s, ex, false, f.prompt, xr13.ptr, plan, nitems, z.ptr, 2 * Il, 0, 2 * Il, 1, 0);
        const ms13 = try tm.ms(s, reps);
        try k.exl3Gemm(s, ex, true, f.prompt, xr2.ptr, plan, nitems, y.ptr, H, 0, H, 1, 0);
        try tm.t0.record(s);
        for (0..reps) |_| try k.exl3Gemm(s, ex, true, f.prompt, xr2.ptr, plan, nitems, y.ptr, H, 0, H, 1, 0);
        const ms2 = try tm.ms(s, reps);
        try z.download(0, zh);
        try y.download(0, yh);
        const same13 = std.mem.eql(u8, zh, zh_ref);
        const same2 = std.mem.eql(u8, yh, yh_ref);
        std.debug.print("  {s}: w13 {d:.3} ms ({d:.1} TFLOP/s), w2 {d:.3} ms ({d:.1} TFLOP/s); bits vs g1: w13 {s}, w2 {s}\n", .{ f.name, ms13, w13_flops / ms13 / 1e9, ms2, w2_flops / ms2 / 1e9, if (same13) "SAME" else "DIFFER", if (same2) "SAME" else "DIFFER" });
    }
    // the engine's grouped kernel (tf_exl3_grouped_2, kern.exl3Grouped) against the forms the forward runs it in place
    // of: g1 on the decode plan (one-row steps), g2 on the prompt plan (2 .. 383 rows); bits against g1's, outputs poisoned
    {
        const Case = struct { name: []const u8, plan: cuda.grouped.Plan, items: usize, tile: usize, prompt: bool };
        const cases = [_]Case{
            .{ .name = "decode plan (items of 16)", .plan = plan_d, .items = nitems_d, .tile = ink.kern.decode_tile, .prompt = false },
            .{ .name = "prompt plan (items of 64)", .plan = plan, .items = nitems, .tile = ink.kern.prompt_tile, .prompt = true },
        };
        for (cases) |cs| {
            if (fast) break;
            try tm.t0.record(s);
            for (0..reps) |_| try k.exl3Gemm(s, ex, false, cs.prompt, xr13.ptr, cs.plan, cs.items, z.ptr, 2 * Il, 0, 2 * Il, 1, 0);
            const ms13_f = try tm.ms(s, reps);
            try tm.t0.record(s);
            for (0..reps) |_| try k.exl3Gemm(s, ex, true, cs.prompt, xr2.ptr, cs.plan, cs.items, y.ptr, H, 0, H, 1, 0);
            const ms2_f = try tm.ms(s, reps);
            try z.fill8(0xff, s.handle);
            try y.fill8(0xff, s.handle);
            try k.exl3Grouped(s, ex, false, xr13.ptr, cs.plan, cs.items, cs.tile, pairs, z.ptr, 1);
            try k.exl3Grouped(s, ex, true, xr2.ptr, cs.plan, cs.items, cs.tile, pairs, y.ptr, 1);
            try s.synchronize();
            try z.download(0, zh);
            try y.download(0, yh);
            const same13 = std.mem.eql(u8, zh, zh_ref);
            const same2 = std.mem.eql(u8, yh, yh_ref);
            try tm.t0.record(s);
            for (0..reps) |_| try k.exl3Grouped(s, ex, false, xr13.ptr, cs.plan, cs.items, cs.tile, pairs, z.ptr, 1);
            const ms13_c = try tm.ms(s, reps);
            try tm.t0.record(s);
            for (0..reps) |_| try k.exl3Grouped(s, ex, true, xr2.ptr, cs.plan, cs.items, cs.tile, pairs, y.ptr, 1);
            const ms2_c = try tm.ms(s, reps);
            std.debug.print("  core grouped, {s}: w13 {d:.3} ms (the family's {s} {d:.3}), w2 {d:.3} ms ({d:.3}); bits vs g1: w13 {s}, w2 {s}\n", .{ cs.name, ms13_c, if (cs.prompt) "g2" else "g1", ms13_f, ms2_c, ms2_f, if (same13) "SAME" else "DIFFER", if (same2) "SAME" else "DIFFER" });
            if (!same13 or !same2) std.debug.print("      {s}; {s}\n", .{ try diff(a, "w13", zh, zh_ref), try diff(a, "w2", yh, yh_ref) });
        }
        if (world == 1 and !fast) {
            // canonical TP: w2 in two K splits (the forward's world-1 down), raw sums in two planes
            var zd_ref = try cuda.DeviceBuffer.alloc(d, 2 * pairs * H * 4);
            defer zd_ref.free();
            var zd = try cuda.DeviceBuffer.alloc(d, 2 * pairs * H * 4);
            defer zd.free();
            try zd.fill8(0xff, s.handle);
            try k.exl3Gemm(s, ex, true, true, xr2.ptr, plan, nitems, zd_ref.ptr, H, 0, H, 2, pairs * H);
            try k.exl3Grouped(s, ex, true, xr2.ptr, plan, nitems, ink.kern.prompt_tile, pairs, zd.ptr, 2);
            try s.synchronize();
            const h_ref = try a.alloc(u8, 2 * pairs * H * 4);
            const h = try a.alloc(u8, 2 * pairs * H * 4);
            try zd_ref.download(0, h_ref);
            try zd.download(0, h);
            std.debug.print("  core grouped, w2 in 2 K splits (canonical TP): bits vs g2's 2 parts: {s}\n", .{if (std.mem.eql(u8, h, h_ref)) "SAME" else "DIFFER"});
        }
    }

    // the wide-chunk path on the engine's prompt hooks (kern.exl3_prompt_configs): w13 + its epilogue -> act, w2 + its
    // epilogue -> y, against g1 then the epilogues (the forward's decode path); y also in 6 column parts (moeOverlap)
    {
        var act_ref = try cuda.DeviceBuffer.alloc(d, pairs * Il * 2);
        defer act_ref.free();
        var act_p = try cuda.DeviceBuffer.alloc(d, pairs * Il * 2);
        defer act_p.free();
        var ye_ref = try cuda.DeviceBuffer.alloc(d, pairs * H * 4);
        defer ye_ref.free();
        try act_ref.fill8(0xff, s.handle);
        try ye_ref.fill8(0xff, s.handle);
        try k.exl3GateUpEpi(s, ex, z_ref.ptr, 2 * Il, plan, nitems, act_ref.ptr);
        try k.exl3DownEpi(s, ex, ye_ref.ptr, H, 0, H, plan, nitems, y_ref.ptr, 1, 0);
        try s.synchronize();
        const ah_ref = try a.alloc(u8, pairs * Il * 2);
        const ah = try a.alloc(u8, pairs * Il * 2);
        try act_ref.download(0, ah_ref);
        try ye_ref.download(0, yh_ref); // y_ref's host copy now holds the epilogue's y
        // the current prompt path's time: g2 then each epilogue
        try tm.t0.record(s);
        for (0..reps) |_| {
            try k.exl3Gemm(s, ex, false, true, xr13.ptr, plan, nitems, z.ptr, 2 * Il, 0, 2 * Il, 1, 0);
            try k.exl3GateUpEpi(s, ex, z.ptr, 2 * Il, plan, nitems, act_p.ptr);
        }
        const ms13_old = try tm.ms(s, reps);
        try tm.t0.record(s);
        for (0..reps) |_| {
            try k.exl3Gemm(s, ex, true, true, xr2.ptr, plan, nitems, y.ptr, H, 0, H, 1, 0);
            try k.exl3DownEpi(s, ex, y.ptr, H, 0, H, plan, nitems, y.ptr, 1, 0);
        }
        const ms2_old = try tm.ms(s, reps);
        std.debug.print("  g2 + epilogues (the current prompt path): w13 {d:.3} ms, w2 {d:.3} ms\n", .{ ms13_old, ms2_old });
        for (ink.kern.exl3_prompt_configs, 0..) |cfg, v| {
            const sfx = cfg.name orelse cfg.sfx[1..];
            if (cfg_filter) |f| {
                var it = std.mem.tokenizeScalar(u8, f, ',');
                const want = while (it.next()) |n| {
                    if (std.mem.eql(u8, n, sfx)) break true;
                } else false;
                if (!want) continue;
            }
            const tile = ink.kern.Kernels.exl3PromptTile(v);
            const pl = if (tile == 128) plan128 else plan;
            const ni = if (tile == 128) nitems128 else nitems;
            try act_p.fill8(0xff, s.handle);
            try y.fill8(0xff, s.handle);
            try k.exl3PromptGu(s, ex, v, xr13.ptr, pl, ni, tile, act_p.ptr);
            try k.exl3PromptDown(s, ex, v, xr2.ptr, pl, ni, tile, y.ptr, H, 0, H, 1);
            try s.synchronize();
            try act_p.download(0, ah);
            try y.download(0, yh);
            const same_act = std.mem.eql(u8, ah, ah_ref);
            const same_y = std.mem.eql(u8, yh, yh_ref);
            // the down in 6 column parts, as moeOverlap launches it
            try y.fill8(0xff, s.handle);
            const part = H / 6;
            for (0..6) |q| try k.exl3PromptDown(s, ex, v, xr2.ptr, pl, ni, tile, y.ptr + q * part * 4, H, q * part, part, 1);
            try s.synchronize();
            try y.download(0, yh);
            const same_parts = std.mem.eql(u8, yh, yh_ref);
            try tm.t0.record(s);
            for (0..reps) |_| try k.exl3PromptGu(s, ex, v, xr13.ptr, pl, ni, tile, act_p.ptr);
            const ms13 = try tm.ms(s, reps);
            try tm.t0.record(s);
            for (0..reps) |_| try k.exl3PromptDown(s, ex, v, xr2.ptr, pl, ni, tile, y.ptr, H, 0, H, 1);
            const ms2 = try tm.ms(s, reps);
            std.debug.print("  prompt {s:<14} (tile {d:>3}): w13+epi {d:7.3} ms ({d:5.1} TFLOP/s), w2+epi {d:7.3} ms ({d:5.1} TFLOP/s), sum {d:7.3} ms; bits vs g1+epilogues: act {s}, y {s}, y in 6 parts {s}\n", .{ sfx, tile, ms13, w13_flops / ms13 / 1e9, ms2, w2_flops / ms2 / 1e9, ms13 + ms2, if (same_act) "SAME" else "DIFFER", if (same_y) "SAME" else "DIFFER", if (same_parts) "SAME" else "DIFFER" });
            if (!same_act) std.debug.print("      act: {s}\n", .{try diffBytes(a, ah, ah_ref)});
            if (!same_y or !same_parts) std.debug.print("      {s}\n", .{try diff(a, "y", yh, yh_ref)});
        }
        if (world == 1) {
            // canonical TP at world 1: w2 in the two K parts a world-2 pair computes, added in rank order
            var zd = try cuda.DeviceBuffer.alloc(d, 2 * pairs * H * 4);
            defer zd.free();
            try ye_ref.fill8(0xff, s.handle);
            try k.exl3Gemm(s, ex, true, false, xr2.ptr, plan, nitems, zd.ptr, H, 0, H, 2, pairs * H);
            try k.exl3DownEpi(s, ex, ye_ref.ptr, H, 0, H, plan, nitems, zd.ptr, 2, pairs * H);
            try s.synchronize();
            try ye_ref.download(0, yh_ref);
            for (ink.kern.exl3_prompt_configs, 0..) |cfg, v| {
                const tile = ink.kern.Kernels.exl3PromptTile(v);
                try y.fill8(0xff, s.handle);
                try k.exl3PromptDown(s, ex, v, xr2.ptr, if (tile == 128) plan128 else plan, if (tile == 128) nitems128 else nitems, tile, y.ptr, H, 0, H, 2);
                try s.synchronize();
                try y.download(0, yh);
                std.debug.print("  prompt {s}, 2 K parts (canonical TP): bits vs g1 + the 2-part epilogue: y {s}\n", .{ cfg.name orelse cfg.sfx[1..], if (std.mem.eql(u8, yh, yh_ref)) "SAME" else "DIFFER" });
            }
        }
        try y_ref.download(0, yh_ref); // back to the raw sums for anything after
    }

    // the rest of a layer's expert work (the forward's order): w13 activation of the rows, its rotation per pair, the
    // gate/up epilogue, w2's activation per pair, its rotation, the down epilogue
    const mode: u8 = if (layer < 40) 1 else 2;
    const Step = struct { name: []const u8, id: u8 };
    const steps = [_]Step{ .{ .name = "act13", .id = 0 }, .{ .name = "rot13", .id = 1 }, .{ .name = "gateup_epi", .id = 2 }, .{ .name = "act2", .id = 3 }, .{ .name = "rot2", .id = 4 }, .{ .name = "down_epi", .id = 5 } };
    for (steps) |st| {
        const runs = reps + 1;
        for (0..runs) |rr| {
            if (rr == 1) try tm.t0.record(s);
            switch (st.id) {
                0 => try k.exl3Act(s, x.ptr, H, T, H, mode, 0, 1, 0, xq.ptr, H),
                1 => try k.exl3Rot(s, ex, xq.ptr, H, topk, plan, nitems, false, xrot.ptr, H),
                2 => try k.exl3GateUpEpi(s, ex, z.ptr, 2 * Il, plan, nitems, act.ptr),
                3 => try k.exl3Act(s, act.ptr, Il, pairs, Il, mode, 0, 1, 0, xq.ptr, Il),
                4 => try k.exl3Rot(s, ex, xq.ptr, Il, 0, plan, nitems, true, xrot.ptr, Il),
                5 => try k.exl3DownEpi(s, ex, y.ptr, H, 0, H, plan, nitems, y_ref.ptr, 1, 0),
                else => unreachable,
            }
        }
        std.debug.print("  {s}: {d:.3} ms\n", .{ st.name, try tm.ms(s, reps) });
    }

    // e-prefill: the fused activation + rotation (exl3ActRot13 / exl3ActRot2) against exl3Act then exl3Rot, every mode,
    // bits and time; the bench's slots are the experts themselves (lut = identity)
    {
        const picks_h = try a.alloc(i32, pairs);
        for (picks, 0..) |e, i| picks_h[i] = @intCast(e);
        var picks_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(picks_h));
        defer picks_d.free();
        const lut_h = try a.alloc(i32, E);
        for (lut_h, 0..) |*v, i| v.* = @intCast(i);
        var lut_d = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(lut_h));
        defer lut_d.free();
        // w2's input rows: random bf16 (the real act's scale: N(0, 1))
        const a2h = try a.alloc(u16, pairs * Il);
        for (a2h) |*v| v.* = bf16(r.floatNorm(f32));
        var a2 = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(a2h));
        defer a2.free();
        // a world-2 amax table: [2][pairs], each entry the pair's max over its own row times 0.5 .. 1.5 (rank 1's part)
        var amax2 = try cuda.DeviceBuffer.alloc(d, 2 * pairs * 4);
        defer amax2.free();
        try k.exl3RowAmax(s, a2.ptr, Il, pairs, Il, amax2.ptr);
        try s.synchronize();
        {
            const am = try a.alloc(f32, 2 * pairs);
            try amax2.download(0, std.mem.sliceAsBytes(am[0..pairs]));
            for (0..pairs) |i| am[pairs + i] = am[i] * (0.5 + r.float(f32));
            try amax2.upload(0, std.mem.sliceAsBytes(am));
        }
        var xr_new = try cuda.DeviceBuffer.alloc(d, pairs * H * 2);
        defer xr_new.free();
        const href = try a.alloc(u8, pairs * H * 2);
        const hnew = try a.alloc(u8, pairs * H * 2);
        for ([_]u8{ 0, 1, 2 }) |md| {
            // w13: the tokens' rows
            try xrot.fill8(0xff, s.handle);
            try xr_new.fill8(0xff, s.handle);
            try k.exl3Act(s, x.ptr, H, T, H, md, 0, 1, 0, xq.ptr, H);
            try k.exl3Rot(s, ex, xq.ptr, H, topk, plan, nitems, false, xrot.ptr, H);
            try k.exl3ActRot13(s, ex, x.ptr, H, T, H, md, picks_d.ptr, topk, lut_d.ptr, xr_new.ptr);
            try s.synchronize();
            try xrot.download(0, href);
            try xr_new.download(0, hnew);
            const same13 = std.mem.eql(u8, href, hnew);
            try tm.t0.record(s);
            for (0..reps) |_| {
                try k.exl3Act(s, x.ptr, H, T, H, md, 0, 1, 0, xq.ptr, H);
                try k.exl3Rot(s, ex, xq.ptr, H, topk, plan, nitems, false, xrot.ptr, H);
            }
            const ms_old13 = try tm.ms(s, reps);
            try tm.t0.record(s);
            for (0..reps) |_| try k.exl3ActRot13(s, ex, x.ptr, H, T, H, md, picks_d.ptr, topk, lut_d.ptr, xr_new.ptr);
            const ms_new13 = try tm.ms(s, reps);
            // w2: the pairs' rows, world 1 (own amax: row_amax first for the fused form) and the world-2 table
            const n2 = pairs * Il * 2;
            var same2 = [2]bool{ true, true };
            var ms2 = [4]f64{ 0, 0, 0, 0 };
            for (0..2) |wv| {
                if (md != 1 and wv == 1) continue;
                const amax_old: u64 = if (wv == 1) amax2.ptr else 0;
                try xrot.fill8(0xff, s.handle);
                try xr_new.fill8(0xff, s.handle);
                try k.exl3Act(s, a2.ptr, Il, pairs, Il, md, amax_old, 2, pairs, xq.ptr, Il);
                try k.exl3Rot(s, ex, xq.ptr, Il, 0, plan, nitems, true, xrot.ptr, Il);
                var amax_new: u64 = amax2.ptr;
                var wn: usize = 2;
                if (md == 1 and wv == 0) {
                    try k.exl3RowAmax(s, a2.ptr, Il, pairs, Il, amax2.ptr + 2 * pairs * 4 - pairs * 4); // rank 1's half: scratch
                    amax_new = amax2.ptr + pairs * 4;
                    wn = 1;
                }
                try k.exl3ActRot2(s, ex, a2.ptr, Il, md, amax_new, wn, pairs, plan, nitems, xr_new.ptr, Il);
                try s.synchronize();
                try xrot.download(0, href[0..n2]);
                try xr_new.download(0, hnew[0..n2]);
                same2[wv] = std.mem.eql(u8, href[0..n2], hnew[0..n2]);
                if (wv == 0) {
                    try tm.t0.record(s);
                    for (0..reps) |_| {
                        try k.exl3Act(s, a2.ptr, Il, pairs, Il, md, 0, 2, pairs, xq.ptr, Il);
                        try k.exl3Rot(s, ex, xq.ptr, Il, 0, plan, nitems, true, xrot.ptr, Il);
                    }
                    ms2[0] = try tm.ms(s, reps);
                    try tm.t0.record(s);
                    for (0..reps) |_| {
                        if (md == 1) try k.exl3RowAmax(s, a2.ptr, Il, pairs, Il, amax2.ptr + pairs * 4);
                        try k.exl3ActRot2(s, ex, a2.ptr, Il, md, amax_new, wn, pairs, plan, nitems, xr_new.ptr, Il);
                    }
                    ms2[1] = try tm.ms(s, reps);
                }
                // world 2's amax table restored for the next mode (the world-1 check wrote rank 1's half)
                if (md == 1 and wv == 0) {
                    const am = try a.alloc(f32, pairs);
                    try amax2.download(0, std.mem.sliceAsBytes(am));
                    for (am) |*v| v.* *= 0.5 + r.float(f32);
                    try amax2.upload(pairs * 4, std.mem.sliceAsBytes(am));
                }
            }
            std.debug.print("  actrot mode {d}: w13 act+rot {d:.3} ms -> fused {d:.3} ms, bits {s}; w2 act+rot {d:.3} ms -> fused {d:.3} ms (world 1 incl. row amax), bits world 1 {s}, world 2 {s}\n", .{ md, ms_old13, ms_new13, if (same13) "SAME" else "DIFFER", ms2[0], ms2[1], if (same2[0]) "SAME" else "DIFFER", if (md != 1) "n/a" else if (same2[1]) "SAME" else "DIFFER" });
            if (!same13) std.debug.print("      w13: {s}\n", .{try diffBytes(a, href, hnew)});
        }
    }
    return 0;
}
