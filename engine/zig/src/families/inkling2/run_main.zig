//! tf-inkling2-run MODEL_DIR RANK WORLD --ids 1,2,3 [options]: one TP rank of Inkling-2 on this GPU (world > 1:
//! NCCL_COMM_ID names rank 0, every rank runs the same command) with TideLM's residency: the arena sized from the
//! memory left above a floor, filled from the prior, the masked router (full substitution) and adaptive passes.
//!
//!   --ids P             the prompt's token ids (comma separated)
//!   --tokens N          greedy output tokens (default 8)
//!   --force F           teacher-forced continuation ids instead of greedy: fed after the prompt, every row's logits
//!   --force-decode 1    the forced ids as serving runs a reply: the prompt prefilled, the prefill->decode residency
//!                       pass, then one decode step a forced id (and the request-end pass); default 0: all in chunks
//!   --rows R            rows a chunk (default 16)            --context C   positions (default 1024)
//!   --prefill-rows P    rows a prompt chunk of the sub arm (default R; exact arms keep R): the prompt's scratch made
//!                       for P rows after the arena is filled (Model.growPrompt), the arena's size unchanged
//!   --prior FILE        prior scores fp32 [L][E] (default: every expert equal)
//!   --floor-gib G       MemAvailable kept free after the arena (default 8)
//!   --per-layer N       resident experts per layer (the uniform quota) instead of the floor's budget
//!   --transient T       exact mode's ring, slots a format (default top-k x rows, at most E)
//!   --arms A,B          arms run in turn on one arena: sub (masked router, full substitution), exact (exact
//!                       streaming: nothing substituted). Default sub
//!   --adapt 0|1         residency passes (default 1; 0 freezes the placement)
//!   --workers W         experts packed at once (default 16)  --readers R   O_DIRECT read threads (default 8)
//!   --out PREFIX        rank 0 writes PREFIX.<arm>.logits (fp32 [rows][unpadded], row i the logits after token i)
//!                       and PREFIX.<arm>.tokens (the output ids, int32); an arm kind named more than once in --arms
//!                       writes PREFIX.<arm><i>.* (i its position in --arms, from 0: TF_INKLING2_ARM_SWITCHES' A/B)
//!   --ids2 Q            e-prefill: a second prompt, for the arms whose TF_INKLING2_ARM_SWITCHES spec says ids=2 (no
//!                       --out, no --force: their rows' logits are not kept)
//!   e-dense (prefill lane phase 5):
//!   --dense-sets S      `name:classes:PACK_DIR[:gemm|tiles|blocks][:dtiles];...` dense EXL3 sets loaded after the
//!                       store / arena (classes attn, router, head, all; `none` = the bf16 weights; wide chunks:
//!                       dense_exl3.Wide, default gemm; dtiles: decode rows on the candidate kernels); every arm runs
//!                       once a set, in order, its lines and files tagged <arm>-<set>
//!   --ids-files F       `name=FILE,...` prompts (comma-separated ids a file) for the logits path: with --out, each
//!                       set's (else the active weights') every row's logits as PREFIX.<name>.<set>.logits through the
//!                       batched head, the sub arm in --prefill-rows chunks, the exact arm in --rows chunks (written a
//!                       head block at a time)
//!   --head-block B      rows a head product in that path (default 128)
//!   --prefill-bench N   `name,...` of --ids-files: each set's prefill of those prompts timed (no logits), twice
//!   --bench-dense 1     layer 0's projections, its router and the head at 1-16 rows: every set's EXL3 weight and the
//!                       bf16 one (cuBLASLt), each timed over 100 calls with CUDA events (us a call, GB/s of its bytes)
//!   --hcap DIR          the calibration Hessians of the dense linears over --ids-files (each from position 0, in
//!                       --prefill-rows chunks): DIR/L<l>-<input>.f32 and head.f32 on each rank (hcap.zig), then exit
//!   --hcap-layers A-B   layers [A, B) only (default all; the ranks take a half each)
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-run MODEL_DIR RANK WORLD --ids 1,2,3 [--tokens N] [--force IDS] [--rows R] [--prefill-rows P] [--context C] [--prior FILE] [--floor-gib G] [--per-layer N] [--transient T] [--arms sub,exact] [--adapt 0|1] [--workers W] [--readers R] [--out PREFIX]\n";

const Arm = enum { sub, exact };

/// e-dense: a --dense-sets entry (classes none: the bf16 weights) and its index among the model's loaded sets.
const DenseSet = struct { name: []const u8, classes: ink.dense_exl3.Classes, dir: []const u8, idx: ?usize = null, wide: ink.dense_exl3.Wide = .gemm, dtiles: bool = false };
/// e-dense: a --ids-files entry.
const IdsFile = struct { name: []const u8, path: []const u8 };

/// e-dense: the comma-separated ids of a file.
fn readIds(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]i32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    var out: std.ArrayList(i32) = .empty;
    try parseIds(a, std.mem.trim(u8, text, " \n\r\t"), &out);
    return out.items;
}

/// e-dense: the set `idx` active (null: the bf16 weights), graphs dropped (they hold the old weights).
fn activateSet(m: *ink.model.Model, idx: ?usize) !void {
    m.graphs.drop(m.gpa);
    if (m.dx) |dx| try dx.activate(m.s, idx);
    try m.s.synchronize();
}

fn ms(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

fn parseIds(a: std.mem.Allocator, v: []const u8, out: *std.ArrayList(i32)) !void {
    var it = std.mem.tokenizeScalar(u8, v, ',');
    while (it.next()) |t| try out.append(a, try std.fmt.parseInt(i32, t, 10));
}

const gib: f64 = 1024.0 * 1024.0 * 1024.0;

/// TF_INKLING2_PROFILE_PREFILL=1: the sub arm's prompt chunks between cuProfilerStart and cuProfilerStop (an nsys
/// --capture-range=cudaProfilerApi window around the prefill alone); e-step (e-decode's modes): =3 the decode loop and
/// the benches after it (--bench-real, --bench), =4 the benches alone; otherwise nothing.
const ProfRange = struct {
    start: ?*const fn () callconv(.c) c_int = null,
    stop: ?*const fn () callconv(.c) c_int = null,
    decode: bool = false, // e-step: the window opens at the decode loop (3)
    benches: bool = false, // e-step: the window closes after the benches (3, 4)
    bench_only: bool = false, // e-step: the window opens at the benches (4)

    fn init() ProfRange {
        const v = std.c.getenv("TF_INKLING2_PROFILE_PREFILL") orelse return .{};
        const sv = std.mem.span(v);
        const mode: u8 = if (sv.len == 1 and (sv[0] == '1' or sv[0] == '3' or sv[0] == '4')) sv[0] - '0' else return .{};
        var lib = std.DynLib.open("libcuda.so.1") catch return .{}; // already loaded: the driver's own handle
        return .{ .start = lib.lookup(*const fn () callconv(.c) c_int, "cuProfilerStart"), .stop = lib.lookup(*const fn () callconv(.c) c_int, "cuProfilerStop"), .decode = mode == 3, .benches = mode >= 3, .bench_only = mode == 4 };
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const dir = args[1];
    const rank = try std.fmt.parseInt(usize, args[2], 10);
    const world = try std.fmt.parseInt(usize, args[3], 10);
    var ids: std.ArrayList(i32) = .empty;
    var ids_alt: std.ArrayList(i32) = .empty; // e-prefill: --ids2, the arms with ids=2
    var force: std.ArrayList(i32) = .empty;
    var force_decode = false;
    var bench: std.ArrayList(usize) = .empty;
    var spec_depth: usize = 0; // --spec D: DSpark speculation, D drafted tokens a verify (1..7) // --bench 1,2,4,8: ms a forward of T rows after the prompt
    var tokens: usize = 8;
    var context: usize = 1024;
    var rows: usize = 16;
    var prefill_rows: ?usize = null;
    var out: ?[]const u8 = null;
    var prior_path: ?[]const u8 = null;
    var floor_gib: f64 = 8;
    var per_layer: ?usize = null;
    var transient: ?usize = null;
    var arms: std.ArrayList(Arm) = .empty;
    var adapt_on = true;
    var workers: usize = 16;
    var readers: usize = 8;
    var prof: usize = 0; // --prof 1: the spec round's parts timed (stream synced); 2: also distinct experts a forward
    var bench_real = false; // --bench-real 1: the bench widths over the spec loop's own verify windows
    var prewarm = true; // --prewarm 0: no load-time warm-up (every width's plans made at first use)
    var bank_gib: ?f64 = null; // x-mixed: --bank-gib G, the native bank's slot bytes at most (TF_INKLING2_MIX=1)
    var bank_per_layer: ?usize = null; // x-mixed: --bank-per-layer N, the bank as N experts a layer in each pool
    var dense_sets: std.ArrayList(DenseSet) = .empty; // e-dense
    var ids_files: std.ArrayList(IdsFile) = .empty;
    var head_block: usize = 128;
    var hcap_dir: ?[]const u8 = null;
    var hcap_lo: usize = 0;
    var hcap_hi: ?usize = null;
    var prefill_bench: std.ArrayList([]const u8) = .empty;
    var bench_dense = false;
    var ai: usize = 4;
    while (ai + 1 < args.len) : (ai += 2) {
        const k = args[ai];
        const v = args[ai + 1];
        if (std.mem.eql(u8, k, "--ids2")) {
            try parseIds(a, v, &ids_alt);
        } else if (std.mem.eql(u8, k, "--ids")) {
            try parseIds(a, v, &ids);
        } else if (std.mem.eql(u8, k, "--ids-file")) {
            // p4-small: the ids from a file (comma separated), for prompts past one argument's 128 KiB (p32k)
            try parseIds(a, std.mem.trim(u8, try std.Io.Dir.cwd().readFileAlloc(io, v, a, .limited(64 << 20)), " \n\r"), &ids);
        } else if (std.mem.eql(u8, k, "--force")) {
            try parseIds(a, v, &force);
        } else if (std.mem.eql(u8, k, "--spec")) {
            spec_depth = try std.fmt.parseInt(usize, v, 10);
            if (spec_depth > ink.dspark.spec) return error.BadSpecDepth;
        } else if (std.mem.eql(u8, k, "--bench")) {
            var it = std.mem.tokenizeScalar(u8, v, ',');
            while (it.next()) |t| try bench.append(a, try std.fmt.parseInt(usize, t, 10));
        } else if (std.mem.eql(u8, k, "--force-decode")) {
            force_decode = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "--tokens")) {
            tokens = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--context")) {
            context = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--rows")) {
            rows = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--prefill-rows")) {
            prefill_rows = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--out")) {
            out = v;
        } else if (std.mem.eql(u8, k, "--prior")) {
            prior_path = v;
        } else if (std.mem.eql(u8, k, "--floor-gib")) {
            floor_gib = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, k, "--per-layer")) {
            per_layer = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--transient")) {
            transient = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--arms")) {
            var it = std.mem.tokenizeScalar(u8, v, ',');
            while (it.next()) |t| try arms.append(a, std.meta.stringToEnum(Arm, t) orelse return error.BadArm);
        } else if (std.mem.eql(u8, k, "--adapt")) {
            adapt_on = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "--workers")) {
            workers = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--readers")) {
            readers = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--prof")) {
            prof = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--prewarm")) {
            prewarm = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "--bench-real")) {
            bench_real = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "--dense-sets")) {
            var it = std.mem.tokenizeScalar(u8, v, ';');
            while (it.next()) |t| {
                var f = std.mem.splitScalar(u8, t, ':');
                const name = f.next() orelse return error.BadDenseSet;
                const cls = try ink.dense_exl3.parseClasses(if (f.next()) |c| (if (std.mem.eql(u8, c, "none")) "" else c) else return error.BadDenseSet);
                var dir_ = f.rest();
                var wide: ink.dense_exl3.Wide = .gemm;
                var dtiles = false;
                while (std.mem.lastIndexOfScalar(u8, dir_, ':')) |c| {
                    const opt = dir_[c + 1 ..];
                    if (std.meta.stringToEnum(ink.dense_exl3.Wide, opt)) |wd| {
                        wide = wd;
                    } else if (std.mem.eql(u8, opt, "dtiles")) {
                        dtiles = true;
                    } else break;
                    dir_ = dir_[0..c];
                }
                try dense_sets.append(a, .{ .name = name, .classes = cls, .dir = dir_, .wide = wide, .dtiles = dtiles });
            }
        } else if (std.mem.eql(u8, k, "--ids-files")) {
            var it = std.mem.tokenizeScalar(u8, v, ',');
            while (it.next()) |t| {
                const eq = std.mem.indexOfScalar(u8, t, '=') orelse return error.BadIdsFile;
                try ids_files.append(a, .{ .name = t[0..eq], .path = t[eq + 1 ..] });
            }
        } else if (std.mem.eql(u8, k, "--prefill-bench")) {
            var it = std.mem.tokenizeScalar(u8, v, ',');
            while (it.next()) |t| try prefill_bench.append(a, t);
        } else if (std.mem.eql(u8, k, "--bench-dense")) {
            bench_dense = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "--head-block")) {
            head_block = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "--hcap")) {
            hcap_dir = v;
        } else if (std.mem.eql(u8, k, "--hcap-layers")) {
            const dash = std.mem.indexOfScalar(u8, v, '-') orelse return error.BadArgument;
            hcap_lo = try std.fmt.parseInt(usize, v[0..dash], 10);
            hcap_hi = try std.fmt.parseInt(usize, v[dash + 1 ..], 10);
        } else if (std.mem.eql(u8, k, "--bank-gib")) { // x-mixed
            bank_gib = try std.fmt.parseFloat(f64, v);
        } else if (std.mem.eql(u8, k, "--bank-per-layer")) { // x-mixed
            bank_per_layer = try std.fmt.parseInt(usize, v, 10);
        } else return error.BadArgument;
    }
    if (ids.items.len == 0 and ids_files.items.len == 0) return error.NoPrompt; // e-dense: or --ids-files
    if (arms.items.len == 0) try arms.append(a, .sub);
    // x-mixed (a measurement aid): TF_INKLING2_ARM_IDS="FILE:FILE:..." gives arm i the prompt of file i (comma separated
    // ids; an empty entry or none: --ids, or the arm's ids=2), so one load measures several prompts; the logits buffer
    // takes the longest
    var arm_prompts: std.ArrayList([]const i32) = .empty;
    if (std.c.getenv("TF_INKLING2_ARM_IDS")) |v| {
        var files = std.mem.splitScalar(u8, std.mem.span(v), ':');
        while (files.next()) |f| {
            if (f.len == 0) {
                try arm_prompts.append(a, ids.items);
                continue;
            }
            var got: std.ArrayList(i32) = .empty;
            try parseIds(a, std.mem.trim(u8, try std.Io.Dir.cwd().readFileAlloc(io, f, a, .limited(64 << 20)), " \n\r"), &got);
            if (got.items.len == 0) return error.NoPrompt;
            try arm_prompts.append(a, got.items);
        }
    }
    var longest: usize = ids.items.len;
    for (arm_prompts.items) |x| longest = @max(longest, x.len);
    var want_exact = false;
    for (arms.items) |x| want_exact = want_exact or x == .exact;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    const group: ?cuda.tp_group.Group = if (world > 1) try cuda.tp_group.Group.open(&driver, rank, world) else null;
    const t0 = std.Io.Clock.awake.now(io);
    const m = try ink.model.Model.open(init.gpa, io, &ctx, dir, rank, world, group, context, rows);
    defer m.close();
    const dense_ms = ms(io, t0);
    if (m.lazy) |lz| lz.floor = @intFromFloat(floor_gib * gib); // p4-mem: what a context growth keeps free
    // x-mixed: the mixed store's loader (its native bank's) reads the release checkpoint with its own config
    if (m.mix and want_exact) return error.MixedStoreHasNoExactArm; // nothing is substituted there: sub is exact
    const loader = try ink.fill.Loader.open(init.gpa, io, if (m.mix_cfg) |nc| nc else &m.c, if (m.mix) m.mix_native else dir, rank, world, workers, readers);
    defer loader.close();
    m.loader = loader;
    const L = m.c.layers;
    const E = m.c.experts;
    // the draft head before the arena is sized: its bytes come out of the room
    var draft: ?*ink.dspark.Draft = null;
    defer if (draft) |dd| dd.close();
    if (spec_depth > 0) {
        try m.enableSpec();
        const td = std.Io.Clock.awake.now(io);
        draft = try ink.dspark.Draft.open(init.gpa, io, m, dir, rows);
        std.debug.print("rank {d}/{d}: DSpark draft head loaded in {d:.0} ms (depth {d})\n", .{ rank, world, ms(io, td), spec_depth });
    }

    // the arena: the uniform quota, or what MemAvailable leaves above the floor (the same on every rank: the minimum)
    const w = m.slotWords();
    const sb = [2]f64{ @floatFromInt(ink.model.Arena.slotBytes(w, 0)), @floatFromInt(ink.model.Arena.slotBytes(w, 1)) };
    const nl = [2]usize{ m.c.countAct(.nvfp4), m.c.countAct(.mxfp8) }; // the arena's pools: release formats
    const tr: usize = if (want_exact) @min(E, transient orelse m.c.topk * rows) else 0;
    const trs = [2]usize{ if (nl[0] > 0) tr else 0, if (nl[1] > 0) tr else 0 };
    const avail0 = try ink.adapt.memAvailable(io, init.gpa);
    const margin: f64 = 3.5 * gib; // host: the gather arena, pack buffers and mappings (2.4 GiB RSS measured); workspace growth; the TP group
    var room = @as(f64, @floatFromInt(avail0)) - floor_gib * gib - margin - @as(f64, @floatFromInt(trs[0])) * sb[0] - @as(f64, @floatFromInt(trs[1])) * sb[1];
    // p4-mem: with the lazy context the arena takes the context memory not mapped (TF_INKLING2_LAZY_ARENA_GIB at most)
    // and leaves the prompt scratch's room, made after it (engine.zig does the same)
    if (m.lazy) |lz| room -= @floatFromInt(lz.heldBack() + m.promptGrowth(@max(prefill_rows orelse rows, rows)));
    if (world > 1) {
        var mine = [1]f64{room};
        var all: [8]f64 = undefined;
        var xb = try cuda.DeviceBuffer.alloc(&driver, 8);
        defer xb.free();
        var xa = try cuda.DeviceBuffer.alloc(&driver, 8 * world);
        defer xa.free();
        try xb.upload(0, std.mem.sliceAsBytes(&mine));
        try m.group.?.allGather(m.s, xb.ptr, xa.ptr, 8);
        try m.s.synchronize();
        try xa.download(0, std.mem.sliceAsBytes(all[0..world]));
        for (all[0..world]) |r| room = @min(room, r);
    }
    var resident: [2]usize = .{ 0, 0 };
    var need: f64 = 0;
    if (m.mix) {
        // ---- x-mixed: the EXL3 slots first (opened below, after the residency), then the bank (arena) from what
        // MemAvailable leaves over the floor, the bank margin and the prompt scratch (engine.zig sizes it the same way)
        need = @floatFromInt(try ink.exl3.Store.fits(init.gpa, io, dir, m, room, floor_gib));
        m.ex = try ink.exl3.Store.open(init.gpa, io, m, dir, floor_gib);
        const xs = m.ex.?.stats;
        std.debug.print("rank {d}/{d}: EXL3 export (activations {t}): all {d} experts of {d} layers resident, {d:.2} GiB of slots, {d} files, {d:.1} GB read ({d:.1} s read, {d:.1} s pack, {d:.1} s upload; paced {d} ms, MemAvailable min {d:.2} GiB)\n", .{ rank, world, m.exl3_act, E, L, @as(f64, @floatFromInt(m.ex.?.bytes)) / gib, xs.files, @as(f64, @floatFromInt(xs.read_bytes)) / 1e9, @as(f64, @floatFromInt(xs.read_ns)) / 1e9, @as(f64, @floatFromInt(xs.pack_ns)) / 1e9, @as(f64, @floatFromInt(xs.upload_ns)) / 1e9, xs.paused_ms, @as(f64, @floatFromInt(xs.min_avail)) / gib });
        var mo = ink.mix.Opts.fromEnv();
        if (bank_gib) |g| mo.bank_gib = g;
        const after: f64 = @floatFromInt(m.promptGrowth(@max(prefill_rows orelse rows, rows)) + 2 * (loader.stage_bytes + 4096) + (if (m.lazy) |lz| lz.heldBack() else 0));
        var broom = @as(f64, @floatFromInt(try ink.adapt.memAvailable(io, init.gpa))) - (floor_gib + mo.margin_gib) * gib - after;
        if (world > 1) {
            var mine = [1]f64{broom};
            var all: [8]f64 = undefined;
            var xb = try cuda.DeviceBuffer.alloc(&driver, 8);
            defer xb.free();
            var xa = try cuda.DeviceBuffer.alloc(&driver, 8 * world);
            defer xa.free();
            try xb.upload(0, std.mem.sliceAsBytes(&mine));
            try m.group.?.allGather(m.s, xb.ptr, xa.ptr, 8);
            try m.s.synchronize();
            try xa.download(0, std.mem.sliceAsBytes(all[0..world]));
            for (all[0..world]) |r| broom = @min(broom, r);
        }
        resident = if (bank_per_layer) |p| .{ p * nl[0], p * nl[1] } else ink.mix.bankSlots(m, mo, broom);
        const bbytes = ink.mix.bankBytes(m, resident);
        if (bbytes > @max(broom, 0) + 1) {
            std.log.err("inkling2: the native bank's {d} + {d} slots need {d:.2} GiB; {d:.2} GiB is left over the floor, the bank margin and the prompt scratch", .{ resident[0], resident[1], bbytes / gib, broom / gib });
            return error.ArenaAboveFloor;
        }
        try m.allocArena(resident, .{ 0, 0 });
        std.debug.print("rank {d}/{d}: x-mixed: the native bank: {d} NVFP4 + {d} MXFP8 slots ({d:.2} GiB of the release's experts, {d:.1} a layer on average, {d:.1} in the NVFP4 layers, {d:.1} in the MXFP8 layers) of {d:.2} GiB over the {d} GiB floor and the {d:.1} GiB bank margin\n", .{ rank, world, resident[0], resident[1], bbytes / gib, @as(f64, @floatFromInt(resident[0] + resident[1])) / @as(f64, @floatFromInt(L)), @as(f64, @floatFromInt(resident[0])) / @as(f64, @floatFromInt(@max(nl[0], 1))), @as(f64, @floatFromInt(resident[1])) / @as(f64, @floatFromInt(@max(nl[1], 1))), @max(broom, 0) / gib, floor_gib, mo.margin_gib });
        // ---- end x-mixed
    } else if (m.c.exl3) {
        // the EXL3 export (exl3.zig): every expert resident (no quota, no prior, no substitution: --arms exact is sub)
        need = @floatFromInt(try ink.exl3.Store.fits(init.gpa, io, dir, m, room, floor_gib)); // x-merge: one check
        try m.allocArena(.{ 0, 0 }, .{ 0, 0 });
    } else {
        if (per_layer) |p| {
            resident = .{ p * nl[0], p * nl[1] };
        } else {
            const per = room / (@as(f64, @floatFromInt(nl[0])) * sb[0] + @as(f64, @floatFromInt(nl[1])) * sb[1]);
            const n = @min(@as(usize, @intFromFloat(@max(per, 0))), E);
            resident = .{ n * nl[0], n * nl[1] };
        }
        need = @as(f64, @floatFromInt(resident[0])) * sb[0] + @as(f64, @floatFromInt(resident[1])) * sb[1];
        if (need > room + 1) {
            std.log.err("inkling2: {d} + {d} resident slots need {d:.2} GiB; {d:.2} GiB is left above the {d} GiB floor", .{ resident[0], resident[1], need / gib, room / gib, floor_gib });
            return error.ArenaAboveFloor;
        }
        if (resident[0] < m.c.topk * nl[0] or resident[1] < m.c.topk * nl[1]) return error.BudgetBelowFloor;
        try m.allocArena(resident, trs);
    }
    const avail1 = try ink.adapt.memAvailable(io, init.gpa);

    // the prior, the residency, the fill
    const prior = try a.alloc(f32, L * E);
    if (prior_path) |p| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, a, .limited(L * E * 4 + 1));
        if (bytes.len != L * E * 4) return error.PriorShape;
        @memcpy(std.mem.sliceAsBytes(prior), bytes);
    } else @memset(prior, 1);
    var policy: ink.residency.Policy = .{};
    policy.adapt = adapt_on and !m.c.exl3; // the EXL3 export: every expert resident, nothing to move
    const ad = try ink.adapt.Adapt.init(init.gpa, io, m, loader, prior, policy);
    defer ad.deinit();
    const t1 = std.Io.Clock.awake.now(io);
    if (m.mix) {
        try ad.bankHistory(dir); // x-mixed: the demand history (TF_INKLING2_MIX_HISTORY) before the boot fill
        try ad.fillBoot(bank_per_layer); // x-mixed: the bank's boot fill (the prior or the history; the slots are open)
    } else if (m.c.exl3) {
        m.ex = try ink.exl3.Store.open(init.gpa, io, m, dir, floor_gib);
        const xs = m.ex.?.stats;
        std.debug.print("rank {d}/{d}: EXL3 export (activations {t}): all {d} experts of {d} layers resident, {d:.2} GiB of slots, {d} files, {d:.1} GB read ({d:.1} s read, {d:.1} s pack, {d:.1} s upload; paced {d} ms, MemAvailable min {d:.2} GiB)\n", .{ rank, world, m.exl3_act, E, L, @as(f64, @floatFromInt(m.ex.?.bytes)) / gib, xs.files, @as(f64, @floatFromInt(xs.read_bytes)) / 1e9, @as(f64, @floatFromInt(xs.read_ns)) / 1e9, @as(f64, @floatFromInt(xs.pack_ns)) / 1e9, @as(f64, @floatFromInt(xs.upload_ns)) / 1e9, xs.paused_ms, @as(f64, @floatFromInt(xs.min_avail)) / gib });
    } else try ad.fillBoot(per_layer);
    const fill_ms = ms(io, t1);
    const avail2 = try ink.adapt.memAvailable(io, init.gpa);
    const st = loader.stats;
    if (rank == 0 or true) std.debug.print(
        "rank {d}/{d}: dense in {d:.0} ms; arena {d} NVFP4 + {d} MXFP8 resident slots ({d:.2} GiB, {d} transient a format), filled in {d:.1} s: {d} experts, {d:.1} GB read O_DIRECT={} ({d:.1} s read, {d:.1} s pack, {d:.1} s upload; paced {d} ms, PSI max {d:.2}, MemAvailable min {d:.2} GiB seen by the pace); MemAvailable {d:.2} -> {d:.2} -> {d:.2} GiB (floor {d})\n",
        .{ rank, world, dense_ms, resident[0], resident[1], need / gib, tr, fill_ms / 1e3, st.experts, @as(f64, @floatFromInt(st.bytes)) / 1e9, loader.direct, @as(f64, @floatFromInt(st.read_ns)) / 1e9, @as(f64, @floatFromInt(st.pack_ns)) / 1e9, @as(f64, @floatFromInt(st.upload_ns)) / 1e9, st.paused_ms, st.max_psi, @as(f64, @floatFromInt(st.min_avail)) / gib, @as(f64, @floatFromInt(avail0)) / gib, @as(f64, @floatFromInt(avail1)) / gib, @as(f64, @floatFromInt(avail2)) / gib, floor_gib },
    );
    ink.page_cache.finish(dir, rank, world); // e-load (x5): the small files' pass, one line a rank
    if (m.lazy) |lz| {
        var lb: [512]u8 = undefined;
        std.debug.print("rank {d}/{d}: {s}\n", .{ rank, world, lz.summary(&lb) });
    }
    if (rank == 0 and (!m.c.exl3 or m.mix)) { // x-mixed: the bank's spread over the layers
        var lo: usize = E;
        var hi: usize = 0;
        for (0..L) |l| {
            const n = ad.residentIn(l);
            lo = @min(lo, n);
            hi = @max(hi, n);
        }
        std.debug.print("rank 0: resident per layer {d}..{d} of {d}\n", .{ lo, hi, E });
    }

    // --prefill-rows: the sub arm's prompt chunks of P rows, their scratch made now (the arena above is sized without
    // it: the residency is the one --rows alone gets); every rank runs the same command, so the same P
    const prompt_rows = @max(prefill_rows orelse rows, rows);
    if (prompt_rows > rows) {
        const grown = m.promptGrowth(prompt_rows);
        try m.growPrompt(prompt_rows);
        std.debug.print("rank {d}/{d}: prompt chunks of up to {d} rows ({d:.2} GiB past the arena); MemAvailable {d:.2} GiB after\n", .{ rank, world, prompt_rows, @as(f64, @floatFromInt(grown)) / gib, @as(f64, @floatFromInt(try ink.adapt.memAvailable(io, init.gpa))) / gib });
    }
    // x-merge: the lazy context's reach without a give-back (lazy.reach; the EXL3 store gives nothing back)
    if (m.lazy) |lz| {
        const av = try ink.adapt.memAvailable(io, init.gpa);
        std.debug.print("rank {d}/{d}: lazy context reach: {d} of {d} positions ({d} mapped; MemAvailable {d:.2} GiB over the {d:.1} GiB floor and {d:.1} GiB margin)\n", .{ rank, world, lz.reach(av -| (lz.floor + lz.margin)), lz.context, lz.mapped, @as(f64, @floatFromInt(av)) / gib, @as(f64, @floatFromInt(lz.floor)) / gib, @as(f64, @floatFromInt(lz.margin)) / gib });
        if (m.mix) { // x-mixed: the reach with the whole native bank given back
            const bank: u64 = @intFromFloat(ink.mix.bankBytes(m, .{ m.arena.slots(0), m.arena.slots(1) }));
            std.debug.print("rank {d}/{d}: x-mixed: with the native bank's {d:.2} GiB given back the context reaches {d} of {d} positions\n", .{ rank, world, @as(f64, @floatFromInt(bank)) / gib, lz.reach((av -| (lz.floor + lz.margin)) + bank), lz.context });
        }
    }
    if (prewarm) {
        const tw = std.Io.Clock.awake.now(io);
        try ink.warm.prewarm(m, draft);
        if (rank == 0) std.debug.print("rank 0: prewarm (every chunk width, the draft) in {d:.0} ms\n", .{ms(io, tw)});
    }
    try ad.bankBoot(); // x-mixed: the bank back inside its band after the prewarm's memory (nothing without it)
    // e-dense: the --dense-sets (after the store / arena, which is sized without them), then --hcap or the logits path
    for (dense_sets.items) |*ds| {
        if (ds.classes.count() == 0) continue;
        if (m.dx == null) m.dx = try ink.dense_exl3.DenseExl3.init(init.gpa, &driver, &m.g, rank, world, m.canon);
        ds.idx = try m.dx.?.addSet(io, ds.name, ds.classes, ds.dir, L, ds.wide, ds.dtiles);
    }
    if (dense_sets.items.len > 0) std.debug.print("rank {d}/{d}: {d} dense sets ({d:.2} GiB of EXL3 dense on the device); MemAvailable {d:.2} GiB\n", .{ rank, world, dense_sets.items.len, @as(f64, @floatFromInt(if (m.dx) |dx| dx.bytes else 0)) / gib, @as(f64, @floatFromInt(try ink.adapt.memAvailable(io, init.gpa))) / gib });
    if (hcap_dir) |hd| {
        m.graphs.on = false; // the capture's gathers (o_proj's inputs) run eagerly
        m.hcap = try ink.hcap.Hcap.init(init.gpa, &driver, &m.g.lt, m.g.handle, L, hcap_lo, hcap_hi orelse L, rank, world);
        const th = std.Io.Clock.awake.now(io);
        for (ids_files.items) |f| {
            const seq = try readIds(a, io, f.path);
            try m.reset();
            var p: usize = 0;
            while (p < seq.len) : (p += prompt_rows) {
                const n = @min(prompt_rows, seq.len - p);
                var am: [1]i32 = undefined;
                _ = try ink.forward.chunk(m, seq[p .. p + n], .last, null, &am);
                m.hcap.?.rows += n;
            }
        }
        try m.s.synchronize();
        const cap_ms = ms(io, th);
        try m.hcap.?.write(io, m.s, hd, rank);
        std.debug.print("rank {d}/{d}: hcap: {d} rows of {d} sequences in {d:.0} ms, layers [{d}, {d}) + head {}, written to {s} in {d:.0} ms; lowest MemAvailable {d:.2} GiB\n", .{ rank, world, m.hcap.?.rows, ids_files.items.len, cap_ms, m.hcap.?.lo, m.hcap.?.hi, m.hcap.?.head, hd, ms(io, th) - cap_ms, @as(f64, @floatFromInt(try ink.adapt.memAvailable(io, init.gpa))) / gib });
        return 0;
    }
    if (ids_files.items.len > 0 and out != null) {
        const exact_path = want_exact;
        m.exact = exact_path;
        const step = if (exact_path) rows else prompt_rows;
        var hb = try ink.forward.HeadBlock.init(m, head_block);
        defer hb.deinit();
        const lbuf = try init.gpa.alloc(f32, head_block * m.c.unpadded_vocab); // a head block's rows (not a chunk's)
        defer init.gpa.free(lbuf);
        const none = [1]DenseSet{.{ .name = "active", .classes = ink.dense_exl3.Classes.empty, .dir = "" }};
        const sets = if (dense_sets.items.len > 0) dense_sets.items else &none;
        for (sets) |ds| {
            if (dense_sets.items.len > 0) try activateSet(m, ds.idx);
            for (ids_files.items) |f| {
                const seq = try readIds(a, io, f.path);
                try m.reset();
                m.exact_loads = 0;
                var file: ?std.Io.File = null;
                var nb: [512]u8 = undefined;
                if (rank == 0) file = try std.Io.Dir.cwd().createFile(io, try std.fmt.bufPrint(&nb, "{s}.{s}.{s}.logits", .{ out.?, f.name, ds.name }), .{});
                defer if (file) |fh| fh.close(io);
                const tl = std.Io.Clock.awake.now(io);
                var p: usize = 0;
                var off: u64 = 0;
                while (p < seq.len) : (p += step) {
                    const n = @min(step, seq.len - p);
                    const U = m.c.unpadded_vocab;
                    try ink.forward.chunkRows(m, seq[p .. p + n]);
                    var r0: usize = 0;
                    while (r0 < n) : (r0 += head_block) {
                        const bn = @min(head_block, n - r0);
                        try ink.forward.blockLogits(m, &hb, r0, bn, lbuf[0 .. bn * U]);
                        if (file) |fh| try fh.writePositionalAll(io, std.mem.sliceAsBytes(lbuf[0 .. bn * U]), off);
                        off += bn * U * 4;
                    }
                }
                std.debug.print("rank {d}/{d} [logits {s}/{s}] {s}: {d} rows in {d:.0} ms; {d} exact-mode loads\n", .{ rank, world, if (exact_path) "exact" else "sub", ds.name, f.name, seq.len, ms(io, tl), m.exact_loads });
            }
        }
        m.exact = false;
    }
    if (prefill_bench.items.len > 0) {
        // the prefill alone (no logits): each named --ids-files prompt in --prefill-rows chunks, twice a set (the first
        // pass makes any width's plans the prewarm did not), the last chunk with the one-row head as a server's
        const none = [1]DenseSet{.{ .name = "active", .classes = ink.dense_exl3.Classes.empty, .dir = "" }};
        const sets = if (dense_sets.items.len > 0) dense_sets.items else &none;
        for (sets) |ds| {
            if (dense_sets.items.len > 0) {
                try activateSet(m, ds.idx);
                if (prewarm) try ink.warm.prewarm(m, draft);
            }
            for (ids_files.items) |f| {
                var want = false;
                for (prefill_bench.items) |nm| want = want or std.mem.eql(u8, nm, f.name);
                if (!want) continue;
                const seq = try readIds(a, io, f.path);
                for (0..2) |pass| {
                    try m.reset();
                    try m.s.synchronize();
                    const tb = std.Io.Clock.awake.now(io);
                    var p: usize = 0;
                    while (p < seq.len) : (p += prompt_rows) {
                        const n = @min(prompt_rows, seq.len - p);
                        _ = try ink.forward.chunk(m, seq[p .. p + n], if (p + n >= seq.len) .last else .none, null, null);
                    }
                    try m.s.synchronize();
                    std.debug.print("rank {d}/{d} [prefill {s}] {s}: {d} rows in {d:.0} ms (pass {d})\n", .{ rank, world, ds.name, f.name, seq.len, ms(io, tb), pass });
                }
            }
        }
    }
    if (bench_dense) try benchDense(m, &driver, dense_sets.items, rank);
    if (ids.items.len == 0 and ids_files.items.len > 0) {
        std.debug.print("rank {d}/{d}: lowest MemAvailable seen {d:.2} GiB\n", .{ rank, world, @as(f64, @floatFromInt(@min(avail2, try ink.adapt.memAvailable(io, init.gpa)))) / gib });
        return 0;
    }
    const V = m.c.unpadded_vocab;
    const total = @max(@max(ids.items.len, ids_alt.items.len), longest) + @max(tokens, force.items.len); // x-mixed: the longest arm prompt
    // p4-dattn: the rows' logits only when they are kept (--out, --force): a long prompt's buffer (16K tokens x the
    // vocabulary x 4 bytes = 12.9 GB, touched by the safe-mode allocator) left the guard's floor and was killed
    const logits = try a.alloc(f32, if (out != null or force.items.len > 0) total * V else 0);
    const dem0 = try a.alloc(f32, L * E * 3);
    const dem1 = try a.alloc(f32, L * E * 3);
    const argmax = try a.alloc(i32, total);
    // e-dense: every arm once per --dense-sets set (else once, on the active weights), its lines and files tagged
    // <arm>-<set>
    const none_set = [1]DenseSet{.{ .name = "", .classes = ink.dense_exl3.Classes.empty, .dir = "" }};
    const run_sets = if (dense_sets.items.len > 0) dense_sets.items else &none_set;
    for (run_sets) |dset| for (arms.items, 0..) |arm, arm_i| {
        if (dense_sets.items.len > 0 and arm_i == 0) {
            try activateSet(m, dset.idx);
            if (prewarm) try ink.warm.prewarm(m, draft);
            if (rank == 0) std.debug.print("[dense-set] {s}\n", .{dset.name});
        }
        // x-merge: a repeated arm kind's outputs carry the arm's position (PREFIX.sub0, PREFIX.sub2, ...); e-dense:
        // then -<set>
        var same: usize = 0;
        for (arms.items) |xa| same += @intFromBool(xa == arm);
        var nb: [32]u8 = undefined;
        const arm_name = if (same > 1) try std.fmt.bufPrint(&nb, "{s}{d}", .{ @tagName(arm), arm_i }) else @tagName(arm);
        var lab_buf: [96]u8 = undefined;
        const label = if (dset.name.len == 0) arm_name else try std.fmt.bufPrint(&lab_buf, "{s}-{s}", .{ arm_name, dset.name });
        // p4-small: TF_INKLING2_ARM_SWITCHES (A/B of the prefill switches in one process); x-merge: prows=N the arm's
        // prompt chunk rows (at most the grown --prefill-rows)
        const arm_set = try armSwitches(m, ad, arm_i, draft); // x-mixed: the bank's switches need the adapt; e-step: the draft for dgraph= / dexl3=
        const arm_rows = arm_set.prows;
        if (arm_rows) |r| if (r == 0 or r > prompt_rows) return error.BadArmSwitches;
        // e-prefill: ids=2, this arm's prompt is --ids2 (its rows' logits are never kept)
        if (arm_set.ids2 and (ids_alt.items.len == 0 or out != null or force.items.len > 0)) return error.BadArmSwitches;
        // x-mixed: or the arm's own prompt (TF_INKLING2_ARM_IDS)
        const arm_ids = if (arm_set.ids2) ids_alt.items else if (arm_i < arm_prompts.items.len) arm_prompts.items[arm_i] else ids.items;
        try m.reset();
        m.exact = arm == .exact;
        m.exact_loads = 0;
        try ad.begin();
        const full = out != null or force.items.len > 0;
        // the prompt (and a forced continuation) in chunks, every row's logits kept when asked for
        var seq: std.ArrayList(i32) = .empty;
        try seq.appendSlice(a, arm_ids);
        if (!force_decode) try seq.appendSlice(a, force.items);
        var at: usize = 0;
        const tp = std.Io.Clock.awake.now(io);
        var p: usize = 0;
        const step = if (arm == .exact) rows else arm_rows orelse prompt_rows; // exact mode's loads take max_rows rows' picks
        const pr = if (arm == .sub) ProfRange.init() else ProfRange{};
        if (pr.start) |f| if (!pr.benches) {
            try m.s.synchronize();
            _ = f();
        };
        while (p < seq.items.len) : (p += step) {
            const n = @min(step, seq.items.len - p);
            const last = p + n >= seq.items.len;
            _ = try ink.forward.chunk(m, seq.items[p .. p + n], if (full) .all else if (last) .last else .none, if (full) logits[at * V .. (at + n) * V] else null, if (full) argmax[at .. at + n] else argmax[at + n - 1 .. at + n]);
            if (draft) |dd| try dd.absorb(m.spec.?.taps.ptr, p, n);
            at += n;
        }
        try m.s.synchronize();
        const prefill_ms = ms(io, tp);
        if (pr.stop) |f| if (!pr.benches) {
            _ = f();
        };
        if (pr.start) |f| if (pr.decode) { // e-step: mode 3 opens at the decode loop
            try m.s.synchronize();
            _ = f();
        };
        var gen: std.ArrayList(i32) = .empty;
        var windows: std.ArrayList([ink.dspark.block]i32) = .empty;
        var dec_ms: f64 = 0;
        if (force.items.len > 0 and force_decode) {
            try ad.afterPrefill(arm_ids.len);
            const td = std.Io.Clock.awake.now(io);
            for (force.items) |t| {
                const one = [1]i32{t};
                _ = try ink.forward.chunk(m, &one, .all, logits[at * V .. (at + 1) * V], argmax[at .. at + 1]);
                at += 1;
                try ad.afterToken();
                try ad.afterRound(); // x-mixed
            }
            dec_ms = ms(io, td);
            try ad.end();
        } else if (force.items.len == 0 and draft != null) {
            // DSpark speculation: draft from the anchor, verify [anchor, drafts] in one forward, keep the longest
            // prefix the target's own greedy picks agree with, plus its pick after it (the next anchor)
            const dd = draft.?;
            try ad.afterPrefill(arm_ids.len);
            var anchor = argmax[at - 1];
            const td = std.Io.Clock.awake.now(io);
            var rounds: usize = 0;
            var drafted: usize = 0;
            var accepted: usize = 0;
            var draft_ms: f64 = 0;
            var verify_ms: f64 = 0;
            var rewind_ms: f64 = 0;
            var absorb_ms: f64 = 0;
            var after_ms: f64 = 0;
            var distinct_sum: f64 = 0;
            // xs: TF_INKLING2_SAMPLED_CHECK=1: every window first verified as a sampled server window twice (every
            // row's logits read back): eagerly (graphs off), rewound, then through its graph (graphs.Head.rows: the key's
            // first window eager and captured, replays after), rewound; both rows compared bit for bit, both timed
            const sc_on = if (std.c.getenv("TF_INKLING2_SAMPLED_CHECK")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
            const scU = m.c.unpadded_vocab;
            const sc_a = try a.alloc(f32, if (sc_on) ink.dspark.block * scU else 0);
            const sc_b = try a.alloc(f32, if (sc_on) ink.dspark.block * scU else 0);
            var sc_windows: usize = 0;
            var sc_replays: usize = 0;
            var sc_differ: usize = 0;
            var sc_eager_ms: f64 = 0;
            var sc_graph_ms: f64 = 0;
            while (gen.items.len < tokens) {
                try gen.append(a, anchor);
                if (gen.items.len == tokens) break;
                const base = m.state.len;
                var d7: [ink.dspark.spec]i32 = undefined;
                const tdr = std.Io.Clock.awake.now(io);
                try dd.draft(anchor, base, &d7);
                draft_ms += ms(io, tdr);
                const depth = @min(spec_depth, tokens - gen.items.len);
                var row: [ink.dspark.block]i32 = undefined;
                row[0] = anchor;
                @memcpy(row[1 .. 1 + depth], d7[0..depth]);
                var g: [ink.dspark.block]i32 = undefined;
                if (sc_on) {
                    const n = 1 + depth;
                    var ga: [ink.dspark.block]i32 = undefined;
                    var gb: [ink.dspark.block]i32 = undefined;
                    const was = m.graphs.on;
                    m.graphs.on = false;
                    const sct0 = std.Io.Clock.awake.now(io);
                    _ = try ink.forward.chunk(m, row[0..n], .all, sc_a[0 .. n * scU], ga[0..n]);
                    sc_eager_ms += ms(io, sct0);
                    m.graphs.on = was;
                    try m.rewind(base, 0);
                    const r0 = m.graphs.replays;
                    const sct1 = std.Io.Clock.awake.now(io);
                    _ = try ink.forward.chunk(m, row[0..n], .all, sc_b[0 .. n * scU], gb[0..n]);
                    const replayed = m.graphs.replays > r0;
                    if (replayed) sc_graph_ms += ms(io, sct1);
                    try m.rewind(base, 0);
                    sc_windows += 1;
                    if (replayed) sc_replays += 1;
                    if (!std.mem.eql(u8, std.mem.sliceAsBytes(sc_a[0 .. n * scU]), std.mem.sliceAsBytes(sc_b[0 .. n * scU])) or !std.mem.eql(i32, ga[0..n], gb[0..n])) sc_differ += 1;
                }
                if (prof >= 2) try demandBefore(m, dem0);
                const tvf = std.Io.Clock.awake.now(io);
                _ = try ink.forward.chunk(m, row[0 .. 1 + depth], .all, null, g[0 .. 1 + depth]);
                verify_ms += ms(io, tvf);
                if (prof >= 2) distinct_sum += try distinctSince(m, dem0, dem1);
                if (depth == ink.dspark.spec) try windows.append(a, row);
                var k: usize = 0;
                while (k < depth and row[1 + k] == g[k]) k += 1;
                const keep = k + 1;
                const trw = std.Io.Clock.awake.now(io);
                if (keep < 1 + depth) try m.rewind(base, keep);
                if (prof >= 1) try m.s.synchronize();
                rewind_ms += ms(io, trw);
                const tab = std.Io.Clock.awake.now(io);
                try dd.absorb(m.spec.?.taps.ptr, base, keep);
                if (prof >= 1) try m.s.synchronize();
                absorb_ms += ms(io, tab);
                for (row[1 .. 1 + k]) |t| {
                    if (gen.items.len == tokens) break;
                    try gen.append(a, t);
                }
                const taf = std.Io.Clock.awake.now(io);
                for (0..keep) |_| try ad.afterToken();
                try ad.afterRound(); // x-mixed: the native bank's tick (nothing without it)
                after_ms += ms(io, taf);
                anchor = g[k];
                rounds += 1;
                drafted += depth;
                accepted += k;
            }
            dec_ms = ms(io, td);
            try ad.end();
            if (rank == 0) std.debug.print("[spec] depth {d}: {d} rounds, {d} of {d} drafts accepted ({d:.3}), {d:.2} tokens a round; draft {d:.1} ms, verify {d:.1} ms a round\n", .{ spec_depth, rounds, accepted, drafted, @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(@max(drafted, 1))), @as(f64, @floatFromInt(gen.items.len - 1)) / @as(f64, @floatFromInt(@max(rounds, 1))), draft_ms / @as(f64, @floatFromInt(@max(rounds, 1))), verify_ms / @as(f64, @floatFromInt(@max(rounds, 1))) });
            if (sc_on and rank == 0) std.debug.print("[sampled-graph] {d} windows, {d} through a graph replay: {d} differ from the eager window (every row's fp32 logits and argmax); eager {d:.2} ms, graphed {d:.2} ms a window\n", .{ sc_windows, sc_replays, sc_differ, sc_eager_ms / @as(f64, @floatFromInt(@max(sc_windows, 1))), sc_graph_ms / @as(f64, @floatFromInt(@max(sc_replays, 1))) });
            if (rank == 0 and prof >= 1) {
                const nr: f64 = @floatFromInt(@max(rounds, 1));
                std.debug.print("[prof] a round: draft {d:.2} ms, verify {d:.2} ms, rewind {d:.2} ms, absorb {d:.2} ms, afterToken {d:.2} ms, rest {d:.2} ms (round {d:.2} ms); distinct experts a layer a verify {d:.1}\n", .{ draft_ms / nr, verify_ms / nr, rewind_ms / nr, absorb_ms / nr, after_ms / nr, (dec_ms - draft_ms - verify_ms - rewind_ms - absorb_ms - after_ms) / nr, dec_ms / nr, distinct_sum / nr });
            }
        } else if (force.items.len == 0) {
            try ad.afterPrefill(arm_ids.len);
            var tok = argmax[at - 1];
            const td = std.Io.Clock.awake.now(io);
            for (0..tokens) |_| {
                try gen.append(a, tok);
                if (gen.items.len == tokens) break;
                const one = [1]i32{tok};
                _ = try ink.forward.chunk(m, &one, if (full) .all else .last, if (full) logits[at * V .. (at + 1) * V] else null, argmax[at .. at + 1]);
                tok = argmax[at];
                at += 1;
                try ad.afterToken();
                try ad.afterRound(); // x-mixed
            }
            dec_ms = ms(io, td);
            try ad.end();
        }
        if (pr.start) |f| if (pr.bench_only) { // e-step: mode 4 opens at the benches
            try m.s.synchronize();
            _ = f();
        };
        if (bench.items.len > 0 and bench_real and arm == .sub and windows.items.len > 0) {
            // the same widths over the spec loop's own windows [anchor, 7 drafts] (their first T rows): a verify's
            // routing, not a repeated token's; rep 0 is the first use at that point (cold plans for unseen widths)
            for (bench.items) |T| {
                var best: f64 = 1e30;
                var sum: f64 = 0;
                var first_ms: f64 = 0;
                var dist: f64 = 0;
                for (0..7) |rep| {
                    const win = windows.items[(rep + T) % windows.items.len];
                    var am: [16]i32 = undefined;
                    if (prof >= 2) try demandBefore(m, dem0);
                    const tb = std.Io.Clock.awake.now(io);
                    _ = try ink.forward.chunk(m, win[0..T], .all, null, am[0..T]);
                    try m.s.synchronize();
                    const dt = ms(io, tb);
                    if (prof >= 2) dist += try distinctSince(m, dem0, dem1);
                    if (rep > 0) {
                        best = @min(best, dt);
                        sum += dt;
                    } else first_ms = dt;
                }
                if (rank == 0) std.debug.print("[bench-real] {s}: {d} rows a forward: mean {d:.1} ms, best {d:.1} ms, first {d:.1} ms; distinct experts a layer {d:.1}\n", .{ @tagName(arm), T, sum / 6, best, first_ms, dist / 7 });
            }
        }
        if (bench.items.len > 0) {
            // forwards of T rows after the prompt (a verify window's shape), 6 each after one warm-up, the
            // token ids the greedy continuation's (their routing a real text's); the sequence grows as it would
            var tok = argmax[at - 1];
            for (bench.items) |T| {
                var ids2: [16]i32 = @splat(tok);
                var best: f64 = 1e30;
                var sum: f64 = 0;
                var dist: f64 = 0;
                for (0..7) |rep| {
                    if (prof >= 2) try demandBefore(m, dem0);
                    const tb = std.Io.Clock.awake.now(io);
                    var am: [16]i32 = undefined;
                    _ = try ink.forward.chunk(m, ids2[0..T], .all, null, am[0..T]);
                    try m.s.synchronize();
                    const dt = ms(io, tb);
                    if (prof >= 2 and rep > 0) dist += try distinctSince(m, dem0, dem1);
                    if (rep > 0) {
                        best = @min(best, dt);
                        sum += dt;
                    }
                    tok = am[T - 1];
                    for (&ids2, 0..) |*d, i| d.* = if (i < T) am[i] else tok;
                }
                if (rank == 0) std.debug.print("[bench] {s}: {d} rows a forward: mean {d:.1} ms, best {d:.1} ms ({d:.1} ms a row)\n", .{ @tagName(arm), T, sum / 6, best, sum / 6 / @as(f64, @floatFromInt(T)) });
                if (rank == 0 and prof >= 2) std.debug.print("[bench] {s}: {d} rows: distinct experts a layer {d:.1}\n", .{ @tagName(arm), T, dist / 6 });
            }
        }
        if (pr.stop) |f| if (pr.benches) { // e-step: modes 3 and 4 close after the benches
            try m.s.synchronize();
            _ = f();
        };
        const steps = if (force_decode and force.items.len > 0) force.items.len + 1 else gen.items.len;
        const per_tok = if (steps > 1) dec_ms / @as(f64, @floatFromInt(steps - 1)) else 0;
        std.debug.print("rank {d}/{d} [{s}]: {d} rows in {d:.0} ms; {d} exact-mode loads; generated {any}; decode {d:.1} ms/token ({d:.2} tok/s); residency {d} passes, {d} swaps, {d:.0} ms, plan mismatches {d}\n", .{ rank, world, if (dset.name.len == 0) @tagName(arm) else label, seq.items.len, prefill_ms, m.exact_loads, gen.items, per_tok, if (per_tok > 0) 1000.0 / per_tok else 0, ad.stats.passes, ad.stats.swaps, @as(f64, @floatFromInt(ad.stats.pass_ns)) / 1e6, ad.stats.plan_mismatch });
        if (out) |prefix| if (rank == 0) {
            var buf: [512]u8 = undefined;
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&buf, "{s}.{s}.logits", .{ prefix, label }), .data = std.mem.sliceAsBytes(logits[0 .. at * V]) });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&buf, "{s}.{s}.tokens", .{ prefix, label }), .data = std.mem.sliceAsBytes(gen.items) });
        };
    };
    std.debug.print("rank {d}/{d}: lowest MemAvailable seen {d:.2} GiB\n", .{ rank, world, @as(f64, @floatFromInt(@min(avail2, try ink.adapt.memAvailable(io, init.gpa)))) / gib });
    if (m.lazy) |lz| std.debug.print("rank {d}/{d}: lazy context: {d} positions mapped at the end (peak {d}), {d} growths, {d} trims, {d} refused; arena {d:.2} GiB given back, {d:.2} GiB mapped back\n", .{ rank, world, lz.mapped, lz.stats.peak, lz.stats.grows, lz.stats.trims, lz.stats.refused, @as(f64, @floatFromInt(lz.stats.gave)) / gib, @as(f64, @floatFromInt(lz.stats.regained)) / gib });
    return 0;
}

// ---- p4-small: TF_INKLING2_ARM_SWITCHES="SPEC:SPEC:..." sets the prefill switches before arm i to SPEC i (comma
// separated rel=0|1|2, fuse=0|1, absorb=0|1, attn=gqa|tiled|tc|fa, mx=0|1, xchg=pair|token; x-merge: dattn=0|1,
// prows=N; e-prefill: exl3=BUILD (TF_INKLING2_EXL3_PROMPT's names), actrot=0|1 (TF_INKLING2_EXL3_ACTROT), ids=1|2 (the
// arm's prompt: --ids or --ids2); e-step: sfuse=0|1, dgroup=0|1, dgraph=0|1, dexl3=0|1; unnamed ones keep their value, but prows and ids are the arm's own; no SPEC i: as
// before). Every rank runs the same command, so the same switches an arm. With --arms sub,sub,... and --adapt 0 one
// process times several settings on one arena (a measurement aid; the engine reads its switches once, Model.open).
const ArmSet = struct { prows: ?usize = null, ids2: bool = false };
/// x-mixed: the native bank's view for an arm (TF_INKLING2_ARM_SWITCHES bank / bankfrac / bankmx): hidden (the EXL3
/// store alone), the whole bank, or a part of it: each pool's top `frac` of its residents by the residency's blended
/// scores (a smaller bank's measure in the same load), the MXFP8 pool hidden when `!mx` (an NVFP4-only bank's). The
/// host's bank is untouched; every rank computes the same view (the same residency state: the same votes).
fn bankView(m: *ink.model.Model, ad: *ink.adapt.Adapt, show: bool, frac: f64, mx: bool) !void {
    if (!show) return m.publishBank(false);
    if (frac >= 1 and mx) return m.publishBank(true);
    const gpa = m.gpa;
    const n = m.c.layers * m.c.experts;
    const sc = try gpa.alloc(f64, n);
    defer gpa.free(sc);
    _ = try ad.r.scores(sc, null);
    const mask = try gpa.alloc(u8, n);
    defer gpa.free(mask);
    @memset(mask, 0);
    const Res = struct { s: f64, i: u32 };
    for (0..2) |f| {
        if (f == 1 and !mx) continue;
        var list: std.ArrayList(Res) = .empty;
        defer list.deinit(gpa);
        for (m.arena.owner[f][0..m.arena.slots(f)]) |o| if (o >= 0) try list.append(gpa, .{ .s = sc[@intCast(o)], .i = @intCast(o) });
        std.mem.sort(Res, list.items, {}, struct {
            fn lt(_: void, x: Res, y: Res) bool {
                if (x.s != y.s) return x.s > y.s;
                return x.i < y.i;
            }
        }.lt);
        const keep: usize = @intFromFloat(@floor(frac * @as(f64, @floatFromInt(list.items.len))));
        for (list.items[0..keep]) |r| mask[r.i] = 1;
    }
    try m.publishBankMask(mask);
}

fn armSwitches(m: *ink.model.Model, ad: *ink.adapt.Adapt, arm_i: usize, draft: ?*ink.dspark.Draft) !ArmSet {
    const all = std.mem.span(std.c.getenv("TF_INKLING2_ARM_SWITCHES") orelse return .{});
    var specs = std.mem.splitScalar(u8, all, ':'); // ':' (not ';': rank 1's command goes through a remote shell)
    var i: usize = 0;
    const spec = while (specs.next()) |sp| : (i += 1) {
        if (i == arm_i) break sp;
    } else return .{};
    var prows: ?usize = null;
    var ids2 = false;
    // x-mixed: the native bank's view for this arm (bank / bankfrac / bankmx), published once the arm's switches are read
    var view_set = false;
    var view_show = true;
    var view_frac: f64 = 1;
    var view_mx = true;
    var it = std.mem.tokenizeScalar(u8, spec, ',');
    while (it.next()) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=') orelse return error.BadArmSwitches;
        const k = kv[0..eq];
        const v = kv[eq + 1 ..];
        if (std.mem.eql(u8, k, "rel")) {
            m.rel_wide = try std.fmt.parseInt(u8, v, 10);
            if (m.rel_wide > 2) return error.BadArmSwitches;
        } else if (std.mem.eql(u8, k, "fuse")) {
            m.fuse = !std.mem.eql(u8, v, "0");
            if (m.fuse and !m.k.arn_big) return error.FuseUnavailable; // Model.open's condition
        } else if (std.mem.eql(u8, k, "absorb")) {
            m.absorb_wide = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "attn")) {
            m.prefill_attn = std.meta.stringToEnum(ink.model.PrefillAttn, v) orelse return error.BadArmSwitches;
        } else if (std.mem.eql(u8, k, "mx")) {
            // p4-mx: mx=0|1 sets Model.mx_act (W8A8 MXFP8 experts); on needs x.mxq (TF_INKLING2_MX_ACT=1 at open).
            // It changes the decode / verify kernels too: graphs captured under the other setting are dropped.
            const on = !std.mem.eql(u8, v, "0");
            if (on and m.x.mxq.ptr == 0) return error.MxActUnavailable;
            if (on != m.mx_act) m.graphs.drop(m.gpa);
            m.mx_act = on;
        } else if (std.mem.eql(u8, k, "xchg")) {
            // p4-xchg: xchg=pair|token sets Model.moe_xchg (the MoE's TP sum). token needs no buffer pair lacks; pair
            // needs y_all at its per-pair size, grown only when the model opened with pair (TF_INKLING2_MOE_XCHG unset
            // or pair). It changes the decode / verify kernels too: graphs captured under the other setting are dropped.
            const want = std.meta.stringToEnum(ink.model.MoeXchg, v) orelse return error.BadArmSwitches;
            const H = m.c.hidden;
            const need = if (m.world == 2 and m.split_moe) (m.prompt_rows + 1) / 2 * m.c.topk * H * 4 else m.world * m.prompt_rows * m.c.topk * H * 4;
            if (want == .pair and m.world > 1 and m.x.y_all.len < need) return error.XchgPairUnavailable;
            if (want != m.moe_xchg) m.graphs.drop(m.gpa);
            m.moe_xchg = want;
        } else if (std.mem.eql(u8, k, "dattn")) {
            // x-merge: dattn=0|1 sets p4-dattn's Model.decode_attn and draft_attn together. On needs their partials
            // (TF_INKLING2_DECODE_ATTN on at open: the default); off needs the draft head's score scratch, which
            // TF_INKLING2_LAZY_CONTEXT=1 does not make beside the draft attention. Graphs of the other setting dropped.
            const on = !std.mem.eql(u8, v, "0");
            if (on and m.fd_part.ptr == 0) return error.DecodeAttnUnavailable;
            if (!on and m.lazy != null and m.draft_attn and m.spec != null) return error.DecodeAttnUnavailable;
            if (on != m.decode_attn or on != m.draft_attn) m.graphs.drop(m.gpa);
            m.decode_attn = on;
            m.draft_attn = on;
        } else if (std.mem.eql(u8, k, "dgraph")) {
            // e-step: dgraph=0|1 sets the draft's TF_INKLING2_DRAFT_GRAPH (on needs the split-K draft attention and
            // graphs); its graph is dropped either way (the next pass runs eagerly and captures)
            const dd = draft orelse return error.BadArmSwitches;
            const on = !std.mem.eql(u8, v, "0");
            if (on and !(m.draft_attn and m.graphs.on)) return error.DraftGraphUnavailable;
            if (dd.dgraph) |e| e.deinit(dd.gpa);
            dd.dgraph = null;
            dd.dgraph_on = on;
        } else if (std.mem.eql(u8, k, "dexl3")) {
            // e-step: dexl3=0|1: the draft's seven projections on their EXL3 weights (TF_INKLING2_DRAFT_EXL3 at open)
            // or the bf16 ones (kept with TF_INKLING2_DRAFT_EXL3_KEEP_BF16=1); the draft graph is dropped
            const dd = draft orelse return error.BadArmSwitches;
            const on = !std.mem.eql(u8, v, "0");
            if (on and dd.dx == null) return error.DraftExl3Unavailable;
            if (!on and !dd.dx_bf16) return error.DraftBf16Unavailable;
            if (dd.dgraph) |e| e.deinit(dd.gpa);
            dd.dgraph = null;
            dd.dx_on = on;
        } else if (std.mem.eql(u8, k, "dgroup")) {
            // e-step: dgroup=0|1 sets Model.dense_group (a layer's 6-bit EXL3 attention-input projections as one group;
            // needs a dense EXL3 set: TF_INKLING2_DENSE_EXL3). On reserves the groups' scratch (graphs dropped after)
            const on = !std.mem.eql(u8, v, "0");
            if (on and m.dx == null) return error.DenseGroupUnavailable;
            if (on) try ink.forward.reserveGroups(m);
            if (on != m.dense_group) m.graphs.drop(m.gpa);
            m.dense_group = on;
        } else if (std.mem.eql(u8, k, "sfuse")) {
            // e-step: sfuse=0|1 sets Model.step_fuse (a decode / verify chunk's fused launches, the same bits; it
            // needs FUSE's shared-memory opt-in). Graphs captured under the other setting are dropped.
            const on = !std.mem.eql(u8, v, "0");
            if (on and !m.k.arn_big) return error.StepFuseUnavailable;
            if (on != m.step_fuse) m.graphs.drop(m.gpa);
            m.step_fuse = on;
        } else if (std.mem.eql(u8, k, "prows")) {
            // x-merge: prows=N, the arm's prompt chunk rows (1 .. --prefill-rows; checked by the caller)
            prows = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "exl3")) {
            // e-prefill: exl3=BUILD, the EXL3 wide-chunk kernels' build (kern.exl3PromptConfig: g2k8s2, v3g4d, ...; every
            // build the same bits); moeExl3 reads it at every chunk (the plan's item size with it)
            m.k.exl3_cfg = try ink.kern.exl3PromptConfig(v);
        } else if (std.mem.eql(u8, k, "actrot")) {
            // e-prefill: actrot=0|1, the EXL3 activation + rotation in one pass (the same bits); graphs captured under the
            // other setting are dropped (they hold the decode chunks' launches)
            const on = !std.mem.eql(u8, v, "0");
            if (on != m.k.exl3_actrot) m.graphs.drop(m.gpa);
            m.k.exl3_actrot = on;
        } else if (std.mem.eql(u8, k, "ids")) {
            // e-prefill: ids=1|2, the arm's prompt (--ids or --ids2; checked by the caller)
            ids2 = std.mem.eql(u8, v, "2");
            if (!ids2 and !std.mem.eql(u8, v, "1")) return error.BadArmSwitches;
        } else if (std.mem.eql(u8, k, "bank")) {
            // x-mixed: bank=0|1 hides the native bank from the device (every pick on the EXL3 store: the EXL3 store
            // alone, in the same load) or shows it again (Model.publishBank; the host's bank is untouched)
            if (!m.mix) return error.BadArmSwitches;
            view_set = true;
            view_show = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "bankfrac")) {
            // x-mixed: bankfrac=F (0 < F <= 1), only each pool's top F of its residents shown (a smaller bank's measure
            // in the same load: bankView); frozen arms only
            if (!m.mix) return error.BadArmSwitches;
            view_set = true;
            view_frac = try std.fmt.parseFloat(f64, v);
            if (!(view_frac > 0 and view_frac <= 1)) return error.BadArmSwitches;
        } else if (std.mem.eql(u8, k, "bankmx")) {
            // x-mixed: bankmx=0 hides the bank's MXFP8 pool (an NVFP4-only bank: the MXFP8 layers' picks on EXL3)
            if (!m.mix) return error.BadArmSwitches;
            view_set = true;
            view_mx = !std.mem.eql(u8, v, "0");
        } else if (std.mem.eql(u8, k, "admit")) {
            // x-mixed: admit=0|1, the native bank's admission for this arm on (the bank adapts) or off (it stays)
            const b = ad.bank orelse return error.BadArmSwitches;
            b.o.admit = !std.mem.eql(u8, v, "0");
            b.o.between = std.mem.eql(u8, v, "between"); // xa: admit=between, admission at the arm's end only
        } else if (std.mem.eql(u8, k, "trimhint")) {
            // xa (a gate): trimhint=N, p4-cache's begin-step hint for this arm's begin (the mapped positions of a state
            // likely to return): its reclaim trims to max(budget, live, N); the engine sets it from the step in xr
            ad.trim_hint = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, k, "giveback")) {
            // xa (a gate): giveback=K, K experts a layer of the bank given back before this arm's begin
            // (Adapt.bankGiveBack: a growing context's give-back of a size MemAvailable does not set), which its
            // reclaim then maps back: the map-back's gate compares the same experts in every load
            if (ad.bank == null) return error.BadArmSwitches;
            try ad.bankGiveBack(try std.fmt.parseInt(usize, v, 10));
        } else if (std.mem.eql(u8, k, "reads") or std.mem.eql(u8, k, "perlayer") or std.mem.eql(u8, k, "every")) {
            // x-mixed: the admission's batch size, per-layer cap and interval for this arm (TF_INKLING2_MIX_READS,
            // _PER_LAYER, _ADMIT_EVERY): a warm-up arm may adapt the bank faster than serving does
            const b = ad.bank orelse return error.BadArmSwitches;
            const n = try std.fmt.parseInt(usize, v, 10);
            if (std.mem.eql(u8, k, "reads")) b.o.reads = std.math.clamp(n, 1, 256) else if (std.mem.eql(u8, k, "perlayer")) b.o.per_layer = @max(n, 1) else b.o.every = @max(n, 1);
        } else if (std.mem.eql(u8, k, "check")) {
            // x-mixed: check=0|1, every layer's split checked on the host (forward.mixCheck); eager forwards for the arm
            if (!m.mix) return error.BadArmSwitches;
            const on = !std.mem.eql(u8, v, "0");
            m.graphs.drop(m.gpa);
            m.graphs.on = !on;
            m.mix_check = on;
        } else return error.BadArmSwitches;
    }
    // x-mixed: the arm's view of the bank (an admission's publish shows the whole bank: no admission beside a part)
    if (view_set) try bankView(m, ad, view_show, view_frac, view_mx);
    if (m.bank_mask != null) if (ad.bank) |bb| if (bb.o.admit) return error.BadArmSwitches;
    // x-mixed: without the score scratch (TF_INKLING2_SCORES=0) only the kernels that read none may run
    if (m.scores_off and !(m.decode_attn and (m.prefill_attn == .fa or m.prefill_attn == .tc))) return error.ScoresRead;
    if (m.rank == 0 and m.mix) std.debug.print("[arm {d}] x-mixed: the native bank {s} (top {d:.2} of each pool's residents, MXFP8 pool {s}), admission {s}, split check {s}\n", .{ arm_i, if (m.bank_hidden) "hidden (the EXL3 store alone)" else if (m.bank_mask != null) "partly shown" else "shown", view_frac, if (view_mx) "shown" else "hidden", if (ad.bank) |bb| (if (bb.o.admit) "on" else "off") else "-", if (m.mix_check) "on (eager)" else "off" });
    if (m.rank == 0) std.debug.print("[arm {d}] switches: rel {d} fuse {} absorb {} attn {s}\n", .{ arm_i, m.rel_wide, m.fuse, m.absorb_wide, @tagName(m.prefill_attn) });
    if (m.rank == 0) std.debug.print("[arm {d}] p4-mx: mx_act {}\n", .{ arm_i, m.mx_act });
    if (m.rank == 0) std.debug.print("[arm {d}] p4-xchg: moe_xchg {s}\n", .{ arm_i, @tagName(m.moe_xchg) }); // p4-xchg
    if (m.rank == 0) std.debug.print("[arm {d}] x-merge: decode_attn {} draft_attn {} prompt rows {?d}\n", .{ arm_i, m.decode_attn, m.draft_attn, prows });
    if (m.rank == 0) std.debug.print("[arm {d}] e-prefill: exl3 build {s} actrot {} prompt {s}\n", .{ arm_i, ink.kern.exl3_prompt_configs[m.k.exl3_cfg].name orelse ink.kern.exl3_prompt_configs[m.k.exl3_cfg].sfx[1..], m.k.exl3_actrot, if (ids2) "--ids2" else "--ids" });
    if (m.rank == 0) std.debug.print("[arm {d}] e-step: step_fuse {} dense_group {} draft graph {} draft exl3 {} draft head {s}\n", .{ arm_i, m.step_fuse, m.dense_group, if (draft) |dd| dd.dgraph_on else false, if (draft) |dd| dd.dx_on else false, if (draft) |dd| (if (dd.head_x != null) "exl3" else "bf16") else "-" });
    return .{ .prows = prows, .ids2 = ids2 };
}
// ---- end p4-small

/// --prof 2: the demand counts before a forward (stream synced).
fn demandBefore(m: *ink.model.Model, dem0: []f32) !void {
    try m.s.synchronize();
    try m.x.demand.download(0, std.mem.sliceAsBytes(dem0));
}

/// --prof 2: distinct wanted experts a layer in the forward since demandBefore (the demand's pick counts that grew).
fn distinctSince(m: *ink.model.Model, dem0: []const f32, dem1: []f32) !f64 {
    try m.s.synchronize();
    try m.x.demand.download(0, std.mem.sliceAsBytes(dem1));
    var n: usize = 0;
    var i: usize = 0;
    while (i < dem1.len) : (i += 3) n += @intFromBool(dem1[i] > dem0[i]);
    return @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(m.c.layers));
}

/// e-dense: --bench-dense. Layer 0's q / k / r / gate / o, its router and the head, at 1, 2, 4, 8 and 16 rows: the bf16
/// weights through cuBLASLt (the forward's own Gemms) and every set's EXL3 weights through dense_exl3 (core's decode
/// linear), 100 calls each between two CUDA events after 5 warm-up calls; rank 0 prints us a call and the GB/s of the
/// weight's bytes on this rank.
fn benchDense(m: *ink.model.Model, d: *const cuda.Driver, sets: []const DenseSet, rank: usize) !void {
    const s = m.s;
    const c = &m.c;
    const H: usize = c.hidden;
    const qw = m.hq() * c.head_dim;
    var x = try cuda.DeviceBuffer.alloc(d, 16 * @max(H, qw) * 2);
    defer x.free();
    try x.fill8(0x3c, s.handle); // bf16 0x3c3c: about 0.0115 everywhere
    var y = try cuda.DeviceBuffer.alloc(d, 16 * (c.vocab / m.world + 256) * 4);
    defer y.free();
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    const l = &m.layers[0];
    const Shape = struct { name: []const u8, w: u64, n: usize, k: usize, f32: bool };
    const shapes = [_]Shape{
        .{ .name = "wq_du", .w = l.wq, .n = qw, .k = H, .f32 = false },
        .{ .name = "wk_dv", .w = l.wk, .n = m.hkv() * c.head_dim, .k = H, .f32 = false },
        .{ .name = "wr_du", .w = l.wr, .n = m.hq() * c.d_rel, .k = H, .f32 = false },
        .{ .name = "gate_attn", .w = l.wgate, .n = qw, .k = H, .f32 = false },
        .{ .name = "wo_ud", .w = l.wo, .n = H, .k = qw, .f32 = true },
        .{ .name = "router", .w = l.router, .n = c.experts, .k = H, .f32 = true },
        .{ .name = "head", .w = m.unembed, .n = m.vocabLocal(), .k = H, .f32 = false },
    };
    const rows = [_]usize{ 1, 2, 4, 8, 16 };
    const iters = 100;
    for (shapes, 0..) |sh, si| {
        if (sh.w != 0) for (rows) |R| {
            for (0..5) |_| try m.g.run(s, x.ptr, sh.w, y.ptr, R, sh.n, sh.k, sh.f32);
            try e0.record(s);
            for (0..iters) |_| try m.g.run(s, x.ptr, sh.w, y.ptr, R, sh.n, sh.k, sh.f32);
            try e1.record(s);
            try e1.synchronize();
            const us = @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / iters;
            const bytes: f64 = @floatFromInt(sh.n * sh.k * 2);
            if (rank == 0) std.debug.print("[dense-bench] bf16 {s} {d}x{d} rows {d}: {d:.1} us ({d:.0} GB/s)\n", .{ sh.name, sh.k, sh.n, R, us, bytes / us / 1e3 });
        };
        const dx = m.dx orelse continue;
        for (sets) |ds| {
            const idx = ds.idx orelse continue;
            const set = &dx.sets.items[idx];
            const lw = &set.layers[0];
            const w: ?ink.dense_exl3.Weight = switch (si) {
                0 => lw.wq,
                1 => lw.wk,
                2 => lw.wr,
                3 => lw.wgate,
                4 => lw.wo,
                5 => lw.router,
                else => set.head,
            };
            const e = w orelse continue;
            try dx.activate(s, idx);
            for (rows) |R| {
                for (0..5) |_| try dx.run(s, e, x.ptr, R, y.ptr, sh.f32);
                try e0.record(s);
                for (0..iters) |_| try dx.run(s, e, x.ptr, R, y.ptr, sh.f32);
                try e1.record(s);
                try e1.synchronize();
                const us = @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / iters;
                const bytes: f64 = @floatFromInt(@as(usize, e.k) * e.n * e.halves / 16 + 2 * @as(usize, e.k) + 2 * @as(usize, e.n));
                if (rank == 0) std.debug.print("[dense-bench] {s} {s} {d}x{d} {d} bits rows {d}: {d:.1} us ({d:.0} GB/s)\n", .{ ds.name, sh.name, e.k, e.n, e.halves / 2, R, us, bytes / us / 1e3 });
            }
        }
    }
    if (m.dx) |dx| try dx.activate(s, null);
    m.graphs.drop(m.gpa);
}
