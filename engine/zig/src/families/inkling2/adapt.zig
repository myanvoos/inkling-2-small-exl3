//! A rank's residency over a request's life, TideLM's adaptive loading (docs/adaptive-experts.md) on residency.zig:
//!   boot     the fill: initial_selection per format pool from the prior (or the uniform per-layer quota)
//!   prefill  the prompt's demand is a vote; a pass when the prompt has >= 32 new tokens and misses >= the gate
//!   decode   every 600 output tokens, a pass on the answer's demand so far (no vote) when that stretch misses
//!            >= the gate
//!   end      the answer's demand is a vote; a pass
//! A pass: rank 0 plans (global_plan per pool on the blended scores), the plan goes to every rank over the TP group,
//! and every rank applies rank 0's plan in order: each swap's slot is re-filled in place (fill.Loader) and the LUT
//! entries are rewritten, so buffers (and graphs captured over them) survive. The stream is idle during a pass.
//!
//! TF_INKLING2_BG_PASS=1: the prefill->decode pass loads in the background while decode starts. Its donors (the
//! lowest-value residents) are unmapped and published before any load begins, so no forward reads a slot being
//! written; at every output token both ranks agree (the minimum over the TP group) on how many of the plan's swaps
//! have landed and map those, so both ranks route the same rows the same way. Any other pass, `end` and `begin`
//! finish it first. A layer the unmapped donors would take under the router's top-k keeps the pass synchronous.
const std = @import("std");
const cuda = @import("cuda");
const model_mod = @import("model.zig");
const fill = @import("fill.zig");
const res = @import("residency.zig");
const lazy = @import("lazy.zig"); // p4-mem
const mix = @import("mix.zig"); // x-mixed

const Model = model_mod.Model;

pub const PassKind = enum { prefill, decode, end };

pub const Stats = struct {
    passes: usize = 0,
    swaps: usize = 0,
    pass_ns: u64 = 0,
    plan_mismatch: usize = 0, // rank > 0: swaps where this rank's own plan differed from rank 0's
};

pub const Adapt = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *Model,
    loader: *fill.Loader,
    r: res.Residency,
    pools: [2][]u32,
    demand: []f32, // host copy of the device demand [L][E][3]
    since: []f32, // the demand at the last decode pass (the stretch's start)
    tokens: usize = 0,
    xmine: cuda.DeviceBuffer,
    xall: cuda.DeviceBuffer,
    stats: Stats = .{},
    log: bool,
    bg_on: bool = false,
    bg: ?Bg = null,
    agree_mine: cuda.DeviceBuffer,
    agree_all: cuda.DeviceBuffer,
    /// x-mixed: the native bank's admission (Model.mix; null otherwise): its knobs, ticks, candidates and stats.
    bank: ?*Bank = null,
    /// xa (p4-cache's begin-step word, built in xr): the mapped positions, at its save, of a long saved state likely to
    /// return (not the one this request continues); 0 none. The engine sets it on every rank before `begin`, whose
    /// reclaim then trims the lazy context to max(budget, the live length, this) and consumes it.
    trim_hint: usize = 0,

    /// A background pass: its plan's jobs (slots unmapped until they land), how many are mapped, when it began.
    const Bg = struct { kind: PassKind, jobs: []fill.Job, mapped: usize = 0, t0: std.Io.Timestamp, tokens: usize, miss: f64, w: f64 };

    const max_swaps = 1024; // two pools of at most max_loads each

    pub fn init(gpa: std.mem.Allocator, io: std.Io, m: *Model, loader: *fill.Loader, prior: []const f32, policy: res.Policy) !*Adapt {
        const a = try gpa.create(Adapt);
        errdefer gpa.destroy(a);
        var pools: [2]std.ArrayList(u32) = .{ .empty, .empty };
        // the arena's pools by release format (Config.act; x-merge: formats for the release)
        for (0..m.c.layers) |l| try pools[if (m.c.act[l] == .nvfp4) 0 else 1].append(gpa, @intCast(l));
        const n = m.c.layers * m.c.experts * res.demand_fields;
        const xbytes = (1 + 4 * max_swaps) * 4;
        a.* = .{
            .gpa = gpa,
            .io = io,
            .m = m,
            .loader = loader,
            .r = try res.Residency.init(gpa, &m.c, policy, prior),
            .pools = .{ try pools[0].toOwnedSlice(gpa), try pools[1].toOwnedSlice(gpa) },
            .demand = try gpa.alloc(f32, n),
            .since = try gpa.alloc(f32, n),
            .xmine = try cuda.DeviceBuffer.alloc(m.d, xbytes),
            .xall = try cuda.DeviceBuffer.alloc(m.d, xbytes * m.world),
            .log = m.rank == 0,
            .bg_on = if (std.c.getenv("TF_INKLING2_BG_PASS")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false,
            .agree_mine = try cuda.DeviceBuffer.alloc(m.d, 16),
            .agree_all = try cuda.DeviceBuffer.alloc(m.d, 16 * m.world),
        };
        @memset(a.demand, 0);
        @memset(a.since, 0);
        // p4-mem: the arena's memory for the context. x-merge: not in the EXL3 store, whose slots are exl3.zig's (every
        // expert resident, the arena empty): a growth past MemAvailable is refused there (lazy.ensure), never given
        // x-mixed: the mixed store's arena is its native bank, which gives slots back (each pick of a given-back expert
        // falls back to its EXL3 trellis), and no layer keeps a floor of residents (nothing is ever substituted)
        if (m.lazy) |lz| if (!m.c.exl3 or m.mix) {
            lz.giveback = .{ .ctx = a, .f = giveBackFn };
        };
        if (m.mix) {
            a.r.floor = 0;
            try a.bankInit();
        }
        return a;
    }

    pub fn deinit(a: *Adapt) void {
        if (a.bank != null) { // x-mixed: no collective at exit (the ranks may stop apart): the batch joined and dropped
            _ = a.loader.joinBackground();
            if (a.bg) |bg| a.gpa.free(bg.jobs);
            a.bg = null;
        }
        a.settle() catch |err| std.log.err("inkling2: the background residency pass failed: {t}", .{err});
        if (a.bank) |b| { // x-mixed
            a.gpa.free(b.eligible);
            a.gpa.destroy(b);
        }
        if (a.m.lazy) |lz| lz.giveback = null; // p4-mem
        a.agree_mine.free();
        a.agree_all.free();
        a.r.deinit();
        a.gpa.free(a.pools[0]);
        a.gpa.free(a.pools[1]);
        a.gpa.free(a.demand);
        a.gpa.free(a.since);
        a.xmine.free();
        a.xall.free();
        a.gpa.destroy(a);
    }

    fn now(a: *const Adapt) std.Io.Timestamp {
        return std.Io.Clock.awake.now(a.io);
    }

    /// The boot fill: `per` experts in every layer (the uniform quota) when given, else initial_selection of each
    /// pool's resident slots, from the blended scores (the prior alone on a cold history).
    pub fn fillBoot(a: *Adapt, per: ?usize) !void {
        const m = a.m;
        const sc = try a.gpa.alloc(f64, m.c.layers * m.c.experts);
        defer a.gpa.free(sc);
        _ = try a.r.scores(sc, null);
        for (0..2) |fi| {
            if (a.pools[fi].len == 0) continue;
            const sel = if (per) |p| try a.r.uniform(a.gpa, sc, a.pools[fi], p) else try a.r.initial(a.gpa, sc, a.pools[fi], m.arena.resident[fi]);
            defer a.gpa.free(sel);
            if (sel.len > m.arena.resident[fi]) return error.ArenaTooSmall;
            const jobs = try a.gpa.alloc(fill.Job, sel.len);
            defer a.gpa.free(jobs);
            for (sel, jobs, 0..) |p, *j, slot| j.* = .{ .layer = p[0], .expert = p[1], .slot = @intCast(slot) };
            try a.loader.load(m, jobs);
        }
        try m.publish();
        try m.s.synchronize();
    }

    /// Experts resident in layer `l` (the arena's resident slots only).
    pub fn residentIn(a: *const Adapt, l: usize) usize {
        const E = a.m.c.experts;
        var n: usize = 0;
        for (a.m.arena.host_routable[l * E ..][0..E]) |r| n += r;
        return n;
    }

    fn pull(a: *Adapt) !void {
        try a.m.s.synchronize();
        try a.m.x.demand.download(0, std.mem.sliceAsBytes(a.demand));
    }

    fn zero(a: *Adapt) !void {
        try a.m.x.demand.fill8(0, a.m.s.handle);
        try a.m.s.synchronize();
        @memset(a.since, 0);
        @memset(a.demand, 0);
    }

    /// A new request: its demand starts at zero.
    pub fn begin(a: *Adapt) !void {
        try a.settle();
        try a.zero();
        a.tokens = 0;
        if (a.bank) |b| { // x-mixed: a request's counters
            b.rounds = 0;
            b.stats.picks = .{ .{ 0, 0 }, .{ 0, 0 } };
            b.stats.mass = .{ .{ 0, 0 }, .{ 0, 0 } };
            b.stats.fmt_picks = .{ .{ .{ 0, 0 }, .{ 0, 0 } }, .{ .{ 0, 0 }, .{ 0, 0 } } };
            b.read0 = a.loader.stats.bytes;
            b.loads0 = b.stats.loads;
        }
        try a.reclaim(); // p4-mem: the lazy context back to its budget, the arena's given-back slots mapped again
    }

    /// The prefill->decode boundary after `prompt_new` new prompt tokens.
    pub fn afterPrefill(a: *Adapt, prompt_new: usize) !void {
        if (a.bank != null) return a.bankPrefill(prompt_new); // x-mixed
        try a.pull();
        const miss = res.missRate(a.demand, null);
        if (a.r.policy.adapt and prompt_new > 0) {
            try a.r.vote(a.demand);
            if (prompt_new >= a.r.policy.min_prompt and miss.rate >= a.r.policy.miss_gate) _ = try a.passMode(.prefill, null, miss.rate, a.bg_on and !a.m.exact);
        }
        if (a.log) std.debug.print("[residency] prefill {d} tokens: miss {d:.4} ({d:.0} of {d:.0} picks substituted)\n", .{ prompt_new, miss.rate, miss.missed, miss.picks });
        try a.zero();
    }

    /// After each output token.
    pub fn afterToken(a: *Adapt) !void {
        a.tokens += 1;
        if (a.bank != null) return; // x-mixed: the bank advances and ticks a round (afterRound)
        if (a.bg != null) try a.advance(false);
        const every = a.r.policy.decode_tokens;
        if (!a.r.policy.adapt or every == 0 or a.tokens % every != 0) return;
        try a.settle();
        try a.pull();
        const miss = res.missRate(a.demand, a.since);
        if (miss.rate >= a.r.policy.miss_gate) {
            if (a.log) std.debug.print("[residency] decode adaptation at output token {d} (stretch miss {d:.4})\n", .{ a.tokens, miss.rate });
            _ = try a.pass(.decode, a.demand, miss.rate);
        }
        @memcpy(a.since, a.demand);
    }

    /// The request's end: its answer's demand is a vote, then a pass.
    pub fn end(a: *Adapt) !void {
        if (a.bank != null) return a.bankEnd(); // x-mixed
        try a.settle();
        try a.pull();
        const miss = res.missRate(a.demand, null);
        if (a.log) std.debug.print("[residency] answer {d} tokens: miss {d:.4} ({d:.0} of {d:.0} picks substituted)\n", .{ a.tokens, miss.rate, miss.missed, miss.picks });
        if (a.r.policy.adapt and a.tokens > 0) {
            try a.r.vote(a.demand);
            _ = try a.pass(.end, null, miss.rate);
        }
        try a.zero();
    }

    /// Every layer keeps at least the router's top-k routable experts with all of `jobs`' donors unmapped at once.
    fn canDefer(a: *const Adapt, jobs: []const fill.Job) bool {
        const m = a.m;
        const E = m.c.experts;
        const out = a.gpa.alloc(usize, m.c.layers) catch return false;
        defer a.gpa.free(out);
        @memset(out, 0);
        for (jobs) |j| {
            const fi = fmtIndex(m.c.act[j.layer]);
            const owner = m.arena.owner[fi][j.slot];
            if (owner < 0) continue;
            out[@as(usize, @intCast(owner)) / E] += 1;
        }
        for (0..m.c.layers) |l| if (a.residentIn(l) < out[l] + m.c.topk) return false;
        return true;
    }

    /// A background pass's landed swaps mapped on every rank: the count both ranks have (the TP group's minimum), or
    /// with `drain` every swap (the load joined first). Publishes what it maps.
    fn advance(a: *Adapt, drain: bool) !void {
        const b = if (a.bg) |*x| x else return;
        const m = a.m;
        var n = a.loader.landed();
        var failed: ?anyerror = null;
        if (drain) {
            if (a.loader.finishBackground()) |jobs| {
                if (jobs) |j| a.gpa.free(j);
                n = b.jobs.len;
            } else |err| failed = err;
        } else if (m.world > 1) {
            var mine = [4]u32{ @intCast(n), 0, 0, 0 };
            try a.agree_mine.uploadAsync(0, std.mem.sliceAsBytes(&mine), m.s.handle);
            try m.group.?.allGather(m.s, a.agree_mine.ptr, a.agree_all.ptr, 16);
            try m.s.synchronize();
            var all: [64]u32 = undefined;
            try a.agree_all.download(0, std.mem.sliceAsBytes(all[0 .. 4 * m.world]));
            for (0..m.world) |r| n = @min(n, all[4 * r]);
        }
        if (n > b.mapped) {
            for (b.jobs[b.mapped..n]) |j| m.place(j.layer, j.expert, fmtIndex(m.c.act[j.layer]), j.slot);
            try m.s.synchronize();
            try m.publish();
            b.mapped = n;
        }
        if (failed) |err| {
            a.gpa.free(b.jobs);
            a.bg = null;
            return err;
        }
        if (b.mapped == b.jobs.len) {
            if (!drain) {
                if (try a.loader.finishBackground()) |j| a.gpa.free(j);
            }
            const ns: u64 = @intCast(b.t0.durationTo(a.now()).toNanoseconds());
            if (a.log) std.debug.print("[residency] {s} pass (background): {d} swaps landed {d:.0} ms after the plan, over {d} output tokens\n", .{ @tagName(b.kind), b.jobs.len, @as(f64, @floatFromInt(ns)) / 1e6, a.tokens - b.tokens });
            a.gpa.free(b.jobs);
            a.bg = null;
        }
    }

    /// A background pass finished: every swap landed and mapped (both ranks call it at the same step).
    pub fn settle(a: *Adapt) !void {
        if (a.bank != null) return a.bankSettle(); // x-mixed: every rank maps what every rank landed
        if (a.bg != null) try a.advance(true);
    }

    /// One pass: rank 0's plan, applied on every rank. Returns the swaps applied.
    pub fn pass(a: *Adapt, kind: PassKind, pending: ?[]const f32, miss: f64) !usize {
        return a.passMode(kind, pending, miss, false);
    }

    /// `pass`, its loads in the background when `background` (and no layer would fall under top-k meanwhile).
    fn passMode(a: *Adapt, kind: PassKind, pending: ?[]const f32, miss: f64, background: bool) !usize {
        try a.settle();
        const m = a.m;
        const t0 = a.now();
        const E = m.c.experts;
        try m.s.synchronize();
        const sc = try a.gpa.alloc(f64, m.c.layers * E);
        defer a.gpa.free(sc);
        const w = try a.r.scores(sc, pending);
        var mine: std.ArrayList(res.Swap) = .empty;
        defer mine.deinit(a.gpa);
        for (0..2) |fi| {
            if (a.pools[fi].len == 0) continue;
            const sw = try a.r.plan(a.gpa, m.arena.host_routable, sc, a.pools[fi]);
            defer a.gpa.free(sw);
            try mine.appendSlice(a.gpa, sw);
        }
        // rank 0's plan to every rank
        var words: [1 + 4 * max_swaps]i32 = undefined;
        const n_mine = @min(mine.items.len, max_swaps);
        words[0] = @intCast(n_mine);
        for (mine.items[0..n_mine], 0..) |s, i| words[1 + 4 * i ..][0..4].* = .{ @intCast(s.lo), @intCast(s.eo), @intCast(s.li), @intCast(s.ei) };
        var plan: [1 + 4 * max_swaps]i32 = words;
        if (m.world > 1) {
            const bytes = (1 + 4 * max_swaps) * 4;
            try a.xmine.upload(0, std.mem.sliceAsBytes(&words));
            try m.group.?.allGather(m.s, a.xmine.ptr, a.xall.ptr, bytes);
            try m.s.synchronize();
            try a.xall.download(0, std.mem.sliceAsBytes(&plan)); // rank 0's segment comes first
            if (m.rank != 0) {
                const n0: usize = @intCast(plan[0]);
                if (n0 != n_mine or !std.mem.eql(i32, plan[1 .. 1 + 4 * n0], words[1 .. 1 + 4 * n0])) a.stats.plan_mismatch += 1;
            }
        }
        const n: usize = @intCast(plan[0]);
        const jobs = try a.gpa.alloc(fill.Job, n);
        defer a.gpa.free(jobs);
        for (jobs, 0..) |*j, i| {
            const q = plan[1 + 4 * i ..][0..4];
            const lo: usize = @intCast(q[0]);
            const eo: usize = @intCast(q[1]);
            const slot = m.arena.host_lut[lo * E + eo];
            if (slot < 0 or m.c.act[lo] != m.c.act[@intCast(q[2])]) return error.BadPlan;
            j.* = .{ .layer = @intCast(q[2]), .expert = @intCast(q[3]), .slot = @intCast(slot) };
        }
        if (background and n > 0 and a.canDefer(jobs)) {
            // the donors unmapped and published (the stream is idle), then the loads start on the loader's thread
            for (jobs) |j| m.evict(fmtIndex(m.c.act[j.layer]), j.slot);
            try m.publish();
            try m.s.synchronize();
            var context: cuda.abi.Context = null;
            try m.d.check(m.d.api.cuCtxGetCurrent(&context), "cuCtxGetCurrent");
            const own = try a.gpa.dupe(fill.Job, jobs);
            errdefer a.gpa.free(own);
            try a.loader.startBackground(m.d, context, fill.Target.ofArena(&m.arena), own);
            a.bg = .{ .kind = kind, .jobs = own, .t0 = t0, .tokens = a.tokens, .miss = miss, .w = w };
            const ns: u64 = @intCast(t0.durationTo(a.now()).toNanoseconds());
            a.stats.passes += 1;
            a.stats.swaps += n;
            a.stats.pass_ns += ns;
            if (a.log) std.debug.print("[residency] {s} pass: miss {d:.4}, observed weight {d:.3}, {d} swaps planned in {d:.0} ms, loading in the background\n", .{ @tagName(kind), miss, w, n, @as(f64, @floatFromInt(ns)) / 1e6 });
            return n;
        }
        const seen = a.loader.stats.min_avail;
        a.loader.stats.min_avail = std.math.maxInt(u64);
        try a.loader.load(m, jobs);
        try m.publish();
        try m.s.synchronize();
        const ns: u64 = @intCast(t0.durationTo(a.now()).toNanoseconds());
        const low = a.loader.stats.min_avail;
        a.loader.stats.min_avail = @min(seen, low);
        a.stats.passes += 1;
        a.stats.swaps += n;
        a.stats.pass_ns += ns;
        if (a.log) std.debug.print("[residency] {s} pass: miss {d:.4}, observed weight {d:.3}, {d} swaps in {d:.0} ms ({d:.2} ms a swap; MemAvailable min {d:.2} GiB, RSS {d:.2} GiB)\n", .{ @tagName(kind), miss, w, n, @as(f64, @floatFromInt(ns)) / 1e6, @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(@max(n, 1))), @as(f64, @floatFromInt(if (n > 0) low else (fill.readAvail() orelse 0))) / (1 << 30), @as(f64, @floatFromInt(rss())) / (1 << 30) });
        return n;
    }

    // ---- p4-mem: the arena's memory and the lazy context (lazy.zig, TF_INKLING2_LAZY_CONTEXT=1)

    /// A give-back's or a reclaim's plan: [ok, slots of NVFP4, slots of MXFP8, experts a layer, (layer, expert) ...],
    /// rank 0's, crossing the TP group in the pass's exchange buffers.
    const plan_words = 1 + 4 * max_swaps;
    const plan_most = (plan_words - 4) / 2; // (layer, expert) pairs a plan holds

    fn giveBackFn(ctx: *anyopaque, short: u64) anyerror!void {
        const a: *Adapt = @ptrCast(@alignCast(ctx));
        return a.giveBack(short);
    }

    /// Rank 0's `plan` on every rank (every rank calls it at the same step).
    fn share(a: *Adapt, plan: *[plan_words]i32) !void {
        const m = a.m;
        if (m.world == 1) return;
        try a.xmine.upload(0, std.mem.sliceAsBytes(plan));
        try m.group.?.allGather(m.s, a.xmine.ptr, a.xall.ptr, plan_words * 4);
        try m.s.synchronize();
        try a.xall.download(0, std.mem.sliceAsBytes(plan)); // rank 0's segment comes first
    }

    /// lazy.zig's give-back: at least `short` bytes of the arena released for a growing context, alike on every rank.
    /// Rank 0 plans (giveBackPlan) and every rank applies its plan: the victims evicted, the residents of the pool's
    /// top slots moved down into the slots the victims left (device copies of their packed words, the LUT re-pointed),
    /// the chunks past the new top unmapped (lazy.arenaShrink). error.ArenaAtFloor when the router's top-k in every
    /// layer leaves too little to give.
    fn giveBack(a: *Adapt, short: u64) !void {
        const m = a.m;
        const lz = m.lazy.?;
        const E = m.c.experts;
        const t0 = a.now();
        try a.settle();
        // exact mode's transient ring sits past the resident slots: no give-back beside it (every rank alike)
        for (0..2) |f| if (m.arena.resident[f] != m.arena.slots(f)) return error.TransientRing;
        var plan: [plan_words]i32 = @splat(0);
        // a plan that fails on rank 0 still crosses (ok 0): the other ranks wait for it
        if (m.rank == 0) a.giveBackPlan(short, &plan) catch |err| {
            std.log.err("inkling2: the context's give-back plan failed: {t}", .{err});
            plan[0] = 0;
        };
        try a.share(&plan);
        if (plan[0] <= 0) return error.ArenaAtFloor;
        try m.s.synchronize();
        const avail0 = fill.readAvail() orelse 0;
        var at: usize = 4;
        var moved: usize = 0;
        var freed: u64 = 0;
        for (0..2) |f| {
            const nv: usize = @intCast(plan[1 + f]);
            if (nv == 0) continue;
            const n = m.arena.slots(f);
            if (nv > n) return error.BadPlan;
            const top = n - nv;
            // every victim unmapped; those under the new top leave holes
            var holes: std.ArrayList(usize) = .empty;
            defer holes.deinit(a.gpa);
            for (0..nv) |_| {
                const l: usize = @intCast(plan[at]);
                const e: usize = @intCast(plan[at + 1]);
                at += 2;
                const slot = m.arena.host_lut[l * E + e];
                if (slot < 0 or fmtIndex(m.c.act[l]) != f) return error.BadPlan;
                const sl: usize = @intCast(slot);
                if (sl < top) try holes.append(a.gpa, sl);
                m.evict(f, sl);
            }
            std.mem.sort(usize, holes.items, {}, std.sort.asc(usize));
            // the residents past the new top into the holes, in slot order
            var hi: usize = 0;
            for (top..n) |sl| {
                const o = m.arena.owner[f][sl];
                if (o < 0) continue;
                if (hi >= holes.items.len) return error.BadPlan;
                const dst = holes.items[hi];
                hi += 1;
                try copySlot(m, f, sl, dst);
                const own: usize = @intCast(o);
                m.evict(f, sl);
                m.place(own / E, own % E, f, dst);
                moved += 1;
            }
            try m.s.synchronize(); // the copies landed before their sources are unmapped
            freed += lz.arenaShrink(&m.arena, f, top);
        }
        try m.publish();
        m.graphs.drop(m.gpa);
        const ns: u64 = @intCast(t0.durationTo(a.now()).toNanoseconds());
        if (a.log) std.debug.print("[residency] context: {d} experts a layer given back ({d} NVFP4 + {d} MXFP8 slots, {d} moved, {d:.2} GiB) in {d:.0} ms; MemAvailable {d:.2} -> {d:.2} GiB\n", .{ plan[3], plan[1], plan[2], moved, @as(f64, @floatFromInt(freed)) / (1 << 30), @as(f64, @floatFromInt(ns)) / 1e6, @as(f64, @floatFromInt(avail0)) / (1 << 30), @as(f64, @floatFromInt(fill.readAvail() orelse 0)) / (1 << 30) });
    }

    /// Rank 0's give-back plan into `plan` (ok 0: none possible): `k` experts a layer fewer in each pool (the arena's
    /// own sizing unit), the fewest whose chunks release `short`; the victims each pool's lowest-value residents on the
    /// scores blended with this request's demand so far (score, then layer, then id ascending), no layer under the
    /// router's top-k.
    fn giveBackPlan(a: *Adapt, short: u64, plan: *[plan_words]i32) !void {
        const m = a.m;
        const lz = m.lazy.?;
        const E = m.c.experts;
        const L = m.c.layers;
        try a.pull();
        const sc = try a.gpa.alloc(f64, L * E);
        defer a.gpa.free(sc);
        _ = try a.r.scores(sc, a.demand);
        const n = [2]usize{ m.arena.slots(0), m.arena.slots(1) };
        const w = m.arena.words;
        var k: usize = 1;
        var cut: [2]usize = undefined;
        while (true) : (k += 1) {
            cut = .{ @min(k * a.pools[0].len, n[0]), @min(k * a.pools[1].len, n[1]) };
            if (cut[0] + cut[1] > plan_most) return;
            if (lz.arenaFreed(w, 0, n[0] - cut[0]) + lz.arenaFreed(w, 1, n[1] - cut[1]) >= short) break;
            if (cut[0] == n[0] and cut[1] == n[1]) return;
        }
        const counts = try a.gpa.alloc(usize, L);
        defer a.gpa.free(counts);
        for (counts, 0..) |*c, l| c.* = a.residentIn(l);
        const Cand = struct { s: f64, l: u32, e: u32 };
        var at: usize = 4;
        for (0..2) |f| {
            var cands: std.ArrayList(Cand) = .empty;
            defer cands.deinit(a.gpa);
            for (a.pools[f]) |l| for (0..E) |e| if (m.arena.host_routable[l * E + e] != 0) try cands.append(a.gpa, .{ .s = sc[l * E + e], .l = l, .e = @intCast(e) });
            std.mem.sort(Cand, cands.items, {}, struct {
                fn lt(_: void, x: Cand, y: Cand) bool {
                    if (x.s != y.s) return x.s < y.s;
                    if (x.l != y.l) return x.l < y.l;
                    return x.e < y.e;
                }
            }.lt);
            var got: usize = 0;
            for (cands.items) |c| {
                if (got == cut[f]) break;
                if (counts[c.l] <= a.r.floor) continue;
                counts[c.l] -= 1;
                plan[at] = @intCast(c.l);
                plan[at + 1] = @intCast(c.e);
                at += 2;
                got += 1;
            }
            if (got < cut[f]) return;
            plan[1 + f] = @intCast(got);
        }
        plan[3] = @intCast(k);
        plan[0] = 1;
    }

    /// Slot `src` of pool `f` copied into slot `dst` on the model's stream (packed words; NVFP4 also its scales).
    fn copySlot(m: *Model, f: usize, src: usize, dst: usize) !void {
        const ar = &m.arena;
        const w = ar.words;
        const s = m.s.handle;
        if (f == 0) {
            try ar.nv_up.copyFrom(dst * w[0][0] * 4, ar.nv_up.ptr + src * w[0][0] * 4, w[0][0] * 4, s);
            try ar.nv_down.copyFrom(dst * w[0][1] * 4, ar.nv_down.ptr + src * w[0][1] * 4, w[0][1] * 4, s);
            try ar.nv_up_scale.copyFrom(dst * 8, ar.nv_up_scale.ptr + src * 8, 8, s);
            try ar.nv_down_scale.copyFrom(dst * 4, ar.nv_down_scale.ptr + src * 4, 4, s);
        } else {
            try ar.mx_up.copyFrom(dst * w[1][0] * 4, ar.mx_up.ptr + src * w[1][0] * 4, w[1][0] * 4, s);
            try ar.mx_down.copyFrom(dst * w[1][1] * 4, ar.mx_down.ptr + src * w[1][1] * 4, w[1][1] * 4, s);
        }
    }

    /// A new sequence (begin): the lazy context unmapped back to its budget (lazy.trim; never under the live
    /// sequence), then the arena's given-back slots mapped again as far as MemAvailable allows over the floor and the
    /// margin on every rank (at most the slots it had at load: the room a growth gave back returns), filled with each
    /// pool's best absent experts (rank 0's choice on the blended scores, loaded as a pass's swaps are).
    fn reclaim(a: *Adapt) !void {
        const m = a.m;
        const lz = m.lazy orelse return;
        // xa: p4-cache's hint keeps a long state that is likely to return mapped (the trim to max(budget, live, hint));
        // `held` is the context memory it keeps beyond the live length's trim, logged with the bank slots it held back
        const hint = a.trim_hint;
        a.trim_hint = 0;
        const tgt = struct { // lazy.trim's own target: a multiple of the step, which need not be a power of two
            fn of(l: *const lazy.Lazy, keep: usize) usize {
                return @min(l.mapped, @max(l.budget, @min(l.context, (keep + l.step - 1) / l.step * l.step)));
            }
        };
        const held: u64 = if (hint > m.state.len) lz.bytesAt(tgt.of(lz, hint)) -| lz.bytesAt(tgt.of(lz, m.state.len)) else 0;
        try lz.trim(m, @max(m.state.len, hint));
        if (hint > 0 and a.log) std.debug.print("[bank] the trim keeps {d} positions mapped (a returning state's hint; {d:.2} GiB of context held)\n", .{ lz.mapped, @as(f64, @floatFromInt(held)) / (1 << 30) });
        if (!lz.arena_on) return;
        const n = [2]usize{ m.arena.slots(0), m.arena.slots(1) };
        if (n[0] >= lz.arena_boot[0] and n[1] >= lz.arena_boot[1]) return;
        for (0..2) |f| if (m.arena.resident[f] != m.arena.slots(f)) return; // exact mode's ring: none given back
        const t0 = a.now();
        const w = m.arena.words;
        const avail0 = fill.readAvail() orelse 0;
        // x-mixed: the bank maps back only what leaves MemAvailable a band over its watch's threshold (bankKeep), so the
        // next request's watch does not shed it again (without the band the two took turns: shed, mapped back, shed)
        const keep = if (a.bank != null) a.bankKeep(lz) else lz.margin;
        const room = try lazy.Lazy.agreeMin(m, avail0 -| (lz.floor + keep));
        var k: usize = 0;
        var to = n;
        while (true) {
            const next = [2]usize{ @min(n[0] + (k + 1) * a.pools[0].len, lz.arena_boot[0]), @min(n[1] + (k + 1) * a.pools[1].len, lz.arena_boot[1]) };
            if (next[0] == to[0] and next[1] == to[1]) break;
            if (next[0] - n[0] + next[1] - n[1] > plan_most) break;
            if (lz.arenaAdded(w, 0, next[0]) + lz.arenaAdded(w, 1, next[1]) > room) break;
            k += 1;
            to = next;
        }
        if (held > 0 and a.log) { // xa: the experts a layer the hint held back: the same steps on the room without it
            var k2: usize = 0;
            var to2 = n;
            while (true) {
                const nx = [2]usize{ @min(n[0] + (k2 + 1) * a.pools[0].len, lz.arena_boot[0]), @min(n[1] + (k2 + 1) * a.pools[1].len, lz.arena_boot[1]) };
                if (nx[0] == to2[0] and nx[1] == to2[1]) break;
                if (nx[0] - n[0] + nx[1] - n[1] > plan_most) break;
                if (lz.arenaAdded(w, 0, nx[0]) + lz.arenaAdded(w, 1, nx[1]) > room + held) break;
                k2 += 1;
                to2 = nx;
            }
            std.debug.print("[bank] the hint held back {d} experts a layer of the bank's map-back ({d} mapped back)\n", .{ k2 -| k, k });
        }
        if (k == 0) return;
        var plan: [plan_words]i32 = @splat(0);
        if (m.rank == 0) a.reclaimPlan(.{ to[0] - n[0], to[1] - n[1] }, &plan) catch |err| {
            std.log.err("inkling2: the arena's reclaim plan failed: {t}", .{err});
            plan[0] = 0;
        };
        try a.share(&plan);
        if (plan[0] <= 0) return;
        // mapped on every rank, or on none
        var failed = false;
        for (0..2) |f| if (to[f] > n[f]) lz.arenaGrow(&m.arena, f, to[f]) catch {
            failed = true;
        };
        if (try lazy.Lazy.agreeMax(m, @intFromBool(failed)) != 0) {
            for (0..2) |f| _ = lz.arenaShrink(&m.arena, f, n[f]);
            std.log.warn("inkling2 rank {d}: mapping the arena's given-back slots failed on a rank; it stays at {d} + {d}", .{ m.rank, n[0], n[1] });
            return;
        }
        const jobs = try a.gpa.alloc(fill.Job, plan_most);
        defer a.gpa.free(jobs);
        var at: usize = 4;
        var j: usize = 0;
        for (0..2) |f| for (0..@as(usize, @intCast(plan[1 + f]))) |i| {
            jobs[j] = .{ .layer = @intCast(plan[at]), .expert = @intCast(plan[at + 1]), .slot = @intCast(n[f] + i) };
            at += 2;
            j += 1;
        };
        if (a.bank) |b| if (!b.o.reclaim_sync) {
            // xa: the map-back as a background batch: the slots are new and empty (no donor), loaded on the loader's
            // stream and mapped as they land (bankAdvance / bankSettle: the least every rank landed), so the request
            // starts at once; LAND_WAIT (a gate) lands it here, the synchronous load's result
            { // the jobs' copy is the batch's from here (bankSettle frees it)
                const own = try a.gpa.dupe(fill.Job, jobs[0..j]);
                errdefer a.gpa.free(own);
                var context: cuda.abi.Context = null;
                try m.d.check(m.d.api.cuCtxGetCurrent(&context), "cuCtxGetCurrent");
                try a.loader.startBackground(m.d, context, fill.Target.ofArena(&m.arena), own);
                a.bg = .{ .kind = .decode, .jobs = own, .t0 = a.now(), .tokens = a.tokens, .miss = 0, .w = 0 };
            }
            b.stats.batches += 1;
            b.stats.planned += j;
            b.stats.reclaims += 1;
            b.stats.reclaimed += j;
            m.graphs.drop(m.gpa);
            const ns0: u64 = @intCast(t0.durationTo(a.now()).toNanoseconds());
            if (a.log) std.debug.print("[residency] context: {d} experts a layer mapping back in the background ({d} NVFP4 + {d} MXFP8 slots; started in {d:.0} ms); MemAvailable {d:.2} -> {d:.2} GiB\n", .{ k, plan[1], plan[2], @as(f64, @floatFromInt(ns0)) / 1e6, @as(f64, @floatFromInt(avail0)) / (1 << 30), @as(f64, @floatFromInt(fill.readAvail() orelse 0)) / (1 << 30) });
            if (b.o.land_wait) try a.bankSettle();
            return;
        };
        try a.loader.load(m, jobs[0..j]);
        try m.publish();
        try m.s.synchronize();
        m.graphs.drop(m.gpa);
        const ns: u64 = @intCast(t0.durationTo(a.now()).toNanoseconds());
        if (a.log) std.debug.print("[residency] context: {d} experts a layer mapped back ({d} NVFP4 + {d} MXFP8 slots loaded) in {d:.0} ms; MemAvailable {d:.2} -> {d:.2} GiB\n", .{ k, plan[1], plan[2], @as(f64, @floatFromInt(ns)) / 1e6, @as(f64, @floatFromInt(avail0)) / (1 << 30), @as(f64, @floatFromInt(fill.readAvail() orelse 0)) / (1 << 30) });
    }

    /// Rank 0's reclaim choice into `plan`: each pool's best absent experts on the blended scores (score descending,
    /// then layer, then id ascending), `want` [f] of them.
    fn reclaimPlan(a: *Adapt, want: [2]usize, plan: *[plan_words]i32) !void {
        const m = a.m;
        const E = m.c.experts;
        const sc = try a.gpa.alloc(f64, m.c.layers * E);
        defer a.gpa.free(sc);
        _ = try a.r.scores(sc, null);
        const Cand = struct { s: f64, l: u32, e: u32 };
        var at: usize = 4;
        for (0..2) |f| {
            var cands: std.ArrayList(Cand) = .empty;
            defer cands.deinit(a.gpa);
            for (a.pools[f]) |l| for (0..E) |e| if (m.arena.host_routable[l * E + e] == 0) try cands.append(a.gpa, .{ .s = sc[l * E + e], .l = l, .e = @intCast(e) });
            std.mem.sort(Cand, cands.items, {}, struct {
                fn lt(_: void, x: Cand, y: Cand) bool {
                    if (x.s != y.s) return x.s > y.s;
                    if (x.l != y.l) return x.l < y.l;
                    return x.e < y.e;
                }
            }.lt);
            const take = @min(want[f], cands.items.len);
            for (cands.items[0..take]) |c| {
                plan[at] = @intCast(c.l);
                plan[at + 1] = @intCast(c.e);
                at += 2;
            }
            plan[1 + f] = @intCast(take);
        }
        plan[0] = 1;
    }

    // ---- x-mixed: the native bank's admission (TF_INKLING2_MIX=1; out/prefill/p5/x-mixed/DESIGN.md section 3)
    // The mixed store's arena is a cache of the release's native experts over the EXL3 store (every expert resident): a
    // pick the bank does not hold runs that expert's EXL3 trellis, so nothing is ever substituted, no layer keeps a floor
    // and no pass runs. The router's own picks are observed (the demand; field [2]: those the bank did not hold), voted
    // into the residency history (a prompt with TF_INKLING2_MIX_PROMPT_WEIGHT, a decode tick with 1), and at a prompt's
    // end and every TF_INKLING2_MIX_ADMIT_EVERY verify rounds rank 0 plans a batch (planBank), which every rank applies:
    // the donors unmapped and published at once (their picks fall back to EXL3), the loads in the background
    // (fill.Loader), each mapped once every rank has it (bankAdvance: the least every rank has landed). A failed load on
    // a rank leaves the same experts native on every rank (bankSettle). Every step below runs on every rank at the same
    // engine step (the collectives' order).

    pub const BankStats = struct {
        ticks: usize = 0,
        batches: usize = 0,
        planned: usize = 0, // experts planned (a failed batch maps fewer)
        loads: usize = 0, // experts mapped
        evictions: usize = 0, // donors unmapped (an empty slot is not one)
        failures: usize = 0, // batches some rank did not finish
        sheds: usize = 0, // the watchdog's give-backs
        reclaims: usize = 0, // xa: begin-time map-backs started in the background
        reclaimed: usize = 0, // xa: their slots
        /// A request's picks (demand field [0]) and their score mass ([1]), all and those the bank served:
        /// [prompt, answer][all, native].
        picks: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } },
        mass: [2][2]f64 = .{ .{ 0, 0 }, .{ 0, 0 } },
        /// The picks by the layer's format: [prompt, answer][NVFP4 layers, MXFP8 layers][all, native].
        fmt_picks: [2][2][2]f64 = .{ .{ .{ 0, 0 }, .{ 0, 0 } }, .{ .{ 0, 0 }, .{ 0, 0 } } },
    };

    pub const Bank = struct {
        o: mix.Opts,
        rounds: usize = 0, // verify rounds (or decode steps) since the request's begin
        eligible: []u8, // [L][E]: picked and served by EXL3 since the last plan (the router-triggered candidates)
        stats: BankStats = .{},
        read0: u64 = 0, // the loader's bytes and the bank's loads at the request's begin
        loads0: usize = 0,
        id: ?[3][32]u8 = null, // the history's identity (bankHistory), when TF_INKLING2_MIX_HISTORY names a file
        hist_ok: bool = false, // the history file may be written (it was absent, or this run's own)
        saves: usize = 0,
    };

    fn bankInit(a: *Adapt) !void {
        const b = try a.gpa.create(Bank);
        errdefer a.gpa.destroy(b);
        b.* = .{ .o = mix.Opts.fromEnv(), .eligible = try a.gpa.alloc(u8, a.m.c.layers * a.m.c.experts) };
        @memset(b.eligible, 0);
        a.r.policy.half_life = b.o.half_life; // in votes: a decode tick or a prompt each
        a.bank = b;
        if (a.log and b.o.admit) std.debug.print("[bank] admission mode: {s}{s}\n", .{ if (b.o.between) "between requests only (TF_INKLING2_MIX_ADMIT=between: a batch at each request's end, landed by the next begin)" else "live (ticks during a request)", if (b.o.land_wait) "; every batch landed before the next forward (TF_INKLING2_MIX_LAND_WAIT=1, a gate)" else "" });
        if (a.log) std.debug.print("[bank] the native bank's admission: {s}, every {d} rounds, at most {d} experts a batch and {d} a layer, a {d:.0}% margin over the donor, half-life {d:.0} votes, prompt weight {d:.2}, watchdog at the floor + {d:.1} GiB; MXFP8 layers x{d:.2} the experts a layer of the NVFP4 layers{s}\n", .{ if (b.o.admit) "on" else "off (TF_INKLING2_MIX_ADMIT=0: the boot fill stays)", b.o.every, b.o.reads, b.o.per_layer, b.o.gain * 100, b.o.half_life, b.o.prompt_weight, b.o.watch_gib, b.o.mx_priority orelse 1, if (b.o.mx_share != null) " (TF_INKLING2_MIX_MX_SHARE sets the bytes instead)" else if (b.o.mx_priority != null) " (TF_INKLING2_MIX_MX_PRIORITY)" else "" });
    }

    /// The pulled demand's picks and score mass, all and native (the picks the bank held: field [0] less field [2], the
    /// mass in the same share), into the request's prompt (0) or answer (1) counters; its EXL3-served experts become
    /// admission candidates.
    fn bankCount(a: *Adapt, part: usize) void {
        const b = a.bank.?;
        var i: usize = 0;
        while (i < a.demand.len) : (i += res.demand_fields) {
            const p: f64 = a.demand[i];
            if (p <= 0) continue;
            const nat = p - a.demand[i + 2];
            b.stats.picks[part][0] += p;
            b.stats.picks[part][1] += nat;
            b.stats.mass[part][0] += a.demand[i + 1];
            b.stats.mass[part][1] += a.demand[i + 1] * nat / p;
            const f = fmtIndex(a.m.c.act[(i / res.demand_fields) / a.m.c.experts]);
            b.stats.fmt_picks[part][f][0] += p;
            b.stats.fmt_picks[part][f][1] += nat;
            if (a.demand[i + 2] > 0) b.eligible[i / res.demand_fields] = 1;
        }
    }

    /// A tick (the stream idle: it follows a step's readback): the demand since the last tick pulled, counted, voted
    /// with weight `w`, zeroed; then, with admission on and no batch in flight, a batch planned and applied.
    fn bankTick(a: *Adapt, part: usize, w: f64) !void {
        const b = a.bank.?;
        try a.pull();
        a.bankCount(part);
        try a.r.voteWeighted(a.demand, w);
        try a.zero();
        b.stats.ticks += 1;
        if (b.o.admit and !b.o.between and a.bg == null) { // xa: between requests, the end admits
            try a.bankAdmit();
            if (b.o.land_wait) try a.bankSettle(); // xa (a gate): landed before the next forward, every rank alike
        }
    }

    fn bankPrefill(a: *Adapt, prompt_new: usize) !void {
        const b = a.bank.?;
        if (a.bg != null) try a.bankAdvance();
        try a.bankTick(0, b.o.prompt_weight);
        if (a.log) {
            const st = &b.stats;
            std.debug.print("[bank] prompt {d} tokens: {d:.1}% of the picks native ({d:.1}% of their score mass); bank {d} NVFP4 + {d} MXFP8 slots; by format: NVFP4 layers {d:.1}%, MXFP8 layers {d:.1}% of their picks native\n", .{ prompt_new, pct(st.picks[0]), pct(st.mass[0]), a.m.arena.slots(0), a.m.arena.slots(1), pct(st.fmt_picks[0][0]), pct(st.fmt_picks[0][1]) });
        }
    }

    /// x-mixed: after a verify round (the engine's commit step) or a decode step, on every rank: the batch in flight
    /// advanced; every TF_INKLING2_MIX_ADMIT_EVERY rounds a tick (the answer's demand, weight 1); every 16 rounds the
    /// watchdog. Nothing without the bank.
    pub fn afterRound(a: *Adapt) !void {
        const b = a.bank orelse return;
        b.rounds += 1;
        if (a.bg != null) try a.bankAdvance();
        if (b.rounds % b.o.every == 0) try a.bankTick(1, 1.0);
        if (b.rounds % 16 == 0) try a.bankWatch();
    }

    fn bankEnd(a: *Adapt) !void {
        const b = a.bank.?;
        try a.bankSettle();
        try a.pull();
        a.bankCount(1);
        if (a.tokens > 0) try a.r.voteWeighted(a.demand, 1.0);
        try a.zero();
        if (b.o.admit and b.o.between and a.bg == null) { // xa: between requests: the next begin lands it
            try a.bankAdmit();
            if (b.o.land_wait) try a.bankSettle();
        }
        if (a.log) {
            const st = &b.stats;
            std.debug.print("[bank] answer {d} tokens: {d:.1}% of the picks native ({d:.1}% of their score mass); request: {d} experts loaded ({d:.2} GB read); run: {d} batches, {d} loaded of {d} planned, {d} evicted, {d} failed batches, {d} sheds; bank {d} NVFP4 + {d} MXFP8 slots; by format: NVFP4 layers {d:.1}%, MXFP8 layers {d:.1}% of the answer's picks native; MemAvailable {d:.2} GiB\n", .{ a.tokens, pct(st.picks[1]), pct(st.mass[1]), st.loads - b.loads0, @as(f64, @floatFromInt(a.loader.stats.bytes -| b.read0)) / 1e9, st.batches, st.loads, st.planned, st.evictions, st.failures, st.sheds, a.m.arena.slots(0), a.m.arena.slots(1), pct(st.fmt_picks[1][0]), pct(st.fmt_picks[1][1]), @as(f64, @floatFromInt(fill.readAvail() orelse 0)) / mix.gib });
        }
        a.bankSave();
    }

    fn pct(x: [2]f64) f64 {
        return if (x[0] > 0) 100 * x[1] / x[0] else 0;
    }

    /// Rank 0's batch into `plan` ([n, (layer, expert, slot) x n]): per pool, the candidates (picked and served by EXL3
    /// since the last plan, not in the bank, a positive blended score) by score descending, each against the pool's
    /// next donor (empty slots first, then the residents by score ascending, any layer), taken when the donor is empty
    /// or the candidate's score beats the donor's by the gain margin; at most `reads` a batch and `per_layer` a layer.
    fn planBank(a: *Adapt, plan: *[plan_words]i32) !void {
        const b = a.bank.?;
        const m = a.m;
        const E = m.c.experts;
        const L = m.c.layers;
        const sc = try a.gpa.alloc(f64, L * E);
        defer a.gpa.free(sc);
        _ = try a.r.scores(sc, null);
        const per = try a.gpa.alloc(usize, L);
        defer a.gpa.free(per);
        @memset(per, 0);
        const Cand = struct { s: f64, l: u32, e: u32, slot: u32 };
        var at: usize = 1;
        var n: usize = 0;
        for (0..2) |f| {
            const slots = m.arena.slots(f);
            if (slots == 0 or a.pools[f].len == 0) continue;
            var donors: std.ArrayList(Cand) = .empty;
            defer donors.deinit(a.gpa);
            for (0..slots) |sl| {
                const o = m.arena.owner[f][sl];
                if (o < 0) {
                    try donors.append(a.gpa, .{ .s = -1, .l = 0, .e = 0, .slot = @intCast(sl) });
                } else {
                    const ow: usize = @intCast(o);
                    try donors.append(a.gpa, .{ .s = sc[ow], .l = @intCast(ow / E), .e = @intCast(ow % E), .slot = @intCast(sl) });
                }
            }
            std.mem.sort(Cand, donors.items, {}, struct {
                fn lt(_: void, x: Cand, y: Cand) bool {
                    if (x.s != y.s) return x.s < y.s;
                    if (x.l != y.l) return x.l < y.l;
                    if (x.e != y.e) return x.e < y.e;
                    return x.slot < y.slot;
                }
            }.lt);
            var incoming: std.ArrayList(Cand) = .empty;
            defer incoming.deinit(a.gpa);
            for (a.pools[f]) |l| for (0..E) |e| {
                const i = l * E + e;
                if (b.eligible[i] != 0 and m.arena.host_lut[i] < 0 and sc[i] > 0) try incoming.append(a.gpa, .{ .s = sc[i], .l = l, .e = @intCast(e), .slot = 0 });
            };
            std.mem.sort(Cand, incoming.items, {}, struct {
                fn lt(_: void, x: Cand, y: Cand) bool {
                    if (x.s != y.s) return x.s > y.s;
                    if (x.l != y.l) return x.l < y.l;
                    return x.e < y.e;
                }
            }.lt);
            var di: usize = 0;
            for (incoming.items) |in| {
                if (n >= b.o.reads or di >= donors.items.len) break;
                if (per[in.l] >= b.o.per_layer) continue;
                const dn = donors.items[di];
                // the donors ascend and the candidates descend: one that cannot beat this donor beats none after it
                if (dn.s >= 0 and in.s <= dn.s * (1 + b.o.gain)) break;
                plan[at] = @intCast(in.l);
                plan[at + 1] = @intCast(in.e);
                plan[at + 2] = @intCast(dn.slot);
                at += 3;
                n += 1;
                di += 1;
                per[in.l] += 1;
            }
        }
        plan[0] = @intCast(n);
    }

    /// Rank 0's batch on every rank: its donors unmapped and published (each pick of theirs runs EXL3 from the next
    /// forward; the stream is idle at a tick, so no forward still reads them), then its loads in the background.
    fn bankAdmit(a: *Adapt) !void {
        const b = a.bank.?;
        const m = a.m;
        const E = m.c.experts;
        var plan: [plan_words]i32 = @splat(0);
        if (m.rank == 0) a.planBank(&plan) catch |err| {
            std.log.err("inkling2: the native bank's plan failed: {t}", .{err});
            plan[0] = 0;
        };
        try a.share(&plan);
        @memset(b.eligible, 0);
        const n: usize = @intCast(@max(plan[0], 0));
        if (n == 0) return;
        if (1 + 3 * n > plan_words) return error.BadPlan;
        const own = try a.gpa.alloc(fill.Job, n);
        errdefer a.gpa.free(own);
        for (own, 0..) |*j, i| {
            const l: usize = @intCast(plan[1 + 3 * i]);
            const e: usize = @intCast(plan[2 + 3 * i]);
            const slot: usize = @intCast(plan[3 + 3 * i]);
            if (l >= m.c.layers or e >= E) return error.BadPlan;
            const f = fmtIndex(m.c.act[l]);
            if (slot >= m.arena.slots(f) or m.arena.host_lut[l * E + e] >= 0) return error.BadPlan;
            if (m.arena.owner[f][slot] >= 0) b.stats.evictions += 1;
            m.evict(f, slot);
            j.* = .{ .layer = @intCast(l), .expert = @intCast(e), .slot = @intCast(slot) };
        }
        try m.s.synchronize();
        try m.publish();
        var context: cuda.abi.Context = null;
        try m.d.check(m.d.api.cuCtxGetCurrent(&context), "cuCtxGetCurrent");
        try a.loader.startBackground(m.d, context, fill.Target.ofArena(&m.arena), own);
        a.bg = .{ .kind = .decode, .jobs = own, .t0 = a.now(), .tokens = a.tokens, .miss = 0, .w = 0 };
        b.stats.batches += 1;
        b.stats.planned += n;
    }

    /// Each rank's landed count and whether its load is done, agreed over the TP group (16 bytes a rank): the least
    /// landed count and whether every rank is done.
    fn bankAgree(a: *Adapt, landed: usize, done: bool) !struct { n: usize, done: bool } {
        const m = a.m;
        var n = landed;
        var all_done = done;
        if (m.world > 1) {
            var mine = [4]u32{ @intCast(landed), @intFromBool(done), 0, 0 };
            try a.agree_mine.uploadAsync(0, std.mem.sliceAsBytes(&mine), m.s.handle);
            try m.group.?.allGather(m.s, a.agree_mine.ptr, a.agree_all.ptr, 16);
            try m.s.synchronize();
            var all: [64]u32 = undefined;
            try a.agree_all.download(0, std.mem.sliceAsBytes(all[0 .. 4 * m.world]));
            for (0..m.world) |r| {
                n = @min(n, all[4 * r]);
                all_done = all_done and all[4 * r + 1] != 0;
            }
        }
        return .{ .n = n, .done = all_done };
    }

    /// The batch's jobs [mapped, n) mapped and published (their bytes landed on every rank).
    fn bankMap(a: *Adapt, n: usize) !void {
        const bg = if (a.bg) |*x| x else return;
        const m = a.m;
        if (n <= bg.mapped) return;
        for (bg.jobs[bg.mapped..n]) |j| m.place(j.layer, j.expert, fmtIndex(m.c.act[j.layer]), j.slot);
        try m.s.synchronize();
        try m.publish();
        a.bank.?.stats.loads += n - bg.mapped;
        bg.mapped = n;
    }

    /// While a batch loads: what every rank has landed mapped; once every rank's load is done, the batch settled.
    fn bankAdvance(a: *Adapt) !void {
        if (a.bg == null) return;
        const g = try a.bankAgree(a.loader.landed(), a.loader.backgroundFinished());
        try a.bankMap(g.n);
        if (g.done) try a.bankSettle();
    }

    /// The batch in flight finished on every rank (each joins its load): the jobs every rank landed mapped, the rest
    /// left empty (their donors are already unmapped: those picks run EXL3), so every rank maps the same experts.
    fn bankSettle(a: *Adapt) !void {
        const bg = a.bg orelse return;
        const r = a.loader.joinBackground();
        const g = try a.bankAgree(r.landed, true);
        var failed: u64 = @intFromBool(r.failed != null);
        failed = try lazy.Lazy.agreeMax(a.m, failed);
        try a.bankMap(g.n);
        if (g.n < bg.jobs.len or failed != 0) {
            a.bank.?.stats.failures += 1;
            const why: []const u8 = if (r.failed) |e| @errorName(e) else "landed";
            std.log.err("inkling2 rank {d}: the native bank's batch: {d} of {d} experts loaded on every rank (this rank: {s}); the rest stay on EXL3", .{ a.m.rank, g.n, bg.jobs.len, why });
        }
        a.gpa.free(bg.jobs);
        a.bg = null;
    }

    /// Every 16 rounds: when MemAvailable is under the floor and TF_INKLING2_MIX_WATCH_GIB on any rank, the bank gives
    /// slots back (giveBack: each pool's lowest-scored experts, whose picks then run EXL3). Needs the lazy context.
    fn bankWatch(a: *Adapt) !void {
        const b = a.bank.?;
        const lz = a.m.lazy orelse return;
        _ = try a.bankShed(lz.floor + @as(u64, @intFromFloat(b.o.watch_gib * mix.gib)));
    }

    /// What a reclaim of the bank leaves over the floor: the larger of the context's margin and the watch's
    /// TF_INKLING2_MIX_WATCH_GIB, plus TF_INKLING2_MIX_HYST_GIB (a band between the reclaim and the watch).
    fn bankKeep(a: *const Adapt, lz: *const lazy.Lazy) u64 {
        const b = a.bank.?;
        const watch: u64 = @intFromFloat(b.o.watch_gib * mix.gib);
        return @max(lz.margin, watch) + @as(u64, @intFromFloat(b.o.hyst_gib * mix.gib));
    }

    /// The bank's slots given back (giveBack, alike on every rank) until MemAvailable is at least `want` on every rank;
    /// whether it gave any. A failed give-back is logged, never fatal.
    fn bankShed(a: *Adapt, want: u64) !bool {
        const b = a.bank.?;
        const m = a.m;
        const avail = fill.readAvail() orelse want;
        const short = try lazy.Lazy.agreeMax(m, want -| avail);
        if (short == 0 or m.arena.slots(0) + m.arena.slots(1) == 0) return false;
        a.giveBack(short) catch |err| {
            std.log.warn("inkling2 rank {d}: the native bank could not give {d:.2} GiB back ({t})", .{ m.rank, @as(f64, @floatFromInt(short)) / mix.gib, err });
            return false;
        };
        b.stats.sheds += 1;
        return true;
    }

    /// xa (a gate: run_main's arm switch giveback=K): K experts a layer of the bank given back now, as a growing
    /// context's give-back (giveBack: each pool's lowest-scored residents, the top slots moved down, the chunks
    /// unmapped) but of a size MemAvailable does not set, so the next begin's reclaim maps back the same experts in
    /// every load (a growth's give-back and its map-back follow MemAvailable, which moves between loads).
    pub fn bankGiveBack(a: *Adapt, k: usize) !void {
        if (k == 0) return;
        if (a.bank == null) return error.NoBank;
        const m = a.m;
        const lz = m.lazy orelse return error.NoLazyContext;
        const n = [2]usize{ m.arena.slots(0), m.arena.slots(1) };
        const cut = [2]usize{ @min(k * a.pools[0].len, n[0]), @min(k * a.pools[1].len, n[1]) };
        const short = lz.arenaFreed(m.arena.words, 0, n[0] - cut[0]) + lz.arenaFreed(m.arena.words, 1, n[1] - cut[1]);
        if (short == 0) return error.NothingToGive;
        try a.giveBack(short);
    }

    /// x-mixed: after the load's prewarm (which takes memory after the bank was sized: about 1.3 GiB at 4096-row chunks
    /// on the head node, graphs and workspaces), on every rank: the bank gives slots back until MemAvailable is at the
    /// reclaim's level (bankKeep over the floor) on every rank, so serving starts inside the band. Nothing without the
    /// bank or the lazy context.
    pub fn bankBoot(a: *Adapt) !void {
        const b = a.bank orelse return;
        const lz = a.m.lazy orelse return;
        try a.zero(); // the prewarm's synthetic routing is no request's demand (the give-back's victims follow the scores)
        const before = b.stats.sheds;
        const shed = try a.bankShed(lz.floor + a.bankKeep(lz));
        if (shed) b.stats.sheds = before; // a boot shed is not a watch's
        if (a.log) std.debug.print("[bank] after the prewarm: {s}; bank {d} NVFP4 + {d} MXFP8 slots; MemAvailable {d:.2} GiB (the reclaim keeps {d:.2} GiB over the floor, the watch sheds under {d:.2})\n", .{ if (shed) "slots given back to the band" else "inside the band", a.m.arena.slots(0), a.m.arena.slots(1), @as(f64, @floatFromInt(fill.readAvail() orelse 0)) / mix.gib, @as(f64, @floatFromInt(a.bankKeep(lz))) / mix.gib, b.o.watch_gib });
    }

    /// x-mixed (TF_INKLING2_MIX_HISTORY): the demand history seeded at boot, before the bank's boot fill (which then
    /// follows the blended scores): rank 0 reads the file and every rank takes its bytes (the TP group), checks them
    /// alike (mix.decodeHistory: identity, geometry, CRC) and restores the residency's evidence. A missing file starts
    /// cold (and is written later); a foreign or damaged one is ignored and never written over. `view` is the EXL3
    /// view's directory (its checkpoint.json is in the identity).
    pub fn bankHistory(a: *Adapt, view: []const u8) !void {
        const b = a.bank orelse return;
        const path = b.o.history orelse return;
        const m = a.m;
        b.id = try mix.historyIdentity(a.gpa, a.io, m.mix_native, view);
        var bytes: []u8 = &.{};
        defer if (bytes.len > 0) a.gpa.free(bytes);
        var state: u64 = 0; // 0 missing, 1 read, 2 unreadable
        if (m.rank == 0) {
            if (std.Io.Dir.cwd().readFileAlloc(a.io, path, a.gpa, .limited(mix.history_max))) |got| {
                bytes = got;
                state = 1;
            } else |err| state = if (err == error.FileNotFound) 0 else 2;
        }
        // rank 0's size and state, then its bytes, to every rank (gathers on the TP group: rank 0's segment first)
        var head = [2]u64{ bytes.len, state };
        if (m.world > 1) {
            var hb = try cuda.DeviceBuffer.alloc(m.d, 16);
            defer hb.free();
            var ha = try cuda.DeviceBuffer.alloc(m.d, 16 * m.world);
            defer ha.free();
            try hb.upload(0, std.mem.sliceAsBytes(&head));
            try m.group.?.allGather(m.s, hb.ptr, ha.ptr, 16);
            try m.s.synchronize();
            try ha.download(0, std.mem.sliceAsBytes(&head));
            const size: usize = @intCast(head[0]);
            if (size > mix.history_max) return error.HistoryDamaged;
            if (size > 0) {
                if (m.rank != 0) bytes = try a.gpa.alloc(u8, size);
                var pb = try cuda.DeviceBuffer.alloc(m.d, size);
                defer pb.free();
                var pa = try cuda.DeviceBuffer.alloc(m.d, size * m.world);
                defer pa.free();
                if (m.rank == 0) try pb.upload(0, bytes);
                try m.group.?.allGather(m.s, pb.ptr, pa.ptr, size);
                try m.s.synchronize();
                try pa.download(0, bytes[0..size]);
            }
        }
        if (head[1] == 0) {
            b.hist_ok = true;
            if (a.log) std.debug.print("[bank] history {s}: none yet; the boot fill follows the prior, the file is written at each request's end\n", .{path});
            return;
        }
        const sd = if (head[1] == 1) mix.decodeHistory(a.gpa, bytes, b.id.?, m.c.layers, m.c.experts, m.world) catch |err| {
            std.log.warn("inkling2 rank {d}: the bank history {s} is not this model's or is damaged ({t}): ignored, and never written over", .{ m.rank, path, err });
            return;
        } else {
            std.log.warn("inkling2 rank {d}: the bank history {s} is unreadable: ignored, and never written over", .{ m.rank, path });
            return;
        };
        defer mix.freeSeed(a.gpa, sd);
        @memcpy(a.r.n, sd.n);
        @memcpy(a.r.hist, sd.hist);
        a.r.votes = @intCast(sd.votes);
        b.hist_ok = true;
        if (a.log) std.debug.print("[bank] history {s}: {d} votes restored ({d} experts were in the bank at the save)\n", .{ path, sd.votes, sd.bank.len });
    }

    /// Rank 0: the demand history written (TF_INKLING2_MIX_HISTORY; at every request's end, atomically). A failure is
    /// logged, never fatal.
    fn bankSave(a: *Adapt) void {
        const b = a.bank orelse return;
        const path = b.o.history orelse return;
        if (a.m.rank != 0 or !b.hist_ok or b.id == null) return;
        a.bankWrite(path) catch |err| {
            std.log.warn("inkling2: the bank history {s} was not written ({t})", .{ path, err });
            return;
        };
        b.saves += 1;
    }

    fn bankWrite(a: *Adapt, path: []const u8) !void {
        const b = a.bank.?;
        const m = a.m;
        const E = m.c.experts;
        const sc = try a.gpa.alloc(f64, m.c.layers * E);
        defer a.gpa.free(sc);
        _ = try a.r.scores(sc, null);
        var bank: std.ArrayList([2]u16) = .empty;
        defer bank.deinit(a.gpa);
        for (m.arena.host_lut, 0..) |slot, i| if (slot >= 0) try bank.append(a.gpa, .{ @intCast(i / E), @intCast(i % E) });
        const Ctx = struct { sc: []const f64, E: usize };
        std.mem.sort([2]u16, bank.items, Ctx{ .sc = sc, .E = E }, struct {
            fn lt(ctx: Ctx, x: [2]u16, y: [2]u16) bool {
                const sx = ctx.sc[@as(usize, x[0]) * ctx.E + x[1]];
                const sy = ctx.sc[@as(usize, y[0]) * ctx.E + y[1]];
                if (sx != sy) return sx > sy;
                if (x[0] != y[0]) return x[0] < y[0];
                return x[1] < y[1];
            }
        }.lt);
        const bytes = try mix.encodeHistory(a.gpa, b.id.?, m.c.layers, E, m.world, a.r.votes, a.r.n, a.r.hist, bank.items);
        defer a.gpa.free(bytes);
        try mix.writeAtomic(a.gpa, path, bytes);
    }
    // ---- end x-mixed

};

/// This process's resident set (VmRSS), in bytes; 0 when /proc says nothing.
fn rss() u64 {
    const fd = std.c.open("/proc/self/status", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var buf: [4096]u8 = undefined;
    var n: usize = 0;
    while (n < buf.len) {
        const r = std.c.read(fd, buf[n..].ptr, buf.len - n);
        if (r <= 0) break;
        n += @intCast(r);
    }
    const at = std.mem.indexOf(u8, buf[0..n], "VmRSS:") orelse return 0;
    var f = std.mem.tokenizeAny(u8, buf[at + "VmRSS:".len .. n], " \tkB\n");
    return (std.fmt.parseInt(u64, f.next() orelse return 0, 10) catch 0) * 1024;
}

fn fmtIndex(f: @import("config.zig").Format) usize {
    return if (f == .nvfp4) 0 else 1;
}

/// MemAvailable from /proc/meminfo, in bytes (read with read(2): procfs reports a size of 0).
pub fn memAvailable(io: std.Io, gpa: std.mem.Allocator) !u64 {
    _ = io;
    _ = gpa;
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.MemInfo;
    defer _ = std.c.close(fd);
    var buf: [8192]u8 = undefined;
    var n: usize = 0;
    while (n < buf.len) {
        const r = std.c.read(fd, buf[n..].ptr, buf.len - n);
        if (r <= 0) break;
        n += @intCast(r);
    }
    var it = std.mem.tokenizeScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \tkB");
        return (try std.fmt.parseInt(u64, f.next() orelse return error.MemInfo, 10)) * 1024;
    };
    return error.MemInfo;
}
