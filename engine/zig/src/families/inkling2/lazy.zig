//! p4-mem: the long-context buffers on demand (TF_INKLING2_LAZY_CONTEXT=1; default 0 = phase 3's buffers made whole
//! at open). A buffer whose size the context sets -- a global layer's K / V cache (position p at slot p), the decode
//! attention's score scratch, the draft head's global-layer context ring, its score scratch and its rope table -- has
//! the whole context's virtual addresses reserved once (cuMemAddressReserve) and memory mapped behind a prefix of them
//! only: the positions `mapped` (TF_INKLING2_CONTEXT_MAPPED tokens at open, default 65536). Addresses never move, so
//! the CUDA graphs captured over them stay valid; a chunk reaching past `mapped` first maps more (ensure: every rank
//! at the same step, up to the next multiple of TF_INKLING2_CONTEXT_STEP tokens, default the budget). The arena is
//! sized after these buffers (engine.zig, run_main.zig), so it is the one a small context gets; of the memory the
//! whole context would have taken at open, the arena takes TF_INKLING2_LAZY_ARENA_GIB at most (default all; 0 sizes
//! it as phase 3 does and leaves the rest free).
//!
//! A growth keeps MemAvailable at the floor plus TF_INKLING2_CONTEXT_MARGIN_GIB (default 1) on every rank. When it
//! cannot, the arena gives memory back first: its four slot arrays are ranges of 64 MiB chunks here (allocArena),
//! and the residency (adapt.zig giveBack) evicts its lowest-value experts, moves the residents of the top slots down
//! into the slots they left and unmaps the chunks past the new top. When even that cannot make room, the chunk fails
//! with error.ContextMemory on every rank (the engine's follower keeps serving). A new sequence (Adapt.begin ->
//! reclaim) unmaps the context back to the budget (never under the live sequence) and maps the given-back slots again.
//!
//! Numerics: none change. The score scratch's stride a (row, head) is the mapped positions' instead of the context's
//! (ink2_attention reads it as a stride only), and a prompt chunk's attention takes as many rows a launch as the mapped
//! scratch holds (the same per-row bits, forward.layers).
const std = @import("std");
const cuda = @import("cuda");
const model_mod = @import("model.zig");
const fill = @import("fill.zig");
const pieces_mod = @import("lazy_pieces.zig"); // xr

const abi = cuda.abi;
const Buf = cuda.DeviceBuffer;
const Model = model_mod.Model;
const gib: f64 = 1024.0 * 1024.0 * 1024.0;

/// The arena's slot arrays map in chunks of this many bytes (a give-back releases whole ones): 64 MiB maps at the
/// driver's bulk rate (vmm_probe: 2.6 ms a chunk, 41 ms a GiB; cuMemAlloc 38 ms a GiB) and reads at its bandwidth.
pub const arena_chunk: usize = 64 << 20;

/// n rounded up to a multiple of `step` (any step >= 1; std.mem.alignForward needs a power of two).
fn roundUp(n: usize, step: usize) usize {
    return (n + step - 1) / step * step;
}

/// CUDA's virtual memory entry points, looked up in the opened driver's libcuda.so.1 (the core's cuda.arena.Vmm does
/// the same for its forward-only chunk allocator; these ranges grow and shrink their mapped prefix in place).
pub const Vmm = struct {
    api: Api,
    prop: Prop,
    access: Access,
    granule: usize,

    const Api = struct {
        reserve: *const fn (*u64, usize, usize, u64, c_ulonglong) callconv(.c) abi.Result,
        addressFree: *const fn (u64, usize) callconv(.c) abi.Result,
        create: *const fn (*u64, usize, *const Prop, c_ulonglong) callconv(.c) abi.Result,
        release: *const fn (u64) callconv(.c) abi.Result,
        map: *const fn (u64, usize, usize, u64, c_ulonglong) callconv(.c) abi.Result,
        unmap: *const fn (u64, usize) callconv(.c) abi.Result,
        setAccess: *const fn (u64, usize, *const Access, usize) callconv(.c) abi.Result,
        granularity: *const fn (*usize, *const Prop, c_int) callconv(.c) abi.Result,
    };
    /// CUmemAllocationProp: pinned memory of device `loc_id` (no RDMA flag: these buffers never cross the TP group).
    const Prop = extern struct { type: c_int = 1, handle_types: c_int = 0, loc_type: c_int = 1, loc_id: c_int = 0, win32: ?*anyopaque = null, compression: u8 = 0, rdma: u8 = 0, usage: u16 = 0, reserved: [4]u8 = .{ 0, 0, 0, 0 } };
    /// CUmemAccessDesc: read-write from device `loc_id`.
    const Access = extern struct { loc_type: c_int = 1, loc_id: c_int, flags: c_int = 3 };

    comptime {
        std.debug.assert(@sizeOf(Prop) == 32 and @offsetOf(Prop, "compression") == 24 and @sizeOf(Access) == 12);
    }

    pub fn open(d: *const cuda.Driver, device: abi.Device) !Vmm {
        var ok: c_int = 0;
        if (d.api.cuDeviceGetAttribute(&ok, .virtual_memory_management_supported, device) != abi.success or ok == 0) return error.NoVirtualMemory;
        var api: Api = undefined;
        const lib = @constCast(&d.lib);
        const names = .{ .{ "reserve", "cuMemAddressReserve" }, .{ "addressFree", "cuMemAddressFree" }, .{ "create", "cuMemCreate" }, .{ "release", "cuMemRelease" }, .{ "map", "cuMemMap" }, .{ "unmap", "cuMemUnmap" }, .{ "setAccess", "cuMemSetAccess" }, .{ "granularity", "cuMemGetAllocationGranularity" } };
        inline for (names) |n| @field(api, n[0]) = lib.lookup(@FieldType(Api, n[0]), n[1]) orelse return error.MissingSymbol;
        const prop: Prop = .{ .loc_id = device };
        var g: usize = 0;
        if (api.granularity(&g, &prop, 1) != abi.success or g == 0) return error.NoVirtualMemory; // RECOMMENDED
        return .{ .api = api, .prop = prop, .access = .{ .loc_id = device }, .granule = @max(g, 2 << 20) };
    }

    fn check(rc: abi.Result, what: []const u8) !void {
        if (rc == abi.success) return;
        std.log.err("inkling2 lazy context: {s} failed (CUresult {d})", .{ what, rc });
        return error.CudaFailed;
    }
};

/// Virtual addresses for `reserved` bytes, the first `mapped` of them backed by memory in `chunk`-byte pieces, one
/// physical allocation each (released when its piece is unmapped): the address stays, the backed prefix moves.
/// xr: a context buffer's range (`whole`) maps each growth's bytes as one piece instead (lazy_pieces.zig): one
/// allocation, one map, one access call where its 2 MiB chunks took three calls each; unmapped a piece at a time.
pub const Range = struct {
    va: u64 = 0,
    reserved: usize = 0,
    chunk: usize = 0,
    mapped: usize = 0,
    whole: ?pieces_mod.Pieces = null, // xr: the pieces (their ends' storage is the Lazy's)

    pub fn reserve(v: *const Vmm, bytes: usize, chunk: usize) !Range {
        const c = std.mem.alignForward(usize, @max(chunk, v.granule), v.granule);
        const len = std.mem.alignForward(usize, @max(bytes, 1), c);
        var va: u64 = 0;
        try Vmm.check(v.api.reserve(&va, len, v.granule, 0, 0), "cuMemAddressReserve");
        return .{ .va = va, .reserved = len, .chunk = c };
    }

    /// xr: a context buffer's range: granule-sized boundaries, each growth one piece of `ends.len` at most.
    pub fn reserveWhole(v: *const Vmm, bytes: usize, ends: []usize) !Range {
        var r = try reserve(v, bytes, v.granule);
        r.whole = .{ .ends = ends };
        return r;
    }

    fn want(r: *const Range, bytes: usize) usize {
        return @min(std.mem.alignForward(usize, bytes, r.chunk), r.reserved);
    }

    /// Bytes [0, bytes) backed (whole chunks, at most the reservation). A failed chunk leaves the ones before it.
    pub fn mapTo(r: *Range, v: *const Vmm, bytes: usize) !void {
        const to = r.want(bytes);
        if (r.whole) |*pc| { // xr: the bytes past the mapped ones as one piece
            const sp = (try pc.next(to)) orelse return;
            const len = sp.end - sp.start;
            var h: u64 = 0;
            try Vmm.check(v.api.create(&h, len, &v.prop, 0), "cuMemCreate");
            defer _ = v.api.release(h); // the mapping holds the memory until it is unmapped
            const at = r.va + sp.start;
            try Vmm.check(v.api.map(at, len, 0, h, 0), "cuMemMap");
            Vmm.check(v.api.setAccess(at, len, &v.access, 1), "cuMemSetAccess") catch |err| {
                _ = v.api.unmap(at, len);
                return err;
            };
            pc.push(sp);
            r.mapped = sp.end;
            return;
        }
        while (r.mapped < to) {
            var h: u64 = 0;
            try Vmm.check(v.api.create(&h, r.chunk, &v.prop, 0), "cuMemCreate");
            defer _ = v.api.release(h); // the mapping holds the memory until it is unmapped
            const at = r.va + r.mapped;
            try Vmm.check(v.api.map(at, r.chunk, 0, h, 0), "cuMemMap");
            Vmm.check(v.api.setAccess(at, r.chunk, &v.access, 1), "cuMemSetAccess") catch |err| {
                _ = v.api.unmap(at, r.chunk);
                return err;
            };
            r.mapped += r.chunk;
        }
    }

    /// Only bytes [0, bytes) stay backed: every chunk wholly past them unmapped, its memory released.
    pub fn unmapTo(r: *Range, v: *const Vmm, bytes: usize) void {
        const keep = @min(r.want(bytes), r.mapped);
        if (r.whole) |*pc| { // xr: the pieces wholly at or past `keep` (the growths' own boundaries: exact)
            while (pc.lastPast(keep)) |sp| {
                _ = v.api.unmap(r.va + sp.start, sp.end - sp.start);
                pc.pop();
                r.mapped = sp.start;
            }
            return;
        }
        while (r.mapped > keep) {
            r.mapped -= r.chunk;
            _ = v.api.unmap(r.va + r.mapped, r.chunk);
        }
    }

    /// Bytes mapTo(bytes) would add.
    pub fn addedAt(r: *const Range, bytes: usize) usize {
        return r.want(bytes) -| r.mapped;
    }

    /// Bytes unmapTo(bytes) would release.
    pub fn freedAt(r: *const Range, bytes: usize) usize {
        if (r.whole) |*pc| return pc.freedAt(@min(r.want(bytes), r.mapped));
        return r.mapped -| @min(r.want(bytes), r.mapped);
    }

    pub fn deinit(r: *Range, v: *const Vmm) void {
        if (r.reserved == 0) return;
        r.unmapTo(v, 0);
        _ = v.api.addressFree(r.va, r.reserved);
        r.* = .{};
    }
};

/// One context-sized buffer: `per_token` bytes a position, for max(mapped + extra, least) positions (the draft's
/// rings hold 64 positions past the target's, its score scratch a block more; the score scratch at least the window
/// and a verify's rows). `view` is the owner's buffer, borrowed: its address the range's, its length the mapped size.
pub const Entry = struct {
    owner: *const anyopaque,
    range: Range,
    view: *Buf,
    per_token: usize,
    extra: usize,
    least: usize,
    name: []const u8,

    pub fn bytesAt(e: *const Entry, tokens: usize) usize {
        return e.per_token * @max(tokens + e.extra, e.least);
    }
};

/// Told when the mapped positions change (old -> new): the draft head's score stride and rope rows (dspark.zig).
pub const Hook = struct { owner: *const anyopaque, ctx: *anyopaque, f: *const fn (ctx: *anyopaque, old: usize, new: usize) anyerror!void };

/// The arena's memory handed back for a growth (adapt.zig giveBack): at least `short` bytes, on every rank alike.
pub const GiveBack = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque, short: u64) anyerror!void };

pub const Stats = struct {
    grows: usize = 0,
    trims: usize = 0,
    peak: usize = 0, // the most positions mapped
    gave: u64 = 0, // arena bytes given back for the context, over the run
    regained: u64 = 0, // arena bytes mapped back (reclaim)
    refused: usize = 0, // growths that failed (error.ContextMemory)
};

pub const Lazy = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    vmm: Vmm,
    rank: usize,
    context: usize, // positions reserved (Model.context)
    budget: usize, // positions mapped at open (TF_INKLING2_CONTEXT_MAPPED)
    step: usize, // a growth maps up to the next multiple of this (TF_INKLING2_CONTEXT_STEP)
    mapped: usize, // positions every entry holds now
    floor: u64 = 8 << 30, // MemAvailable a growth keeps (the engine's floor, set after open) ...
    margin: u64, // ... plus this (TF_INKLING2_CONTEXT_MARGIN_GIB)
    arena_cap: ?u64, // TF_INKLING2_LAZY_ARENA_GIB: of the memory not mapped at open, what the arena may take (null: all)
    /// TF_INKLING2_LAZY_SCORES_MIB (default 1536): bytes the score scratch maps at least, never past the context's own
    /// size (Model.makeScratch). A decode chunk reads scoreCap() floats a (row, head) of it; a prompt chunk's attention
    /// takes as many rows a launch as it holds, so a bigger scratch means fewer, wider launches: 1536 MiB is phase 3's
    /// at 1M (its launches exactly; p16k TTFT 17.07 s vs 17.72 s with the 64K positions' 96 MiB, phase 3 16.79 s).
    scores_min: usize,
    entries: std.ArrayList(Entry) = .empty,
    hooks: std.ArrayList(Hook) = .empty,
    giveback: ?GiveBack = null,
    /// The arena's slot arrays (allocArena): NVFP4 up, NVFP4 down, MXFP8 up, MXFP8 down.
    arena: [4]Range = @splat(.{}),
    arena_on: bool = false,
    arena_boot: [2]usize = .{ 0, 0 }, // each pool's slots at allocArena: a reclaim maps back to these at most
    stats: Stats = .{},

    fn envUsize(name: [*:0]const u8) ?usize {
        const v = std.c.getenv(name) orelse return null;
        return std.fmt.parseInt(usize, std.mem.span(v), 10) catch {
            std.log.warn("inkling2: {s}={s} is not a count; ignored", .{ name, std.mem.span(v) });
            return null;
        };
    }

    fn envGib(name: [*:0]const u8) ?f64 {
        const v = std.c.getenv(name) orelse return null;
        const x = std.fmt.parseFloat(f64, std.mem.span(v)) catch {
            std.log.warn("inkling2: {s}={s} is not a number of GiB; ignored", .{ name, std.mem.span(v) });
            return null;
        };
        return if (x >= 0) x else null;
    }

    /// The lazy context of a model of `context` positions on `device` (Model.open, before its state and scratch).
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, device: abi.Device, rank: usize, context: usize) !*Lazy {
        const lz = try gpa.create(Lazy);
        errdefer gpa.destroy(lz);
        const budget = std.math.clamp(envUsize("TF_INKLING2_CONTEXT_MAPPED") orelse 65536, 1, context);
        lz.* = .{
            .gpa = gpa,
            .d = d,
            .vmm = try Vmm.open(d, device),
            .rank = rank,
            .context = context,
            .budget = budget,
            .step = @max(envUsize("TF_INKLING2_CONTEXT_STEP") orelse budget, 1),
            .mapped = budget,
            .margin = @intFromFloat((envGib("TF_INKLING2_CONTEXT_MARGIN_GIB") orelse 1.0) * gib),
            .arena_cap = if (envGib("TF_INKLING2_LAZY_ARENA_GIB")) |g| @as(u64, @intFromFloat(g * gib)) else null,
            .scores_min = (envUsize("TF_INKLING2_LAZY_SCORES_MIB") orelse 1536) << 20,
        };
        lz.stats.peak = budget;
        return lz;
    }

    pub fn deinit(lz: *Lazy) void {
        for (lz.entries.items) |*e| lz.freeEntry(e);
        for (&lz.arena) |*r| r.deinit(&lz.vmm);
        lz.entries.deinit(lz.gpa);
        lz.hooks.deinit(lz.gpa);
        lz.gpa.destroy(lz);
    }

    /// A context-sized buffer of `owner`'s: the whole context reserved, the mapped positions backed, `view` set to it.
    pub fn register(lz: *Lazy, owner: *const anyopaque, view: *Buf, per_token: usize, extra: usize, least: usize, name: []const u8) !void {
        var e: Entry = .{ .owner = owner, .range = .{}, .view = view, .per_token = per_token, .extra = extra, .least = least, .name = name };
        // xr: a piece a growth step (ensure), the budget's at open: context / step + 3 pieces at most
        const ends = try lz.gpa.alloc(usize, lz.context / lz.step + 3);
        errdefer lz.gpa.free(ends);
        e.range = try Range.reserveWhole(&lz.vmm, e.bytesAt(lz.context), ends);
        errdefer e.range.deinit(&lz.vmm);
        try e.range.mapTo(&lz.vmm, e.bytesAt(lz.mapped));
        view.* = .{ .d = lz.d, .ptr = e.range.va, .len = e.bytesAt(lz.mapped), .borrowed = true };
        try lz.entries.append(lz.gpa, e);
    }

    /// xr: an entry's range unmapped and freed, its pieces' storage with it.
    fn freeEntry(lz: *Lazy, e: *Entry) void {
        const ends: []usize = if (e.range.whole) |pc| pc.ends else &.{};
        e.range.deinit(&lz.vmm);
        if (ends.len > 0) lz.gpa.free(ends);
    }

    pub fn hook(lz: *Lazy, owner: *const anyopaque, ctx: *anyopaque, f: *const fn (ctx: *anyopaque, old: usize, new: usize) anyerror!void) !void {
        try lz.hooks.append(lz.gpa, .{ .owner = owner, .ctx = ctx, .f = f });
    }

    /// `owner`'s buffers unmapped and forgotten, its hooks dropped (the draft head's close).
    pub fn drop(lz: *Lazy, owner: *const anyopaque) void {
        var i: usize = 0;
        while (i < lz.entries.items.len) {
            if (lz.entries.items[i].owner == owner) {
                lz.freeEntry(&lz.entries.items[i]);
                _ = lz.entries.orderedRemove(i);
            } else i += 1;
        }
        i = 0;
        while (i < lz.hooks.items.len) {
            if (lz.hooks.items[i].owner == owner) {
                _ = lz.hooks.orderedRemove(i);
            } else i += 1;
        }
    }

    /// Bytes every entry maps now.
    pub fn mappedBytes(lz: *const Lazy) usize {
        var n: usize = 0;
        for (lz.entries.items) |*e| n += e.range.mapped;
        return n;
    }

    /// Bytes the entries would take at `tokens` positions (unrounded: the buffers' own sizes).
    pub fn bytesAt(lz: *const Lazy, tokens: usize) usize {
        var n: usize = 0;
        for (lz.entries.items) |*e| n += e.bytesAt(tokens);
        return n;
    }

    /// Of the memory the whole context's buffers would take, what is not mapped now.
    pub fn deferred(lz: *const Lazy) usize {
        return lz.bytesAt(lz.context) -| lz.mappedBytes();
    }

    /// Bytes the arena's sizing leaves out (TF_INKLING2_LAZY_ARENA_GIB): what is not mapped past the cap; 0 = all
    /// of it goes to the arena (the default).
    pub fn heldBack(lz: *const Lazy) usize {
        const cap = lz.arena_cap orelse return 0;
        return lz.deferred() -| cap;
    }

    /// Every rank's `v`, the largest (`max`) or the smallest; every rank calls it at the same step.
    fn agree(m: *Model, v: u64, max: bool) !u64 {
        if (m.world == 1) return v;
        var all: [8]u64 = undefined;
        try m.group.?.gather(u64, m.d, m.s, v, all[0..m.world]);
        var x = all[0];
        for (all[1..m.world]) |y| x = if (max) @max(x, y) else @min(x, y);
        return x;
    }

    pub fn agreeMax(m: *Model, v: u64) !u64 {
        return agree(m, v, true);
    }

    pub fn agreeMin(m: *Model, v: u64) !u64 {
        return agree(m, v, false);
    }

    /// Positions [0, need) mapped in every entry (forward.chunk, before a chunk's launches; every rank at the same
    /// step): up to the next multiple of the step, the arena's memory given back first when MemAvailable would fall
    /// under the floor and the margin on any rank. error.ContextMemory (on every rank) when there is no room.
    pub fn ensure(lz: *Lazy, m: *Model, need: usize) !void {
        if (need <= lz.mapped) return;
        if (need > lz.context) return error.ContextFull;
        const old = lz.mapped;
        // e-dense / lead, 2026-11-15: the step need not be a power of two (std.mem.alignForward asserts one: a window
        // under 65536 that is not a power of two panicked at the first trim / ensure); the same values for every
        // power-of-two step
        const new = @min(lz.context, roundUp(need, lz.step));
        var add: usize = 0;
        for (lz.entries.items) |*e| add += e.range.addedAt(e.bytesAt(new));
        const avail0 = fill.readAvail() orelse 0;
        // the most any rank lacks is given back on every rank (the ranks' arenas hold the same experts)
        const short = try agreeMax(m, (lz.floor + lz.margin + add) -| avail0);
        try m.s.synchronize();
        if (short > 0) {
            const gb = lz.giveback orelse {
                lz.stats.refused += 1;
                std.log.err("inkling2 rank {d}: {d} positions need {d:.2} GiB more context memory; MemAvailable {d:.2} GiB leaves {d:.2} GiB short of the {d:.1} GiB floor and margin, and no arena can give it back", .{ lz.rank, need, @as(f64, @floatFromInt(add)) / gib, @as(f64, @floatFromInt(avail0)) / gib, @as(f64, @floatFromInt(short)) / gib, @as(f64, @floatFromInt(lz.floor + lz.margin)) / gib });
                return error.ContextMemory;
            };
            var gave = true;
            gb.f(gb.ctx, short) catch |err| {
                gave = false;
                std.log.err("inkling2 rank {d}: {d} positions need {d:.2} GiB more context memory; the arena cannot give {d:.2} GiB back ({t})", .{ lz.rank, need, @as(f64, @floatFromInt(add)) / gib, @as(f64, @floatFromInt(short)) / gib, err });
            };
            if (try agreeMax(m, @intFromBool(!gave)) != 0) {
                lz.stats.refused += 1;
                return error.ContextMemory;
            }
        }
        // the floor itself is never crossed: checked again after the give-back, on every rank
        const avail1 = fill.readAvail() orelse 0;
        if (try agreeMax(m, @intFromBool(avail1 < lz.floor + add)) != 0) {
            lz.stats.refused += 1;
            std.log.err("inkling2 rank {d}: {d} positions need {d:.2} GiB more context memory; MemAvailable {d:.2} GiB would fall under the {d:.1} GiB floor", .{ lz.rank, need, @as(f64, @floatFromInt(add)) / gib, @as(f64, @floatFromInt(avail1)) / gib, @as(f64, @floatFromInt(lz.floor)) / gib });
            return error.ContextMemory;
        }
        // xr: a step at a time, every entry one piece a step (trims, to a step, then unmap whole pieces exactly)
        const t_map = now();
        var failed = false;
        var s = (old / lz.step + 1) * lz.step;
        map: while (true) : (s += lz.step) {
            const t = @min(s, new);
            for (lz.entries.items) |*e| e.range.mapTo(&lz.vmm, e.bytesAt(t)) catch {
                failed = true;
                break :map;
            };
            if (t == new) break;
        }
        const map_ms = @as(f64, @floatFromInt(now() - t_map)) / 1e6;
        if (try agreeMax(m, @intFromBool(failed)) != 0) {
            for (lz.entries.items) |*e| e.range.unmapTo(&lz.vmm, e.bytesAt(old));
            lz.stats.refused += 1;
            std.log.err("inkling2 rank {d}: mapping {d} positions of context failed on a rank; the chunk is refused", .{ lz.rank, new });
            return error.ContextMemory;
        }
        const t_hooks = now();
        try lz.setMapped(m, old, new);
        const hooks_ms = @as(f64, @floatFromInt(now() - t_hooks)) / 1e6;
        lz.stats.grows += 1;
        lz.stats.peak = @max(lz.stats.peak, new);
        std.log.info("inkling2 rank {d}: context mapped {d} -> {d} of {d} positions (+{d:.2} GiB, {d:.2} GiB mapped; MemAvailable {d:.2} -> {d:.2} GiB; mapped in {d:.0} ms, the views' hooks {d:.0} ms)", .{ lz.rank, old, new, lz.context, @as(f64, @floatFromInt(add)) / gib, @as(f64, @floatFromInt(lz.mappedBytes())) / gib, @as(f64, @floatFromInt(avail0)) / gib, @as(f64, @floatFromInt(fill.readAvail() orelse 0)) / gib, map_ms, hooks_ms });
    }

    /// The positions past max(budget, keep) unmapped (a new sequence: Adapt.reclaim, keep = the live length; every
    /// rank alike). Nothing reads past the sequence's length, and the stream is idle.
    pub fn trim(lz: *Lazy, m: *Model, keep: usize) !void {
        const new = @max(lz.budget, @min(lz.context, roundUp(keep, lz.step)));
        if (new >= lz.mapped) return;
        const old = lz.mapped;
        try m.s.synchronize();
        const before = lz.mappedBytes();
        for (lz.entries.items) |*e| e.range.unmapTo(&lz.vmm, e.bytesAt(new));
        try lz.setMapped(m, old, new);
        lz.stats.trims += 1;
        std.log.info("inkling2 rank {d}: context unmapped {d} -> {d} positions (-{d:.2} GiB; MemAvailable {d:.2} GiB)", .{ lz.rank, old, new, @as(f64, @floatFromInt(before - lz.mappedBytes())) / gib, @as(f64, @floatFromInt(fill.readAvail() orelse 0)) / gib });
    }

    /// The views' lengths and the hooks at `new` positions; the graphs dropped (the score scratch's stride is in them).
    fn setMapped(lz: *Lazy, m: *Model, old: usize, new: usize) !void {
        lz.mapped = new;
        for (lz.entries.items) |*e| e.view.len = e.bytesAt(new);
        for (lz.hooks.items) |h| try h.f(h.ctx, old, new);
        m.graphs.drop(m.gpa);
    }

    // ---- the arena (Model.allocArena; adapt.zig giveBack / reclaim)

    /// Slot array `i` (0 NVFP4 up, 1 NVFP4 down, 2 MXFP8 up, 3 MXFP8 down) as a range of arena_chunk pieces (granule
    /// pieces when small), `bytes` mapped: a borrowed view of it.
    pub fn arenaPart(lz: *Lazy, i: usize, bytes: usize) !Buf {
        const n = @max(bytes, 16);
        var r = try Range.reserve(&lz.vmm, n, if (n >= 4 * arena_chunk) arena_chunk else lz.vmm.granule);
        errdefer r.deinit(&lz.vmm);
        try r.mapTo(&lz.vmm, n);
        lz.arena[i].deinit(&lz.vmm);
        lz.arena[i] = r;
        lz.arena_on = true;
        return .{ .d = lz.d, .ptr = r.va, .len = n, .borrowed = true };
    }

    /// Bytes cutting pool `f` (0 NVFP4, 1 MXFP8) to `n` slots would release (whole chunks past them).
    pub fn arenaFreed(lz: *const Lazy, words: [2][2]usize, f: usize, n: usize) usize {
        return lz.arena[2 * f].freedAt(n * words[f][0] * 4) + lz.arena[2 * f + 1].freedAt(n * words[f][1] * 4);
    }

    /// Bytes mapping pool `f` for `n` slots would add.
    pub fn arenaAdded(lz: *const Lazy, words: [2][2]usize, f: usize, n: usize) usize {
        return lz.arena[2 * f].addedAt(n * words[f][0] * 4) + lz.arena[2 * f + 1].addedAt(n * words[f][1] * 4);
    }

    /// Pool `f` cut to its first `n` slots (they hold every resident: the caller moved them there; the stream is
    /// idle): the chunks past them unmapped, the arena's count and views set. Returns the bytes released.
    pub fn arenaShrink(lz: *Lazy, a: *model_mod.Arena, f: usize, n: usize) u64 {
        const freed = lz.arenaFreed(a.words, f, n);
        lz.arena[2 * f].unmapTo(&lz.vmm, n * a.words[f][0] * 4);
        lz.arena[2 * f + 1].unmapTo(&lz.vmm, n * a.words[f][1] * 4);
        lz.setArenaSlots(a, f, n);
        lz.stats.gave += freed;
        return freed;
    }

    /// Pool `f` mapped for `n` slots (at most its slots at allocArena); the new slots hold nothing yet.
    pub fn arenaGrow(lz: *Lazy, a: *model_mod.Arena, f: usize, n: usize) !void {
        if (n > lz.arena_boot[f]) return error.PastArena;
        const added = lz.arenaAdded(a.words, f, n);
        try lz.arena[2 * f].mapTo(&lz.vmm, n * a.words[f][0] * 4);
        try lz.arena[2 * f + 1].mapTo(&lz.vmm, n * a.words[f][1] * 4);
        lz.setArenaSlots(a, f, n);
        lz.stats.regained += added;
    }

    fn setArenaSlots(lz: *Lazy, a: *model_mod.Arena, f: usize, n: usize) void {
        _ = lz;
        if (f == 0) {
            a.nv_slots = n;
            a.nv_up.len = @max(16, n * a.words[0][0] * 4);
            a.nv_down.len = @max(16, n * a.words[0][1] * 4);
        } else {
            a.mx_slots = n;
            a.mx_up.len = @max(16, n * a.words[1][0] * 4);
            a.mx_down.len = @max(16, n * a.words[1][1] * 4);
        }
        a.resident[f] = n; // a give-back needs no transient ring (adapt.zig refuses one)
    }

    fn now() i128 {
        var t: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t);
        return @as(i128, t.sec) * std.time.ns_per_s + t.nsec;
    }

    /// x-merge: the most positions a growth from now maps with `room` bytes (MemAvailable over the floor and the
    /// margin: ensure's test, in its steps), without a give-back. The EXL3 store's whole reach (its slots are
    /// exl3.zig's and give nothing back); the native store's before its arena gives memory back.
    pub fn reach(lz: *const Lazy, room: u64) usize {
        var best = lz.mapped;
        while (best < lz.context) {
            const next = @min(lz.context, (best / lz.step + 1) * lz.step);
            var add: u64 = 0;
            for (lz.entries.items) |*e| add += e.range.addedAt(e.bytesAt(next));
            if (add > room) break;
            best = next;
        }
        return best;
    }

    /// One line for the logs: what is mapped, what the whole context would take, what of the rest the arena takes.
    pub fn summary(lz: *const Lazy, buf: []u8) []const u8 {
        const g = struct {
            fn f(x: usize) f64 {
                return @as(f64, @floatFromInt(x)) / gib;
            }
        }.f;
        const d = lz.deferred();
        var cap_text: [64]u8 = undefined;
        const how = if (lz.arena_cap) |c| std.fmt.bufPrint(&cap_text, "TF_INKLING2_LAZY_ARENA_GIB={d:.2}", .{g(c)}) catch "" else "TF_INKLING2_LAZY_ARENA_GIB unset: all";
        return std.fmt.bufPrint(buf, "lazy context: {d} of {d} positions mapped ({d:.2} GiB; {d:.2} GiB for the whole context), step {d}, margin {d:.1} GiB over the {d:.1} GiB floor; of the {d:.2} GiB not mapped the arena takes {d:.2} GiB ({s})", .{ lz.mapped, lz.context, g(lz.mappedBytes()), g(lz.bytesAt(lz.context)), lz.step, g(lz.margin), g(lz.floor), g(d), g(d - lz.heldBack()), how }) catch buf[0..0];
    }
};
