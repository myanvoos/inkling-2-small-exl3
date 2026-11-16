//! xk-rows: Inkling-2's conversation states on NVMe through the core's rows store (core/rows_store.zig: rows in
//! blocks states share, each state's own parts in a file), so a conversation's state outlives the requests that
//! take the one slot between its turns (TF_INKLING2_PROMPT_DISK; unset: on, =0 or TF_INKLING2_LEGACY=1: off).
//!
//! Every rank keeps its own shard on its own disk under <TF_INKLING2_ROWS_DIR>/rank<r>/<identity>: rank 0 holds the
//! store proper (its index of states and blocks, the value-and-age eviction under TF_INKLING2_ROWS_GIB, the free-disk
//! floor, the daily write budget, the check of every file when the store opens); the other ranks hold files of the
//! same names with their own heads' rows, which rank 0's steps (engine.zig rows_save / rows_load) name one by one,
//! and a sweep at load removes whatever rank 0's index lacks. A state's image (rows_layout.zig): its own parts (the
//! words, the convolution rings, the draft's f_rnn), the global layers' K / V rows from position 0, the sliding
//! layers' rings of the last window. Rank 0's index keys a state by its salted ids (rows_layout.zig): blocks are
//! shared only where the rows are the same computation, so a state read back is the one saved, bit for bit.
//!
//! Copies run on TF_INKLING2_ROWS_THREADS threads (default 6), each with a pinned stage of TF_INKLING2_ROWS_STAGE_MIB
//! (default 8): device rows to the stage to a file (rows_file.zig's Writer: through a .part name, fdatasync, the pages
//! dropped), or a file read past the page cache (O_DIRECT) into the stage, its hash checked, to the device.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const ink = @import("inkling2");
const reuse = @import("reuse.zig");
const lay = @import("rows_layout.zig");

const pc = api.prompt_cache;
const store_mod = pc.rows; // core/rows_store.zig
const image = store_mod.image;
const file = store_mod.file;
const chain = store_mod.chain;
const Want = store_mod.write.Want;
const Buf = cuda.DeviceBuffer;
const Model = ink.model.Model;
const Draft = ink.dspark.Draft;
const Allocator = std.mem.Allocator;
const gib: u64 = 1 << 30;

/// Where a save's state comes from: the live state (its length), or a snapshot slot (reuse.zig) at its position,
/// the global rows below it from the live caches; xr: a checkpoint copied beside the prefill (`ckpt`): the parts the
/// next chunk overwrites from the store's stage, the rest from the live caches.
pub const Source = union(enum) { live, slot: usize, ckpt };

pub const Settings = struct {
    on: bool = false,
    root: []const u8 = "",
    cap: u64 = 64 * gib,
    free: u64 = 64 * gib,
    daily: u64 = 256 * gib,
    min: usize = 2048,
    every: usize = 8192,
    threads: usize = 6,
    stage: usize = 8 << 20,
    keep_requests: usize = 4, // xr: the trim hint's window (TF_INKLING2_ROWS_KEEP_REQUESTS; 0: no hint)

    fn env(name: [*:0]const u8) ?[]const u8 {
        const v = std.c.getenv(name) orelse return null;
        return std.mem.span(v);
    }

    fn num(name: [*:0]const u8, default: u64) u64 {
        const v = env(name) orelse return default;
        return std.fmt.parseInt(u64, v, 10) catch default;
    }

    /// The settings from the environment; on only beside prompt reuse (its slots and book).
    pub fn fromEnv(gpa: Allocator, reuse_on: bool) !Settings {
        var s: Settings = .{};
        s.on = reuse_on and !ink.defaults.legacy();
        if (env("TF_INKLING2_PROMPT_DISK")) |v| s.on = reuse_on and !std.mem.eql(u8, v, "0");
        if (!s.on) return s;
        s.root = if (env("TF_INKLING2_ROWS_DIR")) |d| try gpa.dupe(u8, d) else try std.fmt.allocPrint(gpa, "{s}/.cache/tensorfold/inkling2-rows", .{env("HOME") orelse "/var/tmp"});
        s.cap = num("TF_INKLING2_ROWS_GIB", 64) * gib;
        s.free = num("TF_INKLING2_ROWS_FREE_GIB", 64) * gib;
        // a long conversation writes ~200 MiB a rank a turn (its new rows, its window's blocks, its own file)
        s.daily = num("TF_INKLING2_ROWS_DAILY_GIB", 256) * gib;
        s.min = @max(num("TF_INKLING2_ROWS_MIN", 2048), 256);
        // checkpoints on the store's blocks (and so on every prompt chunk width up to it: a cold pass cuts no chunk)
        s.every = std.mem.alignBackward(usize, num("TF_INKLING2_ROWS_EVERY", 8192), lay.block_tokens);
        s.threads = std.math.clamp(num("TF_INKLING2_ROWS_THREADS", 6), 1, 32);
        s.stage = std.math.clamp(num("TF_INKLING2_ROWS_STAGE_MIB", 8), 1, 64) << 20;
        s.keep_requests = num("TF_INKLING2_ROWS_KEEP_REQUESTS", 4);
        return s;
    }
};

/// Host bytes a rank's store takes beside the model (its stages, its lists' device buffers): what the bank's sizing
/// must leave for it (engine.zig).
pub fn reserveBytes(s: Settings, world: usize, ckpt_stage: u64) u64 {
    if (!s.on) return 0;
    return s.threads * (s.stage + 4096) + (world + 1) * pay_cap + 4096 + (if (s.every > 0) ckpt_stage else 0);
}

/// xr: device bytes a checkpoint's stage takes: the convolution rings (Model.saveState's first part), then the
/// draft's sliding windows and f_rnn (Draft.saveContext's layout).
pub fn ckptStageBytes(m: *const Model, draft: ?*const Draft) u64 {
    return m.snapRingBytes() + (if (draft) |d| d.snapBytes() else 0);
}

/// A step's file list at most (engine.zig: rows_save and rows_load carry it in a second gather).
pub const pay_cap: usize = 1 << 20;

/// One file of a save or a load: what it is, its group and block (a block's), its name; its bytes once written.
const Item = struct { op: lay.Op, window: u32 = 0, k: u32 = 0, name: chain.Name, bytes: u64 = 0, written: bool = false };

pub const Store = struct {
    gpa: Allocator,
    ctx: *const cuda.Context,
    m: *Model,
    draft: ?*Draft,
    slots: *const reuse.Slots,
    rank: usize,
    world: usize,
    set: Settings,
    dims: lay.Dims,
    local: []bool,
    dlocal: [ink.dspark.layers]bool = undefined,
    ordinal: []u16, // a layer's place among the sliding (or global) layers of its kind
    dordinal: [ink.dspark.layers]u16 = undefined,
    ring_off: [4]u64 = .{ 0, 0, 0, 0 }, // where each convolution ring sits in a snapshot slot's rings buffer
    identity: u64,
    dir: [:0]u8, // this rank's directory of states
    root: [:0]u8, // this rank's root (rank 0's: the store's)
    rows: ?*store_mod.Rows = null, // rank 0
    stages: []cuda.HostBuffer,
    pay_mine: Buf,
    pay_all: Buf,
    ag_mine: Buf,
    ag_all: Buf,
    list: std.ArrayList(u8) = .empty, // rank 0: the next step's file list
    drops: std.ArrayList(lay.Entry) = .empty, // rank 0: files the other ranks still hold and must delete
    pending: ?Pending = null, // rank 0: the save planned, run by the step, then published
    saved_sizes: std.ArrayList(u64) = .empty, // rank 0: the planned save's files' bytes, as the step wrote them
    written: std.ArrayList(Item) = .empty, // every rank: the last save's files on this disk, until the ranks agree on it
    // xr: checkpoints copied beside the prefill (startCkpt, ckptMain, commitCkpts)
    ckpt_stage: ?Buf = null, // the parts the next chunk overwrites: the convolution rings, then the draft's part
    ckpt_event: ?cuda.Event = null, // recorded on the model's stream after the stage's copies
    ckpt_thread: ?std.Thread = null, // the running copy (one at a time: the stage is its)
    ckpt_run: CkptRun = .{},
    ckpt_done: std.ArrayList(CkptRun) = .empty, // copies that ended, awaiting the commit (every rank)
    ckpt_plans: std.ArrayList(Pending) = .empty, // rank 0: their plans, in order
    inflight: std.AutoHashMapUnmanaged(chain.Stem, void) = .empty, // rank 0: new blocks of plans not yet committed
    ckpt_failed: bool = false, // a checkpoint that did not start here: the commit drops them all
    gate: lay.WindowGate = .{}, // the live windows the running copy reads (engine.zig's run waits on it)
    // xr, rank 0: the trim hint's evidence: the long states the last requests left on disk (newest last)
    requests: u64 = 0, // prompts seen (newRequest)
    recent: lay.Recents = .{},
    load_mapped: bool = true, // the last read's context mapped (every rank agrees it: lazy.ensure): else not its files
    stats: Stats = .{},

    const Pending = struct { name: chain.Name, at: u32, tokens: []u32, wants: []Want, new: []usize, bytes: u64, began: i128 };

    pub const Stats = struct { saves: u64 = 0, loads: u64 = 0, failed: u64 = 0, written: u64 = 0, read: u64 = 0, skipped: u64 = 0 };

    // ---- opening (every rank, in Engine.load after the snapshot slots; every collective in the same order)

    /// The store of this rank when every rank's settings and identity agree (null: off on every rank).
    pub fn open(gpa: Allocator, ctx: *const cuda.Context, m: *Model, draft: ?*Draft, slots: *const reuse.Slots, rank: usize, world: usize, dir_model: []const u8, set: Settings) !?*Store {
        var owned = false; // set.root (allocated when on) is the store's once it opens
        defer if (!owned and set.on) gpa.free(set.root);
        // every rank answers, on or off, so the ranks agree on the same exchange
        const mine_id: u64 = if (set.on and slots.n > 0) identityOf(m, draft, dir_model, world) catch 0 else 0;
        const id = try agreeId(m, mine_id, world);
        if (id == 0) {
            if (set.on) std.log.warn("inkling2 rank {d}: conversation states on disk off: the ranks' builds, views or switches differ (or a rank has them off)", .{rank});
            return null;
        }
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa, .ctx = ctx, .m = m, .draft = draft, .slots = slots, .rank = rank, .world = world, .set = set, .dims = undefined, .local = &.{}, .ordinal = &.{}, .identity = id, .dir = undefined, .root = undefined, .stages = &.{}, .pay_mine = undefined, .pay_all = undefined, .ag_mine = undefined, .ag_all = undefined };
        try s.layout();
        errdefer {
            gpa.free(s.local);
            gpa.free(s.ordinal);
        }
        s.root = try std.fmt.allocPrintSentinel(gpa, "{s}/rank{d}", .{ set.root, rank }, 0);
        errdefer gpa.free(s.root);
        var h = std.hash.Wyhash.init(0x726f7773); // the core's directory name of an identity (rows_store.init)
        h.update(std.mem.asBytes(&id));
        s.dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ s.root, h.final() }, 0);
        errdefer gpa.free(s.dir);
        s.pay_mine = try Buf.alloc(m.d, pay_cap);
        errdefer s.pay_mine.free();
        s.pay_all = try Buf.alloc(m.d, pay_cap * world);
        errdefer s.pay_all.free();
        s.ag_mine = try Buf.alloc(m.d, 16);
        errdefer s.ag_mine.free();
        s.ag_all = try Buf.alloc(m.d, 16 * world);
        errdefer s.ag_all.free();
        s.stages = try gpa.alloc(cuda.HostBuffer, set.threads);
        var made: usize = 0;
        errdefer {
            for (s.stages[0..made]) |*b| b.free();
            gpa.free(s.stages);
        }
        while (made < set.threads) : (made += 1) s.stages[made] = try cuda.HostBuffer.alloc(m.d, set.stage + 4096);
        if (set.every > 0) { // xr: checkpoints copied beside the prefill
            s.ckpt_stage = try Buf.alloc(m.d, ckptStageBytes(m, draft));
            s.ckpt_event = cuda.Event.init(m.d, false) catch |err| {
                s.ckpt_stage.?.free();
                s.ckpt_stage = null;
                return err;
            };
        }
        errdefer {
            if (s.ckpt_stage) |*b| b.free();
            if (s.ckpt_event) |*e| e.deinit();
        }
        // the directory (and rank 0's store) made, then agreed: a rank that cannot open leaves every rank without one
        const opened = blk: {
            makePath(s.dir) catch |err| break :blk err;
            if (rank == 0) {
                const vt = &snaps_vtable;
                s.rows = store_mod.Rows.open(gpa, .{ .ptr = s, .vtable = vt }, .{ .root = s.root, .identity = id, .cap = set.cap, .floor = set.free, .daily = set.daily, .sync_writes = true, .staging_bytes = set.stage }) catch |err| break :blk err;
            }
            break :blk {};
        };
        errdefer if (s.rows) |r| r.deinit();
        if (opened) |_| {} else |err| std.log.warn("inkling2 rank {d}: conversation states on disk cannot open under {s} ({t})", .{ rank, s.dir, err });
        if (!try s.agree(if (opened) |_| true else |_| false)) return error.RowsUnavailable;
        try s.sweep();
        if (rank == 0) {
            const r = s.rows.?;
            std.log.info("inkling2 rank 0/{d}: conversation states on disk in {s}: {d} states, {d} MiB of {d} GiB; checkpoints every {d} positions, saves from {d}; {d} I/O threads x {d} MiB pinned", .{ world, r.dir, r.states.items.len, r.held() >> 20, set.cap / gib, set.every, set.min, set.threads, set.stage >> 20 });
        } else std.log.info("inkling2 rank {d}/{d}: conversation states on disk in {s} (rank 0's index names them)", .{ rank, world, s.dir });
        owned = true;
        return s;
    }

    pub fn close(s: *Store) void {
        s.joinCkpt();
        s.dropCkpts(); // what no commit took: its files go (rank 0: its plans' reservations)
        if (s.ckpt_stage) |*b| b.free();
        if (s.ckpt_event) |*e| e.deinit();
        s.ckpt_done.deinit(s.gpa);
        s.ckpt_plans.deinit(s.gpa);
        s.inflight.deinit(s.gpa);
        if (s.rows) |r| r.deinit();
        if (s.pending) |p| s.freePending(p);
        for (s.stages) |*b| b.free();
        s.gpa.free(s.stages);
        s.pay_mine.free();
        s.pay_all.free();
        s.ag_mine.free();
        s.ag_all.free();
        s.list.deinit(s.gpa);
        s.drops.deinit(s.gpa);
        s.saved_sizes.deinit(s.gpa);
        s.written.deinit(s.gpa);
        s.gpa.free(s.local);
        s.gpa.free(s.ordinal);
        s.gpa.free(s.dir);
        s.gpa.free(s.root);
        s.gpa.free(s.set.root);
        s.gpa.destroy(s);
    }

    fn layout(s: *Store) !void {
        const m = s.m;
        const L = m.c.layers;
        s.local = try s.gpa.alloc(bool, L);
        s.ordinal = try s.gpa.alloc(u16, L);
        var nl: u16 = 0;
        var ng: u16 = 0;
        for (m.layers, 0..) |l, i| {
            s.local[i] = l.local;
            s.ordinal[i] = if (l.local) nl else ng;
            if (l.local) nl += 1 else ng += 1;
        }
        var off: u64 = 0;
        for (m.state.rings_all, 0..) |b, k| {
            s.ring_off[k] = off;
            off += b.len;
        }
        const kv: u32 = @intCast(m.hkv() * m.c.head_dim * 2);
        s.dims = .{ .local = s.local, .ring_bytes = .{ m.state.rings_all[0].len, m.state.rings_all[1].len, m.state.rings_all[2].len, m.state.rings_all[3].len }, .kv_bytes = kv, .window = m.c.window };
        if (s.draft) |d| {
            var dl: u16 = 0;
            var dg: u16 = 0;
            for (ink.dspark.local_layers, 0..) |l, i| {
                s.dlocal[i] = l;
                s.dordinal[i] = if (l) dl else dg;
                if (l) dl += 1 else dg += 1;
            }
            s.dims.draft = .{ .local = &s.dlocal, .kv_bytes = @intCast(d.kvwl * 2), .window = ink.dspark.window, .frnn_bytes = ink.dspark.H * 2 };
        }
    }

    /// Every rank's identity agreed: rank 0's when every rank's equals it, else 0 (off everywhere).
    fn agreeId(m: *Model, id: u64, world: usize) !u64 {
        if (world == 1) return id;
        const v = [2]u64{ id, 0 };
        var mine = try Buf.alloc(m.d, 16);
        defer mine.free();
        var all = try Buf.alloc(m.d, 16 * world);
        defer all.free();
        try mine.upload(0, std.mem.sliceAsBytes(&v));
        try ink.forward.gatherNccl(m, m.s, mine.ptr, all.ptr, 16);
        try m.s.synchronize();
        var got: [32]u64 = undefined;
        try all.download(0, std.mem.sliceAsBytes(got[0 .. 2 * world]));
        for (0..world) |r| if (got[2 * r] != got[0]) return 0;
        return got[0];
    }

    /// Whether every rank's `ok` holds (an exchange of a word; every rank calls it at the same point).
    pub fn agree(s: *Store, ok: bool) !bool {
        if (s.world == 1) return ok;
        const v = [4]u32{ @intFromBool(ok), 0, 0, 0 };
        try s.ag_mine.upload(0, std.mem.sliceAsBytes(&v));
        try ink.forward.gatherNccl(s.m, s.m.s, s.ag_mine.ptr, s.ag_all.ptr, 16);
        try s.m.s.synchronize();
        var got: [4 * 32]u32 = undefined;
        try s.ag_all.download(0, std.mem.sliceAsBytes(got[0 .. 4 * s.world]));
        for (0..s.world) |r| if (got[4 * r] == 0) return false;
        return true;
    }

    /// Rank 0's `bytes` to every rank (a list past a step's header, or the sweep's at load) into `out` (the same
    /// length on every rank; rank 0's `out` may be its `bytes`).
    pub fn exchange(s: *Store, bytes: []const u8, out: []u8) !void {
        if (bytes.len > pay_cap or out.len != bytes.len) return error.RowsListTooLong;
        if (s.rank == 0 and out.ptr != bytes.ptr) @memcpy(out, bytes);
        if (s.world == 1) return;
        const n = std.mem.alignForward(usize, @max(bytes.len, 4), 4);
        if (s.rank == 0) try s.pay_mine.uploadAsync(0, bytes, s.m.s.handle);
        try ink.forward.gatherNccl(s.m, s.m.s, s.pay_mine.ptr, s.pay_all.ptr, n);
        try s.m.s.synchronize();
        if (s.rank != 0) try s.pay_all.download(0, out); // rank 0's segment first
    }

    // ---- the image the core's store indexes by (rank 0); its copies run here, not through the store's writer

    const image_vtable: store_mod.Image = .{ .parts = imageParts, .gather = noGather, .blank = noBlank, .scatter = noScatter };
    const snaps_vtable: pc.Snapshots.VTable = .{ .bytes = noBytes, .save = noSave, .restore = noRestore, .drop = noDrop, .rows = &image_vtable };

    fn imageParts(ptr: *anyopaque, at: u32, out: *std.ArrayList(image.Part), a: Allocator) anyerror!void {
        const s: *Store = @ptrCast(@alignCast(ptr));
        var list: std.ArrayList(lay.Part) = .empty;
        defer list.deinit(a);
        try lay.parts(s.dims, at, &list, a);
        for (list.items) |p| try out.append(a, .{ .bytes = p.bytes, .row = p.row, .window = p.window });
    }
    fn noGather(_: *anyopaque, _: pc.Saved, _: []const image.Range, _: []u8) anyerror!void {
        return error.RowsCopiesAreTheFamilys;
    }
    fn noBlank(_: *anyopaque, _: u32) anyerror!pc.Saved {
        return error.RowsCopiesAreTheFamilys;
    }
    fn noScatter(_: *anyopaque, _: pc.Saved, _: []const image.Range, _: []const u8) anyerror!void {
        return error.RowsCopiesAreTheFamilys;
    }
    fn noBytes(_: *anyopaque, _: u32) u64 {
        return 0;
    }
    fn noSave(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
        return error.RowsCopiesAreTheFamilys;
    }
    fn noRestore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.RowsCopiesAreTheFamilys;
    }
    fn noDrop(_: *anyopaque, _: pc.Saved) void {}

    // ---- rank 0: what to save, what to read

    /// The stored state reaching furthest into `prompt` past `longer_than` (its ids the prompt's first, salts aside;
    /// at least one prompt row after it): its index in the store, or null.
    pub fn best(s: *const Store, prompt: []const u32, longer_than: usize) ?usize {
        const r = s.rows orelse return null;
        var out: ?usize = null;
        for (r.states.items, 0..) |st, i| {
            if (st.at <= longer_than or st.at >= prompt.len or st.tokens.len != st.at) continue;
            if (out) |o| if (st.at < r.states.items[o].at or (st.at == r.states.items[o].at and st.used <= r.states.items[o].used)) continue;
            if (!lay.prefixOf(st.tokens, prompt)) continue;
            out = i;
        }
        return out;
    }

    /// State `i`'s salted ids (rank 0).
    pub fn tokensOf(s: *const Store, i: usize) []const u32 {
        return s.rows.?.states.items[i].tokens;
    }

    /// Whether a state of salted ids `tokens` is on disk (touched: the newest, if so).
    pub fn has(s: *Store, tokens: []const u32) bool {
        const r = s.rows orelse return false;
        const name = chain.state(r.keying, tokens, &.{}, &.{}, @intCast(tokens.len));
        const i = r.find(name) orelse return false;
        r.touch(i);
        return true;
    }

    /// Rank 0: the list of a save of the state of salted ids `tokens` (its position their count) into s.list, or
    /// false: on disk already, or refused (the daily budget, the cap, the free-disk floor). Room is made first:
    /// every state the eviction removed goes into the list for the other ranks too.
    pub fn planSave(s: *Store, tokens: []const u32) !bool {
        return s.planFor(tokens, false);
    }

    /// xr: rank 0's plan of a checkpoint copied beside the prefill (rows_ckpt): as planSave, the blocks of earlier
    /// checkpoints not yet committed counted as written (the commit takes them in order), the list in the copy's
    /// order (lay.ckptOrder: the files that read the live windows first).
    pub fn planCkpt(s: *Store, tokens: []const u32) !bool {
        return s.planFor(tokens, true);
    }

    fn present(s: *const Store, r: *const store_mod.Rows, stem: chain.Stem, ckpt: bool) bool {
        return r.blocks.contains(stem) or (ckpt and s.inflight.contains(stem));
    }

    fn planFor(s: *Store, tokens: []const u32, ckpt: bool) !bool {
        const r = s.rows orelse return false;
        const gpa = s.gpa;
        const at: u32 = @intCast(tokens.len);
        if (s.pending) |p| {
            // a plan whose step never ran (its exchange failed): its disk reservation back
            r.admission.finish(r.dir, p.bytes, false);
            s.freePending(p);
            s.pending = null;
        }
        const name = chain.state(r.keying, tokens, &.{}, &.{}, at);
        if (r.find(name)) |i| {
            r.touch(i);
            return false;
        }
        const parts = try r.parts(gpa, at);
        defer gpa.free(parts);
        var wants: std.ArrayList(Want) = .empty;
        errdefer wants.deinit(gpa);
        try r.blockNames(gpa, parts, tokens, &.{}, &.{}, at, &wants);
        var own: std.ArrayList(image.Range) = .empty;
        defer own.deinit(gpa);
        try image.ownRanges(parts, at, r.keying.block, &own, gpa);
        var bytes: u64 = image.total(own.items) + 4 * (3 + @as(u64, at)) + @sizeOf(file.Trailer);
        for (wants.items) |w| if (!s.present(r, chain.stem(w.name), ckpt)) {
            bytes += w.bytes;
        };
        if (!r.budget(bytes)) return s.skip(at, error.DailyBudgetSpent);
        // the store's eviction removes states (and blocks no other names) on this rank: the others follow by name
        var before_states: std.AutoHashMapUnmanaged(chain.Stem, void) = .empty;
        defer before_states.deinit(gpa);
        for (r.states.items) |st| try before_states.put(gpa, chain.stem(st.name), {});
        var before_blocks: std.AutoHashMapUnmanaged(chain.Stem, void) = .empty;
        defer before_blocks.deinit(gpa);
        var bit = r.blocks.keyIterator();
        while (bit.next()) |k| try before_blocks.put(gpa, k.*, {});
        const others_before = lanes.learned_dirs.totalOthers(r.root, r.dir) catch 0;
        // the blocks this state names are the job's while room is made (the store's collect keeps a job's wants: an
        // older checkpoint of this conversation may go, its blocks stay for this one)
        // (xr: and those of the checkpoints not yet committed: their states name them too)
        var guard_wants: std.ArrayList(Want) = .empty;
        defer guard_wants.deinit(gpa);
        try guard_wants.appendSlice(gpa, wants.items);
        for (s.ckpt_plans.items) |cp| try guard_wants.appendSlice(gpa, cp.wants);
        var guard: store_mod.write.Job = .{ .entry = undefined, .starts = &.{}, .name = name, .parts = &.{}, .wants = guard_wants.items, .own = &.{} };
        r.job = &guard;
        const roomy = r.room(bytes);
        r.job = null;
        for (r.states.items) |st| _ = before_states.remove(chain.stem(st.name));
        var sit = before_states.keyIterator();
        while (sit.next()) |k| try s.drop(.drop_state, k.*);
        bit = r.blocks.keyIterator();
        while (bit.next()) |k| _ = before_blocks.remove(k.*);
        var dit = before_blocks.keyIterator();
        while (dit.next()) |k| try s.drop(.drop_block, k.*);
        const wipe = (lanes.learned_dirs.totalOthers(r.root, r.dir) catch 0) < others_before;
        if (!roomy) return s.skip(at, error.OverCap);
        switch (r.admission.reserve(r.dir, bytes)) {
            .ready => {},
            .refused => return s.skip(at, error.UnderFreeDiskFloor),
            .quiet => return s.skip(at, error.RetryAfterFailedWrite), // the admission's backoff after a failed save
        }
        // the list: the other ranks' deletions, the blocks the disk lacks, then the state's own file
        var es: std.ArrayList(lay.Entry) = .empty;
        defer es.deinit(gpa);
        try es.appendSlice(gpa, s.drops.items);
        var new: std.ArrayList(usize) = .empty;
        errdefer new.deinit(gpa);
        for (wants.items, 0..) |w, i| if (!s.present(r, chain.stem(w.name), ckpt)) {
            try es.append(gpa, .{ .op = @intFromEnum(lay.Op.block), .window = w.window, .k = w.k, .name = w.name });
            try new.append(gpa, i);
        };
        try es.append(gpa, .{ .op = @intFromEnum(lay.Op.state), .window = 0, .k = 0, .name = name });
        if (ckpt) _ = try lay.ckptOrder(es.items, gpa);
        try s.encode(.{ .at = at, .count = 0, .flags = if (wipe) lay.flag_wipe else 0 }, es.items);
        s.drops.clearRetainingCapacity();
        s.pending = .{ .name = name, .at = at, .tokens = try gpa.dupe(u32, tokens), .wants = try wants.toOwnedSlice(gpa), .new = try new.toOwnedSlice(gpa), .bytes = bytes, .began = nowNs() };
        return true;
    }

    fn skip(s: *Store, at: u32, err: anyerror) bool {
        s.stats.skipped += 1;
        std.log.warn("inkling2 rows: the state of {d} tokens stays off disk ({t})", .{ at, err });
        return false;
    }

    fn drop(s: *Store, op: lay.Op, stem: chain.Stem) !void {
        var e: lay.Entry = .{ .op = @intFromEnum(op), .window = 0, .k = 0, .name = @splat(0) };
        @memcpy(e.name[0..16], &stem);
        try s.drops.append(s.gpa, e);
    }

    fn freePending(s: *Store, p: Pending) void {
        s.gpa.free(p.tokens);
        s.gpa.free(p.wants);
        s.gpa.free(p.new);
    }

    /// Rank 0: the list of a read of state `i` into s.list; `same_rows`: the live state holds its rows from 0 (the
    /// book's holds), so only its own file and its window's blocks are read.
    pub fn planLoad(s: *Store, i: usize, same_rows: bool) !void {
        const r = s.rows.?;
        const gpa = s.gpa;
        r.touch(i); // read back: the newest for the store's eviction
        const st = r.states.items[i];
        const parts = try r.parts(gpa, st.at);
        defer gpa.free(parts);
        var wants: std.ArrayList(Want) = .empty;
        defer wants.deinit(gpa);
        try r.blockNames(gpa, parts, st.tokens, st.spans, st.starts, st.at, &wants);
        var es: std.ArrayList(lay.Entry) = .empty;
        defer es.deinit(gpa);
        try es.appendSlice(gpa, s.drops.items); // the other ranks' deletions waiting (eviction, a failed read)
        for (wants.items) |w| {
            if (same_rows and w.window == 0) continue;
            try es.append(gpa, .{ .op = @intFromEnum(lay.Op.block), .window = w.window, .k = w.k, .name = w.name });
        }
        try es.append(gpa, .{ .op = @intFromEnum(lay.Op.state), .window = 0, .k = 0, .name = st.name });
        try s.encode(.{ .at = st.at, .count = 0 }, es.items);
        s.drops.clearRetainingCapacity();
    }

    /// Rank 0: a state that failed to read back forgotten here, and named for the other ranks' next list.
    pub fn forget(s: *Store, i: usize) void {
        const r = s.rows.?;
        const name = r.states.items[i].name;
        s.drop(.drop_state, chain.stem(name)) catch {};
        r.remove(i);
    }

    fn encode(s: *Store, h: lay.Head, es: []const lay.Entry) !void {
        const n = lay.listBytes(es.len);
        if (n > pay_cap) return error.RowsListTooLong;
        try s.list.resize(s.gpa, n);
        _ = lay.encode(h, es, s.list.items);
    }

    // ---- every rank: a save, a read (engine.zig's rows_save / rows_load run these on every rank)

    /// The files of `list` written from `src` (the state at the list's position); rank 0's own file carries the
    /// state's ids. Deletions first (the other ranks'). True when every file was written.
    pub fn save(s: *Store, src: Source, list: []const u8) !bool {
        s.joinCkpt(); // xr: the stages are one copy's at a time
        s.written.clearRetainingCapacity(); // (a save that fails before its copies names none of an earlier one's)
        const got = try lay.decode(list);
        const at = got.head.at;
        if (got.head.flags & lay.flag_wipe != 0 and s.rank != 0) s.wipeOthers();
        var items: std.ArrayList(Item) = .empty;
        defer items.deinit(s.gpa);
        for (got.entries) |e| {
            const op = std.enums.fromInt(lay.Op, e.op) orelse return error.BadRowsList;
            switch (op) {
                .drop_block, .drop_state => if (s.rank != 0) unlinkStem(s.dir, e.name[0..16].*, extOf(if (op == .drop_block) .block else .state)),
                .block, .state => try items.append(s.gpa, .{ .op = op, .window = e.window, .k = e.k, .name = e.name }),
                else => return error.BadRowsList,
            }
        }
        const src_at: u32 = switch (src) {
            .live => @intCast(s.m.state.len),
            .slot => |i| @intCast(s.slots.at[i]),
            .ckpt => return error.RowsBadSource, // a checkpoint's copy is startCkpt's
        };
        // the copies run on other threads, outside the model's stream: every kernel and copy it holds done first
        try s.m.s.synchronize();
        if (src_at != at) return error.RowsSourceMoved;
        const meta_counts: [3]u32 = .{ at, 0, 0 };
        var meta: [2][]const u8 = .{ std.mem.asBytes(&meta_counts), &.{} };
        if (s.rank == 0) {
            const p = s.pending orelse return error.RowsNoPlan;
            meta[1] = std.mem.sliceAsBytes(p.tokens);
        }
        var w: Work = .{ .s = s, .src = src, .at = at, .items = items.items, .save = true, .meta = if (s.rank == 0) &meta else &.{} };
        const ok = w.run();
        if (!ok) {
            for (items.items) |it| if (it.written) unlinkStem(s.dir, chain.stem(it.name), extOf(it.op));
            s.stats.failed += 1;
        } else for (items.items) |it| s.written.append(s.gpa, it) catch {}; // (a miss: the next sweep takes it)
        if (s.rank == 0) {
            // the written files' bytes in list order (the new blocks, then the state's own), for publish's index
            s.saved_sizes.clearRetainingCapacity();
            for (items.items) |it| s.saved_sizes.append(s.gpa, it.bytes) catch {};
        }
        s.stats.written += w.bytes.load(.monotonic);
        return ok;
    }

    /// Every rank, once the ranks agreed on a save (`all`: every rank wrote every file): else no file of it stays on
    /// any disk. Rank 0's plan then joins the index, or goes (publish).
    pub fn settleSave(s: *Store, all: bool) void {
        if (!all) for (s.written.items) |it| unlinkStem(s.dir, chain.stem(it.name), extOf(it.op));
        s.written.clearRetainingCapacity();
        if (s.rank == 0) s.publish(all);
    }

    /// Rank 0, after every rank's save: the state and its new blocks into the index (`ok`), else its files gone.
    fn publish(s: *Store, ok: bool) void {
        const r = s.rows orelse return;
        const p = s.pending orelse return;
        defer {
            s.freePending(p);
            s.pending = null;
        }
        if (!ok) {
            for (p.new) |wi| unlinkStem(r.dir, chain.stem(p.wants[wi].name), ".blk");
            unlinkStem(r.dir, chain.stem(p.name), ".st");
            r.admission.finish(r.dir, p.bytes, false);
            return;
        }
        publishInto(s, r, p, s.saved_sizes.items) catch |err| {
            std.log.warn("inkling2 rows: the state of {d} tokens could not join the index ({t})", .{ p.at, err });
            r.admission.finish(r.dir, p.bytes, true);
            return;
        };
        r.admission.finish(r.dir, p.bytes, true);
        s.stats.saves += 1;
        const ms = @as(f64, @floatFromInt(nowNs() - p.began)) / 1e6;
        std.log.info("inkling2 rows: {d} tokens on disk: {d} new blocks, {d} MiB written a rank in {d:.0} ms ({d} states, {d} MiB of {d} GiB)", .{ p.at, p.new.len, p.bytes >> 20, ms, r.states.items.len, r.held() >> 20, s.set.cap / gib });
    }

    /// The plan's new blocks (`sizes`: their files' bytes in p.new's order, then the state's own file) and its state
    /// into rank 0's index.
    fn publishInto(s: *Store, r: *store_mod.Rows, p: Pending, sizes: []const u64) !void {
        _ = s;
        if (sizes.len != p.new.len + 1) return error.RowsSizes;
        try r.states.ensureUnusedCapacity(r.gpa, 1);
        try r.blocks.ensureUnusedCapacity(r.gpa, @intCast(p.new.len));
        for (p.new, 0..) |wi, j| {
            r.blocks.putAssumeCapacity(chain.stem(p.wants[wi].name), .{ .bytes = sizes[j], .refs = 0 });
            r.spent(sizes[j]);
        }
        var st: store_mod.State = .{ .name = p.name, .at = p.at, .tokens = try r.gpa.dupe(u32, p.tokens), .spans = &.{}, .starts = &.{}, .own = sizes[p.new.len], .rows = 0, .used = 0, .shared = false };
        for (p.wants) |w| st.rows += w.bytes;
        r.spent(st.own);
        r.states.appendAssumeCapacity(st);
        r.ref(p.wants, 1);
        r.touch(r.states.items.len - 1);
        r.counts.writes += 1;
    }

    /// The files of `list` read into the live state at the list's position (its positions below mapped first:
    /// lazy.ensure, a collective every rank runs here). True when every file read back whole and checked.
    pub fn load(s: *Store, list: []const u8) !bool {
        s.joinCkpt(); // xr: the stages are one copy's at a time
        const got = try lay.decode(list);
        const at = got.head.at;
        s.load_mapped = true;
        const t0 = nowNs();
        if (s.m.lazy) |lz| lz.ensure(s.m, at) catch |err| {
            std.log.warn("inkling2 rank {d}: a state of {d} tokens cannot be read back: its context does not map ({t})", .{ s.rank, at, err });
            s.load_mapped = false;
            return false;
        };
        var items: std.ArrayList(Item) = .empty;
        defer items.deinit(s.gpa);
        for (got.entries) |e| {
            const op = std.enums.fromInt(lay.Op, e.op) orelse return error.BadRowsList;
            switch (op) {
                .drop_block, .drop_state => if (s.rank != 0) unlinkStem(s.dir, e.name[0..16].*, extOf(if (op == .drop_block) .block else .state)),
                .block, .state => try items.append(s.gpa, .{ .op = op, .window = e.window, .k = e.k, .name = e.name }),
                else => return error.BadRowsList,
            }
        }
        // the copies write the caches from other threads: nothing on the model's stream may still touch them
        try s.m.s.synchronize();
        const t1 = nowNs();
        var w: Work = .{ .s = s, .src = .live, .at = at, .items = items.items, .save = false, .meta = &.{} };
        const ok = w.run();
        if (!ok) s.stats.failed += 1;
        const bytes = w.bytes.load(.monotonic);
        s.stats.read += bytes;
        if (ok) s.stats.loads += 1;
        const map_ms = @as(f64, @floatFromInt(t1 - t0)) / 1e6;
        const read_ms = @as(f64, @floatFromInt(nowNs() - t1)) / 1e6;
        if (ok) std.log.info("inkling2 rank {d}: read back {d} tokens: context mapped in {d:.0} ms, {d} files, {d} MiB in {d:.0} ms ({d:.2} GB/s)", .{ s.rank, at, map_ms, items.items.len, bytes >> 20, read_ms, @as(f64, @floatFromInt(bytes)) / 1e6 / @max(read_ms, 0.001) });
        return ok;
    }

    // ---- xr: checkpoints copied beside the prefill (rows_ckpt), committed between requests (rows_commit)

    /// Every rank, at rows_ckpt (the chunk that ends at the list's position just queued): the parts the next chunk
    /// overwrites (the convolution rings; the draft's sliding windows and f_rnn) copied to the stage on the model's
    /// stream, an event after them, then the copy-out on a thread of its own; the next chunk overlaps it. Its live
    /// window reads go first; engine.zig's run lets one prompt chunk pass before they are done (the rings keep window
    /// + chunk rows), any later chunk or verify waits for them (the gate). Nothing waits on the model's stream.
    pub fn startCkpt(s: *Store, list: []const u8) !void {
        s.joinCkpt(); // the stage is one copy's
        const got = try lay.decode(list);
        const at = got.head.at;
        if (s.m.state.len != at) return error.RowsSourceMoved;
        const stage = s.ckpt_stage orelse return error.RowsNoStage;
        if (s.rank == 0 and s.pending == null) return error.RowsNoPlan;
        var items: std.ArrayList(Item) = .empty;
        errdefer items.deinit(s.gpa);
        var readers: u32 = 0;
        for (got.entries) |e| {
            const op = std.enums.fromInt(lay.Op, e.op) orelse return error.BadRowsList;
            switch (op) {
                .drop_block, .drop_state => if (s.rank != 0) unlinkStem(s.dir, e.name[0..16].*, extOf(if (op == .drop_block) .block else .state)),
                .block, .state => {
                    try items.append(s.gpa, .{ .op = op, .window = e.window, .k = e.k, .name = e.name });
                    if (op == .state or e.window > 0) readers += 1;
                },
                else => return error.BadRowsList,
            }
        }
        if (got.head.flags & lay.flag_wipe != 0 and s.rank != 0) s.wipeOthers();
        // the stage, on the model's stream after the chunk that made the state (stream order: no wait here)
        var off: u64 = 0;
        for (s.m.state.rings_all) |b| {
            try stage.copyFrom(off, b.ptr, b.len, s.m.s.handle);
            off += b.len;
        }
        if (s.draft) |d| try d.saveContext(at, .{ .d = stage.d, .ptr = stage.ptr + s.m.snapRingBytes(), .len = d.snapBytes(), .borrowed = true });
        try s.ckpt_event.?.record(s.m.s);
        s.ckpt_run = .{ .items = try items.toOwnedSlice(s.gpa), .at = at, .tokens = if (s.rank == 0) s.pending.?.tokens else &.{} };
        s.gate.open(readers);
        s.ckpt_thread = std.Thread.spawn(.{}, ckptMain, .{s}) catch |err| {
            s.gate.done(true);
            s.gpa.free(s.ckpt_run.items);
            s.ckpt_run = .{};
            return err;
        };
    }

    /// The copy's thread: the stage's copies landed (the event), then the files (Work: the window readers first).
    fn ckptMain(s: *Store) void {
        const run = &s.ckpt_run;
        run.ok = false;
        defer s.gate.done(true); // nothing reads the live windows past here, whatever happened
        s.ctx.makeCurrent() catch return;
        s.ckpt_event.?.synchronize() catch return;
        const meta_counts: [3]u32 = .{ run.at, 0, 0 };
        const meta: [2][]const u8 = .{ std.mem.asBytes(&meta_counts), std.mem.sliceAsBytes(run.tokens) };
        var w: Work = .{ .s = s, .src = .ckpt, .at = run.at, .items = run.items, .save = true, .meta = if (s.rank == 0) &meta else &.{}, .gate = &s.gate };
        run.ok = w.run();
        run.bytes = w.bytes.load(.monotonic);
    }

    /// The running copy (if any) joined; its outcome kept for the commit (a failed one's files gone now).
    pub fn joinCkpt(s: *Store) void {
        const t = s.ckpt_thread orelse return;
        t.join();
        s.ckpt_thread = null;
        const run = s.ckpt_run;
        s.ckpt_run = .{};
        if (!run.ok) {
            for (run.items) |it| if (it.written) unlinkStem(s.dir, chain.stem(it.name), extOf(it.op));
            s.stats.failed += 1;
        }
        s.stats.written += run.bytes;
        s.ckpt_done.append(s.gpa, run) catch {
            // no room to keep it: its files go now, and the commit drops the rest (ckpt_failed)
            for (run.items) |it| if (it.written) unlinkStem(s.dir, chain.stem(it.name), extOf(it.op));
            s.gpa.free(run.items);
            s.ckpt_failed = true;
        };
    }

    /// Rank 0, after rows_ckpt ran here: the plan waits for the commit (its new blocks count as written meanwhile).
    pub fn queueCkpt(s: *Store) void {
        const p = s.pending orelse return;
        s.pending = null;
        s.ckpt_plans.append(s.gpa, p) catch {
            s.rows.?.admission.finish(s.rows.?.dir, p.bytes, false);
            s.freePending(p);
            s.ckpt_failed = true;
            return;
        };
        for (p.new) |wi| s.inflight.put(s.gpa, chain.stem(p.wants[wi].name), {}) catch {
            s.ckpt_failed = true;
        };
    }

    /// Rank 0: checkpoints sent since the last commit (a rows_commit is due).
    pub fn ckptPending(s: *const Store) bool {
        return s.ckpt_plans.items.len > 0 or s.ckpt_failed;
    }

    /// Every rank, at rows_commit: the copies joined; whether every one of this rank's ended whole.
    pub fn ckptOk(s: *Store) bool {
        s.joinCkpt();
        if (s.ckpt_failed) return false;
        for (s.ckpt_done.items) |d| if (!d.ok) return false;
        return true;
    }

    /// Every rank, once the ranks agreed (`all`): rank 0's plans join the index in order (else every rank's files
    /// of them go, and rank 0's reservations return).
    pub fn commitCkpts(s: *Store, all: bool) void {
        if (!all) {
            s.dropCkpts();
            return;
        }
        if (s.rank == 0) if (s.rows) |r| {
            if (s.ckpt_plans.items.len != s.ckpt_done.items.len) {
                std.log.warn("inkling2 rows: {d} checkpoint plans for {d} copies: none committed", .{ s.ckpt_plans.items.len, s.ckpt_done.items.len });
                s.dropCkpts();
                return;
            }
            for (s.ckpt_plans.items, s.ckpt_done.items) |p, d| s.commitOne(r, p, d);
            s.ckpt_plans.clearRetainingCapacity();
            s.inflight.clearRetainingCapacity();
        };
        for (s.ckpt_done.items) |d| s.gpa.free(d.items);
        s.ckpt_done.clearRetainingCapacity();
        s.ckpt_failed = false;
    }

    fn commitOne(s: *Store, r: *store_mod.Rows, p: Pending, d: CkptRun) void {
        defer s.freePending(p);
        // its files' bytes in the plan's order (the copy's list is in its own); every block it names on disk now
        var sizes: std.ArrayList(u64) = .empty;
        defer sizes.deinit(s.gpa);
        const whole = blk: {
            for (p.new) |wi| {
                const it = findItem(d.items, p.wants[wi].name) orelse break :blk false;
                sizes.append(s.gpa, it.bytes) catch break :blk false;
            }
            const st = findItem(d.items, p.name) orelse break :blk false;
            sizes.append(s.gpa, st.bytes) catch break :blk false;
            for (p.wants) |w| if (!r.blocks.contains(chain.stem(w.name)) and findItem(d.items, w.name) == null) break :blk false;
            break :blk true;
        };
        if (!whole) {
            std.log.warn("inkling2 rows: the checkpoint at {d} lacks a file: not committed", .{p.at});
            for (d.items) |it| if (it.written) unlinkStem(s.dir, chain.stem(it.name), extOf(it.op));
            for (d.items) |it| s.drop(if (it.op == .block) .drop_block else .drop_state, chain.stem(it.name)) catch {};
            r.admission.finish(r.dir, p.bytes, false);
            return;
        }
        publishInto(s, r, p, sizes.items) catch |err| {
            std.log.warn("inkling2 rows: the checkpoint at {d} could not join the index ({t})", .{ p.at, err });
            r.admission.finish(r.dir, p.bytes, true);
            return;
        };
        r.admission.finish(r.dir, p.bytes, true);
        s.stats.saves += 1;
        const ms = @as(f64, @floatFromInt(nowNs() - p.began)) / 1e6;
        std.log.info("inkling2 rows: checkpoint {d} committed: {d} new blocks, {d} MiB a rank, copied beside the prefill, {d:.0} ms from its plan ({d} states, {d} MiB of {d} GiB)", .{ p.at, p.new.len, p.bytes >> 20, ms, r.states.items.len, r.held() >> 20, s.set.cap / gib });
    }

    /// Every pending checkpoint dropped: this rank's files of them, rank 0's plans and reservations (the other ranks'
    /// files named for their next list).
    fn dropCkpts(s: *Store) void {
        for (s.ckpt_done.items) |d| {
            for (d.items) |it| if (it.written) unlinkStem(s.dir, chain.stem(it.name), extOf(it.op));
            if (s.rank == 0) for (d.items) |it| s.drop(if (it.op == .block) .drop_block else .drop_state, chain.stem(it.name)) catch {};
            s.gpa.free(d.items);
        }
        s.ckpt_done.clearRetainingCapacity();
        if (s.rows) |r| for (s.ckpt_plans.items) |p| {
            r.admission.finish(r.dir, p.bytes, false);
            s.freePending(p);
        };
        s.ckpt_plans.clearRetainingCapacity();
        s.inflight.clearRetainingCapacity();
        s.ckpt_failed = false;
    }

    // ---- xr: the trim hint (rank 0): context a recently used long state on disk was mapped for

    /// Rank 0, at a prompt's prefill: one more request.
    pub fn newRequest(s: *Store) void {
        s.requests += 1;
    }

    /// Rank 0, at a request's release, after its live end's save (or with it on disk already): a state past the lazy
    /// context's budget is noted with the positions mapped for it now (its save's), for the next requests' hint.
    pub fn noteLong(s: *Store, ids: []const u32) void {
        const r = s.rows orelse return;
        const lz = s.m.lazy orelse return;
        if (s.set.keep_requests == 0 or lz.mapped <= lz.budget) return;
        const name = chain.state(r.keying, ids, &.{}, &.{}, @intCast(ids.len));
        if (r.find(name) == null) return; // not on disk: nothing would come back from it
        s.recent.note(.{ .name = name, .mapped = @intCast(lz.mapped), .request = s.requests });
    }

    /// Rank 0, at a prompt that does not continue the live state: the most positions a long state left on disk by
    /// one of the last keep_requests requests was mapped at, if the store still holds it (0: none). The begin step
    /// carries it to every rank (engine.zig: e.trim_hint), whose residency's reclaim keeps that much mapped.
    pub fn trimHint(s: *const Store) u32 {
        const r = s.rows orelse return 0;
        return s.recent.hint(s.requests, s.set.keep_requests, r, held);
    }

    fn held(r: *store_mod.Rows, name: chain.Name) bool {
        return r.find(name) != null; // (evicted since: not held)
    }

    // ---- the sweep (every rank, at open): the other ranks keep only what rank 0's index names

    fn sweep(s: *Store) !void {
        const gpa = s.gpa;
        var es: std.ArrayList(lay.Entry) = .empty;
        defer es.deinit(gpa);
        if (s.rows) |r| {
            for (r.states.items) |st| try es.append(gpa, .{ .op = @intFromEnum(lay.Op.keep_state), .window = 0, .k = 0, .name = st.name });
            var it = r.blocks.keyIterator();
            while (it.next()) |k| {
                var e: lay.Entry = .{ .op = @intFromEnum(lay.Op.keep_block), .window = 0, .k = 0, .name = @splat(0) };
                @memcpy(e.name[0..16], k);
                try es.append(gpa, e);
            }
        }
        // the number of pieces first, then each piece (every rank the same exchanges)
        const per = (pay_cap - @sizeOf(lay.Head)) / @sizeOf(lay.Entry);
        var count = [4]u32{ @intCast((es.items.len + per - 1) / per), 0, 0, 0 };
        var cbuf: [16]u8 = undefined;
        try s.exchange(std.mem.asBytes(&count), &cbuf);
        count = std.mem.bytesToValue([4]u32, &cbuf);
        var keep: std.AutoHashMapUnmanaged(chain.Stem, void) = .empty;
        defer keep.deinit(gpa);
        var piece: u32 = 0;
        while (piece < count[0]) : (piece += 1) {
            // every rank learns the piece's length from rank 0 first
            const lo = @min(es.items.len, piece * per);
            const hi = @min(es.items.len, lo + per);
            var len = [4]u32{ @intCast(lay.listBytes(hi - lo)), 0, 0, 0 };
            var lbuf: [16]u8 = undefined;
            try s.exchange(std.mem.asBytes(&len), &lbuf);
            len = std.mem.bytesToValue([4]u32, &lbuf);
            try s.list.resize(gpa, len[0]);
            const tmp = try gpa.alloc(u8, len[0]);
            defer gpa.free(tmp);
            if (s.rank == 0) _ = lay.encode(.{ .at = 0, .count = 0, .flags = if (piece + 1 == count[0]) lay.flag_last else 0 }, es.items[lo..hi], s.list.items);
            try s.exchange(s.list.items[0..len[0]], tmp);
            if (s.rank == 0) continue;
            const got = try lay.decode(tmp);
            for (got.entries) |e| try keep.put(gpa, e.name[0..16].*, {});
        }
        if (s.rank == 0) return;
        // what rank 0's index lacks goes, and the other identities' directories (rank 0's eviction takes them first)
        var removed: usize = 0;
        const d = std.c.opendir(s.dir.ptr) orelse return;
        defer _ = std.c.closedir(d);
        while (std.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (name.len == 0 or name[0] == '.') continue;
            var buf: [1200]u8 = undefined;
            const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ s.dir, name }, 0) catch continue;
            const stem = parseStem(name);
            const kept = if (stem) |st| keep.contains(st) and !std.mem.endsWith(u8, name, ".part") else false;
            if (!kept) {
                _ = std.c.unlink(p);
                removed += 1;
            }
        }
        s.wipeOthers();
        std.log.info("inkling2 rank {d}/{d}: the sweep kept {d} files of rank 0's index, removed {d}", .{ s.rank, s.world, keep.count(), removed });
    }

    /// The other identities' directories under this rank's root removed (rank 0's eviction removed its own).
    fn wipeOthers(s: *Store) void {
        var cursor: ?u64 = null;
        var ids: std.ArrayList(u64) = .empty;
        defer ids.deinit(s.gpa);
        while (lanes.learned_dirs.next(s.root, s.dir, cursor) catch null) |id| {
            ids.append(s.gpa, id) catch break;
            cursor = id;
        }
        for (ids.items) |id| lanes.learned_dirs.removeOther(s.root, s.dir, id) catch {};
    }
};

/// xr: one checkpoint's copy beside the prefill: its files (with their bytes once written), position, ids (rank 0's
/// state file carries them: the plan's, alive until the commit), outcome.
const CkptRun = struct { items: []Item = &.{}, at: u32 = 0, tokens: []const u32 = &.{}, ok: bool = false, bytes: u64 = 0 };

fn findItem(items: []const Item, name: chain.Name) ?Item {
    for (items) |it| if (std.mem.eql(u8, &it.name, &name)) return it;
    return null;
}

// ---- the copies

const Work = struct {
    s: *Store,
    src: Source,
    at: u32,
    items: []Item,
    save: bool,
    meta: []const []const u8, // the state file's metadata (rank 0's saves)
    gate: ?*lay.WindowGate = null, // xr: a checkpoint's: told as each file reading the live windows is done
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
    first: std.atomic.Value(u16) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),

    /// Every item on the store's threads; true when every one succeeded.
    fn run(w: *Work) bool {
        const n = @min(w.s.set.threads, w.items.len);
        if (n == 0) return true;
        var threads: [32]?std.Thread = @splat(null);
        for (0..n) |i| threads[i] = std.Thread.spawn(.{}, worker, .{ w, i }) catch blk: {
            worker(w, i); // a thread that would not start: its share runs here
            break :blk null;
        };
        for (threads[0..n]) |t| if (t) |th| th.join();
        if (w.failed.load(.acquire)) {
            const code = w.first.load(.acquire);
            const err: anyerror = if (code != 0) @errorFromInt(code) else error.RowsCopyFailed;
            std.log.warn("inkling2 rank {d}: {s} {d} files of a state of {d} tokens failed ({t})", .{ w.s.rank, if (w.save) "writing" else "reading", w.items.len, w.at, err });
            return false;
        }
        return true;
    }

    fn fail(w: *Work, err: anyerror) void {
        _ = w.first.cmpxchgStrong(0, @intFromError(err), .acq_rel, .acquire);
        w.failed.store(true, .release);
    }

    fn worker(w: *Work, t: usize) void {
        w.s.ctx.makeCurrent() catch |err| return w.fail(err);
        // xr: copies on a stream of its own (non-blocking: no implicit order with any other stream, the model's
        // included), each stage-full synchronized before the stage is reused
        var st = cuda.Stream.init(w.s.m.d, true) catch |err| return w.fail(err);
        defer st.deinit();
        const hb = w.s.stages[t];
        const base = std.mem.alignForward(usize, @intFromPtr(hb.bytes.ptr), 4096);
        const stage: []align(4096) u8 = @as([*]align(4096) u8, @ptrFromInt(base))[0..w.s.set.stage];
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var lparts: std.ArrayList(lay.Part) = .empty;
        lay.parts(w.s.dims, w.at, &lparts, a) catch |err| return w.fail(err);
        var cparts: std.ArrayList(image.Part) = .empty;
        for (lparts.items) |p| cparts.append(a, .{ .bytes = p.bytes, .row = p.row, .window = p.window }) catch |err| return w.fail(err);
        var ranges: std.ArrayList(image.Range) = .empty;
        while (!w.failed.load(.acquire)) {
            const i = w.next.fetchAdd(1, .monotonic);
            if (i >= w.items.len) break;
            const it = &w.items[i];
            ranges.clearRetainingCapacity();
            const r = if (it.op == .block) image.blockRanges(cparts.items, lay.block_tokens, it.window, it.k, &ranges, a) else image.ownRanges(cparts.items, w.at, lay.block_tokens, &ranges, a);
            r catch |err| return w.fail(err);
            var c: Copier = .{ .w = w, .parts = lparts.items, .a = a, .stream = st };
            if (w.save) c.writeFile(it, ranges.items, stage) catch |err| return w.fail(err) else c.readFile(it, ranges.items, stage) catch |err| return w.fail(err);
            if (w.gate) |g| if (it.op == .state or it.window > 0) g.done(false);
        }
    }
};

/// One file's copies between the device and a stage: an image range through its parts to device addresses.
const Copier = struct {
    w: *Work,
    parts: []const lay.Part,
    a: Allocator,
    stream: cuda.Stream, // xr: the worker's
    words: [lay.words_bytes]u8 = undefined, // a read's words, checked once its file is whole
    words_seen: usize = 0,

    fn writeFile(c: *Copier, it: *Item, ranges: []const image.Range, stage: []u8) !void {
        const s = c.w.s;
        const ext = extOf(it.op);
        var wr = try file.Writer.open(s.dir, &chain.hex(it.name), ext);
        errdefer wr.abort();
        for (ranges) |g| {
            var off: u64 = 0;
            while (off < g.len) {
                const n: usize = @intCast(@min(g.len - off, stage.len));
                try c.copy(g.off + off, stage[0..n], false);
                try c.stream.synchronize(); // the stage holds them now
                try wr.append(stage[0..n]);
                off += n;
            }
        }
        const t: file.Trailer = if (it.op == .block)
            .{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = it.name, .block = lay.block_tokens }
        else
            .{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = it.name, .at = c.w.at, .block = lay.block_tokens };
        it.bytes = try wr.finish(t, if (it.op == .state) c.w.meta else &.{}, it.op == .state);
        it.written = true;
        _ = c.w.bytes.fetchAdd(it.bytes, .monotonic);
    }

    fn readFile(c: *Copier, it: *Item, ranges: []const image.Range, stage: []align(4096) u8) !void {
        const s = c.w.s;
        const ext = extOf(it.op);
        var pbuf: [1200]u8 = undefined;
        const p = try file.path(&pbuf, s.dir, &chain.hex(it.name), ext, false);
        var o = try file.open(p, if (it.op == .block) .block else .state, it.name);
        defer o.close();
        const n = o.trailer.payload;
        if (n != image.total(ranges) or o.trailer.block != lay.block_tokens or (it.op == .state and o.trailer.at != c.w.at)) return error.RowsRead;
        var h = std.hash.Wyhash.init(0);
        var done: u64 = 0;
        var ri: usize = 0; // the range the payload stands in, and how far into it
        var roff: u64 = 0;
        while (done < n) {
            // aligned reads from offset `done` (a multiple of the stage, so of 4096): the last may run into the
            // metadata or the trailer, of which only the payload's bytes count
            const span: usize = @intCast(@min(stage.len, std.mem.alignForward(u64, n - done, 4096)));
            var got: usize = 0;
            const want: usize = @intCast(@min(n - done, span));
            while (got < want) {
                const from = std.mem.alignBackward(usize, got, 4096);
                const r = std.c.pread(o.fd, stage.ptr + from, span - from, @intCast(done + from));
                if (r < 0 and std.c.errno(r) == .INTR) continue;
                if (r <= 0) return error.RowsRead;
                const next = from + @as(usize, @intCast(r));
                if (next <= got) return error.RowsRead;
                got = next;
            }
            const bytes = stage[0..want];
            h.update(bytes);
            // the payload is its ranges back to back: these bytes go where they stand
            var b: usize = 0;
            while (b < bytes.len) {
                const g = ranges[ri];
                const take: usize = @intCast(@min(g.len - roff, bytes.len - b));
                try c.copy(g.off + roff, bytes[b..][0..take], true);
                b += take;
                roff += take;
                if (roff == g.len) {
                    ri += 1;
                    roff = 0;
                }
            }
            try c.stream.synchronize(); // the stage's bytes landed: the next read may overwrite it
            done += want;
        }
        if (h.final() != o.trailer.payload_hash) return error.RowsHash;
        if (it.op == .state) try c.checkWords();
        _ = c.w.bytes.fetchAdd(o.trailer.payload + o.trailer.meta + @sizeOf(file.Trailer), .monotonic);
    }

    fn checkWords(c: *Copier) !void {
        if (c.words_seen != lay.words_bytes) return error.DiskWords;
        const got = std.mem.bytesToValue(lay.Words, &c.words);
        const want = c.wordsNow();
        if (got.magic != want.magic or got.version != want.version or got.at != want.at or got.world != want.world or got.layers != want.layers or got.kv_bytes != want.kv_bytes or got.draft != want.draft) return error.DiskWords;
    }

    fn wordsNow(c: *const Copier) lay.Words {
        const s = c.w.s;
        return .{ .at = c.w.at, .world = @intCast(s.world), .rank = @intCast(s.rank), .layers = @intCast(s.local.len), .kv_bytes = s.dims.kv_bytes, .draft = @intFromBool(s.draft != null) };
    }

    /// Image bytes [off, off + host.len) to `host` (a save) or from it (a read, into the live state).
    fn copy(c: *Copier, off: u64, host: []u8, to_device: bool) !void {
        var pieces: std.ArrayList(lay.Piece) = .empty;
        defer pieces.deinit(c.a);
        try lay.locate(c.parts, off, host.len, &pieces, c.a);
        var segs: std.ArrayList(lay.Segment) = .empty;
        defer segs.deinit(c.a);
        var h: usize = 0;
        for (pieces.items) |pc_| {
            const p = c.parts[pc_.part];
            const dst = host[h..][0..@intCast(pc_.len)];
            h += dst.len;
            if (p.kind == .words) {
                if (to_device) {
                    @memcpy(c.words[@intCast(pc_.off)..][0..dst.len], dst);
                    c.words_seen += dst.len;
                } else {
                    const wd = c.wordsNow();
                    @memcpy(dst, std.mem.asBytes(&wd)[@intCast(pc_.off)..][0..dst.len]);
                }
                continue;
            }
            // the part's own bytes, or runs of rows consecutive both in the image and on the device (rows_layout.zig's
            // segments: host-tested against live rings, positional rows and snapshot windows)
            segs.clearRetainingCapacity();
            try lay.segments(p, c.w.at, pc_.off, pc_.len, try c.place(p), &segs, c.a);
            for (segs.items) |g| try c.move(g.dev, dst[@intCast(g.host)..][0..@intCast(g.len)], to_device);
        }
    }

    fn move(c: *Copier, dev: u64, host: []u8, to_device: bool) !void {
        const d = c.w.s.m.d;
        if (host.len == 0) return;
        // xr: queued on the worker's stream (the stage is pinned); the caller synchronizes a stage-full
        if (to_device) {
            try d.check(d.api.cuMemcpyHtoDAsync_v2(dev, host.ptr, host.len, c.stream.handle), "cuMemcpyHtoDAsync");
        } else try d.check(d.api.cuMemcpyDtoHAsync_v2(host.ptr, dev, host.len, c.stream.handle), "cuMemcpyDtoHAsync");
    }

    /// Where part `p` lives for this copy's source: its device base, the ring it sits in (0: positional), and the
    /// first position a snapshot slot's window rows begin at (subtracted from a run's slot).
    fn place(c: *const Copier, p: lay.Part) !lay.Place {
        const s = c.w.s;
        const m = s.m;
        const at = c.w.at;
        // where the parts a snapshot keeps come from: a slot's buffers, or (xr) a checkpoint's stage (its rings and
        // the draft's part; its sliding windows the live rings'), else the live state's
        const Bases = struct { rings: ?u64 = null, window: ?u64 = null, ctx: ?u64 = null };
        const bases: Bases = switch (c.w.src) {
            .live => .{},
            .slot => |i| .{ .rings = s.slots.rings[i].ptr, .window = s.slots.window[i].ptr, .ctx = s.slots.ctx[i].ptr },
            .ckpt => blk: {
                const st = s.ckpt_stage orelse return error.RowsNoStage;
                break :blk .{ .rings = st.ptr, .ctx = st.ptr + m.snapRingBytes() };
            },
        };
        const rings_base = bases.rings;
        const window_base = bases.window;
        const ctx_base = bases.ctx;
        switch (p.kind) {
            .words => unreachable,
            .conv => return .{ .base = if (rings_base) |b| b + s.ring_off[p.index] else m.state.rings_all[p.index].ptr },
            .frnn => {
                const d = s.draft orelse return error.RowsNoDraft;
                if (ctx_base) |b| {
                    var nl: u64 = 0;
                    for (s.dlocal) |l| nl += @intFromBool(l);
                    return .{ .base = b + 2 * nl * ink.dspark.window * d.kvwl * 2 };
                }
                return .{ .base = d.frnn.ptr };
            },
            .gk => return .{ .base = m.state.kc[p.index].ptr },
            .gv => return .{ .base = m.state.vc[p.index].ptr },
            .dgk, .dgv => {
                const d = s.draft orelse return error.RowsNoDraft;
                const l = &d.layer[p.index];
                return .{ .base = if (p.kind == .dgk) l.kc.ptr else l.vc.ptr };
            },
            .lk, .lv => {
                const kv: u64 = @intFromBool(p.kind == .lv);
                if (window_base) |b| {
                    const j: u64 = s.ordinal[p.index];
                    return .{ .base = b + (2 * j + kv) * m.c.window * p.row, .lo = at - @min(at, m.c.window) };
                }
                return .{ .base = if (kv == 1) m.state.vc[p.index].ptr else m.state.kc[p.index].ptr, .slots = m.state.slots[p.index] };
            },
            .dlk, .dlv => {
                const d = s.draft orelse return error.RowsNoDraft;
                const kv: u64 = @intFromBool(p.kind == .dlv);
                const l = &d.layer[p.index];
                if (ctx_base) |b| {
                    const j: u64 = s.dordinal[p.index];
                    return .{ .base = b + (2 * j + kv) * ink.dspark.window * p.row, .lo = at - @min(at, ink.dspark.window) };
                }
                return .{ .base = if (kv == 1) l.vc.ptr else l.kc.ptr, .slots = l.slots };
            },
        }
    }
};

// ---- the identity: what a state on disk is read under

/// The build (this executable's bytes), the checkpoint view's small files, every TF_INKLING2_* switch but this
/// store's own, the split and the layout: a change of any one starts cold (the store's eviction takes the old first).
fn identityOf(m: *const Model, draft: ?*const Draft, dir: []const u8, world: usize) !u64 {
    var h = std.hash.Wyhash.init(0x696e6b32726f7773);
    h.update("inkling2 rows v1");
    try hashFile(&h, "/proc/self/exe");
    var buf: [1200]u8 = undefined;
    for ([_][]const u8{ "config.json", "checkpoint.json", "model.safetensors.index.json", "dense-exl3/manifest.json", "native/config.json" }) |name| {
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0) catch continue;
        h.update(name);
        hashFile(&h, p) catch {};
    }
    try hashEnv(&h);
    const dims = [_]u64{ world, m.c.layers, m.hkv(), m.c.head_dim, m.c.hidden, m.c.window, m.c.conv, @intFromBool(draft != null), if (draft) |d| d.kvwl else 0, lay.block_tokens };
    h.update(std.mem.sliceAsBytes(&dims));
    const id = h.final();
    return if (id == 0) 1 else id;
}

fn hashFile(h: *std.hash.Wyhash, p: [:0]const u8) !void {
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var buf: [1 << 16]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n < 0 and std.c.errno(n) == .INTR) continue;
        if (n < 0) return error.ReadFailed;
        if (n == 0) return;
        h.update(buf[0..@intCast(n)]);
    }
}

/// TF_INKLING2_* from /proc/self/environ, sorted, but this store's own settings and the debug ones.
fn hashEnv(h: *std.hash.Wyhash) !void {
    const fd = std.c.open("/proc/self/environ", .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    var buf: [1 << 17]u8 = undefined;
    var len: usize = 0;
    while (len < buf.len) {
        const n = std.c.read(fd, buf[len..].ptr, buf.len - len);
        if (n < 0 and std.c.errno(n) == .INTR) continue;
        if (n <= 0) break;
        len += @intCast(n);
    }
    var vars: [256][]const u8 = undefined;
    var k: usize = 0;
    var it = std.mem.splitScalar(u8, buf[0..len], 0);
    while (it.next()) |v| {
        if (!std.mem.startsWith(u8, v, "TF_INKLING2_")) continue;
        if (std.mem.startsWith(u8, v, "TF_INKLING2_ROWS_") or std.mem.startsWith(u8, v, "TF_INKLING2_PROMPT_DISK=") or std.mem.startsWith(u8, v, "TF_INKLING2_DUMP_") or std.mem.startsWith(u8, v, "TF_INKLING2_ROUNDS")) continue;
        if (k < vars.len) {
            vars[k] = v;
            k += 1;
        }
    }
    std.mem.sort([]const u8, vars[0..k], {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (vars[0..k]) |v| {
        h.update(v);
        h.update("\x00");
    }
}

// ---- small file helpers

fn makePath(dir: [:0]const u8) !void {
    var i: usize = 1;
    var buf: [1200]u8 = undefined;
    if (dir.len >= buf.len) return error.NameTooLong;
    while (i <= dir.len) : (i += 1) {
        if (i < dir.len and dir[i] != '/') continue;
        @memcpy(buf[0..i], dir[0..i]);
        buf[i] = 0;
        const p: [*:0]const u8 = @ptrCast(&buf);
        const rc = std.c.mkdir(p, 0o700);
        if (rc != 0 and std.c.errno(rc) != .EXIST) return error.RowsDir;
    }
}

/// A file's extension: a block's, or a state's own file.
fn extOf(op: lay.Op) []const u8 {
    return if (op == .block) ".blk" else ".st";
}

fn unlinkStem(dir: []const u8, stem: chain.Stem, ext: []const u8) void {
    var buf: [1200]u8 = undefined;
    const p = file.path(&buf, dir, &chain.hex(stem), ext, false) catch return;
    _ = std.c.unlink(p);
}

fn parseStem(name: []const u8) ?chain.Stem {
    if (name.len < 32) return null;
    var out: chain.Stem = undefined;
    _ = std.fmt.hexToBytes(&out, name[0..32]) catch return null;
    return out;
}

fn nowNs() i128 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t);
    return @as(i128, t.sec) * std.time.ns_per_s + t.nsec;
}
