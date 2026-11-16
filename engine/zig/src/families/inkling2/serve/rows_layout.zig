//! xk-rows: one TP rank's Inkling-2 sequence state as the core's rows store keeps it (core/rows_image.zig's parts),
//! every part a pass resuming at position `at` needs, in a fixed order:
//!   own (the state's own file): the words (what the state is), the four convolution rings (the last K - 1 inputs
//!     before `at` of every layer), the draft head's f_rnn;
//!   rows from position 0 (blocks of 256 positions, shared by the states of one lineage): each global layer's K, then
//!     V, then the draft head's global layer's;
//!   rings of the last `window` positions (blocks inside the window shared the same way): each sliding layer's K, then
//!     V, then the draft head's sliding layers'.
//! A row is one position's keys (or values) of this rank's kv heads. The live state keeps them elsewhere: a global
//! layer's row of position p at slot p, a sliding layer's at p % its slots, the draft head's sliding layers' at p %
//! theirs; a snapshot slot (serve/reuse.zig) keeps a window as rows 0 .. n - 1 of positions at - n .. at - 1. This
//! file maps between the image and those places (serve/rows.zig copies), and carries the two things the store's names
//! need beyond tokens: the lineage salt and the steps' file lists.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The core store's block (core/rows_store.zig block_tokens).
pub const block_tokens = 256;

pub const Kind = enum(u8) { words, conv, frnn, gk, gv, dgk, dgv, lk, lv, dlk, dlv };

/// One part of the image: what it is, its bytes at the state's position, a row's bytes (0: the state's own), its
/// window (0: rows from position 0). `index`: a conv ring's kind, a model layer (gk .. lv) or a draft layer.
pub const Part = struct {
    kind: Kind,
    index: u16 = 0,
    bytes: u64,
    row: u32 = 0,
    window: u32 = 0,
};

/// What fixes a rank's layout: per target layer whether it slides, the conv rings' bytes, a K (or V) row's bytes,
/// the sliding window; the draft head's, when it is loaded.
pub const Dims = struct {
    local: []const bool,
    ring_bytes: [4]u64,
    kv_bytes: u32,
    window: u32,
    draft: ?Draft = null,

    pub const Draft = struct { local: []const bool, kv_bytes: u32, window: u32, frnn_bytes: u32 };
};

/// The words: a state's own description, checked when it is read back (`at`, the split, the layout).
pub const Words = extern struct {
    magic: u32 = 0x324b4e49, // "INK2"
    version: u32 = 1,
    at: u32,
    world: u32,
    rank: u32,
    layers: u32,
    kv_bytes: u32,
    draft: u32,
};

pub const words_bytes = @sizeOf(Words);

comptime {
    std.debug.assert(words_bytes == 32);
}

/// The image's parts at `at`, appended to `out`.
pub fn parts(d: Dims, at: u32, out: *std.ArrayList(Part), a: Allocator) !void {
    try out.append(a, .{ .kind = .words, .bytes = words_bytes });
    for (d.ring_bytes, 0..) |b, k| try out.append(a, .{ .kind = .conv, .index = @intCast(k), .bytes = b });
    if (d.draft) |dr| try out.append(a, .{ .kind = .frnn, .bytes = dr.frnn_bytes });
    const n: u64 = at;
    for (d.local, 0..) |l, li| if (!l) {
        try out.append(a, .{ .kind = .gk, .index = @intCast(li), .bytes = n * d.kv_bytes, .row = d.kv_bytes });
        try out.append(a, .{ .kind = .gv, .index = @intCast(li), .bytes = n * d.kv_bytes, .row = d.kv_bytes });
    };
    if (d.draft) |dr| for (dr.local, 0..) |l, i| if (!l) {
        try out.append(a, .{ .kind = .dgk, .index = @intCast(i), .bytes = n * dr.kv_bytes, .row = dr.kv_bytes });
        try out.append(a, .{ .kind = .dgv, .index = @intCast(i), .bytes = n * dr.kv_bytes, .row = dr.kv_bytes });
    };
    const w: u64 = @min(at, d.window);
    for (d.local, 0..) |l, li| if (l) {
        try out.append(a, .{ .kind = .lk, .index = @intCast(li), .bytes = w * d.kv_bytes, .row = d.kv_bytes, .window = d.window });
        try out.append(a, .{ .kind = .lv, .index = @intCast(li), .bytes = w * d.kv_bytes, .row = d.kv_bytes, .window = d.window });
    };
    if (d.draft) |dr| {
        const dw: u64 = @min(at, dr.window);
        for (dr.local, 0..) |l, i| if (l) {
            try out.append(a, .{ .kind = .dlk, .index = @intCast(i), .bytes = dw * dr.kv_bytes, .row = dr.kv_bytes, .window = dr.window });
            try out.append(a, .{ .kind = .dlv, .index = @intCast(i), .bytes = dw * dr.kv_bytes, .row = dr.kv_bytes, .window = dr.window });
        };
    }
}

/// The position image row `r` of a ring part of `window` positions holds at `at`: rows hold positions at - n ..
/// at - 1 (n = min(at, window)), position p at row p % window.
pub fn ringPosition(at: u32, window: u32, r: u32) u32 {
    const n = @min(at, window);
    if (n < window) return r; // at < window: positions 0 .. at - 1 at rows 0 ..
    const lo = at - n;
    return lo + (r + window - lo % window) % window;
}

/// The position of row `r` of a row part (window 0: rows from position 0; else the ring's).
pub fn rowPosition(p: Part, at: u32, r: u32) u32 {
    return if (p.window == 0) r else ringPosition(at, p.window, r);
}

/// Where a byte range of the image lies: part `part`, `off` bytes into it, `len` bytes (within the part).
pub const Piece = struct { part: usize, off: u64, len: u64 };

/// The image range [off, off + len) cut at part boundaries, into `out`.
pub fn locate(ps: []const Part, off: u64, len: u64, out: *std.ArrayList(Piece), a: Allocator) !void {
    var base: u64 = 0;
    var at = off;
    var left = len;
    for (ps, 0..) |p, i| {
        if (left == 0) return;
        const end = base + p.bytes;
        if (at < end) {
            const take = @min(left, end - at);
            try out.append(a, .{ .part = i, .off = at - base, .len = take });
            at += take;
            left -= take;
        }
        base = end;
    }
    if (left > 0) return error.RangePastImage;
}

/// A run of whole rows of a row part whose positions sit at consecutive slots of a ring of `slots` (0: positional,
/// slot = position): rows [r, r + n), the first at slot `slot`.
pub const Run = struct { r: u32, n: u32, slot: u64 };

/// The rows [r0, r1) of row part `p` at `at` as runs contiguous both in the image and among `slots` slots
/// (0: positional), into `out`. The image's own ring wraps at the window and the live one at `slots`: both cut runs.
pub fn runs(p: Part, at: u32, r0: u32, r1: u32, slots: u64, out: *std.ArrayList(Run), a: Allocator) !void {
    var r = r0;
    while (r < r1) {
        const pos = rowPosition(p, at, r);
        const slot: u64 = if (slots == 0) pos else pos % slots;
        var n: u32 = 1;
        while (r + n < r1) : (n += 1) {
            const q = rowPosition(p, at, r + n);
            if (q != pos + n) break; // the image ring wrapped
            if (slots != 0 and (slot + n) >= slots) break; // the live ring wrapped
        }
        try out.append(a, .{ .r = r, .n = n, .slot = slot });
        r += n;
    }
}

/// Where a part's rows sit for one copy: the device address of its first slot, the ring the live state keeps them in
/// (0: positional, slot = position), and the position a snapshot slot's first row holds (its window's at - n).
pub const Place = struct { base: u64, slots: u64 = 0, lo: u64 = 0 };

/// `len` bytes at device address `dev`, `host` bytes into the image range being copied.
pub const Segment = struct { dev: u64, host: u64, len: u64 };

/// The device segments of part `p`'s bytes [off, off + len) at `at` placed at `pl`, in image order, into `out`: one for
/// a state's own part, else a run of rows (whole or cut) at a time, cut where the image's or the live ring wraps.
pub fn segments(p: Part, at: u32, off: u64, len: u64, pl: Place, out: *std.ArrayList(Segment), a: Allocator) !void {
    if (len == 0) return;
    if (off + len > p.bytes) return error.RangePastPart;
    if (p.row == 0) return out.append(a, .{ .dev = pl.base + off, .host = 0, .len = len });
    const row: u64 = p.row;
    const r0: u32 = @intCast(off / row);
    const r1: u32 = @intCast((off + len + row - 1) / row);
    var rs: std.ArrayList(Run) = .empty;
    defer rs.deinit(a);
    try runs(p, at, r0, r1, pl.slots, &rs, a);
    for (rs.items) |run| {
        const lo = @max(off, @as(u64, run.r) * row);
        const hi = @min(off + len, @as(u64, run.r + run.n) * row);
        try out.append(a, .{ .dev = pl.base + (run.slot - pl.lo) * row + (lo - @as(u64, run.r) * row), .host = lo - off, .len = hi - lo });
    }
}

/// Where a snapshot slot (serve/reuse.zig: Model.saveState, Draft.saveContext) keeps position `pos` of a window of
/// `window` positions saved at `at`: row pos - (at - n), n = min(at, window).
pub fn slotRow(at: u32, window: u32, pos: u32) u32 {
    return pos - (at - @min(at, window));
}

// ---- the salt
//
// A row's bits are not a function of the tokens alone: they follow the chunking of the pass that made them and the
// experts resident at the time (the mixed store's bank adapts). The store shares a block between states whose tokens
// agree through it, so the tokens a state hands the store carry where each pass of its history began: a pass from
// position s puts its own 26-bit salt in bits 18 .. 30 of positions s and s + 1 (vocabulary ids are under 2^18; bit
// 31 stays clear: the core's image keys take it). Blocks wholly before a pass's start keep their names (their rows
// are the same); from it on they are the pass's own. So a state read back is the exact one it saved, and the live
// state still holds a stored state's rows exactly when the live salted tokens begin with the stored ones.

pub const vocab_bits = 18;
pub const salt_bits = 26;
pub const id_mask: u32 = (1 << vocab_bits) - 1;
const half: u32 = 1 << (salt_bits / 2);

/// Token `t` at offset `i` (0 or 1) from its pass's start, with the pass's `salt`.
pub fn salted(t: u32, salt: u32, i: usize) u32 {
    const part = if (i == 0) salt & (half - 1) else (salt >> (salt_bits / 2)) & (half - 1);
    return (t & id_mask) | (part << vocab_bits);
}

/// Whether stored (salted) tokens are `prompt`'s first ones, the salts set aside.
pub fn prefixOf(stored: []const u32, prompt: []const u32) bool {
    if (stored.len > prompt.len) return false;
    for (stored, prompt[0..stored.len]) |s, p| if (s & id_mask != p) return false;
    return true;
}

/// How many first tokens salted `live` and plain `prompt` share, the salts set aside.
pub fn sharedIds(live: []const u32, prompt: []const u32) usize {
    const n = @min(live.len, prompt.len);
    for (live[0..n], prompt[0..n], 0..) |l, p, i| if (l & id_mask != p) return i;
    return n;
}

/// A new pass's salt from `seed`: nonzero, salt_bits wide.
pub fn newSalt(seed: u64) u32 {
    var h = std.hash.Wyhash.init(0x73616c74);
    h.update(std.mem.asBytes(&seed));
    const s: u32 = @as(u32, @truncate(h.final())) & ((1 << salt_bits) - 1);
    return if (s == 0) 1 else s;
}

// ---- the steps' file lists (engine.zig: rows_save / rows_load / rows_sweep carry them past the step's header)

pub const Op = enum(u32) { block = 1, state = 2, drop_block = 3, drop_state = 4, keep_block = 5, keep_state = 6 };

/// One file of a list: what to do with it, its group and block index (a block), its 32-byte name.
pub const Entry = extern struct { op: u32, window: u32, k: u32, flags: u32 = 0, name: [32]u8 };

/// A list's head: its state's position, the entries that follow, flags (wipe: other identities' directories go;
/// last: the final piece of a sweep).
pub const Head = extern struct { magic: u32 = 0x4c53524b, at: u32, count: u32, flags: u32 = 0 };

pub const flag_wipe: u32 = 1;
pub const flag_last: u32 = 2;

comptime {
    std.debug.assert(@sizeOf(Entry) == 48 and @sizeOf(Head) == 16);
}

/// Bytes a list of `n` entries takes.
pub fn listBytes(n: usize) usize {
    return @sizeOf(Head) + n * @sizeOf(Entry);
}

/// The list as u32 words (its bytes are a multiple of four), into `out` (listBytes(entries.len) bytes).
pub fn encode(h: Head, entries: []const Entry, out: []u8) []u8 {
    var hh = h;
    hh.count = @intCast(entries.len);
    @memcpy(out[0..@sizeOf(Head)], std.mem.asBytes(&hh));
    @memcpy(out[@sizeOf(Head)..][0 .. entries.len * @sizeOf(Entry)], std.mem.sliceAsBytes(entries));
    return out[0..listBytes(entries.len)];
}

/// A list read back: its head and entries (views into `bytes`).
pub fn decode(bytes: []const u8) !struct { head: Head, entries: []align(1) const Entry } {
    if (bytes.len < @sizeOf(Head)) return error.BadRowsList;
    const h = std.mem.bytesToValue(Head, bytes[0..@sizeOf(Head)]);
    if (h.magic != (Head{ .at = 0, .count = 0 }).magic or bytes.len != listBytes(h.count)) return error.BadRowsList;
    return .{ .head = h, .entries = std.mem.bytesAsSlice(Entry, bytes[@sizeOf(Head)..]) };
}

// ---- xr: checkpoints written beside the prefill (rows.zig's checkpoint jobs, engine.zig's gate)

/// A checkpoint's files in the order its copy takes them: the window blocks (window > 0), then the state's own file
/// (both read the live sliding rings, which a later chunk overwrites), then the blocks of rows from position 0
/// (positional: they hold until a pass starts below the checkpoint). Stable within each kind; deletions stay first.
/// Returns how many files read the live rings (the gate's count).
pub fn ckptOrder(es: []Entry, a: Allocator) !u32 {
    const tmp = try a.dupe(Entry, es);
    defer a.free(tmp);
    var n: usize = 0;
    var readers: u32 = 0;
    // deletions, window blocks, the state, blocks from 0: four passes over the copy
    for ([_]u8{ 0, 1, 2, 3 }) |pass| for (tmp) |e| {
        const kind: u8 = if (e.op == @intFromEnum(Op.block)) (if (e.window > 0) 1 else 3) else if (e.op == @intFromEnum(Op.state)) 2 else 0;
        if (kind != pass) continue;
        es[n] = e;
        n += 1;
        if (kind == 1 or kind == 2) readers += 1;
    };
    return readers;
}

/// The live sliding windows a checkpoint's copy reads: the next prompt chunk may run beside it (a ring keeps window +
/// chunk rows: the next chunk leaves the window's slots alone), anything after that waits for those files.
pub const WindowGate = struct {
    left: std.atomic.Value(u32) = .init(0), // files of the running copy that read the live rings, not done yet
    allow: u32 = 0, // prompt chunks that may still run before they are (the main thread's)

    /// A copy starts: `readers` files read the live rings; one prompt chunk may run first.
    pub fn open(g: *WindowGate, readers: u32) void {
        g.allow = 1;
        g.left.store(readers, .release);
    }

    /// The copy's thread: one of those files is done (or, `all`, the copy ended: failed or not, nothing reads them now).
    pub fn done(g: *WindowGate, all: bool) void {
        if (all) {
            g.left.store(0, .release);
            return;
        }
        var v = g.left.load(.acquire);
        while (v > 0) v = g.left.cmpxchgWeak(v, v - 1, .acq_rel, .acquire) orelse return;
    }

    /// The main thread, before a step that writes the caches (`chunk`: a prompt chunk): the ns it waited.
    pub fn before(g: *WindowGate, chunk: bool) u64 {
        if (g.left.load(.acquire) == 0) return 0;
        if (chunk and g.allow > 0) {
            g.allow -= 1;
            return 0;
        }
        const t0 = nowNs();
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 50_000 };
        while (g.left.load(.acquire) != 0) _ = std.c.nanosleep(&ts, null);
        return @intCast(nowNs() - t0);
    }
};

/// xr: the trim hint's evidence (rows.zig, rank 0): the long states the last requests left on disk, the newest last.
/// A state noted again replaces its note; past `cap` notes the oldest goes.
pub const Recents = struct {
    pub const cap = 8;
    pub const Note = struct { name: [32]u8, mapped: u32, request: u64 };

    notes: [cap]Note = undefined,
    n: usize = 0,

    pub fn note(rs: *Recents, e: Note) void {
        for (rs.notes[0..rs.n]) |*x| if (std.mem.eql(u8, &x.name, &e.name)) {
            x.* = e;
            return;
        };
        if (rs.n == cap) {
            std.mem.copyForwards(Note, rs.notes[0 .. cap - 1], rs.notes[1..]);
            rs.n -= 1;
        }
        rs.notes[rs.n] = e;
        rs.n += 1;
    }

    /// The most positions a note from the last `k` requests (of `requests` so far) was mapped at, its state still
    /// held (`held`); 0 for none (k 0: no hint).
    pub fn hint(rs: *const Recents, requests: u64, k: usize, ctx: anytype, comptime held: fn (@TypeOf(ctx), [32]u8) bool) u32 {
        if (k == 0) return 0;
        var most: u32 = 0;
        for (rs.notes[0..rs.n]) |x| {
            if (requests -| x.request > k) continue;
            if (!held(ctx, x.name)) continue;
            most = @max(most, x.mapped);
        }
        return most;
    }
};

fn nowNs() i128 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t);
    return @as(i128, t.sec) * std.time.ns_per_s + t.nsec;
}

// ---- tests

fn testDims(local: []const bool, dlocal: []const bool) Dims {
    return .{ .local = local, .ring_bytes = .{ 24, 24, 96, 96 }, .kv_bytes = 8, .window = 16, .draft = .{ .local = dlocal, .kv_bytes = 4, .window = 16, .frnn_bytes = 12 } };
}

test "the parts: own first, rows from 0 next, rings last; the core's validity rule holds at every position" {
    const a = std.testing.allocator;
    const d = testDims(&.{ true, false, true, true, false }, &.{ true, false });
    for ([_]u32{ 1, 7, 15, 16, 17, 255, 256, 257, 1000 }) |at| {
        var ps: std.ArrayList(Part) = .empty;
        defer ps.deinit(a);
        try parts(d, at, &ps, a);
        // words, 4 rings, f_rnn; 2 global layers x 2; 1 draft global x 2; 3 local x 2; 1 draft local x 2
        try std.testing.expectEqual(@as(usize, 6 + 4 + 2 + 6 + 2), ps.items.len);
        for (ps.items) |p| {
            if (p.row == 0) continue;
            const rows = p.bytes / p.row;
            if (p.window == 0) try std.testing.expectEqual(@as(u64, at), rows) else try std.testing.expectEqual(@as(u64, @min(at, p.window)), rows);
        }
        try std.testing.expect(ps.items[0].kind == .words and ps.items[5].kind == .frnn and ps.items[6].kind == .gk and ps.items[6].index == 1);
    }
}

test "a ring row's position: the one of the last `window` positions sitting at that row" {
    // at < window: rows are positions
    try std.testing.expectEqual(@as(u32, 5), ringPosition(10, 16, 5));
    // at = 2000, window 1024: positions 976 .. 1999, p at row p % 1024
    try std.testing.expectEqual(@as(u32, 976), ringPosition(2000, 1024, 976));
    try std.testing.expectEqual(@as(u32, 1024), ringPosition(2000, 1024, 0));
    try std.testing.expectEqual(@as(u32, 1999), ringPosition(2000, 1024, 975));
    // every row of every position: a permutation of the window, each position at row p % window
    for ([_]u32{ 1024, 1025, 2047, 2048, 5000 }) |at| {
        var seen: [1024]bool = @splat(false);
        for (0..1024) |r| {
            const p = ringPosition(at, 1024, @intCast(r));
            try std.testing.expect(p >= at - 1024 and p < at and p % 1024 == r);
            try std.testing.expect(!seen[p - (at - 1024)]);
            seen[p - (at - 1024)] = true;
        }
    }
}

test "runs cut where the image's ring or the live ring wraps, and cover every row once in order" {
    const a = std.testing.allocator;
    const p: Part = .{ .kind = .lk, .bytes = 16 * 8, .row = 8, .window = 16 };
    for ([_]u32{ 9, 16, 17, 30, 31, 32, 33, 100 }) |at| for ([_]u64{ 0, 21, 23, 40 }) |slots| {
        var list: std.ArrayList(Run) = .empty;
        defer list.deinit(a);
        const rows: u32 = @min(at, 16);
        try runs(p, at, 0, rows, slots, &list, a);
        var r: u32 = 0;
        for (list.items) |x| {
            try std.testing.expectEqual(r, x.r);
            for (0..x.n) |j| {
                const pos = ringPosition(at, 16, x.r + @as(u32, @intCast(j)));
                const want: u64 = if (slots == 0) pos else pos % slots;
                try std.testing.expectEqual(want, x.slot + j); // contiguous in the live ring too
            }
            r += x.n;
        }
        try std.testing.expectEqual(rows, r);
    };
}

test "locate cuts a range at part boundaries" {
    const a = std.testing.allocator;
    const ps = [_]Part{ .{ .kind = .words, .bytes = 32 }, .{ .kind = .conv, .bytes = 100 }, .{ .kind = .gk, .bytes = 64, .row = 8 } };
    var out: std.ArrayList(Piece) = .empty;
    defer out.deinit(a);
    try locate(&ps, 20, 150, &out, a);
    try std.testing.expectEqualSlices(Piece, &.{ .{ .part = 0, .off = 20, .len = 12 }, .{ .part = 1, .off = 0, .len = 100 }, .{ .part = 2, .off = 0, .len = 38 } }, out.items);
    out.clearRetainingCapacity();
    try std.testing.expectError(error.RangePastImage, locate(&ps, 190, 20, &out, a));
}

/// A test device: the live state's buffers (global rows positional, sliding rings of `slots`), a snapshot slot's
/// (window rows from at - n), and each part's place in it; every row filled with its position's own bytes.
const TestDev = struct {
    mem: []u8,
    places: []Place,

    fn fill(row: []u8, part: usize, pos: u32) void {
        for (row, 0..) |*b, i| b.* = @truncate((part *% 131) ^ (@as(usize, pos) *% 7) ^ (i *% 29) ^ (@as(usize, pos) >> 3));
    }

    /// Part `pi` of `ps` at `at`: `slots` its live ring (0: positional), `snap`: kept as a snapshot slot keeps it.
    fn make(a: Allocator, ps: []const Part, at: u32, slots: u64, snap: bool) !TestDev {
        var d: TestDev = .{ .mem = &.{}, .places = try a.alloc(Place, ps.len) };
        var size: u64 = 0;
        for (ps, 0..) |p, i| {
            const rows: u64 = if (p.row == 0) 1 else if (p.window == 0) at + 7 else if (snap) p.window else slots;
            const bytes = if (p.row == 0) p.bytes else rows * p.row;
            d.places[i] = .{ .base = size + 64 };
            if (p.row > 0 and p.window > 0) {
                if (snap) d.places[i].lo = at - @min(at, p.window) else d.places[i].slots = slots;
            }
            size += bytes + 64 * 3;
        }
        d.mem = try a.alloc(u8, size);
        @memset(d.mem, 0xee);
        return d;
    }

    /// The device's row of position `pos` of part `pi` (its own bytes for a part without rows).
    fn rowAt(d: *const TestDev, p: Part, pi: usize, pos: u32) []u8 {
        const pl = d.places[pi];
        if (p.row == 0) return d.mem[@intCast(pl.base)..][0..@intCast(p.bytes)];
        const slot: u64 = if (pl.slots == 0) pos else pos % pl.slots;
        return d.mem[@intCast(pl.base + (slot - pl.lo) * p.row)..][0..p.row];
    }

    /// Every position a state at `at` holds (rows from 0; the last window of a ring) filled with its own bytes.
    fn fillState(d: *TestDev, ps: []const Part, at: u32) void {
        for (ps, 0..) |p, pi| {
            if (p.row == 0) {
                fill(d.rowAt(p, pi, 0), pi, 0);
                continue;
            }
            const lo = if (p.window == 0) 0 else at - @min(at, p.window);
            var pos = lo;
            while (pos < at) : (pos += 1) fill(d.rowAt(p, pi, pos), pi, pos);
        }
    }

    /// Whether every position of the state at `at` holds its own bytes.
    fn holdsState(d: *const TestDev, ps: []const Part, at: u32) bool {
        var want: [64]u8 = undefined;
        for (ps, 0..) |p, pi| {
            const lo: u32 = if (p.row == 0) 0 else if (p.window == 0) 0 else at - @min(at, p.window);
            const hi: u32 = if (p.row == 0) 1 else at;
            var pos = lo;
            while (pos < hi) : (pos += 1) {
                const got = d.rowAt(p, pi, pos);
                fill(want[0..got.len], pi, pos);
                if (!std.mem.eql(u8, got, want[0..got.len])) return false;
            }
        }
        return true;
    }

    /// Image bytes [off, off + host.len) between the device and `host` (to_device: into the device), as rows.zig's
    /// copies cut them: at part boundaries (locate), then into segments.
    fn copy(d: *TestDev, ps: []const Part, at: u32, off: u64, host: []u8, to_device: bool, a: Allocator) !void {
        var pieces: std.ArrayList(Piece) = .empty;
        defer pieces.deinit(a);
        try locate(ps, off, host.len, &pieces, a);
        var segs: std.ArrayList(Segment) = .empty;
        defer segs.deinit(a);
        var h: usize = 0;
        for (pieces.items) |pc| {
            const dst = host[h..][0..@intCast(pc.len)];
            h += dst.len;
            segs.clearRetainingCapacity();
            try segments(ps[pc.part], at, pc.off, pc.len, d.places[pc.part], &segs, a);
            var covered: u64 = 0;
            for (segs.items) |g| {
                try std.testing.expectEqual(covered, g.host); // segments tile the piece in order
                covered += g.len;
                const dev = d.mem[@intCast(g.dev)..][0..@intCast(g.len)];
                const hb = dst[@intCast(g.host)..][0..@intCast(g.len)];
                if (to_device) @memcpy(dev, hb) else @memcpy(hb, dev);
            }
            try std.testing.expectEqual(pc.len, covered);
        }
    }
};

test "segments: a state through any cut of its image moves between live rings, positional rows and snapshot windows" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d: Dims = .{ .local = &.{ true, false, true }, .ring_bytes = .{ 12, 20, 8, 16 }, .kv_bytes = 8, .window = 16, .draft = .{ .local = &.{ true, false }, .kv_bytes = 4, .window = 16, .frnn_bytes = 6 } };
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for ([_]u32{ 5, 16, 17, 31, 33, 40, 77 }) |at| {
        var list: std.ArrayList(Part) = .empty;
        try parts(d, at, &list, a);
        const ps = list.items;
        var total: u64 = 0;
        for (ps) |p| total += p.bytes;
        // the source: the live state (rings of 16 + 9 slots), or a snapshot slot; the image read in cuts of any size
        for ([_]bool{ false, true }) |snap| {
            var src = try TestDev.make(a, ps, at, 25, snap);
            src.fillState(ps, at);
            const img = try a.alloc(u8, @intCast(total));
            var off: u64 = 0;
            while (off < total) {
                const n = @min(total - off, 1 + rnd.uintLessThan(u64, 41));
                try src.copy(ps, at, off, img[@intCast(off)..][0..@intCast(n)], false, a);
                off += n;
            }
            // the image's rows: a ring's row r the position ringPosition gives, rows from 0 in order
            var want: [64]u8 = undefined;
            var base: u64 = 0;
            for (ps, 0..) |p, pi| {
                defer base += p.bytes;
                if (p.row == 0 or p.kind == .words) continue;
                var r: u32 = 0;
                while (@as(u64, r) * p.row < p.bytes) : (r += 1) {
                    const pos = rowPosition(p, at, r);
                    TestDev.fill(want[0..p.row], pi, pos);
                    try std.testing.expectEqualSlices(u8, want[0..p.row], img[@intCast(base + @as(u64, r) * p.row)..][0..p.row]);
                }
            }
            // back into a live state of other rings (16 + 13 slots), in other cuts: every position its own bytes
            var dst = try TestDev.make(a, ps, at, 29, false);
            off = 0;
            while (off < total) {
                const n = @min(total - off, 1 + rnd.uintLessThan(u64, 57));
                try dst.copy(ps, at, off, img[@intCast(off)..][0..@intCast(n)], true, a);
                off += n;
            }
            try std.testing.expect(dst.holdsState(ps, at));
            // and into a snapshot slot's layout
            var sl = try TestDev.make(a, ps, at, 0, true);
            try sl.copy(ps, at, 0, img, true, a);
            try std.testing.expect(sl.holdsState(ps, at));
        }
    }
}

test "a snapshot slot's window row of a position" {
    try std.testing.expectEqual(@as(u32, 0), slotRow(2000, 1024, 976));
    try std.testing.expectEqual(@as(u32, 1023), slotRow(2000, 1024, 1999));
    try std.testing.expectEqual(@as(u32, 7), slotRow(10, 1024, 7));
}

test "the salt: a prompt's own tokens match a state whatever its passes; two passes never salt alike" {
    const p = [_]u32{ 200002, 5, 77, 90000, 3, 8 };
    const s1 = newSalt(1);
    const s2 = newSalt(2);
    try std.testing.expect(s1 != s2 and s1 != 0 and s1 >> salt_bits == 0);
    // pass one from 0, pass two from 3: positions 0, 1 and 3, 4 carry salts
    var st: [6]u32 = p;
    st[0] = salted(p[0], s1, 0);
    st[1] = salted(p[1], s1, 1);
    st[3] = salted(p[3], s2, 0);
    st[4] = salted(p[4], s2, 1);
    var other = st;
    other[3] = salted(p[3], s1, 0); // another pass from 3
    try std.testing.expect(!std.mem.eql(u32, &st, &other));
    for (st) |t| try std.testing.expect(t & 0x8000_0000 == 0); // the store's image keys keep bit 31
    for (st, p) |t, q| try std.testing.expectEqual(q, t & id_mask);
    try std.testing.expect(prefixOf(&st, &p) and prefixOf(st[0..4], &p));
    try std.testing.expect(!prefixOf(&st, p[0..5])); // longer than the prompt
    var q = p;
    q[3] = 1;
    try std.testing.expect(!prefixOf(&st, &q));
    try std.testing.expectEqual(@as(usize, 3), sharedIds(&st, &q));
    try std.testing.expectEqual(@as(usize, 6), sharedIds(&st, &p));
}

test "a checkpoint's files: deletions, the window blocks, the state, the rows from 0; the readers counted" {
    var es = [_]Entry{
        .{ .op = @intFromEnum(Op.block), .window = 0, .k = 0, .name = @splat(1) },
        .{ .op = @intFromEnum(Op.block), .window = 1024, .k = 30, .name = @splat(2) },
        .{ .op = @intFromEnum(Op.drop_block), .window = 0, .k = 0, .name = @splat(3) },
        .{ .op = @intFromEnum(Op.block), .window = 0, .k = 1, .name = @splat(4) },
        .{ .op = @intFromEnum(Op.state), .window = 0, .k = 0, .name = @splat(5) },
        .{ .op = @intFromEnum(Op.block), .window = 1024, .k = 31, .name = @splat(6) },
    };
    try std.testing.expectEqual(@as(u32, 3), try ckptOrder(&es, std.testing.allocator));
    const want = [_]u8{ 3, 2, 6, 5, 1, 4 };
    for (es, want) |e, w| try std.testing.expectEqual(w, e.name[0]);
}

test "the window gate: one prompt chunk beside the copy, then a wait for its ring readers; a verify waits at once" {
    var g: WindowGate = .{};
    try std.testing.expectEqual(@as(u64, 0), g.before(false)); // no copy: nothing waits
    g.open(2);
    try std.testing.expectEqual(@as(u64, 0), g.before(true)); // the next chunk runs beside the copy
    const T = struct {
        fn run(gg: *WindowGate) void {
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000_000 };
            _ = std.c.nanosleep(&ts, null);
            gg.done(false);
            _ = std.c.nanosleep(&ts, null);
            gg.done(false);
        }
    };
    const th = try std.Thread.spawn(.{}, T.run, .{&g});
    const waited = g.before(true); // the chunk after it waits for both readers (~40 ms)
    th.join();
    try std.testing.expect(waited >= 30_000_000);
    try std.testing.expectEqual(@as(u32, 0), g.left.load(.acquire));
    // a verify after a copy waits at once; a copy that ends (failed) releases every wait
    g.open(5);
    const th2 = try std.Thread.spawn(.{}, struct {
        fn run(gg: *WindowGate) void {
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 10_000_000 };
            _ = std.c.nanosleep(&ts, null);
            gg.done(true);
        }
    }.run, .{&g});
    try std.testing.expect(g.before(false) >= 5_000_000);
    th2.join();
    g.done(false); // past zero: stays zero
    try std.testing.expectEqual(@as(u32, 0), g.left.load(.acquire));
}

test "the trim hint: the last k requests' long states still held, the most positions; renoted, aged out, evicted" {
    var rs: Recents = .{};
    const T = struct {
        fn all(_: void, _: [32]u8) bool {
            return true;
        }
        fn not3(_: void, n: [32]u8) bool {
            return n[0] != 3;
        }
    };
    const n1: [32]u8 = @splat(1);
    const n2: [32]u8 = @splat(2);
    const n3: [32]u8 = @splat(3);
    try std.testing.expectEqual(@as(u32, 0), rs.hint(1, 4, {}, T.all)); // nothing noted
    rs.note(.{ .name = n1, .mapped = 196608, .request = 1 }); // A (138K) left at request 1
    try std.testing.expectEqual(@as(u32, 196608), rs.hint(2, 4, {}, T.all)); // B: keep A's mapping
    try std.testing.expectEqual(@as(u32, 196608), rs.hint(5, 4, {}, T.all)); // still within 4 requests
    try std.testing.expectEqual(@as(u32, 0), rs.hint(6, 4, {}, T.all)); // 5 requests on: A's mapping goes
    try std.testing.expectEqual(@as(u32, 0), rs.hint(2, 0, {}, T.all)); // k 0: no hint
    rs.note(.{ .name = n2, .mapped = 131072, .request = 3 });
    rs.note(.{ .name = n1, .mapped = 262144, .request = 4 }); // A again, grown: its note replaced
    try std.testing.expectEqual(@as(usize, 2), rs.n);
    try std.testing.expectEqual(@as(u32, 262144), rs.hint(5, 4, {}, T.all));
    rs.note(.{ .name = n3, .mapped = 327680, .request = 5 });
    try std.testing.expectEqual(@as(u32, 327680), rs.hint(6, 4, {}, T.all));
    try std.testing.expectEqual(@as(u32, 262144), rs.hint(6, 4, {}, T.not3)); // evicted since: not counted
    // past the cap the oldest note goes
    for (0..Recents.cap) |i| rs.note(.{ .name = @splat(@intCast(10 + i)), .mapped = 1, .request = 6 });
    try std.testing.expectEqual(@as(usize, Recents.cap), rs.n);
    try std.testing.expectEqual(@as(u32, 1), rs.hint(6, 4, {}, T.all));
}

test "a file list survives its encoding" {
    var names: [3][32]u8 = undefined;
    for (&names, 0..) |*n, i| @memset(n, @intCast(i + 1));
    const es = [_]Entry{ .{ .op = @intFromEnum(Op.block), .window = 0, .k = 3, .name = names[0] }, .{ .op = @intFromEnum(Op.block), .window = 1024, .k = 7, .name = names[1] }, .{ .op = @intFromEnum(Op.state), .window = 0, .k = 0, .name = names[2] } };
    var buf: [listBytes(3)]u8 = undefined;
    const bytes = encode(.{ .at = 4321, .count = 0, .flags = flag_wipe }, &es, &buf);
    const got = try decode(bytes);
    try std.testing.expectEqual(@as(u32, 4321), got.head.at);
    try std.testing.expectEqual(flag_wipe, got.head.flags);
    try std.testing.expectEqual(@as(usize, 3), got.entries.len);
    try std.testing.expectEqual(@as(u32, 1024), got.entries[1].window);
    try std.testing.expectEqualSlices(u8, &names[2], &got.entries[2].name);
    try std.testing.expectError(error.BadRowsList, decode(bytes[0 .. bytes.len - 4]));
}
