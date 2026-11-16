//! p4-cache: rank 0's book of the reusable sequence state (TF_INKLING2_PROMPT_REUSE=1; reuse.zig holds the device
//! side). `live`: the ids whose state the caches hold now (the last prompt, then every row a verify committed after
//! it; the last drawn token never ran); `ok`: the whole state (convolution rings, sliding windows, the draft's
//! context) is the one after `live`, so a prompt that extends `live` continues from its end. Slots: snapshots of the
//! state at earlier prompts' ends, each at a position `at` with `live[0 .. at]` its prompt (a pass from position s
//! rewrites positions s.. and drops every slot past s), and rank 0's copy of that prompt's last-row logits, so a
//! prompt equal to a slot's runs no rows. A prompt that shares less than `min` ids with either starts at 0.
//! xk-rows: `live` holds the ids salted where each pass of the state's history began (rows_layout.zig: a pass's
//! salt in its first two positions), the names the rows store gives the state's blocks; prompts match it with the
//! salts set aside. A state read back from disk (rows.zig) becomes the live state with its own salted ids.
const std = @import("std");
const lay = @import("rows_layout.zig");

pub const max_slots = 4;

/// Where a request's prompt pass starts.
pub const Plan = struct {
    at: usize = 0, // positions below are reused (0: a new sequence)
    restore: ?usize = null, // the slot restored first (null: the live state as it stands)
    repeat: bool = false, // the prompt is the slot's exactly: no rows run, the first token from its saved logits
    shared: usize = 0, // the prompt's common prefix with `live` (for the log)
    disk: bool = false, // xk-rows: a state read back from disk (rows.zig), the pass from its end

    pub fn mode(pl: Plan) []const u8 {
        return if (pl.disk) "disk" else if (pl.repeat) "repeat" else if (pl.restore != null) "snapshot" else if (pl.at > 0) "continue" else "new";
    }
};

pub const Book = struct {
    gpa: std.mem.Allocator,
    live: std.ArrayList(u32) = .empty,
    ok: bool = false,
    stuck: bool = false, // an append found no room: `live` stopped short of the state until the next pass starts
    broken: bool = false, // a step failed: the state is in question until the next pass starts
    n: usize = 0,
    at: [max_slots]?usize = @splat(null),
    logits: [max_slots][]f32 = @splat(&.{}),
    min: usize = 64,
    // xk-rows: the current pass's start and salt (its first two positions carry it), and where salts come from
    fork_pos: usize = 0,
    fork_salt: u32 = 1,
    seed: u64 = 0,
    passes: u64 = 0,

    /// `n` slots (at most max_slots), room for `context` ids, `vocab` logits a slot; prefixes under `min` not reused.
    pub fn init(gpa: std.mem.Allocator, n: usize, context: usize, vocab: usize, min: usize) !Book {
        var b: Book = .{ .gpa = gpa, .n = @min(n, max_slots), .min = @max(min, 1) };
        errdefer b.deinit();
        try b.live.ensureTotalCapacityPrecise(gpa, context);
        for (0..b.n) |i| b.logits[i] = try gpa.alloc(f32, vocab);
        return b;
    }

    pub fn deinit(b: *Book) void {
        for (&b.logits) |*l| if (l.len > 0) {
            b.gpa.free(l.*);
            l.* = &.{};
        };
        b.live.deinit(b.gpa);
    }

    /// The plan for prompt `p` with the model's length `len` now: continue the live state when `p` extends it by at
    /// least a row; else the slot at the longest position inside the shared prefix (`p` itself: a repeat); else 0.
    pub fn plan(b: *const Book, p: []const u32, len: usize) Plan {
        const live = b.live.items;
        const shared = lay.sharedIds(live, p);
        var pl: Plan = .{ .shared = shared };
        if (b.ok and !b.stuck and len == live.len and shared == live.len and shared >= b.min and shared < p.len) {
            pl.at = shared;
            return pl;
        }
        var best: ?usize = null;
        for (b.at[0..b.n], 0..) |a, i| {
            const q = a orelse continue;
            if (q < b.min or q > shared or q > p.len) continue;
            if (best == null or q > b.at[best.?].?) best = i;
        }
        if (best) |i| {
            pl.at = b.at[i].?;
            pl.restore = i;
            pl.repeat = pl.at == p.len;
        }
        return pl;
    }

    /// A pass from `pl.at` begins: positions from there on are rewritten, so `live` ends there and every slot past it
    /// goes; the state is in question until the pass completes (finish) or stops between chunks (pause). xk-rows: the
    /// pass's own salt from here (fork).
    pub fn start(b: *Book, pl: Plan) void {
        b.ok = false;
        b.stuck = false;
        b.broken = false;
        b.live.shrinkRetainingCapacity(@min(pl.at, b.live.items.len));
        for (b.at[0..b.n]) |*a| if (a.*) |q| {
            if (q > pl.at) a.* = null;
        };
        b.fork(b.live.items.len);
    }

    /// xk-rows: a state read back from disk is the live state now: its salted ids `stored`, through position
    /// `stored.len`. With `same_rows` (the live state held its rows already: holds) the slots below it stay; else
    /// every slot goes (the rows under them were rewritten). The pass from its end forks.
    pub fn startDisk(b: *Book, stored: []const u32, same_rows: bool) void {
        b.ok = false;
        b.stuck = false;
        b.broken = false;
        for (b.at[0..b.n]) |*a| if (a.*) |q| {
            if (!same_rows or q > stored.len) a.* = null;
        };
        b.live.clearRetainingCapacity();
        if (stored.len > b.live.capacity) {
            b.stuck = true;
            return;
        }
        b.live.appendSliceAssumeCapacity(stored);
        b.fork(stored.len);
    }

    /// A new pass's salt from position `pos`: a fresh salt for every pass a process runs (its seed, the count).
    fn fork(b: *Book, pos: usize) void {
        b.passes += 1;
        b.fork_pos = pos;
        b.fork_salt = lay.newSalt(b.seed ^ (b.passes *% 0x9e3779b97f4a7c15));
    }

    /// xk-rows: whether the live state holds stored state `stored`'s rows: the live salted ids begin with its own.
    pub fn holds(b: *const Book, stored: []const u32) bool {
        return b.live.items.len >= stored.len and std.mem.eql(u32, b.live.items[0..stored.len], stored);
    }

    /// A step failed on the way: no continue from this state (the slots, positional below their ends, still hold).
    pub fn fail(b: *Book) void {
        b.broken = true;
        b.ok = false;
    }

    /// xk-rows: the live state was overwritten below its end (a read from disk that failed part way): nothing holds.
    pub fn lose(b: *Book) void {
        b.fail();
        b.live.clearRetainingCapacity();
        for (b.at[0..b.n]) |*a| a.* = null;
    }

    /// Rows now in the caches, in position order: a prompt chunk's, a verify's kept rows, a one-token step's; the
    /// current pass's first two positions take its salt.
    pub fn extend(b: *Book, ids: []const u32) void {
        if (b.stuck) return;
        if (b.live.items.len + ids.len > b.live.capacity) {
            b.stuck = true;
            b.ok = false;
            return;
        }
        const base = b.live.items.len;
        b.live.appendSliceAssumeCapacity(ids);
        for (0..2) |i| {
            const pos = b.fork_pos + i;
            if (pos >= base and pos < base + ids.len) b.live.items[pos] = lay.salted(ids[pos - base], b.fork_salt, i);
        }
    }

    /// The slot a prompt's end at position `len` goes to, or none (a prompt under `min`, a repeat: its slot holds
    /// it): an empty slot, else the one at the lowest position, sparing the slot the pass restored.
    pub fn slotFor(b: *const Book, len: usize, pl: Plan) ?usize {
        if (b.n == 0 or len < b.min or pl.repeat or b.stuck) return null;
        var pick: ?usize = null;
        for (b.at[0..b.n], 0..) |a, i| {
            const q = a orelse return i;
            if (b.n > 1 and pl.restore == i) continue;
            if (pick == null or q < b.at[pick.?].?) pick = i;
        }
        return pick;
    }

    /// Slot `i` is being overwritten (its save step runs next).
    pub fn clear(b: *Book, i: usize) void {
        b.at[i] = null;
    }

    /// The pass completed at position `len` with the state the one after `live` (the prompt); slot `slot`, saved by
    /// the pass's last step, holds it, with the prompt's last-row `logits` (rank 0's).
    pub fn finish(b: *Book, slot: ?usize, len: usize, logits: []const f32) void {
        b.ok = !b.stuck and !b.broken and b.live.items.len == len;
        const i = slot orelse return;
        if (!b.ok or logits.len > b.logits[i].len) return;
        @memcpy(b.logits[i][0..logits.len], logits);
        b.at[i] = len;
    }

    /// The pass stopped between chunks, or the stream ended (its rows committed): the state is the one after `live`
    /// when the model's length `len` says so.
    pub fn pause(b: *Book, len: usize) void {
        b.ok = !b.stuck and !b.broken and len == b.live.items.len;
    }
};

fn seq(a: std.mem.Allocator, from: u32, n: usize) ![]u32 {
    const s = try a.alloc(u32, n);
    for (s, 0..) |*x, i| x.* = from + @as(u32, @intCast(i));
    return s;
}

test "a first prompt starts at 0, a follow-up continues the live state, a re-rendered one restores its slot" {
    const a = std.testing.allocator;
    var b = try Book.init(a, 2, 4096, 8, 64);
    defer b.deinit();
    const p1 = try seq(a, 1000, 100);
    defer a.free(p1);
    try std.testing.expectEqualStrings("new", b.plan(p1, 0).mode());
    // turn 1: prompt 100 ids, its end into a slot, a reply of 20 committed rows
    var pl = b.plan(p1, 0);
    b.start(pl);
    b.extend(p1);
    const s1 = b.slotFor(p1.len, pl).?;
    b.clear(s1);
    const lg = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    b.finish(s1, p1.len, &lg);
    try std.testing.expect(b.ok and b.at[s1].? == 100);
    const reply = try seq(a, 5000, 20);
    defer a.free(reply);
    b.extend(reply);
    b.pause(120);
    try std.testing.expect(b.ok);
    // turn 2 renders the reply as generated: continue at 120
    var p2: std.ArrayList(u32) = .empty;
    defer p2.deinit(a);
    try p2.appendSlice(a, p1);
    try p2.appendSlice(a, reply);
    try p2.appendSlice(a, &.{ 7, 8, 9 });
    pl = b.plan(p2.items, 120);
    try std.testing.expectEqualStrings("continue", pl.mode());
    try std.testing.expectEqual(@as(usize, 120), pl.at);
    // the model's length disagrees: no continue, the slot instead
    try std.testing.expectEqualStrings("snapshot", b.plan(p2.items, 119).mode());
    // the reply re-rendered differently from its 2nd id: back to the prompt's end
    p2.items[101] = 1;
    pl = b.plan(p2.items, 120);
    try std.testing.expectEqualStrings("snapshot", pl.mode());
    try std.testing.expectEqual(@as(usize, 100), pl.at);
    try std.testing.expectEqual(@as(usize, 101), pl.shared);
    // the same prompt again: the slot's state and logits, no rows
    pl = b.plan(p1, 120);
    try std.testing.expectEqualStrings("repeat", pl.mode());
    try std.testing.expectEqual(@as(usize, 100), pl.at);
    // an unrelated prompt, and one sharing fewer than 64 ids: a new sequence
    const other = try seq(a, 90000, 300);
    defer a.free(other);
    try std.testing.expectEqualStrings("new", b.plan(other, 120).mode());
    try std.testing.expectEqualStrings("new", b.plan(p1[0..63], 120).mode());
}

test "a pass from position s drops the slots past s; slots fill empty first, then the lowest, sparing the restored" {
    const a = std.testing.allocator;
    var b = try Book.init(a, 2, 4096, 4, 64);
    defer b.deinit();
    const lg = [_]f32{ 0, 0, 0, 0 };
    const p = try seq(a, 10, 1000);
    defer a.free(p);
    // turn 1 at 200, turn 2 continues to 500: two slots
    var pl = b.plan(p[0..200], 0);
    b.start(pl);
    b.extend(p[0..200]);
    const s1 = b.slotFor(200, pl).?;
    b.finish(s1, 200, &lg);
    b.extend(p[200..250]);
    b.pause(250);
    pl = b.plan(p[0..500], 250);
    try std.testing.expectEqualStrings("continue", pl.mode());
    b.start(pl);
    b.extend(p[250..500]);
    const s2 = b.slotFor(500, pl).?;
    try std.testing.expect(s2 != s1);
    b.finish(s2, 500, &lg);
    // an edit inside turn 2's prompt (after 300): back to the slot at 200; the slot at 500 goes
    var q = try a.dupe(u32, p[0..600]);
    defer a.free(q);
    q[300] = 7;
    pl = b.plan(q, 500);
    try std.testing.expectEqual(@as(usize, 200), pl.at);
    try std.testing.expectEqual(s1, pl.restore.?);
    b.start(pl);
    try std.testing.expect(b.at[s2] == null and b.at[s1].? == 200 and b.live.items.len == 200);
    b.extend(q[200..600]);
    // the new end takes the empty slot, not the restored one
    try std.testing.expectEqual(s2, b.slotFor(600, pl).?);
    b.finish(s2, 600, &lg);
    // a third prompt end replaces the lowest (200)
    b.extend(q[0..50]); // a reply
    b.pause(650);
    var r: std.ArrayList(u32) = .empty;
    defer r.deinit(a);
    try r.appendSlice(a, q[0..600]);
    try r.appendSlice(a, q[0..50]);
    try r.appendSlice(a, &.{ 1, 2 });
    pl = b.plan(r.items, 650);
    try std.testing.expectEqualStrings("continue", pl.mode());
    b.start(pl);
    b.extend(&.{ 1, 2 });
    try std.testing.expectEqual(s1, b.slotFor(652, pl).?);
    // a new sequence drops every slot
    b.start(.{});
    try std.testing.expect(b.at[0] == null and b.at[1] == null and b.live.items.len == 0);
}

test "a failed step keeps the book from continuing the state, even when the lengths agree" {
    const a = std.testing.allocator;
    var b = try Book.init(a, 1, 1024, 1, 4);
    defer b.deinit();
    const p = try seq(a, 0, 100);
    defer a.free(p);
    b.start(.{});
    b.extend(p[0..50]);
    b.finish(b.slotFor(50, .{}), 50, &[_]f32{0});
    b.fail();
    b.pause(50);
    try std.testing.expect(!b.ok);
    const pl = b.plan(p, 50);
    try std.testing.expectEqualStrings("snapshot", pl.mode()); // the slot, not the live state
    b.start(pl);
    try std.testing.expect(!b.broken);
}

test "no room for an append: the book stops claiming the state until the next pass" {
    const a = std.testing.allocator;
    var b = try Book.init(a, 1, 128, 1, 4);
    defer b.deinit();
    const p = try seq(a, 0, 200);
    defer a.free(p);
    b.start(.{});
    b.extend(p[0..100]);
    b.extend(p[100..200]); // past the room
    try std.testing.expect(b.stuck and !b.ok and b.live.items.len == 100);
    b.finish(b.slotFor(200, .{}), 200, &[_]f32{0});
    try std.testing.expect(!b.ok and b.at[0] == null);
    try std.testing.expectEqualStrings("new", b.plan(p, 200).mode());
}

test "xk-rows: every pass salts its first two positions; a rerun of the same ids from the same place salts anew" {
    const a = std.testing.allocator;
    var b = try Book.init(a, 2, 4096, 4, 64);
    defer b.deinit();
    const p = try seq(a, 100, 600);
    defer a.free(p);
    b.start(.{}); // a pass from 0
    b.extend(p[0..300]);
    b.pause(300);
    for (b.live.items, p[0..300], 0..) |s, t, i| {
        if (i < 2) try std.testing.expect(s != t and s & lay.id_mask == t) else try std.testing.expectEqual(t, s);
    }
    const first = try a.dupe(u32, b.live.items);
    defer a.free(first);
    // continue at 300: positions 300 and 301 carry the second pass's salt
    var pl = b.plan(p[0..500], 300);
    try std.testing.expectEqualStrings("continue", pl.mode());
    b.start(pl);
    b.extend(p[300..500]);
    try std.testing.expect(b.live.items[300] != p[300] and b.live.items[301] != p[301] and b.live.items[302] == p[302]);
    try std.testing.expect(b.holds(first)); // the rows below 300 are the first pass's
    const second = try a.dupe(u32, b.live.items);
    defer a.free(second);
    // back to 300 and the same ids again: a third pass, its own salt; the first pass's rows still held
    b.start(.{ .at = 300 });
    b.extend(p[300..500]);
    try std.testing.expect(b.holds(first) and !b.holds(second));
    try std.testing.expectEqual(@as(usize, 500), lay.sharedIds(b.live.items, p));
    // a state read back from disk becomes the live state, salts and all; its pass forks at its end
    b.startDisk(second[0..400], false);
    try std.testing.expect(b.holds(second[0..400]) and b.at[0] == null and b.at[1] == null);
    b.extend(p[400..420]);
    try std.testing.expect(b.live.items[400] != p[400] and b.live.items[402] == p[402]);
    b.lose();
    try std.testing.expect(b.live.items.len == 0 and !b.ok);
}
