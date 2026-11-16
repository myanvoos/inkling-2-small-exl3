//! xr: the pieces a lazy context buffer is mapped in (lazy.zig `Range` with `whole`): each growth's bytes one physical
//! allocation (one cuMemCreate + cuMemMap + cuMemSetAccess, where 2 MiB chunks took ~0.5 ms each: a 138K restore's
//! growth mapped ~1,100 of them), unmapped whole (cuMemUnmap takes a mapping whole). Their ends are kept ascending; a
//! trim to a size unmaps the pieces at or past it, so trims to the growths' own boundaries (lazy.zig grows a step at a
//! time and trims to a step) are exact. std only: host-tested.
const std = @import("std");

pub const Pieces = struct {
    ends: []usize, // the pieces' ends, ascending (the first starts at 0)
    n: usize = 0,

    pub const Span = struct { start: usize, end: usize };

    /// Bytes mapped: the last piece's end.
    pub fn mapped(p: *const Pieces) usize {
        return if (p.n == 0) 0 else p.ends[p.n - 1];
    }

    /// A new piece from mapped() to `to` (past it): its span, for the caller to map. error.TooManyPieces: no room.
    pub fn next(p: *const Pieces, to: usize) !?Span {
        const from = p.mapped();
        if (to <= from) return null;
        if (p.n == p.ends.len) return error.TooManyPieces;
        return .{ .start = from, .end = to };
    }

    /// The piece `next` gave is mapped now.
    pub fn push(p: *Pieces, s: Span) void {
        std.debug.assert(s.start == p.mapped() and s.end > s.start and p.n < p.ends.len);
        p.ends[p.n] = s.end;
        p.n += 1;
    }

    /// The last piece, if it lies wholly at or past `keep`: the next to unmap for a trim to `keep`.
    pub fn lastPast(p: *const Pieces, keep: usize) ?Span {
        if (p.n == 0) return null;
        const start = if (p.n == 1) 0 else p.ends[p.n - 2];
        if (start < keep) return null;
        return .{ .start = start, .end = p.ends[p.n - 1] };
    }

    /// The last piece is unmapped now.
    pub fn pop(p: *Pieces) void {
        std.debug.assert(p.n > 0);
        p.n -= 1;
    }

    /// Bytes a trim to `keep` would unmap (the pieces wholly at or past it).
    pub fn freedAt(p: *const Pieces, keep: usize) usize {
        var i = p.n;
        while (i > 0) : (i -= 1) {
            const start = if (i == 1) 0 else p.ends[i - 2];
            if (start < keep) break;
        }
        return p.mapped() - (if (i == 0) 0 else p.ends[i - 1]);
    }
};

// ---- tests: lazy.zig's sequence (open at the budget, a growth a step at a time, trims to a step) on a byte rule

const step = 65536;
const granule = 2 << 20;

/// lazy.zig's bytes of an entry at `t` positions (per_token, extra, least), rounded to the granule (Range.want).
fn want(per: usize, extra: usize, least: usize, t: usize) usize {
    return std.mem.alignForward(usize, per * @max(t + extra, least), granule);
}

/// A growth from `old` to `new` positions mapped a step at a time; returns the pieces mapped.
fn grow(p: *Pieces, per: usize, extra: usize, least: usize, old: usize, new: usize) !usize {
    var made: usize = 0;
    var s = (old / step + 1) * step;
    while (true) {
        const t = @min(s, new);
        if (try p.next(want(per, extra, least, t))) |sp| {
            p.push(sp);
            made += 1;
        }
        if (t == new) break;
        s += step;
    }
    return made;
}

/// A trim to `keep` positions; returns the pieces unmapped.
fn trim(p: *Pieces, per: usize, extra: usize, least: usize, keep: usize) usize {
    var n: usize = 0;
    const k = want(per, extra, least, keep);
    while (p.lastPast(k)) |_| {
        p.pop();
        n += 1;
    }
    return n;
}

test "a growth maps one piece a step (2 MiB chunks took 32 a step for a 1 KiB row); trims land on the steps" {
    var buf: [16]usize = undefined;
    // a global layer's K cache: 1 KiB a position, no extra
    var p: Pieces = .{ .ends = &buf };
    try std.testing.expectEqual(@as(usize, 1), try grow(&p, 1024, 0, 0, 0, step)); // open at the budget
    try std.testing.expectEqual(@as(usize, 64 << 20), p.mapped());
    // the 138K restore: 65536 -> 196608 in two pieces (it took 64 chunks of 2 MiB)
    try std.testing.expectEqual(@as(usize, 2), try grow(&p, 1024, 0, 0, step, 3 * step));
    try std.testing.expectEqual(@as(usize, 192 << 20), p.mapped());
    // a new sequence trims to the budget: both growth pieces go, exactly
    try std.testing.expectEqual(@as(usize, 64 << 20), p.mapped() - p.freedAt(want(1024, 0, 0, step)));
    try std.testing.expectEqual(@as(usize, 2), trim(&p, 1024, 0, 0, step));
    try std.testing.expectEqual(@as(usize, 64 << 20), p.mapped());
    // nothing under the budget's piece is ever unmapped by a trim to the budget
    try std.testing.expectEqual(@as(usize, 0), trim(&p, 1024, 0, 0, step));
    // a trim to one step of two
    _ = try grow(&p, 1024, 0, 0, step, 3 * step);
    try std.testing.expectEqual(@as(usize, 1), trim(&p, 1024, 0, 0, 2 * step));
    try std.testing.expectEqual(@as(usize, 128 << 20), p.mapped());
}

test "entries with extra and least positions: pieces end where the steps' bytes do; a context end past the last step" {
    var buf: [16]usize = undefined;
    // the draft's rope table: 512 B a position, 64 extra
    var p: Pieces = .{ .ends = &buf };
    _ = try grow(&p, 512, 64, 0, 0, step);
    const at_budget = p.mapped();
    try std.testing.expectEqual(want(512, 64, 0, step), at_budget);
    _ = try grow(&p, 512, 64, 0, step, 500000); // to the context's end (not a step multiple)
    try std.testing.expectEqual(want(512, 64, 0, 500000), p.mapped());
    try std.testing.expectEqual(@as(usize, 7), p.n - 1); // the steps past the budget (6 whole), the last one short
    _ = trim(&p, 512, 64, 0, step);
    try std.testing.expectEqual(at_budget, p.mapped());
    // a score scratch whose least positions cover every step: one piece, never trimmed
    var q: Pieces = .{ .ends = buf[8..] };
    _ = try grow(&q, 4096, 0, 400000, 0, step);
    try std.testing.expectEqual(@as(usize, 0), try grow(&q, 4096, 0, 400000, step, 3 * step));
    try std.testing.expectEqual(@as(usize, 1), q.n);
}

test "a full piece list refuses the next growth (the caller sizes it: context / step + 3)" {
    var buf: [2]usize = undefined;
    var p: Pieces = .{ .ends = &buf };
    _ = try grow(&p, 1024, 0, 0, 0, step);
    _ = try grow(&p, 1024, 0, 0, step, 2 * step);
    try std.testing.expectError(error.TooManyPieces, grow(&p, 1024, 0, 0, 2 * step, 3 * step));
    try std.testing.expectEqual(@as(usize, 2), p.n);
}
