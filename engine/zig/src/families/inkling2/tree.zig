//! Tree verify windows for Inkling-2 (design: docs/design/inkling2-tree-drafts.md): the host bookkeeping a window of
//! R rows with parents needs on top of the chain path, std only (host prototype; nothing here touches the GPU).
//!
//! A window's row r (row 0 the pending token, parents[0] = -1, parents[r] < r) sits at logical position
//! base + depth(r). On the device every row is still written at the chunk's row index (cache slot base + r, the conv
//! history's row K - 1 + r), so the forward's buffers stay those of a chain of R rows; what changes is what each row
//! reads:
//!   attention   committed positions as a chain row at the same logical position would, plus the window rows that
//!               are its ancestors or itself (`anc` bit masks), every distance logical (depth differences);
//!   convolutions lag l of row r is its l-th ancestor's input, or, above row 0, the ring's (`convSources`);
//!   head prep   the global layers' tau(pos) at the logical position (`positions`).
//! After the verify keeps a path (rows 0 = path[0] < path[1] < ... each the parent of the next):
//!   the positional caches move row path[i] to slot base + i (`compaction`, in increasing i: never a clobber);
//!   every conv ring becomes the last K - 1 inputs along the path (`ringAfter`: rows of the recorded history);
//!   the draft absorbs the path's taps, gathered in path order.
const std = @import("std");

pub const max_rows = 32; // a row's ancestors fit one u32 mask
pub const max_lags = 7; // K - 1 at most (Inkling-2: K = 4)

pub const Window = struct {
    rows: usize,
    depth: [max_rows]u8 = undefined,
    anc: [max_rows]u32 = undefined, // bit j: row j is row r's ancestor or row r itself
    parents: [max_rows]i8 = undefined,

    /// `parents` per row (row 0: -1), parents before children.
    pub fn init(parents: []const i32) !Window {
        if (parents.len == 0 or parents.len > max_rows) return error.BadRows;
        if (parents[0] != -1) return error.BadRoot;
        var w: Window = .{ .rows = parents.len };
        w.depth[0] = 0;
        w.anc[0] = 1;
        w.parents[0] = -1;
        for (parents[1..], 1..) |p, r| {
            if (p < 0 or p >= @as(i32, @intCast(r))) return error.ParentAfterChild;
            const q: usize = @intCast(p);
            w.depth[r] = w.depth[q] + 1;
            w.anc[r] = w.anc[q] | (@as(u32, 1) << @intCast(r));
            w.parents[r] = @intCast(p);
        }
        return w;
    }

    pub fn isChain(w: *const Window) bool {
        for (1..w.rows) |r| if (w.parents[r] != @as(i8, @intCast(r - 1))) return false;
        return true;
    }

    /// Each row's logical position (head prep's tau, the rel-bias distances, the sliding window's edge).
    pub fn positions(w: *const Window, base: usize, out: []i32) void {
        for (0..w.rows) |r| out[r] = @intCast(base + w.depth[r]);
    }

    /// The `lags` (= K - 1) inputs a row's convolution reads before its own, nearest first: lag l >= 1 is the l-th
    /// ancestor's chunk row (>= 0), or ring slot k (oldest first, k = lags + depth - l) encoded as -1 - k.
    pub fn convSources(w: *const Window, lags: usize, out: [][max_lags]i8) !void {
        if (lags > max_lags) return error.TooManyLags;
        for (0..w.rows) |r| {
            var at: i32 = @intCast(r);
            for (1..lags + 1) |l| {
                at = if (at >= 0) w.parents[@intCast(at)] else at;
                const d: i32 = @as(i32, w.depth[r]) - @as(i32, @intCast(l));
                out[r][l - 1] = if (d >= 0) @intCast(at) else @intCast(-1 - (@as(i32, @intCast(lags)) + d));
            }
        }
    }

    /// Whether row r's path runs through `path` (rows 0, path[1], ...): the path's rows each the parent of the next.
    pub fn checkPath(w: *const Window, path: []const u32) !void {
        if (path.len == 0 or path[0] != 0) return error.BadPath;
        for (path[1..], 1..) |row, i| {
            if (row >= w.rows or w.parents[row] != @as(i8, @intCast(path[i - 1]))) return error.BadPath;
        }
    }
};

/// The ring after keeping `path` (Model.rewind generalised): ring slot j (oldest first, j < lags) is history row
/// out[j], where the history is [lags ring rows as they were][the chunk's R rows' inputs] (the sconv kernels' `hist`).
pub fn ringAfter(path: []const u32, lags: usize, out: []u32) void {
    const n = path.len; // kept rows
    for (0..lags) |j| {
        // logical offset from base of ring slot j after n rows: n - lags + j
        const off: i64 = @as(i64, @intCast(n)) - @as(i64, @intCast(lags)) + @as(i64, @intCast(j));
        out[j] = if (off >= 0) @intCast(lags + path[@intCast(off)]) else @intCast(@as(i64, @intCast(lags)) + off);
    }
}

/// Cache moves that make the kept path positional: row path[i] to row i (slots base + ...), in increasing i, only
/// where they differ. A move never overwrites a later move's source (path[i] >= i and increasing).
pub fn compaction(path: []const u32, out: [][2]u32) usize {
    var n: usize = 0;
    for (path, 0..) |row, i| if (row != i) {
        out[n] = .{ row, @intCast(i) };
        n += 1;
    };
    return n;
}

/// The draft head's lanes grouped by level for a batched Markov walk: level d (0-based) expands the nodes of depth d - 1
/// (level 0: the anchor alone), one Markov step each, and lane i takes rank `ranks[i]` of its parent's step. `slot[i]`
/// is lane i's parent's index within its level's batch; `width[d]` the batch rows of level d.
pub const Levels = struct {
    width: [16]u8 = @splat(0),
    slot: [max_rows]u8 = undefined,
    levels: usize = 0,

    pub fn init(parents: []const i32, depths: []const u8) !Levels {
        var lv: Levels = .{};
        var expanded: [max_rows]i32 = @splat(-1); // lane -> its index in the next level's batch
        var root_slot: i32 = -1;
        for (parents, depths, 0..) |p, d, i| {
            if (d >= lv.width.len) return error.TooDeep;
            if (p >= @as(i32, @intCast(i))) return error.ParentAfterChild;
            if ((p < 0) != (d == 0)) return error.BadDepth;
            if (p >= 0 and depths[@intCast(p)] + 1 != d) return error.BadDepth;
            const e: *i32 = if (p < 0) &root_slot else &expanded[@intCast(p)];
            if (e.* < 0) {
                e.* = lv.width[d];
                lv.width[d] += 1;
            }
            lv.slot[i] = @intCast(e.*);
            lv.levels = @max(lv.levels, @as(usize, d) + 1);
        }
        return lv;
    }
};

// ------------------------------------------------------------------------------------------------------------ tests

const K = 4; // Inkling-2's short convolutions
const LAGS = K - 1;

/// The reference chain convolution (ink2_sconv_static's arithmetic) over `xs` after `ring`; outputs and the ring after.
fn chainConv(w: [K]f32, ring0: [LAGS]f32, xs: []const f32, ys: []f32, ring_out: *[LAGS]f32) void {
    var win: [K]f32 = undefined;
    @memcpy(win[0..LAGS], &ring0);
    const div = @sqrt(@as(f32, K + 1));
    for (xs, 0..) |x, t| {
        win[K - 1] = x;
        var s: f32 = 0;
        for (0..K) |k| s = @mulAdd(f32, w[k], win[k], s);
        ys[t] = (s + x) / div;
        for (0..LAGS) |k| win[k] = win[k + 1];
    }
    @memcpy(ring_out, win[0..LAGS]);
}

/// The tree convolution as the kernel would run it: each row's window gathered by convSources.
fn treeConv(win_: *const Window, w: [K]f32, ring0: [LAGS]f32, xs: []const f32, ys: []f32) !void {
    var src: [max_rows][max_lags]i8 = undefined;
    try win_.convSources(LAGS, src[0..win_.rows]);
    const div = @sqrt(@as(f32, K + 1));
    for (0..win_.rows) |r| {
        var win: [K]f32 = undefined;
        win[K - 1] = xs[r];
        for (1..K) |l| {
            const s = src[r][l - 1];
            win[K - 1 - l] = if (s >= 0) xs[@intCast(s)] else ring0[@intCast(-1 - @as(i32, s))];
        }
        var s: f32 = 0;
        for (0..K) |k| s = @mulAdd(f32, w[k], win[k], s);
        ys[r] = (s + xs[r]) / div;
    }
}

fn pathTo(win: *const Window, r: usize, out: []u32) []u32 {
    var n: usize = win.depth[r] + 1;
    var at: i32 = @intCast(r);
    var i = n;
    while (i > 0) : (i -= 1) {
        out[i - 1] = @intCast(at);
        at = win.parents[@intCast(at)];
    }
    n = win.depth[r] + 1;
    return out[0..n];
}

fn randomParents(rng: std.Random, rows: usize, out: []i32) void {
    out[0] = -1;
    for (1..rows) |r| out[r] = @intCast(rng.uintLessThan(usize, r));
}

test "tree convolutions equal the chain convolution along every row's path, bitwise; rings after any kept path too" {
    var prng = std.Random.DefaultPrng.init(0x1d2);
    const rng = prng.random();
    for (0..2000) |_| {
        const rows = 1 + rng.uintLessThan(usize, 16);
        var parents: [16]i32 = undefined;
        randomParents(rng, rows, &parents);
        const win = try Window.init(parents[0..rows]);
        var w: [K]f32 = undefined;
        for (&w) |*x| x.* = rng.floatNorm(f32);
        var ring0: [LAGS]f32 = undefined;
        for (&ring0) |*x| x.* = rng.floatNorm(f32);
        var xs: [16]f32 = undefined;
        for (xs[0..rows]) |*x| x.* = rng.floatNorm(f32);
        var ys: [16]f32 = undefined;
        try treeConv(&win, w, ring0, xs[0..rows], ys[0..rows]);
        // the history the kernels record: ring then every row's input
        var hist: [LAGS + 16]f32 = undefined;
        @memcpy(hist[0..LAGS], &ring0);
        @memcpy(hist[LAGS .. LAGS + rows], xs[0..rows]);
        for (0..rows) |r| {
            var pb: [16]u32 = undefined;
            const path = pathTo(&win, r, &pb);
            try win.checkPath(path);
            var px: [16]f32 = undefined;
            for (path, 0..) |row, i| px[i] = xs[row];
            var py: [16]f32 = undefined;
            var ring_ref: [LAGS]f32 = undefined;
            chainConv(w, ring0, px[0..path.len], py[0..path.len], &ring_ref);
            try std.testing.expectEqual(@as(u32, @bitCast(py[path.len - 1])), @as(u32, @bitCast(ys[r])));
            // keeping this path: the ring from the history
            var src: [LAGS]u32 = undefined;
            ringAfter(path, LAGS, &src);
            for (0..LAGS) |j| try std.testing.expectEqual(@as(u32, @bitCast(ring_ref[j])), @as(u32, @bitCast(hist[src[j]])));
        }
    }
}

test "a chain window's sources are the chain's own: the previous rows, then the ring" {
    const win = try Window.init(&.{ -1, 0, 1, 2, 3 });
    try std.testing.expect(win.isChain());
    var src: [5][max_lags]i8 = undefined;
    try win.convSources(LAGS, &src);
    try std.testing.expectEqualSlices(i8, &.{ -3, -2, -1 }, src[0][0..LAGS]); // ring slots 2, 1, 0 (nearest first)
    try std.testing.expectEqualSlices(i8, &.{ 0, -3, -2 }, src[1][0..LAGS]);
    try std.testing.expectEqualSlices(i8, &.{ 3, 2, 1 }, src[4][0..LAGS]);
    var ring: [LAGS]u32 = undefined;
    ringAfter(&.{ 0, 1 }, LAGS, &ring); // Model.rewind(base, 2): hist rows 2, 3, 4
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4 }, &ring);
}

/// The toy attention's visible keys for a row: (logical distance, key identity) pairs, committed keys identified by
/// their position, window rows by 1000 + row.
const Key = struct { dist: i64, id: i64 };

fn keyLess(_: void, a: Key, b: Key) bool {
    return a.dist < b.dist or (a.dist == b.dist and a.id < b.id);
}

test "tree attention sees what a chain row at the same position sees: same keys, same distances, same window" {
    var prng = std.Random.DefaultPrng.init(0x7ee);
    const rng = prng.random();
    for (0..500) |_| {
        const rows = 1 + rng.uintLessThan(usize, 16);
        var parents: [16]i32 = undefined;
        randomParents(rng, rows, &parents);
        const win = try Window.init(parents[0..rows]);
        const base = rng.uintLessThan(usize, 40);
        const window = 1 + rng.uintLessThan(usize, 12); // 0 would be global; the sliding edge is what is tested
        var pos: [16]i32 = undefined;
        win.positions(base, &pos);
        for (0..rows) |r| {
            // tree: committed keys p in (pos - window, base), window rows j with anc bit set
            var tk: [64]Key = undefined;
            var nt: usize = 0;
            const lo: i64 = @max(0, @as(i64, pos[r]) - @as(i64, @intCast(window)) + 1);
            var p: i64 = lo;
            while (p < base) : (p += 1) {
                tk[nt] = .{ .dist = pos[r] - p, .id = p };
                nt += 1;
            }
            for (0..rows) |j| if (win.anc[r] >> @intCast(j) & 1 == 1) {
                const d: i64 = @as(i64, win.depth[r]) - @as(i64, win.depth[j]);
                if (d < window) {
                    tk[nt] = .{ .dist = d, .id = 1000 + @as(i64, @intCast(j)) };
                    nt += 1;
                }
            };
            // chain: the path's rows at positions base .. base + depth, a plain causal sliding window
            var pb: [16]u32 = undefined;
            const path = pathTo(&win, r, &pb);
            var ck: [64]Key = undefined;
            var nc: usize = 0;
            const q: i64 = @intCast(base + path.len - 1);
            var pp: i64 = @max(0, q - @as(i64, @intCast(window)) + 1);
            while (pp <= q) : (pp += 1) {
                ck[nc] = .{ .dist = q - pp, .id = if (pp < base) pp else 1000 + @as(i64, path[@intCast(pp - @as(i64, @intCast(base)))]) };
                nc += 1;
            }
            std.mem.sort(Key, tk[0..nt], {}, keyLess);
            std.mem.sort(Key, ck[0..nc], {}, keyLess);
            try std.testing.expectEqual(nc, nt);
            try std.testing.expectEqualSlices(Key, ck[0..nc], tk[0..nt]);
            try std.testing.expectEqual(q, @as(i64, pos[r]));
        }
    }
}

test "compaction leaves the cache positional: kept rows at base .. base + n - 1, the window behind untouched" {
    var prng = std.Random.DefaultPrng.init(0xc0c);
    const rng = prng.random();
    const window = 8;
    const R = 16;
    const slots = window + R; // a sliding layer's ring (Model.makeState: window + max_rows)
    for (0..1000) |_| {
        const rows = 1 + rng.uintLessThan(usize, R);
        var parents: [R]i32 = undefined;
        randomParents(rng, rows, &parents);
        const win = try Window.init(parents[0..rows]);
        const base = rng.uintLessThan(usize, 100);
        var cache: [slots]i64 = undefined;
        for (0..slots) |s| cache[s] = -1;
        // committed positions [base - slots, base) hold their position; the chunk's rows at slot (base + r) % slots
        var p: usize = if (base > slots) base - slots else 0;
        while (p < base) : (p += 1) cache[p % slots] = @intCast(p);
        for (0..rows) |r| cache[(base + r) % slots] = 1000 + @as(i64, @intCast(r));
        const r = rng.uintLessThan(usize, rows);
        var pb: [R]u32 = undefined;
        const path = pathTo(&win, r, &pb);
        var moves: [R][2]u32 = undefined;
        const n = compaction(path, &moves);
        for (moves[0..n]) |m| cache[(base + m[1]) % slots] = cache[(base + m[0]) % slots];
        for (path, 0..) |row, i| try std.testing.expectEqual(1000 + @as(i64, row), cache[(base + i) % slots]);
        // the next chunk's rows read positions down to (base + path.len) - window: still there
        const next = base + path.len;
        var q: usize = if (next > window) next - window else 0;
        while (q < base) : (q += 1) try std.testing.expectEqual(@as(i64, @intCast(q)), cache[q % slots]);
    }
}

test "levels: a lane tree's Markov walk batched by depth, each lane under its parent's batch row" {
    // anchor -> a (rank 0), b (rank 1); a -> c (rank 0), d (rank 1); b -> e (rank 0); c -> f (rank 0)
    const parents = [_]i32{ -1, -1, 0, 0, 1, 2 };
    const depths = [_]u8{ 0, 0, 1, 1, 1, 2 };
    const lv = try Levels.init(&parents, &depths);
    try std.testing.expectEqual(@as(usize, 3), lv.levels);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 1 }, lv.width[0..3]); // level 1 expands a and b, level 2 c
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0, 1, 0 }, lv.slot[0..6]);
    try std.testing.expectError(error.BadDepth, Levels.init(&.{ -1, 0 }, &.{ 0, 2 }));
}
