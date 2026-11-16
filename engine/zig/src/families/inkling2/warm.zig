//! The load-time warm-up: one forward of every chunk width a request can run (prompt chunks 1 .. max_rows with the
//! one-row head and one of max_rows + 1 when prompt chunks are wider, verify windows 2 .. 8 with the batched head, a
//! rewind, the draft's absorb of every width and one draft), so the first request pays no cuBLASLt plan creation and
//! no first launches. Every rank runs it (it holds the forward's collectives). The sequence is reset after it and the
//! demand zeroed: nothing of it reaches a request (the caches are positional, the rings zeroed by reset, the draft's
//! context positional, its f_rnn set by the next absorb), and the plans it makes are the ones the first use would make
//! (one plan a shape, kept).
const std = @import("std");
const model_mod = @import("model.zig");
const forward = @import("forward.zig");
const dspark = @import("dspark.zig");

const Model = model_mod.Model;

pub fn prewarm(m: *Model, draft: ?*dspark.Draft) !void {
    const was_exact = m.exact;
    m.exact = false;
    defer m.exact = was_exact;
    var ids: [64]i32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast((1000 + 37 * i) % m.c.unpadded_vocab); // ordinary vocabulary ids
    const R = @min(m.max_rows, ids.len);
    try m.reset();
    // prompt chunks: every width, the last row's head (a .none chunk runs the same products before the head)
    for (1..R + 1) |T| {
        if (m.state.len + T + dspark.block > m.context) try m.reset();
        const at = m.state.len;
        _ = try forward.chunk(m, ids[0..T], .last, null, null);
        if (draft) |d| try d.absorb(m.spec.?.taps.ptr, at, T);
    }
    // a wide prompt chunk (Model.prompt_rows past max_rows) of max_rows + 1 rows: the wide path's own kernels and its
    // NCCL point-to-point connections made now (a wider chunk's products make their plans at first use, milliseconds)
    const wide_rows = m.max_rows + 1;
    if (m.prompt_rows > R and wide_rows + dspark.block <= m.context) {
        try m.reset();
        var wide: [64 + 1]i32 = undefined;
        for (wide[0..wide_rows], 0..) |*t, i| t.* = @intCast((1000 + 37 * i) % m.c.unpadded_vocab);
        _ = try forward.chunk(m, wide[0..wide_rows], .last, null, null);
        if (draft) |d| try d.absorb(m.spec.?.taps.ptr, 0, wide_rows);
        try m.reset();
    }
    // verify windows: every row's argmax through the batched head; a rewind to the first row (e-step: up to the graphed
    // verify width, Graphs.rows: 8 = dspark.block unless TF_INKLING2_COPIES widens copy windows to 16)
    if (draft) |d| {
        var am: [@import("graphs.zig").max_rows]i32 = undefined;
        for (2..@min(@max(dspark.block, m.graphs.rows), R) + 1) |T| {
            if (m.state.len + T + dspark.block > m.context) try m.reset();
            const base = m.state.len;
            _ = try forward.chunk(m, ids[0..T], .all, null, am[0..T]);
            try m.rewind(base, 1);
            try d.absorb(m.spec.?.taps.ptr, base, 1);
        }
        var out: [dspark.spec]i32 = undefined;
        if (m.state.len + dspark.block + 1 > m.context) try m.reset();
        try d.draft(ids[0], m.state.len, &out);
    }
    try m.reset();
    try m.x.demand.fill8(0, m.s.handle);
    try m.s.synchronize();
}

// ---- e-step: verify windows, the draft and the commit timed at load (TF_INKLING2_PRICES=timed; serve/engine.zig)
// What the lane core prices a round by (lanes.config.Model.window_costs, the depth rule and the copy allocator), as
// GLM's native family times it (docs/recipes/glm-5.3-flash-cuda.md "Window prices", lanes.cost_rule): a window of w
// rows over real text (its rows route as a reply's do, unlike a repeated token's) from its step to its argmaxes,
// then one row kept, as a served round keeps rows; each width the fastest of cost_rule.reps runs after an untimed
// one. Every rank runs the same sequence (the forwards hold collectives); rank 0's clock is the one served. Nothing
// of it reaches a request: the sequence is reset after it and the demand zeroed, as prewarm leaves them.

/// Widths a table holds at most (engine.max_rows).
pub const max_widths = 16;

/// A load's measured prices (kept in lanes.cost_cache by build, store and settings): ms[w - 1] a verify forward of w
/// rows; draft_ms one DSpark pass (7 drafts, whatever the depth); commit_ms a round's commit (rewind + the kept rows'
/// absorb into the draft's context).
pub const Prices = extern struct {
    ms: [max_widths]f64 = @splat(0),
    rows: u64 = 0,
    draft_ms: f64 = 0,
    commit_ms: f64 = 0,

    /// Whether a table read back holds `rows` widths of finite positive prices.
    pub fn usable(p: Prices, rows: usize) bool {
        if (p.rows != rows or rows == 0 or rows > max_widths) return false;
        for (p.ms[0..rows]) |v| if (!(v > 0) or !std.math.isFinite(v)) return false;
        return p.draft_ms > 0 and std.math.isFinite(p.draft_ms) and p.commit_ms >= 0 and std.math.isFinite(p.commit_ms);
    }
};

/// Timed runs a price keeps the fastest of, after one untimed (lanes.cost_rule.reps).
const reps = 7;
/// The text's tokens before the timed windows (a prompt chunk), and the positions a sequence runs before it restarts.
const prefix = 16;
const room = 512;

const Timer = struct {
    m: *Model,
    d: *dspark.Draft,
    text: []const i32,
    at: usize = prefix, // the next window's pending token in the text (which repeats)
    am: [max_widths]i32 = undefined,

    fn token(t: *const Timer, r: usize) i32 {
        return t.text[(t.at + r) % t.text.len];
    }

    /// The sequence from position 0: the text's first `prefix` tokens as a prompt chunk, its taps absorbed.
    fn begin(t: *Timer) !void {
        try t.m.reset();
        _ = try forward.chunk(t.m, t.text[0..prefix], .last, null, null);
        try t.d.absorb(t.m.spec.?.taps.ptr, 0, prefix);
        t.at = prefix;
    }

    fn roomFor(t: *Timer, rows: usize) !void {
        const end = t.m.state.len + rows + dspark.block + 2;
        if (end > @min(t.m.context, room)) try t.begin();
    }

    /// One verify window of `rows` rows from its launch to its argmaxes (ms), then its first row kept.
    fn window(t: *Timer, rows: usize) !f64 {
        try t.roomFor(rows);
        var ids: [max_widths]i32 = undefined;
        for (ids[0..rows], 0..) |*x, r| x.* = t.token(r);
        const base = t.m.state.len;
        try t.m.s.synchronize(); // the last round's work is not this window's
        const t0 = @import("graphs.zig").now();
        _ = try forward.chunk(t.m, ids[0..rows], .all, null, t.am[0..rows]); // reads the argmaxes back (synchronized)
        const ns = @import("graphs.zig").now() -| t0;
        if (rows > 1) try t.m.rewind(base, 1);
        try t.d.absorb(t.m.spec.?.taps.ptr, base, 1);
        t.at += 1;
        return @as(f64, @floatFromInt(ns)) / 1e6;
    }

    /// One DSpark pass from the next token (ms; draft() reads its tokens back).
    fn draftPass(t: *Timer) !f64 {
        try t.roomFor(1);
        var out: [dspark.spec]i32 = undefined;
        try t.m.s.synchronize();
        const t0 = @import("graphs.zig").now();
        try t.d.draft(t.token(0), t.m.state.len, &out);
        return @as(f64, @floatFromInt(@import("graphs.zig").now() -| t0)) / 1e6;
    }

    /// A round's commit after an 8-row window: 4 rows kept (the rewind) and absorbed into the draft's context (ms).
    fn commit(t: *Timer) !f64 {
        const rows = @min(dspark.block, t.m.max_rows);
        try t.roomFor(rows);
        var ids: [max_widths]i32 = undefined;
        for (ids[0..rows], 0..) |*x, r| x.* = t.token(r);
        const base = t.m.state.len;
        _ = try forward.chunk(t.m, ids[0..rows], .all, null, t.am[0..rows]);
        const t0 = @import("graphs.zig").now();
        try t.m.rewind(base, rows / 2);
        try t.d.absorb(t.m.spec.?.taps.ptr, base, rows / 2);
        try t.m.s.synchronize();
        const ns = @import("graphs.zig").now() -| t0;
        t.at += rows / 2;
        return @as(f64, @floatFromInt(ns)) / 1e6;
    }

    const What = enum { window, draft, commit };

    fn once(t: *Timer, comptime what: What, rows: usize) !f64 {
        return switch (what) {
            .window => t.window(rows),
            .draft => t.draftPass(),
            .commit => t.commit(),
        };
    }

    /// The fastest of `reps` runs after an untimed one (lanes.cost_rule.fastest).
    fn fastest(t: *Timer, comptime what: What, rows: usize) !f64 {
        _ = try t.once(what, rows);
        var best = std.math.inf(f64);
        for (0..reps) |_| best = @min(best, try t.once(what, rows));
        return best;
    }
};

/// Verify windows of 1 .. `rows` rows over `text` (token ids, at least prefix + 1), the draft and the commit, timed
/// on every rank (TF_INKLING2_PRICES=timed); a window's price never falls as its rows grow (a dip is noise).
pub fn timeWindows(m: *Model, draft: *dspark.Draft, text: []const i32, rows: usize) !Prices {
    if (rows == 0 or rows > max_widths or rows > m.max_rows or text.len <= prefix) return error.PriceShape;
    const was_exact = m.exact;
    m.exact = false;
    defer m.exact = was_exact;
    var t: Timer = .{ .m = m, .d = draft, .text = text };
    try t.begin();
    var p: Prices = .{ .rows = rows };
    for (0..rows) |i| p.ms[i] = try t.fastest(.window, i + 1);
    for (1..rows) |i| p.ms[i] = @max(p.ms[i], p.ms[i - 1]);
    p.draft_ms = try t.fastest(.draft, 0);
    p.commit_ms = try t.fastest(.commit, 0);
    try m.reset();
    try m.x.demand.fill8(0, m.s.handle);
    try m.s.synchronize();
    return p;
}
// ---- end e-step
