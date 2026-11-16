//! xs (sampled decode): a sampled row's token as lanes.gpu_full.draw draws it, without the work that cannot change it.
//! The lane core's draw sorts the whole vocabulary (200,058 values) for every row, then races every token within 24
//! of the row's best (value - log(-log(u)), u keyed by seed, position and token): ~11 ms a row on the host, 30-43 ms
//! of a sampled round. The race's noise is a pure function of (seed, position, token), so the token is the race's
//! argmax over the kept set whatever order the tokens are scored in, and three facts let most of the work go:
//!   - with top_k off (or past the vocabulary) and top_p off the sort is never read: only the row's max is (the kept
//!     set is every token at or above max(m + ln(min_p), m - 24));
//!   - a token can add at most noise_max to its value (-log(-log(below_one)) = 16.6355), so one whose value is below
//!     the best score found so far less noise_max (and a rounding margin) can never win nor tie: it is not scored;
//!   - where the order matters (top_k 1..1024: gpu_rule.sample over the first top_k; top_p: the nucleus over the
//!     first 1,024), only that prefix of the order is needed: it is selected exactly (value descending, id ascending,
//!     gpu_rule.better) instead of sorting all of it.
//! Everything that decides the token is the lane core's own arithmetic, in its order: v = l * inv_t, gpu_rule.key, the
//! keyed uniforms, the scores, gpu_full's tie rule, mass()'s 1024-lane sums (copied here: it is private there),
//! gpu_rule.sample's sums, cut and race. So the token is lanes.gpu_full.draw's for every seed and position (the same
//! RNG draws: each scored token's own u). The rest (top_k between 1,024 and the vocabulary, a nucleus past 1,024 tokens)
//! falls back to lanes.gpu_full.draw itself.
const std = @import("std");
const lanes = @import("lanes");

const full = lanes.gpu_full;
const rule = lanes.gpu_rule;
const Cand = rule.Cand;
const Mark = full.Mark;
const Sampling = lanes.Sampling;

/// Above the largest noise a raced token can draw: -log(-log(below_one)) = 16.6355 (u kept below 1 by raceUniform).
const noise_max: f32 = 16.64;
/// The first race's tokens: a high best score before the pass that prunes below it.
const seed_k = 64;

/// The candidates' buffer, kept between rows (no allocation a row).
pub const Scratch = struct {
    gpa: std.mem.Allocator,
    buf: std.ArrayList(Cand) = .empty,
    /// draws that fell back to lanes.gpu_full.draw (top_k between 1,024 and the vocabulary, a nucleus past 1,024)
    fallbacks: u64 = 0,

    pub fn deinit(sc: *Scratch) void {
        sc.buf.deinit(sc.gpa);
    }
};

/// The token lanes.gpu_full.draw(logits, s, position, null) draws; error.NonFiniteLogits as the backend's check.
pub fn draw(sc: *Scratch, logits: []const f32, s: Sampling, position: u32) !u32 {
    const r = rule.rule(s);
    const inv_t = r.inv_t;
    var m = -std.math.inf(f32);
    for (logits) |l| {
        if (!std.math.isFinite(l)) return error.NonFiniteLogits;
        m = @max(m, l * inv_t);
    }
    if (!full.fullVocabulary(s)) return ruleSample(sc, logits, s, r, m, position);
    if (s.top_k != 0 and s.top_k < logits.len) return fallback(sc, logits, s, position);
    const p = full.Plan{ .inv_t = inv_t, .m = m, .lo = @max(m + r.min_log, m - full.never) };
    if (r.top_p > 0.0 and r.top_p < 1.0) return nucleusRace(sc, logits, s, p, r.top_p, position);
    return prunedRace(sc, logits, p, s.seed, position);
}

fn fallback(sc: *Scratch, logits: []const f32, s: Sampling, position: u32) !u32 {
    sc.fallbacks += 1;
    return full.draw(sc.gpa, logits, s, position, null);
}

/// gpu_rule.sample's draw (top_k 1 .. 1,024) on the exact first n of the order: its sums, cut, floor and race.
fn ruleSample(sc: *Scratch, logits: []const f32, s: Sampling, r: rule.Rule, m: f32, position: u32) !u32 {
    const kc = r.top_k;
    const cap: usize = if (kc == 0 or kc > rule.candidates) rule.candidates else kc;
    const n: usize = @min(@min(logits.len, rule.candidates), cap);
    const all = try first(sc, logits, r.inv_t, n);
    var norm: f32 = 0.0; // kc != 0 here: the kept candidates' own sum (the full one, threadSum, is unused)
    for (all[0..n]) |c| norm += @exp(c.v - m);
    var keep: usize = n;
    if (r.top_p > 0.0 and r.top_p < 1.0) {
        var cum: f32 = 0.0;
        for (all[0..n], 0..) |c, j| {
            cum += @exp(c.v - m) / norm;
            if (cum >= r.top_p) {
                keep = j + 1;
                break;
            }
        }
    }
    if (r.min_log > -std.math.inf(f32)) {
        const floor = m + r.min_log;
        var j: usize = 0;
        while (j < keep and all[j].v >= floor) j += 1;
        keep = j;
    }
    var best: usize = 0;
    var best_score = -std.math.inf(f32);
    for (all[0..keep], 0..) |c, t| {
        const score = c.v - @log(-@log(rule.uniform24(s.seed, position, c.i)));
        if (t == 0 or score > best_score) {
            best = t;
            best_score = score;
        }
    }
    return all[best].i;
}

/// gpu_full's nucleus (top_p) and race: the norm over every token (its mass(), copied), the cut found on the exact
/// first 1,024 of the order, the race over the tokens up to the cut; a nucleus that needs more: the lane core's draw.
fn nucleusRace(sc: *Scratch, logits: []const f32, s: Sampling, p: full.Plan, top_p: f32, position: u32) !u32 {
    const norm = mass(logits, p, 1.0, -std.math.inf(f32), start, full.end, full.end, false);
    const n = @min(logits.len, rule.candidates);
    const order = try first(sc, logits, p.inv_t, n);
    var cum: f32 = 0.0;
    const cut: usize = for (order[0..n], 0..) |c, j| {
        cum += @exp(c.v - p.m) / norm;
        if (cum >= top_p) break j;
    } else return fallback(sc, logits, s, position); // extend(): past the 1,024 candidates
    var bs = -std.math.inf(f32);
    var bk: u32 = 0;
    var bi: u32 = 0xFFFF_FFFF;
    for (order[0 .. cut + 1]) |c| {
        if (!(c.v >= p.lo)) continue;
        const k = rule.key(c.v);
        const score = c.v - @log(-@log(full.raceUniform(s.seed, position, c.i)));
        if (beats(score, k, c.i, bs, bk, bi)) {
            bs = score;
            bk = k;
            bi = c.i;
        }
    }
    return @min(bi, @as(u32, @intCast(logits.len - 1)));
}

/// gpu_full.race with no top_k and no top_p (the kept set: v >= lo), scoring only tokens that can still win: the best
/// of the first seed_k tokens of the order gives a score B; a token below B - noise_max can only score below B (the
/// margin covers the scores' rounding), and B only grows, so the threshold follows the best score found.
fn prunedRace(sc: *Scratch, logits: []const f32, p: full.Plan, seed: u64, position: u32) !u32 {
    var bs = -std.math.inf(f32);
    var bk: u32 = 0;
    var bi: u32 = 0xFFFF_FFFF;
    const head = try first(sc, logits, p.inv_t, @min(logits.len, seed_k));
    for (head) |c| {
        if (!(c.v >= p.lo)) continue;
        const k = rule.key(c.v);
        const score = c.v - @log(-@log(full.raceUniform(seed, position, c.i)));
        if (beats(score, k, c.i, bs, bk, bi)) {
            bs = score;
            bk = k;
            bi = c.i;
        }
    }
    var th = below(bs);
    for (logits, 0..) |l, i| {
        const v = l * p.inv_t;
        if (!(v >= p.lo) or v < th) continue;
        const k = rule.key(v);
        const id: u32 = @intCast(i);
        const score = v - @log(-@log(full.raceUniform(seed, position, id)));
        if (beats(score, k, id, bs, bk, bi)) {
            bs = score;
            bk = k;
            bi = id;
            th = below(bs);
        }
    }
    return @min(bi, @as(u32, @intCast(logits.len - 1)));
}

/// The value under which a token's score stays below `b` whatever its noise (with a margin for the f32 rounding).
fn below(b: f32) f32 {
    return b - noise_max - 1e-3 * @max(1.0, @abs(b));
}

/// The first n tokens of the order (gpu_rule.better: value descending, then id ascending), exactly: tokens pass into a
/// buffer while they beat the n-th best kept so far; a full buffer is sorted and cut back to its n best.
fn first(sc: *Scratch, logits: []const f32, inv_t: f32, n: usize) ![]const Cand {
    const room = @max(4 * n, 256);
    sc.buf.clearRetainingCapacity();
    try sc.buf.ensureTotalCapacity(sc.gpa, room);
    var bar: ?Cand = null; // the n-th best after the last cut: a token that does not beat it is outside the first n
    for (logits, 0..) |l, i| {
        const c: Cand = .{ .v = l * inv_t, .i = @intCast(i) };
        if (bar) |b| if (!rule.better({}, c, b)) continue;
        sc.buf.appendAssumeCapacity(c);
        if (sc.buf.items.len == room) {
            std.mem.sort(Cand, sc.buf.items, {}, rule.better);
            sc.buf.shrinkRetainingCapacity(n);
            bar = sc.buf.items[n - 1];
        }
    }
    std.mem.sort(Cand, sc.buf.items, {}, rule.better);
    return sc.buf.items[0..@min(n, sc.buf.items.len)];
}

// ---- gpu_full's private helpers, copied (the same expressions, so the same bits)

const start = Mark{ .key = 0xFFFF_FFFF, .id = 0xFFFF_FFFF };

fn atOrBefore(k: u32, i: u32, m: Mark) bool {
    return k > m.key or (k == m.key and i <= m.id);
}

fn after(k: u32, i: u32, m: Mark) bool {
    return k < m.key or (k == m.key and i > m.id);
}

fn beats(s: f32, k: u32, i: u32, bs: f32, bk: u32, bi: u32) bool {
    return s > bs or (s == bs and (k > bk or (k == bk and i < bi)));
}

/// gpu_full.mass: exp(v - m) / norm over tokens at or above `lo`, after `b`, inside `top` and `cut`, in its order.
fn mass(logits: []const f32, p: full.Plan, norm: f32, lo: f32, b: Mark, top: Mark, cut: Mark, strict: bool) f32 {
    var acc: [1024]f32 = @splat(0.0);
    for (logits, 0..) |l, i| {
        const v = l * p.inv_t;
        if (!(v >= lo)) continue;
        const k = rule.key(v);
        const id: u32 = @intCast(i);
        const inside = if (strict) k > cut.key or (k == cut.key and id < cut.id) else atOrBefore(k, id, cut);
        if (after(k, id, b) and atOrBefore(k, id, top) and inside) acc[i % 1024] += @exp(v - p.m) / norm;
    }
    return rule.lanesSum(&acc);
}

// ---- tests: the token of lanes.gpu_full.draw, row by row, over seeds, positions and settings

fn testRow(a: std.mem.Allocator, n: usize, seed: u64, spread: f32) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const row = try a.alloc(f32, n);
    for (row) |*x| x.* = (r.float(f32) * 2 - 1) * spread + if (r.uintLessThan(u32, 50) == 0) r.float(f32) * 2 * spread else 0;
    return row;
}

test "the fast draw is the lane core's draw: every setting, seeds and positions" {
    const a = std.testing.allocator;
    var sc: Scratch = .{ .gpa = a };
    defer sc.deinit();
    const settings = [_]Sampling{
        .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 0, .temperature = 0.7, .top_k = 0, .top_p = 1.0, .min_p = 0.05 },
        .{ .seed = 0, .temperature = 1.3, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 20, .top_p = 1.0 },
        .{ .seed = 0, .temperature = 0.6, .top_k = 40, .top_p = 0.95 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 1024, .top_p = 1.0, .min_p = 0.02 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 0.95 },
        .{ .seed = 0, .temperature = 0.8, .top_k = 0, .top_p = 0.9, .min_p = 0.01 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 3000, .top_p = 1.0 },
    };
    for ([_]f32{ 3, 12, 30 }, 0..) |spread, ri| {
        const row = try testRow(a, 20_011, 77 + ri, spread);
        defer a.free(row);
        for (settings) |base| for (0..8) |j| {
            var st = base;
            st.seed = 0x9E37_79B9 *% (j + 1) +% ri;
            const pos: u32 = @intCast(1000 + 37 * j);
            const want = try full.draw(a, row, st, pos, null);
            const got = try draw(&sc, row, st, pos);
            try std.testing.expectEqual(want, got);
        };
    }
}

// TF_INKLING2_DRAW_LOGITS=<file of fp32 rows> (TF_INKLING2_DRAW_VOCAB columns, default Inkling's 200,058; up to
// TF_INKLING2_DRAW_ROWS rows, default 64): every row under every setting at 4 seeds, the token against
// lanes.gpu_full.draw's and both timed (skipped without the file).
test "real rows: the lane core's token, and the time a row" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const path = std.mem.span(std.c.getenv("TF_INKLING2_DRAW_LOGITS") orelse return error.SkipZigTest);
    const vocab: usize = if (std.c.getenv("TF_INKLING2_DRAW_VOCAB")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 200_058;
    const want_rows: usize = if (std.c.getenv("TF_INKLING2_DRAW_ROWS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 64;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const total: usize = @intCast(try file.length(io) / (vocab * 4));
    const rows = @min(total, want_rows);
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    var sc: Scratch = .{ .gpa = a };
    defer sc.deinit();
    const settings = [_]Sampling{
        .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 0, .temperature = 0.7, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.05 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 40, .top_p = 0.95 },
        .{ .seed = 0, .temperature = 1.0, .top_k = 0, .top_p = 0.95 },
    };
    var ref_ns: [settings.len]u64 = @splat(0);
    var fast_ns: [settings.len]u64 = @splat(0);
    var n: usize = 0;
    for (0..rows) |ri| {
        const at = (ri * total / rows) * vocab * 4;
        if (try file.readPositionalAll(io, std.mem.sliceAsBytes(row), at) != vocab * 4) return error.ShortRead;
        for (settings, 0..) |base, si| for (0..4) |j| {
            var st = base;
            st.seed = (0xD1B5_4A32_D192_ED03 *% (ri * 4 + j + 1)) & ((@as(u64, 1) << 63) - 1);
            const pos: u32 = @intCast(5000 + 97 * ri + j);
            const t0 = std.Io.Clock.awake.now(io);
            const want = try full.draw(a, row, st, pos, null);
            const t1 = std.Io.Clock.awake.now(io);
            const got = try draw(&sc, row, st, pos);
            const t2 = std.Io.Clock.awake.now(io);
            ref_ns[si] += @intCast(t0.durationTo(t1).toNanoseconds());
            fast_ns[si] += @intCast(t1.durationTo(t2).toNanoseconds());
            if (want != got) std.debug.print("row {d} setting {d} seed {d} pos {d}: lane core {d}, draw.zig {d}\n", .{ ri, si, st.seed, pos, want, got });
            try std.testing.expectEqual(want, got);
            n += 1;
        };
    }
    for (settings, ref_ns, fast_ns) |st, r, f| std.debug.print("[draw] T {d:.1} top_k {d} top_p {d:.2} min_p {d:.2}: lane core {d:.3} ms a row, draw.zig {d:.3} ms a row ({d} rows x 4 seeds)\n", .{ st.temperature, st.top_k, st.top_p, st.min_p, @as(f64, @floatFromInt(r)) / 1e6 / @as(f64, @floatFromInt(rows * 4)), @as(f64, @floatFromInt(f)) / 1e6 / @as(f64, @floatFromInt(rows * 4)), rows });
    std.debug.print("[draw] {d} draws, every token the lane core's; {d} fell back\n", .{ n, sc.fallbacks });
}

test "non-finite logits are refused as the backend refused them" {
    const a = std.testing.allocator;
    var sc: Scratch = .{ .gpa = a };
    defer sc.deinit();
    var row = [_]f32{ 1, 2, std.math.nan(f32), 0 };
    try std.testing.expectError(error.NonFiniteLogits, draw(&sc, &row, .{ .seed = 1, .temperature = 1, .top_k = 0, .top_p = 1 }, 3));
}
