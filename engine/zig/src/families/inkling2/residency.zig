//! TideLM's adaptive expert residency (tidelm-core/src/residency.rs, docs/adaptive-experts.md), ported for Inkling-2:
//! which (layer, expert) pairs hold arena slots, from a prior and the router's observed demand.
//!
//!   initial_selection  the boot fill: top-k per layer (the routing floor), then one ranking of every remaining
//!                      (layer, expert) across layers for the rest of the budget. Ties: lower layer, then lower id.
//!   global_plan        a pass: the highest-value absentee against the lowest-value eligible resident, across layers
//!                      (a donor to another layer must hold more than the floor), at most `max_loads` swaps, a swap
//!                      only above the gain floor. Ties as TideLM's (stable ids).
//!   blend_demand       per layer: w = n / (n + prior votes), scores = (1 - w) prior + w observed / n; the score
//!                      metric (sum of the router's sigmoid scores over its unmasked top-k), request-normalized votes.
//!
//! One difference, from the checkpoint: Inkling stores layers 0-39 NVFP4 (8.96 MB a slot a rank at TP2) and 40-47
//! MXFP8 (16.4 MB), so the arena is two pools and a slot moves between layers of one format only. Every rule above
//! runs per pool, on scores in shared (per-layer-normalized) units.
//!
//! All of it is deterministic (ties by stable ids, no hash order): rank 0 plans and both ranks apply its plan.
const std = @import("std");
const config = @import("config.zig");

pub const Policy = struct {
    /// Demand half-life in votes (TideLM `high`: 10 votes, the newest vote 6.7% of observed demand).
    half_life: f64 = 10,
    /// The prior's weight in votes (DSV41_ADAPT_PRIOR default 4).
    prior_votes: f64 = 4,
    /// A prefill or decode pass needs at least this miss rate (`high`: 1%).
    miss_gate: f64 = 0.01,
    max_loads: usize = 512,
    /// The gain floor, a share of the pool's mean score.
    gain_frac: f64 = 0.005,
    /// A decode pass every this many output tokens (0: none).
    decode_tokens: usize = 600,
    /// The prefill->decode pass needs a prompt of at least this many new tokens.
    min_prompt: usize = 32,
    /// Off: the placement is frozen (TideLM sensitivity `off`).
    adapt: bool = true,

    pub fn decay(p: Policy) f64 {
        return std.math.pow(f64, 2.0, -1.0 / p.half_life);
    }
};

/// One slot move: the resident (lo, eo) gives its slot to the absent (li, ei).
pub const Swap = struct { lo: u32, eo: u32, li: u32, ei: u32, gain: f64 };

/// The demand a forward records per (layer, expert): [0] unmasked top-k picks, [1] their sigmoid score mass,
/// [2] picks the masked router could not take (substituted). fp32, [L][E][3].
pub const demand_fields = 3;

pub const Residency = struct {
    gpa: std.mem.Allocator,
    layers: usize,
    experts: usize,
    floor: usize,
    policy: Policy,
    prior: []f64, // [L][E], each layer summing to 1
    hist: []f64, // [L][E] decayed sum of request-normalized votes
    n: []f64, // [L] decayed vote weight (the evidence)
    votes: usize = 0,

    pub fn init(gpa: std.mem.Allocator, c: *const config.Config, policy: Policy, prior: []const f32) !Residency {
        const L: usize = c.layers;
        const E: usize = c.experts;
        if (prior.len != L * E) return error.PriorShape;
        const r: Residency = .{ .gpa = gpa, .layers = L, .experts = E, .floor = c.topk, .policy = policy, .prior = try gpa.alloc(f64, L * E), .hist = try gpa.alloc(f64, L * E), .n = try gpa.alloc(f64, L) };
        for (0..L) |l| {
            var sum: f64 = 0;
            for (prior[l * E ..][0..E]) |v| {
                if (!std.math.isFinite(v) or v < 0) return error.PriorValue;
                sum += v;
            }
            for (r.prior[l * E ..][0..E], prior[l * E ..][0..E]) |*o, v| o.* = if (sum > 0) v / sum else 1.0 / @as(f64, @floatFromInt(E));
        }
        @memset(r.hist, 0);
        @memset(r.n, 0);
        return r;
    }

    pub fn deinit(r: *Residency) void {
        r.gpa.free(r.prior);
        r.gpa.free(r.hist);
        r.gpa.free(r.n);
    }

    /// A request's demand normalized per layer (its score mass summing to 1), into `out` [L][E]; layers it never
    /// routed stay zero (and `has[l]` false).
    fn normalized(r: *const Residency, demand: []const f32, out: []f64, has: []bool) void {
        const E = r.experts;
        for (0..r.layers) |l| {
            var sum: f64 = 0;
            for (0..E) |e| sum += demand[(l * E + e) * demand_fields + 1];
            has[l] = sum > 0;
            for (0..E) |e| out[l * E + e] = if (sum > 0) demand[(l * E + e) * demand_fields + 1] / sum else 0;
        }
    }

    /// One vote: the history ages one step and takes the request's normalized demand.
    pub fn vote(r: *Residency, demand: []const f32) !void {
        const E = r.experts;
        const d = r.policy.decay();
        const v = try r.gpa.alloc(f64, r.layers * E);
        defer r.gpa.free(v);
        const has = try r.gpa.alloc(bool, r.layers);
        defer r.gpa.free(has);
        r.normalized(demand, v, has);
        for (0..r.layers) |l| {
            for (r.hist[l * E ..][0..E], v[l * E ..][0..E]) |*h, x| h.* = h.* * d + x;
            r.n[l] = r.n[l] * d + (if (has[l]) @as(f64, 1) else 0);
        }
        r.votes += 1;
    }

    /// x-mixed: `vote` weighted `w` (the mixed store's native bank: a decode tick 1, a prompt TF_INKLING2_MIX_PROMPT_WEIGHT):
    /// the history ages one step and takes w times the stretch's demand, normalized per layer; a layer's evidence n
    /// grows by w when it routed. `vote` is voteWeighted(demand, 1) bit for bit (x * 1 is x).
    pub fn voteWeighted(r: *Residency, demand: []const f32, w: f64) !void {
        const E = r.experts;
        const d = r.policy.decay();
        const v = try r.gpa.alloc(f64, r.layers * E);
        defer r.gpa.free(v);
        const has = try r.gpa.alloc(bool, r.layers);
        defer r.gpa.free(has);
        r.normalized(demand, v, has);
        for (0..r.layers) |l| {
            for (r.hist[l * E ..][0..E], v[l * E ..][0..E]) |*h, x| h.* = h.* * d + w * x;
            r.n[l] = r.n[l] * d + (if (has[l]) w else 0);
        }
        r.votes += 1;
    }

    /// blend_demand into `out` [L][E]; `pending` (a request's demand so far) is blended as one more vote without
    /// being committed (TideLM's decode pass). Returns the mean observed weight.
    pub fn scores(r: *const Residency, out: []f64, pending: ?[]const f32) !f64 {
        const E = r.experts;
        const d = r.policy.decay();
        var v: ?[]f64 = null;
        defer if (v) |x| r.gpa.free(x);
        const has = try r.gpa.alloc(bool, r.layers);
        defer r.gpa.free(has);
        if (pending) |p| {
            v = try r.gpa.alloc(f64, r.layers * E);
            r.normalized(p, v.?, has);
        }
        var wsum: f64 = 0;
        for (0..r.layers) |l| {
            const obs_n = if (v != null) r.n[l] * d + (if (has[l]) @as(f64, 1) else 0) else r.n[l];
            var n: f64 = 0;
            for (0..E) |e| n += obs(r, v, d, l, e);
            const w = if (n > 0) obs_n / (obs_n + r.policy.prior_votes) else 0;
            wsum += w;
            for (0..E) |e| out[l * E + e] = (1 - w) * r.prior[l * E + e] + w * (if (n > 0) obs(r, v, d, l, e) / n else 0);
        }
        return wsum / @as(f64, @floatFromInt(r.layers));
    }

    fn obs(r: *const Residency, v: ?[]f64, d: f64, l: usize, e: usize) f64 {
        const i = l * r.experts + e;
        return if (v) |x| r.hist[i] * d + x[i] else r.hist[i];
    }

    /// initial_selection over the layers in `pool`: (layer, expert) pairs, the floor's first (layer order, best
    /// first), then the global ranking's. `budget` must hold the floor of every pool layer and at most all experts.
    pub fn initial(r: *const Residency, gpa: std.mem.Allocator, sc: []const f64, pool: []const u32, budget: usize) ![][2]u32 {
        const E = r.experts;
        if (!(r.floor * pool.len <= budget and budget <= E * pool.len)) return error.BudgetBelowFloor;
        var out: std.ArrayList([2]u32) = .empty;
        errdefer out.deinit(gpa);
        const Extra = struct { s: f64, l: u32, e: u32 };
        var extras: std.ArrayList(Extra) = .empty;
        defer extras.deinit(gpa);
        const order = try gpa.alloc(u32, E);
        defer gpa.free(order);
        for (pool) |l| {
            topIndices(sc[l * E ..][0..E], order);
            for (order[0..r.floor]) |e| try out.append(gpa, .{ l, e });
            for (order[r.floor..]) |e| try extras.append(gpa, .{ .s = sc[l * E + e], .l = l, .e = e });
        }
        std.mem.sort(Extra, extras.items, {}, struct {
            fn lt(_: void, a: Extra, b: Extra) bool {
                if (a.s != b.s) return a.s > b.s;
                if (a.l != b.l) return a.l < b.l;
                return a.e < b.e;
            }
        }.lt);
        for (extras.items[0 .. budget - r.floor * pool.len]) |x| try out.append(gpa, .{ x.l, x.e });
        return out.toOwnedSlice(gpa);
    }

    /// The uniform selection (TideLM build_keep_masks Uniform; the residency plan's quality probe): the top `per`
    /// of every pool layer by score.
    pub fn uniform(r: *const Residency, gpa: std.mem.Allocator, sc: []const f64, pool: []const u32, per: usize) ![][2]u32 {
        const E = r.experts;
        if (per < r.floor or per > E) return error.BudgetBelowFloor;
        var out: std.ArrayList([2]u32) = .empty;
        errdefer out.deinit(gpa);
        const order = try gpa.alloc(u32, E);
        defer gpa.free(order);
        for (pool) |l| {
            topIndices(sc[l * E ..][0..E], order);
            for (order[0..per]) |e| try out.append(gpa, .{ l, e });
        }
        return out.toOwnedSlice(gpa);
    }

    /// global_plan over the pool's layers: `keep` [L][E] (1 resident), scores `sc` [L][E]. min_gain is the policy's
    /// share of the pool's mean score.
    pub fn plan(r: *const Residency, gpa: std.mem.Allocator, keep: []const u8, sc: []const f64, pool: []const u32) ![]Swap {
        const E = r.experts;
        const Cand = struct { s: f64, tie: f64, l: u32, e: u32 };
        var donors: std.ArrayList(Cand) = .empty;
        defer donors.deinit(gpa);
        var incoming: std.ArrayList(Cand) = .empty;
        defer incoming.deinit(gpa);
        const counts = try gpa.alloc(usize, r.layers);
        defer gpa.free(counts);
        @memset(counts, 0);
        var mean: f64 = 0;
        for (pool) |l| {
            for (0..E) |e| {
                const s = sc[l * E + e];
                if (!std.math.isFinite(s) or s < 0) return error.BadScores;
                mean += s;
                if (keep[l * E + e] != 0) {
                    counts[l] += 1;
                    try donors.append(gpa, .{ .s = s, .tie = s, .l = l, .e = @intCast(e) });
                } else if (s > 0) {
                    try incoming.append(gpa, .{ .s = s, .tie = s, .l = l, .e = @intCast(e) });
                }
            }
            if (counts[l] < r.floor) return error.BelowFloor;
        }
        mean /= @as(f64, @floatFromInt(pool.len * E));
        const min_gain = r.policy.gain_frac * mean;
        // donors ascending as tuples (score, tie, layer, expert); incoming: score and tie descending, ids ascending
        std.mem.sort(Cand, donors.items, {}, struct {
            fn lt(_: void, a: Cand, b: Cand) bool {
                if (a.s != b.s) return a.s < b.s;
                if (a.tie != b.tie) return a.tie < b.tie;
                if (a.l != b.l) return a.l < b.l;
                return a.e < b.e;
            }
        }.lt);
        std.mem.sort(Cand, incoming.items, {}, struct {
            fn lt(_: void, a: Cand, b: Cand) bool {
                if (a.s != b.s) return a.s > b.s;
                if (a.tie != b.tie) return a.tie > b.tie;
                if (a.l != b.l) return a.l < b.l;
                return a.e < b.e;
            }
        }.lt);
        var changes: std.ArrayList(Swap) = .empty;
        errdefer changes.deinit(gpa);
        for (incoming.items) |in| {
            if (changes.items.len >= r.policy.max_loads) break;
            var victim: ?usize = null;
            for (donors.items, 0..) |dn, i| if (dn.l == in.l or counts[dn.l] > r.floor) {
                victim = i;
                break;
            };
            const vi = victim orelse continue;
            const old = donors.items[vi];
            const gain = in.s - old.s;
            if (gain <= min_gain) continue;
            _ = donors.orderedRemove(vi);
            try changes.append(gpa, .{ .lo = old.l, .eo = old.e, .li = in.l, .ei = in.e, .gain = gain });
            counts[old.l] -= 1;
            counts[in.l] += 1;
        }
        return changes.toOwnedSlice(gpa);
    }
};

/// Indices of `values` descending, lower id first on ties (residency.rs top_indices), into `out` (len = values.len).
pub fn topIndices(values: []const f64, out: []u32) void {
    for (out, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, out, values, struct {
        fn lt(v: []const f64, a: u32, b: u32) bool {
            if (v[a] != v[b]) return v[a] > v[b];
            return a < b;
        }
    }.lt);
}

/// Miss rate of a demand stretch: substituted picks / picks (fields 2 and 0), over `now` minus `since` (null: 0).
pub fn missRate(now: []const f32, since: ?[]const f32) struct { picks: f64, missed: f64, rate: f64 } {
    var p: f64 = 0;
    var m: f64 = 0;
    var i: usize = 0;
    while (i < now.len) : (i += demand_fields) {
        p += now[i] - (if (since) |s| s[i] else 0);
        m += now[i + 2] - (if (since) |s| s[i + 2] else 0);
    }
    return .{ .picks = p, .missed = m, .rate = if (p > 0) m / p else 0 };
}

// ---- TideLM's own unit tests (residency.rs `mod tests`), on this port

fn testRes(gpa: std.mem.Allocator, layers: usize, experts: usize, floor: usize) !Residency {
    const r: Residency = .{ .gpa = gpa, .layers = layers, .experts = experts, .floor = floor, .policy = .{}, .prior = try gpa.alloc(f64, layers * experts), .hist = try gpa.alloc(f64, layers * experts), .n = try gpa.alloc(f64, layers) };
    @memset(r.prior, 0);
    @memset(r.hist, 0);
    @memset(r.n, 0);
    return r;
}

test "residency: global_plan moves the highest-value absentee onto the cheapest donor" {
    const gpa = std.testing.allocator;
    var r = try testRes(gpa, 1, 3, 1);
    defer r.deinit();
    r.policy.gain_frac = 0;
    const keep = [_]u8{ 1, 1, 0 };
    const sc = [_]f64{ 5.0, 0.1, 9.0 };
    const sw = try r.plan(gpa, &keep, &sc, &.{0});
    defer gpa.free(sw);
    try std.testing.expectEqual(@as(usize, 1), sw.len);
    try std.testing.expectEqual(Swap{ .lo = 0, .eo = 1, .li = 0, .ei = 2, .gain = sw[0].gain }, sw[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 8.9), sw[0].gain, 1e-9);
}

test "residency: global_plan respects the floor and the gain floor" {
    const gpa = std.testing.allocator;
    var r = try testRes(gpa, 2, 2, 1);
    defer r.deinit();
    r.policy.gain_frac = 0;
    const keep = [_]u8{ 1, 0, 1, 0 };
    const sc = [_]f64{ 0.1, 0.0, 0.2, 9.0 };
    const sw = try r.plan(gpa, &keep, &sc, &.{ 0, 1 });
    defer gpa.free(sw);
    // (1,1) can only displace (1,0): a same-layer replacement at the floor
    try std.testing.expectEqual(@as(usize, 1), sw.len);
    try std.testing.expectEqual(@as(u32, 1), sw[0].lo);
    try std.testing.expectEqual(@as(u32, 0), sw[0].eo);
    try std.testing.expectEqual(@as(u32, 1), sw[0].li);
    try std.testing.expectEqual(@as(u32, 1), sw[0].ei);
    // a gain floor above 8.8 suppresses it (mean score of the pool is 9.3 / 4)
    r.policy.gain_frac = 10.0 / (9.3 / 4.0);
    const none = try r.plan(gpa, &keep, &sc, &.{ 0, 1 });
    defer gpa.free(none);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

test "residency: initial_selection keeps the floor per layer and fills the budget exactly" {
    const gpa = std.testing.allocator;
    const L = 40;
    const E = 384;
    var r = try testRes(gpa, L, E, 6);
    defer r.deinit();
    const sc = try gpa.alloc(f64, L * E);
    defer gpa.free(sc);
    for (0..L) |l| for (0..E) |e| {
        sc[l * E + e] = @floatFromInt(e + l);
    };
    var pool: [L]u32 = undefined;
    for (&pool, 0..) |*p, i| p.* = @intCast(i);
    const sel = try r.initial(gpa, sc, &pool, 9500);
    defer gpa.free(sel);
    try std.testing.expectEqual(@as(usize, 9500), sel.len);
    var per: [L]usize = @splat(0);
    for (sel) |p| per[p[0]] += 1;
    for (per) |n| try std.testing.expect(n >= 6);
    try std.testing.expectError(error.BudgetBelowFloor, r.initial(gpa, sc, &pool, 239));
}

test "residency: a cold history changes nothing; observed demand outvotes the prior with evidence" {
    const gpa = std.testing.allocator;
    var r = try testRes(gpa, 1, 2, 1);
    defer r.deinit();
    r.prior[0] = 0.75;
    r.prior[1] = 0.25;
    var out: [2]f64 = undefined;
    try std.testing.expectEqual(@as(f64, 0), try r.scores(&out, null));
    try std.testing.expectEqual([2]f64{ 0.75, 0.25 }, out);
    // 4 votes of evidence for expert 1 against a prior of 4 votes: w = 1/2
    r.policy.prior_votes = 4;
    r.hist[1] = 4;
    r.n[0] = 4;
    const w = try r.scores(&out, null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), w, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.375), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.625), out[1], 1e-12);
}
