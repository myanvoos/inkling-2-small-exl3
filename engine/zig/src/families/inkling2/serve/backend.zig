//! Inkling-2 behind lanes.Backend on rank 0: one stream at a time, every forward a step every rank runs
//! (engine.zig). With the DSpark draft head (mtp.safetensors, TF_INKLING2_SPEC): the lane core asks for a block of
//! drafts (draft), verifies [pending, drafts] in one forward (verify), and keeps the rows its draws agree with
//! (keep: the rest rolled back, the kept rows' taps into the draft's context). Without it: one-token steps.
//! Tokens are drawn on the host from fp32 softcapped logits by the lane core's own sampler (lanes.gpu_full.draw:
//! argmax without sampling settings; a greedy verify reads only each row's argmax). Residency follows TideLM: the
//! prompt's demand at prefill->decode, a pass every 600 output tokens, the answer's demand at release.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("engine.zig");
const reuse = @import("reuse.zig"); // p4-cache
const draw_mod = @import("draw.zig"); // xs: a sampled row's token without the lane core's full sort (draw.zig)

const be = lanes.backend;
const Stream = lanes.Stream;
const Kind = engine.Kind;

const ring = 1024; // drawn tokens a handle names
const max_held = 7;

/// A verify forward's ms by window rows at TP=2 on two GB10s, steady state, taps on, the 8 GiB floor's arena, the
/// batched head: tf-inkling2-run --bench 1..8 --bench-real 1 (each width over the spec loop's own windows [anchor,
/// drafts], mean of 6 after a warm-up). A row costs ~10 ms more: its picks bring ~3.7 more distinct experts a layer
/// (8 rows: 33.7 a layer). (The plain --bench repeats one token a row: 10 distinct experts at 8 rows, 97 ms; it
/// prices a window ~50 ms too cheap.)
const forward_ms = [_]f64{ 80.0, 90.2, 100.1, 111.3, 119.9, 130.7, 138.9, 147.5 };
/// A DSpark block: one draft pass of all 7 drafts whatever the depth (22.7-23 ms in the run tool, 25-27 ms a step in
/// serving) plus the commit (rewind, absorb, ~1 ms), sent as one step: what a drafted round pays past its forward.
const draft_ms = 26.0;
/// The lane core prices a round of d drafts as window_costs[d + 1] + mtp_step_ms x d. The draft's cost does not grow
/// with d, so it is folded into every drafted width (2..8) and mtp_step_ms is 0; width 1 (no drafts) is the plain
/// forward (the depth rule never prices it without plain_guard).
const window_costs = blk: {
    var c: [forward_ms.len]lanes.config.Cost = undefined;
    for (&c, forward_ms, 0..) |*x, ms, i| x.* = .{ .width = i + 1, .ms = if (i == 0) ms else ms + draft_ms };
    break :blk c;
};

// ---- e-step: measured prices (TF_INKLING2_PRICES=timed) and verified copies (TF_INKLING2_COPIES=1)
/// The lane core's window prices from a load's timing (engine.Engine.prices, warm.timeWindows): width 1 the plain
/// round (its forward and commit), every wider width also one DSpark pass (it drafts 7 whatever the depth: priced in
/// the width, mtp_step_ms 0, as the static table does).
fn timedCosts(p: @import("inkling2").warm.Prices, out: []lanes.config.Cost) []lanes.config.Cost {
    const n: usize = @min(@as(usize, @intCast(p.rows)), out.len);
    for (out[0..n], 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = p.ms[i] + p.commit_ms + (if (i == 0) 0 else p.draft_ms) };
    return out[0..n];
}

/// GLM's copy facts (glm/cuda_copy_policy.zig apply, its TF_GLM_COPIES=1) on Inkling's: root and context copies
/// after a history match (16 tokens here), a first copy of 4 rows growing as its rows land, each stream's own copy landing
/// rates, plain rounds priced against drafted ones; a copy window up to `rows` rows (head drafts stay 7).
/// TF_INKLING2_COPY_MATCH=<n> (history tokens a copy needs, default 16; GLM's 4 lost 22% on an agent's tool turn) and
/// TF_INKLING2_COPY_FIRST=<rows> (a first copy window, default 4): a root copy takes its round from DSpark whenever one
/// is found (core lanes root_copy), so a longer match keeps DSpark's rounds where a copy breaks early (a tool call
/// re-writing a file it read, escaped: e-step REPORT section 2.2).
fn copyFacts(base: lanes.Model, rows: usize) lanes.Model {
    var m = base;
    m.copy_drafts = true;
    m.root_copies = true;
    m.context_copies = true;
    m.copy_match = envInt("TF_INKLING2_COPY_MATCH", 16, 1, 64); // job2 R16: the tool turn at DSpark's pace (4: GLM's)
    m.own_copy_rates = true;
    m.plain_guard = true;
    m.first_copy_rows = @intCast(envInt("TF_INKLING2_COPY_FIRST", 4, 2, @intCast(rows)));
    m.exact_width = @intCast(rows);
    m.batch_rows = @intCast(rows);
    m.window_costs_lone = false; // every copy pays for its rows
    m.draft_confidence = null;
    return m;
}

/// An integer setting within [lo, hi]; unset or unreadable: `default`.
fn envInt(name: [*:0]const u8, default: i64, lo: i64, hi: i64) i64 {
    const v = std.c.getenv(name) orelse return default;
    const n = std.fmt.parseInt(i64, std.mem.span(v), 10) catch return default;
    return std.math.clamp(n, lo, hi);
}
// ---- end e-step

pub const Ink = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    busy: ?*Stream = null,
    len: u64 = 0, // positions committed in the caches
    pending: ?usize = null, // rows of the last verify not yet committed
    kept: ?usize = null, // of them, the rows keep() chose (committed with the next step: draft, verify or release)
    held: [max_held]u32 = undefined,
    held_n: usize = 0,
    first_handle: ?u64 = null,
    tokens: [ring]u32 = undefined,
    next: u64 = 0,
    prof: struct { steps: u64 = 0, fwd_ns: u64 = 0, draft_ns: u64 = 0, rows: u64 = 0, kept: u64 = 0, draws: u64 = 0, draw_ns: u64 = 0, fallbacks: u64 = 0, checked: u64 = 0, differ: u64 = 0 } = .{},
    last_fwd_ns: u64 = 0, // the last verify's forward (TF_INKLING2_ROUNDS=1 logs each round)
    last_draw_ns: u64 = 0, // xs: the last verify's sampled draws on the host (0: greedy)
    // xs: TF_INKLING2_FAST_DRAW (default on; 0 or TF_INKLING2_LEGACY=1: the lane core's draw on every sampled row;
    // check: both, the lane core's token served, every disagreement counted and logged) and its candidates' buffer,
    // made at the first sampled draw
    fast_draw: ?bool = null,
    check_draw: bool = false,
    draw_sc: ?draw_mod.Scratch = null,
    trace: bool = false,
    /// p4-cache: rank 0's book of the reusable state (TF_INKLING2_PROMPT_REUSE; null: off, every pass from 0) and
    /// the last verify's ids (its kept rows join the book's live ids when committed).
    book: ?*reuse.Book = null,
    win: [engine.max_rows]u32 = undefined, // e-step: copy windows up to max_rows rows (was max_held + 1)
    costs: [engine.max_rows]lanes.config.Cost = undefined, // e-step: the timed window prices (facts, timedCosts)
    // xk-rows: this request's prompt end and the slot its state went to (release saves them), the debug dump
    cur_prompt: usize = 0,
    cur_slot: ?usize = null,
    dump_dir: ?[]const u8 = null, // TF_INKLING2_DUMP_LOGITS: rank 0 writes each request's first-token logits there
    dump_n: u64 = 0,

    pub fn backend(x: *Ink) be.Backend {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: be.Backend.VTable = .{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .held = heldFn, .release = release, .reports_fill = true };

    /// One stream. With the draft head: DSpark blocks of 7 drafted from the rows a verify kept, windows of up to 8
    /// rows priced by the measured table. Without: one-token steps, no copies (a window past one row needs the
    /// rollback that only the draft head's taps path sets up).
    pub fn facts(x: *Ink) lanes.Model {
        const head = x.e.draft != null;
        var m: lanes.Model = .{
            .exact_width = if (head) max_held + 1 else 1,
            .gpu_tokens = false,
            .mtp = head,
            .speculate = head,
            .speculate_early = false,
            .drafts = if (head) @intCast(x.e.spec) else 0,
            .mtp_step_ms = 0, // a block's 7 drafts are one pass whatever the depth: priced in window_costs (draft_ms)
            .window_costs = if (head) &window_costs else &.{},
            .copy_drafts = false,
            .max_streams = 1,
            .batch_rows = max_held + 1,
        };
        // ---- e-step: the load's timed prices (TF_INKLING2_PRICES=timed) and the copy policy (TF_INKLING2_COPIES=1)
        if (head) if (x.e.prices) |p| {
            m.window_costs = timedCosts(p, &x.costs);
        };
        if (head and x.e.copies) m = copyFacts(m, x.e.window_rows);
        // ---- end e-step
        return m;
    }

    fn self(ptr: *anyopaque) *Ink {
        return @ptrCast(@alignCast(ptr));
    }

    fn now(x: *const Ink) std.Io.Timestamp {
        return std.Io.Clock.awake.now(x.e.io);
    }

    fn since(x: *const Ink, t: std.Io.Timestamp) u64 {
        return @intCast(t.durationTo(x.now()).toNanoseconds());
    }

    fn step(x: *Ink, kind: Kind, ids: []const i32, want: bool, arg: u32) !void {
        try x.stepRaw(.{ .kind = @intFromEnum(kind), .want = @intFromBool(want), .arg = arg }, ids);
    }

    /// A step on every rank; p4-cache: a failed one leaves the state in question (the book continues nothing from it).
    fn stepRaw(x: *Ink, h: engine.Header, ids: []const i32) !void {
        x.e.step(h, ids) catch |err| {
            if (x.book) |b| b.fail();
            return err;
        };
    }

    fn take(x: *Ink, t: u32) u64 {
        const h = x.next;
        x.tokens[@intCast(h % ring)] = t;
        x.next += 1;
        return h;
    }

    /// `logits` drawn at `position` with the stream's sampling. xs: a sampled row through draw.zig (the lane core's
    /// token without its full-vocabulary sort; on by default), timed; greedy rows exactly as before.
    fn drawFrom(x: *Ink, s: *Stream, logits: []const f32, position: u64) !u32 {
        const st = s.sampling orelse {
            for (logits) |v| if (!std.math.isFinite(v)) return error.NonFiniteLogits;
            return lanes.gpu_full.draw(x.gpa, logits, s.sampling, @intCast(position), null);
        };
        const t0 = x.now();
        defer {
            x.prof.draws += 1;
            x.prof.draw_ns += x.since(t0);
        }
        if (x.fastDraw()) {
            if (x.draw_sc == null) x.draw_sc = .{ .gpa = x.gpa };
            const sc = &x.draw_sc.?;
            const before = sc.fallbacks;
            defer x.prof.fallbacks += sc.fallbacks - before;
            const got = try draw_mod.draw(sc, logits, st, @intCast(position));
            if (!x.check_draw) return got;
            const want = try lanes.gpu_full.draw(x.gpa, logits, s.sampling, @intCast(position), null);
            x.prof.checked += 1;
            if (got != want) {
                x.prof.differ += 1;
                std.log.err("inkling2 draw check: position {d} seed {d}: draw.zig {d}, the lane core {d}", .{ position, st.seed, got, want });
            }
            return want;
        }
        for (logits) |v| if (!std.math.isFinite(v)) return error.NonFiniteLogits;
        return lanes.gpu_full.draw(x.gpa, logits, s.sampling, @intCast(position), null);
    }

    /// xs: TF_INKLING2_FAST_DRAW=0 (or TF_INKLING2_LEGACY=1 unset) draws sampled rows with lanes.gpu_full.draw itself.
    fn fastDraw(x: *Ink) bool {
        if (x.fast_draw) |on| return on;
        const env = if (std.c.getenv("TF_INKLING2_FAST_DRAW")) |v| std.mem.span(v) else null;
        const on = if (env) |v| !std.mem.eql(u8, v, "0") else !@import("inkling2").defaults.legacy();
        x.check_draw = if (env) |v| std.mem.eql(u8, v, "check") else false;
        x.fast_draw = on;
        if (x.check_draw) std.log.info("inkling2: TF_INKLING2_FAST_DRAW=check: every sampled row drawn both ways, the lane core's token served", .{});
        std.log.info("inkling2: sampled rows drawn {s}", .{if (on) "by draw.zig (the lane core's token, no full sort; TF_INKLING2_FAST_DRAW=0: the lane core's draw)" else "by the lane core's draw (TF_INKLING2_FAST_DRAW off)"});
        return on;
    }

    /// xk-rows gate (TF_INKLING2_DUMP_LOGITS=dir): this request's first-token logits, fp32, to dir/req-<n>.logits.
    fn dumpLogits(x: *Ink, dir: []const u8) void {
        x.dump_n += 1;
        var buf: [1100]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}/req-{d}.logits", .{ dir, x.dump_n }, 0) catch return;
        const fd = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return;
        defer _ = std.c.close(fd);
        const bytes = std.mem.sliceAsBytes(x.e.logits);
        var done: usize = 0;
        while (done < bytes.len) {
            const n = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
            if (n <= 0) return;
            done += @intCast(n);
        }
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        if (x.busy != null and x.busy.? != s) return error.NoFreeSlot;
        if (s.images.len > 0) return error.InklingTextOnly;
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        if (p.len + @as(usize, s.max_new) + @max(x.e.window_rows, max_held + 1) + 1 > x.e.context()) return error.PastContext; // e-step: copy windows
        x.e.settle(); // xk-rows: the end thread's saves (and its residency pass) before the store's index is read
        if (x.e.disk) |rs| rs.newRequest(); // xr: the trim hint's request count
        x.busy = s;
        x.pending = null;
        x.kept = null;
        x.held_n = 0;
        x.first_handle = null;
        s.cached = 0;
        if (s.reuse.saved != null) s.reuse_failed = true; // no prompt cache: every pass starts at 0
        errdefer x.busy = null;
        // p4-cache: where the pass starts (reuse.zig: the live state, a snapshot slot, or 0); without
        // TF_INKLING2_PROMPT_REUSE at 0, the begin step the reset it always was
        const t0 = x.now();
        var pl: reuse.Plan = if (x.book) |b| b.plan(p, x.e.m.state.len) else .{};
        const live_before: usize = if (x.book) |b| b.live.items.len else 0;
        // xk-rows: a stored state reaching a block or more past the book's own plan is read back from disk first
        // (only its own file and window blocks when the live state still holds its rows from 0), then the pass
        // continues from its end; a read that fails anywhere leaves a new sequence
        var disk_ms: f64 = 0;
        var disk_same = false;
        if (x.book) |b| if (x.e.disk) |rs| if (!pl.repeat) if (rs.best(p, pl.at + 255)) |i| {
            const stored = rs.tokensOf(i);
            disk_same = b.holds(stored);
            const td = x.now();
            const ok = x.e.loadState(i, disk_same) catch false;
            disk_ms = @as(f64, @floatFromInt(x.since(td))) / 1e6;
            const shared = pl.shared; // (the plan's own field, read before the plan is rewritten)
            if (ok) {
                b.startDisk(stored, disk_same);
                pl = .{ .at = stored.len, .disk = true, .shared = shared };
            } else {
                if (rs.load_mapped) rs.forget(i); // a file failed its check (not the context's memory): it goes
                b.lose();
                pl = .{ .shared = shared };
            }
        };
        if (x.book) |b| if (!pl.disk) b.start(pl);
        x.cur_prompt = p.len;
        x.cur_slot = pl.restore;
        // xr: the trim hint, begin's one id: a pass that does not continue the live state leaves mapped the context a
        // recently used long state on disk was mapped for (rows.zig trimHint), so its return needs no growth
        const cont = pl.at > 0 and pl.restore == null and !pl.disk;
        const hint: i32 = if (x.e.disk) |rs| (if (cont) 0 else @intCast(rs.trimHint())) else 0;
        try x.stepRaw(.{ .kind = @intFromEnum(Kind.begin), .want = if (pl.restore) |i| @intCast(i + 1) else 0, .arg = @intCast(pl.at) }, &.{hint});
        if (hint > 0) std.log.info("inkling2 rows: trim hint {d} positions (a long state from the last {d} requests, on disk)", .{ hint, x.e.disk.?.set.keep_requests });
        s.cached = @intCast(pl.at);
        // chunks of the engine's prompt rows (engine.Settings.prompt_rows, up to max_prompt_rows): each reads the
        // experts its rows route to once, so wide chunks read the resident experts once per many rows
        var ids: [engine.max_prompt_rows]i32 = undefined;
        var at: usize = pl.at;
        // xk-rows: checkpoints every TF_INKLING2_ROWS_EVERY positions: a chunk ends on each (a pass from 0 cuts no
        // chunk: the width divides it) and the state there goes to disk, so a later prompt that parts from this one
        // resumes at the last checkpoint before the parting
        const every: usize = if (x.e.disk) |rs| (if (x.book != null) rs.set.every else 0) else 0;
        var checkpoints: usize = 0;
        while (at < p.len) {
            if (s.isCancelled()) {
                if (x.book) |b| b.pause(x.e.m.state.len); // between chunks: the rows run so far are the live state
                return error.Cancelled;
            }
            var n = @min(x.e.prompt_rows, p.len - at);
            if (every > 0) n = @min(n, (at / every + 1) * every - at);
            for (ids[0..n], p[at .. at + n]) |*d, t| d.* = @intCast(t);
            try x.step(.chunk, ids[0..n], at + n == p.len, 0);
            if (x.book) |b| b.extend(p[at .. at + n]);
            at += n;
            s.fill_at.store(@intCast(at), .monotonic);
            if (every > 0 and at % every == 0 and at < p.len) if (x.book) |b| if (!b.stuck and !b.broken and b.live.items.len == at and at >= x.e.disk.?.set.min) {
                // xr: copied beside the next chunk (rows_ckpt), committed by the end thread (rows_commit)
                if (x.e.ckptState(b.live.items[0..at]) catch false) checkpoints += 1;
            };
        }
        x.len = p.len;
        // p4-cache: the prompt's end saved into a snapshot slot by the prefill->decode step (want: the slot + 1)
        const slot: ?usize = if (x.book) |b| b.slotFor(p.len, pl) else null;
        if (x.book) |b| if (slot) |i| b.clear(i);
        if (slot) |i| x.cur_slot = i; // xk-rows: the prompt's end, which release saves
        try x.stepRaw(.{ .kind = @intFromEnum(Kind.prefilled), .want = if (slot) |i| @intCast(i + 1) else 0, .arg = @intCast(p.len - pl.at) }, &.{});
        if (x.book) |b| {
            if (pl.repeat) @memcpy(x.e.logits, b.logits[pl.restore.?][0..x.e.logits.len]); // no rows ran: the slot's
            b.finish(slot, p.len, x.e.logits);
            std.log.info("inkling2 reuse: prompt {d}, {d} shared with the live state's {d}; {s} at {d}, {d} rows run, {d:.0} ms to the first token's logits{s}", .{ p.len, pl.shared, live_before, pl.mode(), pl.at, p.len - pl.at, @as(f64, @floatFromInt(x.since(t0))) / 1e6, if (pl.disk) (if (disk_same) " (read back from disk: its own file and window, the rows below held live)" else " (read back from disk)") else "" });
            if (pl.disk) std.log.info("inkling2 rows: read {d} tokens back in {d:.0} ms", .{ pl.at, disk_ms });
            if (checkpoints > 0) std.log.info("inkling2 rows: {d} checkpoints copied beside this prompt's pass (committed at its end)", .{checkpoints});
        }
        if (x.dump_dir) |d| x.dumpLogits(d); // xk-rows gate: the first token's logits, as they are drawn from
        x.first_handle = x.take(try x.drawFrom(s, x.e.logits, p.len));
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        if (x.busy != s) return error.UnknownStream;
        if (position != x.len) return error.PositionMismatch;
        return x.first_handle orelse error.NoFirstToken;
    }

    fn value(x: *Ink, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(x, h),
        };
    }

    /// Rows the next commit keeps: keep()'s choice, else `kept`, else all of the last verify's (0: none pending).
    fn toCommit(x: *const Ink, kept: ?usize) usize {
        const rows = x.pending orelse return 0;
        return x.kept orelse kept orelse rows;
    }

    /// The last verify's rows (toCommit) committed on every rank in one step (keep + token), and with an `anchor` the
    /// next block drafted from it in the same step; without rows pending, the draft alone. Returns whether it drafted.
    fn commit(x: *Ink, kept: ?usize, anchor: ?u32) !bool {
        const ids = [1]i32{if (anchor) |a| @intCast(a) else 0};
        const send: []const i32 = if (anchor != null) &ids else &.{};
        const rows = x.pending orelse {
            if (anchor == null) return false;
            try x.step(.draft, send, false, 0);
            return true;
        };
        const k = x.toCommit(kept);
        if (k == 0 or k > rows) return error.BadKeep;
        const t0 = x.now();
        try x.step(.commit, send, false, @intCast(k));
        if (x.book) |b| b.extend(x.win[0..k]); // p4-cache: the kept rows are in the caches
        if (x.trace) {
            if (x.last_draw_ns > 0) {
                std.log.info("inkling2 round: {d} rows, {d} kept; verify {d:.1} ms, commit{s} {d:.1} ms; draw {d:.2} ms", .{ rows, k, @as(f64, @floatFromInt(x.last_fwd_ns)) / 1e6, if (anchor != null) " + draft" else "", @as(f64, @floatFromInt(x.since(t0))) / 1e6, @as(f64, @floatFromInt(x.last_draw_ns)) / 1e6 });
            } else std.log.info("inkling2 round: {d} rows, {d} kept; verify {d:.1} ms, commit{s} {d:.1} ms", .{ rows, k, @as(f64, @floatFromInt(x.last_fwd_ns)) / 1e6, if (anchor != null) " + draft" else "", @as(f64, @floatFromInt(x.since(t0))) / 1e6 });
        }
        x.len += k;
        x.prof.kept += k;
        x.pending = null;
        x.kept = null;
        return anchor != null;
    }

    /// No draft head: one token in, the next drawn (a one-row window committed at once).
    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        if (x.busy != s) return error.UnknownStream;
        if (x.e.draft != null) return error.NotPipelined;
        const t = try x.value(feed);
        if (position != x.len + 1) return error.PositionMismatch;
        if (x.len + 1 > x.e.context()) return error.PastContext;
        const ids = [1]i32{@intCast(t)};
        try x.step(.chunk, &ids, true, 1); // the forward, then its token counted (one step)
        x.len += 1;
        if (x.book) |b| b.extend(&[_]u32{t}); // p4-cache
        return x.take(try x.drawFrom(s, x.e.logits, position));
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        if (h >= x.next or x.next - h > ring) return error.NoSuchToken;
        return x.tokens[@intCast(h % ring)];
    }

    /// [pending, drafts] in one forward on every rank; each row drawn with the stream's sampling at its position.
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        for (windows, out) |w, *o| {
            if (w.parents != null) return error.TreesNotBuilt;
            if (w.masks.len != 0) return error.StructuresUnsupported;
            if (w.stream != x.busy) return error.UnknownStream;
            _ = try x.commit(null, null);
            if (w.held > x.held_n) return error.NoHeldDrafts;
            const drafts: []const u32 = if (w.held > 0) x.held[0..w.held] else w.tokens;
            const rows = 1 + drafts.len;
            if (rows > @max(x.e.window_rows, max_held + 1)) return error.WindowTooWide; // e-step: copy windows
            if (x.len + rows > x.e.context()) return error.PastContext;
            for (w.positions, 0..) |p, r| if (p != x.len + 1 + r) return error.PositionMismatch;
            var ids: [engine.max_rows]i32 = undefined;
            ids[0] = @intCast(w.pending);
            for (drafts, 0..) |t, j| ids[1 + j] = @intCast(t);
            const sampled = w.stream.sampling != null;
            const t0 = x.now();
            try x.step(.verify, ids[0..rows], sampled, 0);
            if (x.book != null) for (x.win[0..rows], ids[0..rows]) |*wi, t| {
                wi.* = @intCast(t); // p4-cache: the ids a commit of this window keeps
            };
            x.last_fwd_ns = x.since(t0);
            x.prof.fwd_ns += x.last_fwd_ns;
            x.prof.steps += 1;
            x.prof.rows += rows;
            const V = x.e.vocab();
            const td = x.now();
            for (0..rows) |r| {
                o.sampled[r] = if (sampled) try x.drawFrom(w.stream, x.e.row_logits[r * V ..][0..V], w.positions[r]) else @intCast(x.e.argmax[r]);
            }
            x.last_draw_ns = if (sampled) x.since(td) else 0; // xs: the round trace's draw time
            @memcpy(o.drafts[0..drafts.len], drafts);
            x.pending = rows;
            x.held_n = 0;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            if (w.stream != x.busy) return error.UnknownStream;
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
            // committed with the next step (the draft's, the next verify's or the release's): one step a round less
            const rows = x.pending orelse continue;
            if (path.len > rows) return error.BadKeep;
            x.kept = path.len;
        }
    }

    /// The kept rows committed, then a block drafted from the anchor (the token after the last kept row, or the
    /// prompt's first token) on every rank; up to `depth` of its 7 held for the next window.
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        for (requests) |r| {
            const kept: ?usize = if (r.rows) |rows| rows.len else null;
            if (r.depth == 0 and r.lanes == null) {
                _ = try x.commit(kept, null);
                continue;
            }
            // checked against the length after the commit; on a refusal past here the commit still happens
            const after = x.len + x.toCommit(kept);
            if (x.e.draft == null) return error.NoDraftHead;
            if (r.stream != x.busy) return error.UnknownStream;
            if (r.lanes != null) return error.TreesNotBuilt;
            const anchor: u32 = a: {
                errdefer _ = x.commit(kept, null) catch {};
                if (r.position != after + 1) return error.PositionMismatch;
                break :a if (r.rows) |rows| b: {
                    for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
                    break :b r.follow[rows.len - 1];
                } else try x.value(r.first orelse return error.NoFirstToken);
            };
            if (after + max_held + 2 > x.e.context()) {
                _ = try x.commit(kept, null);
                x.held_n = 0;
                continue;
            }
            // keep + token + draft: one step on every rank
            const t0 = x.now();
            _ = try x.commit(kept, anchor);
            x.prof.draft_ns += x.since(t0);
            const depth = @min(r.depth, max_held);
            for (x.held[0..depth], x.e.drafts[0..depth]) |*h, d| h.* = @intCast(d);
            x.held_n = depth;
        }
    }

    fn heldFn(ptr: *anyopaque, s: *Stream, out: []u32) anyerror!void {
        const x = self(ptr);
        if (x.busy != s) return error.UnknownStream;
        if (out.len > x.held_n) return error.NoHeldDrafts;
        @memcpy(out, x.held[0..out.len]);
    }

    /// The request's end: the last verify's rows committed (the lane core kept what it emits), the answer's demand
    /// a vote and a residency pass (every rank).
    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.busy != s) return;
        const p = x.prof;
        const n: f64 = @floatFromInt(@max(p.steps, 1));
        std.log.info("inkling2 decode: {d} verify forwards, {d:.2} rows and {d:.2} kept each, forward {d:.1} ms, commit + draft {d:.1} ms a step", .{ p.steps, @as(f64, @floatFromInt(p.rows)) / n, @as(f64, @floatFromInt(p.kept)) / n, @as(f64, @floatFromInt(p.fwd_ns)) / 1e6 / n, @as(f64, @floatFromInt(p.draft_ns)) / 1e6 / n });
        // xs: a sampled request's host draws (draw.zig's, or the lane core's with TF_INKLING2_FAST_DRAW=0)
        if (p.draws > 0) std.log.info("inkling2 draw: {d} sampled rows in {d:.1} ms, {d:.3} ms a row ({s}); {d} fell back to the lane core's draw; checked {d}, {d} differ", .{ p.draws, @as(f64, @floatFromInt(p.draw_ns)) / 1e6, @as(f64, @floatFromInt(p.draw_ns)) / 1e6 / @as(f64, @floatFromInt(p.draws)), if (x.check_draw) "check: both" else if (x.fast_draw orelse false) "draw.zig" else "the lane core's draw", p.fallbacks, p.checked, p.differ });
        x.prof = .{};
        _ = x.commit(null, null) catch {};
        if (x.book) |b| b.pause(x.e.m.state.len); // p4-cache: every row committed: the state is the live ids'
        // xk-rows: the prompt's end (its snapshot slot) and the live end go to disk on the end thread, off the reply
        if (x.book) |b| if (x.e.disk) |rs| if (b.ok) {
            var reqs: [2]engine.Engine.SaveReq = undefined;
            var nr: usize = 0;
            const live = b.live.items.len;
            if (x.cur_slot) |sl| if (b.at[sl]) |q| if (q == x.cur_prompt and q >= rs.set.min and q <= live) {
                reqs[nr] = .{ .src = .{ .slot = sl }, .at = q };
                nr += 1;
            };
            if (live >= rs.set.min and live != x.cur_prompt) {
                reqs[nr] = .{ .src = .live, .at = live };
                nr += 1;
            }
            x.e.queueSaves(reqs[0..nr], b.live.items);
        };
        x.e.endAsync(); // between requests: the next request's first step waits for it
        x.busy = null;
        x.len = 0;
        x.pending = null;
        x.kept = null;
        x.held_n = 0;
    }
};
