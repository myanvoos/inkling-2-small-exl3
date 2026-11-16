//! One TP rank of Inkling-2 for serving: the model (inkling2 model / forward), its residency (fill, adapt) and the
//! steps rank 0 sends the other ranks. Every forward and every residency pass holds collectives, so every rank runs
//! the same sequence of steps: rank 0 runs `step` (the step to the others over the TP group, then its own run), the
//! others `follow` (receive, run) until rank 0's `quit`. A step is 16 bytes and at most `max_ids` token ids.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const reuse = @import("reuse.zig"); // p4-cache
const rows_mod = @import("rows.zig"); // xk-rows
const core = @import("core"); // e-step: the tokenizer of the timed windows' text
const lanes = @import("lanes"); // e-step: the cost text and the window-cost cache

const Model = ink.model.Model;
const gib: f64 = 1024.0 * 1024.0 * 1024.0;

pub const max_rows = 16; // rows a verify / decode forward takes (the forward's scratch, graphs, histories)
/// The widest prompt chunk (TF_INKLING2_PREFILL_ROWS caps at it): a prompt chunk's ids past a step's own max_ids cross
/// in a second gather of their own.
pub const max_prompt_rows = 4096; // lead, phase 4: 4096-row chunks pay with the token exchange (TF_INKLING2_PREFILL_ROWS=4096)
const engine_rows = max_rows;
const max_ids = max_rows;

pub const Kind = enum(u32) { begin = 1, chunk, prefilled, token, end, quit, verify, keep, draft, commit, rows_save, rows_load, rows_ckpt, rows_commit };

/// One step as it crosses: kind; begin: a new sequence (arg and want 0), else (p4-cache, TF_INKLING2_PROMPT_REUSE)
/// the pass resumes at position arg, snapshot slot want - 1 restored first (want 0: the live state continues);
/// chunk: ids that follow and whether the last row's logits are read (want 1), and
/// output tokens it commits (arg: a one-token decode step's token, counted after its forward); prefilled: the
/// prompt's new tokens (want: snapshot slot + 1 that keeps the prompt's end, p4-cache); token: output tokens
/// committed (arg); verify: the window's ids (want 1: every row's logits
/// to rank 0's host, for a sampled stream); keep: rows of the last verify kept (arg); draft: the anchor (ids[0]);
/// commit: keep + token + draft in one step (a round's second and last): the last verify's first `arg` rows kept,
/// `arg` tokens counted, then, with one id, a block drafted from that anchor.
/// xk-rows: rows_save / rows_load: a state at position arg written to / read from every rank's disk (rows.zig), the
/// file list (n words) in a second gather; rows_save's want: its source (0 the live state, else snapshot slot want - 1).
/// xr: rows_ckpt: a checkpoint at arg copied beside the prefill (its list as rows_save's); rows_commit: the
/// checkpoints since the last commit into rank 0's index once every rank's copies ended whole (an empty list).
pub const Header = extern struct { kind: u32, n: u32 = 0, want: u32 = 0, arg: u32 = 0 };
const step_bytes = @sizeOf(Header) + max_ids * 4;

pub const Settings = struct {
    context: usize,
    floor_gib: f64 = 8, // MemAvailable left after the arena (TF_INKLING2_FLOOR_GIB)
    prior: ?[]const u8 = null, // TF_INKLING2_PRIOR: fp32 [L][E] scores (the plan-gamma prior)
    adapt: bool = true, // TF_INKLING2_ADAPT=0 freezes the placement
    spec: usize = 7, // TF_INKLING2_SPEC: DSpark drafts a verify (0: no draft head)
    prewarm: bool = true, // TF_INKLING2_PREWARM=0: no load-time warm-up (the first request makes every width's plans)
    /// TF_INKLING2_PREFILL_ROWS: rows a prompt chunk takes (max_rows .. max_prompt_rows; at most what MemAvailable
    /// leaves over the floor and a 1 GiB margin once the arena is filled). Each chunk reads the resident experts it
    /// routes to once, so wide chunks read each expert once per many prompt rows instead of once per 16.
    prompt_rows: usize = 2048,
    workers: usize = 16,
    readers: usize = 8,
    /// e-step: TF_INKLING2_PRICES=timed: the lane core prices rounds by verify windows, the DSpark pass and the commit
    /// timed at load (warm.timeWindows; kept per build, store and settings in lanes.cost_cache), not the static table
    /// of the Nov-13 native build (backend.zig). Unset / static: that table (x2's facts).
    prices: bool = false,
    /// e-step: TF_INKLING2_COPIES=1: the lane core's verified root / context copies (a 16-token history match, copies of
    /// the context verified like any draft; copy windows up to max_rows rows, graphed). Unset / 0: no copies (x2).
    copies: bool = false,

    pub fn fromEnv(context: usize) Settings {
        var s: Settings = .{ .context = context };
        if (!ink.defaults.legacy()) s.prompt_rows = 4096; // the user, 2026-11-15: on by default (TF_INKLING2_PREFILL_ROWS sets any)
        if (std.c.getenv("TF_INKLING2_FLOOR_GIB")) |v| s.floor_gib = std.fmt.parseFloat(f64, std.mem.span(v)) catch s.floor_gib;
        if (std.c.getenv("TF_INKLING2_PRIOR")) |v| s.prior = std.mem.span(v);
        if (std.c.getenv("TF_INKLING2_ADAPT")) |v| s.adapt = !std.mem.eql(u8, std.mem.span(v), "0");
        if (std.c.getenv("TF_INKLING2_PREWARM")) |v| s.prewarm = !std.mem.eql(u8, std.mem.span(v), "0");
        if (std.c.getenv("TF_INKLING2_SPEC")) |v| s.spec = std.fmt.parseInt(usize, std.mem.span(v), 10) catch s.spec;
        if (std.c.getenv("TF_INKLING2_PREFILL_ROWS")) |v| s.prompt_rows = std.fmt.parseInt(usize, std.mem.span(v), 10) catch s.prompt_rows;
        s.prompt_rows = std.math.clamp(s.prompt_rows, max_rows, max_prompt_rows);
        if (s.spec > ink.dspark.spec) s.spec = ink.dspark.spec;
        // ---- e-step
        if (std.c.getenv("TF_INKLING2_PRICES")) |v| s.prices = std.mem.eql(u8, std.mem.span(v), "timed");
        if (std.c.getenv("TF_INKLING2_COPIES")) |v| s.copies = std.mem.eql(u8, std.mem.span(v), "1");
        // ---- end e-step
        return s;
    }

    /// e-step: a verify window's rows at most: 8 (the DSpark head's 7 drafts + the pending row), max_rows with copies.
    pub fn windowRows(s: Settings) usize {
        return if (s.copies and s.spec > 0) max_rows else ink.dspark.block;
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    m: *Model,
    loader: *ink.fill.Loader,
    ad: *ink.adapt.Adapt,
    rank: usize,
    world: usize,
    mine: cuda.DeviceBuffer,
    all: cuda.DeviceBuffer,
    ids: [max_ids]i32 = undefined,
    /// Rows a prompt chunk takes (Model.prompt_rows, the same on every rank); a chunk step's ids past max_ids cross
    /// through these (rank 0's segment first) into prompt_ids.
    prompt_rows: usize = max_rows,
    ids_mine: cuda.DeviceBuffer,
    ids_all: cuda.DeviceBuffer,
    prompt_ids: [max_prompt_rows]i32 = undefined,
    logits: []f32, // rank 0: the last read row's fp32 logits [unpadded]
    resident: [2]usize = .{ 0, 0 },
    draft: ?*ink.dspark.Draft = null,
    spec: usize = 0,
    ctx: *const cuda.Context,
    end_thread: ?std.Thread = null, // the last request's end pass, running between requests
    // a verify's window: where it started, its rows, every row's argmax and (sampled streams) logits [rows][unpadded]
    base: usize = 0,
    rows: usize = 0,
    argmax: [engine_rows]i32 = undefined,
    row_logits: []f32 = &.{},
    drafts: [ink.dspark.spec]i32 = undefined,
    slots: reuse.Slots = .{}, // p4-cache: snapshots of prompts' ends (none with TF_INKLING2_PROMPT_REUSE off)
    // xr: the begin step's trim hint (rank 0's rows.zig trimHint; every rank): context positions a recently used long
    // state on disk was mapped at, which the residency's reclaim keeps mapped (Adapt.trim_hint, when it has one)
    trim_hint: usize = 0,
    // xk-rows: this rank's conversation states on disk (rows.zig; null: off), the list a follower received, rank 0's
    // saves for the end thread (the live ids, salted, they name) and the last read's outcome on every rank
    disk: ?*rows_mod.Store = null,
    recv_list: std.ArrayList(u8) = .empty,
    saves: [2]SaveReq = undefined,
    saves_n: usize = 0,
    save_ids: std.ArrayList(u32) = .empty,
    load_ok: bool = false,
    // ---- e-step: the measured prices rank 0 hands the lane core (TF_INKLING2_PRICES=timed), the copy policy
    // (TF_INKLING2_COPIES) and a verify window's rows at most (8; max_rows with copies)
    prices: ?ink.warm.Prices = null,
    copies: bool = false,
    window_rows: usize = ink.dspark.block,

    /// xk-rows: a state the end thread saves: its source, its position (its ids: save_ids' first `at`).
    pub const SaveReq = struct { src: rows_mod.Source, at: usize };

    /// Loads this rank: dense weights, the arena from what MemAvailable leaves above the floor (the smallest room of
    /// the ranks), the fill from the prior. `group` is open (world > 1).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, rank: usize, world: usize, group: ?cuda.tp_group.Group, s: Settings) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const m = try Model.open(gpa, io, ctx, dir, rank, world, group, s.context, max_rows);
        errdefer m.close();
        if (m.lazy) |lz| lz.floor = @intFromFloat(s.floor_gib * gib); // p4-mem: what a context growth keeps free
        // x-mixed: the mixed store's loader (its native bank's) reads the release checkpoint with the release's own config
        // (the EXL3 view holds no expert tensors)
        const loader = try ink.fill.Loader.open(gpa, io, if (m.mix_cfg) |nc| nc else &m.c, if (m.mix) m.mix_native else dir, rank, world, s.workers, s.readers);
        errdefer loader.close();
        m.loader = loader;
        var draft: ?*ink.dspark.Draft = null;
        errdefer if (draft) |dd| dd.close();
        if (s.spec > 0) {
            try m.enableSpec();
            draft = try ink.dspark.Draft.open(gpa, io, m, dir, max_rows);
            std.log.info("inkling2 rank {d}/{d}: DSpark draft head (mtp.safetensors) loaded; {d} drafts a verify", .{ rank, world, s.spec });
        }
        // xk-rows: the store's settings now, so the bank's sizing leaves its stages (the store opens at the end)
        const rows_set = try rows_mod.Settings.fromEnv(gpa, reuse.slotsWanted(m) > 0);
        var rows_owned = false;
        errdefer if (!rows_owned and rows_set.on) gpa.free(rows_set.root);
        const E = m.c.experts;
        const L = m.c.layers;
        const w = m.slotWords();
        const sb = [2]f64{ @floatFromInt(ink.model.Arena.slotBytes(w, 0)), @floatFromInt(ink.model.Arena.slotBytes(w, 1)) };
        const nl = [2]usize{ m.c.countAct(.nvfp4), m.c.countAct(.mxfp8) }; // the arena's pools: release formats
        const avail: f64 = @floatFromInt(try ink.adapt.memAvailable(io, gpa));
        var room = avail - s.floor_gib * gib - 3.5 * gib;
        // p4-mem: with the lazy context the arena takes the context memory not mapped (TF_INKLING2_LAZY_ARENA_GIB at
        // most) and leaves the prompt scratch's room, made after it (phase 3's comes out of the 3.5 GiB margin: at 1M
        // it fits 2048 rows with ~0.2-0.7 GiB to spare; a bigger arena's rounding left 512)
        if (m.lazy) |lz| room -= @floatFromInt(lz.heldBack() + m.promptGrowth(s.prompt_rows));
        var mine = try cuda.DeviceBuffer.alloc(ctx.d, step_bytes);
        errdefer mine.free();
        var all = try cuda.DeviceBuffer.alloc(ctx.d, step_bytes * world);
        errdefer all.free();
        if (world > 1) {
            var v = [2]f64{ room, 0 };
            try mine.upload(0, std.mem.sliceAsBytes(&v));
            try ink.forward.gatherNccl(m, m.s, mine.ptr, all.ptr, 16);
            try m.s.synchronize();
            var rooms: [16]f64 = undefined;
            try all.download(0, std.mem.sliceAsBytes(rooms[0 .. 2 * world]));
            for (0..world) |r| room = @min(room, rooms[2 * r]);
        }
        // the EXL3 export (exl3.zig): every expert resident, its slots counted against the room above, no residency to
        // size, fill or adapt; else the arena from the room, each layer the same quota
        var per: usize = E;
        var resident = [2]usize{ 0, 0 };
        if (m.mix) {
            // ---- x-mixed: the EXL3 slots first (the EXL3 store's room check), then the native bank from what MemAvailable
            // leaves over the floor, the bank's margin and what is made after it (the prompt scratch at the asked rows,
            // the reuse slots, the loader's stages; with the lazy context the bank takes the deferred context memory and
            // gives it back as a sequence grows): the fewest any rank can (both ranks hold the same experts)
            _ = try ink.exl3.Store.fits(gpa, io, dir, m, room, s.floor_gib);
            const t_ex = std.Io.Clock.awake.now(io);
            m.ex = try ink.exl3.Store.open(gpa, io, m, dir, s.floor_gib);
            const xs = m.ex.?.stats;
            std.log.info("inkling2 rank {d}/{d}: EXL3 export: all {d} experts of {d} layers resident ({d:.2} GiB) in {d:.1} s: {d} files, {d:.1} GB read ({d:.1} s read, {d:.1} s pack, {d:.1} s upload; paced {d} ms, MemAvailable min {d:.2} GiB); floor {d} GiB", .{ rank, world, E, L, @as(f64, @floatFromInt(m.ex.?.bytes)) / gib, @as(f64, @floatFromInt(t_ex.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9, xs.files, @as(f64, @floatFromInt(xs.read_bytes)) / 1e9, @as(f64, @floatFromInt(xs.read_ns)) / 1e9, @as(f64, @floatFromInt(xs.pack_ns)) / 1e9, @as(f64, @floatFromInt(xs.upload_ns)) / 1e9, xs.paused_ms, @as(f64, @floatFromInt(xs.min_avail)) / gib, s.floor_gib });
            const mo = ink.mix.Opts.fromEnv();
            const after: f64 = @floatFromInt(m.promptGrowth(s.prompt_rows) + reuse.slotsWanted(m) * reuse.Slots.bytesEach(m, draft) + 2 * (loader.stage_bytes + 4096) + (if (m.lazy) |lz| lz.heldBack() else 0) + rows_mod.reserveBytes(rows_set, world, rows_mod.ckptStageBytes(m, draft)));
            var broom = @as(f64, @floatFromInt(try ink.adapt.memAvailable(io, gpa))) - (s.floor_gib + mo.margin_gib) * gib - after;
            if (world > 1) {
                var v = [2]f64{ broom, 0 };
                try mine.upload(0, std.mem.sliceAsBytes(&v));
                try ink.forward.gatherNccl(m, m.s, mine.ptr, all.ptr, 16);
                try m.s.synchronize();
                var rooms: [16]f64 = undefined;
                try all.download(0, std.mem.sliceAsBytes(rooms[0 .. 2 * world]));
                for (0..world) |r| broom = @min(broom, rooms[2 * r]);
            }
            resident = ink.mix.bankSlots(m, mo, broom);
            try m.allocArena(resident, .{ 0, 0 });
            per = (resident[0] + resident[1]) / @max(L, 1);
            std.log.info("inkling2 rank {d}/{d}: the native bank: {d} NVFP4 + {d} MXFP8 slots ({d:.2} GiB of the release's experts: {d:.1} a layer on average, {d:.1} in the NVFP4 layers, {d:.1} in the MXFP8 layers) from {d:.2} GiB over the {d} GiB floor and the {d:.1} GiB bank margin, after {d:.2} GiB still to be made (prompt scratch, reuse slots, loader stages)", .{ rank, world, resident[0], resident[1], ink.mix.bankBytes(m, resident) / gib, @as(f64, @floatFromInt(resident[0] + resident[1])) / @as(f64, @floatFromInt(L)), @as(f64, @floatFromInt(resident[0])) / @as(f64, @floatFromInt(@max(nl[0], 1))), @as(f64, @floatFromInt(resident[1])) / @as(f64, @floatFromInt(@max(nl[1], 1))), @max(broom, 0) / gib, s.floor_gib, mo.margin_gib, after / gib });
            // ---- end x-mixed
        } else if (m.c.exl3) {
            _ = try ink.exl3.Store.fits(gpa, io, dir, m, room, s.floor_gib); // x-merge: the room check, one function
            try m.allocArena(.{ 0, 0 }, .{ 0, 0 });
            const t_ex = std.Io.Clock.awake.now(io);
            m.ex = try ink.exl3.Store.open(gpa, io, m, dir, s.floor_gib);
            const xs = m.ex.?.stats;
            std.log.info("inkling2 rank {d}/{d}: EXL3 export: all {d} experts of {d} layers resident ({d:.2} GiB) in {d:.1} s: {d} files, {d:.1} GB read ({d:.1} s read, {d:.1} s pack, {d:.1} s upload; paced {d} ms, MemAvailable min {d:.2} GiB); floor {d} GiB", .{ rank, world, E, L, @as(f64, @floatFromInt(m.ex.?.bytes)) / gib, @as(f64, @floatFromInt(t_ex.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9, xs.files, @as(f64, @floatFromInt(xs.read_bytes)) / 1e9, @as(f64, @floatFromInt(xs.read_ns)) / 1e9, @as(f64, @floatFromInt(xs.pack_ns)) / 1e9, @as(f64, @floatFromInt(xs.upload_ns)) / 1e9, xs.paused_ms, @as(f64, @floatFromInt(xs.min_avail)) / gib, s.floor_gib });
        } else {
            per = @min(@as(usize, @intFromFloat(@max(room / (@as(f64, @floatFromInt(nl[0])) * sb[0] + @as(f64, @floatFromInt(nl[1])) * sb[1]), 0))), E);
            if (per < m.c.topk) {
                std.log.err("inkling2: {d:.1} GiB above the {d} GiB floor holds {d} experts a layer, under the router's {d}", .{ room / gib, s.floor_gib, per, m.c.topk });
                return error.ArenaBelowFloor;
            }
            resident = .{ per * nl[0], per * nl[1] };
            try m.allocArena(resident, .{ 0, 0 });
        }
        const prior = try gpa.alloc(f32, L * E);
        defer gpa.free(prior);
        if (s.prior) |p| {
            const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(L * E * 4 + 1));
            defer gpa.free(bytes);
            if (bytes.len != L * E * 4) return error.PriorShape;
            @memcpy(std.mem.sliceAsBytes(prior), bytes);
        } else {
            std.log.warn("inkling2: no TF_INKLING2_PRIOR: the first fill takes every expert as equal", .{});
            @memset(prior, 1);
        }
        var policy: ink.residency.Policy = .{};
        policy.adapt = s.adapt and !m.c.exl3; // the EXL3 export: nothing to move (every expert resident)
        const ad = try ink.adapt.Adapt.init(gpa, io, m, loader, prior, policy);
        errdefer ad.deinit();
        const t0 = std.Io.Clock.awake.now(io);
        if (m.mix) try ad.bankHistory(dir); // x-mixed: the demand history (TF_INKLING2_MIX_HISTORY) before the boot fill
        if (!m.c.exl3 or m.mix) try ad.fillBoot(null); // x-mixed: the bank's boot fill (the prior, or the history)
        const st = loader.stats;
        if (!m.c.exl3 or m.mix) std.log.info("inkling2 rank {d}/{d}: {d} NVFP4 + {d} MXFP8 experts resident ({d} a layer, {d:.1} GiB) in {d:.1} s, {d:.1} GB read past the page cache (paced {d} ms, PSI max {d:.1}, MemAvailable min {d:.2} GiB); floor {d} GiB", .{ rank, world, resident[0], resident[1], per, (@as(f64, @floatFromInt(resident[0])) * sb[0] + @as(f64, @floatFromInt(resident[1])) * sb[1]) / gib, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e9, @as(f64, @floatFromInt(st.bytes)) / 1e9, st.paused_ms, st.max_psi, @as(f64, @floatFromInt(st.min_avail)) / gib, s.floor_gib });
        ink.page_cache.finish(dir, rank, world); // e-load (x5): the small files' pass, one line a rank
        if (m.lazy) |lz| {
            var lb: [512]u8 = undefined;
            std.log.info("inkling2 rank {d}/{d}: {s}", .{ rank, world, lz.summary(&lb) });
        }
        // prompt chunks of up to s.prompt_rows rows: their scratch is made now, after the arena (sized above from the
        // floor, so the residency is the one it is without them), as wide as MemAvailable allows over the floor and a
        // 1 GiB margin, halving from the asked width; every rank takes the narrowest any rank can
        var prompt_rows = s.prompt_rows;
        const avail_fill: f64 = @floatFromInt(try ink.adapt.memAvailable(io, gpa));
        const spare = avail_fill - (s.floor_gib + 1.0) * gib;
        while (prompt_rows > max_rows and @as(f64, @floatFromInt(m.promptGrowth(prompt_rows))) > spare) prompt_rows /= 2;
        prompt_rows = @max(prompt_rows, max_rows);
        // p4-cache: the snapshot slots every rank keeps (TF_INKLING2_PROMPT_REUSE), the fewest any rank asks for (the
        // exchange's second word, 0 with the switch off)
        var reuse_slots: u64 = reuse.slotsWanted(m);
        if (world > 1) {
            var v = [2]u64{ prompt_rows, reuse_slots };
            try mine.upload(0, std.mem.sliceAsBytes(&v));
            try ink.forward.gatherNccl(m, m.s, mine.ptr, all.ptr, 16);
            try m.s.synchronize();
            var got: [16]u64 = undefined;
            try all.download(0, std.mem.sliceAsBytes(got[0 .. 2 * world]));
            for (0..world) |r| prompt_rows = @min(prompt_rows, got[2 * r]);
            for (0..world) |r| reuse_slots = @min(reuse_slots, got[2 * r + 1]);
        }
        const grown = m.promptGrowth(prompt_rows);
        try m.growPrompt(prompt_rows);
        var ids_mine = try cuda.DeviceBuffer.alloc(ctx.d, max_prompt_rows * 4);
        errdefer ids_mine.free();
        var ids_all = try cuda.DeviceBuffer.alloc(ctx.d, max_prompt_rows * 4 * world);
        errdefer ids_all.free();
        std.log.info("inkling2 rank {d}/{d}: prompt chunks of up to {d} rows ({d:.2} GiB of scratch past the arena; MemAvailable {d:.2} GiB before it, floor {d} GiB)", .{ rank, world, prompt_rows, @as(f64, @floatFromInt(grown)) / gib, avail_fill / gib, s.floor_gib });
        // p4-cache: the slots after the prompt scratch (the arena's size and the chunk width are the ones without them)
        var slots: reuse.Slots = if (reuse_slots > 0) try reuse.Slots.alloc(m, draft, reuse_slots) else .{};
        errdefer slots.free();
        if (slots.n > 0) std.log.info("inkling2 rank {d}/{d}: prompt reuse on (TF_INKLING2_PROMPT_REUSE): {d} snapshot slots of {d:.1} MiB; MemAvailable {d:.2} GiB after them", .{ rank, world, slots.n, @as(f64, @floatFromInt(reuse.Slots.bytesEach(m, draft))) / (1024.0 * 1024.0), @as(f64, @floatFromInt(try ink.adapt.memAvailable(io, gpa))) / gib });
        // x-merge: how far the lazy context reaches now without a give-back (lazy.reach: MemAvailable over the floor and
        // the margin, ensure's test); the EXL3 store gives nothing back, so a chunk past it is refused there
        if (m.lazy) |lz| {
            const av = try ink.adapt.memAvailable(io, gpa);
            const grow_room = av -| (lz.floor + lz.margin);
            const per_pos = (lz.bytesAt(lz.context) -| lz.bytesAt(lz.mapped)) / @max(lz.context -| lz.mapped, 1);
            std.log.info("inkling2 rank {d}/{d}: lazy context reach: {d} of {d} positions ({d} mapped; MemAvailable {d:.2} GiB, {d:.2} GiB over the {d:.1} GiB floor and {d:.1} GiB margin; {d} B a position past the mapped ones){s}", .{ rank, world, lz.reach(grow_room), lz.context, lz.mapped, @as(f64, @floatFromInt(av)) / gib, @as(f64, @floatFromInt(grow_room)) / gib, @as(f64, @floatFromInt(lz.floor)) / gib, @as(f64, @floatFromInt(lz.margin)) / gib, per_pos, if (m.c.exl3 and !m.mix) "; past it the EXL3 store refuses the chunk (its slots give nothing back)" else "; past it the arena gives memory back" });
            // x-mixed: the reach with the whole native bank given back (its picks then all on the EXL3 store)
            if (m.mix) {
                const bank: u64 = @intFromFloat(ink.mix.bankBytes(m, .{ m.arena.slots(0), m.arena.slots(1) }));
                std.log.info("inkling2 rank {d}/{d}: x-mixed: with the native bank's {d:.2} GiB given back the context reaches {d} of {d} positions", .{ rank, world, @as(f64, @floatFromInt(bank)) / gib, lz.reach(grow_room + bank), lz.context });
            }
        }
        // e-step: copy windows past 8 rows are graphed too (TF_INKLING2_COPIES; prewarm captures them)
        const window_rows = s.windowRows();
        if (draft != null) m.graphs.rows = window_rows;
        // every chunk width's plans and first launches now, not in the first request (every rank: collectives)
        const tw = std.Io.Clock.awake.now(io);
        if (s.prewarm) try ink.warm.prewarm(m, draft);
        if (s.prewarm) std.log.info("inkling2 rank {d}/{d}: prewarm (chunk widths 1..{d}, one wide chunk of {d} for prompt chunks up to {d}, verify widths 2..{d}, the draft) in {d:.0} ms", .{ rank, world, max_rows, max_rows + 1, prompt_rows, @max(2, @min(m.graphs.rows, max_rows)), @as(f64, @floatFromInt(tw.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6 });
        // e-step: the windows, the draft and the commit timed now (TF_INKLING2_PRICES=timed; every rank runs them)
        const prices: ?ink.warm.Prices = if (s.prices and draft != null) try timePrices(gpa, io, ctx, dir, m, draft.?, &mine, &all, rank, world, window_rows) else null;
        try ad.bankBoot(); // x-mixed: the bank back inside its band after the prewarm's memory (nothing without it)
        e.* = .{ .gpa = gpa, .io = io, .m = m, .loader = loader, .ad = ad, .rank = rank, .world = world, .mine = mine, .all = all, .prompt_rows = prompt_rows, .ids_mine = ids_mine, .ids_all = ids_all, .logits = try gpa.alloc(f32, m.c.unpadded_vocab), .resident = resident, .draft = draft, .spec = if (draft != null) s.spec else 0, .ctx = ctx, .slots = slots, .prices = prices, .copies = s.copies and draft != null, .window_rows = if (draft != null) window_rows else 1 };
        // every rank: a sampled verify reads each row's logits on every rank (the same collectives everywhere)
        if (draft != null) e.row_logits = try gpa.alloc(f32, @max(ink.dspark.block, window_rows) * m.c.unpadded_vocab);
        // xk-rows: the store last, on every rank (its agreement and sweep are collectives), over the engine's slots
        rows_owned = true;
        e.disk = rows_mod.Store.open(gpa, ctx, m, draft, &e.slots, rank, world, dir, if (e.slots.n > 0) rows_set else blk: {
            var off = rows_set;
            off.on = false; // no slots (prompt reuse off): no book to name states, so no store
            if (rows_set.on) gpa.free(rows_set.root);
            off.root = "";
            break :blk off;
        }) catch |err| blk: {
            // every rank agreed it (rows.zig): serve on without states on disk, as before them
            std.log.warn("inkling2 rank {d}/{d}: conversation states on disk off ({t})", .{ rank, world, err });
            break :blk null;
        };
        return e;
    }

    pub fn deinit(e: *Engine) void {
        if (e.disk) |rs| rs.close(); // xk-rows
        e.recv_list.deinit(e.gpa);
        e.save_ids.deinit(e.gpa);
        e.slots.free(); // p4-cache
        if (e.draft) |dd| dd.close();
        if (e.row_logits.len > 0) e.gpa.free(e.row_logits);
        e.ad.deinit();
        e.m.loader = null;
        e.loader.close();
        e.mine.free();
        e.all.free();
        e.ids_mine.free();
        e.ids_all.free();
        e.gpa.free(e.logits);
        const gpa = e.gpa;
        e.m.close();
        gpa.destroy(e);
    }

    pub fn context(e: *const Engine) usize {
        return e.m.context;
    }

    pub fn vocab(e: *const Engine) usize {
        return e.m.c.unpadded_vocab;
    }

    /// Rank 0: the step to every rank, then run here (after the last request's end pass, if one is running).
    pub fn step(e: *Engine, h: Header, ids: []const i32) !void {
        e.settle();
        try e.stepNow(h, ids);
    }

    /// Waits for the end pass running between requests.
    pub fn settle(e: *Engine) void {
        if (e.end_thread) |t| {
            t.join();
            e.end_thread = null;
        }
    }

    /// Rank 0: the request's end step (its demand a vote, a residency pass: seconds of loads) on a thread of its
    /// own, as TideLM runs it between requests; the next step waits for it (settle), so every rank sees one order.
    pub fn endAsync(e: *Engine) void {
        e.settle();
        e.end_thread = std.Thread.spawn(.{}, endPass, .{e}) catch {
            e.commitCkpts(); // xr
            e.runSaves(); // xk-rows
            e.stepNow(.{ .kind = @intFromEnum(Kind.end) }, &.{}) catch |err| std.log.err("inkling2: the request-end residency pass failed: {t}", .{err});
            return;
        };
    }

    fn endPass(e: *Engine) void {
        e.ctx.makeCurrent() catch |err| {
            std.log.err("inkling2: the end pass's thread has no CUDA context: {t}", .{err});
            return;
        };
        e.commitCkpts(); // xr: the prompt pass's checkpoints into the index (their copies ended long since)
        e.runSaves(); // xk-rows: the request's states to disk first, then its residency pass
        e.stepNow(.{ .kind = @intFromEnum(Kind.end) }, &.{}) catch |err| std.log.err("inkling2: the request-end residency pass failed: {t}", .{err});
    }

    // ---- xk-rows: rank 0's saves and reads (steps every rank runs: rows_save, rows_load)

    /// Rank 0, at a request's release: the states the end thread saves, named by `ids` (the book's salted live ids).
    pub fn queueSaves(e: *Engine, reqs: []const SaveReq, ids: []const u32) void {
        e.saves_n = 0;
        if (e.disk == null or reqs.len == 0) return;
        e.save_ids.clearRetainingCapacity();
        e.save_ids.appendSlice(e.gpa, ids) catch return;
        for (reqs) |r| if (r.at <= ids.len and e.saves_n < e.saves.len) {
            e.saves[e.saves_n] = r;
            e.saves_n += 1;
        };
    }

    /// The queued saves (the end thread's, or the caller's when it cannot start), each a rows_save on every rank.
    fn runSaves(e: *Engine) void {
        defer e.saves_n = 0;
        for (e.saves[0..e.saves_n]) |r| {
            _ = e.saveState(r.src, e.save_ids.items[0..r.at], false) catch |err| blk: {
                std.log.warn("inkling2 rows: saving {d} tokens failed ({t})", .{ r.at, err });
                break :blk false;
            };
            // xr: the live end, on disk now (saved, or there already): the next requests' trim hint
            if (r.src == .live) if (e.disk) |rs| rs.noteLong(e.save_ids.items[0..r.at]);
        }
    }

    /// Rank 0: the state of salted ids `ids` from `src` saved on every rank's disk (`settle`: after the end thread,
    /// as every step from the engine thread); whether it was saved now (false: on disk already, or refused).
    pub fn saveState(e: *Engine, src: rows_mod.Source, ids: []const u32, settle_first: bool) !bool {
        const rs = e.disk orelse return false;
        if (settle_first) e.settle();
        if (!try rs.planSave(ids)) return false;
        const want: u32 = switch (src) {
            .live => 0,
            .slot => |i| @intCast(i + 1),
            .ckpt => return error.RowsBadSource, // lead (x6 build): a checkpoint's copy is ckptState's, as Store.save says
        };
        try e.stepList(.{ .kind = @intFromEnum(Kind.rows_save), .want = want, .arg = @intCast(ids.len) }, rs.list.items);
        return true;
    }

    /// xr, rank 0: a checkpoint of salted ids `ids` (the live state at their count) copied beside the prefill on
    /// every rank (rows_ckpt; committed by the end thread's rows_commit); whether one was started.
    pub fn ckptState(e: *Engine, ids: []const u32) !bool {
        const rs = e.disk orelse return false;
        if (!try rs.planCkpt(ids)) return false;
        try e.stepList(.{ .kind = @intFromEnum(Kind.rows_ckpt), .arg = @intCast(ids.len) }, rs.list.items);
        return true;
    }

    /// xr, rank 0: the checkpoints since the last commit committed (rows_commit on every rank), if any.
    fn commitCkpts(e: *Engine) void {
        const rs = e.disk orelse return;
        if (!rs.ckptPending()) return;
        e.stepList(.{ .kind = @intFromEnum(Kind.rows_commit) }, &.{}) catch |err| std.log.warn("inkling2 rows: committing the checkpoints failed ({t})", .{err});
    }

    /// Rank 0: stored state `i` read back into every rank's live state (`same_rows`: only its own file and window
    /// blocks: the live state holds its rows from 0); whether every rank read it (else every rank reset).
    pub fn loadState(e: *Engine, i: usize, same_rows: bool) !bool {
        const rs = e.disk orelse return false;
        e.settle();
        try rs.planLoad(i, same_rows);
        const at = rs.tokensOf(i).len;
        try e.stepList(.{ .kind = @intFromEnum(Kind.rows_load), .arg = @intCast(at) }, rs.list.items);
        return e.load_ok;
    }

    /// Rank 0: a rows step to every rank, its list in a second gather, then run here.
    fn stepList(e: *Engine, h: Header, list: []const u8) !void {
        const rs = e.disk orelse return error.NoRowsStore;
        if (list.len % 4 != 0 or list.len > rows_mod.pay_cap) return error.RowsListTooLong;
        var hh = h;
        hh.n = @intCast(list.len / 4);
        if (e.world > 1) {
            var buf: [step_bytes]u8 = @splat(0);
            @memcpy(buf[0..@sizeOf(Header)], std.mem.asBytes(&hh));
            try e.mine.uploadAsync(0, &buf, e.m.s.handle);
            try ink.forward.gatherNccl(e.m, e.m.s, e.mine.ptr, e.all.ptr, step_bytes);
        }
        try rs.exchange(list, @constCast(list));
        try e.runList(hh, list);
    }

    /// Every rank: a rows step's work, then the ranks' agreement on it.
    fn runList(e: *Engine, h: Header, list: []const u8) !void {
        const rs = e.disk orelse return error.NoRowsStore;
        const kind = std.enums.fromInt(Kind, h.kind) orelse return error.UnknownStep;
        switch (kind) {
            .rows_save => {
                const src: rows_mod.Source = if (h.want == 0) .live else .{ .slot = h.want - 1 };
                const ok = rs.save(src, list) catch |err| blk: {
                    std.log.warn("inkling2 rank {d}: saving a state of {d} tokens failed ({t})", .{ e.rank, h.arg, err });
                    break :blk false;
                };
                const all = try rs.agree(ok);
                rs.settleSave(all); // a save some rank could not finish leaves no file anywhere
            },
            .rows_ckpt => {
                // xr: no agreement here (nothing waits): a copy that could not start fails the commit on every rank
                rs.startCkpt(list) catch |err| {
                    std.log.warn("inkling2 rank {d}: a checkpoint at {d} could not start its copy ({t})", .{ e.rank, h.arg, err });
                    rs.ckpt_failed = true;
                };
                if (e.rank == 0) rs.queueCkpt();
            },
            .rows_commit => {
                const all = try rs.agree(rs.ckptOk());
                rs.commitCkpts(all);
            },
            .rows_load => {
                const ok = rs.load(list) catch |err| blk: {
                    std.log.warn("inkling2 rank {d}: reading a state of {d} tokens failed ({t})", .{ e.rank, h.arg, err });
                    break :blk false;
                };
                const all = try rs.agree(ok);
                e.load_ok = all;
                if (all) {
                    e.m.state.len = h.arg;
                } else {
                    // a read that failed on any rank left the live state in pieces: a new sequence
                    try e.m.reset();
                }
            },
            else => return error.UnknownStep,
        }
    }

    fn stepNow(e: *Engine, h: Header, ids: []const i32) !void {
        if (ids.len > max_ids and (h.kind != @intFromEnum(Kind.chunk) or ids.len > e.prompt_rows)) return error.TooManyIds;
        var hh = h;
        hh.n = @intCast(ids.len);
        if (e.world > 1) {
            var buf: [step_bytes]u8 = @splat(0);
            @memcpy(buf[0..@sizeOf(Header)], std.mem.asBytes(&hh));
            if (ids.len <= max_ids) @memcpy(buf[@sizeOf(Header)..][0 .. ids.len * 4], std.mem.sliceAsBytes(ids));
            // stream-ordered: the previous step's gather may still read `mine` (a plain upload is not ordered
            // against the model's non-blocking stream); a pageable async copy is staged before it returns
            try e.mine.uploadAsync(0, &buf, e.m.s.handle);
            try ink.forward.gatherNccl(e.m, e.m.s, e.mine.ptr, e.all.ptr, step_bytes); // NCCL: a follower idles here
            if (ids.len > max_ids) {
                // a wide prompt chunk's ids: a second gather of exactly them (the header told the others how many)
                try e.ids_mine.uploadAsync(0, std.mem.sliceAsBytes(ids), e.m.s.handle);
                try ink.forward.gatherNccl(e.m, e.m.s, e.ids_mine.ptr, e.ids_all.ptr, ids.len * 4);
            }
        }
        try e.run(hh, ids);
    }

    /// Ranks past 0: rank 0's steps until its quit.
    pub fn follow(e: *Engine) !void {
        const zero: [step_bytes]u8 = @splat(0);
        try e.mine.upload(0, &zero);
        while (true) {
            try ink.forward.gatherNccl(e.m, e.m.s, e.mine.ptr, e.all.ptr, step_bytes); // NCCL: a follower idles here
            try e.m.s.synchronize();
            var buf: [step_bytes]u8 = undefined;
            try e.all.download(0, &buf); // rank 0's segment
            const h = std.mem.bytesToValue(Header, buf[0..@sizeOf(Header)]);
            const kind = std.enums.fromInt(Kind, h.kind) orelse return error.UnknownStep;
            if (kind == .quit) return;
            if (kind == .rows_save or kind == .rows_load or kind == .rows_ckpt or kind == .rows_commit) {
                // xk-rows: the file list, rank 0's, from the second gather
                const rs = e.disk orelse return error.NoRowsStore;
                try e.recv_list.resize(e.gpa, @as(usize, h.n) * 4);
                try rs.exchange(e.recv_list.items, e.recv_list.items);
                try e.runList(h, e.recv_list.items);
                continue;
            }
            if (h.n > max_ids) {
                // a wide prompt chunk: its ids in a second gather, rank 0's segment first
                if (kind != .chunk or h.n > e.prompt_rows) return error.TooManyIds;
                try ink.forward.gatherNccl(e.m, e.m.s, e.ids_mine.ptr, e.ids_all.ptr, h.n * 4);
                try e.m.s.synchronize();
                try e.ids_all.download(0, std.mem.sliceAsBytes(e.prompt_ids[0..h.n]));
                e.run(h, e.prompt_ids[0..h.n]) catch |err| if (err != error.ContextMemory) return err; // p4-mem
                continue;
            }
            @memcpy(std.mem.sliceAsBytes(e.ids[0..h.n]), buf[@sizeOf(Header)..][0 .. h.n * 4]);
            // p4-mem: a context growth every rank refused (lazy.zig) fails rank 0's request; this rank follows on
            e.run(h, e.ids[0..h.n]) catch |err| if (err != error.ContextMemory) return err;
        }
    }

    fn run(e: *Engine, h: Header, ids: []const i32) !void {
        const kind = std.enums.fromInt(Kind, h.kind) orelse return error.UnknownStep;
        // xr: a checkpoint's copy reads the live sliding windows: one prompt chunk may run beside it, any later chunk
        // or verify (they write the rings' slots of those positions) waits for those files first
        if (kind == .chunk or kind == .verify) if (e.disk) |rs| {
            const ns = rs.gate.before(kind == .chunk);
            if (ns > 1_000_000) std.log.info("inkling2 rank {d}: a {t} step waited {d:.1} ms for a checkpoint's window copy", .{ e.rank, kind, @as(f64, @floatFromInt(ns)) / 1e6 });
        };
        switch (kind) {
            .begin => {
                // xr: a pass may rewrite positions a checkpoint's copy still reads (a new sequence, a slot's restore):
                // that copy ends first (its commit, if no end pass ran, comes with the next one)
                if (e.disk) |rs| rs.joinCkpt();
                // xr: the trim hint (rank 0's, the step's one id) for the residency's reclaim in ad.begin below
                e.trim_hint = if (ids.len > 0 and ids[0] > 0) @intCast(ids[0]) else 0;
                if (comptime @hasField(ink.adapt.Adapt, "trim_hint")) e.ad.trim_hint = e.trim_hint;
                // p4-cache: arg and want 0 (always with TF_INKLING2_PROMPT_REUSE off): a new sequence; else a resume
                if (h.arg == 0 and h.want == 0) try e.m.reset() else try e.resumeAt(h.arg, h.want);
                try e.ad.begin();
            },
            .chunk => {
                const at = e.m.state.len;
                var am: [1]i32 = undefined;
                if (h.want != 0) {
                    _ = try ink.forward.chunk(e.m, ids, .last, e.logits, &am);
                } else _ = try ink.forward.chunk(e.m, ids, .none, null, null);
                if (e.draft) |dd| try dd.absorb(e.m.spec.?.taps.ptr, at, ids.len);
                for (0..h.arg) |_| try e.ad.afterToken();
            },
            .prefilled => {
                if (h.want != 0) try e.slots.save(e.m, e.draft, h.want - 1); // p4-cache: the prompt's end kept
                try e.ad.afterPrefill(h.arg);
            },
            .token => {
                for (0..@max(h.arg, 1)) |_| try e.ad.afterToken();
                try e.ad.afterRound(); // x-mixed: the native bank's tick (nothing without it)
            },
            .verify => {
                // one forward over [pending, drafts]: every row's argmax (and, sampled, every row's logits)
                e.base = e.m.state.len;
                e.rows = ids.len;
                if (h.want != 0) {
                    _ = try ink.forward.chunk(e.m, ids, .all, e.row_logits[0 .. ids.len * e.m.c.unpadded_vocab], e.argmax[0..ids.len]);
                } else _ = try ink.forward.chunk(e.m, ids, .all, null, e.argmax[0..ids.len]);
            },
            .keep => try e.keepRows(h.arg),
            .draft => try e.draft.?.draft(ids[0], e.m.state.len, &e.drafts),
            .commit => {
                try e.keepRows(h.arg);
                for (0..h.arg) |_| try e.ad.afterToken();
                if (ids.len > 0) try e.draft.?.draft(ids[0], e.m.state.len, &e.drafts);
                try e.ad.afterRound(); // x-mixed: the native bank's tick, the stream idle after the draft's readback
            },
            .end => try e.ad.end(),
            .quit => {},
            .rows_save, .rows_load, .rows_ckpt, .rows_commit => return error.RowsStepWithoutList, // xk-rows: through stepList / follow only
        }
    }

    /// p4-cache: a pass resuming at position `at`: snapshot slot `want - 1` restored (the caches' positions below it
    /// are its prompt's), or with want 0 the state as the last request left it (its length must be `at`).
    fn resumeAt(e: *Engine, at: usize, want: u32) !void {
        // p4-mem: the positions below `at` mapped first (lazy.zig; every rank runs this step). A no-op wherever the
        // book allows a resume: the lazy context only unmaps at Adapt.begin, after this, down to max(budget, at), and
        // a pass from s drops every slot past s, so a valid slot's positions were never unmapped
        if (e.m.lazy) |lz| try lz.ensure(e.m, at);
        if (want != 0) try e.slots.restore(e.m, e.draft, want - 1);
        if (e.m.state.len != at) {
            std.log.err("inkling2 rank {d}: a pass resuming at {d} finds the sequence at {d}", .{ e.rank, at, e.m.state.len });
            return error.ResumeMismatch;
        }
    }

    /// The last verify's first `keep` rows stay: the rest rolled back, the kept rows' taps into the draft.
    fn keepRows(e: *Engine, keep: usize) !void {
        if (keep == 0 or keep > e.rows) return error.BadKeep;
        if (keep < e.rows) try e.m.rewind(e.base, keep);
        if (e.draft) |dd| try dd.absorb(e.m.spec.?.taps.ptr, e.base, keep);
        e.rows = 0;
    }

    /// Rank 0: the other ranks leave their loop.
    pub fn quit(e: *Engine) void {
        e.settle();
        if (e.world > 1 and e.rank == 0) e.step(.{ .kind = @intFromEnum(Kind.quit) }, &.{}) catch {};
    }
};

// ---- e-step: measured window prices at load (TF_INKLING2_PRICES=timed), GLM's recipe ("Window prices")

/// Rank 0's prices (the other ranks time along and return null): the table a load of this build, store and settings
/// kept (lanes.cost_cache) when rank 0 holds one, else every rank times warm.timeWindows over lanes.cost_rule.text
/// (each rank tokenizes it with the model's own tokenizer: the same ids) and rank 0 keeps the result.
fn timePrices(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, m: *Model, draft: *ink.dspark.Draft, mine: *cuda.DeviceBuffer, all: *cuda.DeviceBuffer, rank: usize, world: usize, rows: usize) !?ink.warm.Prices {
    var shape: [4096]u8 = undefined;
    var gpu: [256]u8 = undefined;
    const key: ?[32]u8 = if (rank == 0) priceKey(gpa, io, ctx, dir, m, rows, &gpu, &shape) else null;
    var kept: ?ink.warm.Prices = null;
    if (key) |k| if (lanes.cost_cache.load(ink.warm.Prices, gpa, io, k)) |p| {
        if (p.usable(rows)) kept = p;
    };
    // every rank times, or none: rank 0's verdict crosses as the room's did (its segment first)
    var time_it = kept == null;
    if (world > 1) {
        var v = [2]u64{ @intFromBool(time_it), 0 };
        try mine.upload(0, std.mem.sliceAsBytes(&v));
        try ink.forward.gatherNccl(m, m.s, mine.ptr, all.ptr, 16);
        try m.s.synchronize();
        var got: [2]u64 = undefined;
        try all.download(0, std.mem.sliceAsBytes(&got));
        time_it = got[0] != 0;
    }
    if (!time_it) {
        if (kept) |p| report(p, "kept", 0);
        return kept;
    }
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    const ids = try tok.encode(gpa, lanes.cost_rule.text);
    defer gpa.free(ids);
    const text = try gpa.alloc(i32, ids.len);
    defer gpa.free(text);
    for (text, ids) |*t, id| t.* = @intCast(id);
    const t0 = std.Io.Clock.awake.now(io);
    const p = try ink.warm.timeWindows(m, draft, text, rows);
    const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    if (rank != 0) return null;
    if (key) |k| lanes.cost_cache.save(ink.warm.Prices, gpa, io, k, p);
    report(p, "timed", ms);
    return p;
}

fn report(p: ink.warm.Prices, how: []const u8, ms: f64) void {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (p.ms[0..@intCast(p.rows)], 1..) |v, r| w.print("{s}{d}:{d:.2}", .{ if (r == 1) "" else " ", r, v }) catch break;
    std.log.info("inkling2: windows ({s}{s}): verify ms by rows {s}; a DSpark pass {d:.2} ms, a commit {d:.2} ms", .{ how, if (ms > 0) " now" else "", w.buffered(), p.draft_ms, p.commit_ms });
    if (ms > 0) std.log.info("inkling2: the window timing took {d:.0} ms", .{ms});
}

/// The prices' cache identity: the executable (lanes.cost_cache.key), the GPU, the model folder's config, the ranks,
/// the rows, the context and every TF_INKLING2_ / TF_ROCE / NCCL_ setting of this process (any of them may change a
/// window's time); null (no cache) when it does not fit or cannot be made.
fn priceKey(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, m: *const Model, rows: usize, gpu: []u8, shape: []u8) ?[32]u8 {
    var w: std.Io.Writer = .fixed(shape);
    w.print("inkling2 {s} exl3 {d} world {d} rows {d} context {d} layers {d} experts {d}", .{ dir, @intFromBool(m.c.exl3), m.world, rows, m.context, m.c.layers, m.c.experts }) catch return null;
    var names: [256][]const u8 = undefined;
    var n: usize = 0;
    var i: usize = 0;
    while (std.c.environ[i]) |e| : (i += 1) {
        const kv = std.mem.span(e);
        const take = std.mem.startsWith(u8, kv, "TF_INKLING2_") or std.mem.startsWith(u8, kv, "TF_ROCE") or std.mem.startsWith(u8, kv, "NCCL_");
        if (!take) continue;
        if (n == names.len) return null;
        names[n] = kv;
        n += 1;
    }
    std.mem.sort([]const u8, names[0..n], {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    for (names[0..n]) |kv| w.print(" {s}", .{kv}) catch return null;
    const parts = [_][]const u8{ "inkling2-cuda-windows", ctx.name(gpu) catch "gpu", w.buffered() };
    return lanes.cost_cache.key(gpa, io, &parts) catch null;
}
// ---- end e-step
