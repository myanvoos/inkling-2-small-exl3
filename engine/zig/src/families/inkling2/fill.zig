//! Expert loads into the arena's slots: the boot fill, a residency pass's swaps and exact mode's transient loads all go
//! through `Loader.load`. A batch of (layer, expert, slot) jobs ("a stage") has its stored byte ranges
//! (experts.Store.views) read past the page cache (O_DIRECT, reader threads, the mapping never touched) straight into
//! page-locked host memory the GPU maps; the device packers (inkling2.cu ink2_pack_nv / ink2_pack_mx: the byte
//! permutation of Store.packViews, gated byte-equal against the reference packers) write each slot from there on a
//! side stream while the next stage is read into the other buffer. Each job is marked resident on the host LUT (the
//! caller publishes); `load` returns once every slot has landed.
//!
//! Memory stays flat: two stage buffers (TF_INKLING2_STAGE_MIB each, default 8 MXFP8 experts' bytes) are the loader's
//! whole host footprint, reused stage to stage; no pack buffer, no upload copy and no page-cache page is made. Every
//! stage's read waits for the pace (MemAvailable and PSI, under the hosts' mem_guard).
//! TF_INKLING2_CPU_PACK=1: the host path instead (core direct_gather, Store.packViews on threads, Model.put's
//! synchronous uploads), as before the device packers.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const experts_mod = @import("experts.zig");
const model_mod = @import("model.zig");

const gather_mod = core.direct_gather;
const direct_io = core.direct_io;

pub const Job = struct { layer: u32, expert: u32, slot: u32 };

/// core/paced_read.zig's pacing, ported (that module reads a Mac's sysctl counters and is not exported from core):
/// its Memory and its waitFor loop unchanged, under a Linux gauge below.
const Memory = struct {
    free: u64,
    reclaimable: u64,

    fn room(m: Memory) u64 {
        return m.free +| m.reclaimable;
    }
};

/// core/paced_read.zig waitFor: waits while `gauge`'s room is under `want`; still short after `limit_ms`:
/// error.MemoryShort, its reading in `short`.
fn waitFor(gauge: anytype, want: u64, limit_ms: u32, short: *Memory) error{MemoryShort}!void {
    var waited: u32 = 0;
    while (gauge.reading()) |m| : (waited += 2) {
        if (m.room() >= want) return;
        if (waited >= limit_ms) {
            short.* = m;
            return error.MemoryShort;
        }
        gauge.pause(2);
    }
}

/// The pace a batch's read waits for (paced_read's waitFor, on Linux's counters): MemAvailable at least
/// `avail_bytes` and memory PSI (some, avg10) under `psi`. Both sit inside the hosts' mem_guard (it acts at PSI 40
/// and 1280 MiB): reads stop well before the guard would. TF_INKLING2_PACE_PSI / TF_INKLING2_PACE_MIB override.
pub const Pace = struct {
    psi: f64 = 20,
    avail_bytes: u64 = 2 << 30,
    wait_ms: u32 = 30_000, // paced_read.wait_ms

    pub fn fromEnv() Pace {
        var p: Pace = .{};
        if (std.c.getenv("TF_INKLING2_PACE_PSI")) |v| p.psi = std.fmt.parseFloat(f64, std.mem.span(v)) catch p.psi;
        if (std.c.getenv("TF_INKLING2_PACE_MIB")) |v| p.avail_bytes = (std.fmt.parseInt(u64, std.mem.span(v), 10) catch (p.avail_bytes >> 20)) << 20;
        return p;
    }
};

/// Linux's reading for core paced_read.waitFor: MemAvailable as the room, none at all while PSI is at the limit.
const Gauge = struct {
    pace: Pace,
    stats: *Stats,

    fn reading(g: Gauge) ?Memory {
        const avail = readAvail() orelse return null;
        const psi = readPsi() orelse 0;
        g.stats.min_avail = @min(g.stats.min_avail, avail);
        g.stats.max_psi = @max(g.stats.max_psi, psi);
        return .{ .free = if (psi >= g.pace.psi) 0 else avail, .reclaimable = 0 };
    }

    fn pause(g: Gauge, ms: u32) void {
        g.stats.paused_ms += ms;
        const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(@as(u64, ms) * std.time.ns_per_ms) };
        _ = std.c.nanosleep(&ts, null);
    }
};

fn readProc(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const r = std.c.read(fd, buf[n..].ptr, buf.len - n);
        if (r <= 0) break;
        n += @intCast(r);
    }
    return buf[0..n];
}

/// MemAvailable in bytes.
pub fn readAvail() ?u64 {
    var buf: [8192]u8 = undefined;
    const text = readProc("/proc/meminfo", &buf) orelse return null;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \tkB");
        return (std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null) * 1024;
    };
    return null;
}

/// Memory pressure, "some avg10" of /proc/pressure/memory.
pub fn readPsi() ?f64 {
    var buf: [512]u8 = undefined;
    const text = readProc("/proc/pressure/memory", &buf) orelse return null;
    const at = std.mem.indexOf(u8, text, "some avg10=") orelse return null;
    var f = std.mem.tokenizeAny(u8, text[at + "some avg10=".len ..], " \n");
    return std.fmt.parseFloat(f64, f.next() orelse return null) catch null;
}

pub const Stats = struct {
    experts: usize = 0,
    bytes: u64 = 0, // stored bytes read
    read_ns: u64 = 0, // reads (the pace's waits included)
    pack_ns: u64 = 0, // host path: packs; device path: waits for a stage's packs to land
    upload_ns: u64 = 0, // host path: uploads; device path: pack launches
    paused_ms: u64 = 0, // the pace's waits
    max_psi: f64 = 0, // seen by the pace
    min_avail: u64 = std.math.maxInt(u64),
};

/// Where a job's slot lives on the device, per format pool (0 NVFP4, 1 MXFP8): packed words [slot][words] of up and
/// down, NVFP4's global scales (up [slot][2], down [slot]).
pub const Target = struct {
    up: [2]u64,
    down: [2]u64,
    up_scale: u64,
    down_scale: u64,
    words: [2][2]usize,

    pub fn ofArena(a: *const model_mod.Arena) Target {
        return .{ .up = .{ a.nv_up.ptr, a.mx_up.ptr }, .down = .{ a.nv_down.ptr, a.mx_down.ptr }, .up_scale = a.nv_up_scale.ptr, .down_scale = a.nv_down_scale.ptr, .words = a.words };
    }
};

const max_stage_jobs = 32;

/// x-mixed (a gate): TF_INKLING2_MIX_FAIL_READ=R makes every background load of rank R fail before it reads (one
/// rank's failure: both ranks must then map the same experts); unset or not a number: none. Read once.
var fail_read_rank: ?i64 = null;
fn failRead(rank: u32) bool {
    if (fail_read_rank == null) {
        fail_read_rank = if (std.c.getenv("TF_INKLING2_MIX_FAIL_READ")) |x| std.fmt.parseInt(i64, std.mem.span(x), 10) catch -1 else -1;
    }
    return fail_read_rank.? == rank;
}
/// A stored range's bytes a read request takes at most (core direct_gather's chunk).
const piece_bytes = gather_mod.chunk;

/// One stage: its jobs' stored ranges in mapped pinned memory, the device packs that read them, the event they end on.
const Stage = struct {
    host: cuda.HostBuffer,
    win: []align(direct_io.alignment) u8, // the 4 KiB-aligned window the reads land in (host stays the owner)
    dev: u64,
    ev: cuda.Event,
    bad: []u32, // [max_stage_jobs] mapped: the packs' refusal bits
    bad_dev: u64,
    jobs: [max_stage_jobs]Job = undefined,
    n: usize = 0,
    busy: bool = false,
};

const Piece = struct { file: u32, lo: u64, len: usize, dst: usize };

const Gpu = struct {
    d: *const cuda.Driver,
    mod: cuda.Module,
    nv: cuda.Function,
    mx: cuda.Function,
    s: cuda.Stream,
    stages: [2]Stage,
    flags: cuda.HostBuffer, // both stages' bad words
    pieces: std.ArrayList(Piece) = .empty,
};

pub const Loader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    ck: core.checkpoint.Checkpoint,
    store: experts_mod.Store,
    gather: gather_mod.Gather,
    sources: []gather_mod.Source,
    files: []direct_io.File, // every shard, opened once (direct where allowed): the device path's reads
    direct: bool,
    workers: usize,
    readers: usize,
    slots: []experts_mod.Slot = &.{}, // host path: a worker's pack buffers (sized for the larger format, cut per job)
    up_words: usize,
    down_words: usize,
    views: []experts_mod.Store.Views,
    ptrs: []*[]const u8,
    errs: []?anyerror,
    stats: Stats = .{},
    pace: Pace = .{},
    cpu_pack: bool,
    stage_bytes: usize,
    gpu: ?*Gpu = null,
    bg: ?*Background = null,
    /// The device packers' kernel image (inkling2.cu's fatbin, embedded); pack_main may hand the same image from a file.
    image: []const u8 = &.{},

    /// The checkpoint at `dir` for rank `rank` of `world`: its expert tensors mapped (untouched), every NVFP4 global
    /// scale read and checked, every shard opened (O_DIRECT where allowed). `workers` experts a host-path batch,
    /// `readers` read threads. Refuses a shard the file system will not open O_DIRECT unless
    /// TF_INKLING2_ALLOW_CACHED_READS=1.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, c: *const config.Config, dir: []const u8, rank: usize, world: usize, workers: usize, readers: usize) !*Loader {
        const l = try gpa.create(Loader);
        errdefer gpa.destroy(l);
        l.* = undefined;
        l.gpa = gpa;
        l.io = io;
        l.stats = .{};
        l.pace = Pace.fromEnv();
        l.gpu = null;
        l.bg = null;
        l.image = cuda.kernels.inkling2;
        l.slots = &.{};
        l.cpu_pack = envIs("TF_INKLING2_CPU_PACK", "1");
        l.ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
        errdefer l.ck.close();
        l.store = .{ .c = c, .ck = &l.ck, .rank = @intCast(rank), .world = @intCast(world) };
        try l.store.loadGlobals(gpa);
        const files = l.ck.files.items;
        if (files.len > gather_mod.max_sources) return error.TooManyShards;
        l.sources = try gpa.alloc(gather_mod.Source, files.len);
        l.files = try gpa.alloc(direct_io.File, files.len);
        l.direct = true;
        var opened: usize = 0;
        errdefer for (l.files[0..opened]) |*f| f.close();
        for (files, l.sources, l.files) |*f, *s, *df| {
            s.* = .{ .path = f.path, .memory = f.map.memory };
            df.* = try direct_io.File.open(f.path);
            opened += 1;
            if (!df.direct) l.direct = false;
        }
        if (!l.direct and !allowCached()) {
            std.log.err("inkling2: a shard of {s} does not open O_DIRECT; expert reads would fill the page cache (TF_INKLING2_ALLOW_CACHED_READS=1 to accept)", .{dir});
            return error.NoDirectReads;
        }
        l.gather = gather_mod.Gather.init(gpa, readers);
        l.workers = @max(1, workers);
        l.readers = @max(1, @min(readers, gather_mod.max_readers));
        const H = c.hidden;
        const Il = c.inter / world;
        const nv = experts_mod.slotWords(.nvfp4, H, Il);
        const mx = experts_mod.slotWords(.mxfp8, H, Il);
        l.up_words = @max(nv[0], mx[0]);
        l.down_words = @max(nv[1], mx[1]);
        // a stage holds 8 of the largest format's experts unless TF_INKLING2_STAGE_MIB says otherwise
        var most: usize = 0;
        // the EXL3 export's experts are exl3.zig's (its own batch files, every expert resident): this loader never
        // reads one, and the view's index holds none of their tensors
        if (!c.exl3) for (0..c.layers) |li| {
            if (li > 0 and c.formats[li] == c.formats[li - 1]) continue;
            most = @max(most, try l.rawSpan(@intCast(li), 0));
        };
        l.stage_bytes = 8 * most;
        if (std.c.getenv("TF_INKLING2_STAGE_MIB")) |v| l.stage_bytes = (std.fmt.parseInt(usize, std.mem.span(v), 10) catch (l.stage_bytes >> 20)) << 20;
        l.stage_bytes = std.mem.alignForward(usize, @max(l.stage_bytes, most), direct_io.alignment);
        l.views = try gpa.alloc(experts_mod.Store.Views, @max(l.workers, max_stage_jobs));
        l.ptrs = try gpa.alloc(*[]const u8, 4 * l.workers);
        l.errs = try gpa.alloc(?anyerror, l.workers);
        return l;
    }

    pub fn close(l: *Loader) void {
        const gpa = l.gpa;
        if (l.finishBackground()) |jobs| {
            if (jobs) |j| gpa.free(j);
        } else |_| {}
        if (l.gpu) |g| {
            g.s.synchronize() catch {};
            for (&g.stages) |*st| {
                st.ev.deinit();
                st.host.free();
            }
            g.flags.free();
            g.s.deinit();
            g.mod.unload();
            g.pieces.deinit(gpa);
            gpa.destroy(g);
        }
        for (l.slots) |s| {
            gpa.free(s.up);
            gpa.free(s.down);
        }
        if (l.slots.len > 0) gpa.free(l.slots);
        gpa.free(l.views);
        gpa.free(l.ptrs);
        gpa.free(l.errs);
        for (l.files) |*f| f.close();
        gpa.free(l.files);
        gpa.free(l.sources);
        if (l.store.globals) |g| gpa.free(g);
        l.gather.deinit();
        l.ck.close();
        gpa.destroy(l);
    }

    fn allowCached() bool {
        return envIs("TF_INKLING2_ALLOW_CACHED_READS", "1");
    }

    fn envIs(name: [*:0]const u8, want: []const u8) bool {
        const v = std.c.getenv(name) orelse return false;
        return std.mem.eql(u8, std.mem.span(v), want);
    }

    fn now(l: *const Loader) std.Io.Timestamp {
        return std.Io.Clock.awake.now(l.io);
    }

    fn since(l: *const Loader, t: std.Io.Timestamp) u64 {
        return @intCast(t.durationTo(l.now()).toNanoseconds());
    }

    /// Every job's expert packed into its slot of `m`'s arena and marked resident on its host LUT (the caller
    /// publishes). Returns once every slot has landed.
    pub fn load(l: *Loader, m: *model_mod.Model, jobs: []const Job) !void {
        if (l.bg != null) return error.BackgroundBusy; // adapt.zig finishes a background load before any other
        if (l.cpu_pack) return l.loadHost(m, jobs);
        try l.loadInto(m.d, Target.ofArena(&m.arena), jobs, m);
    }

    /// The wait every stage's read makes for the pace (paced_read's waitFor on Linux's gauge).
    fn paceWait(l: *Loader) !void {
        var short: Memory = undefined;
        waitFor(Gauge{ .pace = l.pace, .stats = &l.stats }, l.pace.avail_bytes, l.pace.wait_ms, &short) catch {
            std.log.err("inkling2: expert reads stopped after {d} s: MemAvailable {d:.2} GiB / PSI at or over {d:.0} (the pace keeps {d:.2} GiB and PSI under {d:.0}; mem_guard acts at 1.25 GiB and 40)", .{ l.pace.wait_ms / 1000, @as(f64, @floatFromInt(short.free)) / (1 << 30), l.pace.psi, @as(f64, @floatFromInt(l.pace.avail_bytes)) / (1 << 30), l.pace.psi });
            return error.MemoryShort;
        };
    }

    /// The device path's resources, made on first use: the packers' module, a side stream, two mapped stages.
    fn ensureGpu(l: *Loader, d: *const cuda.Driver) !*Gpu {
        if (l.gpu) |g| return g;
        if (l.image.len == 0) return error.BuiltWithoutKernels;
        const g = try l.gpa.create(Gpu);
        errdefer l.gpa.destroy(g);
        g.* = undefined;
        g.d = d;
        g.pieces = .empty;
        g.mod = try cuda.Module.load(d, l.image);
        errdefer g.mod.unload();
        g.nv = try g.mod.function("ink2_pack_nv");
        g.mx = try g.mod.function("ink2_pack_mx");
        g.s = try cuda.Stream.init(d, true);
        errdefer g.s.deinit();
        g.flags = try cuda.HostBuffer.allocMapped(d, 2 * max_stage_jobs * 4);
        errdefer g.flags.free();
        const flags_dev = try g.flags.device();
        const words = g.flags.slice(u32);
        var made: usize = 0;
        errdefer for (g.stages[0..made]) |*st| {
            st.ev.deinit();
            st.host.free();
        };
        for (&g.stages, 0..) |*st, i| {
            var host = try cuda.HostBuffer.allocMapped(d, l.stage_bytes + direct_io.alignment);
            errdefer host.free();
            const base = @intFromPtr(host.bytes.ptr);
            const off = std.mem.alignForward(usize, base, direct_io.alignment) - base; // cuMemHostAlloc promises 256 B, not 4 KiB
            const win: []align(direct_io.alignment) u8 = @alignCast(host.bytes[off..][0..l.stage_bytes]);
            st.* = .{ .host = host, .win = win, .dev = try host.device() + off, .ev = try cuda.Event.init(d, false), .bad = words[i * max_stage_jobs ..][0..max_stage_jobs], .bad_dev = flags_dev + i * max_stage_jobs * 4 };
            made += 1;
        }
        l.gpu = g;
        return g;
    }

    /// The span (4 KiB-aligned ends) of one job's four stored ranges, as a stage lays them out.
    fn rawSpan(l: *const Loader, layer: u32, e: usize) !usize {
        const v = try l.store.views(layer, e);
        var n: usize = 0;
        for ([_][]const u8{ v.w13, v.s13, v.w2, v.s2 }) |r| n += direct_io.span(@intFromPtr(r.ptr), r.len);
        return n;
    }

    fn source(l: *const Loader, v: []const u8) !usize {
        const p = @intFromPtr(v.ptr);
        for (l.sources, 0..) |s, i| {
            const b = @intFromPtr(s.memory.ptr);
            if (p >= b and p + v.len <= b + s.memory.len) return i;
        }
        return error.UnmappedView;
    }

    /// `jobs` packed into `t`'s slots by the device packers, stage by stage: a stage's reads overlap the previous
    /// stage's packs. With `m`, each job is marked resident on its host LUT as its pack is launched (a slot's previous
    /// owner unmapped first); on an error every slot not proven written is unmapped again, so the host LUT holds only
    /// what landed.
    pub fn loadInto(l: *Loader, d: *const cuda.Driver, t: Target, jobs: []const Job, m: ?*model_mod.Model) !void {
        if (l.bg != null) return error.BackgroundBusy; // the caller finishes a background load first
        return l.loadJobs(d, t, jobs, m, null);
    }

    fn loadJobs(l: *Loader, d: *const cuda.Driver, t: Target, jobs: []const Job, m: ?*model_mod.Model, progress: ?*std.atomic.Value(usize)) !void {
        const g = try l.ensureGpu(d);
        errdefer {
            g.s.synchronize() catch {};
            for (&g.stages) |*st| {
                if (st.busy) if (m) |mm| for (st.jobs[0..st.n]) |j| mm.evict(fmtIndex(mm.c.act[j.layer]), j.slot);
                st.busy = false;
                st.n = 0;
            }
        }
        var i: usize = 0;
        var si: usize = 0;
        while (i < jobs.len) : (si ^= 1) {
            const st = &g.stages[si];
            if (st.busy) try l.retire(g, st, m, progress);
            // the stage's jobs: as many as its bytes hold
            var used: usize = 0;
            st.n = 0;
            g.pieces.clearRetainingCapacity();
            const t0 = l.now();
            while (i < jobs.len and st.n < max_stage_jobs) {
                const j = jobs[i];
                const v = try l.store.views(j.layer, j.expert);
                var need: usize = 0;
                for ([_][]const u8{ v.w13, v.s13, v.w2, v.s2 }) |r| need += direct_io.span(@intFromPtr(r.ptr) - @intFromPtr(l.sources[try l.source(r)].memory.ptr), r.len);
                if (st.n > 0 and used + need > l.stage_bytes) break;
                if (need > l.stage_bytes) return error.StageTooSmall;
                // each range's span at `used`, its view re-pointed into the stage
                var rv: [4][]const u8 = .{ v.w13, v.s13, v.w2, v.s2 };
                for (&rv) |*r| {
                    const f = try l.source(r.*);
                    const lo: u64 = @intFromPtr(r.ptr) - @intFromPtr(l.sources[f].memory.ptr);
                    const alo = std.mem.alignBackward(u64, lo, direct_io.alignment);
                    const end = lo + r.len;
                    var at = alo;
                    while (at < end) : (at += piece_bytes) {
                        try g.pieces.append(l.gpa, .{ .file = @intCast(f), .lo = at, .len = @intCast(@min(piece_bytes, end - at)), .dst = used + @as(usize, @intCast(at - alo)) });
                    }
                    l.stats.bytes += r.len;
                    r.* = st.win[used + @as(usize, @intCast(lo - alo)) ..][0..r.len];
                    used += direct_io.span(lo, r.len);
                }
                l.views[st.n] = .{ .w13 = rv[0], .s13 = rv[1], .w2 = rv[2], .s2 = rv[3] };
                st.jobs[st.n] = j;
                st.n += 1;
                i += 1;
            }
            try l.paceWait();
            try l.readPieces(st, g.pieces.items);
            l.stats.read_ns += l.since(t0);
            // the packs, on the side stream (they read the stage through its device mapping)
            const t1 = l.now();
            const H = l.store.c.hidden;
            const Il = l.store.inter();
            const I = l.store.c.inter;
            @memset(st.bad[0..st.n], 0);
            st.busy = true; // from here an error unmaps the stage's slots
            for (st.jobs[0..st.n], 0..) |j, b| {
                const fmt = l.store.c.formats[j.layer];
                const fi = fmtIndex(fmt);
                const v = l.views[b];
                const off = struct {
                    fn of(stage: *const Stage, r: []const u8) u64 {
                        return stage.dev + (@intFromPtr(r.ptr) - @intFromPtr(stage.win.ptr));
                    }
                };
                if (m) |mm| mm.evict(fi, j.slot);
                var a: cuda.Args = .{};
                a.add(t.up[fi] + j.slot * t.words[fi][0] * 4);
                a.add(t.down[fi] + j.slot * t.words[fi][1] * 4);
                for ([_][]const u8{ v.w13, v.s13, v.w2, v.s2 }) |r| a.add(off.of(st, r));
                for ([_]usize{ H, Il, I, l.store.rank }) |x| a.add(@as(c_int, @intCast(x)));
                const nb = (Il / 32) * (H / 32);
                switch (fmt) {
                    .nvfp4 => {
                        const gl = l.store.globals.?[j.layer * l.store.c.experts + j.expert];
                        a.add(t.up_scale + j.slot * 8);
                        a.add(t.down_scale + j.slot * 4);
                        a.add(gl[0]);
                        a.add(gl[1]);
                        a.add(st.bad_dev + b * 4);
                        try cuda.launch.launch(g.nv, .{ .grid = .{ .x = @intCast(3 * nb), .y = 1, .z = 1 }, .block = .{ .x = 160, .y = 1, .z = 1 } }, g.s, &a);
                    },
                    .mxfp8 => {
                        a.add(st.bad_dev + b * 4);
                        try cuda.launch.launch(g.mx, .{ .grid = .{ .x = @intCast(3 * nb), .y = 1, .z = 1 }, .block = .{ .x = 288, .y = 1, .z = 1 } }, g.s, &a);
                    },
                    .exl3 => return error.Exl3Slot,
                }
                if (m) |mm| mm.place(j.layer, j.expert, fi, j.slot);
            }
            try st.ev.record(g.s);
            l.stats.upload_ns += l.since(t1);
            l.stats.experts += st.n;
            // the previous stage's packs (they ran while this stage was read) retired: its jobs have landed
            const other = &g.stages[si ^ 1];
            if (other.busy) try l.retire(g, other, m, progress);
        }
        for ([_]usize{ si ^ 1, si }) |k| if (g.stages[k].busy) try l.retire(g, &g.stages[k], m, progress);
    }

    /// A stage's packs waited for and their refusals read: a refused expert is an error (its slot unmapped). Its jobs
    /// count into `progress` (jobs land in order: stages are retired in the order they were launched).
    fn retire(l: *Loader, g: *Gpu, st: *Stage, m: ?*model_mod.Model, progress: ?*std.atomic.Value(usize)) !void {
        _ = g;
        _ = m; // a refused stage's slots are unmapped by loadJobs' errdefer
        const t = l.now();
        try st.ev.synchronize();
        l.stats.pack_ns += l.since(t);
        for (st.jobs[0..st.n], st.bad[0..st.n]) |j, bad| if (bad != 0) {
            if (bad & 1 != 0) std.log.err("layer {d} expert {d}: NaN or negative e4m3 block scale (NVFP4)", .{ j.layer, j.expert });
            if (bad & 2 != 0) std.log.err("layer {d} expert {d}: NaN e4m3 weight byte (MXFP8)", .{ j.layer, j.expert });
            if (bad & 4 != 0) std.log.err("layer {d} expert {d}: UE8M0 exponent 0xFF (MXFP8)", .{ j.layer, j.expert });
            return if (bad & 2 != 0) error.BadWeight else error.BadScale; // the errdefer unmaps the stage
        };
        if (progress) |p| _ = p.fetchAdd(st.n, .release);
        st.busy = false;
        st.n = 0;
    }

    // ---- background loads: a pass's packs while the caller's forwards run (adapt.zig maps what landed)

    /// `jobs` (copied) loaded on a thread of its own into `t`'s slots, which no forward may read until they land
    /// (the caller unmapped their owners and published first). `context`: the CUDA context the thread makes current.
    /// The host LUT is not touched: `landed` says how many of `jobs`, in order, are in their slots.
    pub fn startBackground(l: *Loader, d: *const cuda.Driver, context: cuda.abi.Context, t: Target, jobs: []const Job) !void {
        if (l.bg != null) return error.BackgroundBusy;
        _ = try l.ensureGpu(d);
        const b = try l.gpa.create(Background);
        errdefer l.gpa.destroy(b);
        b.* = .{ .l = l, .d = d, .context = context, .t = t, .jobs = try l.gpa.dupe(Job, jobs) };
        errdefer l.gpa.free(b.jobs);
        b.thread = try std.Thread.spawn(.{}, Background.run, .{b});
        l.bg = b;
    }

    /// Jobs of the background load that have landed (a prefix of its jobs); 0 when none runs.
    pub fn landed(l: *const Loader) usize {
        const b = l.bg orelse return 0;
        return b.done.load(.acquire);
    }

    /// x-mixed: whether the background load's thread is done (every job landed, or it failed); false when none runs.
    pub fn backgroundFinished(l: *const Loader) bool {
        const b = l.bg orelse return false;
        return b.finished.load(.acquire);
    }

    /// x-mixed: the background load joined and released: how many of its jobs landed (a prefix: stages retire in
    /// order) and its error, if any. The caller maps what every rank landed (adapt.zig's bank settle), so a failed read
    /// on one rank leaves the same experts native on every rank.
    pub fn joinBackground(l: *Loader) struct { landed: usize, failed: ?anyerror } {
        const b = l.bg orelse return .{ .landed = 0, .failed = null };
        b.thread.join();
        l.bg = null;
        defer {
            l.gpa.free(b.jobs);
            l.gpa.destroy(b);
        }
        return .{ .landed = b.done.load(.acquire), .failed = b.failed };
    }

    /// The background load joined (every job landed, or its error); its jobs, owned by the caller, or null.
    pub fn finishBackground(l: *Loader) !?[]Job {
        const b = l.bg orelse return null;
        b.thread.join();
        l.bg = null;
        defer l.gpa.destroy(b);
        if (b.failed) |err| {
            l.gpa.free(b.jobs);
            return err;
        }
        return b.jobs;
    }

    const Background = struct {
        l: *Loader,
        d: *const cuda.Driver,
        context: cuda.abi.Context,
        t: Target,
        jobs: []Job,
        thread: std.Thread = undefined,
        done: std.atomic.Value(usize) = .init(0),
        failed: ?anyerror = null,
        finished: std.atomic.Value(bool) = .init(false), // x-mixed: the thread is done (backgroundFinished)

        fn run(b: *Background) void {
            defer b.finished.store(true, .release); // x-mixed: after `failed` is set
            if (failRead(b.l.store.rank)) b.failed = error.InjectedReadFailure; // x-mixed: TF_INKLING2_MIX_FAIL_READ (a gate)
            if (b.failed != null) return;
            b.d.check(b.d.api.cuCtxSetCurrent(b.context), "cuCtxSetCurrent") catch |err| {
                b.failed = err;
                return;
            };
            cuda.copy_source.setCurrent(b.context);
            b.l.loadJobs(b.d, b.t, b.jobs, null, &b.done) catch |err| {
                b.failed = err;
            };
        }
    };

    const ReadWork = struct {
        l: *Loader,
        st: *Stage,
        pieces: []const Piece,
        next: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(bool) = .init(false),
        first: anyerror = error.ReadFailed,

        fn run(w: *ReadWork) void {
            while (!w.failed.load(.monotonic)) {
                const i = w.next.fetchAdd(1, .monotonic);
                if (i >= w.pieces.len) return;
                const p = w.pieces[i];
                const dst: direct_io.Buffer = @alignCast(w.st.win[p.dst..][0..direct_io.span(p.lo, p.len)]);
                _ = w.l.files[p.file].read(dst, p.lo, p.len) catch |err| {
                    if (w.failed.cmpxchgStrong(false, true, .monotonic, .monotonic) == null) w.first = err;
                    return;
                };
            }
        }
    };

    /// The pieces read into the stage by `readers` threads (the stage's buffer is idle: its packs were retired).
    fn readPieces(l: *Loader, st: *Stage, pieces: []const Piece) !void {
        var w: ReadWork = .{ .l = l, .st = st, .pieces = pieces };
        var threads: [gather_mod.max_readers]?std.Thread = @splat(null);
        const n = @min(l.readers, pieces.len);
        for (threads[1..@max(n, 1)]) |*t| t.* = std.Thread.spawn(.{}, ReadWork.run, .{&w}) catch null;
        w.run();
        for (threads[1..@max(n, 1)]) |t| if (t) |th| th.join();
        if (w.failed.load(.monotonic)) return w.first;
    }

    /// The host path (TF_INKLING2_CPU_PACK=1): direct_gather reads, packViews on worker threads, Model.put.
    pub fn loadHost(l: *Loader, m: *model_mod.Model, jobs: []const Job) !void {
        if (l.slots.len == 0) {
            l.slots = try l.gpa.alloc(experts_mod.Slot, l.workers);
            for (l.slots) |*s| s.* = .{ .fmt = .nvfp4, .up = try l.gpa.alloc(u32, l.up_words), .down = try l.gpa.alloc(u32, l.down_words) };
        }
        const H = m.c.hidden;
        const Il = m.inter();
        var i: usize = 0;
        while (i < jobs.len) {
            const n = @min(l.workers, jobs.len - i);
            const batch = jobs[i .. i + n];
            // stored byte ranges, read past the page cache (views re-pointed at the gather arena's copies)
            const t0 = l.now();
            for (batch, 0..) |j, b| {
                l.views[b] = try l.store.views(j.layer, j.expert);
                const v = &l.views[b];
                l.ptrs[4 * b + 0] = &v.w13;
                l.ptrs[4 * b + 1] = &v.s13;
                l.ptrs[4 * b + 2] = &v.w2;
                l.ptrs[4 * b + 3] = &v.s2;
                l.stats.bytes += v.w13.len + v.s13.len + v.w2.len + v.s2.len;
            }
            try l.paceWait();
            if (l.direct) try l.gather.read(l.sources, l.ptrs[0 .. 4 * n]);
            l.stats.read_ns += l.since(t0);
            // packs, one a worker
            const t1 = l.now();
            for (batch, 0..) |j, b| {
                const fmt = m.c.act[j.layer]; // the arena's format (Config.act: formats for the release)
                const w = experts_mod.slotWords(fmt, H, Il);
                l.slots[b].fmt = fmt;
                l.slots[b].up = l.slots[b].up.ptr[0..w[0]];
                l.slots[b].down = l.slots[b].down.ptr[0..w[1]];
                l.errs[b] = null;
            }
            var threads: [64]?std.Thread = @splat(null);
            for (1..n) |b| threads[b] = std.Thread.spawn(.{}, packOne, .{ l, batch[b], b }) catch null;
            packOne(l, batch[0], 0);
            for (1..n) |b| if (threads[b]) |t| t.join() else packOne(l, batch[b], b);
            for (batch, 0..) |j, b| if (l.errs[b]) |err| {
                std.log.err("inkling2: layer {d} expert {d}: pack failed: {t}", .{ j.layer, j.expert, err });
                return err;
            };
            l.stats.pack_ns += l.since(t1);
            const t2 = l.now();
            for (batch, 0..) |j, b| try m.put(j.layer, j.expert, j.slot, &l.slots[b]);
            l.stats.upload_ns += l.since(t2);
            l.stats.experts += n;
            i += n;
        }
        // restore the buffers' full lengths for the next batch's cuts
        for (l.slots) |*s| {
            s.up = s.up.ptr[0..l.up_words];
            s.down = s.down.ptr[0..l.down_words];
        }
    }

    fn packOne(l: *Loader, j: Job, b: usize) void {
        l.store.packViews(std.heap.smp_allocator, j.layer, j.expert, l.views[b], &l.slots[b]) catch |err| {
            l.errs[b] = err;
        };
    }
};

fn fmtIndex(f: config.Format) usize {
    return if (f == .nvfp4) 0 else 1;
}
