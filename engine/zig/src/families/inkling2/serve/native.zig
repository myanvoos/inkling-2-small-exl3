//! Inkling-2-Small behind the native CUDA registry: the release checkpoint as stored (NVFP4 experts in layers 0-39,
//! MXFP8 in 40-47, bf16 attention), at TP over hosts (--tp, --rank; NCCL_COMM_ID names rank 0), with TideLM's
//! expert residency (the arena sized from what MemAvailable leaves above TF_INKLING2_FLOOR_GIB, filled from the
//! TF_INKLING2_PRIOR prior, adapted by demand; misses take the next-best resident expert). Rank 0 serves; the other
//! ranks follow its steps (engine.zig).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const ink = @import("inkling2");
const engine = @import("engine.zig");
const backend_mod = @import("backend.zig");
const reuse = @import("reuse.zig"); // p4-cache

pub const model_type = "inkling2";
pub const formats: []const []const u8 = &.{"nvfp4-mxfp8"};
pub const default_context: i64 = 4096;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = engine.max_rows; // the narrowest; Loaded.prompt_rows is the width the load agreed
pub const snap_lookahead: u32 = 0;
pub const explains_open = true;
pub const keeps_no_prompts = true;

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    tp: u32 = 1,
    rank: u32 = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29551,
};

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    prompt_rows: u32, // rows a prompt chunk takes (engine.Settings.prompt_rows within memory, every rank's)
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    target: *anyopaque,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
};

/// The bytes the load ledger counts before the family opens: this rank's dense share (the experts' arena is sized
/// by the family from what is left, above the floor).
pub fn placementWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, tp: u32, rank: u32, drafts: bool) !u64 {
    _ = drafts;
    const c = try ink.config.read(a, io, dir);
    try c.checkSplit(tp);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const entries = try ink.plan.plan(arena.allocator(), &c, rank, tp);
    return ink.plan.bytesOf(entries, .dense);
}

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    b: backend_mod.Ink,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    const world: usize = o.tp;
    const rank: usize = o.rank;
    if (world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    const group: ?cuda.tp_group.Group = if (world > 1) try cuda.tp_group.Group.open(ctx.d, rank, world) else null;
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = try engine.Engine.load(gpa, io, ctx, dir, rank, world, group, engine.Settings.fromEnv(o.context)), .b = undefined };
    own.b = .{ .gpa = gpa, .e = own.e, .trace = if (std.c.getenv("TF_INKLING2_ROUNDS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false };
    // p4-cache: rank 0's book of the reusable state when every rank keeps snapshot slots (TF_INKLING2_PROMPT_REUSE);
    // without one rank 0 asks for no resume and no snapshot, so the other ranks' slots stay idle
    if (rank == 0 and own.e.slots.n > 0) own.b.book = makeBook(gpa, own.e) catch |err| blk: {
        std.log.warn("inkling2: prompt reuse off: its book could not be made ({t})", .{err});
        break :blk null;
    };
    // xk-rows gate: TF_INKLING2_DUMP_LOGITS=dir has rank 0 write each request's first-token logits there
    if (rank == 0) own.b.dump_dir = if (std.c.getenv("TF_INKLING2_DUMP_LOGITS")) |v| std.mem.span(v) else null;
    return .{
        .backend = own.b.backend(),
        .facts = own.b.facts(),
        .rows = if (own.e.draft != null) @intCast(@max(8, own.e.window_rows)) else 1, // e-step: 16 with copy windows
        .prompt_rows = @intCast(own.e.prompt_rows),
        .stream_bytes = 0, // the caches are the model's own, allocated at load
        .free_streams = 1,
        .ctx = own,
        .deinit = close,
        .target = own,
        .follow = if (rank != 0) followRank else null,
    };
}

/// Ranks past 0: rank 0's steps until it quits.
fn followRank(p: *anyopaque) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(p));
    std.log.info("inkling2 rank {d}: following rank 0", .{own.e.rank});
    try own.e.follow();
    std.log.info("inkling2 rank {d}: rank 0 ended", .{own.e.rank});
}

fn close(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.e.quit();
    if (own.b.book) |bk| { // p4-cache
        bk.deinit();
        own.gpa.destroy(bk);
    }
    if (own.b.draw_sc) |*sc| sc.deinit(); // xs: the sampled draws' candidates
    own.e.deinit();
    own.gpa.destroy(own);
}

/// p4-cache: the book (reuse_plan.zig) for the engine's slots: room for a whole context of ids, a slot's logits.
fn makeBook(gpa: std.mem.Allocator, e: *engine.Engine) !*reuse.Book {
    const bk = try gpa.create(reuse.Book);
    errdefer gpa.destroy(bk);
    bk.* = try reuse.Book.init(gpa, e.slots.n, e.context(), e.vocab(), reuse.minShared());
    // xk-rows: the salts this process gives its passes, unlike any other process's (states outlive the process)
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    bk.seed = (@as(u64, @bitCast(@as(i64, ts.sec))) *% 1_000_000_007) ^ @as(u64, @intCast(ts.nsec)) ^ (@as(u64, @intCast(std.c.getpid())) << 40);
    return bk;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoCommId => "Inkling-2 at --tp > 1 meets over NCCL: set NCCL_COMM_ID to rank 0's address:port on every host",
        error.CapturedKernels => "Inkling-2 runs on the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "Inkling-2 serves one stream at a time: retry once the current request ends",
        error.InklingTextOnly => "this Inkling-2 server reads text only: the request has images",
        error.StructuresUnsupported => "structured output is not built for Inkling-2 on CUDA yet",
        error.NonFiniteLogits => "the model's logits went non-finite at this row: the request is stopped rather than answered from them",
        error.ArenaBelowFloor => "the memory left above TF_INKLING2_FLOOR_GIB holds fewer experts a layer than the router takes (8): free memory or lower the floor",
        error.NoDirectReads => "the checkpoint's file system refuses O_DIRECT: expert reads would fill the page cache (TF_INKLING2_ALLOW_CACHED_READS=1 accepts that)",
        error.MemoryShort => "expert reads stopped: memory stayed under the pace (MemAvailable 2 GiB, PSI 20) for 30 s",
        error.ContextMemory => "this request's context needs more memory than the floor leaves, even after the experts' arena gave back what it can above the router's top-k: shorten the prompt or lower TF_INKLING2_FLOOR_GIB", // p4-mem
        else => null,
    };
}

test "refusals explain themselves" {
    for ([_]anyerror{ error.NoCommId, error.PastContext, error.NoFreeSlot, error.ArenaBelowFloor, error.MemoryShort }) |e| try std.testing.expect(explain(null, e) != null);
    try std.testing.expect(explain(null, error.OutOfMemory) == null);
}
