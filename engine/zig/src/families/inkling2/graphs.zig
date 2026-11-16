//! Decode and verify forwards (T = 1 .. max_rows rows) as CUDA graphs: one executable a (rows, head) key, captured
//! from the eager body right after the key's first (eager) forward, which makes its cuBLASLt plans; replayed after.
//! The body's kernels are the eager path's own, so a replay is bitwise the eager forward. What varies a call: the ids
//! (staged into x.ids before the launch), the routing (device: route, plan, items_to_slots over the arena's LUT, which
//! residency swaps edit in place) and the position: head_prep, cache_write and attention take it as a kernel argument,
//! so every replay rewrites that one argument in their nodes (cuGraphExecKernelNodeSetParams) before the launch.
//! TF_INKLING2_GRAPHS=0 runs every forward eagerly.
const std = @import("std");
const cuda = @import("cuda");

const abi = cuda.abi;

/// e-step: keys a Graphs holds, 1 .. 16 rows (engine.max_rows); Graphs.rows says how many it captures: 8 (every verify
/// window the DSpark head drafts), 16 with TF_INKLING2_COPIES (copy windows up to 16 rows).
pub const max_rows = 16;

/// What a captured forward ends with: no head (a prompt chunk; with speculation the final-norm tap), the per-row head
/// on the last row (x.logits fp32, x.argmax[0]), the batched argmax-only head over every row (x.argmax[0 .. T]), or
/// (xs: a sampled verify window, TF_INKLING2_SAMPLED_GRAPH) the per-row head on every row, a row at a time as the eager
/// window runs it, each row's fp32 logits into its slot of x.rows_f32 and its argmax into x.argmax[row].
pub const Head = enum(u2) { none, row, batch, rows };

/// A kernel whose argument `index` (an int) is the chunk's first position.
pub const PosArg = struct { f: cuda.Function, index: usize };

const GetParamInfo = *const fn (abi.Function, usize, *usize, *usize) callconv(.c) abi.Result;

/// One kernel node whose position argument a replay rewrites: its parameters with their values in `storage`.
const Patch = struct {
    node: abi.GraphNode,
    p: abi.KernelNodeParams,
    ptrs: []?*anyopaque,
    storage: []align(16) u8,
    pos_at: usize, // byte offset of the position in storage
};

pub const Entry = struct {
    graph: cuda.graph.Graph, // kept: the patched nodes are its nodes
    exec: cuda.graph.Exec,
    patches: []Patch,

    fn deinit(e: *Entry, gpa: std.mem.Allocator) void {
        for (e.patches) |p| {
            gpa.free(p.ptrs);
            gpa.free(p.storage);
        }
        gpa.free(e.patches);
        e.exec.deinit();
        e.graph.deinit();
        gpa.destroy(e);
    }
};

pub const Graphs = struct {
    on: bool = false,
    rows: usize = 8, // e-step: the widest chunk captured (16 with TF_INKLING2_COPIES); wider ones run eagerly
    entries: [max_rows][4]?*Entry = @splat(@splat(null)), // xs: four head kinds (Head.rows)
    captures: usize = 0,
    replays: usize = 0,
    patch_ns: u64 = 0, // host time rewriting positions, over every replay
    gen: u64 = 0, // e-step: drops so far (the DSpark pass's graph, TF_INKLING2_DRAFT_GRAPH, is dropped with them)
    capture_ns: u64 = 0, // host time capturing, instantiating and uploading, over every capture

    /// On unless TF_INKLING2_GRAPHS is 0 / off.
    pub fn fromEnv() Graphs {
        const v = if (std.c.getenv("TF_INKLING2_GRAPHS")) |p| std.mem.span(p) else return .{ .on = true };
        return .{ .on = !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off")) };
    }

    /// Every captured forward freed (their buffers or kernels change: the arena, speculation, close).
    pub fn drop(g: *Graphs, gpa: std.mem.Allocator) void {
        g.gen +%= 1; // e-step
        for (&g.entries) |*row| for (row) |*e| if (e.*) |x| {
            x.deinit(gpa);
            e.* = null;
        };
    }

    pub fn key(T: usize, head: Head) struct { usize, usize } {
        return .{ T - 1, @intFromEnum(head) };
    }

    /// Ends a capture on `s` into an executable whose position arguments (of `pos_args`' kernels, `expect` nodes,
    /// captured at `pos0`) a replay rewrites.
    pub fn finish(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, pos_args: []const PosArg, expect: usize, pos0: usize) !*Entry {
        var g = try cuda.graph.endCapture(s);
        var owned = true;
        defer if (owned) g.deinit();
        const info = @constCast(&d.lib).lookup(GetParamInfo, "cuFuncGetParamInfo") orelse return error.MissingSymbol;
        const n = try g.nodeCount();
        const nodes = try gpa.alloc(abi.GraphNode, n);
        defer gpa.free(nodes);
        _ = try g.nodes(nodes);
        var patches: std.ArrayList(Patch) = .empty;
        errdefer {
            for (patches.items) |p| {
                gpa.free(p.ptrs);
                gpa.free(p.storage);
            }
            patches.deinit(gpa);
        }
        for (nodes) |node| {
            var p: abi.KernelNodeParams = undefined;
            if (d.api.cuGraphKernelNodeGetParams_v2(node, &p) != abi.success) continue; // not a kernel node
            const pa = for (pos_args) |a| {
                if (a.f.handle == p.func) break a;
            } else continue;
            // the arguments' layout (count, offsets, sizes) from the driver; their values copied out of the node
            var count: usize = 0;
            var end: usize = 0;
            while (count < 64) : (count += 1) {
                var off: usize = 0;
                var size: usize = 0;
                if (info(p.func, count, &off, &size) != abi.success) break;
                end = @max(end, off + size);
            }
            if (pa.index >= count) return error.GraphPositionArgument;
            const ptrs = try gpa.alloc(?*anyopaque, count);
            errdefer gpa.free(ptrs);
            const storage = try gpa.alignedAlloc(u8, .@"16", @max(end, 16));
            errdefer gpa.free(storage);
            var pos_at: usize = 0;
            const src = p.params orelse return error.GraphPositionArgument;
            for (0..count) |i| {
                var off: usize = 0;
                var size: usize = 0;
                _ = info(p.func, i, &off, &size);
                const from: [*]const u8 = @ptrCast(src[i] orelse return error.GraphPositionArgument);
                @memcpy(storage[off..][0..size], from[0..size]);
                ptrs[i] = @ptrCast(&storage[off]);
                if (i == pa.index) {
                    if (size != 4) return error.GraphPositionArgument;
                    pos_at = off;
                }
            }
            // the argument must hold the position it was captured at (a reordered kernel signature fails here)
            if (std.mem.readInt(i32, storage[pos_at..][0..4], .little) != @as(i32, @intCast(pos0))) return error.GraphPositionArgument;
            p.params = ptrs.ptr;
            p.extra = null;
            try patches.append(gpa, .{ .node = node, .p = p, .ptrs = ptrs, .storage = storage, .pos_at = pos_at });
        }
        if (patches.items.len != expect) {
            std.log.err("inkling2 graphs: {d} position arguments in the capture, {d} expected", .{ patches.items.len, expect });
            return error.GraphPositionArgument;
        }
        var exec = try g.instantiate();
        errdefer exec.deinit();
        try exec.upload(s);
        const e = try gpa.create(Entry);
        errdefer gpa.destroy(e);
        const owned_patches = try patches.toOwnedSlice(gpa);
        e.* = .{ .graph = g, .exec = exec, .patches = owned_patches };
        owned = false;
        return e;
    }

    /// The forward at position `pos0` on `s`.
    pub fn replay(g: *Graphs, d: *const cuda.Driver, e: *Entry, s: cuda.Stream, pos0: usize) !void {
        const t0 = now();
        const v: i32 = @intCast(pos0);
        for (e.patches) |*p| {
            std.mem.writeInt(i32, p.storage[p.pos_at..][0..4], v, .little);
            try d.check(d.api.cuGraphExecKernelNodeSetParams_v2(e.exec.handle, p.node, &p.p), "cuGraphExecKernelNodeSetParams");
        }
        g.patch_ns += now() -| t0;
        try e.exec.launchOn(s);
        g.replays += 1;
    }
};

pub fn now() u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
