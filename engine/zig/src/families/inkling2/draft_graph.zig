//! e-step (prefill lane phase 5): the DSpark draft pass as a CUDA graph (TF_INKLING2_DRAFT_GRAPH=1, dspark.zig): its
//! ~110 launches and 17 small all-gathers replayed as one graph instead of launched one by one (the eager pass left the
//! GPU idle ~2 ms of its ~15 between launches). A graph is captured from the pass's own launches right after a pass ran
//! eagerly (its cuBLASLt plans made outside the capture), so a replay is that pass, launch for launch: the same bits.
//! What varies a pass is three integers, kernel arguments that a replay rewrites in their nodes
//! (cuGraphExecKernelNodeSetParams, as graphs.zig rewrites a forward's position): the anchor token (dsp_positions'
//! and every dsp_walk_step's `anchor`), the anchor's position (dsp_positions' pos0, every dsp_fdec's `anchor`) and a
//! sliding layer's first context position (its dsp_fdec's `lo`: the anchor's position minus the window, or 0); a
//! node's layer is found by its context ring (its `kc` argument).
const std = @import("std");
const cuda = @import("cuda");

const abi = cuda.abi;

pub const Rule = enum { anchor, pos, lo_local };

/// A kernel whose node arguments a replay rewrites: (index, rule) pairs; with `local_kc`, the `lo_local` rule applies
/// only to nodes whose argument `kc_index` (a pointer) is one of those rings (the other layers' lo stays 0).
pub const Wanted = struct {
    f: cuda.Function,
    args: []const struct { index: usize, rule: Rule },
    local_kc: []const u64 = &.{},
    kc_index: usize = 0,
};

const max_args = 2;

const Patch = struct {
    node: abi.GraphNode,
    p: abi.KernelNodeParams,
    ptrs: []?*anyopaque,
    storage: []align(16) u8,
    at: [max_args]usize = @splat(0),
    rules: [max_args]Rule = @splat(.anchor),
    n: usize = 0,
};

pub const Entry = struct {
    graph: cuda.graph.Graph,
    exec: cuda.graph.Exec,
    patches: []Patch,
    replays: u64 = 0,

    pub fn deinit(e: *Entry, gpa: std.mem.Allocator) void {
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

const GetParamInfo = *const fn (abi.Function, usize, *usize, *usize) callconv(.c) abi.Result;

/// Ends a capture on `s` into an executable whose wanted arguments (`expect` nodes) a replay rewrites; the captured
/// values must be the capture's (`anchor`, `pos`, `lo` of a sliding layer): a reordered signature fails here.
pub fn finish(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, wanted: []const Wanted, expect: usize, anchor: i32, pos: i32, window: usize) !*Entry {
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
        const w = for (wanted) |x| {
            if (x.f.handle == p.func) break x;
        } else continue;
        var count: usize = 0;
        var end: usize = 0;
        while (count < 64) : (count += 1) {
            var off: usize = 0;
            var size: usize = 0;
            if (info(p.func, count, &off, &size) != abi.success) break;
            end = @max(end, off + size);
        }
        const ptrs = try gpa.alloc(?*anyopaque, count);
        errdefer gpa.free(ptrs);
        const storage = try gpa.alignedAlloc(u8, .@"16", @max(end, 16));
        errdefer gpa.free(storage);
        var offs: [64]usize = undefined;
        var sizes: [64]usize = undefined;
        const src = p.params orelse return error.DraftGraphArgument;
        for (0..count) |i| {
            var off: usize = 0;
            var size: usize = 0;
            _ = info(p.func, i, &off, &size);
            const from: [*]const u8 = @ptrCast(src[i] orelse return error.DraftGraphArgument);
            @memcpy(storage[off..][0..size], from[0..size]);
            ptrs[i] = @ptrCast(&storage[off]);
            offs[i] = off;
            sizes[i] = size;
        }
        var patch: Patch = .{ .node = node, .p = p, .ptrs = ptrs, .storage = storage };
        // a sliding layer's node (by its ring), else its lo is the capture's 0 and stays
        var local = false;
        if (w.local_kc.len > 0) {
            if (w.kc_index >= count or sizes[w.kc_index] != 8) return error.DraftGraphArgument;
            const kc = std.mem.readInt(u64, storage[offs[w.kc_index]..][0..8], .little);
            local = std.mem.indexOfScalar(u64, w.local_kc, kc) != null;
        }
        for (w.args) |a| {
            if (a.rule == .lo_local and !local) continue;
            if (a.index >= count or sizes[a.index] != 4) return error.DraftGraphArgument;
            const want: i32 = switch (a.rule) {
                .anchor => anchor,
                .pos => pos,
                .lo_local => loLocal(pos, window),
            };
            if (std.mem.readInt(i32, storage[offs[a.index]..][0..4], .little) != want) return error.DraftGraphArgument;
            if (patch.n == max_args) return error.DraftGraphArgument;
            patch.at[patch.n] = offs[a.index];
            patch.rules[patch.n] = a.rule;
            patch.n += 1;
        }
        p.params = ptrs.ptr;
        p.extra = null;
        patch.p = p;
        try patches.append(gpa, patch);
    }
    if (patches.items.len != expect) {
        std.log.err("inkling2 draft graph: {d} patched nodes in the capture, {d} expected", .{ patches.items.len, expect });
        return error.DraftGraphArgument;
    }
    var exec = try g.instantiate();
    errdefer exec.deinit();
    try exec.upload(s);
    const e = try gpa.create(Entry);
    errdefer gpa.destroy(e);
    e.* = .{ .graph = g, .exec = exec, .patches = try patches.toOwnedSlice(gpa) };
    owned = false;
    return e;
}

fn loLocal(pos: i32, window: usize) i32 {
    const w: i32 = @intCast(window);
    return if (pos > w) pos - w else 0;
}

/// The pass from `anchor` at position `pos` on `s` (a sliding layer's window `window`).
pub fn replay(d: *const cuda.Driver, e: *Entry, s: cuda.Stream, anchor: i32, pos: i32, window: usize) !void {
    for (e.patches) |*p| {
        for (p.at[0..p.n], p.rules[0..p.n]) |at, rule| {
            const v: i32 = switch (rule) {
                .anchor => anchor,
                .pos => pos,
                .lo_local => loLocal(pos, window),
            };
            std.mem.writeInt(i32, p.storage[at..][0..4], v, .little);
        }
        try d.check(d.api.cuGraphExecKernelNodeSetParams_v2(e.exec.handle, p.node, &p.p), "cuGraphExecKernelNodeSetParams");
    }
    try e.exec.launchOn(s);
    e.replays += 1;
}
