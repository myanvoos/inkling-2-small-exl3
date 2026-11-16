//! The kortexa-ai Inkling-2-Small EXL3 export (checkpoint.json format "kortexa-inkling2-exl3-v1") on one TP rank.
//!
//! The export keeps every dense tensor of the release byte for byte (its dense/*.safetensors, read through the view's
//! model.safetensors.index.json like the release's shards) and replaces the routed experts: per expert, w13 (gate/up
//! fused, H -> 2I, output columns interleaved g0, u0, g1, u1, ...) and w2 (I -> H) as ExLlamaV3 trellises (mul1
//! codebook, 2 or 3 bits a weight, layers/NN/experts-SSS-EEE.safetensors: experts.{e}.{w13,w2}.{trellis,suh,svh,mul1}).
//!
//! A rank keeps intermediate channels [rank Il, (rank + 1) Il) of every expert, as the NVFP4 / MXFP8 lanes do, packed
//! into one slot (out/exl3/ref/slot.py, gated on real bytes: the w13 cut equals the whole matrix's columns bit for
//! bit, the w2 partials add to the whole):
//!   [w13 trellis [H/16][2Il/16][16K] | w2 trellis [Il/16][H/16][16K] | suh13 f16 [H] | svh13 f16 [2Il] |
//!    suh2 f16 [Il] | svh2 f16 [H]]      (each part 16-byte aligned; inkling2_exl3.cu's slot_parts reads the same)
//! At TP2 the rank's slots are 95.6 GB (89.07 GiB: 10,944 K3 slots of 5,997,312 B and 7,488 K2 of 4,006,656 B), which
//! fit with the dense share above the 8 GiB floor of a 128 GB GB10: every expert is resident (the router is never
//! masked, nothing is substituted, no residency pass moves a slot).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const fill = @import("fill.zig");
const model_mod = @import("model.zig");
const defaults = @import("defaults.zig"); // the user, 2026-11-15: everything on by default
const direct_io = core.direct_io;

pub const format_id = "kortexa-inkling2-exl3-v1";
pub const mul1_marker: u32 = 0x83DCD12D;
const gib: f64 = 1024.0 * 1024.0 * 1024.0;

extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
extern "c" fn posix_fadvise(fd: c_int, off: i64, len: i64, advice: c_int) c_int;
const fadv_dontneed: c_int = 4;

// ---------------------------------------------------------------------------------------------------- geometry

/// Byte offsets of one K-bit slot's parts (w13 at 0) and its size, for hidden H and Il channels a rank.
pub const Parts = struct { w2: usize, suh13: usize, svh13: usize, suh2: usize, svh2: usize, total: usize };

fn align16(n: usize) usize {
    return (n + 15) & ~@as(usize, 15);
}

pub fn parts(H: usize, Il: usize, k: usize) Parts {
    const tile = 32 * k; // 16 K int16 a 16 x 16 tile
    var p: Parts = undefined;
    p.w2 = align16((H / 16) * (2 * Il / 16) * tile);
    p.suh13 = p.w2 + align16((Il / 16) * (H / 16) * tile);
    p.svh13 = p.suh13 + align16(2 * H);
    p.suh2 = p.svh13 + align16(4 * Il);
    p.svh2 = p.suh2 + align16(2 * Il);
    p.total = p.svh2 + align16(2 * H);
    return p;
}

/// The cuts fall on the 128-wide Hadamard blocks of both projections: 2 Il (w13's columns) and Il (w2's rows).
pub fn checkSplit(hidden: usize, inter: usize, world: usize) !void {
    if (world == 0 or inter % world != 0) return error.Exl3Split;
    const il = inter / world;
    if (hidden % 128 != 0 or il % 128 != 0 or (2 * il) % 128 != 0) return error.Exl3Split;
    if ((hidden / 16) % 4 != 0 or (il / 16) % 4 != 0) return error.Exl3Split; // the GEMM's 4 warps split K / 16
}

// ---------------------------------------------------------------------------------------------------- the index

pub const Batch = struct { start: usize, end: usize, file: []const u8 };

/// checkpoint.json, checked: the format, a complete conversion, the release's shape, every layer's experts at 2 or 3
/// bits, each layer's batch files covering experts [0, E) in order with relative paths under layers/.
pub const Index = struct {
    arena: std.heap.ArenaAllocator,
    layers: usize,
    experts: usize,
    bits: []u8, // [layer][expert]: 2 or 3
    batches: [][]Batch, // [layer]

    pub fn deinit(ix: *Index) void {
        ix.arena.deinit();
    }

    pub fn k(ix: *const Index, layer: usize, e: usize) u8 {
        return ix.bits[layer * ix.experts + e];
    }

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const config.Config) !Index {
        const path = try std.fs.path.join(gpa, &.{ dir, "checkpoint.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
        defer gpa.free(text);
        return parse(gpa, text, c);
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8, c: *const config.Config) !Index {
        var ix: Index = .{ .arena = std.heap.ArenaAllocator.init(gpa), .layers = c.layers, .experts = c.experts, .bits = &.{}, .batches = &.{} };
        errdefer ix.arena.deinit();
        const a = ix.arena.allocator();
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{ .allocate = .alloc_always });
        const root = switch (parsed.value) {
            .object => |o| o,
            else => return error.Exl3Index,
        };
        if (!strIs(root.get("format"), format_id)) return error.Exl3Format;
        if (!strIs(root.get("status"), "complete")) {
            std.log.err("inkling2 exl3: checkpoint.json status is not \"complete\"", .{});
            return error.Exl3Incomplete;
        }
        const tc = objOf(root.get("text_config")) orelse return error.Exl3Index;
        if (try intOf(tc.get("hidden_size")) != c.hidden or try intOf(tc.get("num_hidden_layers")) != c.layers or
            try intOf(tc.get("n_routed_experts")) != c.experts or try intOf(tc.get("intermediate_size")) != c.inter)
        {
            std.log.err("inkling2 exl3: checkpoint.json's text_config differs from config.json's shape", .{});
            return error.Exl3Shape;
        }
        const layers = objOf(root.get("layers")) orelse return error.Exl3Index;
        ix.bits = try a.alloc(u8, c.layers * c.experts);
        ix.batches = try a.alloc([]Batch, c.layers);
        var key: [16]u8 = undefined;
        for (0..c.layers) |l| {
            const lo = objOf(layers.get(try std.fmt.bufPrint(&key, "{d}", .{l}))) orelse {
                std.log.err("inkling2 exl3: checkpoint.json has no layer {d}", .{l});
                return error.Exl3Index;
            };
            if (!strIs(lo.get("status"), "complete")) return error.Exl3Incomplete;
            const kb = arrOf(lo.get("K_by_expert")) orelse return error.Exl3Index;
            if (kb.len != c.experts) return error.Exl3Index;
            for (kb, 0..) |v, e| {
                const bits = try intOf(v);
                if (bits != 2 and bits != 3) {
                    std.log.err("inkling2 exl3: layer {d} expert {d} at {d} bits (this lane reads 2 or 3)", .{ l, e, bits });
                    return error.Exl3Bits;
                }
                ix.bits[l * c.experts + e] = @intCast(bits);
            }
            const bl = arrOf(lo.get("batches")) orelse return error.Exl3Index;
            const out = try a.alloc(Batch, bl.len);
            var next: usize = 0;
            for (bl, out) |v, *b| {
                const bo = objOf(v) orelse return error.Exl3Index;
                const file = switch (bo.get("file") orelse return error.Exl3Index) {
                    .string => |s| s,
                    else => return error.Exl3Index,
                };
                if (!std.mem.startsWith(u8, file, "layers/") or std.mem.indexOf(u8, file, "..") != null) return error.Exl3Path;
                b.* = .{ .start = try intOf(bo.get("start")), .end = try intOf(bo.get("end")), .file = file };
                if (b.start != next or b.end <= b.start or b.end > c.experts) return error.Exl3Batches;
                next = b.end;
            }
            if (next != c.experts) return error.Exl3Batches;
            ix.batches[l] = out;
        }
        return ix;
    }

    /// Bytes of every slot of this rank (Il channels a rank).
    pub fn rankBytes(ix: *const Index, H: usize, Il: usize) u64 {
        var n: u64 = 0;
        for (ix.bits) |b| n += parts(H, Il, b).total;
        return n;
    }
};

fn strIs(v: ?std.json.Value, want: []const u8) bool {
    const x = v orelse return false;
    return switch (x) {
        .string => |s| std.mem.eql(u8, s, want),
        else => false,
    };
}

fn objOf(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return switch (x) {
        .object => |o| o,
        else => null,
    };
}

fn arrOf(v: ?std.json.Value) ?[]std.json.Value {
    const x = v orelse return null;
    return switch (x) {
        .array => |arr| arr.items,
        else => null,
    };
}

fn intOf(v: ?std.json.Value) !usize {
    const x = v orelse return error.Exl3Index;
    return switch (x) {
        .integer => |i| if (i < 0) error.Exl3Index else @intCast(i),
        else => error.Exl3Index,
    };
}

/// Whether config.json's directory holds this export (checkpoint.json of format_id); another checkpoint.json format
/// is refused by name, none at all is not this export.
pub fn detect(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    const path = try std.fs.path.join(gpa, &.{ dir, "checkpoint.json" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26)) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    defer gpa.free(text);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), text, .{});
    const root = objOf(parsed.value) orelse return error.Exl3Index;
    if (strIs(root.get("format"), format_id)) return true;
    std.log.err("inkling2: {s}/checkpoint.json is not a {s} index", .{ dir, format_id });
    return error.Exl3Format;
}

// ---------------------------------------------------------------------------------------------------- one file

/// A safetensors file's tensors by name (header only) and where its data starts.
const Header = struct {
    arena: std.heap.ArenaAllocator,
    data: usize,
    map: std.StringHashMapUnmanaged(Entry) = .empty,

    const Entry = struct { dtype: []const u8, shape: [4]usize, dims: usize, lo: usize, hi: usize };

    fn deinit(h: *Header) void {
        h.arena.deinit();
    }

    fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Header {
        if (bytes.len < 8) return error.Exl3File;
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        if (n > bytes.len - 8 or n > 1 << 26) return error.Exl3File;
        var h: Header = .{ .arena = std.heap.ArenaAllocator.init(gpa), .data = 8 + n };
        errdefer h.arena.deinit();
        const a = h.arena.allocator();
        const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..n], .{ .allocate = .alloc_always });
        const root = objOf(parsed.value) orelse return error.Exl3File;
        var it = root.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const o = objOf(kv.value_ptr.*) orelse return error.Exl3File;
            const dt = switch (o.get("dtype") orelse return error.Exl3File) {
                .string => |s| s,
                else => return error.Exl3File,
            };
            const sh = arrOf(o.get("shape")) orelse return error.Exl3File;
            const offs = arrOf(o.get("data_offsets")) orelse return error.Exl3File;
            if (sh.len > 4 or offs.len != 2) return error.Exl3File;
            var e: Entry = .{ .dtype = dt, .shape = @splat(1), .dims = sh.len, .lo = try intOf(offs[0]), .hi = try intOf(offs[1]) };
            for (sh, 0..) |v, i| e.shape[i] = try intOf(v);
            if (e.hi < e.lo or h.data + e.hi > bytes.len) return error.Exl3File;
            try h.map.put(a, kv.key_ptr.*, e);
        }
        return h;
    }

    /// The bytes of `name`, its dtype and shape checked (error.Exl3Tensor otherwise; `tensors` names the case).
    fn get(h: *const Header, bytes: []const u8, name: []const u8, dtype: []const u8, shape: []const usize) ![]const u8 {
        const e = h.map.get(name) orelse return error.Exl3Tensor;
        if (!std.mem.eql(u8, e.dtype, dtype) or e.dims != shape.len or !std.mem.eql(usize, e.shape[0..e.dims], shape))
            return error.Exl3Tensor;
        return bytes[h.data + e.lo .. h.data + e.hi];
    }

    /// `get`, logging the refusal with what the file holds.
    fn need(h: *const Header, bytes: []const u8, name: []const u8, dtype: []const u8, shape: []const usize) ![]const u8 {
        return h.get(bytes, name, dtype, shape) catch |err| {
            if (h.map.get(name)) |e| {
                std.log.err("inkling2 exl3: {s} is {s} {any}, expected {s} {any}", .{ name, e.dtype, e.shape[0..e.dims], dtype, shape });
            } else std.log.err("inkling2 exl3: no tensor {s} in its batch file", .{name});
            return err;
        };
    }
};

/// One projection's tensors of one expert, as stored (checked by `tensors`).
pub const Mat = struct { trellis: []const u8, suh: []const u8, svh: []const u8 };

/// Expert e's w13 or w2 from a batch file's bytes: trellis I16 [K/16][N/16][16 k], suh F16 [K], svh F16 [N], mul1 I32
/// scalar = the mul1 codebook's marker (the only codebook this lane decodes).
fn tensors(h: *const Header, bytes: []const u8, e: usize, comptime proj: []const u8, H: usize, I: usize, k: usize) !Mat {
    const K: usize = if (comptime std.mem.eql(u8, proj, "w13")) H else I;
    const N: usize = if (comptime std.mem.eql(u8, proj, "w13")) 2 * I else H;
    var buf: [64]u8 = undefined;
    const p = try std.fmt.bufPrint(&buf, "experts.{d}." ++ proj ++ ".", .{e});
    var nb: [80]u8 = undefined;
    const marker = try h.need(bytes, try std.fmt.bufPrint(&nb, "{s}mul1", .{p}), "I32", &.{});
    if (std.mem.readInt(u32, marker[0..4], .little) != mul1_marker) {
        std.log.err("inkling2 exl3: experts.{d}.{s} is not the mul1 codebook", .{ e, proj });
        return error.Exl3Codebook;
    }
    const tr = try h.need(bytes, try std.fmt.bufPrint(&nb, "{s}trellis", .{p}), "I16", &.{ K / 16, N / 16, 16 * k });
    const suh = try h.need(bytes, try std.fmt.bufPrint(&nb, "{s}suh", .{p}), "F16", &.{K});
    const svh = try h.need(bytes, try std.fmt.bufPrint(&nb, "{s}svh", .{p}), "F16", &.{N});
    return .{ .trellis = tr, .suh = suh, .svh = svh };
}

/// An fp16 scale that is NaN or infinite (exponent all ones).
fn badF16(bytes: []const u8) ?usize {
    var i: usize = 0;
    while (i + 1 < bytes.len) : (i += 2) {
        const v = std.mem.readInt(u16, bytes[i..][0..2], .little);
        if (v & 0x7C00 == 0x7C00) return i / 2;
    }
    return null;
}

/// Rank `rank`'s slot of one expert from its two projections (slot.py pack): w13's output tiles of the rank's 2 Il
/// channels gathered from every input tile row, w2's input tile rows of its Il channels, the scales cut alike.
/// `out` is the slot's `parts(...).total` bytes. A pure byte permutation: no value is decoded or rounded.
pub fn packSlot(w13: Mat, w2: Mat, H: usize, I: usize, rank: usize, world: usize, k: usize, out: []u8) !void {
    const Il = I / world;
    const p = parts(H, Il, k);
    if (out.len != p.total) return error.Exl3Slot;
    const tile = 32 * k;
    // w13: [H/16][2I/16] tiles, the rank's [nt0, nt0 + nt) of every row
    const nt = 2 * Il / 16;
    const row13 = (2 * I / 16) * tile;
    for (0..H / 16) |kt| {
        @memcpy(out[kt * nt * tile ..][0 .. nt * tile], w13.trellis[kt * row13 + rank * nt * tile ..][0 .. nt * tile]);
    }
    // w2: [I/16][H/16] tiles, the rank's rows [rank Il / 16, ...): one range
    const r2 = (H / 16) * tile * (Il / 16);
    @memcpy(out[p.w2..][0..r2], w2.trellis[rank * r2 ..][0..r2]);
    @memcpy(out[p.suh13..][0 .. 2 * H], w13.suh[0 .. 2 * H]);
    @memcpy(out[p.svh13..][0 .. 4 * Il], w13.svh[rank * 4 * Il ..][0 .. 4 * Il]);
    @memcpy(out[p.suh2..][0 .. 2 * Il], w2.suh[rank * 2 * Il ..][0 .. 2 * Il]);
    @memcpy(out[p.svh2..][0 .. 2 * H], w2.svh[0 .. 2 * H]);
}

/// One expert's slot of rank `rank` (the check tool's: one file read, checked and packed as Store.open does it).
/// Returns the slot's bits; `out` must hold parts(H, I / world, 3).total bytes (the slot is its first parts().total).
pub fn loadSlot(gpa: std.mem.Allocator, dir: []const u8, ix: *const Index, layer: usize, e: usize, H: usize, I: usize, rank: usize, world: usize, out: []u8) !u8 {
    const k = ix.k(layer, e);
    const b = for (ix.batches[layer]) |bb| {
        if (bb.start <= e and e < bb.end) break bb;
    } else return error.Exl3Index;
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, b.file }, 0);
    defer gpa.free(path);
    const buf = try gpa.alloc(u8, try fileSize(path));
    defer gpa.free(buf);
    const bytes = try readWhole(path, buf);
    var h = try Header.parse(gpa, bytes);
    defer h.deinit();
    const w13 = try tensors(&h, bytes, e, "w13", H, I, k);
    const w2 = try tensors(&h, bytes, e, "w2", H, I, k);
    for ([_][]const u8{ w13.suh, w13.svh, w2.suh, w2.svh }) |sc| if (badF16(sc) != null) return error.Exl3Scale;
    try packSlot(w13, w2, H, I, rank, world, k, out[0..parts(H, I / world, k).total]);
    return k;
}

/// Every expert of `layer` packed for (rank, world) into `out` (host), expert e's slot at offsets[e] (256-byte aligned,
/// the experts in order) with its bits in ks[e], each batch file read once: the bench's loader (tf-inkling2-exl3-bench;
/// the forward's is Store.open). The bytes used.
pub fn packLayer(gpa: std.mem.Allocator, dir: []const u8, ix: *const Index, layer: usize, H: usize, I: usize, rank: usize, world: usize, out: []u8, offsets: []usize, ks: []u8) !usize {
    var at: usize = 0;
    for (ix.batches[layer]) |b| {
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, b.file }, 0);
        defer gpa.free(path);
        const buf = try gpa.alloc(u8, try fileSize(path));
        defer gpa.free(buf);
        const bytes = try readWhole(path, buf);
        var h = try Header.parse(gpa, bytes);
        defer h.deinit();
        for (b.start..b.end) |e| {
            const k = ix.k(layer, e);
            const w13 = try tensors(&h, bytes, e, "w13", H, I, k);
            const w2 = try tensors(&h, bytes, e, "w2", H, I, k);
            const n = parts(H, I / world, k).total;
            if (at + n > out.len) return error.NoSpace;
            try packSlot(w13, w2, H, I, rank, world, k, out[at .. at + n]);
            offsets[e] = at;
            ks[e] = k;
            at = std.mem.alignForward(usize, at + n, 256);
        }
    }
    return at;
}

// ---------------------------------------------------------------------------------------------------- the store

/// Every expert's slot of this rank on the device, one buffer a layer, and the tables the kernels read:
/// slot_ptr u64 [L E] (the slot's address) and slot_k i32 [L E] (its bits). The LUT maps (layer, expert) to slot
/// layer E + expert.
pub const Store = struct {
    gpa: std.mem.Allocator,
    ix: Index,
    bufs: []cuda.DeviceBuffer,
    slot_ptr: cuda.DeviceBuffer,
    slot_k: cuda.DeviceBuffer,
    // the engine's grouped kernel's tables (kern.exl3Grouped): each slot's w13 and w2 trellis addresses, half-bits
    tp13: cuda.DeviceBuffer,
    tp2: cuda.DeviceBuffer,
    k2: cuda.DeviceBuffer,
    nslots: usize,
    bytes: u64 = 0,
    stats: Stats = .{},

    pub const Stats = struct { files: usize = 0, read_bytes: u64 = 0, read_ns: u64 = 0, pack_ns: u64 = 0, upload_ns: u64 = 0, paused_ms: u64 = 0, min_avail: u64 = std.math.maxInt(u64), direct: DirectStats = .{} };

    pub fn close(st: *Store) void {
        for (st.bufs) |*b| b.free();
        st.gpa.free(st.bufs);
        st.slot_ptr.free();
        st.slot_k.free();
        st.tp13.free();
        st.tp2.free();
        st.k2.free();
        st.ix.deinit();
        st.gpa.destroy(st);
    }

    /// The bytes this rank's slots take (no allocation): what a caller checks against MemAvailable first.
    pub fn need(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, m: *const model_mod.Model) !u64 {
        var ix = try Index.open(gpa, io, dir, &m.c);
        defer ix.deinit();
        return ix.rankBytes(m.c.hidden, m.inter());
    }

    /// The room check before the slots load: this rank's slot bytes (need) against `room` (MemAvailable less the floor
    /// and the caller's margins); error.Exl3DoesNotFit, logged, when they exceed it. Returns the bytes. x-merge: the
    /// engine's and the run tool's check in one function (x-mixed DESIGN 2.7: the mixed store sizes its bank after it).
    pub fn fits(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, m: *const model_mod.Model, room: f64, floor_gib: f64) !u64 {
        const n = try need(gpa, io, dir, m);
        if (@as(f64, @floatFromInt(n)) > room + 1) {
            std.log.err("inkling2 exl3 rank {d}: this rank's {d:.2} GiB of EXL3 slots do not fit the {d:.2} GiB left above the {d} GiB floor and the margins: free memory, lower TF_INKLING2_FLOOR_GIB or the context", .{ m.rank, @as(f64, @floatFromInt(n)) / gib, room / gib, floor_gib });
            return error.Exl3DoesNotFit;
        }
        return n;
    }

    /// e-load: every expert's slot of this rank on the device, by the loader TF_INKLING2_EXL3_LOAD picks (`direct`:
    /// openDirect; unset or anything else: the lane's loader, openHost); TF_INKLING2_EXL3_HASH=PATH then writes each
    /// slot's hash (writeHashes: PATH.r<rank>), so two loads can be compared slot by slot.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, m: *model_mod.Model, dir: []const u8, floor_gib: f64) !*Store {
        const st = switch (loadMode()) {
            .host => try openHost(gpa, io, m, dir, floor_gib),
            .direct, .split => try openDirect(gpa, io, m, dir, floor_gib),
        };
        errdefer st.close();
        if (std.c.getenv("TF_INKLING2_EXL3_HASH")) |p| try writeHashes(st, gpa, io, m, std.mem.span(p));
        return st;
    }

    /// Opens the index, then every expert of every layer into the device: per layer one buffer, each batch file read
    /// whole (pread, its pages dropped after: the page cache keeps none of the 191 GB a rank reads), each of its
    /// experts packed on the host and uploaded into its slot. Before each layer's buffer, MemAvailable must hold it
    /// above `floor_gib` + 1 GiB (else error.Exl3DoesNotFit, with the numbers); each file's read waits for the pace
    /// (fill.Pace: MemAvailable, PSI). Then the LUT maps every (layer, expert) to its slot and marks it routable
    /// (the arena must be allocated: allocArena(.{0, 0}, .{0, 0}) gives the LUT and the mask), published.
    pub fn openHost(gpa: std.mem.Allocator, io: std.Io, m: *model_mod.Model, dir: []const u8, floor_gib: f64) !*Store {
        const c = &m.c;
        const H: usize = c.hidden;
        const I: usize = c.inter;
        const W = m.world;
        const Il = I / W;
        try checkSplit(H, I, W);
        // x-mixed: the mixed store sizes its bank (the arena) after these slots, and its LUT is the bank's
        if (!m.arena_on and !m.mix) return error.Exl3NoArena;
        const st = try gpa.create(Store);
        errdefer gpa.destroy(st);
        st.* = .{ .gpa = gpa, .ix = try Index.open(gpa, io, dir, c), .bufs = &.{}, .slot_ptr = undefined, .slot_k = undefined, .tp13 = undefined, .tp2 = undefined, .k2 = undefined, .nslots = c.layers * c.experts };
        errdefer st.ix.deinit();
        const L = c.layers;
        const E = c.experts;
        st.bufs = try gpa.alloc(cuda.DeviceBuffer, L);
        @memset(st.bufs, .{ .d = m.d, .ptr = 0, .len = 0 });
        errdefer {
            for (st.bufs) |*b| if (b.ptr != 0) b.free();
            gpa.free(st.bufs);
        }
        const host_ptr = try gpa.alloc(u64, L * E);
        defer gpa.free(host_ptr);
        const host_k = try gpa.alloc(i32, L * E);
        defer gpa.free(host_k);
        // the largest batch file and the largest slot: the two host buffers, reused file to file
        var most_file: usize = 0;
        for (0..L) |l| for (st.ix.batches[l]) |b| {
            const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, b.file }, 0);
            defer gpa.free(path);
            most_file = @max(most_file, try fileSize(path));
        };
        const file_buf = try gpa.alloc(u8, most_file);
        defer gpa.free(file_buf);
        const slot_buf = try gpa.alloc(u8, parts(H, Il, 3).total);
        defer gpa.free(slot_buf);
        const pace = fill.Pace.fromEnv();
        const floor_bytes: u64 = @intFromFloat(floor_gib * gib);
        for (0..L) |l| {
            var layer_bytes: usize = 0;
            for (0..E) |e| layer_bytes += parts(H, Il, st.ix.k(l, e)).total;
            const avail = fill.readAvail() orelse 0;
            st.stats.min_avail = @min(st.stats.min_avail, avail);
            if (avail < layer_bytes + floor_bytes + (1 << 30)) {
                std.log.err("inkling2 exl3 rank {d}: layer {d}'s {d:.2} GiB of slots do not fit: MemAvailable {d:.2} GiB, floor {d} GiB + 1 GiB (layers 0..{d} loaded, {d:.2} GiB); a smaller context, a lower floor or fewer other programs", .{ m.rank, l, @as(f64, @floatFromInt(layer_bytes)) / gib, @as(f64, @floatFromInt(avail)) / gib, floor_gib, l, @as(f64, @floatFromInt(st.bytes)) / gib });
                return error.Exl3DoesNotFit;
            }
            st.bufs[l] = try cuda.DeviceBuffer.alloc(m.d, layer_bytes);
            var off: usize = 0;
            var e_next: usize = 0;
            for (st.ix.batches[l]) |b| {
                const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, b.file }, 0);
                defer gpa.free(path);
                try st.paceWait(pace, floor_bytes);
                const t0 = std.Io.Clock.awake.now(io);
                const bytes = try readWhole(path, file_buf);
                st.stats.read_ns += ns(io, t0);
                st.stats.read_bytes += bytes.len;
                st.stats.files += 1;
                var h = try Header.parse(gpa, bytes);
                defer h.deinit();
                for (b.start..b.end) |e| {
                    std.debug.assert(e == e_next);
                    e_next += 1;
                    const k = st.ix.k(l, e);
                    const p = parts(H, Il, k);
                    const t1 = std.Io.Clock.awake.now(io);
                    const w13 = try tensors(&h, bytes, e, "w13", H, I, k);
                    const w2 = try tensors(&h, bytes, e, "w2", H, I, k);
                    for ([_][]const u8{ w13.suh, w13.svh, w2.suh, w2.svh }) |sc| if (badF16(sc)) |at| {
                        std.log.err("inkling2 exl3: layer {d} expert {d}: a non-finite fp16 scale at element {d}", .{ l, e, at });
                        return error.Exl3Scale;
                    };
                    const out = slot_buf[0..p.total];
                    try packSlot(w13, w2, H, I, m.rank, W, k, out);
                    st.stats.pack_ns += ns(io, t1);
                    const t2 = std.Io.Clock.awake.now(io);
                    try st.bufs[l].upload(off, out);
                    st.stats.upload_ns += ns(io, t2);
                    host_ptr[l * E + e] = st.bufs[l].ptr + off;
                    host_k[l * E + e] = k;
                    off += p.total;
                }
            }
            std.debug.assert(off == layer_bytes and e_next == E);
            st.bytes += layer_bytes;
        }
        st.slot_ptr = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(host_ptr));
        errdefer st.slot_ptr.free();
        st.slot_k = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(host_k));
        errdefer st.slot_k.free();
        // the engine's grouped kernel's tables: each slot's w13 trellis (the slot's start), its w2 trellis, its half-bits
        const t13 = try gpa.alloc(u64, L * E);
        defer gpa.free(t13);
        const t2 = try gpa.alloc(u64, L * E);
        defer gpa.free(t2);
        const k2 = try gpa.alloc(i32, L * E);
        defer gpa.free(k2);
        for (0..L * E) |i| {
            const bits: usize = @intCast(host_k[i]);
            t13[i] = host_ptr[i];
            t2[i] = host_ptr[i] + parts(c.hidden, c.inter / m.world, bits).w2;
            k2[i] = @intCast(2 * bits);
        }
        st.tp13 = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(t13));
        errdefer st.tp13.free();
        st.tp2 = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(t2));
        errdefer st.tp2.free();
        st.k2 = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(k2));
        errdefer st.k2.free();
        // the LUT: (layer, expert) -> slot layer E + expert, every expert routable, when no bank owns the arena (x-merge,
        // x-mixed DESIGN 2.7: a mixed store's arena LUT maps its bank's slots; the EXL3 slots stay layer E + expert in
        // slot_ptr / slot_k either way)
        const a = &m.arena;
        if (!m.mix and a.slots(0) + a.slots(1) == 0) { // x-mixed: never in the mixed store (moeMix maps its slots)
            for (0..L) |l| for (0..E) |e| {
                a.host_lut[l * E + e] = @intCast(l * E + e);
                a.host_routable[l * E + e] = 1;
            };
            try m.publish();
        }
        return st;
    }

    fn paceWait(st: *Store, pace: fill.Pace, floor_bytes: u64) !void {
        _ = floor_bytes;
        var waited: u32 = 0;
        while (true) : (waited += 2) {
            const avail = fill.readAvail() orelse return;
            const psi = fill.readPsi() orelse 0;
            st.stats.min_avail = @min(st.stats.min_avail, avail);
            if (avail >= pace.avail_bytes and psi < pace.psi) return;
            if (waited >= pace.wait_ms) return error.MemoryShort;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 2 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
            st.stats.paused_ms += 2;
        }
    }
};

fn ns(io: std.Io, t: std.Io.Timestamp) u64 {
    return @intCast(t.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
}

fn fileSize(path: [:0]const u8) !usize {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.Exl3Open;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.Exl3Open;
    return @intCast(end);
}

/// The whole file into `buf` (pread), its pages dropped from the page cache after; the bytes read.
fn readWhole(path: [:0]const u8, buf: []u8) ![]const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) {
        std.log.err("inkling2 exl3: cannot open {s}", .{path});
        return error.Exl3Open;
    }
    defer _ = std.c.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const r = pread(fd, buf[n..].ptr, buf.len - n, @intCast(n));
        if (r < 0) return error.Exl3Read;
        if (r == 0) break;
        n += @intCast(r);
    }
    _ = posix_fadvise(fd, 0, 0, fadv_dontneed);
    return buf[0..n];
}

// ==================================================================================================== e-load
// (prefill lane, phase 5) The store's load the way fill.zig loads the release's experts, TF_INKLING2_EXL3_LOAD=direct
// (unset or any other value: the lane's loader, Store.openHost, unchanged). loadDirect:
//   1. every batch file opened O_DIRECT, its header read (one aligned read a file, on the reader threads) and its
//      experts' tensors checked as `tensors` checks them (names, dtypes, shapes, bytes inside the file);
//   2. the bytes this rank cuts from each file — w13's whole trellis (the rank's column tiles are strided over every
//      row), the rank's rows of w2's (one range), the four scales whole (checked whole, as the lane does) and the two
//      mul1 markers — read past the page cache by `readers` threads in pieces of up to 8 MiB into a ring of mapped
//      pinned stages: no page-cache page, no host pack, no pageable upload; the readers wait for the device only when
//      the whole ring is ahead of it;
//   3. a stage whose reads are in: the host checks its markers and scales (the lane's refusals, before a byte lands),
//      allocates the buffer of each layer the stage starts (after the lane's floor check), and the device cuts the
//      stage's slots (inkling2_exl3.cu ink2_exl3_cut: packSlot as six copy jobs an expert) on a side stream; the
//      stage goes back to the readers once its cut is done.
// The slots lie as openHost lays them (each layer's experts in order, back to back) and hold its bytes: a permutation
// of stored bytes, nothing decoded or rounded (tf-inkling2-exl3-load compares every slot byte against packLayer;
// TF_INKLING2_EXL3_HASH compares the two loaders' slots by hash in the real run).
// Knobs: TF_INKLING2_EXL3_READERS (8 threads), TF_INKLING2_EXL3_STAGES (4), TF_INKLING2_EXL3_STAGE_MIB (128, raised to
// the largest file's bytes); the stages are the loader's whole host footprint (512 MiB pinned), freed when it returns.

pub const LoadMode = enum { host, direct, split };

/// TF_INKLING2_EXL3_LOAD=direct: openDirect (loadDirect); =split: the same, each of a TP pair's ranks reading half the
/// files and the pair swapping the other's slots (loadDirect's Peer; direct at any other world); unset: direct (the user,
/// 2026-11-15: on by default; host under TF_INKLING2_LEGACY=1); `host` or anything else: the lane's openHost.
pub fn loadMode() LoadMode {
    const v = std.mem.span(std.c.getenv("TF_INKLING2_EXL3_LOAD") orelse return if (defaults.legacy()) .host else .direct);
    if (std.mem.eql(u8, v, "direct")) return .direct;
    if (std.mem.eql(u8, v, "split")) return .split;
    return .host;
}

/// loadDirect's own numbers (Stats.direct).
pub const DirectStats = struct {
    heads_ns: u64 = 0, // step 1: every header
    stream_ns: u64 = 0, // steps 2 and 3: the first piece read to the last cut landed
    wait_ns: u64 = 0, // the main thread waiting for a stage's reads (what the ring could not hide)
    check_ns: u64 = 0, // markers and scales on the host
    alloc_ns: u64 = 0, // the layer buffers (floor checks, cuMemAlloc)
    cut_ns: u64 = 0, // cut launches and their waits
    pace_ns: u64 = 0, // the pace's reads of MemAvailable / PSI (and its waits)
    busy_ns: u64 = 0, // the readers' preads, summed over threads
    exchange_ns: u64 = 0, // split mode: the rounds (statuses, sends and receives), waited for
    sent_bytes: u64 = 0, // split mode: the peer's slots sent
    recv_bytes: u64 = 0, // split mode: this rank's slots received
    cut_bytes: u64 = 0, // slot bytes written
    stages: usize = 0,
    pieces: usize = 0,
    readers: usize = 0,
    ring: usize = 0,
    stage_bytes: usize = 0,
    cached: usize = 0, // files the file system would not open O_DIRECT (read through the page cache)
};

const piece_max: usize = 8 << 20; // bytes a read asks for at most (fill.zig's piece, core direct_gather.chunk)
const join_gap: u64 = 64 << 10; // ranges of one file closer than this are read as one
const head_read: usize = 64 << 10; // a header's first read (this export's are about 5 KiB)
const stage_experts = 64; // experts a stage holds at most: its cut table's rows / 6

/// One copy job of ink2_exl3_cut (inkling2_exl3.cu Ink2CutJob): `rows` rows of `bytes` bytes, src rows `spitch` apart,
/// dst rows `dpitch` apart.
pub const CutJob = extern struct { src: u64, dst: u64, spitch: u64, dpitch: u64, rows: u32, bytes: u32 };

/// File bytes [lo, hi), at stage offset dst + (lo - lo's 4 KiB boundary).
const Range = struct { lo: u64, hi: u64, dst: usize = 0 };

/// One expert's tensors in its batch file: each one's first byte (file offsets).
const At = struct { tr13: u64 = 0, tr2: u64 = 0, suh13: u64 = 0, svh13: u64 = 0, suh2: u64 = 0, svh2: u64 = 0, m13: u64 = 0, m2: u64 = 0 };

/// One batch file of the load: its experts [e0, e1) of `layer`, their tensors, the merged ranges this rank reads and
/// where they land in the file's stage.
const Src = struct {
    path: [:0]const u8,
    layer: u32,
    e0: u32,
    e1: u32,
    at: []At, // [e1 - e0]
    ranges: []Range, // capacity 8 a expert; `nr` in use, sorted, merged
    f: direct_io.File = undefined,
    opened: bool = false,
    nr: usize = 0,
    need: usize = 0, // the ranges' 4 KiB-aligned spans
    stage: u32 = 0,
    own: bool = true, // read by this rank (split mode: every other file)
    xoff: usize = 0, // split mode: the file's peer slots in its stage's send buffer
    err: ?anyerror = null,

    /// The stage offset of file byte `x` (a byte of one of the ranges).
    fn where(s: *const Src, x: u64) !usize {
        for (s.ranges[0..s.nr]) |r| if (x >= r.lo and x < r.hi)
            return r.dst + @as(usize, @intCast(x - std.mem.alignBackward(u64, r.lo, direct_io.alignment)));
        return error.Exl3Plan;
    }
};

const Piece = struct { src: u32, stage: u32, lo: u64, len: usize, dst: usize };

fn monoNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepUs(us: u64) void {
    const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(us * std.time.ns_per_us) };
    _ = std.c.nanosleep(&ts, null);
}

fn envUsize(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

/// Header.parse for a header read alone (`head` holds at least its 8 + n bytes): tensors bound by the file's `size`.
fn parseHead(gpa: std.mem.Allocator, head: []const u8, size: u64) !Header {
    if (head.len < 8) return error.Exl3File;
    const n = std.mem.readInt(u64, head[0..8], .little);
    if (n > head.len - 8 or n > 1 << 26) return error.Exl3File;
    var h: Header = .{ .arena = std.heap.ArenaAllocator.init(gpa), .data = @intCast(8 + n) };
    errdefer h.arena.deinit();
    const a = h.arena.allocator();
    const parsed = try std.json.parseFromSlice(std.json.Value, a, head[8..][0..@intCast(n)], .{ .allocate = .alloc_always });
    const root = objOf(parsed.value) orelse return error.Exl3File;
    var it = root.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
        const o = objOf(kv.value_ptr.*) orelse return error.Exl3File;
        const dt = switch (o.get("dtype") orelse return error.Exl3File) {
            .string => |s| s,
            else => return error.Exl3File,
        };
        const sh = arrOf(o.get("shape")) orelse return error.Exl3File;
        const offs = arrOf(o.get("data_offsets")) orelse return error.Exl3File;
        if (sh.len > 4 or offs.len != 2) return error.Exl3File;
        var e: Header.Entry = .{ .dtype = dt, .shape = @splat(1), .dims = sh.len, .lo = try intOf(offs[0]), .hi = try intOf(offs[1]) };
        for (sh, 0..) |v, i| e.shape[i] = try intOf(v);
        if (e.hi < e.lo or h.data + e.hi > size) return error.Exl3File;
        try h.map.put(a, kv.key_ptr.*, e);
    }
    return h;
}

/// Header.need without the bytes: the entry of `name`, its dtype, shape and byte count checked (logged otherwise).
fn entryOf(h: *const Header, name: []const u8, dtype: []const u8, shape: []const usize, bytes: usize) !Header.Entry {
    const e = h.map.get(name) orelse {
        std.log.err("inkling2 exl3: no tensor {s} in its batch file", .{name});
        return error.Exl3Tensor;
    };
    if (!std.mem.eql(u8, e.dtype, dtype) or e.dims != shape.len or !std.mem.eql(usize, e.shape[0..e.dims], shape) or e.hi - e.lo != bytes) {
        std.log.err("inkling2 exl3: {s} is {s} {any} ({d} bytes), expected {s} {any}", .{ name, e.dtype, e.shape[0..e.dims], e.hi - e.lo, dtype, shape });
        return error.Exl3Tensor;
    }
    return e;
}

/// Expert e's eight tensors in a parsed header, checked as `tensors` checks them (the markers' values are checked
/// once their bytes are read).
fn locate(h: *const Header, e: usize, H: usize, I: usize, k: usize) !At {
    var at: At = .{};
    inline for (.{ "w13", "w2" }) |proj| {
        const w13 = comptime std.mem.eql(u8, proj, "w13");
        const K: usize = if (w13) H else I;
        const N: usize = if (w13) 2 * I else H;
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "experts.{d}." ++ proj ++ ".", .{e});
        var nb: [80]u8 = undefined;
        const mk = try entryOf(h, try std.fmt.bufPrint(&nb, "{s}mul1", .{p}), "I32", &.{}, 4);
        const tr = try entryOf(h, try std.fmt.bufPrint(&nb, "{s}trellis", .{p}), "I16", &.{ K / 16, N / 16, 16 * k }, (K / 16) * (N / 16) * 32 * k);
        const suh = try entryOf(h, try std.fmt.bufPrint(&nb, "{s}suh", .{p}), "F16", &.{K}, 2 * K);
        const svh = try entryOf(h, try std.fmt.bufPrint(&nb, "{s}svh", .{p}), "F16", &.{N}, 2 * N);
        const d: u64 = h.data;
        if (w13) {
            at.m13 = d + mk.lo;
            at.tr13 = d + tr.lo;
            at.suh13 = d + suh.lo;
            at.svh13 = d + svh.lo;
        } else {
            at.m2 = d + mk.lo;
            at.tr2 = d + tr.lo;
            at.suh2 = d + suh.lo;
            at.svh2 = d + svh.lo;
        }
    }
    return at;
}

fn rangeLess(_: void, a: Range, b: Range) bool {
    return a.lo < b.lo;
}

/// File `s`'s experts located in its parsed header and the ranges this rank reads (w13's trellis whole, its rows of
/// w2's — `both`: w2's whole, split mode cuts the peer's slots too —, the four scales whole, the two markers), sorted, merged (gaps under join_gap read through) and laid out from
/// the file's first byte in its stage (the stage plan adds the file's place in the stage).
fn planSrc(s: *Src, h: *const Header, ix: *const Index, H: usize, I: usize, rank: usize, world: usize, both: bool) !void {
    const Il = I / world;
    var nr: usize = 0;
    for (s.e0..s.e1) |e| {
        const k = ix.k(s.layer, e);
        const at = try locate(h, e, H, I, k);
        s.at[e - s.e0] = at;
        const tile = 32 * @as(usize, k);
        const r2: u64 = (H / 16) * tile * (Il / 16);
        const want = [_][2]u64{
            .{ at.tr13, (H / 16) * (2 * I / 16) * tile }, // w13 whole: the rank's column tiles of every row
            if (both) .{ at.tr2, world * r2 } else .{ at.tr2 + rank * r2, r2 }, // w2: the rank's rows (both: whole)
            .{ at.suh13, 2 * H }, .{ at.svh13, 4 * I }, .{ at.suh2, 2 * I }, .{ at.svh2, 2 * H }, // scales whole
            .{ at.m13, 4 }, .{ at.m2, 4 },
        };
        for (want) |r| {
            s.ranges[nr] = .{ .lo = r[0], .hi = r[0] + r[1] };
            nr += 1;
        }
    }
    std.mem.sort(Range, s.ranges[0..nr], {}, rangeLess);
    var m: usize = 0;
    for (s.ranges[0..nr]) |r| {
        if (m > 0 and r.lo <= s.ranges[m - 1].hi + join_gap) {
            s.ranges[m - 1].hi = @max(s.ranges[m - 1].hi, r.hi);
        } else {
            s.ranges[m] = r;
            m += 1;
        }
    }
    s.nr = m;
    var used: usize = 0;
    for (s.ranges[0..m]) |*r| {
        r.dst = used;
        used += direct_io.span(r.lo, @intCast(r.hi - r.lo));
    }
    s.need = used;
}

/// What the lane would refuse in expert `at`'s bytes in its stage `win` (null: nothing): a mul1 marker that is not the
/// mul1 codebook's (that projection), a non-finite fp16 scale (its element) — whole tensors, as openHost checks them.
/// No log here (checkExpert logs): host tests plant both.
const Refusal = union(enum) { codebook: []const u8, scale: usize };

fn refusal(s: *const Src, at: At, win: []const u8, H: usize, I: usize) !?Refusal {
    inline for (.{ .{ "w13", "m13" }, .{ "w2", "m2" } }) |mk| {
        const v = std.mem.readInt(u32, win[try s.where(@field(at, mk[1]))..][0..4], .little);
        if (v != mul1_marker) return .{ .codebook = mk[0] };
    }
    for ([_][2]u64{ .{ at.suh13, 2 * H }, .{ at.svh13, 4 * I }, .{ at.suh2, 2 * I }, .{ at.svh2, 2 * H } }) |sc| {
        const o = try s.where(sc[0]);
        if (badF16(win[o..][0..@intCast(sc[1])])) |bad| return .{ .scale = bad };
    }
    return null;
}

/// `refusal` as the lane's errors, logged with openHost's words.
fn checkExpert(s: *const Src, at: At, win: []const u8, l: usize, e: usize, H: usize, I: usize) !void {
    const r = try refusal(s, at, win, H, I) orelse return;
    switch (r) {
        .codebook => |proj| {
            std.log.err("inkling2 exl3: experts.{d}.{s} is not the mul1 codebook", .{ e, proj });
            return error.Exl3Codebook;
        },
        .scale => |bad| {
            std.log.err("inkling2 exl3: layer {d} expert {d}: a non-finite fp16 scale at element {d}", .{ l, e, bad });
            return error.Exl3Scale;
        },
    }
}

/// packSlot's six cuts of expert `at` (k bits) as ink2_exl3_cut's jobs: src offsets in the stage (Src.where), dst
/// offsets in the slot (the caller adds the stage's and the slot's addresses).
fn cutJobs(s: *const Src, at: At, k: usize, H: usize, I: usize, rank: usize, world: usize) ![6]CutJob {
    const Il = I / world;
    const tile = 32 * k;
    const p = parts(H, Il, k);
    const nt = 2 * Il / 16; // the rank's w13 column tiles of a row
    const row13 = (2 * I / 16) * tile;
    const row2 = (H / 16) * tile;
    const r2 = row2 * (Il / 16);
    return .{
        .{ .src = try s.where(at.tr13 + rank * nt * tile), .dst = 0, .spitch = row13, .dpitch = nt * tile, .rows = @intCast(H / 16), .bytes = @intCast(nt * tile) },
        .{ .src = try s.where(at.tr2 + rank * r2), .dst = p.w2, .spitch = row2, .dpitch = row2, .rows = @intCast(Il / 16), .bytes = @intCast(row2) },
        .{ .src = try s.where(at.suh13), .dst = p.suh13, .spitch = 0, .dpitch = 0, .rows = 1, .bytes = @intCast(2 * H) },
        .{ .src = try s.where(at.svh13 + rank * 4 * Il), .dst = p.svh13, .spitch = 0, .dpitch = 0, .rows = 1, .bytes = @intCast(4 * Il) },
        .{ .src = try s.where(at.suh2 + rank * 2 * Il), .dst = p.suh2, .spitch = 0, .dpitch = 0, .rows = 1, .bytes = @intCast(2 * Il) },
        .{ .src = try s.where(at.svh2), .dst = p.svh2, .spitch = 0, .dpitch = 0, .rows = 1, .bytes = @intCast(2 * H) },
    };
}

/// Step 1, on the reader threads: each file opened O_DIRECT, its header read and parsed, its experts located, the
/// ranges this rank reads merged and laid out (dst from the file's first byte in its stage; the stage is planned after).
const Heads = struct {
    srcs: []Src,
    ix: *const Index,
    H: usize,
    I: usize,
    rank: usize,
    world: usize,
    both: bool, // split mode: both ranks' bytes of each file read here
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(w: *Heads) void {
        const pa = std.heap.page_allocator;
        const buf = pa.alignedAlloc(u8, .fromByteUnits(direct_io.alignment), head_read) catch {
            w.failed.store(true, .release);
            return;
        };
        defer pa.free(buf);
        while (!w.failed.load(.acquire)) {
            const i = w.next.fetchAdd(1, .monotonic);
            if (i >= w.srcs.len) return;
            if (!w.srcs[i].own) continue;
            w.one(&w.srcs[i], buf) catch |err| {
                w.srcs[i].err = err;
                w.failed.store(true, .release);
            };
        }
    }

    fn one(w: *Heads, s: *Src, buf: []align(direct_io.alignment) u8) !void {
        s.f = direct_io.File.open(s.path) catch {
            std.log.err("inkling2 exl3: cannot open {s}", .{s.path});
            return error.Exl3Open;
        };
        s.opened = true;
        const end = std.c.lseek(s.f.fd, 0, std.c.SEEK.END);
        if (end < 8) return error.Exl3File;
        const size: u64 = @intCast(end);
        const first = try s.f.read(buf, 0, @intCast(@min(head_read, size)));
        const n = std.mem.readInt(u64, first[0..8], .little);
        if (n > 1 << 26 or 8 + n > size) return error.Exl3File;
        var h = if (8 + n <= first.len) try parseHead(std.heap.smp_allocator, first, size) else blk: {
            // a header longer than the first read: read it whole
            const pa = std.heap.page_allocator;
            const big = try pa.alignedAlloc(u8, .fromByteUnits(direct_io.alignment), direct_io.span(0, @intCast(8 + n)));
            defer pa.free(big);
            break :blk try parseHead(std.heap.smp_allocator, try s.f.read(big, 0, @intCast(8 + n)), size);
        };
        defer h.deinit();
        try planSrc(s, &h, w.ix, w.H, w.I, w.rank, w.world, w.both);
    }
};

/// One stage of the ring: the reads land in `win` (4 KiB-aligned, mapped: the cut reads it at `dev`), the cut's jobs
/// in `jobs` (mapped), `ev` its cut.
const Ring = struct {
    host: cuda.HostBuffer,
    win: []align(direct_io.alignment) u8,
    dev: u64,
    jobs: cuda.HostBuffer,
    jobs_dev: u64,
    ev: cuda.Event,
    xout: cuda.DeviceBuffer, // split mode: the stage's peer slots, sent round by round (else empty)
};

/// Steps 2 and 3's readers: pieces in order, a stage written only once the main thread gave it back (`allowed`).
const Readers = struct {
    srcs: []Src,
    pieces: []const Piece,
    rings: []Ring,
    done: []std.atomic.Value(usize), // pieces landed, a stage
    next: std.atomic.Value(usize) = .init(0),
    allowed: std.atomic.Value(usize), // the last stage the readers may write
    abort: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    first: anyerror = error.Exl3Read,
    busy_ns: std.atomic.Value(u64) = .init(0),

    fn run(w: *Readers) void {
        while (!w.abort.load(.acquire)) {
            const i = w.next.fetchAdd(1, .monotonic);
            if (i >= w.pieces.len) return;
            const p = w.pieces[i];
            while (w.allowed.load(.acquire) < p.stage) {
                if (w.abort.load(.acquire)) return;
                sleepUs(20);
            }
            const r = &w.rings[p.stage % w.rings.len];
            const t0 = monoNs();
            const dst: direct_io.Buffer = @alignCast(r.win[p.dst..][0..direct_io.span(p.lo, p.len)]);
            _ = w.srcs[p.src].f.read(dst, p.lo, p.len) catch |err| {
                if (w.failed.cmpxchgStrong(false, true, .acq_rel, .monotonic) == null) w.first = err;
                std.log.err("inkling2 exl3: read of {d} bytes at {d} of {s} failed: {t}", .{ p.len, p.lo, w.srcs[p.src].path, err });
                return;
            };
            _ = w.busy_ns.fetchAdd(monoNs() - t0, .monotonic);
            _ = w.done[p.stage].fetchAdd(1, .acq_rel);
        }
    }
};

/// Split mode's TP pair (TF_INKLING2_EXL3_LOAD=split at world 2): rank r reads the batch files i with i % 2 == r and
/// cuts both ranks' slots from them; round j sends the peer its slots of my j-th file and receives mine of the peer's
/// j-th file (NCCL send / recv in one group, after a status word both ways: a rank that failed stops both).
pub const Peer = struct { group: *const cuda.tp_group.Group };

/// Every layer's slots of (rank, world) into `bufs` [L] (each allocated here, after the floor check, the layer's
/// experts back to back in order, as openHost lays them), host_ptr / host_k [L E] their addresses and bits; `cut` is
/// inkling2_exl3.cu's ink2_exl3_cut. With `peer` (world 2): split mode, half the files read here (Peer). Returns the
/// slot bytes. On an error every buffer it allocated is freed again.
pub fn loadDirect(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, cut: cuda.Function, ix: *const Index, dir: []const u8, H: usize, I: usize, rank: usize, world: usize, floor_bytes: u64, bufs: []cuda.DeviceBuffer, host_ptr: []u64, host_k: []i32, stats: *Store.Stats, peer: ?Peer) !u64 {
    _ = io;
    try checkSplit(H, I, world);
    if (peer != null and world != 2) return error.Exl3Split;
    const L = ix.layers;
    const E = ix.experts;
    const Il = I / world;
    const other: usize = if (rank == 0) 1 else 0; // split mode's peer rank
    if (bufs.len != L or host_ptr.len != L * E or host_k.len != L * E) return error.Exl3Shape;
    const ds = &stats.direct;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // ---- step 1: the files and their headers (split mode: this rank's half)
    const t_heads = monoNs();
    var nsrc: usize = 0;
    for (0..L) |l| nsrc += ix.batches[l].len;
    const srcs = try a.alloc(Src, nsrc);
    var si: usize = 0;
    for (0..L) |l| for (ix.batches[l]) |b| {
        const ne = b.end - b.start;
        srcs[si] = .{ .path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, b.file }, 0), .layer = @intCast(l), .e0 = @intCast(b.start), .e1 = @intCast(b.end), .at = try a.alloc(At, ne), .ranges = try a.alloc(Range, 8 * ne), .own = peer == null or si % 2 == rank };
        si += 1;
    };
    defer for (srcs) |*s| if (s.opened) s.f.close();
    // the files this rank reads, in order (split mode: file i with i % 2 == rank)
    var order: std.ArrayList(usize) = .empty;
    for (srcs, 0..) |s, i| if (s.own) try order.append(a, i);
    const readers = std.math.clamp(envUsize("TF_INKLING2_EXL3_READERS", 8), 1, 64);
    ds.readers = readers;
    {
        var w: Heads = .{ .srcs = srcs, .ix = ix, .H = H, .I = I, .rank = rank, .world = world, .both = peer != null };
        var threads: [64]?std.Thread = @splat(null);
        for (threads[1..readers]) |*t| t.* = std.Thread.spawn(.{}, Heads.run, .{&w}) catch null;
        w.run();
        for (threads[1..readers]) |t| if (t) |th| th.join();
        if (w.failed.load(.acquire)) {
            for (srcs) |s| if (s.err) |err| {
                std.log.err("inkling2 exl3: {s}: {t}", .{ s.path, err });
                return err;
            };
            return error.OutOfMemory; // a thread's head buffer
        }
    }
    var most: usize = 0;
    for (order.items) |i| {
        const s = &srcs[i];
        most = @max(most, s.need);
        if (!s.f.direct) ds.cached += 1;
        stats.files += 1;
    }
    if (ds.cached > 0) std.log.warn("inkling2 exl3: {d} batch files do not open O_DIRECT: read through the page cache", .{ds.cached});
    ds.heads_ns = monoNs() - t_heads;

    // ---- the plan: slot offsets, stages (consecutive files of `order` that fit), pieces
    const slot_off = try a.alloc(usize, L * E);
    const layer_bytes = try a.alloc(usize, L);
    for (0..L) |l| {
        var off: usize = 0;
        for (0..E) |e| {
            slot_off[l * E + e] = off;
            off += parts(H, Il, ix.k(l, e)).total;
        }
        layer_bytes[l] = off;
    }
    // a file's slot bytes (one rank's, the same for both) and where they start in its layer's buffer
    const fileSlots = struct {
        fn f(s: *const Src, so: []const usize, x: *const Index, h: usize, il: usize) [2]usize {
            var n: usize = 0;
            for (s.e0..s.e1) |e| n += parts(h, il, x.k(s.layer, e)).total;
            return .{ so[s.layer * x.experts + s.e0], n };
        }
    }.f;
    const nring = std.math.clamp(envUsize("TF_INKLING2_EXL3_STAGES", 4), 2, 16);
    const stage_bytes = std.mem.alignForward(usize, @max(envUsize("TF_INKLING2_EXL3_STAGE_MIB", 128) << 20, most), direct_io.alignment);
    ds.ring = nring;
    ds.stage_bytes = stage_bytes;
    var nst: usize = 0;
    var most_x: usize = 0; // split mode: the most peer-slot bytes a stage sends
    {
        var used: usize = 0;
        var experts: usize = 0;
        var xs: usize = 0;
        for (order.items, 0..) |i, oi| {
            const s = &srcs[i];
            const ne = s.e1 - s.e0;
            if (ne > stage_experts) return error.Exl3Batches;
            if (oi > 0 and (used + s.need > stage_bytes or experts + ne > stage_experts)) {
                nst += 1;
                used = 0;
                experts = 0;
                xs = 0;
            }
            s.stage = @intCast(nst);
            for (s.ranges[0..s.nr]) |*r| r.dst += used;
            used += s.need;
            experts += ne;
            s.xoff = xs;
            xs += fileSlots(s, slot_off, ix, H, Il)[1];
            most_x = @max(most_x, xs);
        }
        nst += 1;
    }
    ds.stages = nst;
    const first = try a.alloc(usize, nst + 1); // stage t's files: order[first[t] .. first[t + 1]]
    @memset(first, order.items.len);
    for (order.items, 0..) |i, oi| first[srcs[i].stage] = @min(first[srcs[i].stage], oi);
    first[nst] = order.items.len;
    const npieces = try a.alloc(usize, nst);
    @memset(npieces, 0);
    var pieces: std.ArrayList(Piece) = .empty;
    for (order.items) |i| {
        const s = &srcs[i];
        for (s.ranges[0..s.nr]) |r| {
            const alo = std.mem.alignBackward(u64, r.lo, direct_io.alignment);
            var at = alo;
            while (at < r.hi) : (at += piece_max) {
                try pieces.append(a, .{ .src = @intCast(i), .stage = s.stage, .lo = at, .len = @intCast(@min(piece_max, r.hi - at)), .dst = r.dst + @as(usize, @intCast(at - alo)) });
                npieces[s.stage] += 1;
            }
        }
    }
    ds.pieces = pieces.items.len;

    // ---- the ring, the side stream, split mode's send buffers and status words
    const rings = try a.alloc(Ring, nring);
    var made: usize = 0;
    defer for (rings[0..made]) |*r| {
        r.ev.deinit();
        r.jobs.free();
        r.host.free();
        if (r.xout.ptr != 0) r.xout.free();
    };
    for (rings) |*r| {
        var host = try cuda.HostBuffer.allocMapped(d, stage_bytes + direct_io.alignment);
        errdefer host.free();
        const base = @intFromPtr(host.bytes.ptr);
        const off = std.mem.alignForward(usize, base, direct_io.alignment) - base; // cuMemHostAlloc promises 256 B
        var jobs = try cuda.HostBuffer.allocMapped(d, 2 * 6 * stage_experts * @sizeOf(CutJob));
        errdefer jobs.free();
        var xout: cuda.DeviceBuffer = .{ .d = d, .ptr = 0, .len = 0 };
        if (peer != null) xout = try cuda.DeviceBuffer.alloc(d, most_x);
        errdefer if (xout.ptr != 0) xout.free();
        r.* = .{ .host = host, .win = @alignCast(host.bytes[off..][0..stage_bytes]), .dev = try host.device() + off, .jobs = jobs, .jobs_dev = try jobs.device(), .ev = try cuda.Event.init(d, false), .xout = xout };
        made += 1;
    }
    var side = try cuda.Stream.init(d, true);
    defer side.deinit();
    var words = try cuda.DeviceBuffer.alloc(d, 16); // split mode: [mine u32][pad] then the gathered [2] u32 at 8
    defer words.free();

    // ---- steps 2 and 3: the readers stream the pieces; this thread checks and cuts each stage in order
    // (a cut may still run when an error leaves: the readers' defer below syncs the side stream before this frees)
    errdefer for (bufs) |*b| if (b.ptr != 0) {
        b.free();
        b.* = .{ .d = d, .ptr = 0, .len = 0 };
    };
    const done = try a.alloc(std.atomic.Value(usize), nst);
    for (done) |*x| x.* = .init(0);
    var sh: Readers = .{ .srcs = srcs, .pieces = pieces.items, .rings = rings, .done = done, .allowed = .init(nring - 1) };
    var threads: [64]?std.Thread = @splat(null);
    defer {
        sh.abort.store(true, .release);
        for (threads[0..readers]) |t| if (t) |th| th.join();
        side.synchronize() catch {};
    }
    const t_stream = monoNs();
    for (threads[0..readers]) |*t| t.* = try std.Thread.spawn(.{}, Readers.run, .{&sh});
    const pace = fill.Pace.fromEnv();
    var lc: LayerCtx = .{ .d = d, .bufs = bufs, .layer_bytes = layer_bytes, .floor_bytes = floor_bytes, .rank = rank, .stats = stats };
    const nrounds = if (peer != null) (srcs.len + 1) / 2 else 0;
    var round: usize = 0; // split mode: rounds done (round j: my j-th file's peer slots out, the peer's j-th file's in)
    var failed: ?anyerror = null;
    stage: for (0..nst) |t| {
        stageWork(t, &sh, rings[t % nring], srcs, order.items[first[t]..first[t + 1]], npieces[t], ix, &lc, slot_off, host_ptr, host_k, H, I, rank, other, world, peer != null, cut, side, ds) catch |err| {
            failed = err;
            break :stage;
        };
        // split mode: this stage's rounds (each waits for the peer's matching round)
        if (peer) |p| for (order.items[first[t]..first[t + 1]]) |i| {
            exchange(p, &words, side, rings[t % nring].xout, srcs, i, round, ix, &lc, slot_off, host_ptr, host_k, H, Il, rank, other, ds) catch |err| {
                failed = err;
                break :stage;
            };
            round += 1;
        };
        const tp = monoNs();
        paceDirect(pace, stats) catch |err| {
            failed = err;
            break :stage;
        };
        ds.pace_ns += monoNs() - tp;
        sh.allowed.store(t + nring, .release);
    }
    // split mode: the peer's rounds past this rank's files (an odd file count), or the peer told of a failure here
    if (peer) |p| {
        if (failed) |err| {
            // the peer waits at this round's status (unless every round is done: then it has gone on); it stops too
            if (err != error.Exl3Peer and round < nrounds) _ = status(p, &words, side, rank, false) catch {};
            return err;
        }
        while (round < nrounds) : (round += 1) try exchange(p, &words, side, rings[0].xout, srcs, null, round, ix, &lc, slot_off, host_ptr, host_k, H, Il, rank, other, ds);
    } else if (failed) |err| return err;
    ds.stream_ns = monoNs() - t_stream;
    ds.busy_ns = sh.busy_ns.load(.acquire);
    for (pieces.items) |p| stats.read_bytes += direct_io.span(p.lo, p.len);
    stats.read_ns = ds.wait_ns;
    stats.pack_ns = ds.check_ns;
    stats.upload_ns = ds.cut_ns;
    std.debug.assert(lc.total == ds.cut_bytes + ds.recv_bytes);
    return lc.total;
}

/// The layer buffers as loadDirect allocates them: on first use, after openHost's floor check.
const LayerCtx = struct {
    d: *const cuda.Driver,
    bufs: []cuda.DeviceBuffer,
    layer_bytes: []const usize,
    floor_bytes: u64,
    rank: usize,
    stats: *Store.Stats,
    total: u64 = 0,

    fn need(c: *LayerCtx, l: usize) !void {
        if (c.bufs[l].ptr != 0) return;
        const ta = monoNs();
        const avail = fill.readAvail() orelse 0;
        c.stats.min_avail = @min(c.stats.min_avail, avail);
        if (avail < c.layer_bytes[l] + c.floor_bytes + (1 << 30)) {
            std.log.err("inkling2 exl3 rank {d}: layer {d}'s {d:.2} GiB of slots do not fit: MemAvailable {d:.2} GiB, floor {d:.2} GiB + 1 GiB (layers 0..{d} loaded, {d:.2} GiB); a smaller context, a lower floor or fewer other programs", .{ c.rank, l, @as(f64, @floatFromInt(c.layer_bytes[l])) / gib, @as(f64, @floatFromInt(avail)) / gib, @as(f64, @floatFromInt(c.floor_bytes)) / gib, l, @as(f64, @floatFromInt(c.total)) / gib });
            return error.Exl3DoesNotFit;
        }
        c.bufs[l] = try cuda.DeviceBuffer.alloc(c.d, c.layer_bytes[l]);
        c.total += c.layer_bytes[l];
        c.stats.direct.alloc_ns += monoNs() - ta;
    }
};

/// One stage of loadDirect: its reads waited for, every expert checked (the lane's refusals), its layers' buffers made,
/// its slots cut on `side` and waited for (split mode: the peer's slots too, into the stage's send buffer).
fn stageWork(t: usize, sh: *Readers, r: Ring, srcs: []Src, files: []const usize, npieces: usize, ix: *const Index, lc: *LayerCtx, slot_off: []const usize, host_ptr: []u64, host_k: []i32, H: usize, I: usize, rank: usize, other: usize, world: usize, split: bool, cut: cuda.Function, side: cuda.Stream, ds: *DirectStats) !void {
    const E = ix.experts;
    const Il = I / world;
    const tw = monoNs();
    while (sh.done[t].load(.acquire) < npieces) {
        if (sh.failed.load(.acquire)) return sh.first;
        sleepUs(20);
    }
    ds.wait_ns += monoNs() - tw;
    const jobs = @as([*]CutJob, @ptrCast(@alignCast(r.jobs.bytes.ptr)))[0 .. 2 * 6 * stage_experts];
    var nj: usize = 0;
    for (files) |i| {
        const s = &srcs[i];
        const l: usize = s.layer;
        try lc.need(l);
        const tc = monoNs();
        var xo: usize = s.xoff; // split mode: the file's peer slots, back to back in the send buffer
        for (s.e0..s.e1) |e| {
            const at = s.at[e - s.e0];
            const k = ix.k(l, e);
            try checkExpert(s, at, r.win, l, e, H, I);
            const n = parts(H, Il, k).total;
            const slot = lc.bufs[l].ptr + slot_off[l * E + e];
            for (try cutJobs(s, at, k, H, I, rank, world)) |j| {
                jobs[nj] = j;
                jobs[nj].src += r.dev;
                jobs[nj].dst += slot;
                nj += 1;
            }
            if (split) {
                for (try cutJobs(s, at, k, H, I, other, world)) |j| {
                    jobs[nj] = j;
                    jobs[nj].src += r.dev;
                    jobs[nj].dst += r.xout.ptr + xo;
                    nj += 1;
                }
                xo += n;
            }
            host_ptr[l * E + e] = slot;
            host_k[l * E + e] = k;
            ds.cut_bytes += n;
        }
        ds.check_ns += monoNs() - tc;
    }
    const tk = monoNs();
    var args: cuda.Args = .{};
    args.add(r.jobs_dev);
    try cuda.launch.launch(cut, .{ .grid = .{ .x = 64, .y = @intCast(nj), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, side, &args);
    try r.ev.record(side);
    try r.ev.synchronize();
    ds.cut_ns += monoNs() - tk;
}

/// Split mode: this rank's status (ok or failed) to the peer and the peer's back; false when either failed.
fn status(p: Peer, words: *cuda.DeviceBuffer, side: cuda.Stream, rank: usize, ok: bool) !bool {
    var w = [2]u32{ @intFromBool(ok), 0 };
    try words.upload(0, std.mem.asBytes(&w[0]));
    try p.group.allGather(side, words.ptr, words.ptr + 8, 4);
    try side.synchronize();
    try p.group.check();
    try words.download(8, std.mem.sliceAsBytes(&w));
    _ = rank;
    return w[0] == 1 and w[1] == 1;
}

/// Split mode's round `round`: statuses both ways, then (one NCCL group) my file `mine`'s peer slots (the stage's send
/// buffer from its xoff) to the peer and the peer's file of this round into my slots; waited for.
fn exchange(p: Peer, words: *cuda.DeviceBuffer, side: cuda.Stream, xout: cuda.DeviceBuffer, srcs: []Src, mine: ?usize, round: usize, ix: *const Index, lc: *LayerCtx, slot_off: []const usize, host_ptr: []u64, host_k: []i32, H: usize, Il: usize, rank: usize, other: usize, ds: *DirectStats) !void {
    const tx = monoNs();
    const E = ix.experts;
    const theirs = 2 * round + other; // the peer's file of this round (file i is rank i % 2's)
    if (theirs < srcs.len) try lc.need(srcs[theirs].layer);
    if (!try status(p, words, side, rank, true)) return error.Exl3Peer;
    const g = p.group;
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        if (mine) |i| {
            const s = &srcs[i];
            var n: usize = 0;
            for (s.e0..s.e1) |e| n += parts(H, Il, ix.k(s.layer, e)).total;
            g.send(side, xout.ptr + s.xoff, n, other) catch |err| break :post err;
            ds.sent_bytes += n;
        }
        if (theirs < srcs.len) {
            const s = &srcs[theirs];
            const l: usize = s.layer;
            var n: usize = 0;
            for (s.e0..s.e1) |e| {
                const k = ix.k(l, e);
                host_ptr[l * E + e] = lc.bufs[l].ptr + slot_off[l * E + e];
                host_k[l * E + e] = k;
                n += parts(H, Il, k).total;
            }
            g.recv(side, lc.bufs[l].ptr + slot_off[l * E + s.e0], n, other) catch |err| break :post err;
            ds.recv_bytes += n;
        }
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    try side.synchronize();
    ds.exchange_ns += monoNs() - tx;
}

/// Store.paceWait's pace (fill.Pace: MemAvailable at least pace.avail_bytes, PSI under pace.psi), counted in `stats`.
fn paceDirect(pace: fill.Pace, stats: *Store.Stats) !void {
    var waited: u32 = 0;
    while (true) : (waited += 2) {
        const avail = fill.readAvail() orelse return;
        const psi = fill.readPsi() orelse 0;
        stats.min_avail = @min(stats.min_avail, avail);
        if (avail >= pace.avail_bytes and psi < pace.psi) return;
        if (waited >= pace.wait_ms) return error.MemoryShort;
        sleepUs(2000);
        stats.paused_ms += 2;
    }
}

/// openHost by loadDirect: the same store (index, per-layer buffers, tables, LUT), its slots read and cut as loadDirect
/// does; logs its numbers.
fn openDirect(gpa: std.mem.Allocator, io: std.Io, m: *model_mod.Model, dir: []const u8, floor_gib: f64) !*Store {
    const c = &m.c;
    const H: usize = c.hidden;
    const I: usize = c.inter;
    const W = m.world;
    try checkSplit(H, I, W);
    // x-mixed: the mixed store sizes its bank (the arena) after these slots, and its LUT is the bank's
    if (!m.arena_on and !m.mix) return error.Exl3NoArena;
    const t0 = monoNs();
    const st = try gpa.create(Store);
    errdefer gpa.destroy(st);
    st.* = .{ .gpa = gpa, .ix = try Index.open(gpa, io, dir, c), .bufs = &.{}, .slot_ptr = undefined, .slot_k = undefined, .tp13 = undefined, .tp2 = undefined, .k2 = undefined, .nslots = c.layers * c.experts };
    errdefer st.ix.deinit();
    const L = c.layers;
    const E = c.experts;
    st.bufs = try gpa.alloc(cuda.DeviceBuffer, L);
    @memset(st.bufs, .{ .d = m.d, .ptr = 0, .len = 0 });
    errdefer {
        for (st.bufs) |*b| if (b.ptr != 0) b.free();
        gpa.free(st.bufs);
    }
    const host_ptr = try gpa.alloc(u64, L * E);
    defer gpa.free(host_ptr);
    const host_k = try gpa.alloc(i32, L * E);
    defer gpa.free(host_k);
    const floor_bytes: u64 = @intFromFloat(floor_gib * gib);
    var peer: ?Peer = null;
    if (loadMode() == .split and W == 2) {
        if (m.group) |*g| peer = .{ .group = g };
    }
    st.bytes = try loadDirect(gpa, io, m.d, try m.k.ex3.function("ink2_exl3_cut"), &st.ix, dir, H, I, m.rank, W, floor_bytes, st.bufs, host_ptr, host_k, &st.stats, peer);
    try tables(st, m, host_ptr, host_k);
    const x = st.stats.direct;
    const s = struct {
        fn f(v: u64) f64 {
            return @as(f64, @floatFromInt(v)) / 1e9;
        }
    }.f;
    std.log.info("inkling2 exl3 rank {d}: {t} load in {d:.1} s: headers {d:.2} s ({d} files), stream {d:.1} s ({d:.1} GB in {d} pieces, {d:.2} GB/s; {d} readers busy {d:.0}%), main thread: reads wait {d:.1} s, checks {d:.2} s, buffers {d:.2} s, cuts {d:.2} s, pace {d:.2} s, exchange {d:.2} s ({d:.1} GB out, {d:.1} GB in); {d} stages of {d} MiB in a ring of {d}; MemAvailable min {d:.2} GiB", .{ m.rank, loadMode(), s(monoNs() - t0), s(x.heads_ns), st.stats.files, s(x.stream_ns), s(st.stats.read_bytes), x.pieces, s(st.stats.read_bytes) / @max(s(x.stream_ns), 1e-9), x.readers, 100 * s(x.busy_ns) / @max(s(x.stream_ns) * @as(f64, @floatFromInt(x.readers)), 1e-9), s(x.wait_ns), s(x.check_ns), s(x.alloc_ns), s(x.cut_ns), s(x.pace_ns), s(x.exchange_ns), s(x.sent_bytes), s(x.recv_bytes), x.stages, x.stage_bytes >> 20, x.ring, @as(f64, @floatFromInt(st.stats.min_avail)) / gib });
    return st;
}

/// openHost's ending, for openDirect: the slot tables (family and engine kernels) and the LUT, every expert routable,
/// published (x1: when the arena holds no slots).
fn tables(st: *Store, m: *model_mod.Model, host_ptr: []const u64, host_k: []const i32) !void {
    const c = &m.c;
    const L = c.layers;
    const E = c.experts;
    st.slot_ptr = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(host_ptr));
    errdefer st.slot_ptr.free();
    st.slot_k = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(host_k));
    errdefer st.slot_k.free();
    const t13 = try st.gpa.alloc(u64, L * E);
    defer st.gpa.free(t13);
    const t2 = try st.gpa.alloc(u64, L * E);
    defer st.gpa.free(t2);
    const k2 = try st.gpa.alloc(i32, L * E);
    defer st.gpa.free(k2);
    for (0..L * E) |i| {
        const bits: usize = @intCast(host_k[i]);
        t13[i] = host_ptr[i];
        t2[i] = host_ptr[i] + parts(c.hidden, c.inter / m.world, bits).w2;
        k2[i] = @intCast(2 * bits);
    }
    st.tp13 = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(t13));
    errdefer st.tp13.free();
    st.tp2 = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(t2));
    errdefer st.tp2.free();
    st.k2 = try cuda.DeviceBuffer.fromHost(m.d, std.mem.sliceAsBytes(k2));
    errdefer st.k2.free();
    // x1 (x-merge, x-mixed DESIGN 2.7): the identity LUT only when no bank owns the arena, as openHost writes it
    const a = &m.arena;
    if (!m.mix and a.slots(0) + a.slots(1) == 0) { // x-mixed: never in the mixed store (moeMix maps its slots)
        for (0..L) |l| for (0..E) |e| {
            a.host_lut[l * E + e] = @intCast(l * E + e);
            a.host_routable[l * E + e] = 1;
        };
        try m.publish();
    }
}

/// fmix64 (MurmurHash3's finalizer), inkling2_exl3.cu ink2_fmix64.
pub fn fmix64(x0: u64) u64 {
    var x = x0;
    x ^= x >> 33;
    x *%= 0xff51afd7ed558ccd;
    x ^= x >> 33;
    x *%= 0xc4ceb9fe1a85ec53;
    x ^= x >> 33;
    return x;
}

/// A slot's hash as ink2_exl3_hash makes it (its bytes a multiple of 8): 256 lanes fold the words t, t + 256, ... with
/// their index, then the lanes' hashes fold in lane order.
pub fn slotHash(bytes: []const u8) u64 {
    const n = bytes.len / 8;
    var out: u64 = n;
    for (0..256) |t| {
        var h: u64 = 0x9E3779B97F4A7C15 ^ @as(u64, t);
        var i: usize = t;
        while (i < n) : (i += 256) h = fmix64(h ^ std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little) ^ (@as(u64, i) *% 0xD6E8FEB86659FD93));
        out = fmix64(out ^ h);
    }
    return out;
}

/// Every slot's hash on the device (ink2_exl3_hash, `f`), [nslots] into `out`: slot_ptr / slot_k are the store's
/// tables (device), H and Il the slots' shape.
pub fn hashSlots(d: *const cuda.Driver, f: cuda.Function, s: cuda.Stream, slot_ptr: u64, slot_k: u64, nslots: usize, H: usize, Il: usize, out: []u64) !void {
    var buf = try cuda.DeviceBuffer.alloc(d, nslots * 8);
    defer buf.free();
    var args: cuda.Args = .{};
    args.add(slot_ptr);
    args.add(slot_k);
    args.add(@as(u64, parts(H, Il, 2).total));
    args.add(@as(u64, parts(H, Il, 3).total));
    args.add(buf.ptr);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(nslots), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &args);
    try s.synchronize();
    try buf.download(0, std.mem.sliceAsBytes(out[0..nslots]));
}

/// TF_INKLING2_EXL3_HASH: each slot's hash (u64 little-endian [L E], layer-major) into PATH.r<rank>, and one line.
fn writeHashes(st: *Store, gpa: std.mem.Allocator, io: std.Io, m: *model_mod.Model, path: []const u8) !void {
    const out = try gpa.alloc(u64, st.nslots);
    defer gpa.free(out);
    try hashSlots(m.d, try m.k.ex3.function("ink2_exl3_hash"), m.s, st.slot_ptr.ptr, st.slot_k.ptr, st.nslots, m.c.hidden, m.inter(), out);
    var all: u64 = st.nslots;
    for (out) |h| all = fmix64(all ^ h);
    const name = try std.fmt.allocPrint(gpa, "{s}.r{d}", .{ path, m.rank });
    defer gpa.free(name);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(out) });
    std.log.info("inkling2 exl3 rank {d}: {d} slot hashes ({t} loader) in {s}; all {x:0>16}", .{ m.rank, st.nslots, loadMode(), name, all });
}

// ---------------------------------------------------------------------------------------------------- tests

test "inkling2 exl3: slot parts at the release's TP2 shape (slot.py's numbers)" {
    const p3 = parts(4608, 1152, 3);
    const p2 = parts(4608, 1152, 2);
    try std.testing.expectEqual(@as(usize, 5_997_312), p3.total);
    try std.testing.expectEqual(@as(usize, 4_006_656), p2.total);
    try std.testing.expectEqual(@as(usize, 288 * 144 * 96), p3.w2);
    try std.testing.expectEqual(@as(usize, 3_981_312 + 1_990_656), p3.suh13);
    try checkSplit(4608, 2304, 2);
    try checkSplit(4608, 2304, 1);
    try std.testing.expectError(error.Exl3Split, checkSplit(4608, 2304, 4)); // 576 channels: w2's 128-row blocks cut
}

test "inkling2 exl3: packSlot takes the rank's w13 output tiles from every row and w2's input tile rows" {
    const gpa = std.testing.allocator;
    const H = 256;
    const I = 256;
    const k = 2;
    const tile = 32 * k;
    // tile bytes tagged by (row tile, column tile)
    const t13 = try gpa.alloc(u8, (H / 16) * (2 * I / 16) * tile);
    defer gpa.free(t13);
    for (0..H / 16) |r| for (0..2 * I / 16) |cc| @memset(t13[(r * (2 * I / 16) + cc) * tile ..][0..tile], @intCast((r * 7 + cc) & 0xFF));
    const t2 = try gpa.alloc(u8, (I / 16) * (H / 16) * tile);
    defer gpa.free(t2);
    for (t2, 0..) |*b, i| b.* = @intCast((i / tile) & 0xFF);
    const s = try gpa.alloc(u8, 4 * I + 2 * H);
    defer gpa.free(s);
    for (s, 0..) |*b, i| b.* = @intCast(i & 0xFF);
    const w13: Mat = .{ .trellis = t13, .suh = s[0 .. 2 * H], .svh = s[0 .. 4 * I] };
    const w2: Mat = .{ .trellis = t2, .suh = s[0 .. 2 * I], .svh = s[0 .. 2 * H] };
    const p = parts(H, I / 2, k);
    const out = try gpa.alloc(u8, p.total);
    defer gpa.free(out);
    try packSlot(w13, w2, H, I, 1, 2, k, out);
    // rank 1's w13 column tiles are [16, 32) of 32; row tile 3, its first tile = column tile 16
    try std.testing.expectEqual(@as(u8, @intCast(3 * 7 + 16)), out[(3 * 16 + 0) * tile]);
    // w2: rank 1's first row tile is 8: its tiles start at tile index 8 * 16
    try std.testing.expectEqual(@as(u8, @intCast((8 * 16) & 0xFF)), out[p.w2]);
    // svh13: the rank's 2 Il fp16 values from byte 4 Il (I / 2 channels a rank)
    try std.testing.expectEqual(s[4 * (I / 2)], out[p.svh13]);
    try std.testing.expectEqual(s[2 * (I / 2)], out[p.suh2]);
}

test "inkling2 exl3: the header reader finds a tensor and checks dtype and shape" {
    const gpa = std.testing.allocator;
    const json = "{\"experts.0.w13.mul1\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}";
    var buf: [8 + json.len + 4]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], json.len, .little);
    @memcpy(buf[8..][0..json.len], json);
    std.mem.writeInt(u32, buf[8 + json.len ..][0..4], mul1_marker, .little);
    var h = try Header.parse(gpa, &buf);
    defer h.deinit();
    const b = try h.get(&buf, "experts.0.w13.mul1", "I32", &.{});
    try std.testing.expectEqual(mul1_marker, std.mem.readInt(u32, b[0..4], .little));
    try std.testing.expectError(error.Exl3Tensor, h.get(&buf, "experts.0.w13.mul1", "I16", &.{}));
}

test "inkling2 exl3 e-load: header, ranges, stage and cut jobs give packSlot's bytes; the refusals hold" {
    // a batch file laid out as the export's (markers, every expert's scales, then each expert's w13 and w2 trellis), two
    // experts at 2 and 3 bits, at a shape whose other-rank w2 half (128 KiB) is wider than join_gap
    const gpa = std.testing.allocator;
    const H = 1024;
    const I = 1024;
    const world = 2;
    var bits = [_]u8{ 2, 3 };
    var ix: Index = .{ .arena = std.heap.ArenaAllocator.init(gpa), .layers = 1, .experts = 2, .bits = &bits, .batches = &.{} };
    defer ix.deinit();
    const Tn = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: usize };
    var names: [2][8][48]u8 = undefined;
    var list: [16]Tn = undefined;
    var n: usize = 0;
    // order: markers, scales, trellises (as the export writes them)
    for (0..2) |e| for ([_][]const u8{ "w13", "w2" }, 0..) |proj, pi| {
        list[n] = .{ .name = try std.fmt.bufPrint(&names[e][pi], "experts.{d}.{s}.mul1", .{ e, proj }), .dtype = "I32", .shape = &.{}, .bytes = 4 };
        n += 1;
    };
    const sh13 = [_]usize{H};
    const sv13 = [_]usize{2 * I};
    const sh2 = [_]usize{I};
    const sv2 = [_]usize{H};
    for (0..2) |e| {
        list[n] = .{ .name = try std.fmt.bufPrint(&names[e][2], "experts.{d}.w13.suh", .{e}), .dtype = "F16", .shape = &sh13, .bytes = 2 * H };
        list[n + 1] = .{ .name = try std.fmt.bufPrint(&names[e][3], "experts.{d}.w13.svh", .{e}), .dtype = "F16", .shape = &sv13, .bytes = 4 * I };
        list[n + 2] = .{ .name = try std.fmt.bufPrint(&names[e][4], "experts.{d}.w2.suh", .{e}), .dtype = "F16", .shape = &sh2, .bytes = 2 * I };
        list[n + 3] = .{ .name = try std.fmt.bufPrint(&names[e][5], "experts.{d}.w2.svh", .{e}), .dtype = "F16", .shape = &sv2, .bytes = 2 * H };
        n += 4;
    }
    var tshape: [2][2][3]usize = undefined;
    for (0..2) |e| {
        tshape[e][0] = .{ H / 16, 2 * I / 16, 16 * @as(usize, bits[e]) };
        tshape[e][1] = .{ I / 16, H / 16, 16 * @as(usize, bits[e]) };
        list[n] = .{ .name = try std.fmt.bufPrint(&names[e][6], "experts.{d}.w13.trellis", .{e}), .dtype = "I16", .shape = &tshape[e][0], .bytes = (H / 16) * (2 * I / 16) * 32 * @as(usize, bits[e]) };
        list[n + 1] = .{ .name = try std.fmt.bufPrint(&names[e][7], "experts.{d}.w2.trellis", .{e}), .dtype = "I16", .shape = &tshape[e][1], .bytes = (I / 16) * (H / 16) * 32 * @as(usize, bits[e]) };
        n += 2;
    }
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(gpa);
    try json.append(gpa, '{');
    var off: usize = 0;
    for (list[0..n], 0..) |t, i| {
        if (i > 0) try json.append(gpa, ',');
        try json.print(gpa, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        for (t.shape, 0..) |x, j| try json.print(gpa, "{s}{d}", .{ if (j > 0) "," else "", x });
        try json.print(gpa, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.bytes });
        off += t.bytes;
    }
    try json.append(gpa, '}');
    while ((8 + json.items.len) % 8 != 0) try json.append(gpa, ' '); // safetensors pads the header to 8 bytes
    const data = 8 + json.items.len;
    const file = try gpa.alloc(u8, data + off);
    defer gpa.free(file);
    std.mem.writeInt(u64, file[0..8], json.items.len, .little);
    @memcpy(file[8..data], json.items);
    var prng = std.Random.DefaultPrng.init(0x5eed);
    prng.random().bytes(file[data..]);
    // markers and finite scales (fp16 1.0 .. 2.0)
    off = data;
    for (list[0..n]) |t| {
        if (std.mem.endsWith(u8, t.name, "mul1")) std.mem.writeInt(u32, file[off..][0..4], mul1_marker, .little);
        if (std.mem.eql(u8, t.dtype, "F16")) {
            var i: usize = 0;
            while (i < t.bytes) : (i += 2) std.mem.writeInt(u16, file[off + i ..][0..2], 0x3C00 | @as(u16, @intCast(i & 0x3FF)), .little);
        }
        off += t.bytes;
    }
    for (0..world) |rank| {
        var at_buf: [2]At = undefined;
        var range_buf: [16]Range = undefined;
        var s: Src = .{ .path = "x", .layer = 0, .e0 = 0, .e1 = 2, .at = &at_buf, .ranges = &range_buf };
        var h = try parseHead(gpa, file[0..data], file.len);
        defer h.deinit();
        try planSrc(&s, &h, &ix, H, I, rank, world, false);
        try std.testing.expect(s.nr >= 2); // the other rank's w2 halves are not read
        var read: usize = 0;
        for (s.ranges[0..s.nr]) |r| read += r.hi - r.lo;
        try std.testing.expect(read < file.len - data);
        // the reads, as O_DIRECT lands them: each range's 4 KiB-aligned span at its dst
        const stage = try gpa.alloc(u8, s.need);
        defer gpa.free(stage);
        @memset(stage, 0xEE);
        for (s.ranges[0..s.nr]) |r| {
            const alo: usize = @intCast(std.mem.alignBackward(u64, r.lo, direct_io.alignment));
            const end = @min(alo + direct_io.span(r.lo, @intCast(r.hi - r.lo)), file.len);
            @memcpy(stage[r.dst..][0 .. end - alo], file[alo..end]);
        }
        var hw = try Header.parse(gpa, file);
        defer hw.deinit();
        for (0..2) |e| {
            const k: usize = bits[e];
            try std.testing.expectEqual(@as(?Refusal, null), try refusal(&s, s.at[e], stage, H, I));
            const p = parts(H, I / world, k);
            const got = try gpa.alloc(u8, p.total);
            defer gpa.free(got);
            @memset(got, 0xAA);
            for (try cutJobs(&s, s.at[e], k, H, I, rank, world)) |j| for (0..j.rows) |row| {
                @memcpy(got[j.dst + row * j.dpitch ..][0..j.bytes], stage[j.src + row * j.spitch ..][0..j.bytes]);
            };
            const want = try gpa.alloc(u8, p.total);
            defer gpa.free(want);
            try packSlot(try tensors(&hw, file, e, "w13", H, I, k), try tensors(&hw, file, e, "w2", H, I, k), H, I, rank, world, k, want);
            try std.testing.expectEqualSlices(u8, want, got);
            try std.testing.expectEqual(slotHash(want), slotHash(got));
        }
        // the refusals: a non-finite scale, a marker that is not mul1's
        const sv = try s.where(s.at[1].svh2 + 6);
        const keep = stage[sv + 1];
        stage[sv + 1] = 0x7C;
        try std.testing.expectEqual(@as(usize, 3), (try refusal(&s, s.at[1], stage, H, I)).?.scale);
        stage[sv + 1] = keep;
        const mk = try s.where(s.at[0].m2);
        stage[mk] ^= 1;
        try std.testing.expectEqualStrings("w2", (try refusal(&s, s.at[0], stage, H, I)).?.codebook);
    }
}
