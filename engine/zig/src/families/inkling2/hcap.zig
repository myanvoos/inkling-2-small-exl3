//! e-dense (prefill lane phase 5): the calibration Hessians of Inkling-2's dense linears, H = sum over rows of x^T x in
//! fp32, captured inside the engine's forward (the run tool's --hcap DIR over a calibration set: the served model's own
//! activations, the EXL3 export's experts included) for the fork's quantizer (tools/quantize_dense.py: ExLlamaV3's
//! quantize_exl3, which divides by the row count, regularizes and factors H itself).
//! Inputs a layer: `q` the attention input (x.xq: wq_du, gate_attn, wr_du, and wk_dv / wv_dv on sliding layers),
//! `k` / `v` a global layer's own k / v inputs, `o` o_proj's input (the gated attention output, every rank's heads:
//! gathered, so H is the whole [6144][6144]), `m` the router's input (x.xm); and the head's input (the final norm).
//! Each product is one cuBLASLt GEMM (bf16 operands, fp32 sums, beta 1) into a [K][K] fp32 buffer. The ranks split
//! the layers in two halves (each holds ~8.4 GiB at 24 layers); rank 0 also takes the head. Off unless the run tool
//! opens it.
const std = @import("std");
const cuda = @import("cuda");

const lt_ = cuda.cublaslt;

pub const Input = enum { q, k, v, o, m };

const Key = struct { m: u64, k: u64, lda: u64, ldc: u64 };

pub const Hcap = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lt: *const lt_.Library,
    handle: lt_.Handle,
    ws: cuda.DeviceBuffer,
    gemms: std.AutoHashMapUnmanaged(Key, lt_.Gemm) = .empty,
    /// This rank's layers [lo, hi) and whether it takes the head's input.
    lo: usize,
    hi: usize,
    head: bool,
    bufs: [][5]?cuda.DeviceBuffer,
    head_buf: ?cuda.DeviceBuffer = null,
    widths: [][5]usize,
    head_k: usize = 0,
    /// o_proj's inputs of every rank, gathered: [world][rows][qw] bf16.
    gather: ?cuda.DeviceBuffer = null,
    /// Rows accumulated (every captured H the same).
    rows: u64 = 0,

    const ws_bytes: usize = 32 << 20;

    /// Layers [lo, hi) of `layers` split in two halves over two ranks (world 1: all of them), the head on rank 0.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lt: *const lt_.Library, handle: lt_.Handle, layers: usize, lo: usize, hi: usize, rank: usize, world: usize) !*Hcap {
        const h = try gpa.create(Hcap);
        errdefer gpa.destroy(h);
        const mid = lo + (hi - lo + 1) / 2;
        h.* = .{ .gpa = gpa, .d = d, .lt = lt, .handle = handle, .ws = try cuda.DeviceBuffer.alloc(d, ws_bytes), .lo = if (world == 1 or rank == 0) lo else mid, .hi = if (world == 1 or rank == 1) hi else mid, .head = rank == 0, .bufs = try gpa.alloc([5]?cuda.DeviceBuffer, layers), .widths = try gpa.alloc([5]usize, layers) };
        for (h.bufs) |*b| b.* = @splat(null);
        for (h.widths) |*w| w.* = @splat(0);
        return h;
    }

    pub fn deinit(h: *Hcap) void {
        var it = h.gemms.valueIterator();
        while (it.next()) |g| g.deinit();
        h.gemms.deinit(h.gpa);
        for (h.bufs) |*row| for (row) |*b| if (b.*) |*x| x.free();
        if (h.head_buf) |*b| b.free();
        if (h.gather) |*b| b.free();
        h.ws.free();
        h.gpa.free(h.bufs);
        h.gpa.free(h.widths);
        h.gpa.destroy(h);
    }

    /// C [K][K] (column-major, `ldc`) += A A'^T for column-major A, A' [K][T] (`lda`): row-major x [T][K]'s x^T x'.
    fn syrk(h: *Hcap, s: cuda.Stream, a: u64, b: u64, c: u64, K: usize, T: usize, lda: usize, ldc: usize) !void {
        const key: Key = .{ .m = K, .k = T, .lda = lda, .ldc = ldc };
        const got = try h.gemms.getOrPut(h.gpa, key);
        if (!got.found_existing) {
            got.value_ptr.* = lt_.Gemm.init(h.lt, h.handle, .{ .ta = .n, .tb = .t, .m = K, .n = K, .k = T, .a = .bf16, .b = .bf16, .c = .f32, .lda = lda, .ldb = lda, .ldc = ldc }, ws_bytes) catch |err| {
                _ = h.gemms.remove(key);
                return err;
            };
        }
        try got.value_ptr.run(1, a, b, 1, c, h.ws.ptr, ws_bytes, s.handle);
    }

    fn buf(h: *Hcap, s: cuda.Stream, slot: *?cuda.DeviceBuffer, K: usize) !u64 {
        if (slot.* == null) {
            var b = try cuda.DeviceBuffer.alloc(h.d, K * K * 4);
            errdefer b.free();
            try b.fill8(0, s.handle);
            slot.* = b;
        }
        return slot.*.?.ptr;
    }

    fn mine(h: *const Hcap, li: usize) bool {
        return li >= h.lo and li < h.hi;
    }

    /// x [T][K] bf16 (rows contiguous) into layer li's H of `which` (a replicated input: every rank has the same rows).
    pub fn add(h: *Hcap, s: cuda.Stream, li: usize, which: Input, x: u64, T: usize, K: usize) !void {
        if (!h.mine(li)) return;
        const c = try h.buf(s, &h.bufs[li][@intFromEnum(which)], K);
        h.widths[li][@intFromEnum(which)] = K;
        try h.syrk(s, x, x, c, K, T, K, K);
    }

    /// o_proj's input: every rank's [T][qw] gathered (all ranks call this for every layer: a collective), then this
    /// rank's layers' blocks H[i][j] = x_i^T x_j of the whole [W qw][W qw].
    pub fn addO(h: *Hcap, s: cuda.Stream, group: ?*const cuda.tp_group.Group, world: usize, li: usize, x: u64, T: usize, qw: usize) !void {
        var all = x;
        if (world > 1) {
            const need = world * T * qw * 2;
            if (h.gather == null or h.gather.?.len < need) {
                if (h.gather) |*g| g.free();
                h.gather = null;
                try s.synchronize();
                h.gather = try cuda.DeviceBuffer.alloc(h.d, need);
            }
            try group.?.allGather(s, x, h.gather.?.ptr, T * qw * 2);
            all = h.gather.?.ptr;
        }
        if (!h.mine(li)) return;
        const K = world * qw;
        const c = try h.buf(s, &h.bufs[li][@intFromEnum(Input.o)], K);
        h.widths[li][@intFromEnum(Input.o)] = K;
        for (0..world) |i| for (0..world) |j| {
            try h.syrk(s, all + i * T * qw * 2, all + j * T * qw * 2, c + (i * qw + j * qw * K) * 4, qw, T, qw, K);
        };
    }

    /// The head's input (the final norm's rows), on the rank that takes it.
    pub fn addHead(h: *Hcap, s: cuda.Stream, x: u64, T: usize, K: usize) !void {
        if (!h.head) return;
        const c = try h.buf(s, &h.head_buf, K);
        h.head_k = K;
        try h.syrk(s, x, x, c, K, T, K, K);
    }

    /// Writes every H this rank holds as `DIR/L<li>-<input>.f32` and `DIR/head.f32` (fp32 [K][K], symmetric), with
    /// `DIR/hcap-r<rank>.json` listing them and the row count.
    pub fn write(h: *Hcap, io: std.Io, s: cuda.Stream, dir: []const u8, rank: usize) !void {
        try s.synchronize();
        try std.Io.Dir.cwd().createDirPath(io, dir);
        var list: std.ArrayList(u8) = .empty;
        defer list.deinit(h.gpa);
        try list.print(h.gpa, "{{\"rank\": {d}, \"rows\": {d}, \"layers\": [{d}, {d}], \"files\": [", .{ rank, h.rows, h.lo, h.hi });
        var first = true;
        var name: [512]u8 = undefined;
        for (h.bufs, 0..) |row, li| for (row, 0..) |b, wi| if (b) |x| {
            const K = h.widths[li][wi];
            const file = try std.fmt.bufPrint(&name, "L{d:0>2}-{s}.f32", .{ li, @tagName(@as(Input, @enumFromInt(wi))) });
            try h.dump(io, dir, file, x, K);
            try list.print(h.gpa, "{s}{{\"file\": \"{s}\", \"layer\": {d}, \"input\": \"{s}\", \"k\": {d}}}", .{ if (first) "" else ", ", file, li, @tagName(@as(Input, @enumFromInt(wi))), K });
            first = false;
        };
        if (h.head_buf) |x| {
            try h.dump(io, dir, "head.f32", x, h.head_k);
            try list.print(h.gpa, "{s}{{\"file\": \"head.f32\", \"layer\": -1, \"input\": \"head\", \"k\": {d}}}", .{ if (first) "" else ", ", h.head_k });
        }
        try list.appendSlice(h.gpa, "]}\n");
        const json = try std.fmt.bufPrint(&name, "hcap-r{d}.json", .{rank});
        var path_buf: [1024]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, json }), .data = list.items });
    }

    fn dump(h: *Hcap, io: std.Io, dir: []const u8, file: []const u8, x: cuda.DeviceBuffer, K: usize) !void {
        const host = try h.gpa.alloc(u8, K * K * 4);
        defer h.gpa.free(host);
        try x.download(0, host);
        var path_buf: [1024]u8 = undefined;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, file }), .data = host });
    }
};
