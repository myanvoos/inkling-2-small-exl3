//! Inkling-2's dense products on cuBLASLt, fp32 compute (no TF32): torch's bf16 nn.Linear (bf16 out: one rounding
//! of the fp32 sum), its fp32-out variant (TP partials, the router's logits: bf16 x bf16 products are exact in fp32,
//! so this is the reference's fp32 x fp32 GEMM up to summation order), and the relative bias product
//! rel [m][d_rel] . proj [d_rel][extent] (the weight not transposed). Plans are made once a shape and kept.
const std = @import("std");
const cuda = @import("cuda");

const lt_ = cuda.cublaslt;

const Key = struct { m: u64, n: u64, k: u64, out_f32: bool, plain: bool };

pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: lt_.Library,
    handle: lt_.Handle,
    ws: cuda.DeviceBuffer,
    linear: std.AutoHashMapUnmanaged(Key, lt_.Linear) = .empty,
    plain: std.AutoHashMapUnmanaged(Key, lt_.Gemm) = .empty,

    const ws_bytes: usize = 32 << 20;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try lt_.Library.open();
        errdefer lt.close();
        var h: lt_.Handle = null;
        try lt.check(lt.api.cublasLtCreate(&h), "cublasLtCreate");
        return .{ .gpa = gpa, .lt = lt, .handle = h, .ws = try cuda.DeviceBuffer.alloc(d, ws_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.linear.valueIterator();
        while (it.next()) |v| v.deinit();
        g.linear.deinit(g.gpa);
        var pit = g.plain.valueIterator();
        while (pit.next()) |v| v.deinit();
        g.plain.deinit(g.gpa);
        _ = g.lt.api.cublasLtDestroy(g.handle);
        g.ws.free();
        g.lt.close();
    }

    /// d [m][n] = x [m][k] . w[n][k]^T; x and w bf16, d bf16 or fp32 (`f32`).
    pub fn run(g: *Gemms, s: cuda.Stream, x: u64, w: u64, d: u64, m: usize, n: usize, k: usize, out_f32: bool) !void {
        const key: Key = .{ .m = m, .n = n, .k = k, .out_f32 = out_f32, .plain = false };
        const got = try g.linear.getOrPut(g.gpa, key);
        if (!got.found_existing) {
            got.value_ptr.* = lt_.Linear.init(&g.lt, m, n, k, if (out_f32) .f32 else .bf16, ws_bytes) catch |err| {
                _ = g.linear.remove(key);
                return err;
            };
        }
        try got.value_ptr.run(x, w, d, g.ws.ptr, ws_bytes, s.handle);
    }

    /// d [m][n] = x [m][k] . p [k][n], all bf16 row-major (the relative bias: rel . rel_logits_proj.proj).
    pub fn rowsTimes(g: *Gemms, s: cuda.Stream, x: u64, p: u64, d: u64, m: usize, n: usize, k: usize) !void {
        const key: Key = .{ .m = m, .n = n, .k = k, .out_f32 = false, .plain = true };
        const got = try g.plain.getOrPut(g.gpa, key);
        if (!got.found_existing) {
            // column-major: D^T [n][m] = P^T [n][k] . X^T [k][m]; a row-major [r][c] matrix is column-major [c][r]
            got.value_ptr.* = lt_.Gemm.init(&g.lt, g.handle, .{ .m = n, .n = m, .k = k, .a = .bf16, .b = .bf16, .c = .bf16, .lda = n, .ldb = k, .ldc = n }, ws_bytes) catch |err| {
                _ = g.plain.remove(key);
                return err;
            };
        }
        try got.value_ptr.run(1, p, x, 0, d, g.ws.ptr, ws_bytes, s.handle);
    }
};
