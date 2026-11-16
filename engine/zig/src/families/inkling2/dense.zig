//! A TP rank's dense (non-expert) weights on the GPU exactly as stored: each planned tensor's rank slice is cut on
//! the host from the mapped shard (whole rows, or the split dimension's [lo, hi) of every outer row), checked for
//! non-finite values and uploaded once. No dtype is converted; the experts are the residency arena's (plan.Kind).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const plan_mod = @import("plan.zig");
const Config = @import("config.zig").Config;

const Entry = plan_mod.Entry;

pub const Tensor = struct {
    name: []const u8,
    entry: Entry,
    buf: cuda.DeviceBuffer,
};

pub const Dense = struct {
    gpa: std.mem.Allocator,
    tensors: std.ArrayList(Tensor) = .empty,
    bytes: u64 = 0,

    pub fn deinit(d: *Dense) void {
        for (d.tensors.items) |*t| t.buf.free();
        d.tensors.deinit(d.gpa);
        d.* = undefined;
    }

    /// The tensor named `name` (its original checkpoint name), or null.
    pub fn get(d: *const Dense, name: []const u8) ?*const Tensor {
        for (d.tensors.items) |*t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }
};

/// Copies the rank's slice of `src` (the whole tensor's bytes, row-major) into `out`.
pub fn cut(e: *const Entry, src: []const u8, out: []u8) void {
    const es = e.dtype.size();
    const d = e.split orelse {
        @memcpy(out, src);
        return;
    };
    var outer: usize = 1;
    for (e.shape[0..d]) |x| outer *= x;
    var inner: usize = es;
    for (e.shape[d + 1 .. e.dims]) |x| inner *= x;
    const mid = e.shape[d];
    const take = (e.hi - e.lo) * inner;
    for (0..outer) |o| {
        const from = o * mid * inner + e.lo * inner;
        @memcpy(out[o * take ..][0..take], src[from..][0..take]);
    }
}

/// The first non-finite element's index of a bf16 or f32 slice, or null (other dtypes: null).
pub fn firstNonFinite(e: *const Entry, bytes: []const u8) ?usize {
    switch (e.dtype) {
        .bf16 => {
            var i: usize = 0;
            while (i + 1 < bytes.len) : (i += 2) {
                const v = std.mem.readInt(u16, bytes[i..][0..2], .little);
                if (v & 0x7F80 == 0x7F80) return i / 2;
            }
        },
        .f32 => {
            var i: usize = 0;
            while (i + 3 < bytes.len) : (i += 4) {
                const v = std.mem.readInt(u32, bytes[i..][0..4], .little);
                if (v & 0x7F800000 == 0x7F800000) return i / 4;
            }
        },
        else => {},
    }
    return null;
}

/// Loads every dense entry of `entries` from `ck` onto the device: dtype and whole shape checked (`expect`),
/// non-finite values refused by name. Needs a current CUDA context.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ck: *core.checkpoint.Checkpoint, entries: []const Entry) !Dense {
    var out: Dense = .{ .gpa = gpa };
    errdefer out.deinit();
    var host: std.ArrayList(u8) = .empty;
    defer host.deinit(gpa);
    for (entries) |*e| {
        if (e.kind != .dense) continue;
        const t = try ck.expect(e.name, e.dtype, e.wholeShape());
        const n = e.rankBytes();
        try host.resize(gpa, n);
        cut(e, t.bytes, host.items);
        if (firstNonFinite(e, host.items)) |at| {
            std.log.err("{s}: non-finite value at element {d} of this rank's slice", .{ e.name, at });
            return error.NonFinite;
        }
        const buf = try cuda.DeviceBuffer.fromHost(d, host.items);
        try out.tensors.append(gpa, .{ .name = e.name, .entry = e.*, .buf = buf });
        out.bytes += n;
    }
    return out;
}

test "inkling2 dense: the cut takes rows, columns and middle dimensions of row-major bytes" {
    // a [2, 3, 2] u8 tensor 0..11; rank 1 of 3 along dim 1 takes [1, 2): elements (o, 1, i) for o in 0..2, i in 0..2
    var e: Entry = .{ .name = "t", .dtype = .u8, .dims = 3, .shape = .{ 2, 3, 2 }, .split = 1, .lo = 1, .hi = 2 };
    var src: [12]u8 = undefined;
    for (&src, 0..) |*x, i| x.* = @intCast(i);
    var out: [4]u8 = undefined;
    cut(&e, &src, &out);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3, 8, 9 }, &out);
    // columns of a [3, 4] bf16 matrix, rank 1 of 2: elements 2..3 of every row
    e = .{ .name = "m", .dtype = .bf16, .dims = 2, .shape = .{ 3, 4, 1 }, .split = 1, .lo = 2, .hi = 4 };
    var m: [24]u8 = undefined;
    for (&m, 0..) |*x, i| x.* = @intCast(i);
    var mo: [12]u8 = undefined;
    cut(&e, &m, &mo);
    try std.testing.expectEqualSlices(u8, &.{ 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23 }, &mo);
}

test "inkling2 dense: non-finite bf16 and f32 values are found, finite ones pass" {
    const e16: Entry = .{ .name = "a", .dtype = .bf16, .dims = 1, .shape = .{ 3, 1, 1 } };
    try std.testing.expectEqual(@as(?usize, null), firstNonFinite(&e16, &.{ 0x80, 0x3F, 0x00, 0x00, 0x7F, 0xFF }));
    try std.testing.expectEqual(@as(?usize, 1), firstNonFinite(&e16, &.{ 0x80, 0x3F, 0x80, 0x7F, 0x00, 0x00 }));
    const e32: Entry = .{ .name = "b", .dtype = .f32, .dims = 1, .shape = .{ 2, 1, 1 } };
    try std.testing.expectEqual(@as(?usize, 1), firstNonFinite(&e32, &.{ 0, 0, 0x80, 0x3F, 0, 0, 0xC0, 0x7F }));
}
