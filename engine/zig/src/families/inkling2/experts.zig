//! One TP rank's routed experts, one (layer, expert) at a time, packed into the residency arena's slot layouts from the
//! stored bytes. Nothing is decoded or rounded: packing is a byte permutation, and the NVFP4 global scale is the
//! reference's one division g = fp32(amax / denominator).
//!
//! A rank's channels [rank I / world, ...) are one contiguous range of w13's stored rows (gate and up interleave per
//! channel: g0, u0, g1, u1, ...): the even rows are the gate's, the odd rows the up's. w2 is cut by columns.
//! NVFP4 slot (core cuda/experts.zig Nvfp4Experts): up [I/32][H/32][2][144] int32 (gate and up per block), down
//! [H/32][I/32][1][144], scales (g13, g13) and g2. MXFP8 slot (the reference mxfp8_lane.pack): up [I/32][H/32][2][264],
//! down [H/32][I/32][1][264] (256 words of e4m3 codes, then 8 of UE8M0 exponents).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const Config = config.Config;

const Nv = cuda.experts.Nvfp4Experts;
pub const nv_words = Nv.words; // 144
pub const mx_words = 264;

pub const Slot = struct {
    fmt: config.Format,
    up: []u32,
    down: []u32,
    g13: f32 = 0, // NVFP4: the gate's and up's global scale (one w13 amax)
    g2: f32 = 0,
};

/// Words of one slot's up and down for this rank's `inter` channels (I / world).
pub fn slotWords(fmt: config.Format, hidden: usize, inter: usize) [2]usize {
    const w: usize = if (fmt == .nvfp4) nv_words else mx_words;
    return .{ (inter / 32) * (hidden / 32) * 2 * w, (hidden / 32) * (inter / 32) * w };
}

/// The checkpoint's expert tensors of one layer (whole, mapped) and what a rank takes of them.
pub const Store = struct {
    c: *const Config,
    ck: *core.checkpoint.Checkpoint,
    rank: u32,
    world: u32,
    /// [layer][expert] (g13, g2) for NVFP4 layers, from `loadGlobals` (the fill's pack threads must not touch `ck`);
    /// null: `pack` reads them itself.
    globals: ?[]const [2]f32 = null,

    /// Every NVFP4 expert's global scales, checked (`global`), so packs on threads need nothing from the checkpoint.
    pub fn loadGlobals(s: *Store, gpa: std.mem.Allocator) !void {
        const E = s.c.experts;
        const g = try gpa.alloc([2]f32, s.c.layers * E);
        @memset(g, .{ 0, 0 });
        for (0..s.c.layers) |l| if (s.c.formats[l] == .nvfp4) for (0..E) |e| {
            g[l * E + e] = .{ try s.global(@intCast(l), "w13", e), try s.global(@intCast(l), "w2", e) };
        };
        s.globals = g;
    }

    pub fn inter(s: *const Store) usize {
        return s.c.inter / s.world;
    }

    fn tensor(s: *const Store, layer: u32, comptime suffix: []const u8) !core.safetensors.Tensor {
        var buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.mlp.experts.{s}", .{ layer, suffix });
        return s.ck.get(name);
    }

    /// The byte ranges of expert `e` of `layer` this rank reads, as views into the checkpoint's mapping (no byte is
    /// touched: `pack` reads them through the mapping, the fill re-points them at O_DIRECT copies first).
    /// w13 / its scales: this rank's 2 Il stored rows (one contiguous range); w2 / its scales: the expert's whole rows
    /// (the column cut is the pack's).
    pub const Views = struct { w13: []const u8, s13: []const u8, w2: []const u8, s2: []const u8 };

    pub fn views(s: *const Store, layer: u32, e: usize) !Views {
        const c = s.c;
        const H: usize = c.hidden;
        const I: usize = c.inter;
        const Il = s.inter();
        const fmt = c.formats[layer];
        const per_byte: usize = if (fmt == .nvfp4) 2 else 1;
        const block: usize = if (fmt == .nvfp4) 16 else 32;
        const w13 = try s.tensor(layer, "w13_weight");
        const s13 = try s.tensor(layer, "w13_weight_scale");
        const w2 = try s.tensor(layer, "w2_weight");
        const s2 = try s.tensor(layer, "w2_weight_scale");
        const row13 = H / per_byte;
        const srow13 = H / block;
        const r0 = 2 * Il * s.rank;
        if (w13.bytes.len != c.experts * 2 * I * row13 or s13.bytes.len != c.experts * 2 * I * srow13 or
            w2.bytes.len != c.experts * H * (I / per_byte) or s2.bytes.len != c.experts * H * (I / block)) return error.ExpertBytes;
        return .{
            .w13 = w13.bytes[e * 2 * I * row13 + r0 * row13 ..][0 .. 2 * Il * row13],
            .s13 = s13.bytes[e * 2 * I * srow13 + r0 * srow13 ..][0 .. 2 * Il * srow13],
            .w2 = w2.bytes[e * H * (I / per_byte) ..][0 .. H * (I / per_byte)],
            .s2 = s2.bytes[e * H * (I / block) ..][0 .. H * (I / block)],
        };
    }

    /// Expert `e` of `layer` for this rank, packed into `out` (its `up` / `down` sized by `slotWords`) from the
    /// mapping. Refuses NaN or negative e4m3 block scales (NVFP4), NaN e4m3 codes and 0xFF exponents (MXFP8),
    /// non-finite or non-positive amax.
    pub fn pack(s: *const Store, gpa: std.mem.Allocator, layer: u32, e: usize, out: *Slot) !void {
        try s.packViews(gpa, layer, e, try s.views(layer, e), out);
    }

    /// `pack` from `v` (the mapping's views or their copies). Thread-safe with a thread-safe `gpa`.
    pub fn packViews(s: *const Store, gpa: std.mem.Allocator, layer: u32, e: usize, v: Views, out: *Slot) !void {
        const c = s.c;
        const H: usize = c.hidden;
        const I: usize = c.inter;
        const Il = s.inter();
        const fmt = c.formats[layer];
        out.fmt = fmt;
        const per_byte: usize = if (fmt == .nvfp4) 2 else 1; // codes a byte
        const block: usize = if (fmt == .nvfp4) 16 else 32; // inputs a scale
        // w13: this rank's 2 Il stored rows, split into gate (even) and up (odd) rows
        const row13 = H / per_byte;
        const srow13 = H / block;
        const g_codes = try gpa.alloc(u8, Il * row13);
        defer gpa.free(g_codes);
        const u_codes = try gpa.alloc(u8, Il * row13);
        defer gpa.free(u_codes);
        const g_sc = try gpa.alloc(u8, Il * srow13);
        defer gpa.free(g_sc);
        const u_sc = try gpa.alloc(u8, Il * srow13);
        defer gpa.free(u_sc);
        for (0..Il) |ch| {
            const gr = 2 * ch;
            @memcpy(g_codes[ch * row13 ..][0..row13], v.w13[gr * row13 ..][0..row13]);
            @memcpy(u_codes[ch * row13 ..][0..row13], v.w13[(gr + 1) * row13 ..][0..row13]);
            @memcpy(g_sc[ch * srow13 ..][0..srow13], v.s13[gr * srow13 ..][0..srow13]);
            @memcpy(u_sc[ch * srow13 ..][0..srow13], v.s13[(gr + 1) * srow13 ..][0..srow13]);
        }
        // w2: [H][I / per_byte] codes and [H][I / block] scales, this rank's columns
        const c2 = Il / per_byte;
        const sc2 = Il / block;
        const d_codes = try gpa.alloc(u8, H * c2);
        defer gpa.free(d_codes);
        const d_sc = try gpa.alloc(u8, H * sc2);
        defer gpa.free(d_sc);
        for (0..H) |h| {
            @memcpy(d_codes[h * c2 ..][0..c2], v.w2[h * (I / per_byte) + s.rank * c2 ..][0..c2]);
            @memcpy(d_sc[h * sc2 ..][0..sc2], v.s2[h * (I / block) + s.rank * sc2 ..][0..sc2]);
        }
        const words = slotWords(fmt, H, Il);
        std.debug.assert(out.up.len == words[0] and out.down.len == words[1]);
        switch (fmt) {
            .nvfp4 => {
                for ([_][]const u8{ g_sc, u_sc, d_sc }) |sc| if (badE4m3Scale(sc)) {
                    std.log.err("layer {d} expert {d}: NaN or negative e4m3 block scale (NVFP4)", .{ layer, e });
                    return error.BadScale;
                };
                Nv.packGateUp(out.up, .{ g_codes, g_sc }, .{ u_codes, u_sc }, Il, H);
                Nv.packOne(out.down, d_codes, d_sc, H, Il);
                if (s.globals) |g| {
                    out.g13 = g[layer * c.experts + e][0];
                    out.g2 = g[layer * c.experts + e][1];
                } else {
                    out.g13 = try global(s, layer, "w13", e);
                    out.g2 = try global(s, layer, "w2", e);
                }
            },
            .mxfp8 => {
                for ([_][]const u8{ g_codes, u_codes, d_codes }) |w| if (nanE4m3(w)) {
                    std.log.err("layer {d} expert {d}: NaN e4m3 weight byte (MXFP8)", .{ layer, e });
                    return error.BadWeight;
                };
                for ([_][]const u8{ g_sc, u_sc, d_sc }) |sc| if (std.mem.indexOfScalar(u8, sc, 0xFF) != null) {
                    std.log.err("layer {d} expert {d}: UE8M0 exponent 0xFF (MXFP8)", .{ layer, e });
                    return error.BadScale;
                };
                packMx(out.up, 2, 0, g_codes, g_sc, Il, H);
                packMx(out.up, 2, 1, u_codes, u_sc, Il, H);
                packMx(out.down, 1, 0, d_codes, d_sc, H, Il);
            },
            .exl3 => return error.Exl3Slot, // the EXL3 export's slots are exl3.zig's (packSlot), never this arena's
        }
    }

    /// g = fp32(amax[e]) / fp32(denominator): the reference's one division (weights_quant.dequant_stack).
    fn global(s: *const Store, layer: u32, comptime w: []const u8, e: usize) !f32 {
        const amax = try s.tensor(layer, w ++ "_weight_amax");
        const den = try s.tensor(layer, w ++ "_weight_outer_scale_denominator");
        const a = std.mem.bytesToValue(f32, amax.bytes[e * 4 ..][0..4]);
        const d = std.mem.bytesToValue(f32, den.bytes[0..4]);
        if (d != 2688.0) return error.Denominator;
        if (!std.math.isFinite(a) or a <= 0) return error.BadAmax;
        return a / d;
    }
};

/// An e4m3 scale byte that is NaN (S.1111.111) or negative (sign set, not -0).
fn badE4m3Scale(sc: []const u8) bool {
    for (sc) |b| if (b & 0x7F == 0x7F or (b & 0x80 != 0 and b != 0x80)) return true;
    return false;
}

fn nanE4m3(w: []const u8) bool {
    for (w) |b| if (b & 0x7F == 0x7F) return true;
    return false;
}

/// The reference's mxfp8_lane.pack for one matrix `part` of `parts`: codes [n][k] e4m3 and exponents [n][k/32] into
/// blocks [n/32][k/32][parts][264] int32. A block's 1024 code bytes are ordered (gq, t, j, h, i) for row
/// cb 32 + 8 j + gq and input g 32 + 16 h + 4 t + i; its 32 exponent bytes (t, j, c) for row cb 32 + 8 j + 2 t + c.
pub fn packMx(out: []u32, parts: usize, part: usize, codes: []const u8, exps: []const u8, n: usize, k: usize) void {
    const kg = k / 32;
    std.debug.assert(n % 32 == 0 and k % 32 == 0 and out.len == n / 32 * kg * parts * mx_words);
    const bytes = std.mem.sliceAsBytes(out);
    for (0..n / 32) |cb| for (0..kg) |g| {
        const blk = bytes[((cb * kg + g) * parts + part) * mx_words * 4 ..][0 .. mx_words * 4];
        for (0..8) |gq| for (0..4) |t| for (0..4) |j| for (0..2) |h| for (0..4) |i| {
            const row = cb * 32 + j * 8 + gq;
            const col = g * 32 + h * 16 + t * 4 + i;
            blk[(((gq * 4 + t) * 4 + j) * 2 + h) * 4 + i] = codes[row * k + col];
        };
        for (0..4) |t| for (0..4) |j| for (0..2) |c| {
            const row = cb * 32 + j * 8 + t * 2 + c;
            blk[1024 + (t * 4 + j) * 2 + c] = exps[row * kg + g];
        };
    };
}

test "inkling2 experts: slot sizes for the release at TP2" {
    const nv = slotWords(.nvfp4, 4608, 1152);
    try std.testing.expectEqual(@as(usize, 36 * 144 * 144 * 2), nv[0]); // 36 x 144 blocks, two parts, 144 words
    const mx = slotWords(.mxfp8, 4608, 1152);
    try std.testing.expectEqual(@as(usize, 144 * 36 * 264), mx[1]);
}

test "inkling2 experts: the MXFP8 pack places one code byte and one exponent where pack8 does" {
    const n = 32;
    const k = 64;
    var codes: [n * k]u8 = undefined;
    for (&codes, 0..) |*x, i| x.* = @truncate(i *% 31 +% 7);
    var exps: [n * k / 32]u8 = undefined;
    for (&exps, 0..) |*x, i| x.* = @truncate(i * 3 + 1);
    var out: [1 * 2 * mx_words]u32 = undefined;
    packMx(&out, 1, 0, &codes, &exps, n, k);
    const b = std.mem.sliceAsBytes(&out);
    // block g = 1: row 8 j + gq with j 2, gq 3 (row 19), input 32 + 16 + 4 t + i with t 1, i 2 (input 54)
    try std.testing.expectEqual(codes[19 * k + 54], b[1056 + (((3 * 4 + 1) * 4 + 2) * 2 + 1) * 4 + 2]);
    // its exponent (t 1, j 2, c 1): row 16 + 2 + 1 = 19, group 1
    try std.testing.expectEqual(exps[19 * 2 + 1], b[1056 + 1024 + (1 * 4 + 2) * 2 + 1]);
}

test "inkling2 experts: the byte checks refuse what the kernels cannot decode (NaN e4m3 codes and scales, UE8M0 0xFF)" {
    // MXFP8's decode moves e4m3 fields into bf16 (exact for every finite code) and reads 0x7F / 0xFF as 480 / -480:
    // a NaN code must be refused before packing, as must a negative or NaN NVFP4 block scale
    try std.testing.expect(nanE4m3(&.{ 0x00, 0x7f }));
    try std.testing.expect(nanE4m3(&.{0xff}));
    try std.testing.expect(!nanE4m3(&.{ 0x7e, 0xfe, 0x80, 0x00 }));
    try std.testing.expect(badE4m3Scale(&.{0x7f}));
    try std.testing.expect(badE4m3Scale(&.{0x81}));
    try std.testing.expect(!badE4m3Scale(&.{ 0x80, 0x00, 0x38, 0x7e }));
}
