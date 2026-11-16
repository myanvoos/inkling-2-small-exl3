//! Inkling-2's routed experts under the contract, launched exactly as the family launches them (inkling2 kern.zig):
//!   NVFP4 (inkling2_nvfp4.cu, derived from the core nvfp4_experts.cu):
//!     decode gate-up nvfp4_expert_kernel<2, 4, 4>, the softcapped SwiGLU: float64 reference within the bound
//!     prompt gate-up ink2_nvfp4_fact_gate_up: replaces the decode gate-up bit for bit
//!     decode down <1, 0, 4> and prompt ink2_nvfp4_fact_down_f32: each replaces the CORE nvfp4_expert_kernel<1, 0, 4>
//!       (GLM's, untouched) bit for bit, so the derived file's down path is the served one
//!   MXFP8 (inkling2_mxfp8.cu, the overlay's mxfp8_grouped.cu):
//!     ink2_mx8_gate_up_r1 (softcapped SwiGLU, sums rounded to bf16 first) and ink2_mx8_down_f32_r1: float64 reference
//!     the prompt entries _r4: each replaces its _r1 bit for bit
//!     the wide prompt chunks' entries ink2_mx8_wide_gate_up / _down_f32 (p3-expert: 128-pair plan items, weights
//!       staged in shared memory for the CTA's warps): each replaces its _r1 bit for bit
//!   controls: the core plain-SwiGLU gate-up <2, 2, 4> posing as Inkling's, and the MXFP8 gate-up with its caps
//!     swapped (gate 25, up 4): each must fail gate 3.
//! The softcap: 4 tanh(g / 4) sigmoid(g) 25 tanh(u / 25) on the bf16-rounded scaled sums (cg 4, cu 25, the
//! reference's Inkling2SoftcappedSwiGLU), output rounded to bf16.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const fx = cuda.experts.Nvfp4Experts;
const grouped = cuda.grouped;
const kern = ink.kern;

pub const Kind = enum { nv_up, nv_fact_up, nv_down, nv_fact_down, mx_up, mx_up_r4, mx_down, mx_down_r4, mx_up_wide, mx_down_wide, nv_control, mx_control };

const experts = 4;
const slots = 2; // pairs a token row, both routed
const cg: f32 = 4;
const cu: f32 = 25;

fn isDown(k: Kind) bool {
    return switch (k) {
        .nv_down, .nv_fact_down, .mx_down, .mx_down_r4, .mx_down_wide => true,
        else => false,
    };
}

fn isMx(k: Kind) bool {
    return switch (k) {
        .mx_up, .mx_up_r4, .mx_down, .mx_down_r4, .mx_up_wide, .mx_down_wide, .mx_control => true,
        else => false,
    };
}

/// The prompt form (64-pair plan items) rather than decode (16).
fn isPrompt(k: Kind) bool {
    return switch (k) {
        .nv_fact_up, .nv_fact_down, .mx_up_r4, .mx_down_r4, .mx_up_wide, .mx_down_wide => true,
        else => false,
    };
}

/// The wide prompt chunks' MXFP8 form (128-pair plan items, kern.wide_tile).
fn isWide(k: Kind) bool {
    return k == .mx_up_wide or k == .mx_down_wide;
}

fn depth(s: c.Shape) usize {
    return s.k / 16 + 16;
}

const rule = "as the kernel each replaces gives them: NaN and infinite global scales, a NaN block scale, NaN and infinite inputs";
const why = "a bf16 sum that collapses moves the output by about itself, and the epilogue's two bf16 roundings already allow 2 u of it";

// Inkling-2-Small at TP=2 (hidden 4,608, a rank's 1,152 channels) and TP=1 (2,304)
const up_served = [_]c.Shape{ .{ .rows = 1, .n = 1152, .k = 4608 }, .{ .rows = 4, .n = 1152, .k = 4608 }, .{ .rows = 16, .n = 1152, .k = 4608 }, .{ .rows = 4, .n = 2304, .k = 4608 } };
const down_served = [_]c.Shape{ .{ .rows = 1, .n = 4608, .k = 1152 }, .{ .rows = 4, .n = 4608, .k = 1152 }, .{ .rows = 16, .n = 4608, .k = 1152 }, .{ .rows = 4, .n = 4608, .k = 2304 } };
const up_prompt = [_]c.Shape{ .{ .rows = 64, .n = 1152, .k = 4608 }, .{ .rows = 1, .n = 1152, .k = 4608 }, .{ .rows = 33, .n = 1152, .k = 4608 } };
const down_prompt = [_]c.Shape{ .{ .rows = 64, .n = 4608, .k = 1152 }, .{ .rows = 1, .n = 4608, .k = 1152 }, .{ .rows = 33, .n = 4608, .k = 1152 } };
// the wide form: 160 rows x 2 slots over 4 experts puts items past 64 pairs (two warp rows) as well as short ones
const up_wide = [_]c.Shape{ .{ .rows = 160, .n = 1152, .k = 4608 }, .{ .rows = 1, .n = 1152, .k = 4608 }, .{ .rows = 33, .n = 1152, .k = 4608 } };
const down_wide = [_]c.Shape{ .{ .rows = 160, .n = 4608, .k = 1152 }, .{ .rows = 1, .n = 4608, .k = 1152 }, .{ .rows = 33, .n = 4608, .k = 1152 } };

pub fn spec(kind: Kind) c.Spec {
    const unit = [3]c.Tile{ .{ .size = 16 / slots }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 32, .step = 32, .min = 32 } };
    const wide = [3]c.Tile{ .{ .size = 64 / slots }, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = 64, .step = 32, .min = 32 } };
    const wide128 = [3]c.Tile{ .{ .size = 128 / slots }, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = 64, .step = 32, .min = 32 } };
    return switch (kind) {
        .nv_up => .{ .name = "cuda inkling2 nvfp4 decode gate-up nvfp4_expert_kernel<2, 4, 4> (softcapped SwiGLU)", .accum = .f32, .depth = depth, .ops = 6, .mids = 2, .out = .bf16, .tiles = unit, .served = &up_served, .sums = .{ .bf16 = 30, .why = why } },
        .nv_fact_up => .{ .name = "cuda inkling2 nvfp4 prompt gate-up ink2_nvfp4_fact_gate_up (decode sums)", .accum = .f32, .depth = depth, .ops = 6, .mids = 2, .out = .bf16, .replaces = "inkling2 nvfp4_expert_kernel<2, 4, 4>", .nan = .{ .stated = rule }, .tiles = wide, .served = &up_prompt, .sums = .{ .bf16 = 30, .why = why } },
        .nv_down => .{ .name = "cuda inkling2 nvfp4 decode down nvfp4_expert_kernel<1, 0, 4> (the derived file's)", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "core nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = unit, .served = &down_served },
        .nv_fact_down => .{ .name = "cuda inkling2 nvfp4 prompt down ink2_nvfp4_fact_down_f32 (decode sums)", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "core nvfp4_expert_kernel<1, 0, 4>", .nan = .{ .stated = rule }, .tiles = wide, .served = &down_prompt },
        .mx_up => .{ .name = "cuda inkling2 mxfp8 decode gate-up ink2_mx8_gate_up_r1 (softcapped SwiGLU)", .accum = .f32, .depth = depth, .ops = 6, .mids = 2, .out = .bf16, .tiles = unit, .served = &up_served, .sums = .{ .bf16 = 30, .why = why } },
        .mx_up_r4 => .{ .name = "cuda inkling2 mxfp8 prompt gate-up ink2_mx8_gate_up_r4", .accum = .f32, .depth = depth, .ops = 6, .mids = 2, .out = .bf16, .replaces = "ink2_mx8_gate_up_r1", .nan = .{ .stated = rule }, .tiles = wide, .served = &up_prompt, .sums = .{ .bf16 = 30, .why = why } },
        .mx_down => .{ .name = "cuda inkling2 mxfp8 decode down ink2_mx8_down_f32_r1", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .tiles = unit, .served = &down_served },
        .mx_down_r4 => .{ .name = "cuda inkling2 mxfp8 prompt down ink2_mx8_down_f32_r4", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "ink2_mx8_down_f32_r1", .nan = .{ .stated = rule }, .tiles = wide, .served = &down_prompt },
        .mx_up_wide => .{ .name = "cuda inkling2 mxfp8 wide prompt gate-up ink2_mx8_wide_gate_up", .accum = .f32, .depth = depth, .ops = 6, .mids = 2, .out = .bf16, .replaces = "ink2_mx8_gate_up_r1", .nan = .{ .stated = rule }, .tiles = wide128, .served = &up_wide, .sums = .{ .bf16 = 30, .why = why } },
        .mx_down_wide => .{ .name = "cuda inkling2 mxfp8 wide prompt down ink2_mx8_wide_down_f32", .accum = .f32, .depth = depth, .ops = 1, .out = .f32, .replaces = "ink2_mx8_down_f32_r1", .nan = .{ .stated = rule }, .tiles = wide128, .served = &down_wide },
        .nv_control => .{ .name = "control: inkling2 gate-up replaced by the core plain-SwiGLU gate-up nvfp4_expert_kernel<2, 2, 4> posing as Inkling's softcapped one", .accum = .f32, .depth = depth, .ops = 6, .mids = 2, .out = .bf16, .tiles = unit, .served = up_served[1..2] },
        .mx_control => .{ .name = "control: inkling2 ink2_mx8_gate_up_r1 with its caps swapped (gate 25, up 4)", .accum = .f32, .depth = depth, .ops = 6, .mids = 2, .out = .bf16, .tiles = unit, .served = up_served[1..2] },
    };
}

fn replacing(k: Kind) bool {
    return spec(k).replaces != null;
}

pub const Entry = struct {
    rig: rig_.Rig,
    kind: Kind,
    ik: *const kern.Kernels,
    core4: *const fx, // the core NVFP4 kernels (GLM's), the down baseline and the control
    router: grouped.Router,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    codes: []u8 = &.{}, // NVFP4 [E][parts][n][k/2] nibbles; MXFP8 [E][parts][n][k] e4m3
    scales: []u8 = &.{}, // NVFP4 [E][parts][n][k/16] e4m3; MXFP8 [E][parts][n][k/32] UE8M0
    globals: [experts * 2]f32 = undefined, // NVFP4 [E][parts]
    picks: []i32 = &.{},
    allow_extra: []f64 = &.{},
    wd: u64 = 0,
    sd: u64 = 0,
    xd: u64 = 0,
    plan: grouped.Plan = undefined, // 16-pair items
    wide: grouped.Plan = undefined, // 64-pair items
    wide128: grouped.Plan = undefined, // 128-pair items (the wide form)
    out: u64 = 0,
    base: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Entry) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Entry) void {
        const a = self.rig.gpu.gpa;
        a.free(self.x);
        a.free(self.codes);
        a.free(self.scales);
        a.free(self.picks);
        a.free(self.allow_extra);
        self.x = &.{};
        self.codes = &.{};
        self.scales = &.{};
        self.picks = &.{};
        self.allow_extra = &.{};
    }

    const vt_replacing = g.hooks(Entry);
    const vt_bounded = blk: {
        var v = g.hooks(Entry);
        v.check = null;
        v.replaced = null;
        break :blk v;
    };

    pub fn kernel(self: *Entry) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = if (replacing(self.kind)) &vt_replacing else &vt_bounded, .control = switch (self.kind) {
            .nv_control, .mx_control => .shapes,
            else => null,
        } };
    }

    fn parts(self: *const Entry) usize {
        return if (isDown(self.kind)) 1 else 2;
    }
    fn pairs(self: *const Entry) usize {
        return self.shape.rows * slots;
    }
    fn xRows(self: *const Entry) usize {
        return if (!isDown(self.kind)) self.shape.rows else self.pairs();
    }
    fn row(self: *const Entry, pair: usize) usize {
        return if (!isDown(self.kind)) pair / slots else pair;
    }
    pub fn count(self: *const Entry) usize {
        return self.pairs() * self.shape.n;
    }
    fn outBytes(self: *const Entry) usize {
        return self.count() * c.size(spec(self.kind).out);
    }
    fn codesPer(self: *const Entry) usize {
        return if (isMx(self.kind)) self.shape.n * self.shape.k else self.shape.n * self.shape.k / 2;
    }
    fn scalesPer(self: *const Entry) usize {
        return self.shape.n * self.shape.k / (if (isMx(self.kind)) @as(usize, 32) else 16);
    }
    fn codesOf(self: *const Entry, e: usize, part: usize) []u8 {
        return self.codes[(e * self.parts() + part) * self.codesPer() ..][0..self.codesPer()];
    }
    fn scalesOf(self: *const Entry, e: usize, part: usize) []u8 {
        return self.scales[(e * self.parts() + part) * self.scalesPer() ..][0..self.scalesPer()];
    }

    fn route(self: *Entry, ids: u64, tile: usize, zeros: []const u8) !grouped.Plan {
        const blocks = (self.pairs() + 1023) / 1024;
        const p: grouped.Plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. grouped.maxItems(self.pairs(), experts, tile) * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(blocks * experts * 4) };
        try self.router.route(self.rig.s, ids, self.pairs(), experts, tile, p);
        return p;
    }

    pub fn prepare(self: *Entry, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const n = shape.n;
        const k = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(u16, self.xRows() * k);
        self.codes = try a.alloc(u8, experts * self.parts() * self.codesPer());
        self.scales = try a.alloc(u8, experts * self.parts() * self.scalesPer());
        self.picks = try a.alloc(i32, self.pairs());
        self.allow_extra = try a.alloc(f64, self.count());
        if (isMx(self.kind)) {
            // finite e4m3 codes (not S.1111.111), exponents near 2^-6 so the softcap's curve is exercised
            for (self.codes) |*b| b.* = blk: {
                var v = rng.int(u8);
                while (v & 0x7f == 0x7f) v = rng.int(u8);
                break :blk v;
            };
            for (self.scales) |*s| s.* = 116 + rng.uintLessThan(u8, 6);
        } else {
            rng.bytes(self.codes);
            for (self.scales) |*s| s.* = 0x20 + rng.uintLessThan(u8, 0x28);
        }
        for (&self.globals) |*v| v.* = 0.005 + rng.float(f32) * 0.03;
        for (self.x) |*v| v.* = c.toBf16((rng.float(f32) - 0.5) * 2);
        for (self.picks, 0..) |*p, i| p.* = @intCast(if (i < experts) i else rng.uintLessThan(usize, experts));
        switch (fill) {
            .random => {},
            .nonfinite => {
                // MXFP8: NaN codes (S.1111.111) and 0xFF exponents never reach the kernel -- the loader refuses them
                // by name (experts.zig nanE4m3 / BadScale; its host test) and the kernel's exact bit-move decode
                // reads 0x7F as 480 -- so only what can reach it is planted: NaN and infinite inputs
                if (!isMx(self.kind)) {
                    self.globals[1 * self.parts()] = std.math.nan(f32);
                    self.globals[2 * self.parts()] = std.math.inf(f32);
                    self.scales[0] = 0x7f;
                }
                self.x[(self.xRows() - 1) * k] = c.toBf16(std.math.nan(f32));
                if (self.xRows() > 1) self.x[(self.xRows() - 2) * k + 1] = c.toBf16(std.math.inf(f32));
            },
            .stress => self.stress(rng),
        }
        self.xd = try self.rig.input(std.mem.sliceAsBytes(self.x));
        // pack as the family's loader does: NVFP4 the core slot layout, MXFP8 experts.packMx
        if (isMx(self.kind)) {
            const per = (n / 32) * (k / 32) * self.parts() * ink.experts.mx_words;
            const words = try a.alloc(u32, experts * per);
            defer a.free(words);
            for (0..experts) |e| for (0..self.parts()) |part| ink.experts.packMx(words[e * per ..][0..per], self.parts(), part, self.codesOf(e, part), self.scalesOf(e, part), n, k);
            self.wd = try self.rig.input(std.mem.sliceAsBytes(words));
        } else {
            const per = fx.bytes(1, n, k, self.parts()) / 4;
            const words = try a.alloc(u32, experts * per);
            defer a.free(words);
            for (0..experts) |e| {
                const o = words[e * per ..][0..per];
                if (self.parts() == 2) fx.packGateUp(o, .{ self.codesOf(e, 0), self.scalesOf(e, 0) }, .{ self.codesOf(e, 1), self.scalesOf(e, 1) }, n, k) else fx.packOne(o, self.codesOf(e, 0), self.scalesOf(e, 0), n, k);
            }
            self.wd = try self.rig.input(std.mem.sliceAsBytes(words));
        }
        self.sd = try self.rig.input(std.mem.sliceAsBytes(self.globals[0 .. experts * self.parts()]));
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.picks));
        const zeros = try a.alloc(u8, @max(self.pairs() * 4, grouped.maxItems(self.pairs(), experts, 16) * 12, 8));
        defer a.free(zeros);
        @memset(zeros, 0);
        self.plan = try self.route(ids, kern.decode_tile, zeros);
        self.wide = try self.route(ids, kern.prompt_tile, zeros);
        self.wide128 = try self.route(ids, kern.wide_tile, zeros);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.outBytes());
        self.base = try self.rig.scratch(self.outBytes());
        return true;
    }

    /// Gate 4: bf16-integer inputs, weights 4 and 6 at scale 1, K's halves mirrored: fp32 sums exact in any order.
    fn stress(self: *Entry, rng: std.Random) void {
        const k = self.shape.k;
        @memset(&self.globals, 1);
        if (isMx(self.kind)) @memset(self.scales, 127) else @memset(self.scales, 0x38);
        for (0..self.xRows()) |r| for (0..k / 2) |i| {
            const v: f32 = @floatFromInt(128 + rng.uintLessThan(u32, 128));
            self.x[r * k + i] = c.toBf16(v);
            self.x[r * k + k / 2 + i] = c.toBf16(if (i == 0) v - 1 else v);
        };
        for (0..experts * self.parts() * self.shape.n) |j| for (0..k / 2) |i| {
            if (isMx(self.kind)) {
                const v: u8 = if (rng.boolean()) 0x48 else 0x4c; // e4m3 4 and 6
                self.codes[j * k + i] = v;
                self.codes[j * k + k / 2 + i] = v | 0x80;
            } else {
                const v: u8 = if (rng.boolean()) 6 else 7; // e2m1 4 and 6
                put(self.codes[j * k / 2 ..], i, v);
                put(self.codes[j * k / 2 ..], k / 2 + i, v | 8);
            }
        };
    }

    fn put(codes: []u8, i: usize, v: u8) void {
        const b = &codes[i / 2];
        b.* = if (i % 2 == 0) (b.* & 0xf0) | v else (b.* & 0x0f) | v << 4;
    }

    fn weight(self: *const Entry, e: usize, part: usize, j: usize, i: usize) f64 {
        const k = self.shape.k;
        if (isMx(self.kind)) {
            const sc = self.scalesOf(e, part)[j * (k / 32) + i / 32];
            return core.quant.fp8.e4m3(self.codesOf(e, part)[j * k + i]) * std.math.pow(f64, 2, @as(f64, @floatFromInt(sc)) - 127);
        }
        const b = self.codesOf(e, part)[j * k / 2 + i / 2];
        const nib: u4 = @truncate(if (i % 2 == 0) b & 0xf else b >> 4);
        return core.quant.nvfp4.e2m1(nib) * core.quant.fp8.e4m3(self.scalesOf(e, part)[j * k / 16 + i / 16]);
    }

    fn terms(self: *const Entry, p: usize, part: usize, j: usize, t: []f64) void {
        const e: usize = @intCast(self.picks[p]);
        const xr = self.x[self.row(p) * self.shape.k ..][0..self.shape.k];
        for (t, 0..) |*o, i| o.* = @as(f64, c.fromBf16(xr[i])) * self.weight(e, part, j, i);
    }

    /// The global scale of pair p's part (NVFP4), 1 for MXFP8 (its block exponents are in the weights).
    fn gl(self: *const Entry, p: usize, part: usize) f64 {
        if (isMx(self.kind)) return 1;
        const e: usize = @intCast(self.picks[p]);
        return self.globals[e * self.parts() + part];
    }

    fn ex(self: *const Entry, kind_cg: f32, kind_cu: f32) kern.Kernels.Experts {
        const up = !isDown(self.kind);
        const n = self.shape.n;
        const k = self.shape.k;
        var x: kern.Kernels.Experts = .{ .fmt = if (isMx(self.kind)) .mxfp8 else .nvfp4, .w = self.wd, .scale = if (isMx(self.kind)) 0 else self.sd, .slots = experts, .hidden = if (up) k else n, .inter = if (up) n else k, .cg = kind_cg, .cu = kind_cu };
        x.mx_ri = switch (self.kind) {
            .mx_up_r4, .mx_down_r4 => 2,
            .mx_up, .mx_down, .mx_control => 0,
            else => null,
        };
        x.wide = isWide(self.kind);
        return x;
    }

    fn launch(self: *Entry, out: u64) !void {
        const up = !isDown(self.kind);
        const prompt = isPrompt(self.kind);
        const plan = if (isWide(self.kind)) self.wide128 else if (prompt) self.wide else self.plan;
        const items = grouped.maxItems(self.pairs(), experts, if (isWide(self.kind)) kern.wide_tile else if (prompt) kern.prompt_tile else kern.decode_tile);
        if (self.kind == .nv_control) {
            const in: cuda.experts.Rows = .{ .x = self.xd, .stride = self.shape.k, .slots = slots };
            try self.core4.run(self.rig.s, .up, in, self.coreLayer(), self.plan, out, items, -1);
        } else {
            const swapped = self.kind == .mx_control;
            try self.ik.experts(self.rig.s, up, self.ex(if (swapped) cu else cg, if (swapped) cg else cu), self.xd, self.shape.k, if (up) slots else 0, plan, items, prompt, out);
        }
        try self.rig.s.synchronize();
    }

    fn coreLayer(self: *const Entry) cuda.experts.Layer {
        const up = !isDown(self.kind);
        return .{ .format = .nvfp4, .up = self.wd, .down = self.wd, .up_scale = self.sd, .down_scale = self.sd, .width = if (up) self.shape.n else self.shape.k, .dims = if (up) self.shape.k else self.shape.n, .experts = experts };
    }

    pub fn run(self: *Entry) !void {
        try self.launch(self.out);
    }

    /// The baseline a replacing kind must equal: Inkling's own decode form, or the core down kernel.
    pub fn replaced(self: *Entry) ![]const u8 {
        const items = grouped.maxItems(self.pairs(), experts, kern.decode_tile);
        switch (self.kind) {
            .nv_down, .nv_fact_down => {
                const in: cuda.experts.Rows = .{ .x = self.xd, .stride = self.shape.k, .slots = 0 };
                try self.core4.run(self.rig.s, .down_f32, in, self.coreLayer(), self.plan, self.base, items, -1);
            },
            .nv_fact_up, .mx_up_r4, .mx_down_r4, .mx_up_wide, .mx_down_wide => {
                const up = !isDown(self.kind);
                var x = self.ex(cg, cu);
                if (isMx(self.kind)) x.mx_ri = 0;
                x.wide = false;
                try self.ik.experts(self.rig.s, up, x, self.xd, self.shape.k, if (up) slots else 0, self.plan, items, false, self.base);
            },
            else => return error.NotAReplacement,
        }
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.outBytes());
    }

    /// Gate 1 of a replacing kind: the baseline's bytes, with some output non-finite (the planted values reach it).
    pub fn check(self: *Entry, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from {s}'s", .{ at, got.len, spec(self.kind).replaces.? });
        const out = spec(self.kind).out;
        for (0..got.len / c.size(out)) |i| if (c.classOf(c.read(out, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    fn sigmoid(v: f64) f64 {
        return 1 / (1 + @exp(-v));
    }

    fn capped(v: f64, cap: f64) f64 {
        return cap * std.math.tanh(v / cap);
    }

    /// The softcapped SwiGLU in float64 on (g, u).
    fn act(gv: f64, uv: f64) f64 {
        return capped(gv, cg) * sigmoid(gv) * capped(uv, cu);
    }

    /// The kernel's epilogue on pair p's sums: scaled, rounded to bf16 for up, then the softcapped SwiGLU.
    fn epilogue(self: *const Entry, p: usize, s0: f64, s1: f64) f64 {
        if (isDown(self.kind)) return s0 * self.gl(p, 0);
        const gv = c.fromBf16(c.toBf16(@floatCast(s0 * self.gl(p, 0))));
        const uv = c.fromBf16(c.toBf16(@floatCast(s1 * self.gl(p, 1))));
        return act(gv, uv);
    }

    pub fn reference(self: *Entry, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        const n = self.shape.n;
        for (0..self.pairs()) |p| for (0..n) |j| {
            const o = p * n + j;
            var sum = [2]f64{ 0, 0 };
            var abs = [2]f64{ 0, 0 };
            for (0..self.parts()) |part| {
                self.terms(p, part, j, t);
                for (t) |v| {
                    sum[part] += v;
                    abs[part] += @abs(v);
                }
            }
            if (isDown(self.kind)) {
                want[o] = sum[0] * self.gl(p, 0);
                mag[o] = @abs(self.gl(p, 0)) * abs[0];
                self.allow_extra[o] = 0;
                continue;
            }
            const gs = sum[0] * self.gl(p, 0);
            const us = sum[1] * self.gl(p, 1);
            want[o] = act(gs, us);
            // d/dg of cg tanh(g / cg) sigmoid(g) is at most 1.1, d/du of cu tanh(u / cu) at most 1
            mag[o] = 1.1 * @abs(capped(us, cu)) * @abs(self.gl(p, 0)) * abs[0] + @abs(capped(gs, cg) * sigmoid(gs)) * @abs(self.gl(p, 1)) * abs[1];
            // the kernel rounds g and u to bf16 before the epilogue: each moves the output by its slope's share
            self.allow_extra[o] = 0x1p-8 * (1.1 * @abs(gs) * @abs(capped(us, cu)) + @abs(capped(gs, cg) * sigmoid(gs)) * @abs(us)) * 1.01;
        };
    }

    pub fn bounds(self: *Entry, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.allow_extra) |*l, x| l.* += x;
    }

    pub fn simulate(self: *Entry, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        const n = self.shape.n;
        for (0..self.pairs()) |p| for (0..n) |j| {
            var s = [2]f64{ 0, 0 };
            for (0..self.parts()) |part| {
                self.terms(p, part, j, t);
                s[part] = contract.rows.sumIn(acc, order, t);
            }
            out[p * n + j] = self.epilogue(p, s[0], s[1]);
        };
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    ik: kern.Kernels,
    core4: fx,
    entries: [kinds.len]Entry,
    const kinds = std.enums.values(Kind);

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, stream: cuda.Stream, sms: usize, router: grouped.Router, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.a = a;
        self.ik = try kern.Kernels.load(gpu.d, sms);
        errdefer self.ik.unload();
        self.core4 = try fx.loadWarps(gpu.d, sms, 4);
        errdefer self.core4.unload();
        for (&self.entries, kinds) |*x, kd| x.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .kind = kd, .ik = &self.ik, .core4 = &self.core4, .router = router };
        errdefer for (&self.entries) |*x| x.deinit();
        for (&self.entries) |*x| try kernels.append(a, x.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.entries) |*x| x.deinit();
        self.core4.unload();
        self.ik.unload();
        self.a.destroy(self);
    }
};
