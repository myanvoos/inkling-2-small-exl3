//! tf-inkling2-exl3-check VIEW LAYER EXPERT RANK WORLD MODE X OUT: one expert of the EXL3 export through the forward's
//! own kernels on this GPU, every stage written to OUT for the CPU reference (out/exl3/ref/check_kernels.py):
//!   the rank's slot (exl3.loadSlot) on the device; the two matrices' decoded W_q (ink2_exl3_dequant: wq13.f16,
//!   wq2.f16); X (bf16 [R][H] from the file X) as R pairs of one plan item (members 0..R-1): ink2_exl3_act (MODE 0
//!   none, 1 NVFP4, 2 MXFP8: xq13.f16), ink2_exl3_rot (xr13.f16), ink2_exl3_gemm_g1 and _g2 (z13_g1.f32, z13_g2.f32:
//!   a pair's bits must not depend on the form), ink2_exl3_gateup_epi (act.bf16), the w2 activation under each
//!   row's own amax (xq2.f16), its rotation (xr2.f16), the down GEMM g1 / g2 (y_g1.f32, y_g2.f32: raw), and the down
//!   epilogue on the g1 sums (y.f32); then the engine's forms: tf_exl3_grouped_2 on both matrices (z13_core.f32,
//!   y_core.f32) and the wide-chunk kernels on its prompt hooks (act_p.bf16, y_p.f32). One GPU, a few MB: no TP group.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");

const usage = "usage: tf-inkling2-exl3-check VIEW LAYER EXPERT RANK WORLD MODE X_BF16_FILE OUT_DIR\n";

fn write(io: std.Io, out: []const u8, name: []const u8, bytes: []const u8, gpa: std.mem.Allocator) !void {
    const p = try std.fs.path.join(gpa, &.{ out, name });
    defer gpa.free(p);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = bytes });
}

fn dl(gpa: std.mem.Allocator, b: cuda.DeviceBuffer, n: usize) ![]u8 {
    const host = try gpa.alloc(u8, n);
    try b.download(0, host);
    return host;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 9) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const dir = args[1];
    const layer = try std.fmt.parseInt(usize, args[2], 10);
    const e = try std.fmt.parseInt(usize, args[3], 10);
    const rank = try std.fmt.parseInt(usize, args[4], 10);
    const world = try std.fmt.parseInt(usize, args[5], 10);
    const mode = try std.fmt.parseInt(u8, args[6], 10);
    const out = args[8];
    if (mode > 2 or rank >= world) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const c = try ink.config.read(a, io, dir);
    if (!c.exl3) {
        std.debug.print("{s} is not a view of the EXL3 export (no checkpoint.json of {s})\n", .{ dir, ink.exl3.format_id });
        return 2;
    }
    const H: usize = c.hidden;
    const I: usize = c.inter;
    const Il = I / world;
    try ink.exl3.checkSplit(H, I, world);
    var ix = try ink.exl3.Index.open(a, io, dir, &c);
    defer ix.deinit();
    const xb = try std.Io.Dir.cwd().readFileAlloc(io, args[7], a, .limited(1 << 30));
    if (xb.len == 0 or xb.len % (2 * H) != 0) return error.BadInput;
    const R = xb.len / (2 * H);
    if (R > 4096) return error.BadInput;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try ctx.makeCurrent();
    const d = &driver;
    var k = try ink.kern.Kernels.load(d, @intCast(@max(try ctx.attribute(.multiprocessor_count), 1)));
    defer k.unload();
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();

    // the slot and its one-entry tables
    const slot_host = try a.alloc(u8, ink.exl3.parts(H, Il, 3).total);
    const bits = try ink.exl3.loadSlot(a, dir, &ix, layer, e, H, I, rank, world, slot_host);
    const total = ink.exl3.parts(H, Il, bits).total;
    var slot = try cuda.DeviceBuffer.fromHost(d, slot_host[0..total]);
    defer slot.free();
    var tptr = try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&slot.ptr));
    defer tptr.free();
    const kk: i32 = bits;
    var tk = try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&kk));
    defer tk.free();
    // the engine's grouped kernel's tables for the one slot: w13's and w2's trellis, half-bits
    const t13_h: u64 = slot.ptr;
    const t2_h: u64 = slot.ptr + ink.exl3.parts(H, Il, bits).w2;
    const k2_h: i32 = 2 * @as(i32, bits);
    var t13 = try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&t13_h));
    defer t13.free();
    var t2 = try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&t2_h));
    defer t2.free();
    var tk2 = try cuda.DeviceBuffer.fromHost(d, std.mem.asBytes(&k2_h));
    defer tk2.free();
    const ex: ink.kern.Kernels.Exl3 = .{ .slot_ptr = tptr.ptr, .slot_k = tk.ptr, .nslots = 1, .hidden = H, .inter = Il, .cg = c.gate_cap, .cu = c.up_cap, .tp13 = t13.ptr, .tp2 = t2.ptr, .k2 = tk2.ptr };

    // one plan item: slot 0, members 0..R-1 (pair p reads row p)
    const items_h = [3]i32{ 0, 0, @intCast(R) };
    var items = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&items_h));
    defer items.free();
    const counts_h = [4]i32{ 1, 0, 0, 0 };
    var counts = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&counts_h));
    defer counts.free();
    const mem_h = try a.alloc(i32, R);
    for (mem_h, 0..) |*v, i| v.* = @intCast(i);
    var members = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(mem_h));
    defer members.free();
    const plan: cuda.grouped.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = 0, .hist = 0 };

    var x = try cuda.DeviceBuffer.fromHost(d, xb);
    defer x.free();
    var xq13 = try cuda.DeviceBuffer.alloc(d, R * H * 2);
    defer xq13.free();
    var xr13 = try cuda.DeviceBuffer.alloc(d, R * H * 2);
    defer xr13.free();
    var z1 = try cuda.DeviceBuffer.alloc(d, R * 2 * Il * 4);
    defer z1.free();
    var z2 = try cuda.DeviceBuffer.alloc(d, R * 2 * Il * 4);
    defer z2.free();
    var act = try cuda.DeviceBuffer.alloc(d, R * Il * 2);
    defer act.free();
    var xq2 = try cuda.DeviceBuffer.alloc(d, R * Il * 2);
    defer xq2.free();
    var xr2 = try cuda.DeviceBuffer.alloc(d, R * Il * 2);
    defer xr2.free();
    var y1 = try cuda.DeviceBuffer.alloc(d, R * H * 4);
    defer y1.free();
    var y2 = try cuda.DeviceBuffer.alloc(d, R * H * 4);
    defer y2.free();
    var wq13 = try cuda.DeviceBuffer.alloc(d, H * 2 * Il * 2);
    defer wq13.free();
    var wq2 = try cuda.DeviceBuffer.alloc(d, Il * H * 2);
    defer wq2.free();
    for ([_]*cuda.DeviceBuffer{ &z1, &z2, &y1, &y2 }) |b| try b.fill8(0, s.handle);

    try k.exl3Dequant(s, slot.ptr, bits, false, H, Il, wq13.ptr);
    try k.exl3Dequant(s, slot.ptr, bits, true, H, Il, wq2.ptr);
    try k.exl3Act(s, x.ptr, H, R, H, mode, 0, 1, 0, xq13.ptr, H);
    try k.exl3Rot(s, ex, xq13.ptr, H, 0, plan, 1, false, xr13.ptr, H);
    try k.exl3Gemm(s, ex, false, false, xr13.ptr, plan, 1, z1.ptr, 2 * Il, 0, 2 * Il, 1, 0);
    try k.exl3Gemm(s, ex, false, true, xr13.ptr, plan, 1, z2.ptr, 2 * Il, 0, 2 * Il, 1, 0);
    try k.exl3GateUpEpi(s, ex, z1.ptr, 2 * Il, plan, 1, act.ptr);
    try k.exl3Act(s, act.ptr, Il, R, Il, mode, 0, 1, 0, xq2.ptr, Il);
    try k.exl3Rot(s, ex, xq2.ptr, Il, 0, plan, 1, true, xr2.ptr, Il);
    try k.exl3Gemm(s, ex, true, false, xr2.ptr, plan, 1, y1.ptr, H, 0, H, 1, 0);
    try k.exl3Gemm(s, ex, true, true, xr2.ptr, plan, 1, y2.ptr, H, 0, H, 1, 0);
    // y.f32: the down epilogue in place on a copy of the g1 sums
    var y = try cuda.DeviceBuffer.alloc(d, R * H * 4);
    defer y.free();
    try y.copyFrom(0, y1.ptr, R * H * 4, s.handle);
    try k.exl3DownEpi(s, ex, y.ptr, H, 0, H, plan, 1, y.ptr, 1, 0);
    // the engine's forms: tf_exl3_grouped_2 on both matrices (z13_core, y_core: g1's bits), and the wide-chunk kernels
    // on the engine's prompt hooks (act_p, y_p: the GEMM then the epilogue's bits)
    var z3 = try cuda.DeviceBuffer.alloc(d, R * 2 * Il * 4);
    defer z3.free();
    var y3 = try cuda.DeviceBuffer.alloc(d, R * H * 4);
    defer y3.free();
    var act_p = try cuda.DeviceBuffer.alloc(d, R * Il * 2);
    defer act_p.free();
    var y_p = try cuda.DeviceBuffer.alloc(d, R * H * 4);
    defer y_p.free();
    for ([_]*cuda.DeviceBuffer{ &z3, &y3, &act_p, &y_p }) |b| try b.fill8(0xff, s.handle);
    try k.exl3Grouped(s, ex, false, xr13.ptr, plan, 1, ink.kern.decode_tile, R, z3.ptr, 1);
    try k.exl3Grouped(s, ex, true, xr2.ptr, plan, 1, ink.kern.decode_tile, R, y3.ptr, 1);
    try k.exl3PromptGu(s, ex, k.exl3_cfg, xr13.ptr, plan, 1, ink.kern.prompt_tile, act_p.ptr);
    try k.exl3PromptDown(s, ex, k.exl3_cfg, xr2.ptr, plan, 1, ink.kern.prompt_tile, y_p.ptr, H, 0, H, 1);
    try s.synchronize();

    try std.Io.Dir.cwd().createDirPath(io, out);
    const files = [_]struct { name: []const u8, b: cuda.DeviceBuffer, n: usize }{
        .{ .name = "wq13.f16", .b = wq13, .n = H * 2 * Il * 2 }, .{ .name = "wq2.f16", .b = wq2, .n = Il * H * 2 },
        .{ .name = "xq13.f16", .b = xq13, .n = R * H * 2 },       .{ .name = "xr13.f16", .b = xr13, .n = R * H * 2 },
        .{ .name = "z13_g1.f32", .b = z1, .n = R * 2 * Il * 4 },  .{ .name = "z13_g2.f32", .b = z2, .n = R * 2 * Il * 4 },
        .{ .name = "act.bf16", .b = act, .n = R * Il * 2 },       .{ .name = "xq2.f16", .b = xq2, .n = R * Il * 2 },
        .{ .name = "xr2.f16", .b = xr2, .n = R * Il * 2 },        .{ .name = "y_g1.f32", .b = y1, .n = R * H * 4 },
        .{ .name = "y_g2.f32", .b = y2, .n = R * H * 4 },         .{ .name = "y.f32", .b = y, .n = R * H * 4 },
        .{ .name = "z13_core.f32", .b = z3, .n = R * 2 * Il * 4 }, .{ .name = "y_core.f32", .b = y3, .n = R * H * 4 },
        .{ .name = "act_p.bf16", .b = act_p, .n = R * Il * 2 },    .{ .name = "y_p.f32", .b = y_p, .n = R * H * 4 },
    };
    for (files) |f| try write(io, out, f.name, try dl(a, f.b, f.n), a);
    const meta = try std.fmt.allocPrint(a, "{{\"layer\": {d}, \"expert\": {d}, \"bits\": {d}, \"rows\": {d}, \"rank\": {d}, \"world\": {d}, \"mode\": {d}, \"hidden\": {d}, \"inter\": {d}}}\n", .{ layer, e, bits, R, rank, world, mode, H, I });
    try write(io, out, "meta.json", meta, a);
    std.debug.print("exl3-check: layer {d} expert {d} ({d} bits), rank {d}/{d}, mode {d}, {d} rows: stages written to {s}\n", .{ layer, e, bits, rank, world, mode, R, out });
    return 0;
}
