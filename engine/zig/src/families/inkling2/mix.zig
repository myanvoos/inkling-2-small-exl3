//! x-mixed: the mixed expert store's knobs and the native bank's sizing (TF_INKLING2_MIX=1;
//! out/prefill/p5/x-mixed/DESIGN.md). Every expert stays resident as its EXL3 trellis (exl3.Store); the arena is a native
//! bank of the release's own NVFP4 / MXFP8 bytes, sized after the EXL3 slots from what MemAvailable leaves over the floor,
//! the bank margin and what is still to be made (the prompt scratch, the reuse slots, the loader's stages). The bank is
//! elastic: a growing context or a low MemAvailable takes slots back (adapt.zig's give-back), which only moves picks to
//! EXL3, so the bank may take everything above the floor and the margin, including the lazy context's deferred memory.
const std = @import("std");
const model_mod = @import("model.zig");

const Model = model_mod.Model;
const Arena = model_mod.Arena;

pub const gib: f64 = 1024.0 * 1024.0 * 1024.0;

/// The bank's VMM chunks cost this much over its slots' bytes (p4-mem: a 91.21 GiB arena took 92.54-92.61 GiB of
/// MemAvailable).
pub const vmm_overhead: f64 = 1.015;

/// TF_INKLING2_MIX_* (read once; every rank must run the same values: the plans and the agreements assume it).
pub const Opts = struct {
    /// TF_INKLING2_MIX_BANK_GIB: the bank's slot bytes at most (unset: what the room allows).
    bank_gib: ?f64 = null,
    /// TF_INKLING2_MIX_MARGIN_GIB: MemAvailable left over the floor once the bank is made (and what is made after it).
    margin_gib: f64 = 2,
    /// TF_INKLING2_MIX_MX_SHARE: the MXFP8 pool's share of the bank's bytes (unset: TF_INKLING2_MIX_MX_PRIORITY's, else the
    /// same experts a layer as NVFP4).
    mx_share: ?f64 = null,
    /// TF_INKLING2_MIX_MX_PRIORITY=w: the MXFP8 layers (the release's 40-47: their EXL3 copies drop 8 to 3 bits) hold w
    /// times the experts a layer the NVFP4 layers hold, in the same bytes (unset or 1: equal). The two pools never compete
    /// for a slot (an expert fits only its format's), so a priority is a share of the bank's bytes: the boot fill, the
    /// admission, the give-back (k experts a layer from each pool) and the reclaim (each pool back to its boot slots) all
    /// keep it.
    mx_priority: ?f64 = null,
    /// TF_INKLING2_MIX_ADMIT=0: the bank frozen (its boot fill; deterministic measurements).
    admit: bool = true,
    /// xa (TF_INKLING2_MIX_ADMIT=between): admission only between requests. The votes run as in live admission (the
    /// prompt's end, every ADMIT_EVERY rounds), but a batch is planned and started only at a request's end, and the next
    /// request's begin lands it first: no expert loads while a request runs.
    between: bool = false,
    /// xa (TF_INKLING2_MIX_LAND_WAIT=1, a gate): every batch landed before the next forward (its start is followed by a
    /// settle), so the swap points are the admission rounds themselves, whatever the loader's speed: runs are
    /// reproducible and two builds' admissions compare bitwise.
    land_wait: bool = false,
    /// xa (TF_INKLING2_MIX_RECLAIM=sync): a request's begin maps the bank's given-back slots back with x1's synchronous
    /// load (the old behaviour: the gate's reference). Default (async): the map-back is a background batch, its slots
    /// mapped as they land, so nothing loads synchronously at a request's start.
    reclaim_sync: bool = false,
    /// TF_INKLING2_MIX_ADMIT_EVERY: verify rounds (or decode steps) between admission ticks.
    every: usize = 4,
    /// TF_INKLING2_MIX_READS: experts an admission batch loads at most.
    reads: usize = 16,
    /// TF_INKLING2_MIX_PER_LAYER: admissions a layer a batch at most.
    per_layer: usize = 1,
    /// TF_INKLING2_MIX_GAIN: an admitted expert's score beats its donor's by this share.
    gain: f64 = 0.05,
    /// TF_INKLING2_MIX_HALF_LIFE: the demand history's half-life, in votes (ticks).
    half_life: f64 = 32,
    /// TF_INKLING2_MIX_PROMPT_WEIGHT: a prompt's demand votes with this weight (a decode tick's: 1).
    prompt_weight: f64 = 0.25,
    /// TF_INKLING2_MIX_WATCH_GIB: the bank gives slots back when MemAvailable falls under the floor and this.
    watch_gib: f64 = 1,
    /// TF_INKLING2_MIX_HYST_GIB: a reclaim maps the bank's given-back slots back only while MemAvailable stays this much
    /// over the watch's threshold (and the boot shed leaves it there), so the next request's watch does not shed them
    /// again (adapt.zig bankKeep).
    hyst_gib: f64 = 0.5,
    /// TF_INKLING2_MIX_HISTORY: the demand history file (unset or "off": none).
    history: ?[]const u8 = null,

    fn getF(name: [*:0]const u8) ?f64 {
        const v = std.c.getenv(name) orelse return null;
        return std.fmt.parseFloat(f64, std.mem.span(v)) catch {
            std.log.warn("inkling2: {s}={s} is not a number; its default", .{ name, std.mem.span(v) });
            return null;
        };
    }

    fn getU(name: [*:0]const u8) ?usize {
        const v = std.c.getenv(name) orelse return null;
        return std.fmt.parseInt(usize, std.mem.span(v), 10) catch {
            std.log.warn("inkling2: {s}={s} is not a count; its default", .{ name, std.mem.span(v) });
            return null;
        };
    }

    pub fn fromEnv() Opts {
        var o: Opts = .{};
        o.bank_gib = getF("TF_INKLING2_MIX_BANK_GIB");
        if (getF("TF_INKLING2_MIX_MARGIN_GIB")) |x| o.margin_gib = @max(x, 0);
        if (getF("TF_INKLING2_MIX_MX_SHARE")) |x| o.mx_share = std.math.clamp(x, 0, 1);
        if (getF("TF_INKLING2_MIX_MX_PRIORITY")) |x| o.mx_priority = @max(x, 0);
        if (std.c.getenv("TF_INKLING2_MIX_ADMIT")) |v| {
            o.admit = !std.mem.eql(u8, std.mem.span(v), "0");
            o.between = std.mem.eql(u8, std.mem.span(v), "between"); // xa
        }
        if (std.c.getenv("TF_INKLING2_MIX_LAND_WAIT")) |v| o.land_wait = std.mem.eql(u8, std.mem.span(v), "1"); // xa
        if (std.c.getenv("TF_INKLING2_MIX_RECLAIM")) |v| o.reclaim_sync = std.mem.eql(u8, std.mem.span(v), "sync"); // xa
        if (getU("TF_INKLING2_MIX_ADMIT_EVERY")) |x| o.every = @max(x, 1);
        if (getU("TF_INKLING2_MIX_READS")) |x| o.reads = std.math.clamp(x, 1, 256);
        if (getU("TF_INKLING2_MIX_PER_LAYER")) |x| o.per_layer = @max(x, 1);
        if (getF("TF_INKLING2_MIX_GAIN")) |x| o.gain = @max(x, 0);
        if (getF("TF_INKLING2_MIX_HALF_LIFE")) |x| o.half_life = @max(x, 0.5);
        if (getF("TF_INKLING2_MIX_PROMPT_WEIGHT")) |x| o.prompt_weight = std.math.clamp(x, 0, 1);
        if (getF("TF_INKLING2_MIX_WATCH_GIB")) |x| o.watch_gib = @max(x, 0);
        if (getF("TF_INKLING2_MIX_HYST_GIB")) |x| o.hyst_gib = @max(x, 0);
        if (std.c.getenv("TF_INKLING2_MIX_HISTORY")) |v| {
            const p = std.mem.span(v);
            if (p.len > 0 and !std.mem.eql(u8, p, "off")) o.history = p;
        }
        return o;
    }
};

/// The bank's slots of each pool (0 NVFP4, 1 MXFP8: the release formats, Config.act) for `room` bytes (MemAvailable
/// over the floor, the margin and what is made after the bank): the VMM chunks' overhead kept out, at most `bank_gib`,
/// split by bytes so each pool holds the same experts a layer (or `mx_share` of the bytes to MXFP8), every pool at most
/// its layers' experts.
pub fn bankSlots(m: *const Model, o: Opts, room: f64) [2]usize {
    const w = m.slotWords();
    const sb = [2]f64{ @floatFromInt(Arena.slotBytes(w, 0)), @floatFromInt(Arena.slotBytes(w, 1)) };
    const nl = [2]f64{ @floatFromInt(m.c.countAct(.nvfp4)), @floatFromInt(m.c.countAct(.mxfp8)) };
    var bytes = @max(room, 0) / vmm_overhead;
    if (o.bank_gib) |cap| bytes = @min(bytes, @max(cap, 0) * gib);
    const prio = o.mx_priority orelse 1; // the MXFP8 layers' experts a layer over the NVFP4 layers'
    const one = nl[0] * sb[0] + prio * nl[1] * sb[1]; // an expert in every NVFP4 layer and `prio` in every MXFP8 layer
    const mx_share = o.mx_share orelse (if (one > 0) prio * nl[1] * sb[1] / one else 0);
    const E: f64 = @floatFromInt(m.c.experts);
    const nv = if (nl[0] > 0) @min(@floor(bytes * (1 - mx_share) / sb[0]), nl[0] * E) else 0;
    const mx = if (nl[1] > 0) @min(@floor(bytes * mx_share / sb[1]), nl[1] * E) else 0;
    return .{ @intFromFloat(nv), @intFromFloat(mx) };
}

/// The bank's slot bytes for `slots` (0 NVFP4, 1 MXFP8).
pub fn bankBytes(m: *const Model, slots: [2]usize) f64 {
    const w = m.slotWords();
    return @as(f64, @floatFromInt(slots[0] * Arena.slotBytes(w, 0) + slots[1] * Arena.slotBytes(w, 1)));
}

// ---- the demand history (TF_INKLING2_MIX_HISTORY; DESIGN 3.7): aggregate demand only, never prompts, tokens, logits
// or weights. Rank 0 writes it (mode 0600, atomic: a temporary file, fsync, rename, the directory's fsync) at every
// request's end; at boot rank 0 reads it and every rank takes rank 0's bytes (adapt.zig), checks them alike and seeds
// the residency's history with them (the bank's boot fill then follows the blended scores). A file of another model,
// geometry or format version, or a damaged one, is ignored, never overwritten (writes off for the run).
//
//   "INK2BANK" | version u32 = 1 | L u32 | E u32 | world u32 | votes u64
//   identity: sha256 of the release's config.json, of its model.safetensors.index.json, of the view's checkpoint.json
//   n [L] f64 | hist [L][E] f64           (residency.Residency's decayed weighted votes)
//   bank u32 | (layer u16, expert u16) x bank, by score (the bank at the save: diagnostic, the restore recomputes it)
//   crc32 u32 of everything before it

pub const history_version: u32 = 1;
pub const history_max = 8 << 20;

/// The history's identity: the release's config.json and model.safetensors.index.json, and the view's checkpoint.json
/// (their sha256: the model, its geometry and the EXL3 export the bank sits over).
pub fn historyIdentity(gpa: std.mem.Allocator, io: std.Io, native_dir: []const u8, view_dir: []const u8) ![3][32]u8 {
    var id: [3][32]u8 = undefined;
    const files = [3][2][]const u8{ .{ native_dir, "config.json" }, .{ native_dir, "model.safetensors.index.json" }, .{ view_dir, "checkpoint.json" } };
    for (files, 0..) |f, i| {
        const path = try std.fs.path.join(gpa, &.{ f[0], f[1] });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(bytes);
        std.crypto.hash.sha2.Sha256.hash(bytes, &id[i], .{});
    }
    return id;
}

/// A decoded history: the residency's evidence and votes, and the saved bank's experts (score order).
pub const Seed = struct { votes: u64, n: []f64, hist: []f64, bank: [][2]u16 };

/// The bytes of a history: `n` [L], `hist` [L][E], `bank` (layer, expert) pairs.
pub fn encodeHistory(gpa: std.mem.Allocator, id: [3][32]u8, L: usize, E: usize, world: usize, votes: u64, n: []const f64, hist: []const f64, bank: []const [2]u16) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "INK2BANK");
    for ([_]u32{ history_version, @intCast(L), @intCast(E), @intCast(world) }) |v| try out.appendSlice(gpa, std.mem.asBytes(&v));
    try out.appendSlice(gpa, std.mem.asBytes(&votes));
    for (id) |h| try out.appendSlice(gpa, &h);
    try out.appendSlice(gpa, std.mem.sliceAsBytes(n));
    try out.appendSlice(gpa, std.mem.sliceAsBytes(hist));
    const nb: u32 = @intCast(bank.len);
    try out.appendSlice(gpa, std.mem.asBytes(&nb));
    for (bank) |p| for (p) |v| try out.appendSlice(gpa, std.mem.asBytes(&v));
    const crc = std.hash.Crc32.hash(out.items);
    try out.appendSlice(gpa, std.mem.asBytes(&crc));
    return out.toOwnedSlice(gpa);
}

/// `bytes` checked (magic, version, identity, geometry, CRC, finite non-negative values, the bank's experts in range
/// and distinct) and decoded into `gpa`'s memory; error.HistoryMismatch / error.HistoryDamaged otherwise.
pub fn decodeHistory(gpa: std.mem.Allocator, bytes: []const u8, id: [3][32]u8, L: usize, E: usize, world: usize) !Seed {
    const head = 8 + 4 * 4 + 8 + 3 * 32;
    if (bytes.len < head + 8 or bytes.len > history_max) return error.HistoryDamaged;
    const crc = std.mem.readInt(u32, bytes[bytes.len - 4 ..][0..4], .little);
    if (std.hash.Crc32.hash(bytes[0 .. bytes.len - 4]) != crc) return error.HistoryDamaged;
    if (!std.mem.eql(u8, bytes[0..8], "INK2BANK")) return error.HistoryDamaged;
    const rd = struct {
        fn u(b: []const u8, at: usize) u32 {
            return std.mem.readInt(u32, b[at..][0..4], .little);
        }
    };
    if (rd.u(bytes, 8) != history_version or rd.u(bytes, 12) != L or rd.u(bytes, 16) != E or rd.u(bytes, 20) != world) return error.HistoryMismatch;
    const votes = std.mem.readInt(u64, bytes[24..32], .little);
    for (0..3) |i| if (!std.mem.eql(u8, bytes[32 + 32 * i ..][0..32], &id[i])) return error.HistoryMismatch;
    var at: usize = head;
    if (bytes.len < at + (L + L * E) * 8 + 4 + 4) return error.HistoryDamaged;
    const n = try gpa.alloc(f64, L);
    errdefer gpa.free(n);
    const hist = try gpa.alloc(f64, L * E);
    errdefer gpa.free(hist);
    @memcpy(std.mem.sliceAsBytes(n), bytes[at..][0 .. L * 8]);
    at += L * 8;
    @memcpy(std.mem.sliceAsBytes(hist), bytes[at..][0 .. L * E * 8]);
    at += L * E * 8;
    for (n) |v| if (!std.math.isFinite(v) or v < 0) return error.HistoryDamaged;
    for (hist) |v| if (!std.math.isFinite(v) or v < 0) return error.HistoryDamaged;
    const nb = rd.u(bytes, at);
    at += 4;
    if (nb > L * E or bytes.len != at + nb * 4 + 4) return error.HistoryDamaged;
    const bank = try gpa.alloc([2]u16, nb);
    errdefer gpa.free(bank);
    for (bank, 0..) |*p, i| {
        p.* = .{ std.mem.readInt(u16, bytes[at + 4 * i ..][0..2], .little), std.mem.readInt(u16, bytes[at + 4 * i + 2 ..][0..2], .little) };
        if (p[0] >= L or p[1] >= E) return error.HistoryDamaged;
        for (bank[0..i]) |q| if (q[0] == p[0] and q[1] == p[1]) return error.HistoryDamaged;
    }
    return .{ .votes = votes, .n = n, .hist = hist, .bank = bank };
}

pub fn freeSeed(gpa: std.mem.Allocator, sd: Seed) void {
    gpa.free(sd.n);
    gpa.free(sd.hist);
    gpa.free(sd.bank);
}

/// `bytes` into `path` atomically (a temporary file beside it, mode 0600, fsync, rename, the directory's fsync).
pub fn writeAtomic(gpa: std.mem.Allocator, path: []const u8, bytes: []const u8) !void {
    const dir = std.fs.path.dirname(path) orelse ".";
    const tmp = try std.fmt.allocPrintSentinel(gpa, "{s}.tmp{d}", .{ path, std.c.getpid() }, 0);
    defer gpa.free(tmp);
    const dst = try std.fmt.allocPrintSentinel(gpa, "{s}", .{path}, 0);
    defer gpa.free(dst);
    const dz = try std.fmt.allocPrintSentinel(gpa, "{s}", .{dir}, 0);
    defer gpa.free(dz);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c_uint, 0o600));
    if (fd < 0) return error.HistoryWrite;
    var ok = true;
    var off: usize = 0;
    while (off < bytes.len) {
        const w = std.c.write(fd, bytes[off..].ptr, bytes.len - off);
        if (w <= 0) {
            ok = false;
            break;
        }
        off += @intCast(w);
    }
    if (ok and std.c.fsync(fd) != 0) ok = false;
    _ = std.c.close(fd);
    if (!ok or std.c.rename(tmp, dst) != 0) {
        _ = std.c.unlink(tmp);
        return error.HistoryWrite;
    }
    const dfd = std.c.open(dz, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true });
    if (dfd >= 0) {
        _ = std.c.fsync(dfd);
        _ = std.c.close(dfd);
    }
}

test "x-mixed: a history round-trips, and a damaged or foreign one is refused" {
    const gpa = std.testing.allocator;
    var id: [3][32]u8 = undefined;
    for (&id, 0..) |*h, i| @memset(h, @intCast(i + 1));
    const n = [_]f64{ 1.5, 0 };
    const hist = [_]f64{ 0.25, 0.5, 0, 0.125, 0, 0 };
    const bank = [_][2]u16{ .{ 0, 1 }, .{ 1, 2 } };
    const bytes = try encodeHistory(gpa, id, 2, 3, 2, 7, &n, &hist, &bank);
    defer gpa.free(bytes);
    const sd = try decodeHistory(gpa, bytes, id, 2, 3, 2);
    defer freeSeed(gpa, sd);
    try std.testing.expectEqual(@as(u64, 7), sd.votes);
    try std.testing.expectEqualSlices(f64, &hist, sd.hist);
    try std.testing.expectEqual(@as(u16, 2), sd.bank[1][1]);
    var other = id;
    other[2][0] ^= 1;
    try std.testing.expectError(error.HistoryMismatch, decodeHistory(gpa, bytes, other, 2, 3, 2));
    const bad = try gpa.dupe(u8, bytes);
    defer gpa.free(bad);
    bad[40] ^= 0x10;
    try std.testing.expectError(error.HistoryDamaged, decodeHistory(gpa, bad, id, 2, 3, 2));
}

test "x-mixed: an MXFP8 priority of 2 gives the MXFP8 layers twice the experts a layer in the same bytes" {
    const nv: f64 = 8_957_964;
    const mx: f64 = 16_422_912;
    const bytes = 7.3 * gib;
    const share = 2 * 8 * mx / (40 * nv + 2 * 8 * mx);
    const n_nv = @floor(bytes * (1 - share) / nv); // NVFP4 slots (40 layers)
    const n_mx = @floor(bytes * share / mx); // MXFP8 slots (8 layers)
    try std.testing.expectApproxEqAbs(@as(f64, 2), (n_mx / 8) / (n_nv / 40), 0.02);
    try std.testing.expect(n_nv * nv + n_mx * mx <= bytes);
}

test "x-mixed: the bank's pools at the release's TP2 shape hold equal experts a layer" {
    // an NVFP4 slot 8,957,964 B, an MXFP8 slot 16,422,912 B; one expert in each of 40 + 8 layers = 489,701,856 B
    const nv: f64 = 8_957_964;
    const mx: f64 = 16_422_912;
    const one = 40 * nv + 8 * mx;
    try std.testing.expectApproxEqAbs(@as(f64, 489_701_856), one, 0.5);
    // DESIGN 1.4's 500K bank: 6.93 GiB of slots -> 607 NVFP4 + 121 MXFP8 (15.2 experts a layer)
    const bytes = 6.93 * gib;
    const share = 8 * mx / one;
    try std.testing.expectEqual(@as(f64, 607), @floor(bytes * (1 - share) / nv));
    try std.testing.expectEqual(@as(f64, 121), @floor(bytes * share / mx));
}
