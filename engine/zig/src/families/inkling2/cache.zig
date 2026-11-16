//! e-load (x5): the page cache a load leaves behind (the user, 2026-11-15: "Might be cached mem. you can sudoless
//! dropcache maybe"). GB10's CPU and GPU share one memory; MemAvailable counts clean page cache as room, but CUDA does
//! not get it back the way MemAvailable assumes. So each file a load reads through the page cache is dropped from it
//! (posix_fadvise DONTNEED) once its bytes are on the device: a checkpoint's shards (core.checkpoint maps them:
//! Model.open's dense weights, the dense EXL3 pack, mtp.safetensors) right after the checkpoint's mapping is closed
//! (`Files`), the small files (configs, indexes, the tokenizer) at the end of the load (`dropNames`). Only clean,
//! unmapped pages go; nothing is written, and a later read reads the file again (no numeric change). The expert stores
//! read past the cache already (fill.zig and exl3.zig's direct loader: O_DIRECT; exl3.zig openHost: DONTNEED a file).
//! Default on; TF_INKLING2_DROP_CACHE=0 (or, unset, TF_INKLING2_LEGACY=1): off. `report`: one line a rank.
const std = @import("std");
const core = @import("core");
const defaults = @import("defaults.zig");

extern "c" fn posix_fadvise(fd: c_int, off: i64, len: i64, advice: c_int) c_int;
const fadv_dontneed: c_int = 4;
const gib: f64 = 1024.0 * 1024.0 * 1024.0;

/// TF_INKLING2_DROP_CACHE=0: off; any other value: on; unset: on unless TF_INKLING2_LEGACY=1.
pub fn on() bool {
    if (std.c.getenv("TF_INKLING2_DROP_CACHE")) |v| return !std.mem.eql(u8, std.mem.span(v), "0");
    return !defaults.legacy();
}

/// /proc/meminfo's MemFree and Cached, in bytes (0 when unreadable).
pub const Mem = struct { free: u64 = 0, cached: u64 = 0 };

pub fn mem() Mem {
    var m: Mem = .{};
    var buf: [8192]u8 = undefined;
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return m;
    defer _ = std.c.close(fd);
    var n: usize = 0;
    while (n < buf.len) {
        const r = std.c.read(fd, buf[n..].ptr, buf.len - n);
        if (r <= 0) break;
        n += @intCast(r);
    }
    var it = std.mem.tokenizeScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| {
        inline for (.{ .{ "MemFree:", "free" }, .{ "Cached:", "cached" } }) |k| if (std.mem.startsWith(u8, line, k[0])) {
            var f = std.mem.tokenizeAny(u8, line[k[0].len..], " \tkB");
            @field(m, k[1]) = (std.fmt.parseInt(u64, f.next() orelse "0", 10) catch 0) * 1024;
        };
    }
    return m;
}

// this process's passes, for `report` (loads run on one thread)
var passes: usize = 0;
var files: usize = 0;
var first: Mem = .{}; // before the first pass
var last: Mem = .{}; // after the last pass
var freed_cached: i64 = 0; // Cached, before - after, summed over the passes
var freed_free: i64 = 0; // MemFree, after - before, summed over the passes

fn dropOne(path: []const u8) bool {
    var pb: [4096]u8 = undefined;
    if (path.len >= pb.len) return false;
    @memcpy(pb[0..path.len], path);
    pb[path.len] = 0;
    const fd = std.c.open(@ptrCast(&pb), .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    return posix_fadvise(fd, 0, 0, fadv_dontneed) == 0;
}

/// One pass: `paths` dropped from the page cache (when on), its MemFree / Cached before and after counted.
pub fn dropPaths(paths: []const []const u8) void {
    if (!on() or paths.len == 0) return;
    const before = mem();
    var n: usize = 0;
    for (paths) |p| n += @intFromBool(dropOne(p));
    const after = mem();
    if (passes == 0) first = before;
    last = after;
    passes += 1;
    files += n;
    freed_cached += @as(i64, @intCast(before.cached)) - @as(i64, @intCast(after.cached));
    freed_free += @as(i64, @intCast(after.free)) - @as(i64, @intCast(before.free));
}

/// A checkpoint's shard paths, copied (the checkpoint frees its own at close), dropped once its mapping is closed:
/// `var f = cache.Files.of(gpa, &ck); defer f.drop();` placed before `defer ck.close();` (defers run last first).
/// No path is kept when the drops are off or an allocation fails (the load never fails for it).
pub const Files = struct {
    gpa: std.mem.Allocator,
    all: [][]const u8 = &.{},
    n: usize = 0,

    pub fn of(gpa: std.mem.Allocator, ck: *const core.checkpoint.Checkpoint) Files {
        var f: Files = .{ .gpa = gpa };
        f.take(ck);
        return f;
    }

    /// The paths of `ck`'s files so far (any taken before replaced).
    pub fn take(f: *Files, ck: *const core.checkpoint.Checkpoint) void {
        f.free();
        if (!on() or ck.files.items.len == 0) return;
        f.all = f.gpa.alloc([]const u8, ck.files.items.len) catch return;
        for (ck.files.items) |*file| {
            f.all[f.n] = f.gpa.dupe(u8, file.path) catch break;
            f.n += 1;
        }
    }

    pub fn drop(f: *Files) void {
        dropPaths(f.all[0..f.n]);
        f.free();
    }

    fn free(f: *Files) void {
        for (f.all[0..f.n]) |p| f.gpa.free(p);
        if (f.all.len > 0) f.gpa.free(f.all);
        f.all = &.{};
        f.n = 0;
    }
};

/// The small files a load reads through the page cache, under the model folder (absent ones are skipped).
pub const small_files = [_][]const u8{
    "config.json",            "hf_quant_config.json",  "model.safetensors.index.json", "checkpoint.json",
    "tokenizer.json",         "tokenizer_config.json", "special_tokens_map.json",      "chat_template.jinja",
    "generation_config.json", "processor_config.json",
};

/// One pass over `names` under `dir`.
pub fn dropNames(dir: []const u8, names: []const []const u8) void {
    if (!on()) return;
    var bufs: [16][1024]u8 = undefined;
    var ps: [16][]const u8 = undefined;
    var n: usize = 0;
    for (names) |nm| {
        if (n == ps.len) break;
        ps[n] = std.fmt.bufPrint(&bufs[n], "{s}/{s}", .{ dir, nm }) catch continue;
        n += 1;
    }
    dropPaths(ps[0..n]);
}

/// The end of a load (the run tool, the engine): the small files' pass, then one line — the passes and files,
/// MemFree / Cached before the first pass and after the last, what the passes freed; with the drops off, MemFree /
/// Cached now.
pub fn finish(dir: []const u8, rank: usize, world: usize) void {
    dropNames(dir, &small_files);
    const g = struct {
        fn f(v: u64) f64 {
            return @as(f64, @floatFromInt(v)) / gib;
        }
        fn s(v: i64) f64 {
            return @as(f64, @floatFromInt(v)) / gib;
        }
    };
    if (!on()) {
        const now = mem();
        std.log.info("inkling2 rank {d}/{d}: page cache drops off (TF_INKLING2_DROP_CACHE=0 or TF_INKLING2_LEGACY=1); after the load MemFree {d:.2} GiB, Cached {d:.2} GiB", .{ rank, world, g.f(now.free), g.f(now.cached) });
        return;
    }
    std.log.info("inkling2 rank {d}/{d}: page cache: {d} files dropped in {d} passes (fadvise DONTNEED once on the device); MemFree {d:.2} -> {d:.2} GiB, Cached {d:.2} -> {d:.2} GiB (before the first pass -> after the last); the passes themselves: Cached -{d:.2} GiB, MemFree +{d:.2} GiB", .{ rank, world, files, passes, g.f(first.free), g.f(last.free), g.f(first.cached), g.f(last.cached), g.s(freed_cached), g.s(freed_free) });
}

test "inkling2 cache: /proc/meminfo parses, a pass over a real file and a missing one is harmless" {
    const m = mem();
    try std.testing.expect(m.free > 0 and m.cached > 0);
    dropPaths(&.{ "/proc/self/exe", "/nonexistent/e-load" });
}
