//! Inkling-2's shape from config.json's text_config and its expert formats from hf_quant_config.json, checked.
//! The defaults are the release (ThinkingMachines/Inkling-2-Small); smaller synthetic shapes parse too (the gates).
const std = @import("std");

pub const max_layers = 64;

/// A layer's routed-expert storage: NVFP4 (e2m1, e4m3 per 16, amax / 2688) or MXFP8 (e4m3, UE8M0 per 32), or the
/// kortexa-ai EXL3 export's trellises (exl3.zig: mul1, 2 or 3 bits an expert; its layers' activation quantization
/// keeps the release's NVFP4 / MXFP8 format, Config.act).
pub const Format = enum { nvfp4, mxfp8, exl3 };

pub const Config = struct {
    hidden: u32 = 4608,
    layers: u32 = 48,
    vocab: u32 = 201024, // padded rows of embed / unembed
    unpadded_vocab: u32 = 200058, // logits past it are cut (model._logits)
    heads: u32 = 48,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    d_rel: u32 = 16,
    rel_extent: u32 = 1024, // global layers' relative-bias extent; sliding layers use the window
    window: u32 = 1024, // sliding_window_size
    eps: f32 = 1e-6,
    qk_clip: f32 = 22.627416997969522, // qk_l2_clip_threshold
    rel_clip: f32 = 8.0, // rel_l2_clip_threshold
    log_floor: f32 = 128000.0, // log_scaling_n_floor (global layers' query / rel scaling)
    log_alpha: f32 = 0.1,
    conv: u32 = 4, // sconv_kernel_size
    block: u32 = 6, // n_layers_per_attn_res_block
    experts: u32 = 384,
    topk: u32 = 8,
    inter: u32 = 2304, // intermediate_size: each routed expert's channels
    gate_cap: f32 = 4.0, // mlp_gate_softcap
    up_cap: f32 = 25.0, // mlp_up_softcap
    route_scale: f32 = 8.0,
    logit_cap: f32 = 30.0, // final_logit_softcapping
    eos: u32 = 200006,
    /// Layer i is sliding-window (local) when set, global otherwise.
    local: std.StaticBitSet(max_layers) = defaultLocal(),
    /// Each layer's routed-expert format (hf_quant_config.json's parameter groups; .exl3 for every layer of the EXL3
    /// export).
    formats: [max_layers]Format = defaultFormats(),
    /// Each layer's release format (hf_quant_config.json's), also for the EXL3 export: the activation quantization its
    /// experts' inputs take (NVFP4 per 16 under the row's amax, or MXFP8 per 32; TF_INKLING2_EXL3_ACT=bf16: none).
    act: [max_layers]Format = defaultFormats(),
    /// The routed experts are the kortexa-ai EXL3 export's (checkpoint.json beside config.json: exl3.zig).
    exl3: bool = false,

    pub fn isLocal(c: *const Config, i: u32) bool {
        return c.local.isSet(i);
    }

    /// The attn_res block boundary: the layer input becomes a block state (layer_idx % block == 0).
    pub fn isBoundary(c: *const Config, i: u32) bool {
        return i % c.block == 0;
    }

    pub fn qWidth(c: *const Config) u32 {
        return c.heads * c.head_dim;
    }

    pub fn kvWidth(c: *const Config) u32 {
        return c.kv_heads * c.head_dim;
    }

    pub fn countFormat(c: *const Config, f: Format) u32 {
        var n: u32 = 0;
        for (c.formats[0..c.layers]) |x| n += @intFromBool(x == f);
        return n;
    }

    /// Layers whose release format (Config.act) is `f`: the arena's pools (the native bank). x-merge (x-mixed DESIGN
    /// 2.7): countFormat's count for the release; the EXL3 export's release formats otherwise.
    pub fn countAct(c: *const Config, f: Format) u32 {
        var n: u32 = 0;
        for (c.act[0..c.layers]) |x| n += @intFromBool(x == f);
        return n;
    }

    /// Exactly the released checkpoint's shape: the family contract (every tensor's name, dtype and shape) applies.
    pub fn isRelease(c: *const Config) bool {
        const r = Config{};
        if (!c.local.eql(r.local)) return false;
        for (0..c.layers) |i| if (c.act[i] != r.formats[i]) return false; // the EXL3 export: the release's geometry
        return c.hidden == r.hidden and c.layers == r.layers and c.vocab == r.vocab and
            c.unpadded_vocab == r.unpadded_vocab and c.heads == r.heads and c.kv_heads == r.kv_heads and
            c.head_dim == r.head_dim and c.d_rel == r.d_rel and c.rel_extent == r.rel_extent and
            c.window == r.window and c.experts == r.experts and c.topk == r.topk and c.inter == r.inter and
            c.block == r.block and c.eos == r.eos;
    }

    /// Refuse a TP split the kernels cannot run: heads, kv heads, vocab rows and expert channels (in 32s) per rank.
    pub fn checkSplit(c: *const Config, world: u32) !void {
        if (world == 0 or world > 8) return error.BadWorld;
        if (c.heads % world != 0 or c.kv_heads % world != 0) return error.HeadsSplit;
        if (c.vocab % world != 0) return error.VocabSplit;
        if (c.inter % (32 * world) != 0) return error.ExpertSplit; // NVFP4 / MXFP8 blocks are 32 inputs wide
        if (c.exl3) {
            // the EXL3 cuts fall on the 128-wide Hadamard blocks (w13's 2 Il columns, w2's Il rows), and the GEMM's 4
            // warps split K / 16 (exl3.checkSplit)
            const il = c.inter / world;
            if (c.hidden % 128 != 0 or il % 128 != 0 or (c.hidden / 16) % 4 != 0 or (il / 16) % 4 != 0) return error.Exl3Split;
        }
    }
};

/// `dir`'s config.json and hf_quant_config.json, parsed and checked (`parse`).
pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
    const cp = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(cp);
    const qp = try std.fs.path.join(gpa, &.{ dir, "hf_quant_config.json" });
    defer gpa.free(qp);
    const cj = try std.Io.Dir.cwd().readFileAlloc(io, cp, gpa, .limited(1 << 22));
    defer gpa.free(cj);
    const qj = try std.Io.Dir.cwd().readFileAlloc(io, qp, gpa, .limited(1 << 24));
    defer gpa.free(qj);
    var c = try parse(gpa, cj, qj);
    // the kortexa-ai EXL3 export (a view of it: exl3.zig): its experts are trellises in every layer; the release's
    // formats stay the activation formats
    if (try @import("exl3.zig").detect(gpa, io, dir)) {
        c.exl3 = true;
        for (0..c.layers) |i| c.formats[i] = .exl3;
    }
    return c;
}

fn defaultLocal() std.StaticBitSet(max_layers) {
    var s = std.StaticBitSet(max_layers).empty;
    for (0..48) |i| if (i % 6 != 5) s.set(i); // global layers 5, 11, ..., 47
    return s;
}

fn defaultFormats() [max_layers]Format {
    var f: [max_layers]Format = @splat(.nvfp4);
    for (40..48) |i| f[i] = .mxfp8;
    return f;
}

fn get(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadConfig;
}

fn int(o: std.json.ObjectMap, key: []const u8) !u32 {
    return switch (try get(o, key)) {
        .integer => |i| if (i < 0 or i > std.math.maxInt(u32)) error.BadConfig else @intCast(i),
        else => error.BadConfig,
    };
}

fn float(o: std.json.ObjectMap, key: []const u8) !f32 {
    return switch (try get(o, key)) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn flag(o: std.json.ObjectMap, key: []const u8) !bool {
    return switch (try get(o, key)) {
        .bool => |b| b,
        else => error.BadConfig,
    };
}

/// config.json (top level or its text_config) and hf_quant_config.json. Refuses, by name, any option this family
/// does not compute (biases, pre-affine norms, shared experts, ...) and any shape its kernels cannot run.
pub fn parse(gpa: std.mem.Allocator, config_json: []const u8, quant_json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, config_json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const mt = try get(root, "model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, "inkling2")) return error.NotInkling2;
    const t = if (root.get("text_config")) |v| v.object else root;
    var c = Config{};
    c.hidden = try int(t, "hidden_size");
    c.layers = try int(t, "num_hidden_layers");
    c.vocab = try int(t, "vocab_size");
    c.unpadded_vocab = try int(t, "unpadded_vocab_size");
    c.heads = try int(t, "num_attention_heads");
    c.kv_heads = try int(t, "num_key_value_heads");
    c.head_dim = try int(t, "head_dim");
    c.d_rel = try int(t, "d_rel");
    c.rel_extent = try int(t, "rel_extent");
    c.window = try int(t, "sliding_window_size");
    c.eps = try float(t, "rms_norm_eps");
    c.qk_clip = try float(t, "qk_l2_clip_threshold");
    c.rel_clip = try float(t, "rel_l2_clip_threshold");
    c.log_floor = try float(t, "log_scaling_n_floor");
    c.log_alpha = try float(t, "log_scaling_alpha");
    c.conv = try int(t, "sconv_kernel_size");
    c.block = try int(t, "n_layers_per_attn_res_block");
    c.experts = try int(t, "n_routed_experts");
    c.topk = try int(t, "num_experts_per_tok");
    c.inter = try int(t, "intermediate_size");
    c.gate_cap = try float(t, "mlp_gate_softcap");
    c.up_cap = try float(t, "mlp_up_softcap");
    c.route_scale = try float(t, "route_scale");
    c.logit_cap = try float(t, "final_logit_softcapping");
    c.eos = switch (try get(root, "eos_token_id")) {
        .integer => |i| @intCast(i),
        else => return error.BadConfig,
    };
    // the attention options this family computes: one shape for sliding and global heads, no biases, gamma-less v norm
    if (try int(t, "swa_head_dim") != c.head_dim or try int(t, "swa_num_attention_heads") != c.heads or
        try int(t, "swa_num_key_value_heads") != c.kv_heads) return error.SlidingShapeDiffers;
    if (try flag(t, "q_bias") or try flag(t, "o_bias")) return error.AttentionBias;
    if (try flag(t, "pre_norm_pre_affine") or try flag(t, "final_norm_pre_affine")) return error.PreAffineNorm;
    if (!try flag(t, "use_value_norm_no_gamma") or !try flag(t, "attn_res_normalize_values")) return error.NormOption;
    if (try int(t, "n_shared_experts") != 0) return error.SharedExperts;
    if (!try flag(t, "use_gate_bias")) return error.NoGateBias;
    c.local = std.StaticBitSet(max_layers).empty;
    for ((try get(t, "local_layer_ids")).array.items) |v| {
        if (v != .integer or v.integer < 0 or v.integer >= c.layers) return error.BadConfig;
        c.local.set(@intCast(v.integer));
    }
    // the kernels' shapes: 32-wide blocks, top-k within the routing scratch, the short convolution's 4 taps
    if (c.layers == 0 or c.layers > max_layers or c.block == 0) return error.BadConfig;
    if (c.hidden % 32 != 0 or c.inter % 32 != 0 or c.heads % c.kv_heads != 0 or c.head_dim % 32 != 0) return error.Shape;
    if (c.topk == 0 or c.topk > 16 or c.experts < c.topk or c.conv != 4) return error.Shape;
    if (c.unpadded_vocab > c.vocab or c.eos >= c.unpadded_vocab) return error.Shape;
    try formatsOf(gpa, quant_json, &c);
    c.act = c.formats;
    return c;
}

/// Each layer's expert format: the parameter group whose tensors name its `mlp.experts.w13_weight`. NVFP4 must be the
/// amax-denominator scale format with [1, 16] blocks; anything else is refused.
fn formatsOf(gpa: std.mem.Allocator, quant_json: []const u8, c: *Config) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, quant_json, .{});
    defer parsed.deinit();
    const q = parsed.value.object;
    const qm = try get(q, "quant_method");
    if (qm != .string or !std.mem.eql(u8, qm.string, "inkling2")) return error.QuantMethod;
    var seen = std.StaticBitSet(max_layers).empty;
    for ((try get((try get(q, "model")).object, "parameter_groups")).array.items) |g| {
        const go = g.object;
        const dt = (try get(go, "compute_dtype")).string;
        const fmt: ?Format = if (std.mem.eql(u8, dt, "nvfp4")) .nvfp4 else if (std.mem.eql(u8, dt, "mxfp8")) .mxfp8 else null;
        for ((try get(go, "tensors")).array.items) |tv| {
            const name = tv.string;
            const layer = expertLayer(name) orelse continue;
            const f = fmt orelse return error.ExpertFormat; // an expert tensor in a float group
            if (layer >= c.layers or seen.isSet(layer) and c.formats[layer] != f) return error.ExpertFormat;
            if (f == .nvfp4) {
                const sf = try get(go, "weight_scale_format");
                if (sf != .string or !std.mem.eql(u8, sf.string, "amax_denominator")) return error.Nvfp4ScaleFormat;
            }
            c.formats[layer] = f;
            seen.set(layer);
        }
    }
    for (0..c.layers) |i| if (!seen.isSet(i)) return error.ExpertFormat;
}

/// N of "model.llm.layers.N.mlp.experts.w13_weight", else null.
fn expertLayer(name: []const u8) ?u32 {
    const pre = "model.llm.layers.";
    const post = ".mlp.experts.w13_weight";
    if (!std.mem.startsWith(u8, name, pre) or !std.mem.endsWith(u8, name, post)) return null;
    const mid = name[pre.len .. name.len - post.len];
    return std.fmt.parseInt(u32, mid, 10) catch null;
}

// ----------------------------------------------------------------------------------------------- tests

/// A synthetic text config: the release's values except what `t` names (the gates' shapes).
pub const TextShape = struct {
    layers: u32 = 48,
    hidden: u32 = 4608,
    experts: u32 = 384,
    topk: u32 = 8,
    inter: u32 = 2304,
    vocab: u32 = 201024,
    unpadded: u32 = 200058,
    eos: u32 = 200006,
};

pub fn textConfig(gpa: std.mem.Allocator, t: TextShape) ![]u8 {
    var locals: std.ArrayList(u8) = .empty;
    defer locals.deinit(gpa);
    var first = true;
    for (0..t.layers) |i| if (i % 6 != 5) {
        try locals.print(gpa, "{s}{d}", .{ if (first) "" else ", ", i });
        first = false;
    };
    return std.fmt.allocPrint(gpa,
        \\{{"model_type": "inkling2", "eos_token_id": {d}, "text_config": {{"hidden_size": {d},
        \\"num_hidden_layers": {d}, "vocab_size": {d}, "unpadded_vocab_size": {d}, "num_attention_heads": 48,
        \\"num_key_value_heads": 8, "head_dim": 128, "swa_head_dim": 128, "swa_num_attention_heads": 48,
        \\"swa_num_key_value_heads": 8, "d_rel": 16, "rel_extent": 1024, "sliding_window_size": 1024,
        \\"q_bias": false, "o_bias": false, "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1,
        \\"rms_norm_eps": 1e-06, "pre_norm_pre_affine": false, "final_norm_pre_affine": false,
        \\"use_value_norm_no_gamma": true, "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0,
        \\"local_layer_ids": [{s}], "sconv_kernel_size": 4, "final_logit_softcapping": 30.0,
        \\"n_layers_per_attn_res_block": 6, "attn_res_normalize_values": true, "n_routed_experts": {d},
        \\"num_experts_per_tok": {d}, "n_shared_experts": 0, "intermediate_size": {d}, "mlp_gate_softcap": 4.0,
        \\"mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": true}}}}
    , .{ t.eos, t.hidden, t.layers, t.vocab, t.unpadded, locals.items, t.experts, t.topk, t.inter });
}

/// hf_quant_config.json with layers [0, nv) NVFP4 and [nv, layers) MXFP8.
pub fn quantConfig(gpa: std.mem.Allocator, layers: u32, nv: u32) ![]u8 {
    var nvl: std.ArrayList(u8) = .empty;
    defer nvl.deinit(gpa);
    var mxl: std.ArrayList(u8) = .empty;
    defer mxl.deinit(gpa);
    for (0..layers) |i| {
        const l = if (i < nv) &nvl else &mxl;
        try l.print(gpa, "{s}\"model.llm.layers.{d}.mlp.experts.w13_weight\"", .{ if (l.items.len == 0) "" else ", ", i });
    }
    return std.fmt.allocPrint(gpa,
        \\{{"quant_method": "inkling2", "model": {{"parameter_groups": [
        \\{{"compute_dtype": "bfloat16", "tensors": ["model.llm.embed.weight"]}},
        \\{{"compute_dtype": "nvfp4", "weight_scale_format": "amax_denominator", "tensors": [{s}]}},
        \\{{"compute_dtype": "mxfp8", "tensors": [{s}]}}]}}}}
    , .{ nvl.items, mxl.items });
}

test "inkling2 config: the release values parse to the defaults; the release is recognized" {
    const gpa = std.testing.allocator;
    const t = try textConfig(gpa, .{});
    defer gpa.free(t);
    const q = try quantConfig(gpa, 48, 40);
    defer gpa.free(q);
    const c = try parse(gpa, t, q);
    try std.testing.expect(c.isRelease());
    try std.testing.expectEqual(@as(u32, 40), c.countFormat(.nvfp4));
    try std.testing.expectEqual(@as(u32, 8), c.countFormat(.mxfp8));
    try std.testing.expect(c.isLocal(0) and !c.isLocal(5) and !c.isLocal(47) and c.isLocal(46));
    try std.testing.expect(c.isBoundary(0) and c.isBoundary(6) and !c.isBoundary(5));
    try c.checkSplit(2);
    try c.checkSplit(4); // 576 expert channels a rank, 18 blocks of 32
    try std.testing.expectError(error.HeadsSplit, c.checkSplit(3)); // 8 kv heads over 3 ranks
}

test "inkling2 config: a smaller synthetic shape parses but is not the release; refusals are named" {
    const gpa = std.testing.allocator;
    const t = try textConfig(gpa, .{ .layers = 2, .hidden = 256, .experts = 4, .topk = 2, .inter = 128, .vocab = 512, .unpadded = 500, .eos = 499 });
    defer gpa.free(t);
    const q = try quantConfig(gpa, 2, 1);
    defer gpa.free(q);
    const c = try parse(gpa, t, q);
    try std.testing.expect(!c.isRelease());
    try std.testing.expectEqual(Format.mxfp8, c.formats[1]);
    const bad = try std.mem.replaceOwned(u8, gpa, t, "\"q_bias\": false", "\"q_bias\": true");
    defer gpa.free(bad);
    try std.testing.expectError(error.AttentionBias, parse(gpa, bad, q));
    const short = try quantConfig(gpa, 1, 1); // layer 1 has no expert format
    defer gpa.free(short);
    try std.testing.expectError(error.ExpertFormat, parse(gpa, t, short));
}
