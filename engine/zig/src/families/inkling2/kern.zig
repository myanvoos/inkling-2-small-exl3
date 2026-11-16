//! Inkling-2's kernels on CUDA, typed: its ops (inkling2.cu), its NVFP4 experts (inkling2_nvfp4.cu, the core experts
//! with the softcapped SwiGLU epilogue), its MXFP8 experts (inkling2_mxfp8.cu), the EXL3 export's experts
//! (inkling2_exl3.cu: the universal trellis decode under this plan) and the core's grouped plan (experts.cu).
//! Each wrapper passes the kernel's arguments in its own order; the arithmetic is the kernels' (see their comments).
const std = @import("std");
const cuda = @import("cuda");
const defaults = @import("defaults.zig"); // the user, 2026-11-15: everything on by default

const Function = cuda.Function;
const Stream = cuda.Stream;
const Args = cuda.Args;

/// The derived NVFP4 decode instances (cuobjdump -elf of inkling2_nvfp4.fatbin), at 4 warps a CTA.
const nv_sym = struct {
    const gate_up = "_ZN17tf_inkling2_nvfp419nvfp4_expert_kernelILi2ELi4ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    const down_f32 = "_ZN17tf_inkling2_nvfp419nvfp4_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
};
const nv_warps = 4;
pub const nv_cols = 32;
pub const decode_tile = 16; // pairs a decode plan item holds (the core experts' TILE)
pub const prompt_tile = 64; // a prompt plan's item (the fact entries' 64-row stages)
/// The EXL3 wide-chunk kernels' builds (ink2_exl3_prompt_{gu,down}<suffix>: g 16-row tiles a program, its k tiles a stage,
/// stages, programs an SM): grid z = an item's passes (prompt_tile / 16 g). Index 0 is the forward's.
/// The fewest rows a chunk takes the fused prompt kernels at (exl3PromptGu / exl3PromptDown): below, most experts get
/// a few pairs and g2's four K chains in parallel beat one warp's walk of all of K (tf-inkling2-exl3-bench, layer 1,
/// skew 0.8, w13 old / new: 256 rows 8.54 / 10.07 ms, 512 rows 12.28 / 11.11 ms, 2048 rows 31.7 / 15.0 ms; w2 alike).
/// The bits are the same either side.
pub const exl3_fused_rows = 384;
/// The fewest rows a chunk's w13 takes the engine's grouped kernel at (exl3Grouped, tf_exl3_grouped_2), below
/// exl3_fused_rows: from here it is not slower than g2 on either kind of layer (tf-inkling2-exl3-bench, skew 0.8, w13
/// core / g2: layer 1 128 rows 5.22 / 5.82 ms, 256 rows 7.31 / 8.19; layer 45 128 rows 6.59 / 6.70, 256 rows 8.82 /
/// 9.25), while below it and for w2 the family's forms are faster (decode: one row g1 0.105 ms vs 0.124, verify's 8 rows
/// g2 0.749 vs 0.823). The bits are the same either way.
pub const exl3_core_rows = 128;
pub const exl3_prompt_configs = [_]struct { sfx: []const u8, g: usize, threads: usize = 256, p2: ?P2 = null, name: ?[]const u8 = null }{
    .{ .sfx = "_g4k2s2", .g = 4 }, .{ .sfx = "_g2k8s2", .g = 2 }, .{ .sfx = "_g2k4s3", .g = 2 }, .{ .sfx = "_g2k4s2", .g = 2 },
    // e-prefill (phase 5): the v3 build (inkling2_exl3.cu ink2_p2: the same bits; out/prefill/p5/e-prefill/REPORT.md):
    // gate-up kc k tiles a stage / st stages, down kcd / std; .tile: the plan item size moeExl3 makes for it (v3g4dt128:
    // the same kernels on 128-pair plans)
    .{ .sfx = "_v3g4d", .g = 4, .p2 = .{ .kc = 4, .st = 4, .kcd = 3, .std = 5 } },
    .{ .sfx = "_v3g4d", .g = 4, .p2 = .{ .kc = 4, .st = 4, .kcd = 3, .std = 5, .tile = 128 }, .name = "v3g4dt128" },
};
/// e-prefill: a v3 build's pipelines (gate-up: k tiles a stage, stages; down: kcd, std) and the plan item size moeExl3
/// makes for it.
pub const P2 = struct { kc: usize, st: usize, kcd: usize, std: usize, tile: usize = prompt_tile };
/// ink2_p2::smem_bytes<G, KC, ST>(): ST stages of 16 G rows' KC k tiles ((16 KC + 8) halves a row) and 8 K3 weight tiles a
/// k tile (96 bytes each), or E [16 G][132] fp32 when larger (E aliases the stages).
pub fn p2Smem(g: usize, kc: usize, st: usize) u32 {
    const stage = 16 * g * (16 * kc + 8) * 2 + 8 * kc * 24 * 4;
    return @intCast(@max(st * stage, 16 * g * 132 * 4));
}
/// e-prefill: the index of the build named `want` (exl3_prompt_configs: its name, else its suffix without "_").
pub fn exl3PromptConfig(want: []const u8) !usize {
    for (exl3_prompt_configs, 0..) |cfg, at| {
        if (std.mem.eql(u8, cfg.name orelse cfg.sfx[1..], want)) return at;
    }
    return error.Exl3PromptConfig; // a build this binary has
}
/// The largest prompt plan (grouped.maxItems at 8 rows x top-8: 65 items) the MXFP8 experts run with the 16-row form
/// (r1): such an item holds at most 8 pairs, and r1 streams the weights with less waste than r4 (verify windows:
/// 7-18% faster in the out/kernels bench, every byte equal).
const max_r1_items = 65;
const fact_wn = [2]usize{ 4, 8 }; // ink2_nvfp4_fact_gate_up, _down_f32 warps (the core's fact_wn)
// p3-expert: a wide prompt chunk's MXFP8 plan item (TF_INKLING2_PREFILL_EXPERTS): the wide form's two warp rows of 64
pub const wide_tile = 128;
const wide_wn = 2; // ink2_mx8_wide_*: 2 x 2 warps over 128 rows x 2 blocks, or 1 x 4 over 64 rows x 4 blocks

/// WideMx<M, 2>::SMEM (inkling2_mxfp8.cu; static_asserts there): 2 stages of the larger layout's stage, 128 rows' or 64
/// rows' 64 inputs and 2 or 4 blocks (1056 bytes) of 2 k32 groups of `parts`.
fn wideSmem(parts: usize) u32 {
    const su = @max(128 * 8 + wide_wn * 2 * parts * 66, 64 * 8 + 2 * wide_wn * 2 * parts * 66);
    return @intCast(2 * su * 16);
}

fn promptSmem(parts: usize, wn: usize) u32 {
    return cuda.experts.Nvfp4Experts.promptSmem(parts, wn);
}

pub const Kernels = struct {
    ops: cuda.Module,
    nv: cuda.Module,
    mx: cuda.Module,
    ex3: cuda.Module, // inkling2_exl3.cu: the kortexa-ai EXL3 export's experts (exl3.zig)
    core3: cuda.Module, // the engine's EXL3 experts (exl3_experts.cu): tf_exl3_grouped_2, the decode-side grouped GEMM
    exl3_cfg: usize = 0, // the wide-chunk kernels' build (exl3_prompt_configs; TF_INKLING2_EXL3_PROMPT=<suffix> picks one)
    // e-prefill: TF_INKLING2_EXL3_ACTROT=1: each projection's activation and rotation in one pass (exl3ActRot13 / 2;
    // the same bits as exl3Act then exl3Rot); default off
    exl3_actrot: bool = false,
    dec: Exl3Dec, // e-decode: the decode / verify windows' EXL3 kernels (Exl3Dec, the end of this struct)
    plans: cuda.Module,
    router: cuda.grouped.Router,
    f: struct {
        rmsnorm: Function,
        sconv_static: Function,
        sconv_dynamic: Function,
        head_prep: Function,
        attention: Function,
        attention_gqa: Function,
        sconv_edges: Function,
        sconv_static_seg: Function,
        sconv_dynamic_seg: Function,
        rank_sum_pair_bf16: Function,
        cache_write: Function,
        attn_res: Function,
        block_push: Function,
        residual: Function,
        layer_out: Function,
        bf16_to_f32: Function,
        f32_to_bf16: Function,
        rank_sum: Function,
        rank_sum_bf16: Function,
        embed: Function,
        logits: Function,
        route: Function,
        combine: Function,
        combine_parts: Function,
        combine_pair: Function,
        items_to_slots: Function,
        demand: Function,
        unsplit: Function,
        nv_gate_up: Function,
        nv_down: Function,
        nv_fact_gate_up: Function,
        nv_fact_down: Function,
        mx_gate_up: [3]Function, // 16, 32, 64 rows a pass
        mx_down: [3]Function,
        // p3-expert: the wide prompt chunks' MXFP8 form (128-pair items; inkling2_mxfp8.cu's end)
        mx_wide_gate_up: Function,
        mx_wide_down: Function,
        attention_tc: [2]Function, // p3-tc: ink2_attention_tc_d64, _d128 (TF_INKLING2_PREFILL_ATTN=tc)
        // p3-attn: the tiled prompt attention (attentionTiled)
        attn_tiled_scores: Function,
        attn_tiled_softmax: Function,
        attn_tiled_values: Function,
        // p3-comm: the wide chunk's down projection and combine in column parts (forward.moeOverlap)
        combine_pair_cols: Function,
        nv_fact_down_cols: Function,
        mx_down_cols: Function, // the r4 form's
        // p4-dattn: the decode / verify attention, split-K flash decoding (attentionFdec; Model.decode_attn)
        fdec: [2]Function, // ink2_fdec_d64, _d128
        fdec_merge: Function,
        // p4-small: the small prefill items (inkling2.cu's end; their wrappers at the end of Kernels)
        attn_tiled_scores_rel: Function,
        attn_res_norm: Function,
        sconv_static_seg_out: Function,
        sconv_static_seg_ld: Function,
        attention_fa: [2]Function, // ink2_attention_fa_d64, _d128 (TF_INKLING2_PREFILL_ATTN=fa)
        attention_far: [2]Function, // ink2_attention_far_d64, _d128 (fa with TF_INKLING2_REL_WIDE=2)
        // p4-mx: the W8A8 forms (TF_INKLING2_MX_ACT; inkling2_mxfp8.cu's end)
        mx8_quant: Function,
        mx8a_gate_up: [3]Function, // 16, 32, 64 rows a pass, quantized inputs
        mx8a_down: [3]Function,
        mx8a_wide_gate_up: Function, // a wide chunk's 128-pair items, shared-memory stages
        mx8a_wide_down: Function,
        // p4-xchg: the MoE's TP sum by token rows (forward.moeTok; TF_INKLING2_MOE_XCHG=token)
        xchg_tok_rows: Function,
        // the EXL3 export's experts (inkling2_exl3.cu): activation, rotation, decode-GEMM (decode / prompt plans),
        // the two epilogues, the per-row amax (TP's NVFP4 activation) and the check's dequantization
        exl3_act: Function,
        exl3_row_amax: Function,
        exl3_rot: Function,
        exl3_gemm: [2]Function, // g1 (items of up to 16 pairs), g2 (prompt plans: two member tiles a pass)
        exl3_grouped: Function, // the engine's tf_exl3_grouped_2 (mul1): exl3Grouped
        exl3_prompt_gu: [exl3_prompt_configs.len]Function, // the wide-chunk path: w13 + its epilogue (engine hooks)
        exl3_prompt_down: [exl3_prompt_configs.len]Function, // w2 + its epilogue, column blocks and K parts
        exl3_actrot13: Function, // e-prefill: ink2_exl3_actrot13 / 2 (TF_INKLING2_EXL3_ACTROT)
        exl3_actrot2: Function,
        exl3_gateup_epi: Function,
        exl3_down_epi: Function,
        exl3_dequant: Function,
        mix_split: Function, // x-mixed: ink2_mix_split (forward.moeMix)
    },
    nv_resident: [2]usize, // CTAs the grid may hold for the two decode forms
    arn_big: bool = false, // p4-small: ink2_attn_res_norm may take attn_res_norm_smem bytes of shared memory
    // e-step: a decode / verify chunk's fused launches (inkling2.cu's e-step block; TF_INKLING2_STEP_FUSE)
    es: struct { sconv_static2: Function, rank_sum_sconv: Function, sconv_static_out: Function, head_prep4: Function, route: Function, rot5: Function, group5: Function, head_best: Function, head_pick: Function },

    pub fn load(d: *const cuda.Driver, sms: usize) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.ops = try cuda.Module.load(d, cuda.kernels.inkling2);
        errdefer k.ops.unload();
        k.nv = try cuda.Module.load(d, cuda.kernels.inkling2_nvfp4);
        errdefer k.nv.unload();
        k.mx = try cuda.Module.load(d, cuda.kernels.inkling2_mxfp8);
        errdefer k.mx.unload();
        k.ex3 = try cuda.Module.load(d, cuda.kernels.inkling2_exl3);
        errdefer k.ex3.unload();
        k.core3 = try cuda.Module.load(d, cuda.kernels.exl3_experts);
        errdefer k.core3.unload();
        k.plans = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.plans.unload();
        k.router = try cuda.grouped.Router.resolve(k.plans);
        inline for (.{ "rmsnorm", "sconv_static", "sconv_dynamic", "head_prep", "attention", "cache_write", "attn_res", "block_push", "residual", "layer_out", "bf16_to_f32", "f32_to_bf16", "rank_sum", "rank_sum_bf16", "embed", "logits", "route", "combine", "combine_parts", "combine_pair", "items_to_slots", "demand", "unsplit", "attention_gqa", "sconv_edges", "sconv_static_seg", "sconv_dynamic_seg", "rank_sum_pair_bf16", "attn_tiled_scores", "attn_tiled_softmax", "attn_tiled_values", "combine_pair_cols" }) |n| {
            @field(k.f, n) = try k.ops.function("ink2_" ++ n);
        }
        k.f.nv_gate_up = try k.nv.function(nv_sym.gate_up);
        k.f.nv_down = try k.nv.function(nv_sym.down_f32);
        k.f.nv_fact_gate_up = try k.nv.function("ink2_nvfp4_fact_gate_up");
        k.f.nv_fact_down = try k.nv.function("ink2_nvfp4_fact_down_f32");
        try k.f.nv_fact_gate_up.allowDynamicShared(promptSmem(2, fact_wn[0]));
        try k.f.nv_fact_down.allowDynamicShared(promptSmem(1, fact_wn[1]));
        inline for (.{ "r1", "r2", "r4" }, 0..) |r, ri| {
            k.f.mx_gate_up[ri] = try k.mx.function("ink2_mx8_gate_up_" ++ r);
            k.f.mx_down[ri] = try k.mx.function("ink2_mx8_down_f32_" ++ r);
        }
        k.f.attention_tc = .{ try k.ops.function("ink2_attention_tc_d64"), try k.ops.function("ink2_attention_tc_d128") };
        // p3-expert: the MXFP8 wide form (gate-up's 50176 bytes of dynamic shared memory are past the 48 KiB default)
        k.f.mx_wide_gate_up = try k.mx.function("ink2_mx8_wide_gate_up");
        k.f.mx_wide_down = try k.mx.function("ink2_mx8_wide_down_f32");
        try k.f.mx_wide_gate_up.allowDynamicShared(wideSmem(2));
        try k.f.mx_wide_down.allowDynamicShared(wideSmem(1));
        // p3-comm: the down form in column parts (forward.moeOverlap)
        k.f.nv_fact_down_cols = try k.nv.function("ink2_nvfp4_fact_down_f32_cols");
        try k.f.nv_fact_down_cols.allowDynamicShared(promptSmem(1, fact_wn[1]));
        k.f.mx_down_cols = try k.mx.function("ink2_mx8_down_f32_r4_cols");
        // p4-dattn: the decode attention (attentionFdec)
        k.f.fdec = .{ try k.ops.function("ink2_fdec_d64"), try k.ops.function("ink2_fdec_d128") };
        k.f.fdec_merge = try k.ops.function("ink2_fdec_merge");
        // p4-small
        inline for (.{ "attn_tiled_scores_rel", "attn_res_norm", "sconv_static_seg_out", "sconv_static_seg_ld" }) |n| @field(k.f, n) = try k.ops.function("ink2_" ++ n);
        k.f.attention_fa = .{ try k.ops.function("ink2_attention_fa_d64"), try k.ops.function("ink2_attention_fa_d128") };
        k.f.attention_far = .{ try k.ops.function("ink2_attention_far_d64"), try k.ops.function("ink2_attention_far_d128") };
        // p4-small: FUSE's staged block states need the opt-in past 48 KiB (Model.open turns FUSE off without it)
        k.arn_big = if (k.f.attn_res_norm.allowDynamicShared(@intCast(attn_res_norm_smem))) |_| true else |_| false;
        // p4-mx: the W8A8 forms (the block-scaled e4m3 mma: the module is built for sm_121a)
        k.f.mx8_quant = try k.mx.function("ink2_mx8_quant");
        inline for (.{ "r1", "r2", "r4" }, 0..) |r, ri| {
            k.f.mx8a_gate_up[ri] = try k.mx.function("ink2_mx8a_gate_up_" ++ r);
            k.f.mx8a_down[ri] = try k.mx.function("ink2_mx8a_down_f32_" ++ r);
        }
        k.f.mx8a_wide_gate_up = try k.mx.function("ink2_mx8a_wide_gate_up");
        k.f.mx8a_wide_down = try k.mx.function("ink2_mx8a_wide_down_f32");
        try k.f.mx8a_wide_gate_up.allowDynamicShared(wideQ8Smem(true));
        try k.f.mx8a_wide_down.allowDynamicShared(wideQ8Smem(false));
        // p4-xchg: the MoE's TP sum by token rows (forward.moeTok)
        k.f.xchg_tok_rows = try k.ops.function("ink2_xchg_tok_rows");
        // e-step: the decode / verify chunk's fused launches
        inline for (.{ "sconv_static2", "rank_sum_sconv", "sconv_static_out", "head_prep4", "route", "head_best", "head_pick" }) |n| @field(k.es, n) = try k.ops.function("ink2_es_" ++ n);
        k.es.rot5 = try k.ex3.function("ink2_es_rot_group5"); // e-step: the q6 projection group (inkling2_exl3.cu)
        k.es.group5 = try k.ex3.function("ink2_es_group5_12_2_4");
        inline for (.{ "act", "row_amax", "rot", "gateup_epi", "down_epi", "dequant" }) |n| {
            @field(k.f, "exl3_" ++ n) = try k.ex3.function("ink2_exl3_" ++ n);
        }
        k.f.exl3_gemm = .{ try k.ex3.function("ink2_exl3_gemm_g1"), try k.ex3.function("ink2_exl3_gemm_g2") };
        k.f.exl3_grouped = try k.core3.function("tf_exl3_grouped_2");
        inline for (exl3_prompt_configs, 0..) |cfg, v| {
            k.f.exl3_prompt_gu[v] = try k.ex3.function("ink2_exl3_prompt_gu" ++ cfg.sfx);
            k.f.exl3_prompt_down[v] = try k.ex3.function("ink2_exl3_prompt_down" ++ cfg.sfx);
            if (cfg.p2) |p2| { // e-prefill: the v3 build's stages in dynamic shared memory (49 KB, past the 48 KB default)
                try k.f.exl3_prompt_gu[v].allowDynamicShared(p2Smem(cfg.g, p2.kc, p2.st));
                try k.f.exl3_prompt_down[v].allowDynamicShared(p2Smem(cfg.g, p2.kcd, p2.std));
            }
        }
        k.exl3_cfg = 0; // `k` starts undefined: no field default applies
        // the user, 2026-11-15 ("Turn everything on by default"): e-prefill's v3g4d (bitwise to the lane's builds) unless
        // TF_INKLING2_LEGACY=1 (g4k2s2); TF_INKLING2_EXL3_PROMPT picks any build
        if (!defaults.legacy()) k.exl3_cfg = try exl3PromptConfig("v3g4d");
        // a build this binary has (exl3_prompt_configs: its name, else its suffix without "_")
        if (std.c.getenv("TF_INKLING2_EXL3_PROMPT")) |v| k.exl3_cfg = try exl3PromptConfig(std.mem.span(v));
        k.f.exl3_actrot13 = try k.ex3.function("ink2_exl3_actrot13");
        k.f.exl3_actrot2 = try k.ex3.function("ink2_exl3_actrot2");
        k.exl3_actrot = if (std.c.getenv("TF_INKLING2_EXL3_ACTROT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else !defaults.legacy();
        k.dec = try Exl3Dec.load(k.ex3); // e-decode
        k.nv_resident = .{
            @max(1, try k.f.nv_gate_up.occupancy(nv_warps * 32, 0)) * sms,
            @max(1, try k.f.nv_down.occupancy(nv_warps * 32, 0)) * sms,
        };
        k.f.mix_split = try k.ops.function("ink2_mix_split"); // x-mixed
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.plans.unload();
        k.core3.unload();
        k.ex3.unload();
        k.mx.unload();
        k.nv.unload();
        k.ops.unload();
    }

    fn go(f: Function, s: Stream, grid: usize, block: usize, shared: u32, a: *Args) !void {
        if (grid == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = @intCast(block), .y = 1, .z = 1 }, .shared = shared }, s, a);
    }

    fn i(v: usize) c_int {
        return @intCast(v);
    }

    fn blocks(n: usize) usize {
        return (n + 255) / 256;
    }

    pub fn rmsnorm(k: *const Kernels, s: Stream, in: u64, in_f32: bool, w: u64, out: u64, rows: usize, h: usize, eps: f32) !void {
        var a: Args = .{};
        a.add(in);
        a.add(@as(c_int, @intFromBool(in_f32)));
        for ([_]u64{ w, out }) |v| a.add(v);
        a.add(i(h));
        a.add(eps);
        try go(k.f.rmsnorm, s, rows, 256, 0, &a);
    }

    pub fn sconvStatic(k: *const Kernels, s: Stream, x: u64, y: u64, w: u64, ring: u64, t: usize, c: usize, kk: usize, hist: u64) !void {
        var a: Args = .{};
        for ([_]u64{ x, y, w, ring }) |v| a.add(v);
        for ([_]usize{ t, c, kk }) |v| a.add(i(v));
        a.add(hist);
        try go(k.f.sconv_static, s, blocks(c), 256, 0, &a);
    }

    pub fn sconvDynamic(k: *const Kernels, s: Stream, x: u64, y: u64, static_w: u64, dyn: u64, ring: u64, t: usize, c: usize, kk: usize, dpk: usize, hist: u64) !void {
        var a: Args = .{};
        for ([_]u64{ x, y, static_w, dyn, ring }) |v| a.add(v);
        for ([_]usize{ t, c, kk, dpk }) |v| a.add(i(v));
        a.add(hist);
        try go(k.f.sconv_dynamic, s, blocks(c), 256, 0, &a);
    }

    pub const Prep = enum(c_int) { scaled_clip = 0, clip = 1, rms = 2 };

    pub fn headPrep(k: *const Kernels, s: Stream, v: u64, scale: u64, rows: usize, heads: usize, d: usize, mode: Prep, thr: f32, eps: f32, tau: bool, pos0: usize, alpha: f32, n_floor: f32) !void {
        var a: Args = .{};
        a.add(v);
        a.add(scale);
        a.add(i(heads));
        a.add(i(d));
        a.add(@intFromEnum(mode));
        a.add(thr);
        a.add(eps);
        a.add(@as(c_int, @intFromBool(tau)));
        a.add(i(pos0));
        a.add(alpha);
        a.add(n_floor);
        try go(k.f.head_prep, s, rows * heads, 128, 0, &a);
    }

    pub const Attn = struct { q: u64, kc: u64, vc: u64, rel: u64, sinks: u64, gate: u64, out: u64, scores: u64, heads: usize, kv_heads: usize, d: usize, q0: usize, window: usize, slots: usize, extent: usize, scaling: f32, cap: usize };

    pub fn attention(k: *const Kernels, s: Stream, rows: usize, p: Attn) !void {
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, p.vc, p.rel, p.sinks, p.gate, p.out, p.scores }) |v| a.add(v);
        for ([_]usize{ p.heads, p.kv_heads, p.d, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        a.add(i(p.cap));
        try go(k.f.attention, s, rows * p.heads, 128, 0, &a);
    }

    /// ink2_attention_gqa's dynamic shared memory: a tile of 128 keys (d / 8 + 1 16-byte groups a key), the
    /// `groups` heads' queries (fp32), a tile's probabilities and the reductions' second stage.
    pub fn attnGqaSmem(d: usize, groups: usize) u32 {
        return @intCast(128 * (d / 8 + 1) * 16 + groups * d * 4 + groups * 128 * 4 + groups * 16);
    }

    /// Whether ink2_attention_gqa runs this shape: d a multiple of 8 up to 128, at most 8 query heads a kv head.
    pub fn gqaFits(heads: usize, kv_heads: usize, d: usize) bool {
        return d % 8 == 0 and d <= 128 and kv_heads > 0 and heads % kv_heads == 0 and heads / kv_heads <= 8;
    }

    /// attention for a prompt chunk's rows, each kv head's query heads in one block that stages every tile of keys
    /// and values once (ink2_attention_gqa: ink2_attention's arithmetic thread for thread, the same bits).
    pub fn attentionGqa(k: *const Kernels, s: Stream, rows: usize, p: Attn) !void {
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, p.vc, p.rel, p.sinks, p.gate, p.out, p.scores }) |v| a.add(v);
        for ([_]usize{ p.heads, p.kv_heads, p.d, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        a.add(i(p.cap));
        const groups = p.heads / p.kv_heads;
        try go(k.f.attention_gqa, s, rows * p.kv_heads, groups * 128, attnGqaSmem(p.d, groups), &a);
    }

    /// Rows a segment of the parallel short convolutions (ink2_sconv_*_seg's SC_SEG).
    pub const sconv_seg: usize = 16;

    /// Bytes of the parallel convolutions' saved inputs for `t` rows of `c` channels (`kk` taps).
    pub fn sconvEdgeBytes(t: usize, c: usize, kk: usize) usize {
        return (t + sconv_seg - 1) / sconv_seg * (kk - 1) * c * 4;
    }

    /// sconvStatic on a prompt chunk's `t` rows in parallel, in place, no history: the segments' saved inputs (and
    /// the ring advanced), then every (segment, channel) as the one-thread loop (the same bits).
    pub fn sconvStaticRows(k: *const Kernels, s: Stream, x: u64, w: u64, ring: u64, edges: u64, t: usize, c: usize, kk: usize) !void {
        const units = (t + sconv_seg - 1) / sconv_seg * c;
        var e: Args = .{};
        for ([_]u64{ x, ring, edges }) |v| e.add(v);
        for ([_]usize{ t, c, kk }) |v| e.add(i(v));
        try go(k.f.sconv_edges, s, blocks(units), 256, 0, &e);
        var a: Args = .{};
        for ([_]u64{ x, w, edges }) |v| a.add(v);
        for ([_]usize{ t, c, kk }) |v| a.add(i(v));
        try go(k.f.sconv_static_seg, s, blocks(units), 256, 0, &a);
    }

    /// sconvDynamic on a prompt chunk's rows in parallel, as sconvStaticRows.
    pub fn sconvDynamicRows(k: *const Kernels, s: Stream, x: u64, static_w: u64, dyn: u64, ring: u64, edges: u64, t: usize, c: usize, kk: usize, dpk: usize) !void {
        const units = (t + sconv_seg - 1) / sconv_seg * c;
        var e: Args = .{};
        for ([_]u64{ x, ring, edges }) |v| e.add(v);
        for ([_]usize{ t, c, kk }) |v| e.add(i(v));
        try go(k.f.sconv_edges, s, blocks(units), 256, 0, &e);
        var a: Args = .{};
        for ([_]u64{ x, static_w, dyn, edges }) |v| a.add(v);
        for ([_]usize{ t, c, kk, dpk }) |v| a.add(i(v));
        try go(k.f.sconv_dynamic_seg, s, blocks(units), 256, 0, &a);
    }

    /// rankSumBf16 at two ranks with each rank's fp32 partials in a buffer of its own: out = bf16(y0 + y1).
    pub fn rankSumPairBf16(k: *const Kernels, s: Stream, y0: u64, y1: u64, out: u64, n: usize) !void {
        var a: Args = .{};
        for ([_]u64{ y0, y1, out }) |v| a.add(v);
        a.add(@as(u64, n));
        try go(k.f.rank_sum_pair_bf16, s, blocks(n), 256, 0, &a);
    }

    pub fn cacheWrite(k: *const Kernels, s: Stream, kk: u64, v: u64, kc: u64, vc: u64, t: usize, width: usize, q0: usize, slots: usize) !void {
        var a: Args = .{};
        for ([_]u64{ kk, v, kc, vc }) |x| a.add(x);
        for ([_]usize{ t, width, q0, slots }) |x| a.add(i(x));
        try go(k.f.cache_write, s, blocks(t * width), 256, 0, &a);
    }

    pub fn attnRes(k: *const Kernels, s: Stream, blks: u64, n: usize, smax: usize, partial: u64, partial_f32: bool, query: u64, nq: usize, out: u64, rows: usize, h: usize) !void {
        var a: Args = .{};
        a.add(blks);
        a.add(i(n));
        a.add(i(smax));
        a.add(partial);
        a.add(@as(c_int, @intFromBool(partial_f32)));
        a.add(query);
        a.add(i(nq));
        a.add(out);
        a.add(i(h));
        try go(k.f.attn_res, s, rows * nq, 1024, 0, &a); // four threads a partial thread of the 256 the sums are defined by
    }

    pub fn blockPush(k: *const Kernels, s: Stream, x: u64, blks: u64, t: usize, h: usize, at: usize, smax: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(blks);
        for ([_]usize{ t, h, at, smax }) |v| a.add(i(v));
        try go(k.f.block_push, s, blocks(t * h), 256, 0, &a);
    }

    pub fn residual(k: *const Kernels, s: Stream, h: u64, input: u64, res: u64, boundary: bool, n: usize) !void {
        var a: Args = .{};
        for ([_]u64{ h, input, res }) |v| a.add(v);
        a.add(@as(c_int, @intFromBool(boundary)));
        a.add(@as(u64, n));
        try go(k.f.residual, s, blocks(n), 256, 0, &a);
    }

    pub fn layerOut(k: *const Kernels, s: Stream, res: u64, mlp: u64, out: u64, n: usize) !void {
        var a: Args = .{};
        for ([_]u64{ res, mlp, out }) |v| a.add(v);
        a.add(@as(u64, n));
        try go(k.f.layer_out, s, blocks(n), 256, 0, &a);
    }

    pub fn toF32(k: *const Kernels, s: Stream, x: u64, y: u64, n: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(u64, n));
        try go(k.f.bf16_to_f32, s, blocks(n), 256, 0, &a);
    }

    pub fn toBf16(k: *const Kernels, s: Stream, x: u64, y: u64, n: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(u64, n));
        try go(k.f.f32_to_bf16, s, blocks(n), 256, 0, &a);
    }

    pub fn rankSum(k: *const Kernels, s: Stream, parts: u64, out: u64, world: usize, n: usize, is_f32: bool) !void {
        var a: Args = .{};
        a.add(parts);
        a.add(out);
        a.add(i(world));
        a.add(@as(u64, n));
        a.add(@as(c_int, @intFromBool(is_f32)));
        try go(k.f.rank_sum, s, blocks(n), 256, 0, &a);
    }

    /// fp32 parts [world][n] summed in rank order and rounded once to bf16 (rankSum's f32 form, then toBf16).
    pub fn rankSumBf16(k: *const Kernels, s: Stream, parts: u64, out: u64, world: usize, n: usize) !void {
        var a: Args = .{};
        a.add(parts);
        a.add(out);
        a.add(i(world));
        a.add(@as(u64, n));
        try go(k.f.rank_sum_bf16, s, blocks(n), 256, 0, &a);
    }

    pub fn embed(k: *const Kernels, s: Stream, ids: u64, table: u64, out: u64, t: usize, h: usize, lo: usize, hi: usize) !void {
        var a: Args = .{};
        for ([_]u64{ ids, table, out }) |v| a.add(v);
        for ([_]usize{ t, h, lo, hi }) |v| a.add(i(v));
        try go(k.f.embed, s, blocks(t * h), 256, 0, &a);
    }

    pub fn logits(k: *const Kernels, s: Stream, lg: u64, rows: usize, v: usize, unpadded: usize, cap: f32, out_f32: u64, argmax: u64) !void {
        var a: Args = .{};
        a.add(lg);
        a.add(i(v));
        a.add(i(unpadded));
        a.add(cap);
        a.add(out_f32);
        a.add(argmax);
        try go(k.f.logits, s, rows, 1024, 0, &a);
    }

    pub fn route(k: *const Kernels, s: Stream, lg: u64, bias: u64, routable: u64, t: usize, e: usize, topk: usize, scale: f32, picks: u64, weights: u64, want_ids: u64, want_scores: u64) !void {
        var a: Args = .{};
        for ([_]u64{ lg, bias, routable }) |v| a.add(v);
        for ([_]usize{ t, e, topk }) |v| a.add(i(v));
        a.add(scale);
        for ([_]u64{ picks, weights, want_ids, want_scores }) |v| a.add(v);
        try go(k.f.route, s, t, 64, 0, &a); // a row a block: warp 0 the masked pick, warp 1 the demand
    }

    pub fn combine(k: *const Kernels, s: Stream, y: u64, picks: u64, weights: u64, out: u64, t: usize, topk: usize, h: usize) !void {
        var a: Args = .{};
        for ([_]u64{ y, picks, weights, out }) |v| a.add(v);
        a.add(i(topk));
        a.add(i(h));
        if (t == 0) return;
        // a block a (row, 256 outputs): each output's sum is its own, the split never moves a bit
        try cuda.launch.launch(k.f.combine, .{ .grid = .{ .x = @intCast(t), .y = @intCast((h + 255) / 256), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// combine on the gathered fp32 parts y [world][t][topk][h]: each pair's rank-order sum, then combine's arithmetic.
    pub fn combineParts(k: *const Kernels, s: Stream, y: u64, picks: u64, weights: u64, out: u64, t: usize, topk: usize, h: usize, world: usize) !void {
        var a: Args = .{};
        for ([_]u64{ y, picks, weights, out }) |v| a.add(v);
        a.add(i(topk));
        a.add(i(h));
        a.add(i(world));
        a.add(@as(u64, t * topk * h));
        try go(k.f.combine_parts, s, t * ((h + 255) / 256), 256, 0, &a);
    }

    /// combineParts at two ranks with each rank's partials in a buffer of its own (a wide prompt chunk's row half):
    /// y0 rank 0's fp32 [t][topk][h], y1 rank 1's; picks / weights / out at the half's first row.
    pub fn combinePair(k: *const Kernels, s: Stream, y0: u64, y1: u64, picks: u64, weights: u64, out: u64, t: usize, topk: usize, h: usize) !void {
        var a: Args = .{};
        for ([_]u64{ y0, y1, picks, weights, out }) |v| a.add(v);
        a.add(i(topk));
        a.add(i(h));
        try go(k.f.combine_pair, s, t * ((h + 255) / 256), 256, 0, &a);
    }

    /// The layer's unmasked wants (`n` = rows x top-k) into its demand rows [E][3] (residency.zig).
    pub fn demand(k: *const Kernels, s: Stream, want_ids: u64, want_scores: u64, routable: u64, n: usize, out: u64) !void {
        var a: Args = .{};
        for ([_]u64{ want_ids, want_scores, routable }) |v| a.add(v);
        a.add(i(n));
        a.add(out);
        try go(k.f.demand, s, (n + 255) / 256, 256, 0, &a);
    }

    /// Gathered head parts [world][t][vl] into rows [t][world * vl].
    pub fn unsplit(k: *const Kernels, s: Stream, parts: u64, rows: u64, t: usize, vl: usize, world: usize) !void {
        var a: Args = .{};
        for ([_]u64{ parts, rows }) |v| a.add(v);
        for ([_]usize{ t, vl, world }) |v| a.add(i(v));
        if (t == 0) return;
        try cuda.launch.launch(k.f.unsplit, .{ .grid = .{ .x = 64, .y = @intCast(t), .z = @intCast(world) }, .block = .{ .x = 256 } }, s, &a);
    }

    // ---- the EXL3 export's experts (inkling2_exl3.cu; exl3.zig's slots, items mapped to slots layer E + expert) ----

    /// What the EXL3 kernels read of the store: the slot tables, the slot count, this rank's shape and the SwiGLU caps.
    /// The EXL3 store's tables (exl3.Store): slot addresses and bits (the family's kernels), and for the engine's grouped
    /// kernel each slot's w13 and w2 trellis addresses (int64) and half-bits (int32, 2 x bits).
    pub const Exl3 = struct { slot_ptr: u64, slot_k: u64, nslots: usize, hidden: usize, inter: usize, cg: f32, cu: f32, tp13: u64 = 0, tp2: u64 = 0, k2: u64 = 0 };

    fn go3(f: Function, s: Stream, gx: usize, gy: usize, gz: usize, block: usize, a: *Args) !void {
        if (gx == 0 or gy == 0 or gz == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = @intCast(gz) }, .block = .{ .x = @intCast(block), .y = 1, .z = 1 }, .shared = 0 }, s, a);
    }

    fn go2(f: Function, s: Stream, gx: usize, gy: usize, block: usize, a: *Args) !void {
        if (gx == 0 or gy == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block), .y = 1, .z = 1 }, .shared = 0 }, s, a);
    }

    /// `rows` rows of `kk` bf16 values (row stride x_stride) into fp16 rows (out_stride): mode 0 as they are, 1 NVFP4
    /// under the row's amax (amax_all [world][amax_stride] the ranks' parts' maxima, or 0: the row's own), 2 MXFP8.
    pub fn exl3Act(k: *const Kernels, s: Stream, x: u64, x_stride: usize, rows: usize, kk: usize, mode: u8, amax_all: u64, world: usize, amax_stride: usize, out: u64, out_stride: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(i(x_stride));
        a.add(i(kk));
        a.add(@as(c_int, mode));
        a.add(amax_all);
        a.add(i(world));
        a.add(i(amax_stride));
        a.add(out);
        a.add(i(out_stride));
        try go(k.f.exl3_act, s, rows, 128, 0, &a);
    }

    /// Each of `rows` rows' max |x| over its `kk` bf16 values into amax (fp32 [rows]).
    pub fn exl3RowAmax(k: *const Kernels, s: Stream, x: u64, x_stride: usize, rows: usize, kk: usize, amax: u64) !void {
        var a: Args = .{};
        a.add(x);
        a.add(i(x_stride));
        a.add(i(kk));
        a.add(amax);
        try go(k.f.exl3_row_amax, s, rows, 128, 0, &a);
    }

    /// Every routed pair's rotated input into out [pair][kk] fp16: row pair / row_slots (0: pair) of xq times its
    /// slot's suh (w2: which), H128, / sqrt(128).
    pub fn exl3Rot(k: *const Kernels, s: Stream, e: Exl3, xq: u64, xq_stride: usize, row_slots: usize, p: cuda.grouped.Plan, items: usize, w2: bool, out: u64, kk: usize) !void {
        if (kk % 128 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(xq);
        a.add(i(xq_stride));
        a.add(i(row_slots));
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(@as(c_int, @intFromBool(w2)));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(out);
        a.add(i(kk));
        try go2(k.f.exl3_rot, s, items, kk / 128, 32, &a);
    }

    /// The decode-GEMM of one projection (w2: down) over its output columns [c0, c0 + cn) (multiples of 128) into
    /// out[pair][col - c0] (row stride ldo), raw fp32 sums; a prompt plan's items take two member tiles a pass (the
    /// same bits a pair). `kparts` K parts, part q's sums in plane out + q * plane floats (canonical TP: 2 at world 1
    /// for the down projection, the parts a world-2 pair computes; else 1).
    pub fn exl3Gemm(k: *const Kernels, s: Stream, e: Exl3, w2: bool, prompt: bool, x: u64, p: cuda.grouped.Plan, items: usize, out: u64, ldo: usize, c0: usize, cn: usize, kparts: usize, plane: usize) !void {
        const kk = if (w2) e.inter else e.hidden;
        const n = if (w2) e.hidden else 2 * e.inter;
        if (kparts == 0 or kk % (64 * kparts) != 0 or (kk / 16 / kparts) % 4 != 0 or c0 % 128 != 0 or cn % 128 != 0 or c0 + cn > n) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(@as(c_int, @intFromBool(w2)));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(out);
        a.add(i(ldo));
        a.add(i(c0));
        a.add(i(kparts));
        a.add(@as(c_longlong, @intCast(plane)));
        try go2(k.f.exl3_gemm[@intFromBool(prompt)], s, items, cn / 128, 128, &a);
    }

    /// One matrix of the routed pairs on the engine's grouped EXL3 kernel (exl3_grouped.cuh grouped_body, mul1:
    /// tf_exl3_grouped_2), its plan-item ("standard") mode on this plan: x [pair][K] fp16 (rotated) -> out[split][pair][N]
    /// fp32 raw sums, K in kparts splits (plane pairs N). Its 8 column tiles a program, 4 warp chains summed in warp order
    /// and the engine's decoder are exl3Gemm's arithmetic: the same bits (tf-inkling2-exl3-bench). Items of up to `tile`
    /// pairs (16 or 64); grid (items, N / 128, member tiles x kparts), 128 threads.
    pub fn exl3Grouped(k: *const Kernels, s: Stream, e: Exl3, w2: bool, x: u64, p: cuda.grouped.Plan, items: usize, tile: usize, pairs: usize, out: u64, kparts: usize) !void {
        const kk = if (w2) e.inter else e.hidden;
        const n = if (w2) e.hidden else 2 * e.inter;
        const mt: usize = switch (tile) {
            16 => 1,
            64 => 4,
            else => return error.Exl3Shape,
        };
        if (e.tp13 == 0 or kparts == 0 or kk % (64 * kparts) != 0 or (kk / 16 / kparts) % 4 != 0 or n % 128 != 0) return error.Exl3Shape;
        if (items == 0 or pairs == 0) return;
        const tp = if (w2) e.tp2 else e.tp13;
        var a: Args = .{};
        // GroupedArgs, in its field order: x0 x1 tp0 tp1 k2_0 k2_1 uids ucount members z, then k n p sk maxm slots
        // nexp_max mats ns standard skip (one matrix: x1 = x0, tp1 = tp0; uids = the plan's items)
        for ([_]u64{ x, x, tp, tp, e.k2, e.k2, p.items, p.counts, p.members, out }) |v| a.add(v);
        for ([_]usize{ kk, n, pairs, kparts, tile, 0, items, 1, 0, 1 }) |v| a.add(@as(i32, @intCast(v)));
        a.add(@as(i32, -1));
        try cuda.launch.launch(k.f.exl3_grouped, .{ .grid = .{ .x = @intCast(items), .y = @intCast(n / 128), .z = @intCast(mt * kparts) }, .block = .{ .x = 128, .y = 1, .z = 1 }, .shared = 0 }, s, &a);
    }

    /// Grid z of the fused EXL3 prompt kernels for plan items of up to `tile` pairs: the item's passes of 16 g members
    /// (build cfg), at least one (x-merge; was prompt_tile / (16 g): the same grid for prompt plans).
    fn exl3Passes(cfg: usize, tile: usize) usize {
        const pm = 16 * exl3_prompt_configs[cfg].g;
        return @max(1, (tile + pm - 1) / pm);
    }

    /// A wide prompt chunk's w13 and its epilogue in one launch (inkling2_exl3.cu ink2_prompt, on the engine's EXL3
    /// prompt hooks): x [pair][H] fp16 (rotated) -> act [pair][Il] bf16; the bits of exl3Gemm then exl3GateUpEpi.
    /// Grid (2 Il / 128, items, the item's passes), 256 threads; plan items of up to `tile` pairs (x-merge: the plan's,
    /// x-mixed DESIGN 2.3: grid z = its passes of 16 g members, so a 128-pair item takes one program a pass; the
    /// kernels walk passes past grid z by stride, the same bits either way).
    pub fn exl3PromptGu(k: *const Kernels, s: Stream, e: Exl3, cfg: usize, x: u64, p: cuda.grouped.Plan, items: usize, tile: usize, act: u64) !void {
        if (e.hidden % 64 != 0 or (e.hidden / 16) % 4 != 0 or (2 * e.inter) % 128 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(e.cg);
        a.add(e.cu);
        a.add(act);
        const c = exl3_prompt_configs[cfg];
        if (c.p2) |p2| return p2Launch(k.f.exl3_prompt_gu[cfg], s, p2Smem(c.g, p2.kc, p2.st), 2 * e.inter / 128, items, exl3Passes(cfg, tile), &a);
        try go3(k.f.exl3_prompt_gu[cfg], s, 2 * e.inter / 128, items, exl3Passes(cfg, tile), c.threads, &a);
    }

    /// e-prefill: a v3 build's launch: grid (ncb column blocks x npass, items), npass = the item's passes of 16 g members
    /// (exl3Passes; the kernel's last argument: the programs of one item run side by side in x), 256 threads, `smem`
    /// bytes of dynamic shared memory (p2Smem).
    fn p2Launch(f: Function, s: Stream, smem: u32, ncb: usize, items: usize, npass: usize, a: *Args) !void {
        a.add(i(npass));
        if (ncb == 0 or items == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(ncb * npass), .y = @intCast(items), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = smem }, s, a);
    }

    /// e-prefill: the plan item size moeExl3 makes for a wide chunk's fused EXL3 path under build `cfg` (prompt_tile, or
    /// the v3 build's: 128 for v3g4dt128).
    pub fn exl3PromptTile(cfg: usize) usize {
        return if (exl3_prompt_configs[cfg].p2) |p2| p2.tile else prompt_tile;
    }

    /// A wide prompt chunk's w2 and its epilogue on y's columns [c0, c0 + cn) (y's column 0 = model column c0, row
    /// stride ldy), K in kparts parts added in rank order (canonical TP); the bits of exl3Gemm then exl3DownEpi.
    /// `tile`: the plan's item size, as exl3PromptGu's.
    pub fn exl3PromptDown(k: *const Kernels, s: Stream, e: Exl3, cfg: usize, x: u64, p: cuda.grouped.Plan, items: usize, tile: usize, y: u64, ldy: usize, c0: usize, cn: usize, kparts: usize) !void {
        if (kparts == 0 or e.inter % (64 * kparts) != 0 or (e.inter / 16 / kparts) % 4 != 0 or c0 % 128 != 0 or cn % 128 != 0 or c0 + cn > e.hidden) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(y);
        a.add(i(ldy));
        a.add(i(c0));
        a.add(i(kparts));
        const c = exl3_prompt_configs[cfg];
        if (c.p2) |p2| return p2Launch(k.f.exl3_prompt_down[cfg], s, p2Smem(c.g, p2.kcd, p2.std), cn / 128, items, exl3Passes(cfg, tile), &a);
        try go3(k.f.exl3_prompt_down[cfg], s, cn / 128, items, exl3Passes(cfg, tile), c.threads, &a);
    }

    /// w13's epilogue: z [pair][2 Il] (row stride ldz) -> act [pair][Il] bf16 (the rotation, svh13, the interleaved
    /// gate / up pairs' softcapped SwiGLU).
    pub fn exl3GateUpEpi(k: *const Kernels, s: Stream, e: Exl3, z: u64, ldz: usize, p: cuda.grouped.Plan, items: usize, act: u64) !void {
        var a: Args = .{};
        a.add(z);
        a.add(i(ldz));
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(e.cg);
        a.add(e.cu);
        a.add(act);
        try go2(k.f.exl3_gateup_epi, s, items, 2 * e.inter / 128, 32, &a);
    }

    /// w2's epilogue on y's columns [c0, c0 + cn) (y's column 0 = model column c0, row stride ldy) from the raw sums
    /// of `parts` K parts (z + q * zplane floats; z = y, parts 1: in place), the parts' terms added in rank order.
    pub fn exl3DownEpi(k: *const Kernels, s: Stream, e: Exl3, y: u64, ldy: usize, c0: usize, cn: usize, p: cuda.grouped.Plan, items: usize, z: u64, parts: usize, zplane: usize) !void {
        if (c0 % 128 != 0 or cn % 128 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(y);
        a.add(i(ldy));
        a.add(i(c0));
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(z);
        a.add(i(parts));
        a.add(@as(c_longlong, @intCast(zplane)));
        try go2(k.f.exl3_down_epi, s, items, cn / 128, 32, &a);
    }

    /// e-prefill: exl3Act (on T rows of kk bf16 values, mode 1 under each row's own amax) then exl3Rot with row_slots
    /// topk in one pass: out [t topk + j][kk] fp16, pair (t, j)'s slot lut[picks[t topk + j]] (lut: this layer's
    /// expert -> slot table; a negative entry skips the pick). The same bits. Grid T, 256 threads.
    pub fn exl3ActRot13(k: *const Kernels, s: Stream, e: Exl3, x: u64, x_stride: usize, rows: usize, kk: usize, mode: u8, picks: u64, topk: usize, lut: u64, out: u64) !void {
        if (kk % 128 != 0 or topk > 16) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        a.add(i(x_stride));
        a.add(i(kk));
        a.add(@as(c_int, mode));
        a.add(picks);
        a.add(i(topk));
        for ([_]u64{ lut, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(out);
        try go(k.f.exl3_actrot13, s, rows, 256, 0, &a);
    }

    /// e-prefill: exl3Act (each routed pair's row of x, kk bf16 values; mode 1 under amax_all's max over `world`
    /// entries, which the caller provides at every world) then exl3Rot (w2's suh2, row_slots 0) in one pass: out
    /// [pair][kk] fp16, the same bits. Grid (items, kk / 128), 32 threads.
    pub fn exl3ActRot2(k: *const Kernels, s: Stream, e: Exl3, x: u64, x_stride: usize, mode: u8, amax_all: u64, world: usize, amax_stride: usize, p: cuda.grouped.Plan, items: usize, out: u64, kk: usize) !void {
        if (kk % 128 != 0 or (mode == 1 and amax_all == 0)) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        a.add(i(x_stride));
        a.add(@as(c_int, mode));
        a.add(amax_all);
        a.add(i(world));
        a.add(i(amax_stride));
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(out);
        a.add(i(kk));
        try go2(k.f.exl3_actrot2, s, items, kk / 128, 32, &a);
    }

    /// The check's decode: W_q [K][N] fp16 of one slot's matrix (w2: down) into out.
    pub fn exl3Dequant(k: *const Kernels, s: Stream, slot_addr: u64, bits: usize, w2: bool, hidden: usize, inter: usize, out: u64) !void {
        const kk = if (w2) inter else hidden;
        const n = if (w2) hidden else 2 * inter;
        var a: Args = .{};
        a.add(slot_addr);
        a.add(i(bits));
        a.add(@as(c_int, @intFromBool(w2)));
        a.add(i(hidden));
        a.add(i(inter));
        a.add(out);
        try go2(k.f.exl3_dequant, s, kk / 16, n / 16, 32, &a);
    }

    pub fn itemsToSlots(k: *const Kernels, s: Stream, items: u64, counts: u64, lut: u64) !void {
        var a: Args = .{};
        for ([_]u64{ items, counts, lut }) |v| a.add(v);
        try go(k.f.items_to_slots, s, 1, 1024, 0, &a);
    }

    /// One routed expert matrix product over a plan: `up` (gate | up -> the softcapped SwiGLU, bf16 [pairs][inter]) or
    /// down (bf16 act -> fp32 [pairs][hidden]). `w` / `scale`: the arena's slots (scale: NVFP4 only).
    /// `act8` (p4-mx, MXFP8 only): W8A8, the checkpoint's declared numerics (expertsQ8): `in` holds q8 rows.
    pub const Experts = struct { fmt: @import("config.zig").Format, w: u64, scale: u64, slots: usize, hidden: usize, inter: usize, cg: f32, cu: f32, mx_ri: ?usize = null, wide: bool = false, act8: bool = false };

    pub fn experts(k: *const Kernels, s: Stream, up: bool, x: Experts, in: u64, in_stride: usize, row_slots: usize, p: cuda.grouped.Plan, items: usize, prompt: bool, out: u64) !void {
        const n = if (up) x.inter else x.hidden;
        const kin = if (up) x.hidden else x.inter;
        var a: Args = .{};
        a.add(in);
        a.add(i(in_stride));
        a.add(i(if (up) row_slots else 0));
        a.add(x.w);
        switch (x.fmt) {
            .nvfp4 => {
                a.add(x.scale);
                a.add(i(kin / 32));
                a.add(i(n / nv_cols));
                for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
                a.add(i(n));
                a.add(@as(f32, 0));
                a.add(@as(c_int, -1));
                const nb = n / nv_cols;
                if (prompt) {
                    const wn = fact_wn[if (up) 0 else 1];
                    try go(if (up) k.f.nv_fact_gate_up else k.f.nv_fact_down, s, items * ((nb + wn - 1) / wn), wn * 32, promptSmem(if (up) 2 else 1, wn), &a);
                } else {
                    const grid = @min((items * nb + nv_warps - 1) / nv_warps, k.nv_resident[if (up) 0 else 1]);
                    try go(if (up) k.f.nv_gate_up else k.f.nv_down, s, grid, nv_warps * 32, 0, &a);
                }
            },
            .mxfp8 => {
                if (x.act8) return k.expertsQ8(s, up, x, in, row_slots, p, items, prompt, out); // p4-mx
                a.add(i(x.slots));
                a.add(i(kin / 32));
                a.add(i(n / 32));
                for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
                a.add(i(n));
                a.add(x.cg);
                a.add(x.cu);
                a.add(@as(c_int, 1)); // round the gate and up sums to bf16 first (HF's bf16 gate_up)
                a.add(@as(c_int, -1));
                // r1 for decode (16-pair items) and for a prompt plan of at most 8 rows (65 items: a verify
                // window, at most 8 pairs an item), r4 for longer prompts (64-pair items); a pair's bits do not
                // depend on the form (the contract's r4 entries replace r1 bit for bit)
                if (x.wide) {
                    // p3-expert: a wide chunk's 128-pair plan (wide_tile) on the shared-memory form: grid items x
                    // ceil(NB / 2), 4 warps a CTA (inkling2_mxfp8.cu's end; the same bits as r1 / r4 pair for pair)
                    const nb = n / 32;
                    return go(if (up) k.f.mx_wide_gate_up else k.f.mx_wide_down, s, items * ((nb + wide_wn - 1) / wide_wn), 2 * wide_wn * 32, wideSmem(if (up) 2 else 1), &a);
                }
                const ri: usize = x.mx_ri orelse @as(usize, if (prompt and items > max_r1_items) 2 else 0);
                const units = items * (n / 32);
                try go(if (up) k.f.mx_gate_up[ri] else k.f.mx_down[ri], s, @min((units + 3) / 4, 4096), 128, 0, &a);
            },
            .exl3 => return error.Exl3Layer, // exl3Gemm and its kernels run these layers
        }
    }

    // ---- p3-tc: the tensor-core prompt attention (opt-in, TF_INKLING2_PREFILL_ATTN=tc; not bitwise) ----

    /// Keys a staged tile of ink2_attention_tc_d* (its TC_KEYS) and query rows a block (TC_ROWS).
    pub const tc_keys: usize = 32;
    pub const tc_rows: usize = 16;

    /// ink2_attention_tc_d*'s dynamic shared memory: two stages of a key tile and a value tile, d + 8 bf16 a key.
    pub fn attnTcSmem(d: usize) u32 {
        return @intCast(2 * 2 * tc_keys * (d + 8) * 2);
    }

    /// Whether ink2_attention_tc runs this shape: d 64 or 128, at most 8 query heads (warps) a kv head.
    pub fn tcFits(heads: usize, kv_heads: usize, d: usize) bool {
        return (d == 64 or d == 128) and kv_heads > 0 and heads % kv_heads == 0 and heads / kv_heads <= 8;
    }

    /// attention for a prompt chunk's `rows` rows on tensor cores (ink2_attention_tc_d*): the reference's rounding
    /// points, its bf16 matmuls' fp32 sums in the mma's order (NOT ink2_attention's bits). Reads no score scratch
    /// (p.scores, p.cap unused): every row in one launch, a block a (16-row tile, kv head).
    pub fn attentionTc(k: *const Kernels, s: Stream, rows: usize, p: Attn) !void {
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, p.vc, p.rel, p.sinks, p.gate, p.out }) |v| a.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        const f = if (p.d == 64) k.f.attention_tc[0] else k.f.attention_tc[1];
        const groups = p.heads / p.kv_heads;
        try go(f, s, (rows + tc_rows - 1) / tc_rows * p.kv_heads, groups * 32, attnTcSmem(p.d), &a);
    }
    // ---- end p3-tc ----

    // ---- p3-attn: the tiled prompt attention (inkling2.cu's ink2_attn_tiled_*; Model.prefill_attn == .tiled)

    /// Whether the tiled attention runs this shape: 4 or 6 query heads a kv head, d a multiple of 32 up to 128.
    pub fn tiledFits(heads: usize, kv_heads: usize, d: usize) bool {
        if (kv_heads == 0 or heads % kv_heads != 0) return false;
        const g = heads / kv_heads;
        return (g == 4 or g == 6) and d % 32 == 0 and d > 0 and d <= 128;
    }

    /// The tiled attention's score scratch a (row, head), in bf16 elements, for rows reading at most `cap` keys: the
    /// scores of positions lo & ~7 .. pos and the 16-byte groups around them (a multiple of 8: aligned groups).
    pub fn tiledStride(cap: usize) usize {
        return (cap + 16 + 7) / 8 * 8;
    }

    /// attention for a prompt chunk's rows (ink2_attention_gqa's outputs, bit for bit) in three launches: the scores
    /// tiled 16 rows x the kv group's heads x 128 keys a block (bf16 into the scratch at tiledStride(p.cap) elements a
    /// (row, head)), the softmax a warp a (row, head) (the exact exp-sum order of ink2_attention's 128 threads), the
    /// values tiled 8 rows x the kv group's heads x d a block. p.scores holds rows * heads * tiledStride(p.cap) bf16.
    pub fn attentionTiled(k: *const Kernels, s: Stream, rows: usize, p: Attn) !void {
        if (rows == 0) return;
        const g = p.heads / p.kv_heads;
        const stride = tiledStride(p.cap);
        // key tiles a row tile may need: its last row's position less its first row's lo & ~7, plus one
        const span = if (p.window > 0) @min(p.window + 24, p.q0 + rows) else p.q0 + rows;
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, p.rel, p.scores }) |v| a.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.d, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        a.add(i(stride));
        const sc_smem: u32 = @intCast((32 * 16 * g + 32 * 128) * 4);
        try cuda.launch.launch(k.f.attn_tiled_scores, .{ .grid = .{ .x = @intCast((rows + 15) / 16), .y = @intCast((span + 127) / 128), .z = @intCast(p.kv_heads) }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = sc_smem }, s, &a);
        var b: Args = .{};
        for ([_]u64{ p.scores, p.sinks }) |v| b.add(v);
        for ([_]usize{ rows, p.heads, p.q0, p.window, stride }) |v| b.add(i(v));
        try go(k.f.attn_tiled_softmax, s, (rows * p.heads + 7) / 8, 256, 0, &b);
        // the values' blocks take all d dims (the kernel can split them over gridDim.z blocks, the same bits, but
        // measured slower at every launch width on the GB10)
        const split: usize = 1;
        const dh = p.d / split;
        var c: Args = .{};
        for ([_]u64{ p.scores, p.vc, p.gate, p.out }) |v| c.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.d, p.q0, p.window, p.slots, stride }) |v| c.add(i(v));
        const v_smem: u32 = @intCast((32 * dh + 32 * 8 * g) * 4);
        try cuda.launch.launch(k.f.attn_tiled_values, .{ .grid = .{ .x = @intCast((rows + 7) / 8), .y = @intCast(p.kv_heads), .z = @intCast(split) }, .block = .{ .x = @intCast(dh), .y = 1, .z = 1 }, .shared = v_smem }, s, &c);
    }

    // ---- p3-comm: the wide chunk's down projection and combine in column parts (forward.moeOverlap)

    /// Output columns a part of the down projection starts and ends on: the NVFP4 prompt form's CTA tile (fact_wn[1]
    /// blocks of nv_cols); MXFP8 parts take the same bounds.
    pub const part_cols: usize = fact_wn[1] * nv_cols;

    /// experts(down) of a prompt plan over the output columns [c0, c0 + cn) only (multiples of part_cols), into `out`
    /// as rows of cn fp32 (pair p's part at out + p cn): ink2_nvfp4_fact_down_f32_cols (the fact form's CTA tiles of
    /// those columns) or ink2_mx8_down_f32_r4_cols (the r4 form's units of those blocks): every column's sums are the
    /// whole launch's, so the parts together are its bits, each weight read once.
    pub fn downCols(k: *const Kernels, s: Stream, x: Experts, in: u64, p: cuda.grouped.Plan, items: usize, out: u64, c0: usize, cn: usize) !void {
        const n = x.hidden;
        const kin = x.inter;
        if (c0 % part_cols != 0 or cn % part_cols != 0 or c0 + cn > n) return error.BadColumnPart;
        var a: Args = .{};
        a.add(in);
        a.add(i(kin));
        a.add(i(0));
        a.add(x.w);
        switch (x.fmt) {
            .nvfp4 => {
                a.add(x.scale);
                a.add(i(kin / 32));
                a.add(i(n / nv_cols));
                for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
                a.add(i(cn));
                a.add(@as(f32, 0));
                a.add(@as(c_int, -1));
                const wn = fact_wn[1];
                a.add(i(c0 / part_cols));
                a.add(i(cn / part_cols));
                try go(k.f.nv_fact_down_cols, s, items * (cn / part_cols), wn * 32, promptSmem(1, wn), &a);
            },
            .mxfp8 => {
                if (x.act8) return k.downColsQ8(s, x, in, p, items, out, c0, cn); // p4-mx
                a.add(i(x.slots));
                a.add(i(kin / 32));
                a.add(i(n / 32));
                for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
                a.add(i(cn));
                a.add(x.cg);
                a.add(x.cu);
                a.add(@as(c_int, 1));
                a.add(@as(c_int, -1));
                a.add(i(c0 / 32));
                a.add(i(cn / 32));
                const units = items * (cn / 32);
                try go(k.f.mx_down_cols, s, @min((units + 3) / 4, 4096), 128, 0, &a);
            },
            .exl3 => return error.Exl3Layer, // exl3Gemm and its kernels run these layers
        }
    }

    /// combinePair on the output columns [c0, c0 + cn) of `t` rows: y0 / y1 each rank's fp32 partials of those
    /// columns [t][topk][cn]; out at the rows' first row and column c0 (rows h apart). Each element combinePair's.
    pub fn combinePairCols(k: *const Kernels, s: Stream, y0: u64, y1: u64, picks: u64, weights: u64, out: u64, t: usize, topk: usize, cn: usize, h: usize) !void {
        var a: Args = .{};
        for ([_]u64{ y0, y1, picks, weights, out }) |v| a.add(v);
        a.add(i(topk));
        a.add(i(cn));
        a.add(i(h));
        try go(k.f.combine_pair_cols, s, t * ((cn + 255) / 256), 256, 0, &a);
    }

    // ---- p4-dattn: decode / verify attention, split-K flash decoding (inkling2.cu's end, inkling2_fdec.cuh)

    /// Keys a staged tile of the split kernels (inkling2_fdec.cuh KT) and their dynamic shared memory: two stages of
    /// a key tile and a value tile, d + 8 bf16 a row.
    pub const fdec_keys: usize = 32;
    pub fn fdecSmem(d: usize) u32 {
        return @intCast(2 * 2 * fdec_keys * (d + 8) * 2);
    }

    /// The most splits a launch takes (Model.fd_part is sized for it) and the most rows (a warp a 16 queries, 8
    /// warps a block at most).
    pub const fdec_max_splits: usize = 64;

    /// Whether the split kernels run this shape: d 64 or 128, at most 8 warps of 16 queries (rows x query heads of a
    /// kv group) a block.
    pub fn fdecFits(rows: usize, heads: usize, kv_heads: usize, d: usize) bool {
        if ((d != 64 and d != 128) or kv_heads == 0 or heads % kv_heads != 0) return false;
        return rows * (heads / kv_heads) <= 8 * 16;
    }

    /// Bytes of the splits' partials for `rows` rows: [rows][heads][splits][2 + d] fp32 (m, l, o).
    pub fn fdecPartBytes(rows: usize, heads: usize, d: usize, splits: usize) usize {
        return rows * heads * splits * (2 + d) * 4;
    }

    /// attention for a chunk of `rows` rows (decode, a verify window, a short prompt chunk): ink2_fdec_d* over
    /// (`splits` key splits, kv heads), each block every row x query head of its kv group (the K / V tiles read once
    /// a launch), its partials into `part`, then ink2_fdec_merge a block a (row, head): the splits in order, the sink,
    /// the bf16 output and its gate. NOT ink2_attention's bits (the reference's rounding points but p's and the sums'
    /// order: inkling2.cu). Reads no score scratch (p.scores, p.cap unused). The position (p.q0) is the split kernel's
    /// argument 8 (graphs patch it); the merge takes none.
    pub fn attentionFdec(k: *const Kernels, s: Stream, rows: usize, p: Attn, part: u64, splits: usize) !void {
        if (rows == 0) return;
        if (splits == 0 or splits > fdec_max_splits or !fdecFits(rows, p.heads, p.kv_heads, p.d)) return error.FdecShape;
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, p.vc, p.rel, part }) |v| a.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        const warps = (rows * (p.heads / p.kv_heads) + 15) / 16;
        const f = if (p.d == 64) k.f.fdec[0] else k.f.fdec[1];
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(splits), .y = @intCast(p.kv_heads), .z = 1 }, .block = .{ .x = @intCast(32 * warps), .y = 1, .z = 1 }, .shared = fdecSmem(p.d) }, s, &a);
        var b: Args = .{};
        for ([_]u64{ part, p.sinks, p.gate, p.out }) |v| b.add(v);
        b.add(i(p.d));
        b.add(i(splits));
        try go(k.f.fdec_merge, s, rows * p.heads, p.d, 0, &b);
    }
    // ---- end p4-dattn
    // ---- p4-small: the small prefill items (Model.rel_wide, Model.fuse; inkling2.cu's p4-small block)

    /// The relative bias's width ink2_attn_tiled_scores_rel makes (its RL_DR).
    pub const rel_dr: usize = 16;

    /// attentionTiled with each relative bias made inside the scores launch (ink2_attn_tiled_scores_rel: bf16 of a
    /// sequential fmaf chain over the 16 products, not cuBLASLt's sum) from the launch's r rows `r` (bf16 [rows][heads]
    /// [16]) and the layer's projection `proj` (bf16 [16][extent]); p.rel unused. Softmax and values: attentionTiled's.
    pub fn attentionTiledRel(k: *const Kernels, s: Stream, rows: usize, p: Attn, r: u64, proj: u64, d_rel: usize) !void {
        if (rows == 0) return;
        if (d_rel != rel_dr) return error.RelWidth;
        const g = p.heads / p.kv_heads;
        const stride = tiledStride(p.cap);
        const span = if (p.window > 0) @min(p.window + 24, p.q0 + rows) else p.q0 + rows;
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, r, proj, p.scores }) |v| a.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.d, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        a.add(i(stride));
        // the stages' floats; the bias inputs (16 g rows of 16 r values, 16 x 144 projection values) reuse them
        const sc_smem: u32 = @intCast(@max(32 * 16 * g + 32 * 128, 16 * g * 16 + 16 * 144) * 4);
        try cuda.launch.launch(k.f.attn_tiled_scores_rel, .{ .grid = .{ .x = @intCast((rows + 15) / 16), .y = @intCast((span + 127) / 128), .z = @intCast(p.kv_heads) }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = sc_smem }, s, &a);
        var b: Args = .{};
        for ([_]u64{ p.scores, p.sinks }) |v| b.add(v);
        for ([_]usize{ rows, p.heads, p.q0, p.window, stride }) |v| b.add(i(v));
        try go(k.f.attn_tiled_softmax, s, (rows * p.heads + 7) / 8, 256, 0, &b);
        const split: usize = 1;
        const dh = p.d / split;
        var c: Args = .{};
        for ([_]u64{ p.scores, p.vc, p.gate, p.out }) |v| c.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.d, p.q0, p.window, p.slots, stride }) |v| c.add(i(v));
        const v_smem: u32 = @intCast((32 * dh + 32 * 8 * g) * 4);
        try cuda.launch.launch(k.f.attn_tiled_values, .{ .grid = .{ .x = @intCast((rows + 7) / 8), .y = @intCast(p.kv_heads), .z = @intCast(split) }, .block = .{ .x = @intCast(dh), .y = 1, .z = 1 }, .shared = v_smem }, s, &c);
    }

    /// Whether ink2_attention_fa_d* run this shape: d 64 or 128, 6 query heads (warps) a kv head (the kernels are
    /// bounded for 2 blocks of 192 threads an SM: inkling2.cu FA_THREADS).
    pub fn faFits(heads: usize, kv_heads: usize, d: usize) bool {
        return (d == 64 or d == 128) and kv_heads > 0 and heads == 6 * kv_heads;
    }

    /// attentionTc's launch on ink2_attention_fa_d* (the online softmax: one sweep over the keys; tc's blocks and
    /// shared memory, faFits' shapes). Reads no score scratch.
    pub fn attentionFa(k: *const Kernels, s: Stream, rows: usize, p: Attn) !void {
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, p.vc, p.rel, p.sinks, p.gate, p.out }) |v| a.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        const f = if (p.d == 64) k.f.attention_fa[0] else k.f.attention_fa[1];
        const groups = p.heads / p.kv_heads;
        try go(f, s, (rows + tc_rows - 1) / tc_rows * p.kv_heads, groups * 32, attnTcSmem(p.d), &a);
    }

    /// attentionFa with each tile's relative bias made in the kernel (ink2_attention_far_d*: one m16n8k16 product a
    /// block of 8 dists from the r rows `r` (bf16 [rows][heads][16]) and the projection `proj` (bf16 [16][extent]),
    /// rounded to bf16); p.rel unused. Shared memory: tc's stages, the staged projection columns and a warp's
    /// [16][48] bf16 bias tile.
    pub fn attentionFaRel(k: *const Kernels, s: Stream, rows: usize, p: Attn, r: u64, proj: u64, d_rel: usize) !void {
        if (d_rel != rel_dr) return error.RelWidth;
        var a: Args = .{};
        for ([_]u64{ p.q, p.kc, p.vc, r, proj, p.sinks, p.gate, p.out }) |v| a.add(v);
        for ([_]usize{ rows, p.heads, p.kv_heads, p.q0, p.window, p.slots, p.extent }) |v| a.add(i(v));
        a.add(p.scaling);
        const f = if (p.d == 64) k.f.attention_far[0] else k.f.attention_far[1];
        const groups = p.heads / p.kv_heads;
        const smem: u32 = attnTcSmem(p.d) + 16 * 72 * 2 + @as(u32, @intCast(groups)) * 16 * 48 * 2;
        try go(f, s, (rows + tc_rows - 1) / tc_rows * p.kv_heads, groups * 32, smem, &a);
    }

    /// ink2_attn_res_norm's dynamic shared memory at most (opted in at load): H x 4 + S x H x 2 bytes (92,160 at
    /// H 4608 and 8 block states); with its 4,640 static bytes 100,896, under the 101,376 a block of sm_12x.
    pub const attn_res_norm_smem: usize = 94 * 1024;

    /// Where ink2_attn_res_norm takes its partial source from: the layer input's bf16 rows, fp32 rows, or
    /// ink2_residual's sum made on the fly from the attention's output (`partial`) and the layer input (`hh`).
    pub const ResPartial = enum(c_int) { bf16 = 0, f32 = 1, residual = 2 };

    /// attnRes and the rmsnorm of each query's output in one launch (ink2_attn_res_norm: the two kernels' arithmetic,
    /// the same bits; the fp32 rows never written): query q's output normed with weight w[q] into out[q] (bf16
    /// [rows][h]). `hh` / `boundary`: ResPartial.residual only.
    // ---- e-step: a decode / verify chunk's fused launches (TF_INKLING2_STEP_FUSE; inkling2.cu's e-step block)

    /// The sliding layers' k and v short convolutions (sconvStatic in place, each with its history) in one launch.
    pub fn sconvStatic2(k: *const Kernels, s: Stream, x0: u64, w0: u64, ring0: u64, hist0: u64, x1: u64, w1: u64, ring1: u64, hist1: u64, t: usize, c: usize, kk: usize) !void {
        var a: Args = .{};
        for ([_]u64{ x0, w0, ring0, hist0, x1, w1, ring1, hist1 }) |v| a.add(v);
        for ([_]usize{ t, c, kk }) |v| a.add(i(v));
        try cuda.launch.launch(k.es.sconv_static2, .{ .grid = .{ .x = @intCast(blocks(c)), .y = 2, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 }, .shared = 0 }, s, &a);
    }

    /// o_proj's TP sum (rankSumBf16 over `world` fp32 parts) and the attention's short convolution (sconvStatic, its
    /// history) in one launch: the conv's output into `out`.
    pub fn rankSumSconv(k: *const Kernels, s: Stream, parts: u64, world: usize, w: u64, ring: u64, hist: u64, out: u64, t: usize, c: usize, kk: usize) !void {
        var a: Args = .{};
        a.add(parts);
        a.add(i(world));
        for ([_]u64{ w, ring, hist, out }) |v| a.add(v);
        for ([_]usize{ t, c, kk }) |v| a.add(i(v));
        try go(k.es.rank_sum_sconv, s, blocks(c), 256, 0, &a);
    }

    /// The MLP's short convolution (sconvStatic's loop and history; x read only) with residual + layerOut folded in:
    /// h = bf16(res + conv(x)), res = float(h) + float(o) (o alone at a boundary).
    pub fn sconvStaticOut(k: *const Kernels, s: Stream, x: u64, w: u64, ring: u64, o: u64, h: u64, boundary: bool, t: usize, c: usize, kk: usize, hist: u64) !void {
        var a: Args = .{};
        for ([_]u64{ x, w, ring, o, h }) |v| a.add(v);
        a.add(@as(c_int, @intFromBool(boundary)));
        for ([_]usize{ t, c, kk }) |v| a.add(i(v));
        a.add(hist);
        try go(k.es.sconv_static_out, s, blocks(c), 256, 0, &a);
    }

    /// One headPrep call's arguments (headPrep's own, without the stream, pos0 and eps the calls share).
    pub const Prep1 = struct { v: u64, scale: u64, heads: usize, d: usize, mode: Prep, thr: f32, tau: bool, alpha: f32, n_floor: f32 };

    /// Four headPrep calls of `rows` rows at pos0 in one launch (ink2_es_head_prep4; its pos0 is the last argument).
    pub fn headPrep4(k: *const Kernels, s: Stream, calls: [4]Prep1, rows: usize, eps: f32, pos0: usize) !void {
        const EsPrep = extern struct { v: [4]u64, scale: [4]u64, heads: [4]c_int, d: [4]c_int, mode: [4]c_int, tau_on: [4]c_int, thr: [4]f32, alpha: [4]f32, n_floor: [4]f32, first: [5]c_int };
        var p: EsPrep = undefined;
        var at: usize = 0;
        for (calls, 0..) |cl, j| {
            p.v[j] = cl.v;
            p.scale[j] = cl.scale;
            p.heads[j] = i(cl.heads);
            p.d[j] = i(cl.d);
            p.mode[j] = @intFromEnum(cl.mode);
            p.tau_on[j] = @intFromBool(cl.tau);
            p.thr[j] = cl.thr;
            p.alpha[j] = cl.alpha;
            p.n_floor[j] = cl.n_floor;
            p.first[j] = i(at);
            at += rows * cl.heads;
        }
        p.first[4] = i(at);
        var a: Args = .{};
        a.add(p);
        a.add(eps);
        a.add(i(pos0));
        try go(k.es.head_prep4, s, at, 128, 0, &a);
    }
    /// core's exl3 LinearArgs as linear.cu reads them (exl3.zig LinearArgsC), five of them and their block ends.
    const EsLinearC = extern struct { xh: u64, t: u64, stride_k: i64, stride_nb: i64, svh: u64, bias: u64, y: u64, z: u64, counters: u64, y_dtype: i32, n: i32, sk: i32, m: i32, k: i32 };
    const EsGroup5C = extern struct { l: [5]EsLinearC, ends: [5]i32, members: i32 };
    const EsRot5C = extern struct { x: u64, suh: [5]u64, xh: [5]u64, x_dtype: i32, m: i32, k: i32 };

    /// Whether `ws` run as one exl3Group5 at `m` rows: 2..5 members, 6-bit mul1 (halves 12), one K, each plan's warps
    /// 4 (core's linear_body<12, 2, 4>: the kernel each member's own launch runs at 1-16 rows), 1..16 rows.
    pub fn group5able(ws: []const cuda.exl3.Weight, m: usize) bool {
        if (ws.len < 2 or ws.len > 5 or m < 1 or m > 16) return false;
        for (ws) |w| {
            const p = cuda.exl3.planOf(w);
            if (w.halves != 12 or w.codebook != .mul1 or w.k != ws[0].k or p.wk != 4 or w.k % 128 != 0 or w.n % 128 != 0) return false;
            if ((w.k / 16) % (p.sk * p.wk) != 0) return false;
        }
        return true;
    }

    /// The scratch exl3Group5 needs past core's Dense scratch layout (Dense.groupScratch: members side by side).
    pub fn group5Bytes(ws: []const cuda.exl3.Weight, m: usize) struct { z: usize, counters: usize, xh: usize } {
        var z: usize = 0;
        var c: usize = 0;
        for (ws) |w| {
            z += cuda.exl3.Linear.zBytes(w, 128);
            c += cuda.exl3.Linear.counterBytes(w);
        }
        return .{ .z = z, .counters = c, .xh = ws.len * m * ws[0].k * 2 };
    }

    /// y_i [m][n_i] = x [m][k] (bf16) through each of `ws` (group5able) in two launches: every member's input rotation
    /// (ink2_es_rot_group5), then every member's decode-row linear on its own blocks (ink2_es_group5_12_2_4); each
    /// member's arithmetic is its own matmul's (core's rot_in and linear_body: the same bits as dense_exl3's per-tensor
    /// calls). xh / z / counters: Dense.groupScratch's (z, counters zeroed; each block's last program resets its own).
    pub fn exl3Group5(k: *const Kernels, s: Stream, ws: []const cuda.exl3.Weight, x: u64, m: usize, ys: []const u64, y_f32: bool, xh: u64, z: u64, counters: u64, pdl: bool) !void {
        if (!group5able(ws, m) or ys.len != ws.len) return error.Exl3Group;
        const kk = ws[0].k;
        var rot: EsRot5C = .{ .x = x, .suh = @splat(0), .xh = @splat(0), .x_dtype = @backingInt(cuda.exl3.DType.bf16), .m = @intCast(m), .k = @intCast(kk) };
        var g: EsGroup5C = .{ .l = undefined, .ends = @splat(0), .members = @intCast(ws.len) };
        var zat = z;
        var cat = counters;
        var end: i32 = 0;
        const yd: i32 = @backingInt(if (y_f32) cuda.exl3.DType.f32 else cuda.exl3.DType.bf16);
        for (ws, ys, 0..) |w, y, j| {
            const p = cuda.exl3.planOf(w);
            if (p.sk > 1 and z == 0) return error.Exl3Scratch;
            rot.suh[j] = w.suh;
            rot.xh[j] = xh + j * m * kk * 2;
            g.l[j] = .{ .xh = rot.xh[j], .t = w.trellis, .stride_k = @intCast(w.stride_k), .stride_nb = @intCast(w.stride_nb), .svh = w.svh, .bias = w.bias, .y = y, .z = if (p.sk > 1) zat else 0, .counters = cat, .y_dtype = yd, .n = @intCast(w.n), .sk = @intCast(p.sk), .m = @intCast(m), .k = @intCast(kk) };
            zat += cuda.exl3.Linear.zBytes(w, m);
            cat += cuda.exl3.Linear.counterBytes(w);
            end += @intCast(w.n / 128 * p.sk);
            g.ends[j] = end;
        }
        for (ws.len..5) |j| g.ends[j] = end;
        var ra: Args = .{};
        ra.add(rot);
        try cuda.launch.launch(k.es.rot5, .{ .grid = .{ .x = @intCast((kk / 128 + 3) / 4), .y = @intCast(m), .z = @intCast(ws.len) }, .block = .{ .x = 128, .y = 1, .z = 1 }, .pdl = pdl }, s, &ra);
        var ga: Args = .{};
        ga.add(g);
        try cuda.launch.launch(k.es.group5, .{ .grid = .{ .x = @intCast(end), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 }, .shared = cuda.exl3.Linear.sharedBytes(ws[0], m), .pdl = pdl }, s, &ga);
    }

    /// Each row's (value, global index) of ink2_logits' argmax over this rank's columns [lo, lo + vl) of `rows`
    /// [t][vl] (bf16), into best [t] (float2: the softcapped value, the index's bits).
    pub fn esHeadBest(k: *const Kernels, s: Stream, rows: u64, t: usize, vl: usize, lo: usize, unpadded: usize, cap: f32, best: u64) !void {
        var a: Args = .{};
        a.add(rows);
        for ([_]usize{ vl, lo, unpadded }) |v| a.add(i(v));
        a.add(cap);
        a.add(best);
        try go(k.es.head_best, s, t, 1024, 0, &a);
    }

    /// Each row's pick over the ranks' pairs all [world][stride] (float2) into argmax [t] (i32).
    pub fn esHeadPick(k: *const Kernels, s: Stream, all: u64, world: usize, t: usize, stride: usize, argmax: u64) !void {
        var a: Args = .{};
        a.add(all);
        for ([_]usize{ world, t, stride }) |v| a.add(i(v));
        a.add(argmax);
        try go(k.es.head_pick, s, 1, 64, 0, &a);
    }

    /// route's outputs, bit for bit, from the compact ink2_es_route (a block a (row, pass): pass 0 the masked pick,
    /// pass 1 the demand when want_ids is set).
    pub fn esRoute(k: *const Kernels, s: Stream, lg: u64, bias: u64, routable: u64, t: usize, e: usize, topk: usize, scale: f32, picks: u64, weights: u64, want_ids: u64, want_scores: u64) !void {
        if (t == 0) return;
        if (e > 1024 or topk > 16) return error.EsRouteShape;
        var a: Args = .{};
        for ([_]u64{ lg, bias, routable }) |v| a.add(v);
        for ([_]usize{ t, e, topk }) |v| a.add(i(v));
        a.add(scale);
        for ([_]u64{ picks, weights, want_ids, want_scores }) |v| a.add(v);
        try cuda.launch.launch(k.es.route, .{ .grid = .{ .x = @intCast(t), .y = 2, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }
    // ---- end e-step

    pub fn attnResNorm(k: *const Kernels, s: Stream, blks: u64, n: usize, smax: usize, partial: u64, pm: ResPartial, hh: u64, boundary: bool, query: u64, nq: usize, w: [3]u64, out: [3]u64, rows: usize, h: usize, eps: f32) !void {
        const Arn = extern struct { w: [3]u64, out: [3]u64 };
        var a: Args = .{};
        a.add(blks);
        a.add(i(n));
        a.add(i(smax));
        a.add(partial);
        a.add(@intFromEnum(pm));
        a.add(hh);
        a.add(@as(c_int, @intFromBool(boundary)));
        a.add(query);
        a.add(i(nq));
        a.add(Arn{ .w = w, .out = out });
        a.add(i(h));
        a.add(eps);
        // the staged partial / output row (fp32) and the n block states (bf16): 92,160 bytes at H 4608 and 8 blocks
        const smem = h * 4 + n * h * 2;
        if (smem > attn_res_norm_smem or (smem > 48 * 1024 and !k.arn_big)) return error.AttnResNormSmem;
        try go(k.f.attn_res_norm, s, rows * nq, 1024, @intCast(smem), &a);
    }

    /// sconvStaticRows with the segments' loads ahead of their stores (ink2_sconv_static_seg_ld: the same bits).
    pub fn sconvStaticRowsLd(k: *const Kernels, s: Stream, x: u64, w: u64, ring: u64, edges: u64, t: usize, c: usize, kk: usize) !void {
        const units = (t + sconv_seg - 1) / sconv_seg * c;
        var e: Args = .{};
        for ([_]u64{ x, ring, edges }) |v| e.add(v);
        for ([_]usize{ t, c, kk }) |v| e.add(i(v));
        try go(k.f.sconv_edges, s, blocks(units), 256, 0, &e);
        var a: Args = .{};
        for ([_]u64{ x, w, edges }) |v| a.add(v);
        for ([_]usize{ t, c, kk }) |v| a.add(i(v));
        try go(k.f.sconv_static_seg_ld, s, blocks(units), 256, 0, &a);
    }

    /// The MLP's short convolution on a wide chunk with the layer's output add folded in (sconvStaticRows' edges,
    /// then ink2_sconv_static_seg_out): h = bf16(res + conv(x)), res = float(h) + float(o) (o alone at a boundary).
    /// x (the MLP output) is read only; the same bits as sconvStaticRows + residual + layerOut.
    pub fn sconvStaticRowsOut(k: *const Kernels, s: Stream, x: u64, w: u64, ring: u64, edges: u64, o: u64, h: u64, boundary: bool, t: usize, c: usize, kk: usize) !void {
        const units = (t + sconv_seg - 1) / sconv_seg * c;
        var e: Args = .{};
        for ([_]u64{ x, ring, edges }) |v| e.add(v);
        for ([_]usize{ t, c, kk }) |v| e.add(i(v));
        try go(k.f.sconv_edges, s, blocks(units), 256, 0, &e);
        var a: Args = .{};
        for ([_]u64{ x, w, edges, o, h }) |v| a.add(v);
        a.add(@as(c_int, @intFromBool(boundary)));
        for ([_]usize{ t, c, kk }) |v| a.add(i(v));
        try go(k.f.sconv_static_seg_out, s, blocks(units), 256, 0, &a);
    }
    // ---- end p4-small
    // ---- p4-mx: the MXFP8 experts in the checkpoint's W8A8 numerics (TF_INKLING2_MX_ACT=1; inkling2_mxfp8.cu's end)

    /// Bytes of a quantized MXFP8 row of `kin` inputs (inkling2_mxfp8.cu q8_stride): kin e4m3 codes in fragment order,
    /// kin / 32 ue8m0 exponents, padded to 16 bytes.
    pub fn q8Stride(kin: usize) usize {
        return (kin + kin / 32 + 15) / 16 * 16;
    }

    /// The wide W8A8 forms' warps a column (WN: 2 WN warps a CTA, WN 32-column blocks) and shared-memory stages
    /// (inkling2_mxfp8.cu INK2_MX8A_UP_* / _DN_*; picked by the out/prefill/p4/p4-mx/bench sweep).
    const q8_wide = struct {
        const up_wn: usize = 2;
        const up_st: usize = 2;
        const dn_wn: usize = 2;
        const dn_st: usize = 3;
    };

    /// WideQ8<M, WN, ST>::SMEM (inkling2_mxfp8.cu wide_q8_smem): ST stages of the larger layout's, 128 rows' 4 code
    /// chunks and WN blocks (66 uint4 a matrix) or 64 rows' and 2 WN blocks, of 2 k32 groups, + 32 uint4 of the
    /// rows' exponent words.
    fn wideQ8Smem(up: bool) u32 {
        const parts: usize = if (up) 2 else 1;
        const wn = if (up) q8_wide.up_wn else q8_wide.dn_wn;
        const st = if (up) q8_wide.up_st else q8_wide.dn_st;
        return @intCast(st * (32 + @max(128 * 4 + wn * 2 * parts * 66, 64 * 4 + 2 * wn * 2 * parts * 66)) * 16);
    }

    /// `rows` bf16 rows of `kin` inputs, `x_stride` elements apart, into q8 rows at `q` (ink2_mx8_quant: the EXL3
    /// lane's mxfp8_block arithmetic, sglang / vLLM PR 67325 _mxfp8_block; a thread a 32-input group).
    pub fn mx8Quant(k: *const Kernels, s: Stream, x: u64, x_stride: usize, rows: usize, kin: usize, q: u64) !void {
        var a: Args = .{};
        a.add(x);
        a.add(i(x_stride));
        a.add(i(rows));
        a.add(i(kin));
        a.add(q);
        a.add(i(q8Stride(kin)));
        try go(k.f.mx8_quant, s, blocks(rows * (kin / 32)), 256, 0, &a);
    }

    /// experts() on quantized inputs (x.act8): `in` holds q8 rows, the tokens' (gate-up: q8Stride(hidden) bytes each,
    /// pair p reading row p / row_slots) or the pairs' (down: the q8Stride(inter) rows the gate-up wrote into its
    /// `out`). Gate-up: the softcapped SwiGLU, rounded to bf16, quantized into q8 rows; down: fp32 [pairs][hidden].
    /// A wide chunk's 128-pair plan on the shared-memory form; r1 for decode and a prompt plan of at most 65 items,
    /// r4 otherwise (every form a pair's bits: the same mma sequence).
    fn expertsQ8(k: *const Kernels, s: Stream, up: bool, x: Experts, in: u64, row_slots: usize, p: cuda.grouped.Plan, items: usize, prompt: bool, out: u64) !void {
        const n = if (up) x.inter else x.hidden;
        const kin = if (up) x.hidden else x.inter;
        const nb = n / 32;
        var a: Args = .{};
        a.add(in);
        a.add(i(q8Stride(kin)));
        a.add(i(if (up) row_slots else 0));
        a.add(x.w);
        a.add(i(x.slots));
        a.add(i(kin / 32));
        a.add(i(nb));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(i(n));
        a.add(x.cg);
        a.add(x.cu);
        a.add(@as(c_int, -1));
        a.add(@as(c_int, 0)); // every column block
        a.add(i(nb));
        // a wide chunk's 128-pair plan on the shared-memory form: grid items x ceil(NB / WN), 2 WN warps a CTA
        if (x.wide) {
            const wn = if (up) q8_wide.up_wn else q8_wide.dn_wn;
            return go(if (up) k.f.mx8a_wide_gate_up else k.f.mx8a_wide_down, s, items * ((nb + wn - 1) / wn), 2 * wn * 32, wideQ8Smem(up), &a);
        }
        const ri: usize = x.mx_ri orelse @as(usize, if (prompt and items > max_r1_items) 2 else 0);
        const units = items * nb;
        try go(if (up) k.f.mx8a_gate_up[ri] else k.f.mx8a_down[ri], s, @min((units + 3) / 4, 4096), 128, 0, &a);
    }

    /// downCols() on the gate-up's q8 rows (x.act8) over the column blocks [c0 / 32, (c0 + cn) / 32), each part's rows
    /// cn floats wide (pair p's part at out + p cn): a wide chunk's 128-pair plan on the shared-memory form
    /// (ink2_mx8a_wide_down_f32), else ink2_mx8a_down_f32_r4.
    fn downColsQ8(k: *const Kernels, s: Stream, x: Experts, in: u64, p: cuda.grouped.Plan, items: usize, out: u64, c0: usize, cn: usize) !void {
        const kin = x.inter;
        var a: Args = .{};
        a.add(in);
        a.add(i(q8Stride(kin)));
        a.add(i(0));
        a.add(x.w);
        a.add(i(x.slots));
        a.add(i(kin / 32));
        a.add(i(x.hidden / 32));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(i(cn));
        a.add(x.cg);
        a.add(x.cu);
        a.add(@as(c_int, -1));
        a.add(i(c0 / 32));
        a.add(i(cn / 32));
        if (x.wide) return go(k.f.mx8a_wide_down, s, items * ((cn / 32 + q8_wide.dn_wn - 1) / q8_wide.dn_wn), 2 * q8_wide.dn_wn * 32, wideQ8Smem(false), &a);
        const units = items * (cn / 32);
        try go(k.f.mx8a_down[2], s, @min((units + 3) / 4, 4096), 128, 0, &a);
    }
    // ---- end p4-mx

    // ---- p4-xchg: the MoE's TP sum by token rows (TF_INKLING2_MOE_XCHG=token; forward.moeTok, inkling2.cu's end)

    /// Threads a block of ink2_xchg_tok_rows: four columns each, 512 columns a block.
    const xchg_threads: usize = 128;

    /// A rank's token rows: its fp32 down partials y [t][topk][n] (n = hidden, or a column part's width: pair p's part
    /// at y + p n) times their routing weights, summed over each row's picks in ascending expert id into out [t][n]
    /// fp32 (ink2_xchg_tok_rows: no pair's bf16 rounding; the row's rounding is the rank sum's, rankSumBf16 /
    /// rankSumPairBf16).
    pub fn xchgTokRows(k: *const Kernels, s: Stream, y: u64, picks: u64, weights: u64, out: u64, t: usize, topk: usize, n: usize) !void {
        if (n % 4 != 0 or topk > 16 or y % 16 != 0 or out % 16 != 0) return error.BadTokenRows;
        var a: Args = .{};
        for ([_]u64{ y, picks, weights, out }) |v| a.add(v);
        a.add(i(topk));
        a.add(i(n));
        const per = xchg_threads * 4;
        try go(k.f.xchg_tok_rows, s, t * ((n + per - 1) / per), xchg_threads, 0, &a);
    }
    // ---- end p4-xchg

    // ---- e-decode: the decode / verify windows' EXL3 experts in fewer launches (inkling2_exl3.cu ink2_dec;
    // TF_INKLING2_EXL3_DEC=1, Model.exl3_dec, forward.exl3DecExperts). The lean port: the one shipping build. ----------

    /// The decode-path GEMM builds (ink2_exl3_dgu<sfx> / ink2_exl3_ddown<sfx>: grp column groups of 4 chain warps a
    /// program, 128 grp threads; _t*: the swapped operands); every build gives g1's bits. TF_INKLING2_EXL3_DEC_CFG=<sfx
    /// without _> picks one (default t1p2). This tree has t1p2 only (e-decode's eleven others: out/prefill/p5/e-decode.patch).
    pub const exl3_dec_configs = [_]struct { sfx: []const u8, grp: usize }{
        // the swapped operands (the decoded tile as the mma's A operand: one mma a tile and member group of 8)
        .{ .sfx = "_t1p2", .grp = 1 },
    };

    pub const Exl3Dec = struct {
        rot13: Function,
        rot2: Function,
        ramax: Function, // ink2_exl3_drow_amax: ink2_exl3_row_amax's maxima by 16-byte loads
        gu: [exl3_dec_configs.len]Function,
        down: [exl3_dec_configs.len]Function,
        cfg: usize,
        swap: bool, // grid (column blocks, items) instead of (items, column blocks); TF_INKLING2_EXL3_DEC_SWAP=0: off
        pdl: bool, // the GEMMs as programmatic dependent launches (weights stream in early); TF_INKLING2_EXL3_DEC_PDL=0: off

        fn load(m: cuda.Module) !Exl3Dec {
            var e: Exl3Dec = undefined;
            e.rot13 = try m.function("ink2_exl3_drot13");
            e.rot2 = try m.function("ink2_exl3_drot2");
            e.ramax = try m.function("ink2_exl3_drow_amax");
            inline for (exl3_dec_configs, 0..) |c, v| {
                e.gu[v] = try m.function("ink2_exl3_dgu" ++ c.sfx);
                e.down[v] = try m.function("ink2_exl3_ddown" ++ c.sfx);
            }
            // the defaults: the fastest bit-identical build on real slots (tf-inkling2-exl3-dbench, out/prefill/p5/e-decode
            // bench/v3.txt): the swapped operands with 4 warps and 2 k steps ahead, the swapped grid, dependent launches
            e.cfg = for (exl3_dec_configs, 0..) |c, at| {
                if (std.mem.eql(u8, c.sfx, "_t1p2")) break at;
            } else 0;
            e.swap = if (std.c.getenv("TF_INKLING2_EXL3_DEC_SWAP")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
            e.pdl = if (std.c.getenv("TF_INKLING2_EXL3_DEC_PDL")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
            if (std.c.getenv("TF_INKLING2_EXL3_DEC_CFG")) |v| {
                const want = std.mem.span(v);
                e.cfg = for (exl3_dec_configs, 0..) |c, at| {
                    if (std.mem.eql(u8, c.sfx[1..], want)) break at;
                } else return error.Exl3DecConfig;
            }
            return e;
        }
    };

    /// w13's input of every routed pair (ink2_exl3_drot13): row pair / row_slots of x (bf16, stride xs) through the
    /// activation (mode; NVFP4 under row_amax [rows], exl3DecRowAmax's) and the rotation by its slot's suh13 into out
    /// [pair][H] fp16; w2_amax (or 0) [pairs] zeroed for exl3DecGu. Grid (items, H / 128), 32 threads.
    pub fn exl3DecRot13(k: *const Kernels, s: Stream, e: Exl3, x: u64, xs: usize, row_slots: usize, mode: u8, row_amax: u64, p: cuda.grouped.Plan, items: usize, out: u64, w2_amax: u64) !void {
        if (e.hidden % 128 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        a.add(i(xs));
        a.add(i(row_slots));
        a.add(@as(c_int, mode));
        a.add(row_amax);
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(out);
        a.add(w2_amax);
        try go2(k.dec.rot13, s, items, e.hidden / 128, 32, &a);
    }

    /// w13 and its epilogue for windows of up to max_rows rows (ink2_exl3_dgu<cfg>): x [pair][H] fp16 (rotated) ->
    /// act [pair][act_stride] bf16 (exl3Gemm then exl3GateUpEpi's bits); with amax (or 0), each pair's max |act| over
    /// the rank's channels by atomicMax (zeroed by exl3DecRot13). Grid (items, 2 Il / 128).
    pub fn exl3DecGu(k: *const Kernels, s: Stream, e: Exl3, cfg: usize, x: u64, p: cuda.grouped.Plan, items: usize, act: u64, act_stride: usize, amax: u64) !void {
        if ((e.hidden / 16) % 4 != 0 or (2 * e.inter) % 128 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(e.cg);
        a.add(e.cu);
        a.add(act);
        a.add(i(act_stride));
        a.add(amax);
        a.add(@as(c_int, @intFromBool(k.dec.swap)));
        const nb = 2 * e.inter / 128;
        try k.decLaunch(k.dec.gu[cfg], s, if (k.dec.swap) nb else items, if (k.dec.swap) items else nb, 128 * exl3_dec_configs[cfg].grp, &a);
    }

    /// w2's input of every routed pair (ink2_exl3_drot2): act [pair][as] bf16 through the activation (NVFP4: under the
    /// pair's amax, the max over amax_all's world entries [r][amax_stride]) and the rotation by suh2 into out [pair][Il]
    /// fp16. Grid (items, Il / 128), 32 threads.
    pub fn exl3DecRot2(k: *const Kernels, s: Stream, e: Exl3, act: u64, as: usize, mode: u8, amax_all: u64, world: usize, amax_stride: usize, p: cuda.grouped.Plan, items: usize, out: u64) !void {
        if (e.inter % 128 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(act);
        a.add(i(as));
        a.add(@as(c_int, mode));
        a.add(amax_all);
        a.add(i(world));
        a.add(i(amax_stride));
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(out);
        try go2(k.dec.rot2, s, items, e.inter / 128, 32, &a);
    }

    /// w2 and its epilogue for windows of up to max_rows rows (ink2_exl3_ddown<cfg>): x [pair][Il] fp16 (rotated) ->
    /// y [pair][ldy] fp32, every member's row written whole (exl3Gemm then exl3DownEpi's bits, one K part). Grid
    /// (items, H / 128).
    pub fn exl3DecDown(k: *const Kernels, s: Stream, e: Exl3, cfg: usize, x: u64, p: cuda.grouped.Plan, items: usize, y: u64, ldy: usize) !void {
        if ((e.inter / 16) % 4 != 0 or e.hidden % 128 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        for ([_]u64{ p.items, p.counts, p.members, e.slot_ptr, e.slot_k }) |v| a.add(v);
        a.add(i(e.nslots));
        a.add(i(e.hidden));
        a.add(i(e.inter));
        a.add(y);
        a.add(i(ldy));
        a.add(@as(c_int, @intFromBool(k.dec.swap)));
        const nb = e.hidden / 128;
        try k.decLaunch(k.dec.down[cfg], s, if (k.dec.swap) nb else items, if (k.dec.swap) items else nb, 128 * exl3_dec_configs[cfg].grp, &a);
    }

    /// A decode-path GEMM's launch: grid (gx, gy), `block` threads, as a programmatic dependent launch with Exl3Dec.pdl
    /// (the kernel waits before reading its input; inkling2_exl3.cu ink2_dec::dec_tiles).
    fn decLaunch(k: *const Kernels, f: Function, s: Stream, gx: usize, gy: usize, block: usize, a: *Args) !void {
        if (gx == 0 or gy == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(gx), .y = @intCast(gy), .z = 1 }, .block = .{ .x = @intCast(block), .y = 1, .z = 1 }, .shared = 0, .pdl = k.dec.pdl }, s, a);
    }

    /// Each of `rows` rows' max |x| over its `kk` bf16 values (kk % 8 == 0, rows 16-byte aligned: x_stride % 8 == 0) into
    /// amax (fp32 [rows]): exl3RowAmax's values by 16-byte loads (ink2_exl3_drow_amax, one block of 256 a row).
    pub fn exl3DecRowAmax(k: *const Kernels, s: Stream, x: u64, x_stride: usize, rows: usize, kk: usize, amax: u64) !void {
        if (kk % 8 != 0 or x_stride % 8 != 0) return error.Exl3Shape;
        var a: Args = .{};
        a.add(x);
        a.add(i(x_stride));
        a.add(i(kk));
        a.add(amax);
        try go(k.dec.ramax, s, rows, 256, 0, &a);
    }
    // ---- end e-decode

    // ---- x-mixed: the mixed store's item split (TF_INKLING2_MIX=1; forward.moeMix, inkling2.cu's end)

    /// A layer's plan items (`items` / `counts`, expert ids) split by the native bank's map `nmap` (the layer's arena LUT
    /// row): the bank's items with their bank slots into items_n / counts_n, the rest with the EXL3 store's slots
    /// (`xbase` + expert, xbase = layer E) into items_x / counts_x, each list in the plan's order (ink2_mix_split). One
    /// block; device-side counts only, so a captured graph needs no host value.
    pub fn mixSplit(k: *const Kernels, s: Stream, items: u64, counts: u64, nmap: u64, xbase: usize, items_n: u64, counts_n: u64, items_x: u64, counts_x: u64) !void {
        var a: Args = .{};
        for ([_]u64{ items, counts, nmap }) |v| a.add(v);
        a.add(i(xbase));
        for ([_]u64{ items_n, counts_n, items_x, counts_x }) |v| a.add(v);
        try go(k.f.mix_split, s, 1, 1024, 0, &a);
    }
    // ---- end x-mixed
};
