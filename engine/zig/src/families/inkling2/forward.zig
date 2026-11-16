//! Inkling-2's forward on one TP rank for a chunk of T rows at positions len .. len + T - 1 of the model's sequence,
//! in the reference's order (modeling_inkling2 Inkling2TextModel.forward, its decoder layers, Inkling2ForCausalLM's
//! logits). TP exchanges (tp_group all-gathers, then a rank-order sum on every rank): the embedding (vocab-split, one
//! non-zero part a row: exact), o_proj's fp32 partials, the routed experts' fp32 (row, slot) partials before their
//! bf16 rounding, and the head's vocabulary-split logits (a concatenation). Every rank runs every step. At two ranks
//! the gathers go by the group's RoCE writes (Model.open opens them; TF_ROCE=0 keeps NCCL); o_proj's sum is fused
//! with its bf16 rounding and the experts' sum with their combine (the same rank-order fp32 adds, the same bits).
const std = @import("std");
const cuda = @import("cuda");
const model = @import("model.zig");
const kern = @import("kern.zig");
const graphs = @import("graphs.zig");
const dense_exl3 = @import("dense_exl3.zig"); // e-dense

const Model = model.Model;

/// What a chunk asks for: its token ids (host), and whether to read the logits of its last row (`last`) or of every
/// row (`all`, the gates: `out` holds T x unpadded fp32) or none (a prompt chunk before the last).
pub const Want = enum { none, last, all };

/// rows [T][n] bytes `bytes_each` a rank, gathered and summed in rank order into `out` (fp32 when `f32`).
pub fn reduce(m: *Model, mine: u64, all: u64, out: u64, n: usize, is_f32: bool) !void {
    const es: usize = if (is_f32) 4 else 2;
    if (m.world == 1) {
        if (out != mine) {
            var b: cuda.DeviceBuffer = .{ .d = m.d, .ptr = out, .len = n * es, .borrowed = true };
            try b.copyFrom(0, mine, n * es, m.s.handle);
        }
        return;
    }
    try m.group.?.allGather(m.s, mine, all, n * es);
    try m.k.rankSum(m.s, all, out, m.world, n, is_f32);
}

/// An all-gather that always crosses NCCL, never the group's RoCE path: for an exchange a rank may wait on without
/// bound (a follower's next step header while the server idles). The RoCE gather's device wait poisons the path
/// once its spin limit passes; NCCL's waits as long as it takes. Every rank must call it for the same exchange.
pub fn gatherNccl(m: *Model, s: cuda.Stream, mine: u64, all: u64, bytes: usize) !void {
    const g = &m.group.?;
    try g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, s.handle), "ncclAllGather");
}

/// The stream drained, then a failed RoCE exchange's error (its rows hold nothing valid) instead of its rows.
fn drain(m: *Model) !void {
    try m.s.synchronize();
    if (m.group) |*g| try g.check();
}

/// Runs T rows (ids on the host); returns the argmax of the last row (or of every row into `argmax_out`), and with
/// `want == .all` copies every row's fp32 softcapped logits [T][unpadded] into `logits_out`.
pub fn chunk(m: *Model, ids: []const i32, want: Want, logits_out: ?[]f32, argmax_out: ?[]i32) !i32 {
    const T = ids.len;
    if (T == 0 or T > m.prompt_rows) return error.ChunkRows;
    // past max_rows only a prompt chunk: no exact mode (its loads take at most 128 rows' picks), no batched head
    if (T > m.max_rows and (m.exact or (want == .all and logits_out == null))) return error.ChunkRows;
    if (m.state.len + T > m.context) return error.ContextFull;
    // p4-mem: positions past the mapped context mapped first (lazy.zig; every rank at the same step)
    if (m.lazy) |lz| try lz.ensure(m, m.state.len + T);
    const c = &m.c;
    const s = m.s;
    const x = &m.x;
    for (ids) |id| if (id < 0 or id >= c.vocab) return error.TokenOutOfVocabulary;
    try x.ids.uploadAsync(0, std.mem.sliceAsBytes(ids), s.handle);
    if (graphHead(m, T, want, logits_out)) |head| {
        const kk = graphs.Graphs.key(T, head);
        const g = &m.graphs;
        if (g.entries[kk[0]][kk[1]]) |e| {
            try g.replay(m.d, e, s, m.state.len);
            m.state.len += T;
            switch (head) {
                .none => return -1,
                .row => {
                    var am: [1]i32 = undefined;
                    try x.argmax.downloadAsync(0, std.mem.sliceAsBytes(&am), s.handle);
                    if (logits_out) |lo| try x.logits.downloadAsync(0, std.mem.sliceAsBytes(lo[0..c.unpadded_vocab]), s.handle);
                    try s.synchronize();
                    if (argmax_out) |ao| ao[0] = am[0];
                    return am[0];
                },
                .batch => {
                    var am: [graphs.max_rows]i32 = undefined;
                    try x.argmax.downloadAsync(0, std.mem.sliceAsBytes(am[0..T]), s.handle);
                    try s.synchronize();
                    if (argmax_out) |ao| @memcpy(ao[0..T], am[0..T]);
                    return am[T - 1];
                },
                .rows => {
                    // xs: every row's logits [T][unpadded] (the eager window's layout) and argmaxes in one readback
                    var am: [graphs.max_rows]i32 = undefined;
                    try x.argmax.downloadAsync(0, std.mem.sliceAsBytes(am[0..T]), s.handle);
                    try x.rows_f32.downloadAsync(0, std.mem.sliceAsBytes(logits_out.?[0 .. T * c.unpadded_vocab]), s.handle);
                    try s.synchronize();
                    if (argmax_out) |ao| @memcpy(ao[0..T], am[0..T]);
                    return am[T - 1];
                },
            }
        }
        // a key's first forward runs eagerly (its cuBLASLt plans made outside a capture); its graph is captured after
        const r = try eager(m, T, want, logits_out, argmax_out);
        const t0 = graphs.now();
        g.entries[kk[0]][kk[1]] = capture(m, T, head) catch |err| {
            std.log.warn("inkling2 rank {d}: graph capture of {d} rows ({s}) failed ({s}); forwards run eagerly from here", .{ m.rank, T, @tagName(head), @errorName(err) });
            g.drop(m.gpa);
            g.on = false;
            return r;
        };
        g.captures += 1;
        g.capture_ns += graphs.now() -| t0;
        return r;
    }
    return eager(m, T, want, logits_out, argmax_out);
}

/// The chunk's launches and readbacks one by one (ids already staged in x.ids).
fn eager(m: *Model, T: usize, want: Want, logits_out: ?[]f32, argmax_out: ?[]i32) !i32 {
    const c = &m.c;
    const s = m.s;
    const x = &m.x;
    try layers(m, T, m.state.len);
    m.state.len += T;
    if (m.exact) {
        // the last layer's transients unmapped: between chunks the LUT holds only the resident set
        try drain(m);
        m.dropTransient();
        try m.publish();
    }
    if (want == .none and m.spec == null) return -1;
    try readout(m, T);
    if (want == .none) return -1;
    const first: usize = if (want == .all) 0 else T - 1;
    var last: i32 = -1;
    if (want == .all and logits_out == null and T > 1) {
        // argmaxes only (a verify window): one head product over the rows, one gather, one argmax launch
        try headBatch(m, T);
        var am: [64]i32 = undefined;
        try x.argmax.downloadAsync(0, std.mem.sliceAsBytes(am[0..T]), s.handle);
        try drain(m);
        if (argmax_out) |ao| @memcpy(ao[0..T], am[0..T]);
        return am[T - 1];
    }
    for (first..T) |row| {
        try headRow(m, row);
        var am: [1]i32 = undefined;
        try x.argmax.downloadAsync(0, std.mem.sliceAsBytes(&am), s.handle);
        if (logits_out) |lo| {
            try x.logits.downloadAsync(0, std.mem.sliceAsBytes(lo[(row - first) * c.unpadded_vocab ..][0..c.unpadded_vocab]), s.handle);
        }
        try drain(m);
        if (argmax_out) |ao| ao[row - first] = am[0];
        last = am[0];
    }
    return last;
}

/// The graphed form of a chunk, or null for the eager path: graphs on, not exact mode (its host round trip a layer),
/// at most graphs.max_rows rows, and a head whose readback is the argmaxes and at most one row's logits; xs: or every
/// row's logits (a sampled verify window) as the eager window computes them, under Model.sampled_graph.
fn graphHead(m: *const Model, T: usize, want: Want, logits_out: ?[]f32) ?graphs.Head {
    if (!m.graphs.on or m.exact or T > m.graphs.rows) return null; // e-step: Graphs.rows (8; 16 with copy windows)
    return switch (want) {
        .none => .none,
        .last => .row,
        .all => if (T == 1) .row else if (logits_out == null) .batch else if (m.sampled_graph) .rows else null,
    };
}

/// A chunk's device work at `head` captured as a graph (nothing runs; the replay follows).
fn capture(m: *Model, T: usize, head: graphs.Head) !*graphs.Entry {
    const s = m.s;
    try s.synchronize();
    try cuda.graph.beginCapture(s, .thread_local);
    body(m, T, head) catch |err| {
        if (cuda.graph.endCapture(s)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return err;
    };
    const k = &m.k;
    // p4-dattn: with decode_attn a layer's attention is ink2_fdec_d* (its position argument 8) and the merge (none):
    // still one position node an attention, 6 a layer
    // e-step: with TF_INKLING2_STEP_FUSE the four head preps are one ink2_es_head_prep4 (its position argument 2): 3 a layer
    const pos_args = [_]graphs.PosArg{ .{ .f = k.f.head_prep, .index = 8 }, .{ .f = k.f.cache_write, .index = 6 }, .{ .f = k.f.attention, .index = 11 }, .{ .f = k.f.fdec[0], .index = 8 }, .{ .f = k.f.fdec[1], .index = 8 }, .{ .f = k.es.head_prep4, .index = 2 } };
    return graphs.Graphs.finish(m.gpa, m.d, s, &pos_args, (if (stepFused(m, T)) @as(usize, 3) else 6) * m.c.layers, m.state.len);
}

/// The eager chunk's launches at `head`, none of its host work: what a graph holds.
fn body(m: *Model, T: usize, head: graphs.Head) !void {
    try layers(m, T, m.state.len);
    if (head == .none and m.spec == null) return;
    try readout(m, T);
    switch (head) {
        .none => {},
        .row => try headRow(m, T - 1),
        .batch => try headBatch(m, T),
        // xs: the eager window's per-row head on every row, each into its own slot (the same launches, the same bits)
        .rows => for (0..T) |r| try headRowTo(m, r, m.x.rows_f32.ptr + r * m.c.unpadded_vocab * 4, m.x.argmax.ptr + r * 4),
    }
}

/// The embedding and every layer over the chunk's T rows (ids staged in x.ids) at positions pos0 ..
fn layers(m: *Model, T: usize, pos0: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    const hd: usize = c.head_dim;
    const hq = m.hq();
    const hkv = m.hkv();
    const qw = hq * hd;
    const kvw = hkv * hd;
    const K: usize = c.conv;
    // the rings' histories (speculation's rollback) hold max_rows rows: only a verify window is ever cut back, so a
    // wider prompt chunk records none (a chunk's history is its own; the next one starts it over)
    const hist_on = T <= m.max_rows;
    // a prompt chunk wider than max_rows on its own kernels (Model.prefill_kernels): the same bits
    const wide = T > m.max_rows and m.prefill_kernels;
    // p4-small (TF_INKLING2_FUSE, Model.fuse): a wide chunk's memory-bound kernels fused, the same bits
    const fused = wide and m.fuse;
    // e-step (TF_INKLING2_STEP_FUSE, Model.step_fuse): a decode / verify chunk's too (stepFused), the same bits
    const sfused = stepFused(m, T);
    // embedding (M:1218-1224): this rank's vocabulary rows, summed over ranks (exact), then embed_norm
    if (m.canon) {
        // canonical TP: the pair's two vocabulary halves (one row of the table each, zeros else), summed in rank order
        const mid = c.vocab / 2;
        try k.embed(s, x.ids.ptr, m.embed, x.emb_all.ptr, T, H, 0, mid);
        try k.embed(s, x.ids.ptr, m.embed + mid * H * 2, x.emb_all.ptr + T * H * 2, T, H, mid, c.vocab);
        try k.rankSum(s, x.emb_all.ptr, x.emb.ptr, 2, T * H, false);
    } else {
        try k.embed(s, x.ids.ptr, m.embed, x.emb.ptr, T, H, m.vocab_lo, m.vocab_lo + m.vocabLocal());
        try reduce(m, x.emb.ptr, x.emb_all.ptr, x.emb.ptr, T * H, false);
    }
    try k.rmsnorm(s, x.emb.ptr, false, m.embed_norm, x.h.ptr, T, H, c.eps);
    var nblk: usize = 0;
    const smax = m.smax();
    for (m.layers, 0..) |*l, li| {
        if (m.spec) |sp| for (sp.tap_layers, 0..) |tl, ti| if (tl == li) try sp.tap(m, x.h.ptr, T, ti);
        // the attention's inputs (M:782-793 / M:840-857)
        if ((fused or sfused) and l.nq > 0 and (l.local or l.nq == 3)) { // e-step: sfused too
            // p4-small (TF_INKLING2_FUSE): each query's block attention residual and its input norm in one launch
            const w3 = if (l.local) [3]u64{ l.norm_attn, 0, 0 } else [3]u64{ l.norm_q, l.norm_k, l.norm_v };
            try k.attnResNorm(s, x.blocks.ptr, nblk, smax, x.h.ptr, .bf16, 0, false, l.attn_query, l.nq, w3, .{ x.xq.ptr, x.xk.ptr, x.xv.ptr }, T, H, c.eps);
        } else {
            var src_q: u64 = x.h.ptr;
            var src_q_f32 = false;
            var src_k: u64 = x.h.ptr;
            var src_v: u64 = x.h.ptr;
            if (l.nq > 0) {
                try k.attnRes(s, x.blocks.ptr, nblk, smax, x.h.ptr, false, l.attn_query, l.nq, x.ar.ptr, T, H);
                src_q = x.ar.ptr;
                src_q_f32 = true;
                src_k = if (l.nq == 3) x.ar.ptr + T * H * 4 else x.ar.ptr;
                src_v = if (l.nq == 3) x.ar.ptr + 2 * T * H * 4 else x.ar.ptr;
            }
            if (l.local) {
                try k.rmsnorm(s, src_q, src_q_f32, l.norm_attn, x.xq.ptr, T, H, c.eps);
            } else {
                try k.rmsnorm(s, src_q, src_q_f32, l.norm_q, x.xq.ptr, T, H, c.eps);
                try k.rmsnorm(s, src_k, src_q_f32, l.norm_k, x.xk.ptr, T, H, c.eps);
                try k.rmsnorm(s, src_v, src_q_f32, l.norm_v, x.xv.ptr, T, H, c.eps);
            }
        }
        const xk = if (l.local) x.xq.ptr else x.xk.ptr;
        const xv = if (l.local) x.xq.ptr else x.xv.ptr;
        // e-dense: the calibration Hessians of the attention inputs (the run tool's --hcap only)
        if (m.hcap) |hc| {
            try hc.add(s, li, .q, x.xq.ptr, T, H);
            if (!l.local) {
                try hc.add(s, li, .k, x.xk.ptr, T, H);
                try hc.add(s, li, .v, x.xv.ptr, T, H);
            }
        }
        // projections (bf16 nn.Linear): q, gate, r, sinks from the q side; k, v from their own sides; e-dense: q, k, v,
        // r and gate through the active dense EXL3 set's weights where it has them (projx; p4-small's FUSE writes the
        // same x.xq / xk / xv rows as the unfused norms, so both read them as they are)
        const lx = dxLayer(m, li);
        if (try projGroup(m, lx, l.local, T)) {
            // e-step: TF_INKLING2_DENSE_GROUP=1: q, r, gate (and on a sliding layer k, v) in one group: the same bits
            if (!l.local) {
                try projx(m, xk, l.wk, if (lx) |a| a.wk else null, x.k.ptr, T, kvw, H);
                try projx(m, xv, l.wv, if (lx) |a| a.wv else null, x.v.ptr, T, kvw, H);
            }
        } else {
            try projx(m, x.xq.ptr, l.wq, if (lx) |a| a.wq else null, x.q.ptr, T, qw, H);
            try projx(m, xk, l.wk, if (lx) |a| a.wk else null, x.k.ptr, T, kvw, H);
            try projx(m, xv, l.wv, if (lx) |a| a.wv else null, x.v.ptr, T, kvw, H);
            try projx(m, x.xq.ptr, l.wr, if (lx) |a| a.wr else null, x.r.ptr, T, hq * c.d_rel, H);
            try projx(m, x.xq.ptr, l.wgate, if (lx) |a| a.wgate else null, x.gate.ptr, T, qw, H);
        }
        try proj(m, x.xq.ptr, l.wsink, x.sinks.ptr, T, hq, H);
        // the k / v short convolutions (in place)
        if (l.local and sfused) {
            // e-step: both sconvStatic calls in one launch (the same bits)
            try k.sconvStatic2(s, x.k.ptr, l.kconv, m.state.ring_k[li].ptr, if (hist_on) m.hist(li, 0) else 0, x.v.ptr, l.vconv, m.state.ring_v[li].ptr, if (hist_on) m.hist(li, 1) else 0, T, kvw, K);
        } else if (l.local) {
            try sconv(m, wide, x.k.ptr, l.kconv, m.state.ring_k[li].ptr, T, kvw, if (hist_on) m.hist(li, 0) else 0);
            try sconv(m, wide, x.v.ptr, l.vconv, m.state.ring_v[li].ptr, T, kvw, if (hist_on) m.hist(li, 1) else 0);
        } else {
            try proj(m, xk, l.kproj, x.dyn.ptr, T, hkv * K, H);
            try k.toF32(s, x.dyn.ptr, x.dyn32.ptr, T * hkv * K);
            if (wide) {
                try k.sconvDynamicRows(s, x.k.ptr, l.kstatic, x.dyn32.ptr, m.state.ring_k[li].ptr, x.edges.ptr, T, kvw, K, hd);
            } else try k.sconvDynamic(s, x.k.ptr, x.k.ptr, l.kstatic, x.dyn32.ptr, m.state.ring_k[li].ptr, T, kvw, K, hd, if (hist_on) m.hist(li, 0) else 0);
            try proj(m, xv, l.vproj, x.dyn.ptr, T, hkv * K, H);
            try k.toF32(s, x.dyn.ptr, x.dyn32.ptr, T * hkv * K);
            if (wide) {
                try k.sconvDynamicRows(s, x.v.ptr, l.vstatic, x.dyn32.ptr, m.state.ring_v[li].ptr, x.edges.ptr, T, kvw, K, hd);
            } else try k.sconvDynamic(s, x.v.ptr, x.v.ptr, l.vstatic, x.dyn32.ptr, m.state.ring_v[li].ptr, T, kvw, K, hd, if (hist_on) m.hist(li, 1) else 0);
        }
        // clips and norms (M:465-470 / M:595-602)
        const tau = !l.local;
        if (sfused) {
            // e-step: the four calls in one launch (ink2_es_head_prep4: each block one call's, the same bits)
            try k.headPrep4(s, .{
                .{ .v = x.q.ptr, .scale = l.qscale, .heads = hq, .d = hd, .mode = .scaled_clip, .thr = c.qk_clip, .tau = tau, .alpha = c.log_alpha, .n_floor = c.log_floor },
                .{ .v = x.k.ptr, .scale = 0, .heads = hkv, .d = hd, .mode = .clip, .thr = c.qk_clip, .tau = false, .alpha = 0, .n_floor = 1 },
                .{ .v = x.v.ptr, .scale = 0, .heads = hkv, .d = hd, .mode = .rms, .thr = 0, .tau = false, .alpha = 0, .n_floor = 1 },
                .{ .v = x.r.ptr, .scale = l.relscale, .heads = hq, .d = c.d_rel, .mode = .scaled_clip, .thr = c.rel_clip, .tau = tau, .alpha = c.log_alpha, .n_floor = c.log_floor },
            }, T, c.eps, pos0);
        } else {
            try k.headPrep(s, x.q.ptr, l.qscale, T, hq, hd, .scaled_clip, c.qk_clip, c.eps, tau, pos0, c.log_alpha, c.log_floor);
            try k.headPrep(s, x.k.ptr, 0, T, hkv, hd, .clip, c.qk_clip, c.eps, false, pos0, 0, 1);
            try k.headPrep(s, x.v.ptr, 0, T, hkv, hd, .rms, 0, c.eps, false, pos0, 0, 1);
            try k.headPrep(s, x.r.ptr, l.relscale, T, hq, c.d_rel, .scaled_clip, c.rel_clip, c.eps, tau, pos0, c.log_alpha, c.log_floor);
        }
        // the chunk's keys and values into the cache, the relative bias rows, the attention
        const slots = m.state.slots[li];
        try k.cacheWrite(s, x.k.ptr, x.v.ptr, m.state.kc[li].ptr, m.state.vc[li].ptr, T, kvw, pos0, slots);
        // the relative bias rows, max_rows rows a product (a wider prompt chunk in pieces: each the product a chunk of
        // max_rows makes), then the attention: a row's attention reads only the caches, written for the chunk above
        const rel_row = hq * l.extent * 2;
        // ---- p4-small (TF_INKLING2_REL_WIDE): a wide chunk's bias rows in one product (1), or none at all (2: the
        // tiled scores kernel makes each bias it reads from the r rows and the projection, attentionTiledRel)
        const rel_in_scores = T > m.max_rows and m.rel_wide == 2 and wide and c.d_rel == kern.Kernels.rel_dr and
            ((m.prefill_attn == .tiled and kern.Kernels.tiledFits(hq, hkv, hd)) or (m.prefill_attn == .fa and kern.Kernels.faFits(hq, hkv, hd)));
        if (T > m.max_rows and m.rel_wide != 0) {
            if (!rel_in_scores) try relRows(m, l, 0, T); // x-merge: relRows = the product, or canonical TP's halves
        } else {
            // ---- end p4-small
            var r0: usize = 0;
            while (r0 < T) : (r0 += m.max_rows) {
                const n = @min(m.max_rows, T - r0);
                try relRows(m, l, r0, n);
            }
        }
        // a launch's rows each take `cap` floats of the score scratch: a verify / decode chunk the context's (one
        // launch, as captured in graphs), a wider prompt chunk only the most keys its rows read (the window, or its
        // last position on a global layer), so as many rows a launch as the scratch then holds (at least max_rows)
        const static_cap = m.scoreCap(); // p4-mem: max(context, window + max_rows), or the mapped positions' (lazy.zig)
        const cap = if (T <= m.max_rows) static_cap else if (l.local) @min(c.window, pos0 + T) else pos0 + T;
        // TF_INKLING2_PREFILL_ATTN=tc (opt-in, not bitwise): a wide chunk's attention on tensor cores, every row in
        // one launch (the kernel makes its scores again for each pass instead of keeping them: no score scratch).
        // Else the tiled kernels (the default; scores kept as bf16 at tiledStride(cap) elements a (row, head): twice
        // the rows a launch), else each kv head's keys and values staged once for its query heads (the same bits)
        const tc = wide and m.prefill_attn == .tc and kern.Kernels.tcFits(hq, hkv, hd);
        const fa = wide and m.prefill_attn == .fa and kern.Kernels.faFits(hq, hkv, hd); // p4-small: tc's online form
        const tiled = wide and m.prefill_attn == .tiled and kern.Kernels.tiledFits(hq, hkv, hd);
        const per = if (T <= m.max_rows or tc or fa) T else if (tiled) @max(m.max_rows, x.scores.len / (hq * kern.Kernels.tiledStride(cap) * 2)) else @max(m.max_rows, x.scores.len / (hq * cap * 4));
        const gqa = wide and kern.Kernels.gqaFits(hq, hkv, hd);
        // p4-dattn: TF_INKLING2_DECODE_ATTN=1: a chunk of at most max_rows rows on the split-K flash-decoding kernels
        // (one launch: the score scratch unused; not bitwise)
        const fdec = T <= m.max_rows and m.decode_attn and kern.Kernels.fdecFits(T, hq, hkv, hd);
        var a0: usize = 0;
        while (a0 < T) : (a0 += per) {
            const n = @min(per, T - a0);
            const p: kern.Kernels.Attn = .{ .q = x.q.ptr + a0 * qw * 2, .kc = m.state.kc[li].ptr, .vc = m.state.vc[li].ptr, .rel = x.rel.ptr + a0 * rel_row, .sinks = x.sinks.ptr + a0 * hq * 2, .gate = x.gate.ptr + a0 * qw * 2, .out = x.attn.ptr + a0 * qw * 2, .scores = x.scores.ptr, .heads = hq, .kv_heads = hkv, .d = hd, .q0 = pos0 + a0, .window = if (l.local) c.window else 0, .slots = slots, .extent = l.extent, .scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .cap = cap };
            if (fdec) {
                try k.attentionFdec(s, n, p, m.fd_part.ptr, m.fd_splits[@intFromBool(l.local)]);
            } else if (tc) {
                try k.attentionTc(s, n, p);
            } else if (fa) {
                // p4-small (TF_INKLING2_PREFILL_ATTN=fa; with REL_WIDE=2 the bias made in the kernel)
                if (rel_in_scores) {
                    try k.attentionFaRel(s, n, p, x.r.ptr + a0 * hq * c.d_rel * 2, l.relproj, c.d_rel);
                } else try k.attentionFa(s, n, p);
            } else if (rel_in_scores) {
                // p4-small: the launch's r rows [n][hq][d_rel] and the layer's projection in place of the bias rows
                try k.attentionTiledRel(s, n, p, x.r.ptr + a0 * hq * c.d_rel * 2, l.relproj, c.d_rel);
            } else if (tiled) {
                try k.attentionTiled(s, n, p);
            } else if (gqa) try k.attentionGqa(s, n, p) else try k.attention(s, n, p);
        }
        // o_proj: fp32 partials over this rank's heads, summed, rounded once to bf16 (e-dense: through the active
        // dense EXL3 set's wo_ud where it has it, this rank's 128-row blocks of K; its Hessian first, --hcap only)
        if (m.hcap) |hc| try hc.addO(s, if (m.group) |*g| g else null, m.world, li, x.attn.ptr, T, qw);
        const wo_x: ?dense_exl3.Weight = if (lx) |a| a.wo else null;
        if (m.canon) {
            // canonical TP: the pair's two half-K products (its ranks' heads and wo columns) added in rank order
            const half = qw / 2;
            for (0..2) |r| {
                try model.copy2d(m, x.canon_tmp.ptr, half * 2, x.attn.ptr + r * half * 2, qw * 2, half * 2, T);
                if (wo_x) |e| {
                    try m.dx.?.run(s, try dense_exl3.half(e, .rows, r), x.canon_tmp.ptr, T, x.o_all.ptr + r * T * H * 4, true);
                } else try m.g.run(s, x.canon_tmp.ptr, l.wo_half[r], x.o_all.ptr + r * T * H * 4, T, H, half, true);
            }
            try k.rankSumBf16(s, x.o_all.ptr, x.o.ptr, 2, T * H);
        } else if (wo_x) |e| {
            try m.dx.?.run(s, e, x.attn.ptr, T, x.o32.ptr, true);
        } else try m.g.run(s, x.attn.ptr, l.wo, x.o32.ptr, T, H, qw, true);
        if (m.canon) {} else if (wide and m.world == 2 and m.split_moe) {
            try oprojSplit(m, T);
        } else if (m.world > 1 and sfused) {
            // e-step: the gather, then the rank-order sum, its rounding and the attention's conv in one launch
            try m.group.?.allGather(s, x.o32.ptr, x.o_all.ptr, T * H * 4);
            try k.rankSumSconv(s, x.o_all.ptr, m.world, l.conv_attn, m.state.ring_attn[li].ptr, if (hist_on) m.hist(li, 2) else 0, x.o.ptr, T, H, K);
        } else if (m.world > 1) {
            // the gather, then the rank-order fp32 sum and its bf16 rounding in one launch (the same bits)
            try m.group.?.allGather(s, x.o32.ptr, x.o_all.ptr, T * H * 4);
            try k.rankSumBf16(s, x.o_all.ptr, x.o.ptr, m.world, T * H);
        } else {
            try reduce(m, x.o32.ptr, x.o_all.ptr, x.o32.ptr, T * H, true);
            try k.toBf16(s, x.o32.ptr, x.o.ptr, T * H);
        }
        if (!(m.world > 1 and sfused and !m.canon)) try sconv(m, wide, x.o.ptr, l.conv_attn, m.state.ring_attn[li].ptr, T, H, if (hist_on) m.hist(li, 2) else 0); // e-step: done above
        // block state and residual (M:798-802)
        if (l.boundary) {
            try k.blockPush(s, x.h.ptr, x.blocks.ptr, T, H, nblk, smax);
            nblk += 1;
        }
        if (fused) {
            // p4-small (TF_INKLING2_FUSE): the residual (x.res) never written: the MoE's input made from the attention
            // output and the layer input on the fly with its block residual and norm in one launch, and the MLP's
            // short convolution writing the layer's output (each ink2_residual's add, then ink2_layer_out's)
            try k.attnResNorm(s, x.blocks.ptr, nblk, smax, x.o.ptr, .residual, x.h.ptr, l.boundary, l.mlp_query, 1, .{ l.norm_mlp, 0, 0 }, .{ x.xm.ptr, 0, 0 }, T, H, c.eps);
            if (m.hcap) |hc| try hc.add(s, li, .m, x.xm.ptr, T, H); // e-dense: the router's input (--hcap only)
            try moe(m, li, T);
            try k.sconvStaticRowsOut(s, x.mo.ptr, l.conv_mlp, m.state.ring_mlp[li].ptr, x.edges.ptr, x.o.ptr, x.h.ptr, l.boundary, T, H, c.conv);
            continue;
        }
        if (sfused) {
            // e-step: FUSE's MoE input (the residual made on the fly, x.res never written) and the MLP's conv writing the
            // layer output with its history (ink2_es_sconv_static_out): the same bits as the launches below
            try k.attnResNorm(s, x.blocks.ptr, nblk, smax, x.o.ptr, .residual, x.h.ptr, l.boundary, l.mlp_query, 1, .{ l.norm_mlp, 0, 0 }, .{ x.xm.ptr, 0, 0 }, T, H, c.eps);
            try moe(m, li, T);
            try k.sconvStaticOut(s, x.mo.ptr, l.conv_mlp, m.state.ring_mlp[li].ptr, x.o.ptr, x.h.ptr, l.boundary, T, H, c.conv, if (hist_on) m.hist(li, 3) else 0);
            continue;
        }
        try k.residual(s, x.o.ptr, x.h.ptr, x.res.ptr, l.boundary, T * H);
        // the MoE's input (M:804-805)
        try k.attnRes(s, x.blocks.ptr, nblk, smax, x.res.ptr, true, l.mlp_query, 1, x.ar.ptr, T, H);
        try k.rmsnorm(s, x.ar.ptr, true, l.norm_mlp, x.xm.ptr, T, H, c.eps);
        if (m.hcap) |hc| try hc.add(s, li, .m, x.xm.ptr, T, H); // e-dense: the router's input (--hcap only)
        try moe(m, li, T);
        try sconv(m, wide, x.mo.ptr, l.conv_mlp, m.state.ring_mlp[li].ptr, T, H, if (hist_on) m.hist(li, 3) else 0);
        try k.layerOut(s, x.res.ptr, x.mo.ptr, x.h.ptr, T * H);
    }
}

/// The final readout and norm (M:1311-1312) of the chunk's rows into x.xm; with speculation, every chunk's final-norm
/// rows are the draft's last tap.
fn readout(m: *Model, T: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    var nblk: usize = 0; // the blocks pushed by the layers' boundaries
    for (m.layers) |l| nblk += @intFromBool(l.boundary);
    if ((T > m.max_rows and m.prefill_kernels and m.fuse) or stepFused(m, T)) { // e-step: a narrow chunk too
        // p4-small (TF_INKLING2_FUSE): the final block residual and norm in one launch (the same bits)
        try k.attnResNorm(s, x.blocks.ptr, nblk, m.smax(), x.h.ptr, .bf16, 0, false, m.final_query, 1, .{ m.norm, 0, 0 }, .{ x.xm.ptr, 0, 0 }, T, H, c.eps);
    } else {
        try k.attnRes(s, x.blocks.ptr, nblk, m.smax(), x.h.ptr, false, m.final_query, 1, x.fin.ptr, T, H);
        try k.rmsnorm(s, x.fin.ptr, true, m.norm, x.xm.ptr, T, H, c.eps);
    }
    if (m.hcap) |hc| try hc.addHead(s, x.xm.ptr, T, H); // e-dense: the head's input (--hcap only)
    if (m.spec) |sp| try sp.tap(m, x.xm.ptr, T, sp.tap_layers.len);
}

/// Argmaxes only (a verify window): one head product over the rows, one gather, one argmax launch into x.argmax[0 .. T].
fn headBatch(m: *Model, T: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    if (dxHead(m)) |set| return headBatchX(m, set, T); // e-dense
    const vl = m.vocabLocal();
    if (m.canon) {
        // canonical TP: the pair's two vocabulary halves at the rank's shape, concatenated as the gather does
        const hv = c.vocab / 2;
        for (0..2) |r| try m.g.run(s, x.xm.ptr, m.unembed + r * hv * H * 2, x.head_rows_all.ptr + r * T * hv * 2, T, hv, H, false);
        try k.unsplit(s, x.head_rows_all.ptr, x.head_cat.ptr, T, hv, 2);
        return k.logits(s, x.head_cat.ptr, T, c.vocab, c.unpadded_vocab, c.logit_cap, 0, x.argmax.ptr);
    }
    try m.g.run(s, x.xm.ptr, m.unembed, x.head_rows.ptr, T, vl, H, false);
    if (m.world > 1 and stepFused(m, T)) return headArgmax(m, x.head_rows.ptr, T, vl, m.vocab_lo); // e-step
    if (m.world > 1) {
        try m.group.?.allGather(s, x.head_rows.ptr, x.head_rows_all.ptr, T * vl * 2);
        try k.unsplit(s, x.head_rows_all.ptr, x.head_cat.ptr, T, vl, m.world);
    } else try x.head_cat.copyFrom(0, x.head_rows.ptr, T * vl * 2, s.handle);
    try k.logits(s, x.head_cat.ptr, T, c.vocab, c.unpadded_vocab, c.logit_cap, 0, x.argmax.ptr);
}

/// The head on row `row`: its fp32 softcapped logits in x.logits, its argmax in x.argmax[0].
fn headRow(m: *Model, row: usize) !void {
    return headRowTo(m, row, m.x.logits.ptr, m.x.argmax.ptr);
}

/// headRow with its outputs at `out_f32` (fp32 [unpadded]) and `am` (one i32): xs, a sampled window's rows in slots.
fn headRowTo(m: *Model, row: usize, out_f32: u64, am: u64) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    if (dxHead(m)) |set| return headRowX(m, set, row, out_f32, am); // e-dense
    if (m.canon) {
        // canonical TP: the pair's two vocabulary halves at the rank's shape, one after the other (the gather's order)
        const hv = c.vocab / 2;
        for (0..2) |r| try m.g.run(s, x.xm.ptr + row * H * 2, m.unembed + r * hv * H * 2, x.head_all.ptr + r * hv * 2, 1, hv, H, false);
        return k.logits(s, x.head_all.ptr, 1, c.vocab, c.unpadded_vocab, c.logit_cap, out_f32, am);
    }
    try m.g.run(s, x.xm.ptr + row * H * 2, m.unembed, x.head.ptr, 1, m.vocabLocal(), H, false);
    if (m.world > 1) {
        try m.group.?.allGather(s, x.head.ptr, x.head_all.ptr, m.vocabLocal() * 2);
    } else {
        try x.head_all.copyFrom(0, x.head.ptr, m.vocabLocal() * 2, s.handle);
    }
    try k.logits(s, x.head_all.ptr, 1, c.vocab, c.unpadded_vocab, c.logit_cap, out_f32, am);
}

/// The routed experts of layer `li` on x.xm (bf16 [T][H]) into x.mo (bf16 [T][H]): masked router (+ demand), the
/// plan over expert ids, items to arena slots, gate-up (softcapped SwiGLU), down (fp32 partials), TP sum, combine.
/// x-merge (x-mixed DESIGN 2.7): the native forms in nativeUp / nativeExperts, the TP sum and combine in moeTail
/// (shared with moeExl3); every path's launches and their order are phase 4's.
fn moe(m: *Model, li: usize, T: usize) !void {
    if (m.mix) return moeMix(m, li, T); // x-mixed: the mixed store (TF_INKLING2_MIX=1)
    if (m.layers[li].fmt == .exl3) return moeExl3(m, li, T);
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const l = &m.layers[li];
    const E: usize = c.experts;
    const topk: usize = c.topk;
    const pairs = T * topk;
    try router(m, li, T); // e-dense: the active dense EXL3 set's router where it has one, else the bf16 product
    try routeRows(m, l, li, T); // e-step: ink2_route, or its compact form (the same bits) under TF_INKLING2_STEP_FUSE
    try k.demand(s, x.want_ids.ptr, x.want_sc.ptr, m.arena.routable.ptr + li * E, pairs, x.demand.ptr + li * E * 3 * 4);
    if (m.exact) try exactLoads(m, li, T);
    const prompt = T > 1;
    // p3-expert: a wide prompt chunk's MXFP8 pairs in items of up to 128 on the wide form (Model.prefill_experts)
    const wide_mx = T > m.max_rows and m.prefill_experts and c.act[li] == .mxfp8;
    const tile: usize = if (wide_mx) kern.wide_tile else if (prompt) kern.prompt_tile else kern.decode_tile;
    const plan: cuda.grouped.Plan = .{ .members = x.members.ptr, .items = x.items.ptr, .counts = x.counts.ptr, .rank = x.plan_rank.ptr, .hist = x.plan_hist.ptr };
    try k.router.route(s, x.picks.ptr, pairs, E, tile, plan);
    try k.itemsToSlots(s, x.items.ptr, x.counts.ptr, m.arena.lut.ptr + li * E * 4);
    const items = cuda.grouped.maxItems(pairs, E, tile);
    // the overlapped pair exchange runs the down in column parts beside the exchange (moeOverlap); every other form
    // runs it whole into x.y, then the tail
    if (overlapParts(m, T)) |n| return moeOverlap(m, T, try nativeUp(m, li, T, plan, items, prompt, wide_mx), plan, items, n, null);
    try zeroPairs(m, T);
    try nativeExperts(m, li, T, plan, items, prompt, wide_mx);
    return moeTail(m, T);
}

/// The native forms' gate-up of a layer's routed pairs on the arena's slots, the layer's release format (Config.act):
/// p4-mx's W8A8 input (TF_INKLING2_MX_ACT, MXFP8 layers) and the gate-up (softcapped SwiGLU) into x.act. Returns the
/// down's descriptor (nativeExperts, moeOverlap). x-merge: moe's body, factored (x-mixed DESIGN 2.7).
fn nativeUp(m: *Model, li: usize, T: usize, plan: cuda.grouped.Plan, items: usize, prompt: bool, wide_mx: bool) !kern.Kernels.Experts {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    const a = &m.arena;
    const fmt = c.act[li];
    // ---- p4-mx: TF_INKLING2_MX_ACT=1 runs an MXFP8 layer's experts in the checkpoint's W8A8 numerics: the post-norm
    // rows quantized to MXFP8 (q8 rows in x.mxq), the gate-up's act quantized by its epilogue into x.act (q8 rows),
    // the down on those (kern.expertsQ8 / downColsQ8; every other step of the layer unchanged)
    const mx_act = fmt == .mxfp8 and m.mx_act;
    if (mx_act) try k.mx8Quant(s, x.xm.ptr, H, T, H, x.mxq.ptr);
    // ---- end p4-mx
    const ex: kern.Kernels.Experts = switch (fmt) {
        .nvfp4 => .{ .fmt = .nvfp4, .w = a.nv_up.ptr, .scale = a.nv_up_scale.ptr, .slots = a.nv_slots, .hidden = H, .inter = m.inter(), .cg = c.gate_cap, .cu = c.up_cap },
        .mxfp8 => .{ .fmt = .mxfp8, .w = a.mx_up.ptr, .scale = 0, .slots = a.mx_slots, .hidden = H, .inter = m.inter(), .cg = c.gate_cap, .cu = c.up_cap, .wide = wide_mx, .act8 = mx_act },
        .exl3 => return error.Exl3Layer, // Config.act holds release formats only
    };
    try k.experts(s, true, ex, if (mx_act) x.mxq.ptr else x.xm.ptr, H, c.topk, plan, items, prompt, x.act.ptr);
    var dn = ex;
    dn.w = if (fmt == .nvfp4) a.nv_down.ptr else a.mx_down.ptr;
    dn.scale = if (fmt == .nvfp4) a.nv_down_scale.ptr else 0;
    return dn;
}

/// The native forms of a layer's routed pairs: nativeUp's gate-up, then the down's fp32 pair rows of this rank's
/// channels into x.y (what moeTail reads; the caller zeroes the chunk's rows first where zeroPairs' rule asks, once
/// for every form that writes them). x-merge: moe's body, factored (x-mixed DESIGN 2.7).
fn nativeExperts(m: *Model, li: usize, T: usize, plan: cuda.grouped.Plan, items: usize, prompt: bool, wide_mx: bool) !void {
    const dn = try nativeUp(m, li, T, plan, items, prompt, wide_mx);
    try m.k.experts(m.s, false, dn, m.x.act.ptr, m.inter(), 0, plan, items, prompt, m.x.y.ptr);
}

/// A chunk's pair rows of x.y zeroed before its down forms: not for a wide prompt chunk (its down forms write every
/// routed pair's row whole: each pick is routable, so resident with a slot, and the combine reads routed pairs only),
/// nor with p4-xchg's token rows (TF_INKLING2_MOE_XCHG=token at world > 1: they read routed pairs only, and nothing
/// gathers y); else the chunk's pairs only (y holds a wide prompt chunk's rows, growPrompt: a decode / verify forward
/// zeroing all of it would write ~150 MB a layer for nothing). x-merge: moe's and moeExl3's rule, in one place, called
/// before the expert forms (their gate-ups never touch y: the same bits as between the gate-up and the down).
fn zeroPairs(m: *Model, T: usize) !void {
    if (T > m.max_rows or (m.moe_xchg == .token and m.world > 1)) return;
    var rows: cuda.DeviceBuffer = .{ .d = m.d, .ptr = m.x.y.ptr, .len = T * m.c.topk * m.c.hidden * 4, .borrowed = true };
    try rows.fill8(0, m.s.handle);
}

/// The column parts of the overlapped pair exchange (moeOverlap) for a chunk of T rows, or null: a wide chunk at two
/// ranks with the split exchange and its comm stream (Model.ov), the pair exchange (p4-xchg's token rows run their
/// down whole), and at least two parts. x-merge: moe's and moeExl3's condition, in one place.
fn overlapParts(m: *const Model, T: usize) ?usize {
    if (m.moe_xchg == .token and m.world > 1) return null;
    if (!(T > m.max_rows and m.world == 2 and m.split_moe and m.ov != null and m.c.hidden % kern.Kernels.part_cols == 0)) return null;
    const n = overlapPartsFor(m, T);
    return if (n >= 2) n else null;
}

/// The TP sum and the combine of a layer's routed pairs from x.y (each routed pair's fp32 partial of this rank's
/// channels, whichever form wrote it) into x.mo: p4-xchg's token rows (TF_INKLING2_MOE_XCHG=token at world > 1,
/// moeTokRows); else a wide chunk's row split at two ranks (moeSplit); else the gather and each pair's rank-order
/// fp32 sum inside the combine (the same bits, one launch fewer); else (one rank) the sum and the combine.
/// x-merge: moe's and moeExl3's tail, one function (x-mixed DESIGN 2.7).
fn moeTail(m: *Model, T: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    const topk: usize = c.topk;
    const pairs = T * topk;
    if (m.moe_xchg == .token and m.world > 1) return moeTokRows(m, T); // p4-xchg
    if (T > m.max_rows and m.world == 2 and m.split_moe) return moeSplit(m, T);
    if (m.world > 1) {
        // the gather, then each pair's rank-order fp32 sum inside the combine (the same bits, one launch fewer)
        try m.group.?.allGather(s, x.y.ptr, x.y_all.ptr, pairs * H * 4);
        return k.combineParts(s, x.y_all.ptr, x.picks.ptr, x.wts.ptr, x.mo.ptr, T, topk, H, m.world);
    }
    try reduce(m, x.y.ptr, x.y_all.ptr, x.ysum.ptr, pairs * H, true);
    try k.combine(s, x.ysum.ptr, x.picks.ptr, x.wts.ptr, x.mo.ptr, T, topk, H);
}

/// The routed experts of an EXL3 layer (the kortexa-ai export: exl3.zig, inkling2_exl3.cu) on x.xm into x.mo: the
/// router, plan, items-to-slots, TP sum and combine of `moe` unchanged (every expert resident: the routable mask is
/// all ones, the LUT maps (layer, expert) to slot layer E + expert); the expert arithmetic is the export's:
///   w13: xq = activation(xm) (NVFP4 / MXFP8 under Model.exl3_act, else none), each pair's rotation by its expert's
///        suh13, the trellis decode-GEMM over the rank's 2 Il interleaved columns, the epilogue (rotation, svh13, bf16,
///        softcapped SwiGLU, bf16) into x.act;
///   w2:  each pair's activation (NVFP4: under the pair's amax over every rank's channels, one small gather), its
///        rotation by suh2, the decode-GEMM into x.y (fp32 partials of the rank's channels), the epilogue (rotation,
///        svh2) in place; the TP sum, its row split / exchange overlap and the combine are moe's (moeTail,
///        moeOverlap; with p4-xchg's TF_INKLING2_MOE_XCHG=token the token rows).
/// x-merge (x-mixed DESIGN 2.7): the body in exl3Up / exl3Down / exl3Experts; the launches and their order are the
/// lane's (the token exchange as moe's).
fn moeExl3(m: *Model, li: usize, T: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const l = &m.layers[li];
    const H: usize = c.hidden;
    const E: usize = c.experts;
    const topk: usize = c.topk;
    const pairs = T * topk;
    try router(m, li, T); // e-dense: the active dense EXL3 set's router where it has one, else the bf16 product
    try routeRows(m, l, li, T); // e-step: ink2_route, or its compact form (the same bits) under TF_INKLING2_STEP_FUSE
    try k.demand(s, x.want_ids.ptr, x.want_sc.ptr, m.arena.routable.ptr + li * E, pairs, x.demand.ptr + li * E * 3 * 4);
    const prompt = T > 1;
    const wide = T > m.max_rows;
    // e-prefill: a fused wide chunk's plan items as its build takes them (kern.Kernels.exl3PromptTile: 64, or 128 for
    // v3g4dt128); a pair's bits do not depend on its item
    const tile: usize = if (wide and T >= kern.exl3_fused_rows) kern.Kernels.exl3PromptTile(k.exl3_cfg) else if (prompt) kern.prompt_tile else kern.decode_tile;
    const plan: cuda.grouped.Plan = .{ .members = x.members.ptr, .items = x.items.ptr, .counts = x.counts.ptr, .rank = x.plan_rank.ptr, .hist = x.plan_hist.ptr };
    try k.router.route(s, x.picks.ptr, pairs, E, tile, plan);
    try k.itemsToSlots(s, x.items.ptr, x.counts.ptr, m.arena.lut.ptr + li * E * 4);
    const items = cuda.grouped.maxItems(pairs, E, tile);
    if (overlapParts(m, T)) |n| {
        const ex = try exl3Up(m, li, T, plan, items, tile, prompt, wide, x.act.ptr);
        // moeOverlap runs the EXL3 down for each part (ex); its NVFP4 / MXFP8 descriptor is never read on this path
        const none: kern.Kernels.Experts = .{ .fmt = .exl3, .w = 0, .scale = 0, .slots = 0, .hidden = H, .inter = m.inter(), .cg = c.gate_cap, .cu = c.up_cap };
        return moeOverlap(m, T, none, plan, items, n, ex);
    }
    if (!exl3DecOn(m, wide)) try zeroPairs(m, T); // e-decode: exl3DecExperts writes every routed pair's row whole
    try exl3Experts(m, li, T, plan, items, tile, prompt, wide, x.act.ptr);
    return moeTail(m, T);
}

/// The EXL3 export's experts of a layer's routed pairs, from the input activation to the down epilogue into x.y
/// (exl3Up, then exl3Down), the SwiGLU rows into `act_out` (x.act here); the caller zeroes x.y's rows first as for
/// nativeExperts. x-merge: moeExl3's body, factored (x-mixed DESIGN 2.7: the mixed store runs it on the items its bank
/// does not hold, with its own act_out).
fn exl3Experts(m: *Model, li: usize, T: usize, plan: cuda.grouped.Plan, items: usize, tile: usize, prompt: bool, wide: bool, act_out: u64) !void {
    // e-decode (TF_INKLING2_EXL3_DEC=1): a decode / verify window at two ranks in fewer launches, the same bits a pair;
    // exl3Up is not run, so its activation and rotation (either form: TF_INKLING2_EXL3_ACTROT) run once, inside it
    if (exl3DecOn(m, wide)) return exl3DecExperts(m, T, plan, items, m.exl3View(), m.exl3Mode(li), act_out);
    const ex = try exl3Up(m, li, T, plan, items, tile, prompt, wide, act_out);
    try exl3Down(m, T, ex, plan, items, tile, prompt, wide);
}

/// The EXL3 export's w13 of a layer's routed pairs and w2's input: the tokens' activation (Model.exl3Mode) and each
/// pair's rotation, the decode-GEMM and the SwiGLU epilogue into `act_out` (bf16 [pair][Il]), then from act_out the
/// NVFP4 mode's per-pair amax over every rank's channels (one small gather), w2's activation and rotation into x.e_xr.
/// `tile`: the plan's item size (the fused kernels' passes; the engine's grouped kernel's member tiles). Returns the
/// store's view for the down. x-merge: moeExl3's body, factored.
fn exl3Up(m: *Model, li: usize, T: usize, plan: cuda.grouped.Plan, items: usize, tile: usize, prompt: bool, wide: bool, act_out: u64) !kern.Kernels.Exl3 {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    const topk: usize = c.topk;
    const pairs = T * topk;
    const Il = m.inter();
    const ex = m.exl3View();
    const mode = m.exl3Mode(li);
    // a wide prompt chunk of kern.exl3_fused_rows rows or more: its GEMMs take their epilogues in the same launch, on
    // the engine's EXL3 prompt hooks (kern.exl3PromptGu / exl3PromptDown: grid (column block, item), staged rows and
    // tiles, the engine's decoder), z never written; the same bits as the GEMM then the epilogue (tf-inkling2-exl3-bench)
    const fused = wide and T >= kern.exl3_fused_rows;
    // w13: the tokens' activation, each pair's rotation (row = pair / topk), the decode-GEMM, the SwiGLU epilogue
    if (k.exl3_actrot) {
        // e-prefill (TF_INKLING2_EXL3_ACTROT=1): the activation and the rotation in one pass, the same bits; each pick's
        // EXL3 slot from a per-(layer, expert) table (slot = lut[pick]; -1 skips the pick): the EXL3 store's arena LUT
        // (l E + e), or under TF_INKLING2_MIX x.xlut, the same identity. There the arena's LUT is the bank's (a bank slot
        // or -1): it would skip every EXL3 pick (its row of e_xr never written) and rotate the bank's picks with another
        // expert's vectors. With x.xlut the bank's picks are rotated too, into rows no EXL3 item reads (plan_x's members)
        const xlut = if (m.mix) x.xlut.ptr else m.arena.lut.ptr;
        try k.exl3ActRot13(s, ex, x.xm.ptr, H, T, H, mode, x.picks.ptr, topk, xlut + li * c.experts * 4, x.e_xr.ptr);
    } else {
        try k.exl3Act(s, x.xm.ptr, H, T, H, mode, 0, 1, 0, x.e_xq.ptr, H);
        try k.exl3Rot(s, ex, x.e_xq.ptr, H, topk, plan, items, false, x.e_xr.ptr, H);
    }
    if (fused) {
        try k.exl3PromptGu(s, ex, k.exl3_cfg, x.e_xr.ptr, plan, items, tile, act_out);
    } else {
        // from kern.exl3_core_rows rows the engine's grouped kernel (tf_exl3_grouped_2) is the faster form; the same bits
        if (T >= kern.exl3_core_rows) {
            try k.exl3Grouped(s, ex, false, x.e_xr.ptr, plan, items, tile, pairs, x.e_z.ptr, 1);
        } else try k.exl3Gemm(s, ex, false, prompt, x.e_xr.ptr, plan, items, x.e_z.ptr, 2 * Il, 0, 2 * Il, 1, 0);
        try k.exl3GateUpEpi(s, ex, x.e_z.ptr, 2 * Il, plan, items, act_out);
    }
    // w2's input: a pair's NVFP4 activation scales by its amax over the whole row, every rank's channels
    var amax_all: u64 = 0;
    if (mode == 1 and m.world > 1) {
        try k.exl3RowAmax(s, act_out, Il, pairs, Il, x.e_amax.ptr);
        try m.group.?.allGather(s, x.e_amax.ptr, x.e_amax_all.ptr, pairs * 4);
        amax_all = x.e_amax_all.ptr;
    }
    if (k.exl3_actrot) {
        // e-prefill: one pass; at world 1 the pairs' own amax first (exl3Act's, a max: the same bits)
        var world_eff = m.world;
        if (mode == 1 and m.world == 1) {
            try k.exl3RowAmax(s, act_out, Il, pairs, Il, x.e_amax.ptr);
            amax_all = x.e_amax.ptr;
            world_eff = 1;
        }
        try k.exl3ActRot2(s, ex, act_out, Il, mode, amax_all, world_eff, pairs, plan, items, x.e_xr.ptr, Il);
    } else {
        try k.exl3Act(s, act_out, Il, pairs, Il, mode, amax_all, m.world, pairs, x.e_xq.ptr, Il);
        try k.exl3Rot(s, ex, x.e_xq.ptr, Il, 0, plan, items, true, x.e_xr.ptr, Il);
    }
    return ex;
}

/// The EXL3 export's w2 of a layer's routed pairs from x.e_xr (exl3Up) into x.y: the decode-GEMM and its epilogue
/// (rotation, svh2), fp32 partials of this rank's channels; in one launch from kern.exl3_fused_rows rows (the engine's
/// EXL3 prompt hooks); under canonical TP the two K parts a world-2 pair computes, their terms added in rank order.
/// x-merge: moeExl3's body, factored.
fn exl3Down(m: *Model, T: usize, ex: kern.Kernels.Exl3, plan: cuda.grouped.Plan, items: usize, tile: usize, prompt: bool, wide: bool) !void {
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = m.c.hidden;
    const pairs = T * m.c.topk;
    const fused = wide and T >= kern.exl3_fused_rows;
    if (m.canon) {
        // canonical TP (world 1): the two K parts a world-2 pair computes, their terms added in rank order
        if (fused) try k.exl3PromptDown(s, ex, k.exl3_cfg, x.e_xr.ptr, plan, items, tile, x.y.ptr, H, 0, H, 2) else {
            try k.exl3Gemm(s, ex, true, prompt, x.e_xr.ptr, plan, items, x.e_zd.ptr, H, 0, H, 2, pairs * H);
            try k.exl3DownEpi(s, ex, x.y.ptr, H, 0, H, plan, items, x.e_zd.ptr, 2, pairs * H);
        }
    } else if (fused) {
        try k.exl3PromptDown(s, ex, k.exl3_cfg, x.e_xr.ptr, plan, items, tile, x.y.ptr, H, 0, H, 1);
    } else {
        try k.exl3Gemm(s, ex, true, prompt, x.e_xr.ptr, plan, items, x.y.ptr, H, 0, H, 1, 0);
        try k.exl3DownEpi(s, ex, x.y.ptr, H, 0, H, plan, items, x.y.ptr, 1, 0);
    }
}

// ---- x-mixed: the mixed expert store (TF_INKLING2_MIX=1, Model.mix; out/prefill/p5/x-mixed/DESIGN.md section 2)

/// The routed experts of layer `li` in the mixed store: every expert resident as its EXL3 trellis (Model.ex), a native
/// bank of the release's own bytes in the arena (its LUT row: the bank slot of an expert, or -1). The router is never
/// masked (routable null: the picks are its own top-k, nothing substituted); the demand counts, as field [2], the picks
/// the bank does not hold. One plan for both families: its items split on the device by the bank's map
/// (ink2_mix_split: no host value, so decode / verify graphs capture it and a bank change only edits the LUT); the
/// native forms run the bank's items (nativeExperts: W4A16 NVFP4, MXFP8 W8A16 or W8A8 under TF_INKLING2_MX_ACT), the
/// EXL3 forms the rest (exl3Experts: the export's own activation quantization), each writing its pairs' fp32 rows of
/// x.y (every pair in exactly one list), zeroed first once for both (zeroPairs); then the shared tail (moeTail: the
/// token exchange, the row split or the gather). The plan's tile: the MXFP8 layers' 128-pair items of the native wide
/// forms on a wide chunk, but 64-pair items at kern.exl3_core_rows .. kern.exl3_fused_rows - 1 rows, where the EXL3
/// w13 runs the engine's grouped kernel (16- and 64-pair items only). Never the overlapped exchange (moeOverlap runs
/// one family's down in column parts).
fn moeMix(m: *Model, li: usize, T: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const l = &m.layers[li];
    const E: usize = c.experts;
    const topk: usize = c.topk;
    const pairs = T * topk;
    try router(m, li, T); // e-dense: the active dense EXL3 set's router where it has one, else the bf16 product
    try k.route(s, x.lg.ptr, l.bias, 0, T, E, topk, c.route_scale, x.picks.ptr, x.wts.ptr, x.want_ids.ptr, x.want_sc.ptr);
    try k.demand(s, x.want_ids.ptr, x.want_sc.ptr, m.arena.routable.ptr + li * E, pairs, x.demand.ptr + li * E * 3 * 4);
    const prompt = T > 1;
    const wide = T > m.max_rows;
    const grouped_rows = T >= kern.exl3_core_rows and T < kern.exl3_fused_rows;
    const wide_mx = wide and m.prefill_experts and c.act[li] == .mxfp8 and !grouped_rows;
    const tile: usize = if (wide_mx) kern.wide_tile else if (prompt) kern.prompt_tile else kern.decode_tile;
    const plan: cuda.grouped.Plan = .{ .members = x.members.ptr, .items = x.items.ptr, .counts = x.counts.ptr, .rank = x.plan_rank.ptr, .hist = x.plan_hist.ptr };
    try k.router.route(s, x.picks.ptr, pairs, E, tile, plan);
    try k.mixSplit(s, x.items.ptr, x.counts.ptr, m.arena.lut.ptr + li * E * 4, li * E, x.items_n.ptr, x.counts_n.ptr, x.items_x.ptr, x.counts_x.ptr);
    if (m.mix_check) try mixCheck(m, li, E);
    const items = cuda.grouped.maxItems(pairs, E, tile);
    const plan_n: cuda.grouped.Plan = .{ .members = x.members.ptr, .items = x.items_n.ptr, .counts = x.counts_n.ptr, .rank = x.plan_rank.ptr, .hist = x.plan_hist.ptr };
    const plan_x: cuda.grouped.Plan = .{ .members = x.members.ptr, .items = x.items_x.ptr, .counts = x.counts_x.ptr, .rank = x.plan_rank.ptr, .hist = x.plan_hist.ptr };
    try zeroPairs(m, T);
    try nativeExperts(m, li, T, plan_n, items, prompt, wide_mx);
    try exl3Experts(m, li, T, plan_x, items, tile, prompt, wide, x.e_act.ptr);
    return moeTail(m, T);
}

/// TF_INKLING2_MIX_CHECK=1 (Model.mix_check; eager forwards only: Model.open turns the graphs off): the layer's split
/// read back and checked on the host against the plan and the bank's host LUT: items_n holds exactly the plan's items
/// whose experts the bank maps (with their slots), items_x every other (with layer E + expert), each in the plan's
/// order, and both counts' [0] and [1]. error.MixSplit on a difference (logged).
fn mixCheck(m: *Model, li: usize, E: usize) !void {
    const x = &m.x;
    try m.s.synchronize();
    var cn: [4]i32 = undefined;
    try x.counts.download(0, std.mem.sliceAsBytes(cn[0..2]));
    const n: usize = @intCast(cn[0]);
    const gpa = m.gpa;
    const plan = try gpa.alloc(i32, 3 * n);
    defer gpa.free(plan);
    const got_n = try gpa.alloc(i32, 3 * n);
    defer gpa.free(got_n);
    const got_x = try gpa.alloc(i32, 3 * n);
    defer gpa.free(got_x);
    var c2: [4]i32 = undefined;
    try x.items.download(0, std.mem.sliceAsBytes(plan));
    try x.counts_n.download(0, std.mem.sliceAsBytes(c2[0..2]));
    try x.counts_x.download(0, std.mem.sliceAsBytes(c2[2..4]));
    const nn: usize = @intCast(@max(c2[0], 0));
    const nx: usize = @intCast(@max(c2[2], 0));
    var ok = nn + nx == n;
    if (ok) {
        try x.items_n.download(0, std.mem.sliceAsBytes(got_n[0 .. 3 * nn]));
        try x.items_x.download(0, std.mem.sliceAsBytes(got_x[0 .. 3 * nx]));
        var jn: usize = 0;
        var jx: usize = 0;
        var heads = [2]i32{ 0, 0 };
        for (0..n) |it| {
            const e: usize = @intCast(plan[3 * it]);
            const head = it == 0 or plan[3 * (it - 1)] != plan[3 * it];
            const shown = !m.bank_hidden and (if (m.bank_mask) |mk| mk[li * E + e] != 0 else true);
            const slot = if (shown) m.arena.host_lut[li * E + e] else -1;
            const want = [3]i32{ if (slot >= 0) slot else @intCast(li * E + e), plan[3 * it + 1], plan[3 * it + 2] };
            if (slot >= 0) {
                ok = ok and jn < nn and std.mem.eql(i32, got_n[3 * jn ..][0..3], &want);
                jn += 1;
                heads[0] += @intFromBool(head);
            } else {
                ok = ok and jx < nx and std.mem.eql(i32, got_x[3 * jx ..][0..3], &want);
                jx += 1;
                heads[1] += @intFromBool(head);
            }
        }
        ok = ok and jn == nn and jx == nx and heads[0] == c2[1] and heads[1] == c2[3];
    }
    m.mix_checks += 1;
    if (!ok) {
        std.log.err("inkling2 rank {d}: layer {d}: the mixed store's split differs from the plan and the bank's LUT ({d} items: {d} bank + {d} EXL3)", .{ m.rank, li, n, nn, nx });
        return error.MixSplit;
    }
}
// ---- end x-mixed

/// A wide prompt chunk's expert sum at two ranks, its rows split: rank 0 combines rows [0, th), rank 1 [th, T)
/// (th = ceil(T / 2)). Each rank sends the other its fp32 partials of the other's rows and receives the other's of its
/// own (one NCCL group), combines its rows with each pair's rank-order sum (ink2_combine_pair: combine_parts's
/// arithmetic, the same bits), then the bf16 halves are gathered in place (rank 1's last row padding when T is odd:
/// x.mo holds T + 1 rows). Half the bytes of gathering every pair's partials, and half the combine on each rank.
fn moeSplit(m: *Model, T: usize) !void {
    const c = &m.c;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    const topk: usize = c.topk;
    const g = &m.group.?;
    const th = (T + 1) / 2;
    const lo = if (m.rank == 0) 0 else th;
    const mine = if (m.rank == 0) th else T - th;
    const other_lo = if (m.rank == 0) th else 0;
    const other = T - mine;
    const peer = 1 - m.rank;
    const row_bytes = topk * H * 4;
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        if (other > 0) g.send(s, x.y.ptr + other_lo * row_bytes, other * row_bytes, peer) catch |err| break :post err;
        if (mine > 0) g.recv(s, x.y_all.ptr, mine * row_bytes, peer) catch |err| break :post err;
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    // rank 0's partials first in each sum: rank 0 holds its own in y and rank 1's in y_all, rank 1 the reverse
    const own = x.y.ptr + lo * row_bytes;
    const y0 = if (m.rank == 0) own else x.y_all.ptr;
    const y1 = if (m.rank == 0) x.y_all.ptr else own;
    if (mine > 0) try m.k.combinePair(s, y0, y1, x.picks.ptr + lo * topk * 4, x.wts.ptr + lo * topk * 4, x.mo.ptr + lo * H * 2, mine, topk, H);
    if (m.prefill_kernels) return exchangeHalves(m, x.mo.ptr, H * 2, T);
    try gatherNccl(m, s, x.mo.ptr + m.rank * th * H * 2, x.mo.ptr, th * H * 2);
}

/// moeSplit with its exchange overlapped (Model.overlap, TF_INKLING2_PREFILL_OVERLAP): the down projection in n
/// column parts (overlapPartsFor) of whole part_cols tiles, y laid out [part][pairs][part's columns] so a part's rows
/// are contiguous. Once part q is written (event down[q] on the model stream) the comm stream sends the peer its rows
/// of it and receives this rank's (one NCCL group a part; the communicator runs its groups in issue order) while part
/// q + 1 computes; part q's combine waits for its own exchange only (recvd[q]) and runs while later parts are on the
/// wire. Every element's arithmetic is moeSplit's (the down form's sums by column, ink2_combine_pair_cols =
/// ink2_combine_pair by column): only the order of execution changes. The comm stream's reads of y and writes of
/// y_all are done before the next layer's down form (it follows every combine on the model stream, and each combine
/// waited for its part).
fn moeOverlap(m: *Model, T: usize, dn: kern.Kernels.Experts, plan: cuda.grouped.Plan, items: usize, n: usize, ex3: ?kern.Kernels.Exl3) !void {
    const c = &m.c;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    const topk: usize = c.topk;
    const g = &m.group.?;
    const ov = m.ov.?;
    const th = (T + 1) / 2;
    const lo = if (m.rank == 0) 0 else th;
    const mine = if (m.rank == 0) th else T - th;
    const other_lo = if (m.rank == 0) th else 0;
    const other = T - mine;
    const peer = 1 - m.rank;
    const pairs = T * topk;
    const tiles = H / kern.Kernels.part_cols;
    // part q: tiles [tiles q / n, tiles (q + 1) / n) = columns [c0, c0 + cn); its rows at y + c0 pairs 4 bytes, the
    // peer's partials of this rank's rows at y_all + c0 mine topk 4 bytes
    for (0..n) |q| {
        const c0 = tiles * q / n * kern.Kernels.part_cols;
        const cn = tiles * (q + 1) / n * kern.Kernels.part_cols - c0;
        const yq = x.y.ptr + c0 * pairs * 4;
        if (ex3) |e3| {
            // the EXL3 export's down and its epilogue (rotation, svh2) on the part's columns: in one launch from
            // kern.exl3_fused_rows rows (kern.exl3PromptDown, the engine's EXL3 prompt hooks), the same bits either way
            if (T >= kern.exl3_fused_rows) {
                // moeExl3's fused wide plans (e-prefill: their items as the build takes them, exl3PromptTile)
                try m.k.exl3PromptDown(s, e3, m.k.exl3_cfg, x.e_xr.ptr, plan, items, kern.Kernels.exl3PromptTile(m.k.exl3_cfg), yq, cn, c0, cn, 1);
            } else {
                try m.k.exl3Gemm(s, e3, true, true, x.e_xr.ptr, plan, items, yq, cn, c0, cn, 1, 0);
                try m.k.exl3DownEpi(s, e3, yq, cn, c0, cn, plan, items, yq, 1, 0);
            }
        } else try m.k.downCols(s, dn, x.act.ptr, plan, items, yq, c0, cn);
        try ov.down[q].record(s);
        try ov.comm.wait(ov.down[q]);
        const row_bytes = topk * cn * 4;
        try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            if (other > 0) g.send(ov.comm, yq + other_lo * row_bytes, other * row_bytes, peer) catch |err| break :post err;
            if (mine > 0) g.recv(ov.comm, x.y_all.ptr + c0 * mine * topk * 4, mine * row_bytes, peer) catch |err| break :post err;
            break :post {};
        };
        try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
        try posted;
        try ov.recvd[q].record(ov.comm);
    }
    for (0..n) |q| {
        const c0 = tiles * q / n * kern.Kernels.part_cols;
        const cn = tiles * (q + 1) / n * kern.Kernels.part_cols - c0;
        try s.wait(ov.recvd[q]);
        // rank 0's partials first in each sum: rank 0 holds its own in y and rank 1's in y_all, rank 1 the reverse
        const own = x.y.ptr + c0 * pairs * 4 + lo * topk * cn * 4;
        const got = x.y_all.ptr + c0 * mine * topk * 4;
        const y0 = if (m.rank == 0) own else got;
        const y1 = if (m.rank == 0) got else own;
        if (mine > 0) try m.k.combinePairCols(s, y0, y1, x.picks.ptr + lo * topk * 4, x.wts.ptr + lo * topk * 4, x.mo.ptr + lo * H * 2 + c0 * 2, mine, topk, cn, H);
    }
    if (m.prefill_kernels) return exchangeHalves(m, x.mo.ptr, H * 2, T);
    try gatherNccl(m, s, x.mo.ptr + m.rank * th * H * 2, x.mo.ptr, th * H * 2);
}

/// A chunk's rows per column part of moeOverlap at least: a part then moves at least this many rows' partials of a
/// rank's half (64 x 8 x H x 4 / 2 bytes = 4.7 MB at H 4608), more than its NCCL group's and launch's fixed costs.
const overlap_part_rows = 64;

/// The column parts moeOverlap splits a wide chunk of T rows into: Model.overlap at most, at most one a part_cols
/// tile, at most one per overlap_part_rows rows. Below 2 the chunk takes moeSplit (nothing to overlap).
fn overlapPartsFor(m: *const Model, T: usize) usize {
    return @min(m.overlap, m.c.hidden / kern.Kernels.part_cols, T / overlap_part_rows);
}

/// o_proj's sum for a wide prompt chunk at two ranks, its rows split as moeSplit's: each rank sends the other its fp32
/// partials of the other's rows and receives the other's of its own, sums its rows in rank order and rounds them once
/// (ink2_rank_sum_pair_bf16: rank_sum_bf16's add, the same bits), then the bf16 halves are exchanged.
fn oprojSplit(m: *Model, T: usize) !void {
    const H: usize = m.c.hidden;
    const s = m.s;
    const x = &m.x;
    const g = &m.group.?;
    const th = (T + 1) / 2;
    const lo = if (m.rank == 0) 0 else th;
    const mine = if (m.rank == 0) th else T - th;
    const other_lo = if (m.rank == 0) th else 0;
    const other = T - mine;
    const peer = 1 - m.rank;
    const row_bytes = H * 4;
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        if (other > 0) g.send(s, x.o32.ptr + other_lo * row_bytes, other * row_bytes, peer) catch |err| break :post err;
        if (mine > 0) g.recv(s, x.o_all.ptr, mine * row_bytes, peer) catch |err| break :post err;
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    const own = x.o32.ptr + lo * row_bytes;
    const y0 = if (m.rank == 0) own else x.o_all.ptr;
    const y1 = if (m.rank == 0) x.o_all.ptr else own;
    if (mine > 0) try m.k.rankSumPairBf16(s, y0, y1, x.o.ptr + lo * H * 2, mine * H);
    try exchangeHalves(m, x.o.ptr, H * 2, T);
}

/// A wide chunk's row halves of `buf` exchanged at two ranks (rows [0, ceil(T / 2)) rank 0's, the rest rank 1's;
/// `row_bytes` a row), so both ranks hold every row: one NCCL group, a send and a receive (its point-to-point
/// protocol moves these sizes at about twice the rate of its ring all-gather here).
fn exchangeHalves(m: *Model, buf: u64, row_bytes: usize, T: usize) !void {
    const g = &m.group.?;
    const th = (T + 1) / 2;
    const lo = if (m.rank == 0) 0 else th;
    const mine = if (m.rank == 0) th else T - th;
    const other_lo = if (m.rank == 0) th else 0;
    const other = T - mine;
    const peer = 1 - m.rank;
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        if (mine > 0) g.send(m.s, buf + lo * row_bytes, mine * row_bytes, peer) catch |err| break :post err;
        if (other > 0) g.recv(m.s, buf + other_lo * row_bytes, other * row_bytes, peer) catch |err| break :post err;
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
}

/// A layer's relative bias rows [r0, r0 + n) (x.r . rel_logits_proj.proj -> x.rel, bf16) in one product of n x hq rows;
/// under canonical TP (Model.canon) each half's heads as a pair's rank has them (its rows contiguous), the product at
/// the rank's shape, the rows back in place. x-merge: the EXL3 lane's canonical body, shared by phase 3's max_rows-row
/// pieces and p4-small's one product a wide chunk (TF_INKLING2_REL_WIDE=1: a pair's rank makes T x hq / 2 rows at
/// once).
fn relRows(m: *Model, l: *const model.Layer, r0: usize, n: usize) !void {
    const c = &m.c;
    const s = m.s;
    const x = &m.x;
    const hq = m.hq();
    const rel_row = hq * l.extent * 2;
    if (!m.canon) return m.g.rowsTimes(s, x.r.ptr + r0 * hq * c.d_rel * 2, l.relproj, x.rel.ptr + r0 * rel_row, n * hq, l.extent, c.d_rel);
    // canonical TP: each half's heads as a pair's rank has them (its rows contiguous), the product at the rank's shape,
    // the rows back in place
    const hh = hq / 2;
    const tin = x.canon_tmp.ptr;
    const tout = tin + n * hh * c.d_rel * 2;
    for (0..2) |r| {
        try model.copy2d(m, tin, hh * c.d_rel * 2, x.r.ptr + r0 * hq * c.d_rel * 2 + r * hh * c.d_rel * 2, hq * c.d_rel * 2, hh * c.d_rel * 2, n);
        try m.g.rowsTimes(s, tin, l.relproj, tout, n * hh, l.extent, c.d_rel);
        try model.copy2d(m, x.rel.ptr + r0 * rel_row + r * hh * l.extent * 2, rel_row, tout, hh * l.extent * 2, hh * l.extent * 2, n);
    }
}

/// A head-split product d [T][n] = x [T][k] . w [n][k]^T (bf16 out): one GEMM, or under canonical TP the pair's two
/// half-width GEMMs (each rank's rows of w, its shape's cuBLASLt plan), each half's rows moved into place.
fn proj(m: *Model, xin: u64, w: u64, out: u64, T: usize, n: usize, k: usize) !void {
    if (!m.canon) return m.g.run(m.s, xin, w, out, T, n, k, false);
    const h = n / 2;
    for (0..2) |r| {
        try m.g.run(m.s, xin, w + r * h * k * 2, m.x.canon_tmp.ptr, T, h, k, false);
        try model.copy2d(m, out + r * h * 2, n * 2, m.x.canon_tmp.ptr, h * 2, h * 2, T);
    }
}

/// A static short convolution in place: a wide prompt chunk's rows in parallel (ink2_sconv_*_seg, no history),
/// else the one-thread loop (ink2_sconv_static, its history when `hist`): the same bits.
fn sconv(m: *Model, wide: bool, xy: u64, w: u64, ring: u64, T: usize, C: usize, hist: u64) !void {
    if (wide and m.fuse) return m.k.sconvStaticRowsLd(m.s, xy, w, ring, m.x.edges.ptr, T, C, m.c.conv); // p4-small
    if (wide) return m.k.sconvStaticRows(m.s, xy, w, ring, m.x.edges.ptr, T, C, m.c.conv);
    try m.k.sconvStatic(m.s, xy, xy, w, ring, T, C, m.c.conv, hist);
}

/// Exact mode: the layer's unmasked picks that are not resident, loaded into the transient ring (the previous layer's
/// transients dropped first: the stream is idle once the picks are on the host), then the route run again: with
/// every wanted expert routable, the masked pick is the unmasked one. Both ranks route the same rows the same way
/// (the router's input is bit-equal across ranks), so both load the same experts.
fn exactLoads(m: *Model, li: usize, T: usize) !void {
    const c = &m.c;
    const E: usize = c.experts;
    const pairs = T * c.topk;
    const x = &m.x;
    var want: [64 * 16]i32 = undefined;
    if (pairs > want.len) return error.ChunkRows;
    try x.want_ids.downloadAsync(0, std.mem.sliceAsBytes(want[0..pairs]), m.s.handle);
    try m.s.synchronize();
    m.dropTransient();
    const fi: usize = if (c.act[li] == .nvfp4) 0 else 1; // the arena's pool (Config.act)
    const a = &m.arena;
    const cap = a.slots(fi) - a.resident[fi];
    var jobs: [64 * 16]@import("fill.zig").Job = undefined;
    var n: usize = 0;
    var seen = std.StaticBitSet(1024).empty;
    for (want[0..pairs]) |e| {
        if (e < 0) continue;
        const ue: usize = @intCast(e);
        if (seen.isSet(ue) or a.host_lut[li * E + ue] >= 0) continue;
        seen.set(ue);
        if (n >= cap) return error.TransientRingFull;
        jobs[n] = .{ .layer = @intCast(li), .expert = @intCast(ue), .slot = @intCast(a.resident[fi] + n) };
        n += 1;
    }
    if (n > 0) {
        try m.loader.?.load(m, jobs[0..n]);
        m.transient_used[fi] = n;
        m.exact_loads += n;
    }
    try m.publish();
    if (n > 0) {
        const l = &m.layers[li];
        try m.k.route(m.s, x.lg.ptr, l.bias, a.routable.ptr + li * E, T, E, c.topk, c.route_scale, x.picks.ptr, x.wts.ptr, 0, 0);
    }
}

// ---- p4-xchg: the MoE's TP sum by token rows (TF_INKLING2_MOE_XCHG=token, Model.moe_xchg; inkling2.cu's block)
// Each rank first combines its own fp32 down partials of a token's routed pairs into one fp32 row
// (ink2_xchg_tok_rows: weight times partial, summed over the picks in ascending expert id), the ranks exchange those
// rows (top-k = 8 times fewer bytes than the pairs' partials: 38 MB instead of 302 MB a layer at 2048 rows), and each
// row's ranks' sums are added in rank order and rounded once to bf16: the combine's last rounding, where it was. Not
// phase 3's bits: each pair's bf16 rounding of its down output (it needs the pair's sum over both ranks) is gone, and
// the adds run in another order (measured: out/prefill/p4/p4-xchg/REPORT.md). The rows live in o_proj's buffers
// (its sum is done before the MoE starts; the next layer's o_proj writes them after this layer's exchange): x.o32
// this rank's rows [T][H] fp32, x.o_all the gathered or received ones. A wide chunk's down form runs whole, not in
// moeOverlap's column parts: with the smaller exchange the whole-width kernels (the NVFP4 fact form, the MXFP8 wide
// form) are faster than the parts' forms by more than the overlap hid (REPORT.md: measured with and without).

/// The TP sum and combine (-> x.mo) of a layer's down partials (x.y, every routed pair's fp32 row) by token rows at
/// world > 1 (p4-xchg's moeTok): each rank's token rows (ink2_xchg_tok_rows into x.o32); a wide chunk at two ranks
/// splits its rows (moeTokSplit, as moeSplit); anything else (decode, verify, chunks of at most max_rows rows: graphs
/// capture it) gathers every rank's rows (the group's RoCE gather at two ranks: T x H x 4 bytes a rank, not T x top-k
/// x H x 4) and sums them in rank order with the bf16 rounding (ink2_rank_sum_bf16). x-merge: moeTok's last steps
/// (its down now nativeExperts'), moeTail's token form for both stores.
fn moeTokRows(m: *Model, T: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    try k.xchgTokRows(s, x.y.ptr, x.picks.ptr, x.wts.ptr, x.o32.ptr, T, c.topk, H);
    if (T > m.max_rows and m.world == 2 and m.split_moe) return moeTokSplit(m, T);
    try m.group.?.allGather(s, x.o32.ptr, x.o_all.ptr, T * H * 4);
    try k.rankSumBf16(s, x.o_all.ptr, x.mo.ptr, m.world, T * H);
}

/// moeSplit by token rows (its rows x.o32 [T][H]): each rank sends the peer its rows of the peer's half and receives
/// the peer's of its own half (x.o_all), sums its half in rank order and rounds it (ink2_rank_sum_pair_bf16), then the
/// bf16 halves are exchanged as moeSplit's.
fn moeTokSplit(m: *Model, T: usize) !void {
    const H: usize = m.c.hidden;
    const s = m.s;
    const x = &m.x;
    const g = &m.group.?;
    const th = (T + 1) / 2;
    const lo = if (m.rank == 0) 0 else th;
    const mine = if (m.rank == 0) th else T - th;
    const other_lo = if (m.rank == 0) th else 0;
    const other = T - mine;
    const peer = 1 - m.rank;
    const row_bytes = H * 4;
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = post: {
        if (other > 0) g.send(s, x.o32.ptr + other_lo * row_bytes, other * row_bytes, peer) catch |err| break :post err;
        if (mine > 0) g.recv(s, x.o_all.ptr, mine * row_bytes, peer) catch |err| break :post err;
        break :post {};
    };
    try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
    try posted;
    const own = x.o32.ptr + lo * row_bytes;
    const r0 = if (m.rank == 0) own else x.o_all.ptr;
    const r1 = if (m.rank == 0) x.o_all.ptr else own;
    if (mine > 0) try m.k.rankSumPairBf16(s, r0, r1, x.mo.ptr + lo * H * 2, mine * H);
    if (m.prefill_kernels) return exchangeHalves(m, x.mo.ptr, H * 2, T);
    try gatherNccl(m, s, x.mo.ptr + m.rank * th * H * 2, x.mo.ptr, th * H * 2);
}
// ---- end p4-xchg

// ---- e-step: TF_INKLING2_STEP_FUSE (Model.step_fuse): a decode / verify chunk of at most max_rows rows takes the fused
// launches (the same bits): FUSE's block attention residual + its norm(s), the MoE input's residual folded in, the MLP
// conv writing the layer output (with the history a rewind restores), the sliding layers' k / v convs in one launch,
// the four head preps in one, o_proj's TP sum with the attention's conv (at two ranks or more). Canonical TP keeps the
// unfused launches.
fn stepFused(m: *const Model, T: usize) bool {
    return m.step_fuse and T <= m.max_rows and !m.canon;
}

/// A batched head's argmaxes into x.argmax[0 .. T] without its logits' exchange (ink2_es_head_best / _pick: the same
/// argmaxes as the gather, unsplit and ink2_logits): each rank's (value, index) pair a row from its columns `rows`
/// [T][vl] (global index lo + i), the pairs gathered (16-byte multiples: the RoCE gather's unit) into the full gather's
/// buffer (x.head_rows_all, unused on this path: the pairs, then every rank's), each row's pick over the ranks.
fn headArgmax(m: *Model, rows: u64, T: usize, vl: usize, lo: usize) !void {
    const x = &m.x;
    const c = &m.c;
    const stride = (T + 1) / 2 * 2; // pairs a rank's segment holds: 16-byte multiples
    const best = x.head_rows_all.ptr;
    const all = best + 256; // 16-row windows' pairs (128 bytes at most) before it
    if (stride * 8 > 256) return error.ChunkRows;
    try m.k.esHeadBest(m.s, rows, T, vl, lo, c.unpadded_vocab, c.logit_cap, best);
    try m.group.?.allGather(m.s, best, all, stride * 8);
    try m.k.esHeadPick(m.s, all, m.world, T, stride, x.argmax.ptr);
}

/// A layer's routed picks of x.lg (ink2_route's outputs: picks, weights, the demand's wants): its compact form
/// ink2_es_route for a step-fused chunk (the same bits), else ink2_route.
fn routeRows(m: *Model, l: *const model.Layer, li: usize, T: usize) !void {
    const c = &m.c;
    const x = &m.x;
    const E: usize = c.experts;
    const r = m.arena.routable.ptr + li * E;
    if (stepFused(m, T)) return m.k.esRoute(m.s, x.lg.ptr, l.bias, r, T, E, c.topk, c.route_scale, x.picks.ptr, x.wts.ptr, x.want_ids.ptr, x.want_sc.ptr);
    try m.k.route(m.s, x.lg.ptr, l.bias, r, T, E, c.topk, c.route_scale, x.picks.ptr, x.wts.ptr, x.want_ids.ptr, x.want_sc.ptr);
}
// ---- end e-step

// ---- e-dense (prefill lane phase 5): the non-expert linears through an active dense EXL3 set (dense_exl3.zig;
// TF_INKLING2_DENSE_EXL3, default off: none of this runs and the forward is x2's) ----

/// The active dense EXL3 set's weights of layer li, or null.
fn dxLayer(m: *const Model, li: usize) ?*const dense_exl3.LayerW {
    const dx = m.dx orelse return null;
    const set = dx.active() orelse return null;
    return &set.layers[li];
}

/// The active dense EXL3 set when it holds the head, else null.
fn dxHead(m: *const Model) ?*const dense_exl3.Set {
    const dx = m.dx orelse return null;
    const set = dx.active() orelse return null;
    return if (set.head != null) set else null;
}

/// A head-split product d [T][n] (bf16) through the EXL3 weight `wx` when there is one (canonical TP: the pair's two
/// column halves as views, each rank's own shape, moved into place as proj does), else proj's bf16 GEMM.
fn projx(m: *Model, xin: u64, w: u64, wx: ?dense_exl3.Weight, out: u64, T: usize, n: usize, k: usize) !void {
    const e = wx orelse return proj(m, xin, w, out, T, n, k);
    const dx = m.dx.?;
    if (!m.canon) return dx.run(m.s, e, xin, T, out, false);
    const h = n / 2;
    for (0..2) |r| {
        try dx.run(m.s, try dense_exl3.half(e, .cols, r), xin, T, m.x.canon_tmp.ptr, false);
        try model.copy2d(m, out + r * h * 2, n * 2, m.x.canon_tmp.ptr, h * 2, h * 2, T);
    }
}

/// The router's fp32 logits x.lg [T][E] of x.xm: the active set's EXL3 router (whole on every rank: one plan, the same
/// bits on every rank, so the routing agrees) or the bf16 product.
fn router(m: *Model, li: usize, T: usize) !void {
    const x = &m.x;
    const rx: ?dense_exl3.Weight = if (dxLayer(m, li)) |lx| lx.router else null;
    if (rx) |e| return m.dx.?.run(m.s, e, x.xm.ptr, T, x.lg.ptr, true);
    try m.g.run(m.s, x.xm.ptr, m.layers[li].router, x.lg.ptr, T, m.c.experts, m.c.hidden, true);
}

/// headRow through the set's EXL3 head: this rank's head_n / world columns (whole 128-blocks), gathered in rank order
/// into [head_n] (canonical TP: the pair's two halves), then the softcapped logits over the real tokens.
fn headRowX(m: *Model, set: *const dense_exl3.Set, row: usize, out_f32: u64, am: u64) !void { // xs: outputs as arguments
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = c.hidden;
    const e = set.head.?;
    const V = set.head_n;
    if (m.canon) {
        const hv = V / 2;
        for (0..2) |r| try m.dx.?.run(s, try dense_exl3.half(e, .cols, r), x.xm.ptr + row * H * 2, 1, x.head_all.ptr + r * hv * 2, false);
        return k.logits(s, x.head_all.ptr, 1, V, c.unpadded_vocab, c.logit_cap, out_f32, am);
    }
    const vl = V / m.world;
    try m.dx.?.run(s, e, x.xm.ptr + row * H * 2, 1, x.head.ptr, false);
    if (m.world > 1) {
        try m.group.?.allGather(s, x.head.ptr, x.head_all.ptr, vl * 2);
    } else try x.head_all.copyFrom(0, x.head.ptr, vl * 2, s.handle);
    try k.logits(s, x.head_all.ptr, 1, V, c.unpadded_vocab, c.logit_cap, out_f32, am);
}

/// headBatch through the set's EXL3 head (argmaxes only into x.argmax[0 .. T]).
fn headBatchX(m: *Model, set: *const dense_exl3.Set, T: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const e = set.head.?;
    const V = set.head_n;
    if (m.canon) {
        const hv = V / 2;
        for (0..2) |r| try m.dx.?.run(s, try dense_exl3.half(e, .cols, r), x.xm.ptr, T, x.head_rows_all.ptr + r * T * hv * 2, false);
        try k.unsplit(s, x.head_rows_all.ptr, x.head_cat.ptr, T, hv, 2);
        return k.logits(s, x.head_cat.ptr, T, V, c.unpadded_vocab, c.logit_cap, 0, x.argmax.ptr);
    }
    const vl = V / m.world;
    try m.dx.?.run(s, e, x.xm.ptr, T, x.head_rows.ptr, false);
    if (m.world > 1 and stepFused(m, T)) return headArgmax(m, x.head_rows.ptr, T, vl, m.rank * vl); // e-step
    if (m.world > 1) {
        try m.group.?.allGather(s, x.head_rows.ptr, x.head_rows_all.ptr, T * vl * 2);
        try k.unsplit(s, x.head_rows_all.ptr, x.head_cat.ptr, T, vl, m.world);
    } else try x.head_cat.copyFrom(0, x.head_rows.ptr, T * vl * 2, s.handle);
    try k.logits(s, x.head_cat.ptr, T, V, c.unpadded_vocab, c.logit_cap, 0, x.argmax.ptr);
}

/// The run tool's logits path (--ids-files with --out): the head over a chunk's rows in blocks of `rows` rows, every
/// block's fp32 softcapped logits [n][unpadded] at once (one product a block, as headBatch's, instead of headRow's
/// product a row: the same rounding points, the products summed in the block GEMM's order).
pub const HeadBlock = struct {
    rows: usize,
    part: cuda.DeviceBuffer, // [rows][vl] bf16, this rank's vocabulary columns
    all: cuda.DeviceBuffer, // [planes][rows][vl]
    cat: cuda.DeviceBuffer, // [rows][V]
    f32: cuda.DeviceBuffer, // [rows][unpadded]
    am: cuda.DeviceBuffer, // [rows] i32

    pub fn init(m: *const Model, rows: usize) !HeadBlock {
        const planes = @max(m.world, 2);
        const vl = m.c.vocab / @max(m.world, 1) + 128; // the bf16 head's or an EXL3 head's half, whichever is wider
        var hb: HeadBlock = undefined;
        hb.rows = rows;
        hb.part = try cuda.DeviceBuffer.alloc(m.d, rows * vl * 2);
        errdefer hb.part.free();
        hb.all = try cuda.DeviceBuffer.alloc(m.d, planes * rows * vl * 2);
        errdefer hb.all.free();
        hb.cat = try cuda.DeviceBuffer.alloc(m.d, rows * (m.c.vocab + 256) * 2);
        errdefer hb.cat.free();
        hb.f32 = try cuda.DeviceBuffer.alloc(m.d, rows * m.c.unpadded_vocab * 4);
        errdefer hb.f32.free();
        hb.am = try cuda.DeviceBuffer.alloc(m.d, rows * 4);
        return hb;
    }

    pub fn deinit(hb: *HeadBlock) void {
        for ([_]*cuda.DeviceBuffer{ &hb.part, &hb.all, &hb.cat, &hb.f32, &hb.am }) |b| b.free();
    }
};

/// The head on rows [r0, r0 + n) of x.xm (after readout) into hb.f32 [n][unpadded]: the active set's EXL3 head or the
/// bf16 one, this rank's vocabulary columns, gathered and put in vocabulary order, softcapped.
fn headBlock(m: *Model, hb: *HeadBlock, r0: usize, n: usize) !void {
    const c = &m.c;
    const k = &m.k;
    const s = m.s;
    const H: usize = c.hidden;
    const xin = m.x.xm.ptr + r0 * H * 2;
    const set = dxHead(m);
    const V = if (set) |st| st.head_n else c.vocab;
    if (m.canon) {
        const hv = V / 2;
        for (0..2) |r| {
            if (set) |st| {
                try m.dx.?.run(s, try dense_exl3.half(st.head.?, .cols, r), xin, n, hb.all.ptr + r * n * hv * 2, false);
            } else try m.g.run(s, xin, m.unembed + r * hv * H * 2, hb.all.ptr + r * n * hv * 2, n, hv, H, false);
        }
        try k.unsplit(s, hb.all.ptr, hb.cat.ptr, n, hv, 2);
    } else {
        const vl = V / m.world;
        if (set) |st| {
            try m.dx.?.run(s, st.head.?, xin, n, hb.part.ptr, false);
        } else try m.g.run(s, xin, m.unembed, hb.part.ptr, n, vl, H, false);
        if (m.world > 1) {
            try m.group.?.allGather(s, hb.part.ptr, hb.all.ptr, n * vl * 2);
            try k.unsplit(s, hb.all.ptr, hb.cat.ptr, n, vl, m.world);
        } else try hb.cat.copyFrom(0, hb.part.ptr, n * vl * 2, s.handle);
    }
    try k.logits(s, hb.cat.ptr, n, V, c.unpadded_vocab, c.logit_cap, hb.f32.ptr, hb.am.ptr);
}

/// A chunk's rows (ids on the host) through the model and the final readout (eager; exact mode drops its transients
/// after the layers, as `eager` does); its logits then come a head block at a time (blockLogits), so a 4096-row chunk's
/// fp32 logits (3.3 GB) are never all on the host.
pub fn chunkRows(m: *Model, ids: []const i32) !void {
    const T = ids.len;
    const c = &m.c;
    const s = m.s;
    if (T == 0 or T > m.prompt_rows or (m.exact and T > m.max_rows)) return error.ChunkRows;
    if (m.state.len + T > m.context) return error.ContextFull;
    if (m.lazy) |lz| try lz.ensure(m, m.state.len + T); // p4-mem: as chunk's, the positions mapped first
    for (ids) |id| if (id < 0 or id >= c.vocab) return error.TokenOutOfVocabulary;
    try m.x.ids.uploadAsync(0, std.mem.sliceAsBytes(ids), s.handle);
    try layers(m, T, m.state.len);
    m.state.len += T;
    if (m.exact) {
        try drain(m);
        m.dropTransient();
        try m.publish();
    }
    try readout(m, T);
}

/// Rows [r0, r0 + n) (n <= hb.rows) of the chunk chunkRows ran: their fp32 softcapped logits [n][unpadded] into `out`
/// (host) through `hb` (headBlock).
pub fn blockLogits(m: *Model, hb: *HeadBlock, r0: usize, n: usize, out: []f32) !void {
    const U = m.c.unpadded_vocab;
    if (n == 0 or n > hb.rows or out.len < n * U) return error.ChunkRows;
    try headBlock(m, hb, r0, n);
    try hb.f32.downloadAsync(0, std.mem.sliceAsBytes(out[0 .. n * U]), m.s.handle);
    try drain(m); // hb is reused by the next block
}

// ---- e-step: TF_INKLING2_DENSE_GROUP=1 (Model.dense_group): a layer's EXL3 projections of the attention input as one
// projection group (kern.exl3Group5: every member's rotation in one launch, every member's decode-row linear on its own
// blocks of one grid; each member's arithmetic its own dense_exl3 call's: the same bits). q, r, gate, and on a sliding
// layer k and v too (they read the same rows there); decode / verify chunks (1-16 rows) at two ranks, 6-bit sets.

/// The group's members of layer `lx` (null: not grouped) and their outputs.
fn groupMembers(m: *Model, lx: *const dense_exl3.LayerW, local: bool, ws: *[5]dense_exl3.Weight, ys: *[5]u64) ?usize {
    const x = &m.x;
    const q = lx.wq orelse return null;
    const r = lx.wr orelse return null;
    const g = lx.wgate orelse return null;
    ws[0] = q;
    ys[0] = x.q.ptr;
    ws[1] = r;
    ys[1] = x.r.ptr;
    ws[2] = g;
    ys[2] = x.gate.ptr;
    if (!local) return 3;
    ws[3] = lx.wk orelse return null;
    ys[3] = x.k.ptr;
    ws[4] = lx.wv orelse return null;
    ys[4] = x.v.ptr;
    return 5;
}

/// The layer's grouped projections when the switch, the chunk and the weights allow it (true: done).
fn projGroup(m: *Model, lx: ?*const dense_exl3.LayerW, local: bool, T: usize) !bool {
    if (!m.dense_group or m.canon or m.world != 2 or T > m.max_rows) return false;
    const l = lx orelse return false;
    var ws: [5]dense_exl3.Weight = undefined;
    var ys: [5]u64 = undefined;
    const n = groupMembers(m, l, local, &ws, &ys) orelse return false;
    if (!kern.Kernels.group5able(ws[0..n], T)) return false;
    const dx = m.dx.?;
    const sc = try dx.op.groupScratch(m.s, ws[0..n], T);
    try m.k.exl3Group5(m.s, ws[0..n], m.x.xq.ptr, T, ys[0..n], false, sc.xh, sc.z, sc.counters, dx.op.lin.pdl);
    return true;
}

/// Every layer's group scratch at max_rows rows, reserved before any graph capture (Model.open, with the switch).
pub fn reserveGroups(m: *Model) !void {
    const dx = m.dx orelse return;
    for (m.layers, 0..) |l, li| {
        const lx = dxLayer(m, li) orelse continue;
        var ws: [5]dense_exl3.Weight = undefined;
        var ys: [5]u64 = undefined;
        const n = groupMembers(m, lx, l.local, &ws, &ys) orelse continue;
        if (!kern.Kernels.group5able(ws[0..n], m.max_rows)) continue;
        _ = try dx.op.groupScratch(m.s, ws[0..n], m.max_rows);
    }
    try m.s.synchronize();
}
// ---- end e-step
// ---- e-decode (prefill lane phase 5): the EXL3 decode / verify windows' experts in fewer launches
// (TF_INKLING2_EXL3_DEC, default off: none of this runs and the forward is x4's) ----

/// Whether a chunk's EXL3 experts run on exl3DecExperts: TF_INKLING2_EXL3_DEC (Model.exl3_dec), the EXL3 store, a chunk
/// of at most max_rows rows (decode, verify windows, short prompt chunks), two ranks (world 1 / canonical TP and wide
/// chunks keep exl3Up / exl3Down).
fn exl3DecOn(m: *const Model, wide: bool) bool {
    return m.exl3_dec and m.c.exl3 and !wide and m.world == 2;
}

/// The EXL3 experts of a chunk of up to max_rows rows at two ranks (exl3DecOn), after its plan, from the input
/// activation x.xm to every routed pair's fp32 partial row in x.y (written whole: no zeroing needed; moeTail reads it
/// either way: p4-xchg's token rows or the pair gather), the SwiGLU rows into `act_out` (bf16 [pair][Il]): five launches
/// on an NVFP4 layer (four on an MXFP8 one) and the amax gather, where exl3Up / exl3Down take ten, the gather and a
/// zeroing; every pair's act and y are the forms' bits (e-decode's tf-inkling2-exl3-dbench).
///   the rows' amax (NVFP4: exl3DecRowAmax into e_xq's first T floats: e_xq is not used on this path), w13's activation
///   and rotation (exl3DecRot13; it zeroes e_amax), w13 + its epilogue (exl3DecGu; each pair's max |act| into e_amax),
///   the gather of e_amax, w2's activation and rotation (exl3DecRot2), w2 + its epilogue (exl3DecDown).
fn exl3DecExperts(m: *Model, T: usize, plan: cuda.grouped.Plan, items: usize, ex: kern.Kernels.Exl3, mode: u8, act_out: u64) !void {
    const k = &m.k;
    const s = m.s;
    const x = &m.x;
    const H: usize = m.c.hidden;
    const topk: usize = m.c.topk;
    const pairs = T * topk;
    const Il = m.inter();
    const nv = mode == 1;
    if (nv) try k.exl3DecRowAmax(s, x.xm.ptr, H, T, H, x.e_xq.ptr);
    try k.exl3DecRot13(s, ex, x.xm.ptr, H, topk, mode, x.e_xq.ptr, plan, items, x.e_xr.ptr, if (nv) x.e_amax.ptr else 0);
    try k.exl3DecGu(s, ex, k.dec.cfg, x.e_xr.ptr, plan, items, act_out, Il, if (nv) x.e_amax.ptr else 0);
    if (nv) try m.group.?.allGather(s, x.e_amax.ptr, x.e_amax_all.ptr, pairs * 4);
    try k.exl3DecRot2(s, ex, act_out, Il, mode, x.e_amax_all.ptr, m.world, pairs, plan, items, x.e_xr.ptr);
    try k.exl3DecDown(s, ex, k.dec.cfg, x.e_xr.ptr, plan, items, x.y.ptr, H);
}
// ---- end e-decode
