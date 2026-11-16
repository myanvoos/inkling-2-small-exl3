//! p4-cache: a request resumes from the state the last request left (TF_INKLING2_PROMPT_REUSE=1; default 0: every
//! request prefills from position 0, phase 3's behaviour). Rank 0 decides where a prompt pass starts (reuse_plan.zig:
//! continue the live state, restore a snapshot of an earlier prompt's end, or position 0) and tells the other ranks
//! with the `begin` step (engine.zig); every rank keeps the same snapshot slots here and saves into one with the
//! prompt's `prefilled` step. A slot holds what a pass resuming at its position needs beyond what stays in place
//! (Model.saveState: the convolution rings and the sliding layers' last window of K / V; Draft.saveContext: the draft
//! head's sliding windows and f_rnn); the global layers' K / V and the draft's global context are positional and
//! stay in the caches. Settings: TF_INKLING2_PROMPT_REUSE_SLOTS (1 .. 4, default 2) and TF_INKLING2_PROMPT_REUSE_MIN
//! (the fewest shared ids worth a resume, default 64).
//! Numerics: a resumed state is the one the earlier request computed (its chunking, its arena; a reply's rows from
//! verify windows), not the bits a fresh prefill of the same prompt makes.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const plan_mod = @import("reuse_plan.zig");

const Buf = cuda.DeviceBuffer;
const Model = ink.model.Model;
const Draft = ink.dspark.Draft;

pub const Book = plan_mod.Book;
pub const Plan = plan_mod.Plan;
pub const max_slots = plan_mod.max_slots;

/// Slots asked for: 0 with the switch off, else TF_INKLING2_PROMPT_REUSE_SLOTS (default 2, at most max_slots).
pub fn slotsWanted(m: *const Model) usize {
    if (!m.prompt_reuse) return 0;
    const v = std.c.getenv("TF_INKLING2_PROMPT_REUSE_SLOTS") orelse return 2;
    const n = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return 2;
    return std.math.clamp(n, 1, max_slots);
}

/// TF_INKLING2_PROMPT_REUSE_MIN: the fewest ids a resume reuses (default 64).
pub fn minShared() usize {
    const v = std.c.getenv("TF_INKLING2_PROMPT_REUSE_MIN") orelse return 64;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 64;
}

/// A rank's snapshot slots on the device.
pub const Slots = struct {
    n: usize = 0,
    at: [max_slots]usize = @splat(0),
    rings: [max_slots]Buf = undefined,
    window: [max_slots]Buf = undefined,
    ctx: [max_slots]Buf = undefined, // the draft head's (none without one)

    /// Device bytes one slot takes on this rank.
    pub fn bytesEach(m: *const Model, draft: ?*const Draft) usize {
        return m.snapRingBytes() + m.snapWindowBytes() + if (draft) |d| d.snapBytes() else 0;
    }

    pub fn alloc(m: *const Model, draft: ?*const Draft, n: usize) !Slots {
        var sl: Slots = .{};
        errdefer sl.free();
        for (0..@min(n, max_slots)) |i| {
            sl.rings[i] = try Buf.alloc(m.d, m.snapRingBytes());
            sl.window[i] = Buf.alloc(m.d, m.snapWindowBytes()) catch |err| {
                sl.rings[i].free();
                return err;
            };
            sl.ctx[i] = Buf.alloc(m.d, if (draft) |d| d.snapBytes() else 0) catch |err| {
                sl.rings[i].free();
                sl.window[i].free();
                return err;
            };
            sl.n = i + 1;
        }
        return sl;
    }

    pub fn free(sl: *Slots) void {
        for (0..sl.n) |i| {
            sl.rings[i].free();
            sl.window[i].free();
            sl.ctx[i].free();
        }
        sl.n = 0;
    }

    /// The state at the model's length into slot `i` (copies on the model's stream).
    pub fn save(sl: *Slots, m: *Model, draft: ?*Draft, i: usize) !void {
        if (i >= sl.n) return error.NoReuseSlot;
        try m.saveState(sl.rings[i], sl.window[i]);
        if (draft) |d| try d.saveContext(m.state.len, sl.ctx[i]);
        sl.at[i] = m.state.len;
    }

    /// Slot `i`'s state back in place: the model's length becomes its position.
    pub fn restore(sl: *Slots, m: *Model, draft: ?*Draft, i: usize) !void {
        if (i >= sl.n) return error.NoReuseSlot;
        try m.restoreState(sl.at[i], sl.rings[i], sl.window[i]);
        if (draft) |d| try d.restoreContext(sl.at[i], sl.ctx[i]);
    }
};
