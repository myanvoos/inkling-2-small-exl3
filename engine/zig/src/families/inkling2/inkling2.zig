//! Inkling-2-Small (model_type inkling2) on CUDA in Zig: the family's modules. Bring-up stage: config and TP plan.
pub const config = @import("config.zig");
pub const plan = @import("plan.zig");
pub const dense = @import("dense.zig");
pub const experts = @import("experts.zig");
pub const kern = @import("kern.zig");
pub const gemm = @import("gemm.zig");
pub const model = @import("model.zig");
pub const forward = @import("forward.zig");
pub const fill = @import("fill.zig");
pub const residency = @import("residency.zig");
pub const adapt = @import("adapt.zig");
pub const dspark = @import("dspark.zig");
pub const warm = @import("warm.zig");
pub const lazy = @import("lazy.zig"); // p4-mem
pub const exl3 = @import("exl3.zig");
pub const mix = @import("mix.zig"); // x-mixed
pub const page_cache = @import("cache.zig"); // e-load (x5): the page cache a load leaves, dropped
pub const dense_exl3 = @import("dense_exl3.zig"); // e-dense
pub const hcap = @import("hcap.zig"); // e-dense
pub const defaults = @import("defaults.zig"); // the user, 2026-11-15: everything on by default
pub const Config = config.Config;

test {
    const refAll = @import("std").testing.refAllDecls;
    refAll(config);
    refAll(plan);
    refAll(dense);
    refAll(experts);
    refAll(residency);
    _ = mix; // x-mixed: the bank's sizing test
    _ = exl3; // its tests (slot geometry, packSlot, the header reader); Store needs the GPU model, not analyzed here
    _ = page_cache; // e-load (x5): /proc/meminfo, a drop pass
    _ = dense_exl3; // e-dense: class parsing, the canonical halves' addressing
}
