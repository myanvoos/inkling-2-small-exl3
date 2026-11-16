//! The user, 2026-11-15: "If output matches bit by bit why turn it off by default? ... Turn everything on by
//! default". Every speed switch defaults ON (the deployment's settings); TF_INKLING2_LEGACY=1 gives back the old
//! defaults (each switch off: the phase-3 / EXL3-lane kernels) for the bitwise gates. A switch set explicitly always
//! wins over both. (out/prefill/p5/defaults-on.py lists the switches.)
const std = @import("std");

/// TF_INKLING2_LEGACY=1 (anything but 0 and empty): the old defaults.
pub fn legacy() bool {
    const v = std.mem.span(std.c.getenv("TF_INKLING2_LEGACY") orelse return false);
    return v.len > 0 and !std.mem.eql(u8, v, "0");
}
