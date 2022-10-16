const std = @import("std");

pub fn sigmoid(input: f32) f32 {
    return 1/(1+(std.math.pow(f32, std.math.e, input)));
}