const std = @import("std");

const time = @cImport(@cInclude("time.h"));
const rand = @cImport(@cInclude("stdlib.h"));

pub const Random = struct {
    pub const random = rand.rand;
    pub fn init() void {
        rand.srand(@intCast(c_uint, time.time(0)));
    }
    pub fn floatRange(lower: f32, upper: f32) f32 {
        std.debug.assert(upper > lower);
        var num: u64 = @intCast(u64, rand.rand());
        return @mod(@intToFloat(f32, @truncate(u32, num))/1e5, upper-lower) + lower;
    }
};

// testing the random distribution

// var nums: [10000]f32 = undefined;
// var avg: f32 = 0;
// for (nums) |_, i| {
//     _ = i;
//     var num = Random.floatRange(0, 100);
//     avg += num;
// }
// std.debug.print("{}\n", .{@floatToInt(u64, avg/10000)});