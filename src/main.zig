const std = @import("std");

const Random = @import("random.zig").Random;
const Network = @import("network.zig").Network;

pub fn main() !void {
    // init the random number generator
    Random.init();
    // create a network
    const networkType = Network(2, 3, 100, 2);
    var network = networkType.newNetwork();
    // test the network
    const output: [2]f32 = network.forward([2]f32{1, 0});
    for (output) |value, i| {
        std.debug.print("{}: {d:.5}\n", .{i, value});
    }
}