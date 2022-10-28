const std = @import("std");

const Random = @import("random.zig").Random;
const Network = @import("network.zig").Network;


const numNetworks: u32 = 100;
const numBatches: u32 = 1000;
const numInputs: u32 = 4;
const numOutputs: u32 = 4;

const networkType = Network(numInputs, 3, 8, numOutputs);

pub fn main() !void {
    // init the random number generator
    Random.init();

    // create networks
    var networks: [numNetworks]networkType = undefined;
    for (networks) |_, i| {
        networks[i] = networkType.newNetwork();
        networks[i].setRandomWeights();
    }

    // create random data
    var data: [numBatches][numInputs]f32 = undefined;
    for (data) |_, i| {
        for (data[i]) |_, j| {
            data[i][j] = @intToFloat(f32, @floatToInt(u32, Random.floatRange(0, 100)) % 2);
        }
    }

    // var num: u32 = 0;
    while (true) : (_ = true) {
        // ask for each of the networks 
        // responses for each test case
        var networkResponses: [numNetworks][numBatches][numOutputs]f32 = undefined;
        for (networkResponses[0]) |_, i| { // for each batch
            for (networks) |_, j| { // for each network
                networkResponses[j][i] = networks[j].forward(data[i]);
            }
        }

        // calculate the accuracy of each network
        for (networkResponses) |_, i| {
            networks[i].fitness = fitness(data, networkResponses[i]);
        }
        
        // sort accuracies
        std.sort.sort(networkType, &networks, {}, fitnessSorter);

        // print the sorted accuracies
        std.debug.print("Top 3 network performances:\n", .{});
        for (networks[0..3]) |_, i| {
            std.debug.print("\t{}:{d:.3}\n", .{i, networks[i].fitness});
        }

        // copy the top network
        for (networks) |_, i| {
            if (i > 0) {
                networks[i] = networks[0];
            }
        }

        // mutate networks based on their ranking
        for (networks) |_, i| {
            if (i > 0) {
                networks[i].mutate(@intToFloat(f32, i)/10, @intCast(u32, i));
            }
        }
    }
}

// sorts the inputed networks by fitness
fn fitnessSorter(context: void, a: networkType, b: networkType) bool {
    _ = context;
    return a.fitness < b.fitness;
}

// given the inputs and the network's outputs,
// compute the expected output and a number
// indicating how close the network's output was
// to the expected output (lesser values mean
// less difference)
pub fn fitness(inputs: [numBatches][numInputs]f32, outputs: [numBatches][numOutputs]f32) f32 {
    var difference: f32 = 0;
    for (inputs) |_, i| {
        // calcuate the expected output from the input
        var input: u32 = 0;
        for (inputs[i]) |_, j| {
            input |= @floatToInt(u32, @round(inputs[i][j])) << @intCast(u5, j);
        }
        var expected: u32 = ~input;
        // compare actual outputs and add to difference
        for (outputs[i]) |_, j| {
            const correct: f32 = @intToFloat(f32, (expected >> @intCast(u5, j))%2);
            difference += @fabs(correct-outputs[i][j]);
        }
    }
    return difference / @intToFloat(f32, numBatches);
}

// pub fn main() !void {
//     // init the random number generator
//     Random.init();
//     // create a network
//     const networkType = Network(1, 1, 3, 1);
//     var network = networkType.newNetwork();
//     network.setRandomWeights();
//     network.setGeneticCode([_]f32{0,1,2,3,4,5,6,7,8});
//     _ = network.getGeneticCode();
//     // test the network
//     const output: [1]f32 = network.forward([_]f32{0});
//     for (output) |value, i| {
//         std.debug.print("{}: {d:.5}\n", .{i, value});
//     }
// }