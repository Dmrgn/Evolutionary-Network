const std = @import("std");

const Random = @import("random.zig").Random;
const Sigmoid = @import("sigmoid.zig");

pub fn Node(comptime numWeights: u32) type {
    return struct {
        // reference to our own type
        const self: type = Node(numWeights);
        // the current value of this node
        value: f32,
        // the weights of this node
        weights: [numWeights]f32,
        // the bias of this node
        bias: f32,

        // create a new node with 0 for 
        // the weights and biases
        pub fn newNode() self {
            var weights: [numWeights]f32 = undefined;
            for (weights) |_, i| {
                weights[i] = Random.floatRange(-10, 10);
            }
            return self {
                .value = 0,
                .weights = weights,
                .bias = 0
            };
        }

        // compute this node's value given
        // the previous layer's weighted values
        pub fn compute(node:self, prevLayer: []f32) f32 {
            // sum weighted values
            var value: f32 = 0;
            for (prevLayer) |input| {
                value += input;
            }
            // add this node's bias
            value += node.bias;
            // sigmoid the result
            return Sigmoid.sigmoid(value);
        }
        // turn this node into a portable genetic code
        pub fn getGeneticCode(node: self) []f32 {
            _ = node;
        }
    };
}