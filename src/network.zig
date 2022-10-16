const std = @import("std");

const Node = @import("node.zig").Node;

pub fn Network(comptime numInputs: u32, comptime numHiddens: u32, comptime hiddenDepth: u32, comptime numOutputs: u32) type {
    return struct {
        // reference to our own type
        const self: type = Network(numInputs, numHiddens, hiddenDepth, numOutputs);
        // neuron types
        const InputNode = Node(numHiddens);
        const HiddenNode = Node(numHiddens);
        const LastHiddenNode = Node(numOutputs);
        const OutputNode = Node(0);
        // layer of input neurons
        inputs: [numInputs]InputNode,
        // layer(s) of hidden neurons
        hiddens: [hiddenDepth-1][numHiddens]HiddenNode,
        // last layer of hidden neurons
        lastHiddens: [numHiddens]LastHiddenNode,
        // layer of output neurons
        outputs: [numOutputs]OutputNode,

        // create a new network with the
        // specified super parameters
        pub fn newNetwork() self {
            var inputs: [numInputs]InputNode = initLayer(InputNode, numInputs);

            var hiddens: [hiddenDepth-1][numHiddens]HiddenNode = undefined;
            var i: u32 = 0;
            while (i < hiddens.len) : (i+=1) {
                hiddens[i] = initLayer(HiddenNode, numHiddens);
            }
            // last hidden layer needs to be manually inited
            var lastHiddens: [numHiddens]LastHiddenNode = initLayer(LastHiddenNode, numHiddens);

            var outputs: [numOutputs]OutputNode = initLayer(OutputNode, numOutputs);

            return self {
                .inputs = inputs,
                .hiddens = hiddens,
                .lastHiddens = lastHiddens,
                .outputs = outputs,
            };
        }
        // init a layer to have all its
        // weights and biases set to 0
        pub fn initLayer(comptime T: type, comptime numLayer: u32) [numLayer]T {
            var layer: [numLayer]T = undefined;
            for (layer) |_, i| {
                layer[i] = T.newNode();
            }
            return layer;
        }

        // calculates the value of each node in
        // the specified layer given the previous
        // layer
        pub fn calculateLayerWeightedSum(comptime LayerNodeType: type, comptime numLayer: u32, comptime PrevLayerNodeType: type, comptime numPrevLayer: u32, layer: *[numLayer]LayerNodeType, prevLayer: [numPrevLayer]PrevLayerNodeType) void {
            for (layer) |_, i| {
                var node: *LayerNodeType = &layer[i];
                var values: [numPrevLayer]f32 = undefined;
                for (prevLayer) |_, j| {
                    const curInput = prevLayer[j];
                    const weightedSum = curInput.value * curInput.weights[i];
                    values[j] = weightedSum;
                }
                node.value = node.compute(&values);
            }
        }
        // feed input through this network and
        // return the output layer of neurons
        pub fn forward(networkPointer: *self, input: [numInputs]f32) [numOutputs]f32 {
            var network = networkPointer.*;
            // set the input neurons to have
            // the passed values
            for (network.inputs) |_, i| {
                network.inputs[i].value = input[i];
            }
            // pass the values of the input neurons into
            // the first layer of hidden neurons
            calculateLayerWeightedSum(HiddenNode, numHiddens, InputNode, numInputs, &network.hiddens[0], network.inputs);
            // for each of the remaining hidden layers, collect the
            // values of the neurons in the previous layer,
            // calculate each hidden neuron's value and pass
            // it to the next hidden layer
            for (network.hiddens[1..hiddenDepth-1]) |_, i| {
                var layer: *[numHiddens]HiddenNode = &network.hiddens[i+1];
                calculateLayerWeightedSum(HiddenNode, numHiddens, HiddenNode, numHiddens, layer, network.hiddens[i]);
            }
            // pass the values of the final hidden layer (index hiddenDepth-2) 
            // into the last hidden layer (index hiddenDepth-1)
            calculateLayerWeightedSum(LastHiddenNode, numHiddens, HiddenNode, numHiddens, &network.lastHiddens, network.hiddens[hiddenDepth-2]);
            // pass the values of the last hidden layer into the output layer
            calculateLayerWeightedSum(OutputNode, numOutputs, LastHiddenNode, numHiddens, &network.outputs, network.lastHiddens);
            // return the output layer values
            var outputValues: [numOutputs]f32 = undefined;
            for (network.outputs) |_, i| {
                outputValues[i] = network.outputs[i].value;
            }
            return outputValues;
        }
        // turn this network into a portable genetic code
        pub fn getGeneticCode(network: self) []f32 {
            _ = network;
        }
    };
}