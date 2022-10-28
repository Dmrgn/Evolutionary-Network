const std = @import("std");

const Random = @import("random.zig").Random;
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
        // the length of this network type's genetic code
        const numInputGenes: u32 = numInputs*numHiddens + numInputs;
        const numHiddenGenes: u32 = (hiddenDepth-1)*(numHiddens*numHiddens + numHiddens);
        const numLastHiddenGenes: u32 = numHiddens*numOutputs + numHiddens;
        const numOutputGenes: u32 = numOutputs;
        const numGenes: u32 = numInputGenes + numHiddenGenes + numLastHiddenGenes + numOutputGenes;
        // layer of input neurons
        inputs: [numInputs]InputNode,
        // layer(s) of hidden neurons
        hiddens: [hiddenDepth-1][numHiddens]HiddenNode,
        // last layer of hidden neurons
        lastHiddens: [numHiddens]LastHiddenNode,
        // layer of output neurons
        outputs: [numOutputs]OutputNode,
        // the networks current fitness
        fitness: f32 = 0,

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
        // sets all the weights in the specified layer to random values
        pub fn setRandomLayerWeights(comptime NodeType: type, comptime numLayer: u32, comptime numNextLayer: u32, layer: *[numLayer]NodeType) void {
            for (layer) |_, i| {
                const weights: [numNextLayer]f32 = Random.randomFloatArray(numNextLayer, -10, 10);
                layer.*[i].weights = weights;
            }
        }

        // sets all of the weights in the network to random values
        pub fn setRandomWeights(network: *self) void {
            setRandomLayerWeights(InputNode, numInputs, numHiddens, &network.inputs);
            for (network.hiddens) |_, i| {
                setRandomLayerWeights(HiddenNode, numHiddens, numHiddens, &network.hiddens[i]);
            }
            setRandomLayerWeights(LastHiddenNode, numHiddens, numOutputs, &network.lastHiddens);
        }
        // feed input through this network and
        // return the output layer of neurons
        pub fn forward(network: *self, input: [numInputs]f32) [numOutputs]f32 {
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
        // mutate the genetic code of this network by the specified amount
        pub fn mutate(network: *self, factor: f32, amount: u32) void {
            var code: [numGenes]f32 = network.getGeneticCode();
            var num: u32 = 0;
            while (num < amount) : (num += 1) {
                code[@floatToInt(u32, Random.floatRange(0, 1000000)) % numGenes] += Random.floatRange(-factor, factor);
            }
            network.setGeneticCode(code);
        }
        // get the genetic code of the specified layer
        pub fn getLayerGeneticCode(comptime NodeType: type, comptime numLayer: u32, layer: [numLayer]NodeType, index: *u32, code: *[numGenes]f32) void {
            for (layer) |_, i| {
                for (layer[i].weights) |_, j| {
                    code[index.*] = layer[i].weights[j];
                    index.*+=1;
                }
                code[index.*] = layer[i].bias;
                index.*+=1;
            }
        }
        // sets the genetic code of the specified layer
        pub fn setLayerGeneticCode(comptime NodeType: type, comptime numLayer: u32, layer: *[numLayer]NodeType, index: *u32, code: [numGenes]f32) void {
            for (layer) |_, i| {
                for (layer[i].weights) |_, j| {
                    layer[i].weights[j] = code[index.*];
                    index.*+=1;
                }
                layer[i].bias = code[index.*];
                index.*+=1;
            }
        }
        // turn this network into a portable genetic code
        pub fn getGeneticCode(network: self) [numGenes]f32 {
            var index: u32 = 0;
            var code: [numGenes]f32 = undefined;
            getLayerGeneticCode(InputNode, numInputs, network.inputs, &index, &code);
            for (network.hiddens) |_, i| {
                getLayerGeneticCode(HiddenNode, numHiddens, network.hiddens[i], &index, &code);
            }
            getLayerGeneticCode(LastHiddenNode, numHiddens, network.lastHiddens, &index, &code);
            for (network.outputs) |_, i| {
                code[index] = network.outputs[i].bias;
                index+=1;
            }
            return code;
        }
        // set this network's weights and biases to the values
        // in the specified genetic code
        pub fn setGeneticCode(network: *self, code: [numGenes]f32) void {
            var index: u32 = 0;
            setLayerGeneticCode(InputNode, numInputs, &network.inputs, &index, code);
            for (network.hiddens) |_, i| {
                setLayerGeneticCode(HiddenNode, numHiddens, &network.hiddens[i], &index, code);
            }
            setLayerGeneticCode(LastHiddenNode, numHiddens, &network.lastHiddens, &index, code);
            for (network.outputs) |_, i| {
                network.outputs[i].bias = code[index];
                index+=1;
            }
        }
        // turn this network's genetic code into portable
        // u8 characters to be written to a file
        // pub fn getU8GeneticCode(network: *self, ) [32*numGenes]u8 {
        //     const code: [numGenes]f32 = network.getGeneticCode();
        //     var binaryCode: [32*numGenes]u8 = undefined;

        // }
    };
}