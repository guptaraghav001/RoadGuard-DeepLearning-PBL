
package com.roadguard.neuralnetwork;

public class ArtificialNeuron {

    private final double[] weights;
    private final double bias;

    public ArtificialNeuron(double[] weights, double bias) {
        this.weights = weights.clone();
        this.bias = bias;
    }

    public double weightedSum(double[] inputs) {

        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(
                "Input and weight sizes must match"
            );
        }

        double sum = bias;

        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }

        return sum;
    }

    public int predict(double[] inputs) {
        return weightedSum(inputs) >= 0 ? 1 : 0;
    }

    public static void main(String[] args) {

        double[] weights = {1.0, 1.0};

        ArtificialNeuron neuron =
            new ArtificialNeuron(weights, -1.5);

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        System.out.println("Artificial Neuron - AND Gate");

        for (double[] input : inputs) {

            int output = neuron.predict(input);

            System.out.printf(
                "%.0f AND %.0f = %d%n",
                input[0], input[1], output
            );
        }
    }
}
