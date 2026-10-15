
package com.roadguard.neuralnetwork;

public class SigmoidNeuron {

    private final double[] weights;
    private final double bias;

    public SigmoidNeuron(double[] weights, double bias) {
        this.weights = weights.clone();
        this.bias = bias;
    }

    public static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        double expZ = Math.exp(z);
        return expZ / (1.0 + expZ);
    }

    public double forward(double[] inputs) {

        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(
                "Input and weight sizes must match"
            );
        }

        double z = bias;

        for (int i = 0; i < inputs.length; i++) {
            z += inputs[i] * weights[i];
        }

        return sigmoid(z);
    }

    public static void main(String[] args) {

        SigmoidNeuron neuron = new SigmoidNeuron(
            new double[]{0.5, 0.3}, 0.2
        );

        double[] inputs = {2.0, 3.0};

        double output = neuron.forward(inputs);

        System.out.printf(
            "Sigmoid neuron output: %.6f%n", output
        );
    }
}
