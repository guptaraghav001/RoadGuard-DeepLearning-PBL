
package com.roadguard.neuralnetwork;

import java.util.Random;

public class MultilayerPerceptron {

    private final double[][] hiddenWeights;
    private final double[] hiddenBiases;

    private final double[] outputWeights;
    private final double outputBias;

    public MultilayerPerceptron() {

        Random random = new Random(42);

        hiddenWeights = new double[3][2];
        hiddenBiases = new double[3];

        outputWeights = new double[3];

        // Initialize hidden layer weights
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                hiddenWeights[i][j] =
                    random.nextDouble() * 2 - 1;
            }

            hiddenBiases[i] = 0.0;
        }

        // Initialize output layer weights
        for (int i = 0; i < 3; i++) {
            outputWeights[i] =
                random.nextDouble() * 2 - 1;
        }

        outputBias = 0.0;
    }

    private double sigmoid(double z) {

        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        double expZ = Math.exp(z);
        return expZ / (1.0 + expZ);
    }

    public double forward(double[] inputs) {

        if (inputs == null || inputs.length != 2) {
            throw new IllegalArgumentException(
                "MLP requires exactly 2 inputs"
            );
        }

        double[] hiddenOutputs = new double[3];

        // Hidden layer forward propagation
        for (int i = 0; i < 3; i++) {

            double sum = hiddenBiases[i];

            for (int j = 0; j < 2; j++) {
                sum += hiddenWeights[i][j] * inputs[j];
            }

            hiddenOutputs[i] = sigmoid(sum);
        }

        // Output layer forward propagation
        double outputSum = outputBias;

        for (int i = 0; i < 3; i++) {
            outputSum +=
                outputWeights[i] * hiddenOutputs[i];
        }

        return sigmoid(outputSum);
    }

    public static void main(String[] args) {

        MultilayerPerceptron mlp =
            new MultilayerPerceptron();

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        System.out.println(
            "ROADGUARD - Multilayer Perceptron"
        );

        for (double[] input : inputs) {

            double output = mlp.forward(input);

            System.out.printf(
                "Input: [%.0f, %.0f] -> Output: %.6f%n",
                input[0], input[1], output
            );
        }
    }
}
