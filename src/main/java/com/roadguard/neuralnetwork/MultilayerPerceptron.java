
package com.roadguard.neuralnetwork;

import java.util.Random;

public class MultilayerPerceptron {

    private final double[][] hiddenWeights = new double[3][2];
    private final double[] hiddenBiases = new double[3];

    private final double[] outputWeights = new double[3];
    private double outputBias = 0.0;

    public MultilayerPerceptron() {
        Random random = new Random(42);

        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 2; i++) {
                hiddenWeights[j][i] =
                    random.nextDouble() * 2 - 1;
            }
            outputWeights[j] =
                random.nextDouble() * 2 - 1;
        }
    }

    private static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        }
        double e = Math.exp(z);
        return e / (1.0 + e);
    }

    private static void checkInputs(double[] inputs) {
        if (inputs == null || inputs.length != 2) {
            throw new IllegalArgumentException(
                "Exactly two inputs required"
            );
        }
        for (double value : inputs) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "Inputs must be finite"
                );
            }
        }
    }

    private double[] hiddenForward(double[] inputs) {
        double[] hidden = new double[3];

        for (int j = 0; j < 3; j++) {
            double z = hiddenBiases[j];

            for (int i = 0; i < 2; i++) {
                z += hiddenWeights[j][i] * inputs[i];
            }

            hidden[j] = sigmoid(z);
        }
        return hidden;
    }

    private double outputForward(double[] hidden) {
        double z = outputBias;

        for (int j = 0; j < 3; j++) {
            z += outputWeights[j] * hidden[j];
        }

        return sigmoid(z);
    }

    public double forward(double[] inputs) {
        checkInputs(inputs);
        return outputForward(hiddenForward(inputs));
    }

    public double trainSample(
            double[] inputs,
            double target,
            double learningRate) {

        checkInputs(inputs);

        if ((target != 0.0 && target != 1.0)
                || !Double.isFinite(learningRate)
                || learningRate <= 0) {
            throw new IllegalArgumentException(
                "Invalid target or learning rate"
            );
        }

        double[] hidden = hiddenForward(inputs);
        double prediction = outputForward(hidden);

        // Binary cross-entropy loss
        double p = Math.max(
            1e-12, Math.min(1 - 1e-12, prediction)
        );

        double loss = -(
            target * Math.log(p)
            + (1 - target) * Math.log(1 - p)
        );

        // Output-layer gradient
        double outputDelta = prediction - target;

        // Hidden-layer gradients
        double[] hiddenDeltas = new double[3];

        for (int j = 0; j < 3; j++) {
            hiddenDeltas[j] =
                outputDelta
                * outputWeights[j]
                * hidden[j]
                * (1 - hidden[j]);
        }

        // Update output weights
        for (int j = 0; j < 3; j++) {
            outputWeights[j] -=
                learningRate * outputDelta * hidden[j];
        }

        outputBias -= learningRate * outputDelta;

        // Update hidden weights and biases
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 2; i++) {
                hiddenWeights[j][i] -=
                    learningRate * hiddenDeltas[j] * inputs[i];
            }

            hiddenBiases[j] -=
                learningRate * hiddenDeltas[j];
        }

        return loss;
    }

    public double averageLoss(
            double[][] inputs, double[] targets) {

        if (inputs.length == 0 ||
                inputs.length != targets.length) {
            throw new IllegalArgumentException(
                "Invalid dataset dimensions"
            );
        }

        double total = 0;

        for (int i = 0; i < inputs.length; i++) {
            double p = forward(inputs[i]);
            p = Math.max(1e-12, Math.min(1 - 1e-12, p));

            total -= targets[i] * Math.log(p)
                + (1 - targets[i]) * Math.log(1 - p);
        }

        return total / inputs.length;
    }

    public static void main(String[] args) {

        double[][] inputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        double[] targets = {0, 1, 1, 0};

        MultilayerPerceptron mlp =
            new MultilayerPerceptron();

        double learningRate = 0.5;
        int epochs = 20000;

        System.out.printf(
            "Initial loss: %.6f%n",
            mlp.averageLoss(inputs, targets)
        );

        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                mlp.trainSample(
                    inputs[i], targets[i], learningRate
                );
            }

            if (epoch % 2000 == 0) {
                System.out.printf(
                    "Epoch %d | Loss: %.6f%n",
                    epoch,
                    mlp.averageLoss(inputs, targets)
                );
            }
        }

        System.out.println("\nXOR predictions:");

        for (double[] input : inputs) {
            double prediction = mlp.forward(input);

            System.out.printf(
                "%.0f XOR %.0f = %.6f (class %d)%n",
                input[0], input[1],
                prediction,
                prediction >= 0.5 ? 1 : 0
            );
        }
    }


    // Returns all 13 parameters in a fixed order.
    public double[] getParameters() {
        double[] p = new double[13];
        int k = 0;

        for (int j = 0; j < 3; j++)
            for (int i = 0; i < 2; i++)
                p[k++] = hiddenWeights[j][i];

        for (int j = 0; j < 3; j++)
            p[k++] = hiddenBiases[j];

        for (int j = 0; j < 3; j++)
            p[k++] = outputWeights[j];

        p[k] = outputBias;
        return p;
    }

    public void setParameters(double[] p) {
        if (p == null || p.length != 13)
            throw new IllegalArgumentException(
                "Exactly 13 parameters required"
            );

        for (double value : p)
            if (!Double.isFinite(value))
                throw new IllegalArgumentException(
                    "Parameters must be finite"
                );

        int k = 0;

        for (int j = 0; j < 3; j++)
            for (int i = 0; i < 2; i++)
                hiddenWeights[j][i] = p[k++];

        for (int j = 0; j < 3; j++)
            hiddenBiases[j] = p[k++];

        for (int j = 0; j < 3; j++)
            outputWeights[j] = p[k++];

        outputBias = p[k];
    }

    // Analytical gradient for one training example.
    // Parameter ordering matches getParameters().
    public double[] analyticalGradients(
            double[] inputs, double target) {

        checkInputs(inputs);

        if (target != 0.0 && target != 1.0)
            throw new IllegalArgumentException(
                "Binary target required"
            );

        double[] hidden = hiddenForward(inputs);
        double prediction = outputForward(hidden);

        double outputDelta = prediction - target;
        double[] hiddenDelta = new double[3];

        for (int j = 0; j < 3; j++) {
            hiddenDelta[j] =
                outputDelta * outputWeights[j]
                * hidden[j] * (1.0 - hidden[j]);
        }

        double[] gradients = new double[13];
        int k = 0;

        for (int j = 0; j < 3; j++)
            for (int i = 0; i < 2; i++)
                gradients[k++] = hiddenDelta[j] * inputs[i];

        for (int j = 0; j < 3; j++)
            gradients[k++] = hiddenDelta[j];

        for (int j = 0; j < 3; j++)
            gradients[k++] = outputDelta * hidden[j];

        gradients[k] = outputDelta;

        return gradients;
    }



}
