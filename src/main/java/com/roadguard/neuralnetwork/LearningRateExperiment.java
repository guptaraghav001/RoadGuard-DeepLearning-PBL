
package com.roadguard.neuralnetwork;

public class LearningRateExperiment {

    public static void main(String[] args) {

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[] targets = {0, 1, 1, 0};

        double[] learningRates = {
            0.01, 0.1, 0.5
        };

        int epochs = 2000;

        System.out.println(
            "===== LEARNING RATE EXPERIMENT ====="
        );

        for (double rate : learningRates) {

            // Same random seed for fair comparison
            MultilayerPerceptron mlp =
                new MultilayerPerceptron();

            double initialLoss =
                mlp.averageLoss(inputs, targets);

            for (int epoch = 0; epoch < epochs; epoch++) {

                for (int i = 0; i < inputs.length; i++) {
                    mlp.trainSample(
                        inputs[i], targets[i], rate
                    );
                }
            }

            double finalLoss =
                mlp.averageLoss(inputs, targets);

            int correct = 0;

            for (int i = 0; i < inputs.length; i++) {

                double prediction =
                    mlp.forward(inputs[i]);

                int predictedClass =
                    prediction >= 0.5 ? 1 : 0;

                if (predictedClass == (int) targets[i]) {
                    correct++;
                }
            }

            System.out.printf(
                "Learning rate: %.2f | "
                + "Initial loss: %.6f | "
                + "Final loss: %.6f | "
                + "Accuracy: %.1f%%%n",
                rate,
                initialLoss,
                finalLoss,
                100.0 * correct / inputs.length
            );
        }
    }
}
