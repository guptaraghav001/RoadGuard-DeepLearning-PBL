package com.roadguard.neuralnetwork;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MultilayerPerceptronTest {

    @Test
    void testOutputRange() {

        MultilayerPerceptron mlp =
            new MultilayerPerceptron();

        double output =
            mlp.forward(new double[]{1.0, 0.0});

        assertTrue(output > 0.0 && output < 1.0);
    }

    @Test
    void testReproducibility() {

        MultilayerPerceptron mlp1 =
            new MultilayerPerceptron();

        MultilayerPerceptron mlp2 =
            new MultilayerPerceptron();

        double[] input = {1.0, 1.0};

        assertEquals(
            mlp1.forward(input),
            mlp2.forward(input),
            1e-12
        );
    }

    @Test
    void testInvalidInputSize() {

        MultilayerPerceptron mlp =
            new MultilayerPerceptron();

        assertThrows(
            IllegalArgumentException.class,
            () -> mlp.forward(new double[]{1.0})
        );
    }

    @Test
    void testAllBinaryInputs() {

        MultilayerPerceptron mlp =
            new MultilayerPerceptron();

        double[][] inputs = {
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };

        for (double[] input : inputs) {
            double output = mlp.forward(input);
            assertTrue(Double.isFinite(output));
        }
    }



@Test
void testTrainingReducesLoss() {

    double[][] inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    double[] targets = {0, 1, 1, 0};

    MultilayerPerceptron mlp =
        new MultilayerPerceptron();

    double initialLoss =
        mlp.averageLoss(inputs, targets);

    for (int epoch = 0; epoch < 2000; epoch++) {
        for (int i = 0; i < inputs.length; i++) {
            mlp.trainSample(inputs[i], targets[i], 0.5);
        }
    }

    double finalLoss =
        mlp.averageLoss(inputs, targets);

    assertTrue(
        finalLoss < initialLoss,
        "Training should reduce the loss"
    );
}

}
