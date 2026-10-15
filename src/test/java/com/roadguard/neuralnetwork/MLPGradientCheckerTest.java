
package com.roadguard.neuralnetwork;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MLPGradientCheckerTest {

    @Test
    void testAllMLPGradients() {
        double error = MLPGradientChecker.check();

        assertTrue(
            Double.isFinite(error) && error < 1e-6,
            "All 13 gradients must agree"
        );
    }

    @Test
    void testParameterRoundTrip() {
        MultilayerPerceptron mlp =
            new MultilayerPerceptron();

        double[] original = mlp.getParameters();
        mlp.setParameters(original);

        assertArrayEquals(
            original, mlp.getParameters(), 0.0
        );
    }
}
