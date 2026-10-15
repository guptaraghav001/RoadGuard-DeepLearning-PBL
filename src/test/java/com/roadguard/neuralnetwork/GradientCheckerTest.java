
package com.roadguard.neuralnetwork;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class GradientCheckerTest {

    @Test
    void testAnalyticalAndNumericalGradients() {

        double analytical =
            GradientChecker.analyticalGradient(
                0.5, 2.0, 1.0
            );

        double numerical =
            GradientChecker.numericalGradient(
                0.5, 2.0, 1.0
            );

        assertEquals(analytical, numerical, 1e-7);
    }

    @Test
    void testNegativeTargetGradient() {

        double analytical =
            GradientChecker.analyticalGradient(
                -0.3, 1.5, 0.0
            );

        double numerical =
            GradientChecker.numericalGradient(
                -0.3, 1.5, 0.0
            );

        assertEquals(analytical, numerical, 1e-7);
    }

    @Test
    void testZeroInputGradient() {

        double analytical =
            GradientChecker.analyticalGradient(
                0.7, 0.0, 1.0
            );

        double numerical =
            GradientChecker.numericalGradient(
                0.7, 0.0, 1.0
            );

        assertEquals(0.0, analytical, 1e-12);
        assertEquals(0.0, numerical, 1e-12);
    }
}
