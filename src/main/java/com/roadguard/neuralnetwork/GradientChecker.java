
package com.roadguard.neuralnetwork;

public class GradientChecker {

    private static final double EPSILON = 1e-5;

    public static double sigmoid(double z) {
        if (z >= 0) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        double expZ = Math.exp(z);
        return expZ / (1.0 + expZ);
    }

    // Numerically stable BCE from the logit
    public static double loss(
            double weight, double input, double target) {

        double z = weight * input;

        return Math.max(z, 0.0)
            - z * target
            + Math.log1p(Math.exp(-Math.abs(z)));
    }

    public static double analyticalGradient(
            double weight, double input, double target) {

        double prediction = sigmoid(weight * input);

        return (prediction - target) * input;
    }

    public static double numericalGradient(
            double weight, double input, double target) {

        double lossPlus = loss(
            weight + EPSILON, input, target
        );

        double lossMinus = loss(
            weight - EPSILON, input, target
        );

        return (lossPlus - lossMinus) / (2 * EPSILON);
    }

    public static double relativeError(
            double analytical, double numerical) {

        return Math.abs(analytical - numerical)
            / Math.max(
                1e-8,
                Math.abs(analytical) + Math.abs(numerical)
            );
    }

    public static void main(String[] args) {

        double weight = 0.5;
        double input = 2.0;
        double target = 1.0;

        double analytical = analyticalGradient(
            weight, input, target
        );

        double numerical = numericalGradient(
            weight, input, target
        );

        double error = relativeError(
            analytical, numerical
        );

        System.out.printf(
            "Analytical gradient: %.10f%n", analytical
        );

        System.out.printf(
            "Numerical gradient:  %.10f%n", numerical
        );

        System.out.printf(
            "Relative error:      %.12f%n", error
        );

        System.out.println(
            error < 1e-7
                ? "GRADIENT CHECK PASSED"
                : "GRADIENT CHECK FAILED"
        );
    }
}
