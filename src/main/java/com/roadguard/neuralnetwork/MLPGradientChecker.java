
package com.roadguard.neuralnetwork;

public class MLPGradientChecker {

    private static final double EPS = 1e-5;

    public static double check() {

        MultilayerPerceptron mlp =
            new MultilayerPerceptron();

        double[] input = {1.0, 0.0};
        double target = 1.0;

        double[] parameters = mlp.getParameters();
        double[] analytical =
            mlp.analyticalGradients(input, target);

        double maxError = 0.0;

        System.out.println(
            "===== FULL MLP GRADIENT CHECK ====="
        );

        try {
            for (int i = 0; i < parameters.length; i++) {

                double original = parameters[i];

                parameters[i] = original + EPS;
                mlp.setParameters(parameters);

                double lossPlus = mlp.averageLoss(
                    new double[][]{input},
                    new double[]{target}
                );

                parameters[i] = original - EPS;
                mlp.setParameters(parameters);

                double lossMinus = mlp.averageLoss(
                    new double[][]{input},
                    new double[]{target}
                );

                parameters[i] = original;
                mlp.setParameters(parameters);

                double numerical =
                    (lossPlus - lossMinus) / (2 * EPS);

                double error =
                    Math.abs(analytical[i] - numerical)
                    / Math.max(
                        1e-8,
                        Math.abs(analytical[i])
                        + Math.abs(numerical)
                    );

                maxError = Math.max(maxError, error);

                System.out.printf(
                    "Parameter %02d | Analytical: %.9f "
                    + "| Numerical: %.9f | Error: %.3e%n",
                    i + 1,
                    analytical[i],
                    numerical,
                    error
                );
            }
        } finally {
            mlp.setParameters(parameters);
        }

        System.out.printf(
            "Maximum relative error: %.3e%n",
            maxError
        );

        System.out.println(
            maxError < 1e-6
                ? "FULL MLP GRADIENT CHECK PASSED"
                : "FULL MLP GRADIENT CHECK FAILED"
        );

        return maxError;
    }

    public static void main(String[] args) {
        double error = check();

        if (error >= 1e-6)
            throw new IllegalStateException(
                "Gradient verification failed"
            );
    }
}
