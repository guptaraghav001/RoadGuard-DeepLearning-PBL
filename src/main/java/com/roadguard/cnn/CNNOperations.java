
package com.roadguard.cnn;

public class CNNOperations {

    private static void validate(double[][] input) {
        if (input == null || input.length == 0
                || input[0] == null
                || input[0].length == 0) {
            throw new IllegalArgumentException(
                "Input matrix cannot be empty"
            );
        }

        int width = input[0].length;

        for (double[] row : input) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException(
                    "Matrix must be rectangular"
                );
            }
            for (double value : row) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Matrix values must be finite"
                    );
                }
            }
        }
    }

    public static double[][] relu(double[][] input) {
        validate(input);

        double[][] output =
            new double[input.length][input[0].length];

        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                output[i][j] = Math.max(0, input[i][j]);
            }
        }

        return output;
    }

    public static double[][] maxPool(
            double[][] input, int poolSize, int stride) {

        validate(input);

        if (poolSize <= 0 || stride <= 0
                || poolSize > input.length
                || poolSize > input[0].length) {
            throw new IllegalArgumentException(
                "Invalid pooling configuration"
            );
        }

        int outputH =
            (input.length - poolSize) / stride + 1;

        int outputW =
            (input[0].length - poolSize) / stride + 1;

        double[][] output =
            new double[outputH][outputW];

        for (int i = 0; i < outputH; i++) {
            for (int j = 0; j < outputW; j++) {

                double max = Double.NEGATIVE_INFINITY;

                for (int pi = 0; pi < poolSize; pi++) {
                    for (int pj = 0; pj < poolSize; pj++) {
                        max = Math.max(
                            max,
                            input[i * stride + pi]
                                 [j * stride + pj]
                        );
                    }
                }

                output[i][j] = max;
            }
        }

        return output;
    }

    public static void printMatrix(double[][] matrix) {
        validate(matrix);

        for (double[] row : matrix) {
            for (double value : row) {
                System.out.printf("%8.2f", value);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {

        double[][] featureMap = {
            {-2, 3, 1, -1},
            {4, -1, 2, 5},
            {1, 6, -3, 2},
            {-2, 4, 1, 3}
        };

        double[][] activated = relu(featureMap);

        double[][] pooled = maxPool(
            activated, 2, 2
        );

        System.out.println("===== ReLU OUTPUT =====");
        printMatrix(activated);

        System.out.println("\n===== MAX POOLING OUTPUT =====");
        printMatrix(pooled);
    }
}
