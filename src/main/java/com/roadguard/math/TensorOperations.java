
package com.roadguard.math;

public class TensorOperations {

    private static void validate(double[][][] tensor) {
        if (tensor == null || tensor.length == 0
                || tensor[0] == null
                || tensor[0].length == 0
                || tensor[0][0] == null
                || tensor[0][0].length == 0) {
            throw new IllegalArgumentException(
                "Tensor cannot be empty"
            );
        }

        int rows = tensor[0].length;
        int cols = tensor[0][0].length;

        for (double[][] slice : tensor) {
            if (slice == null || slice.length != rows) {
                throw new IllegalArgumentException(
                    "Tensor shape must be rectangular"
                );
            }

            for (double[] row : slice) {
                if (row == null || row.length != cols) {
                    throw new IllegalArgumentException(
                        "Tensor shape must be rectangular"
                    );
                }

                for (double value : row) {
                    if (!Double.isFinite(value)) {
                        throw new IllegalArgumentException(
                            "Tensor values must be finite"
                        );
                    }
                }
            }
        }
    }

    public static double[][][] add(
            double[][][] a, double[][][] b) {

        validate(a);
        validate(b);

        if (a.length != b.length
                || a[0].length != b[0].length
                || a[0][0].length != b[0][0].length) {
            throw new IllegalArgumentException(
                "Tensor dimensions must match"
            );
        }

        double[][][] result =
            new double[a.length][a[0].length][a[0][0].length];

        for (int d = 0; d < a.length; d++) {
            for (int i = 0; i < a[0].length; i++) {
                for (int j = 0; j < a[0][0].length; j++) {
                    result[d][i][j] =
                        a[d][i][j] + b[d][i][j];
                }
            }
        }

        return result;
    }

    public static double[][][] multiplyScalar(
            double[][][] tensor, double scalar) {

        validate(tensor);

        if (!Double.isFinite(scalar)) {
            throw new IllegalArgumentException(
                "Scalar must be finite"
            );
        }

        double[][][] result =
            new double[tensor.length]
                [tensor[0].length]
                [tensor[0][0].length];

        for (int d = 0; d < tensor.length; d++) {
            for (int i = 0; i < tensor[0].length; i++) {
                for (int j = 0; j < tensor[0][0].length; j++) {
                    result[d][i][j] =
                        tensor[d][i][j] * scalar;
                }
            }
        }

        return result;
    }

    public static double sum(double[][][] tensor) {

        validate(tensor);

        double total = 0;

        for (double[][] slice : tensor) {
            for (double[] row : slice) {
                for (double value : row) {
                    total += value;
                }
            }
        }

        return total;
    }

    public static void main(String[] args) {

        double[][][] tensor = {
            {{1, 2}, {3, 4}},
            {{5, 6}, {7, 8}}
        };

        System.out.println("===== TENSOR OPERATIONS =====");

        System.out.println(
            "Shape: 2 x 2 x 2"
        );

        System.out.println(
            "Sum: " + sum(tensor)
        );

        double[][][] doubled =
            multiplyScalar(tensor, 2.0);

        System.out.println(
            "Sum after scaling: " + sum(doubled)
        );

        double[][][] added = add(tensor, tensor);

        System.out.println(
            "Sum after addition: " + sum(added)
        );
    }
}
