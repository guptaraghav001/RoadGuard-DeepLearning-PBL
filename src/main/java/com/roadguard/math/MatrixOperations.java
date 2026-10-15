
package com.roadguard.math;

public class MatrixOperations {

    private static void validate(double[][] matrix) {
        if (matrix == null || matrix.length == 0
                || matrix[0] == null
                || matrix[0].length == 0) {
            throw new IllegalArgumentException(
                "Matrix cannot be empty"
            );
        }

        int columns = matrix[0].length;

        for (double[] row : matrix) {
            if (row == null || row.length != columns) {
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

    public static double[][] add(
            double[][] a, double[][] b) {

        validate(a);
        validate(b);

        if (a.length != b.length
                || a[0].length != b[0].length) {
            throw new IllegalArgumentException(
                "Matrix dimensions must match"
            );
        }

        double[][] result =
            new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }

        return result;
    }

    public static double[][] subtract(
            double[][] a, double[][] b) {

        validate(a);
        validate(b);

        if (a.length != b.length
                || a[0].length != b[0].length) {
            throw new IllegalArgumentException(
                "Matrix dimensions must match"
            );
        }

        double[][] result =
            new double[a.length][a[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }

        return result;
    }

    public static double[][] multiply(
            double[][] a, double[][] b) {

        validate(a);
        validate(b);

        if (a[0].length != b.length) {
            throw new IllegalArgumentException(
                "Incompatible matrix dimensions"
            );
        }

        double[][] result =
            new double[a.length][b[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < a[0].length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        return result;
    }

    public static double[][] transpose(double[][] a) {

        validate(a);

        double[][] result =
            new double[a[0].length][a.length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                result[j][i] = a[i][j];
            }
        }

        return result;
    }

    public static void printMatrix(double[][] matrix) {

        validate(matrix);

        for (double[] row : matrix) {
            for (double value : row) {
                System.out.printf("%9.3f", value);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {

        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};

        System.out.println("===== MATRIX ADDITION =====");
        printMatrix(add(a, b));

        System.out.println("\n===== MATRIX SUBTRACTION =====");
        printMatrix(subtract(a, b));

        System.out.println("\n===== MATRIX MULTIPLICATION =====");
        printMatrix(multiply(a, b));

        System.out.println("\n===== MATRIX TRANSPOSE =====");
        printMatrix(transpose(a));
    }
}
