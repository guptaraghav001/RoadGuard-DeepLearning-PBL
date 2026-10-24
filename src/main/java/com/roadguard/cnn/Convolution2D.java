
package com.roadguard.cnn;

public class Convolution2D {

    public static double[][] apply(
            double[][] input,
            double[][] kernel,
            int stride,
            int padding) {

        validate(input);
        validate(kernel);

        if (stride <= 0 || padding < 0) {
            throw new IllegalArgumentException(
                "Invalid stride or padding"
            );
        }

        int inputH = input.length;
        int inputW = input[0].length;
        int kernelH = kernel.length;
        int kernelW = kernel[0].length;

        int paddedH = inputH + 2 * padding;
        int paddedW = inputW + 2 * padding;

        if (kernelH > paddedH || kernelW > paddedW) {
            throw new IllegalArgumentException(
                "Kernel larger than padded input"
            );
        }

        double[][] padded =
            new double[paddedH][paddedW];

        for (int i = 0; i < inputH; i++) {
            for (int j = 0; j < inputW; j++) {
                padded[i + padding][j + padding] =
                    input[i][j];
            }
        }

        int outputH =
            (paddedH - kernelH) / stride + 1;

        int outputW =
            (paddedW - kernelW) / stride + 1;

        double[][] output =
            new double[outputH][outputW];

        for (int i = 0; i < outputH; i++) {
            for (int j = 0; j < outputW; j++) {

                double sum = 0.0;

                for (int ki = 0; ki < kernelH; ki++) {
                    for (int kj = 0; kj < kernelW; kj++) {
                        sum += padded[i * stride + ki]
                                     [j * stride + kj]
                            * kernel[ki][kj];
                    }
                }

                output[i][j] = sum;
            }
        }

        return output;
    }

    private static void validate(double[][] matrix) {
        if (matrix == null || matrix.length == 0
                || matrix[0] == null
                || matrix[0].length == 0) {
            throw new IllegalArgumentException(
                "Matrix cannot be empty"
            );
        }

        int width = matrix[0].length;

        for (double[] row : matrix) {
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

    public static void main(String[] args) {

        double[][] input = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        double[][] kernel = {
            {1, 0},
            {0, -1}
        };

        double[][] result = apply(
            input, kernel, 1, 0
        );

        System.out.println("Convolution output:");

        for (double[] row : result) {
            for (double value : row) {
                System.out.printf("%7.2f", value);
            }
            System.out.println();
        }
    }
}
