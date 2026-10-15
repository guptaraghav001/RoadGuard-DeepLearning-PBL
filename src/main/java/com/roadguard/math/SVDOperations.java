
package com.roadguard.math;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class SVDOperations {

    public static double[] singularValues(double[][] data) {
        RealMatrix matrix =
            new Array2DRowRealMatrix(data);

        SingularValueDecomposition svd =
            new SingularValueDecomposition(matrix);

        return svd.getSingularValues();
    }

    public static double reconstructionError(double[][] data) {
        RealMatrix matrix =
            new Array2DRowRealMatrix(data);

        SingularValueDecomposition svd =
            new SingularValueDecomposition(matrix);

        RealMatrix reconstructed = svd.getU()
            .multiply(svd.getS())
            .multiply(svd.getVT());

        return matrix.subtract(reconstructed)
            .getFrobeniusNorm();
    }

    public static void main(String[] args) {
        double[][] matrix = {
            {1, 2},
            {3, 4}
        };

        System.out.println("===== SVD EXPERIMENT =====");

        double[] values = singularValues(matrix);

        for (int i = 0; i < values.length; i++) {
            System.out.printf(
                "Singular value %d: %.6f%n",
                i + 1, values[i]
            );
        }

        System.out.printf(
            "Reconstruction error: %.12f%n",
            reconstructionError(matrix)
        );
    }
}
