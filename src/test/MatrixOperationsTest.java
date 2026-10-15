
package com.roadguard.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixOperationsTest {

    private static void assertMatrixEquals(
            double[][] expected, double[][] actual) {

        assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(
                expected[i], actual[i], 1e-9
            );
        }
    }

    @Test
    void testAddition() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};

        assertMatrixEquals(
            new double[][]{{6, 8}, {10, 12}},
            MatrixOperations.add(a, b)
        );
    }

    @Test
    void testSubtraction() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};

        assertMatrixEquals(
            new double[][]{{-4, -4}, {-4, -4}},
            MatrixOperations.subtract(a, b)
        );
    }

    @Test
    void testMultiplication() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};

        assertMatrixEquals(
            new double[][]{{19, 22}, {43, 50}},
            MatrixOperations.multiply(a, b)
        );
    }

    @Test
    void testTranspose() {
        double[][] a = {{1, 2}, {3, 4}};

        assertMatrixEquals(
            new double[][]{{1, 3}, {2, 4}},
            MatrixOperations.transpose(a)
        );
    }

    @Test
    void testInvalidDimensions() {
        double[][] a = {{1, 2}};
        double[][] b = {{1, 2}};

        assertThrows(
            IllegalArgumentException.class,
            () -> MatrixOperations.multiply(a, b)
        );
    }
}
