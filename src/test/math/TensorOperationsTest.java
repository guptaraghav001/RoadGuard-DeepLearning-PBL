
package com.roadguard.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TensorOperationsTest {

    @Test
    void testTensorSum() {
        double[][][] tensor = {
            {{1, 2}, {3, 4}},
            {{5, 6}, {7, 8}}
        };

        assertEquals(
            36.0,
            TensorOperations.sum(tensor),
            1e-9
        );
    }

    @Test
    void testScalarMultiplication() {
        double[][][] tensor = {
            {{1, 2}, {3, 4}}
        };

        double[][][] result =
            TensorOperations.multiplyScalar(tensor, 2);

        assertEquals(2.0, result[0][0][0], 1e-9);
        assertEquals(8.0, result[0][1][1], 1e-9);
    }

    @Test
    void testTensorAddition() {
        double[][][] a = {
            {{1, 2}, {3, 4}}
        };

        double[][][] result =
            TensorOperations.add(a, a);

        assertEquals(2.0, result[0][0][0], 1e-9);
        assertEquals(8.0, result[0][1][1], 1e-9);
    }

    @Test
    void testInvalidDimensions() {
        double[][][] a = {
            {{1, 2}}
        };

        double[][][] b = {
            {{1, 2, 3}}
        };

        assertThrows(
            IllegalArgumentException.class,
            () -> TensorOperations.add(a, b)
        );
    }
}
