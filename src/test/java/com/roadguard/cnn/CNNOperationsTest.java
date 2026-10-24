
package com.roadguard.cnn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CNNOperationsTest {

    @Test
    void testReLU() {
        double[][] input = {
            {-2, 3},
            {4, -1}
        };

        double[][] result = CNNOperations.relu(input);

        assertArrayEquals(
            new double[]{0, 3}, result[0], 1e-9
        );
        assertArrayEquals(
            new double[]{4, 0}, result[1], 1e-9
        );
    }

    @Test
    void testMaxPooling() {
        double[][] input = {
            {1, 3, 2, 4},
            {5, 6, 7, 8},
            {9, 2, 1, 3},
            {4, 5, 6, 7}
        };

        double[][] result =
            CNNOperations.maxPool(input, 2, 2);

        assertArrayEquals(
            new double[]{6, 8}, result[0], 1e-9
        );
        assertArrayEquals(
            new double[]{9, 7}, result[1], 1e-9
        );
    }

    @Test
    void testInvalidPoolingSize() {
        assertThrows(
            IllegalArgumentException.class,
            () -> CNNOperations.maxPool(
                new double[][]{{1, 2}},
                3, 1
            )
        );
    }
}
