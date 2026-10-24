
package com.roadguard.cnn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class Convolution2DTest {

    private static void assertMatrixEquals(
            double[][] expected,
            double[][] actual) {

        assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(
                expected[i], actual[i], 1e-9
            );
        }
    }

    @Test
    void testBasicConvolution() {
        double[][] input = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };

        double[][] kernel = {
            {1, 0},
            {0, -1}
        };

        assertMatrixEquals(
            new double[][]{
                {-4, -4},
                {-4, -4}
            },
            Convolution2D.apply(input, kernel, 1, 0)
        );
    }

    @Test
    void testStrideTwo() {
        double[][] input = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };

        double[][] kernel = {
            {1, 1},
            {1, 1}
        };

        assertMatrixEquals(
            new double[][]{
                {14, 22},
                {46, 54}
            },
            Convolution2D.apply(input, kernel, 2, 0)
        );
    }

    @Test
    void testPadding() {
        double[][] input = {
            {1, 2},
            {3, 4}
        };

        double[][] kernel = {
            {1, 1},
            {1, 1}
        };

        double[][] result =
            Convolution2D.apply(input, kernel, 1, 1);

        assertEquals(3, result.length);
        assertEquals(3, result[0].length);
        assertEquals(1.0, result[0][0], 1e-9);
        assertEquals(10.0, result[1][1], 1e-9);
    }

    @Test
    void testInvalidStride() {
        assertThrows(
            IllegalArgumentException.class,
            () -> Convolution2D.apply(
                new double[][]{{1}},
                new double[][]{{1}},
                0, 0
            )
        );
    }
}
