
package com.roadguard.cnn;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class CNNFeatureExtractorTest {

    @Test
    void testGrayscaleConversion() {

        double[][][] rgb = {
            {
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0}
            }
        };

        double[][] gray =
            CNNFeatureExtractor.toGrayscale(rgb);

        assertEquals(0.299, gray[0][0], 1e-9);
        assertEquals(0.587, gray[0][1], 1e-9);
    }

    @Test
    void testFeatureMapShape() {

        double[][][] rgb = new double[224][224][3];

        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                rgb[y][x][0] = x / 223.0;
                rgb[y][x][1] = y / 223.0;
                rgb[y][x][2] = 0.5;
            }
        }

        double[][] result =
            CNNFeatureExtractor.extract(rgb);

        assertEquals(112, result.length);
        assertEquals(112, result[0].length);

        for (double[] row : result) {
            for (double value : row) {
                assertTrue(Double.isFinite(value));
                assertTrue(value >= 0);
            }
        }
    }

    @Test
    void testInvalidRGBValues() {

        double[][][] invalid = {
            {
                {1.5, 0.0, 0.0}
            }
        };

        assertThrows(
            IllegalArgumentException.class,
            () -> CNNFeatureExtractor.extract(invalid)
        );
    }
}
