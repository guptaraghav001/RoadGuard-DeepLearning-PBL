
package com.roadguard.cnn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.awt.image.BufferedImage;

public class FeatureMapVisualizerTest {

    @Test
    void testImageDimensions() {

        double[][] map = {
            {0, 1},
            {2, 3}
        };

        BufferedImage result =
            FeatureMapVisualizer.visualize(map);

        assertEquals(2, result.getWidth());
        assertEquals(2, result.getHeight());
    }

    @Test
    void testNormalization() {

        double[][] map = {
            {0, 10}
        };

        BufferedImage result =
            FeatureMapVisualizer.visualize(map);

        assertEquals(
            0, result.getRaster().getSample(0, 0, 0)
        );

        assertEquals(
            255, result.getRaster().getSample(1, 0, 0)
        );
    }

    @Test
    void testConstantFeatureMap() {

        double[][] map = {
            {5, 5},
            {5, 5}
        };

        BufferedImage result =
            FeatureMapVisualizer.visualize(map);

        assertEquals(
            0, result.getRaster().getSample(0, 0, 0)
        );
    }

    @Test
    void testInvalidValues() {

        double[][] invalid = {
            {Double.NaN}
        };

        assertThrows(
            IllegalArgumentException.class,
            () -> FeatureMapVisualizer.visualize(invalid)
        );
    }
}
