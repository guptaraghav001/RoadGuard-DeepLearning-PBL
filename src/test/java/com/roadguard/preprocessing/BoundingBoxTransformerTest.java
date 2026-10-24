
package com.roadguard.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class BoundingBoxTransformerTest {

    @Test
    void testLandscapeImage() {
        double[] result = BoundingBoxTransformer.transform(
            new double[]{0.5, 0.5, 0.4, 0.2},
            400, 200
        );

        assertArrayEquals(
            new double[]{0.5, 0.5, 0.4, 0.1},
            result, 1e-9
        );
    }

    @Test
    void testSquareImage() {
        double[] result = BoundingBoxTransformer.transform(
            new double[]{0.5, 0.5, 0.4, 0.2},
            224, 224
        );

        assertArrayEquals(
            new double[]{0.5, 0.5, 0.4, 0.2},
            result, 1e-9
        );
    }

    @Test
    void testInvalidBox() {
        assertThrows(
            IllegalArgumentException.class,
            () -> BoundingBoxTransformer.transform(
                new double[]{1.2, 0.5, 0.4, 0.2},
                400, 200
            )
        );
    }
}
