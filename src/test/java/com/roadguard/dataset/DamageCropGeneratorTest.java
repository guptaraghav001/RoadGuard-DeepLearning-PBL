
package com.roadguard.dataset;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;

public class DamageCropGeneratorTest {

    @Test
    void testNullOutputDirectory() {

        assertThrows(
            IllegalArgumentException.class,
            () -> DamageCropGenerator.processImage(
                "sample.jpg",
                "sample.txt",
                "train",
                null
            )
        );
    }

    @Test
    void testInvalidSplit() {

        assertThrows(
            IllegalArgumentException.class,
            () -> DamageCropGenerator.processImage(
                "sample.jpg",
                "sample.txt",
                "invalid",
                Path.of("data/processed/test_output")
            )
        );
    }
}
