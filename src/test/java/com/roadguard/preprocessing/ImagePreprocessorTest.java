
package com.roadguard.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

public class ImagePreprocessorTest {

    @Test
    void testImageShapeAndNormalization() throws Exception {

        Path image = Files.createTempFile("roadguard-", ".png");

        try {
            BufferedImage sample = new BufferedImage(
                10, 10, BufferedImage.TYPE_INT_RGB
            );

            // Create a solid red image.
            for (int y = 0; y < 10; y++) {
                for (int x = 0; x < 10; x++) {
                    sample.setRGB(x, y, 0xFF0000);
                }
            }

            ImageIO.write(sample, "png", image.toFile());

            double[][][] tensor =
                ImagePreprocessor.preprocess(image);

            assertEquals(224, tensor.length);
            assertEquals(224, tensor[0].length);
            assertEquals(3, tensor[0][0].length);

            assertEquals(1.0, tensor[0][0][0], 1e-9);
            assertEquals(0.0, tensor[0][0][1], 1e-9);
            assertEquals(0.0, tensor[0][0][2], 1e-9);

        } finally {
            Files.deleteIfExists(image);
        }
    }

    @Test
    void testInvalidImage() throws Exception {

        Path invalid = Files.createTempFile(
            "roadguard-invalid-", ".jpg"
        );

        try {
            assertThrows(
                java.io.IOException.class,
                () -> ImagePreprocessor.preprocess(invalid)
            );
        } finally {
            Files.deleteIfExists(invalid);
        }
    }
}
