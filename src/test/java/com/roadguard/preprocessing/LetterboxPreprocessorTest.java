
package com.roadguard.preprocessing;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

public class LetterboxPreprocessorTest {

    @Test
    void testOutputSizeAndPadding() throws Exception {

        Path temp = Files.createTempFile(
            "roadguard-letterbox-", ".png"
        );

        try {
            BufferedImage image = new BufferedImage(
                400, 200, BufferedImage.TYPE_INT_RGB
            );

            for (int y = 0; y < 200; y++) {
                for (int x = 0; x < 400; x++) {
                    image.setRGB(x, y, 0xFF0000);
                }
            }

            ImageIO.write(image, "png", temp.toFile());

            BufferedImage result =
                LetterboxPreprocessor.resize(temp);

            assertEquals(224, result.getWidth());
            assertEquals(224, result.getHeight());

            // A 400x200 image becomes 224x112,
            // with 56 pixels padding at top/bottom.
            assertEquals(
                0xFF727272,
                result.getRGB(0, 0)
            );

            assertEquals(
                0xFFFF0000,
                result.getRGB(112, 112)
            );

        } finally {
            Files.deleteIfExists(temp);
        }
    }

    @Test
    void testSquareImage() throws Exception {

        Path temp = Files.createTempFile(
            "roadguard-square-", ".png"
        );

        try {
            BufferedImage image = new BufferedImage(
                100, 100, BufferedImage.TYPE_INT_RGB
            );

            ImageIO.write(image, "png", temp.toFile());

            BufferedImage result =
                LetterboxPreprocessor.resize(temp);

            assertEquals(224, result.getWidth());
            assertEquals(224, result.getHeight());

        } finally {
            Files.deleteIfExists(temp);
        }
    }
}
