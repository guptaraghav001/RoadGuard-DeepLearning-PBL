
package com.roadguard.preprocessing;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;

public class LetterboxPreprocessor {

    public static final int SIZE = 224;

    public static BufferedImage resize(Path imagePath)
            throws IOException {

        BufferedImage original =
            ImageIO.read(imagePath.toFile());

        if (original == null) {
            throw new IOException("Invalid image");
        }

        int width = original.getWidth();
        int height = original.getHeight();

        double scale = Math.min(
            (double) SIZE / width,
            (double) SIZE / height
        );

        int newWidth = Math.max(
            1, (int) Math.round(width * scale)
        );

        int newHeight = Math.max(
            1, (int) Math.round(height * scale)
        );

        int padX = (SIZE - newWidth) / 2;
        int padY = (SIZE - newHeight) / 2;

        BufferedImage output = new BufferedImage(
            SIZE, SIZE, BufferedImage.TYPE_INT_RGB
        );

        Graphics2D g = output.createGraphics();

        try {
            g.setColor(new Color(114, 114, 114));
            g.fillRect(0, 0, SIZE, SIZE);

            g.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
            );

            g.drawImage(
                original, padX, padY,
                newWidth, newHeight, null
            );
        } finally {
            g.dispose();
        }

        return output;
    }

    public static void main(String[] args)
            throws IOException {

        if (args.length == 0) {
            System.out.println("Provide an image path.");
            return;
        }

        BufferedImage result = resize(Path.of(args[0]));

        ImageIO.write(
            result, "png",
            Path.of("letterbox_output.png").toFile()
        );

        System.out.println(
            "Letterbox image saved: letterbox_output.png"
        );
    }
}
