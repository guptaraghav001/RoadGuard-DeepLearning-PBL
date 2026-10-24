
package com.roadguard.preprocessing;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import javax.imageio.ImageIO;

public class ImagePreprocessor {

    public static final int IMAGE_SIZE = 224;

    public static double[][][] preprocess(Path imagePath)
            throws IOException {

        BufferedImage original =
            ImageIO.read(imagePath.toFile());

        if (original == null) {
            throw new IOException(
                "Unsupported or invalid image: " + imagePath
            );
        }

        BufferedImage resized = new BufferedImage(
            IMAGE_SIZE,
            IMAGE_SIZE,
            BufferedImage.TYPE_INT_RGB
        );

        Graphics2D graphics = resized.createGraphics();

        try {
            graphics.setRenderingHint(
                RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR
            );

            graphics.drawImage(
                original,
                0, 0,
                IMAGE_SIZE, IMAGE_SIZE,
                null
            );
        } finally {
            graphics.dispose();
        }

        // Tensor shape: [height][width][channels]
        double[][][] tensor =
            new double[IMAGE_SIZE][IMAGE_SIZE][3];

        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {

                int rgb = resized.getRGB(x, y);

                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;

                tensor[y][x][0] = red / 255.0;
                tensor[y][x][1] = green / 255.0;
                tensor[y][x][2] = blue / 255.0;
            }
        }

        return tensor;
    }

    public static void main(String[] args)
            throws IOException {

        if (args.length == 0) {
            System.out.println(
                "Please provide an image path."
            );
            return;
        }

        Path path = Path.of(args[0]);

        double[][][] image = preprocess(path);

        System.out.println(
            "ROADGUARD IMAGE PREPROCESSING"
        );

        System.out.println(
            "Tensor shape: "
            + image.length + " x "
            + image[0].length + " x "
            + image[0][0].length
        );

        System.out.printf(
            "First pixel RGB: %.4f %.4f %.4f%n",
            image[0][0][0],
            image[0][0][1],
            image[0][0][2]
        );
    }
}
