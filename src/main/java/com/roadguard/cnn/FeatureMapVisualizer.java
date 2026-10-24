
package com.roadguard.cnn;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

public class FeatureMapVisualizer {

    public static BufferedImage visualize(double[][] map) {

        if (map == null || map.length == 0
                || map[0] == null || map[0].length == 0) {
            throw new IllegalArgumentException(
                "Feature map cannot be empty"
            );
        }

        int height = map.length;
        int width = map[0].length;

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;

        for (double[] row : map) {
            if (row == null || row.length != width) {
                throw new IllegalArgumentException(
                    "Feature map must be rectangular"
                );
            }

            for (double value : row) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                        "Feature map values must be finite"
                    );
                }
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
        }

        BufferedImage image = new BufferedImage(
            width, height, BufferedImage.TYPE_BYTE_GRAY
        );

        double range = max - min;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                int gray = range == 0
                    ? 0
                    : (int) Math.round(
                        255.0 * (map[y][x] - min) / range
                    );

                gray = Math.max(0, Math.min(255, gray));

                image.getRaster().setSample(x, y, 0, gray);
            }
        }

        return image;
    }

    public static void main(String[] args)
            throws IOException {

        if (args.length == 0) {
            System.out.println(
                "Provide a road image path."
            );
            return;
        }

        Path input = Path.of(args[0]);

        double[][] featureMap =
            CNNFeatureExtractor.extractFromImage(input);

        BufferedImage visualization = visualize(featureMap);

        Path output = Path.of(
            "feature_map_output.png"
        );

        ImageIO.write(
            visualization, "png", output.toFile()
        );

        System.out.println(
            "===== FEATURE MAP VISUALIZATION ====="
        );

        System.out.println(
            "Feature map size: "
            + visualization.getWidth()
            + " x " + visualization.getHeight()
        );

        System.out.println(
            "Saved to: " + output.toAbsolutePath()
        );
    }
}
