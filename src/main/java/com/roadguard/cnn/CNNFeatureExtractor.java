
package com.roadguard.cnn;

import com.roadguard.preprocessing.ImagePreprocessor;

import java.io.IOException;
import java.nio.file.Path;

public class CNNFeatureExtractor {

    // Fixed vertical edge detection kernel
    private static final double[][] EDGE_KERNEL = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    public static double[][] toGrayscale(
            double[][][] rgb) {

        if (rgb == null || rgb.length == 0
                || rgb[0] == null
                || rgb[0].length == 0) {
            throw new IllegalArgumentException(
                "Invalid RGB tensor"
            );
        }

        int height = rgb.length;
        int width = rgb[0].length;

        double[][] gray = new double[height][width];

        for (int y = 0; y < height; y++) {

            if (rgb[y] == null || rgb[y].length != width) {
                throw new IllegalArgumentException(
                    "Inconsistent image width"
                );
            }

            for (int x = 0; x < width; x++) {

                if (rgb[y][x] == null
                        || rgb[y][x].length != 3) {
                    throw new IllegalArgumentException(
                        "Exactly 3 RGB channels required"
                    );
                }

                double r = rgb[y][x][0];
                double g = rgb[y][x][1];
                double b = rgb[y][x][2];

                if (!Double.isFinite(r)
                        || !Double.isFinite(g)
                        || !Double.isFinite(b)
                        || r < 0 || r > 1
                        || g < 0 || g > 1
                        || b < 0 || b > 1) {
                    throw new IllegalArgumentException(
                        "RGB values must be between 0 and 1"
                    );
                }

                gray[y][x] =
                    0.299 * r + 0.587 * g + 0.114 * b;
            }
        }

        return gray;
    }

    public static double[][] extract(
            double[][][] rgb) {

        double[][] gray = toGrayscale(rgb);

        // Stride 1, padding 1
        double[][] convolved = Convolution2D.apply(
            gray, EDGE_KERNEL, 1, 1
        );

        double[][] activated =
            CNNOperations.relu(convolved);

        // 2x2 max pooling with stride 2
        return CNNOperations.maxPool(
            activated, 2, 2
        );
    }

    public static double[][] extractFromImage(
            Path imagePath) throws IOException {

        double[][][] rgb =
            ImagePreprocessor.preprocess(imagePath);

        return extract(rgb);
    }

    public static void main(String[] args)
            throws IOException {

        if (args.length == 0) {
            System.out.println(
                "Please provide an image path."
            );
            return;
        }

        Path imagePath = Path.of(args[0]);

        double[][] featureMap =
            extractFromImage(imagePath);

        System.out.println(
            "===== ROADGUARD CNN FEATURE EXTRACTOR ====="
        );

        System.out.println(
            "Feature map height: " + featureMap.length
        );

        System.out.println(
            "Feature map width: " + featureMap[0].length
        );

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;

        for (double[] row : featureMap) {
            for (double value : row) {
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
            }
        }

        double mean = sum /
            (featureMap.length * featureMap[0].length);

        System.out.printf("Minimum: %.6f%n", min);
        System.out.printf("Maximum: %.6f%n", max);
        System.out.printf("Mean: %.6f%n", mean);
    }
}
