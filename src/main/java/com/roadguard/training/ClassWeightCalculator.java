
package com.roadguard.training;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

public class ClassWeightCalculator {

    private static final Path TRAIN_DIR = Path.of(
        "data/processed/cnn_crops_complete/train"
    );

    private static final int NUM_CLASSES = 8;

    public static void main(String[] args)
            throws IOException {

        System.out.println(
            "===== ROADGUARD CLASS DISTRIBUTION ====="
        );

        long[] counts = new long[NUM_CLASSES];
        long total = 0;

        for (int i = 0; i < NUM_CLASSES; i++) {

            Path classDir = TRAIN_DIR.resolve(
                "class_" + i
            );

            if (!Files.isDirectory(classDir)) {
                throw new IOException(
                    "Missing class directory: " + classDir
                );
            }

            try (Stream<Path> files = Files.list(classDir)) {

                counts[i] = files
                    .filter(Files::isRegularFile)
                    .filter(p -> p.getFileName()
                        .toString()
                        .toLowerCase()
                        .endsWith(".png"))
                    .count();
            }

            if (counts[i] == 0) {
                throw new IllegalStateException(
                    "No training samples for class_" + i
                );
            }

            total += counts[i];
        }

        double[] weights = new double[NUM_CLASSES];

        System.out.println(
            "\nClass | Images | Percentage | Weight"
        );

        for (int i = 0; i < NUM_CLASSES; i++) {

            weights[i] = (double) total /
                (NUM_CLASSES * counts[i]);

            double percentage =
                100.0 * counts[i] / total;

            System.out.printf(
                "%d | %d | %.2f%% | %.4f%n",
                i,
                counts[i],
                percentage,
                weights[i]
            );
        }

        System.out.println(
            "\nTotal training crops: " + total
        );

        System.out.println(
            "\nBalanced class weights:"
        );

        System.out.print("new float[]{");

        for (int i = 0; i < NUM_CLASSES; i++) {

            System.out.printf(
                "%.6ff%s",
                weights[i],
                i == NUM_CLASSES - 1 ? "" : ", "
            );
        }

        System.out.println("};");

        System.out.println(
            "\nCLASS WEIGHT CALCULATION COMPLETED"
        );
    }
}
