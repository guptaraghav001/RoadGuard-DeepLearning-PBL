
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class DatasetInspector {

    public static void main(String[] args) {

        Path folder = Path.of(
            "data/raw/RDD2022ES/combined_annotatedv2"
        );

        if (!Files.isDirectory(folder)) {
            System.out.println("Dataset folder not found!");
            return;
        }

        try {
            List<String> classes = Files.readAllLines(
                folder.resolve("classes.txt")
            );

            long[] classCounts = new long[classes.size()];

            long imageCount = 0;
            long annotationCount = 0;
            long emptyAnnotations = 0;
            long missingAnnotations = 0;
            long invalidLines = 0;

            try (Stream<Path> stream = Files.list(folder)) {

                List<Path> images = stream
                    .filter(p -> p.toString()
                        .toLowerCase().endsWith(".jpg"))
                    .toList();

                imageCount = images.size();

                for (Path image : images) {

                    String filename =
                        image.getFileName().toString();

                    String base = filename.substring(
                        0, filename.lastIndexOf('.')
                    );

                    Path annotation =
                        folder.resolve(base + ".txt");

                    if (!Files.exists(annotation)) {
                        missingAnnotations++;
                        continue;
                    }

                    annotationCount++;

                    List<String> lines =
                        Files.readAllLines(annotation);

                    boolean hasLabel = false;

                    for (String line : lines) {

                        if (line.isBlank()) {
                            continue;
                        }

                        String[] parts =
                            line.trim().split("\\s+");

                        try {
                            if (parts.length != 5) {
                                invalidLines++;
                                continue;
                            }

                            int classId =
                                Integer.parseInt(parts[0]);

                            if (classId < 0 ||
                                classId >= classes.size()) {
                                invalidLines++;
                                continue;
                            }

                            boolean valid = true;

                            for (int i = 1; i < 5; i++) {
                                double value =
                                    Double.parseDouble(parts[i]);

                                if (!Double.isFinite(value)
                                    || value < 0 || value > 1) {
                                    valid = false;
                                }
                            }

                            if (!valid) {
                                invalidLines++;
                                continue;
                            }

                            classCounts[classId]++;
                            hasLabel = true;

                        } catch (NumberFormatException e) {
                            invalidLines++;
                        }
                    }

                    if (!hasLabel) {
                        emptyAnnotations++;
                    }
                }
            }

            System.out.println(
                "\n===== ROADGUARD DATASET REPORT ====="
            );

            System.out.println("Total images: " + imageCount);
            System.out.println(
                "Paired annotation files: " + annotationCount
            );
            System.out.println(
                "Images without valid labels: " + emptyAnnotations
            );
            System.out.println(
                "Missing annotations: " + missingAnnotations
            );
            System.out.println(
                "Invalid annotation lines: " + invalidLines
            );

            System.out.println("\nDamage class distribution:");

            for (int i = 0; i < classes.size(); i++) {
                System.out.printf(
                    "%d - %s : %d%n",
                    i, classes.get(i), classCounts[i]
                );
            }

        } catch (IOException e) {
            System.out.println("Dataset reading error:");
            e.printStackTrace();
        }
    }
}
