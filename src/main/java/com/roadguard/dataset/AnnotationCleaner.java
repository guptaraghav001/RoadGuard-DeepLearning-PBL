
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class AnnotationCleaner {

    private static final Path RAW = Path.of(
        "data/raw/RDD2022ES/combined_annotatedv2"
    );

    private static final Path CLEAN = Path.of(
        "data/processed/RDD2022ES/labels"
    );

    private static final Path SPLITS = Path.of("data/splits");

    private static final double EPSILON = 1e-5;

    public static void main(String[] args)
            throws IOException {

        Files.createDirectories(CLEAN);

        Set<String> labelFiles = new TreeSet<>();

        for (String split : List.of("train", "val", "test")) {

            List<String> rows = Files.readAllLines(
                SPLITS.resolve(split + ".csv")
            );

            for (int i = 1; i < rows.size(); i++) {
                if (rows.get(i).isBlank()) continue;

                String[] parts = rows.get(i).split(",", -1);

                if (parts.length != 2) {
                    throw new IOException(
                        "Invalid CSV row: " + rows.get(i)
                    );
                }

                labelFiles.add(parts[1].trim());
            }
        }

        int corrected = 0;
        int unchanged = 0;
        int rejected = 0;

        for (String filename : labelFiles) {

            Path source = RAW.resolve(filename);
            Path destination = CLEAN.resolve(filename);

            List<String> output = new ArrayList<>();

            for (String line : Files.readAllLines(source)) {

                if (line.isBlank()) continue;

                String[] parts = line.trim().split("\\s+");

                if (parts.length != 5) {
                    rejected++;
                    throw new IOException(
                        "Malformed annotation in " + filename
                    );
                }

                int classId = Integer.parseInt(parts[0]);

                if (classId < 0 || classId > 7) {
                    throw new IOException(
                        "Invalid class in " + filename
                    );
                }

                double x = Double.parseDouble(parts[1]);
                double y = Double.parseDouble(parts[2]);
                double w = Double.parseDouble(parts[3]);
                double h = Double.parseDouble(parts[4]);

                if (!Double.isFinite(x)
                    || !Double.isFinite(y)
                    || !Double.isFinite(w)
                    || !Double.isFinite(h)
                    || w <= 0 || h <= 0) {

                    throw new IOException(
                        "Invalid coordinates in " + filename
                    );
                }

                double x1 = x - w / 2;
                double y1 = y - h / 2;
                double x2 = x + w / 2;
                double y2 = y + h / 2;

                if (x1 < -EPSILON || y1 < -EPSILON
                    || x2 > 1 + EPSILON
                    || y2 > 1 + EPSILON) {

                    rejected++;

                    throw new IOException(
                        "Box exceeds tolerance in " + filename
                    );
                }

                double nx1 = Math.max(0, x1);
                double ny1 = Math.max(0, y1);
                double nx2 = Math.min(1, x2);
                double ny2 = Math.min(1, y2);

                double newX = (nx1 + nx2) / 2;
                double newY = (ny1 + ny2) / 2;
                double newW = nx2 - nx1;
                double newH = ny2 - ny1;

                if (newW <= 0 || newH <= 0) {
                    throw new IOException(
                        "Zero-area box in " + filename
                    );
                }

                if (nx1 != x1 || ny1 != y1
                    || nx2 != x2 || ny2 != y2) {
                    corrected++;
                } else {
                    unchanged++;
                }

                output.add(String.format(
                    Locale.US,
                    "%d %.10f %.10f %.10f %.10f",
                    classId, newX, newY, newW, newH
                ));
            }

            Files.write(
                destination,
                output,
                StandardCharsets.UTF_8
            );
        }

        System.out.println(
            "===== ANNOTATION CLEANING REPORT ====="
        );
        System.out.println(
            "Annotation files processed: " + labelFiles.size()
        );
        System.out.println("Corrected boxes: " + corrected);
        System.out.println("Unchanged boxes: " + unchanged);
        System.out.println("Rejected boxes: " + rejected);
        System.out.println(
            "Cleaned labels saved in: " + CLEAN.toAbsolutePath()
        );
    }
}
