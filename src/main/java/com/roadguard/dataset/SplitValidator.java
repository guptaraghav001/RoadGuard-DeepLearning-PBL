
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

public class SplitValidator {

    private static final Path DATA = Path.of(
        "data/raw/RDD2022ES/combined_annotatedv2"
    );

    private static final Path CLEAN_LABELS = Path.of(
    "data/processed/RDD2022ES/labels"
);

    private static final Path SPLITS = Path.of("data/splits");

    private static final String[] NAMES = {
        "train", "val", "test"
    };

    public static void main(String[] args)
            throws IOException {

        List<String> classes = Files.readAllLines(
            DATA.resolve("classes.txt")
        );

        int classCount = classes.size();

        Map<String, String> ownership = new HashMap<>();
        Map<String, long[]> distributions = new LinkedHashMap<>();

        int errors = 0;

        System.out.println(
            "===== ROADGUARD SPLIT VALIDATION ====="
        );

        for (String split : NAMES) {

            Path csv = SPLITS.resolve(split + ".csv");
            List<String> rows = Files.readAllLines(csv);

            long[] counts = new long[classCount];
            distributions.put(split, counts);

            int entries = 0;
            int missingFiles = 0;
            int mirroredEntries = 0;
            int invalidLabels = 0;
            int leakage = 0;

            Set<String> seenInSplit = new HashSet<>();

            for (int i = 1; i < rows.size(); i++) {

                String row = rows.get(i).trim();

                if (row.isEmpty()) {
                    continue;
                }

                String[] parts = row.split(",", -1);

                if (parts.length != 2) {
                    errors++;
                    System.out.println(
                        "Malformed CSV row: " + row
                    );
                    continue;
                }

                String image = parts[0].trim();
                String label = parts[1].trim();

                entries++;

                boolean mirrored =
                    image.startsWith("xmirror_");

                if (mirrored) {
                    mirroredEntries++;
                }

                String original = mirrored
                    ? image.substring("xmirror_".length())
                    : image;

                if (!seenInSplit.add(image)) {
                    leakage++;
                    System.out.println(
                        "Duplicate entry: " + image
                    );
                }

                String previous =
                    ownership.putIfAbsent(original, split);

                if (previous != null
                    && !previous.equals(split)) {

                    leakage++;
                    System.out.println(
                        "Cross-split leakage: " + original
                    );
                }

                Path imagePath = DATA.resolve(image);
Path labelPath = CLEAN_LABELS.resolve(label);

                if (!Files.isRegularFile(imagePath)
                    || !Files.isRegularFile(labelPath)) {

                    missingFiles++;
                    continue;
                }

                for (String line :
                        Files.readAllLines(labelPath)) {

                    if (line.isBlank()) {
                        continue;
                    }

                    String[] values =
                        line.trim().split("\\s+");

                    try {
                        if (values.length != 5) {
                            invalidLabels++;
                            continue;
                        }

                        int id =
                            Integer.parseInt(values[0]);

                        if (id < 0 || id >= classCount) {
                            invalidLabels++;
                            continue;
                        }

                        double[] box = new double[4];

                        for (int j = 0; j < 4; j++) {
                            box[j] =
                                Double.parseDouble(values[j + 1]);
                        }

                        boolean valid = true;

                        for (double value : box) {
                            if (!Double.isFinite(value)
                                || value < 0 || value > 1) {
                                valid = false;
                            }
                        }

                        
double EPSILON = 1e-9;
if (box[2] <= 0 || box[3] <= 0
    || box[0] - box[2] / 2 < -EPSILON
    || box[0] + box[2] / 2 > 1 + EPSILON
    || box[1] - box[3] / 2 < -EPSILON
    || box[1] + box[3] / 2 > 1 + EPSILON) {

    valid = false;
}


if (!valid && invalidLabels < 5) {

    System.out.println("Invalid label: " + label);

    System.out.printf(
        java.util.Locale.US,
        "Bounds: x1=%.15f y1=%.15f x2=%.15f y2=%.15f%n",
        box[0] - box[2] / 2,
        box[1] - box[3] / 2,
        box[0] + box[2] / 2,
        box[1] + box[3] / 2
    );
}


                       if (valid) {
    counts[id]++;
} else {
    invalidLabels++;

    if (invalidLabels <= 10) {
        System.out.println(
            "Invalid box in: " + label
            + " | Line: " + line
        );
    }
}

                    } catch (NumberFormatException e) {
                        invalidLabels++;
                    }
                }
            }

            if (!split.equals("train")
                && mirroredEntries > 0) {

                System.out.println(
                    "WARNING: Mirrors found in " + split
                );
                errors += mirroredEntries;
            }

            errors += missingFiles
                + invalidLabels + leakage;

            System.out.println("\nSplit: " + split);
            System.out.println("Entries: " + entries);
            System.out.println(
                "Mirrored entries: " + mirroredEntries
            );
            System.out.println(
                "Missing image/label pairs: " + missingFiles
            );
            System.out.println(
                "Invalid label lines: " + invalidLabels
            );
            System.out.println(
                "Duplicate/leakage findings: " + leakage
            );
        }

        System.out.println(
            "\n===== CLASS DISTRIBUTION ====="
        );

        for (int id = 0; id < classCount; id++) {

            System.out.println(
                id + " - " + classes.get(id)
            );

            for (String split : NAMES) {

                long count =
                    distributions.get(split)[id];

                System.out.println(
                    "  " + split + ": " + count
                );

                if (count == 0) {
                    System.out.println(
                        "  WARNING: No instances in "
                        + split
                    );
                }
            }
        }

        System.out.println(
            "\nTotal validation errors: " + errors
        );

        if (errors == 0) {
            System.out.println(
                "STRUCTURAL VALIDATION PASSED"
            );
        } else {
            System.out.println(
                "VALIDATION ISSUES FOUND"
            );
            System.exit(1);
        }
    }
}
