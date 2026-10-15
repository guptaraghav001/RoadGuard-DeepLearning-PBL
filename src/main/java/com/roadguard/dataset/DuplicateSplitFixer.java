
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class DuplicateSplitFixer {

    private static final Path SPLITS =
        Path.of("data/splits");

    private static final Path RAW = Path.of(
        "data/raw/RDD2022ES/combined_annotatedv2"
    );

    public static void main(String[] args)
            throws IOException {

        Path trainFile = SPLITS.resolve("train.csv");
        Path valFile = SPLITS.resolve("val.csv");
        Path testFile = SPLITS.resolve("test.csv");

        List<String> train =
            new ArrayList<>(Files.readAllLines(trainFile));

        List<String> val =
            new ArrayList<>(Files.readAllLines(valFile));

        List<String> test =
            new ArrayList<>(Files.readAllLines(testFile));

        // Verify expected starting assignments.
        requireRow(val, "Czech_002030.jpg");
        requireRow(test, "Japan_012647.jpg");
        requireRow(train, "Czech_002124.jpg");
        requireRow(val, "Japan_001402.jpg");

        // Create backups before making changes.
        for (Path file :
                List.of(trainFile, valFile, testFile)) {

            Files.copy(
                file,
                file.resolveSibling(
                    file.getFileName() + ".backup"
                ),
                StandardCopyOption.REPLACE_EXISTING
            );
        }

        // Move Czech image from validation to training.
        String czechRow =
            removeRow(val, "Czech_002030.jpg");

        train.add(czechRow);

        String mirrorImage =
            "xmirror_Czech_002030.jpg";

        String mirrorLabel =
            "xmirror_Czech_002030.txt";

        if (!Files.isRegularFile(RAW.resolve(mirrorImage))
            || !Files.isRegularFile(
                RAW.resolve(mirrorLabel))) {

            throw new IOException(
                "Czech mirror image or label missing"
            );
        }

        train.add(mirrorImage + "," + mirrorLabel);

        // Move Japanese image from testing to validation.
        String japanRow =
            removeRow(test, "Japan_012647.jpg");

        val.add(japanRow);

        // Save updated CSV manifests.
        Files.write(
            trainFile, train, StandardCharsets.UTF_8
        );

        Files.write(
            valFile, val, StandardCharsets.UTF_8
        );

        Files.write(
            testFile, test, StandardCharsets.UTF_8
        );

        System.out.println(
            "===== DUPLICATE SPLIT FIX REPORT ====="
        );

        System.out.println(
            "Czech_002030.jpg moved: val -> train"
        );

        System.out.println(
            "Czech mirror added to training"
        );

        System.out.println(
            "Japan_012647.jpg moved: test -> val"
        );

        System.out.println(
            "CSV backups created"
        );

        System.out.println(
            "Updated train entries: " + (train.size() - 1)
        );

        System.out.println(
            "Updated val entries: " + (val.size() - 1)
        );

        System.out.println(
            "Updated test entries: " + (test.size() - 1)
        );
    }

    private static void requireRow(
        List<String> rows, String filename
    ) throws IOException {

        long count = rows.stream()
            .filter(row -> row.startsWith(filename + ","))
            .count();

        if (count != 1) {
            throw new IOException(
                "Expected exactly one row for "
                + filename + ", found " + count
            );
        }
    }

    private static String removeRow(
        List<String> rows, String filename
    ) throws IOException {

        for (int i = 1; i < rows.size(); i++) {

            if (rows.get(i).startsWith(filename + ",")) {
                return rows.remove(i);
            }
        }

        throw new IOException(
            "Image not found: " + filename
        );
    }
}
