
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class FullDatasetCropGenerator {

    private static final Path SPLIT_DIR =
        Path.of("data/splits");

  private static final Path FINAL_OUTPUT_DIR =
    Path.of("data/processed/cnn_crops_complete");



    private static final Path LOG_DIR =
        Path.of("experiments/dataset_audit");

    private static final String[] SPLITS = {
        "train", "val", "test"
    };

    public static void processSplit(
            String split, int limit) throws IOException {

        if (!List.of(SPLITS).contains(split)) {
            throw new IllegalArgumentException(
                "Invalid split: " + split
            );
        }

        Path csvPath = SPLIT_DIR.resolve(
            split + ".csv"
        );

        Files.createDirectories(LOG_DIR);

        Path failureLog = LOG_DIR.resolve(
            split + "_failures.txt"
        );

        Path rejectionLog = LOG_DIR.resolve(
            split + "_rejections.txt"
        );

        // Image-level failures
        List<String> failures = new ArrayList<>();

        // Individual invalid YOLO annotations
        List<String> rejections = new ArrayList<>();

        List<String> lines = Files.readAllLines(
            csvPath, StandardCharsets.UTF_8
        );

        if (lines.isEmpty()
                || !lines.get(0).trim()
                    .equalsIgnoreCase("image,label")) {

            throw new IOException(
                "Invalid CSV header: " + csvPath
            );
        }

        int processed = 0;
        int generated = 0;
        int failed = 0;

        for (int i = 1; i < lines.size(); i++) {

            if (limit > 0 && processed >= limit) {
                break;
            }

            String line = lines.get(i).trim();

            if (line.isEmpty()) {
                continue;
            }

            processed++;

            String[] columns = line.split(",", -1);

            if (columns.length != 2) {

                failed++;

                failures.add(
                    split + " | CSV row " + (i + 1)
                    + " | Invalid row format: " + line
                );

                continue;
            }

            String imageName = columns[0].trim();
            String labelName = columns[1].trim();

            if (imageName.isEmpty()
                    || labelName.isEmpty()) {

                failed++;

                failures.add(
                    split + " | CSV row " + (i + 1)
                    + " | Empty filename: " + line
                );

                continue;
            }

            try {

                // Five-argument method with rejection logging
                int count =
                    DamageCropGenerator.processImage(
                        imageName,
                        labelName,
                        split,
                        FINAL_OUTPUT_DIR,
                        rejections
                    );

                generated += count;

            } catch (IOException
                     | IllegalArgumentException e) {

                failed++;

                String message =
                    split + " | " + imageName
                    + " | " + e.getMessage();

                failures.add(message);

                System.out.println(
                    "Failed: " + message
                );
            }

            if (processed % 1000 == 0) {

                System.out.println(
                    split
                    + " | Images processed: " + processed
                    + " | Crops generated: " + generated
                    + " | Failed entries: " + failed
                    + " | Rejections: " + rejections.size()
                );
            }
        }

        // Write image-level failure log
        Files.write(
            failureLog,
            failures,
            StandardCharsets.UTF_8
        );

        // Write annotation-level rejection log
        Files.write(
            rejectionLog,
            rejections,
            StandardCharsets.UTF_8
        );

        System.out.println(
            "\n===== " + split.toUpperCase()
            + " CROP REPORT ====="
        );

        System.out.println(
            "Images processed: " + processed
        );

        System.out.println(
            "Crops generated: " + generated
        );

        System.out.println(
            "Failed entries: " + failed
        );

        System.out.println(
            "Rejected annotations: " + rejections.size()
        );

        System.out.println(
            "Failure log: " + failureLog
        );

        System.out.println(
            "Rejection log: " + rejectionLog
        );
    }

    public static void main(String[] args)
            throws IOException {

        // Default: 10 entries per split
        int limit = 10;

        if (args.length > 0) {

            try {
                limit = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                    "Limit must be an integer", e
                );
            }

            if (limit < 0) {
                throw new IllegalArgumentException(
                    "Limit cannot be negative"
                );
            }
        }

        System.out.println(
            "===== ROADGUARD CNN DATASET GENERATION ====="
        );

        System.out.println(
            "Output directory: "
            + FINAL_OUTPUT_DIR.toAbsolutePath()
        );

        System.out.println(
            "Entry limit per split: " + limit
        );

        if (limit == 0 && Files.exists(FINAL_OUTPUT_DIR)) {
    throw new IOException(
        "Full-run output directory already exists: "
        + FINAL_OUTPUT_DIR
        + ". Use a fresh directory."
    );
}


        for (String split : SPLITS) {

            processSplit(split, limit);
        }

        System.out.println(
            "\nDATASET GENERATION COMPLETED"
        );
    }
}
