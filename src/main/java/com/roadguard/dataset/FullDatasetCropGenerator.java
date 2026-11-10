
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class FullDatasetCropGenerator {

    private static final Path SPLIT_DIR =
        Path.of("data/splits");

    private static final String[] SPLITS = {
        "train", "val", "test"
    };

    public static void processSplit(
            String split, int limit) throws IOException {

        Path csvPath = SPLIT_DIR.resolve(
            split + ".csv"
        );

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

        int[] classCounts = new int[8];

        for (int i = 1; i < lines.size(); i++) {

            if (limit > 0 && processed >= limit) {
                break;
            }

            String line = lines.get(i).trim();

            if (line.isEmpty()) {
                continue;
            }

            String[] columns = line.split(",", -1);

            if (columns.length != 2) {
                failed++;
                continue;
            }

            String imageName = columns[0].trim();
            String labelName = columns[1].trim();

            try {
                int count =
                    DamageCropGenerator.processImage(
                        imageName, labelName, split
                    );

                generated += count;

            } catch (IOException
                     | IllegalArgumentException e) {

                failed++;

                System.out.println(
                    "Failed: " + imageName
                    + " | " + e.getMessage()
                );
            }

            processed++;

            if (processed % 1000 == 0) {
                System.out.println(
                    split + " | Processed: "
                    + processed
                    + " | Crops: " + generated
                );
            }
        }

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
    }

    public static void main(String[] args)
            throws IOException {

        // Default: process only 10 images per split.
        int limit = 10;

        if (args.length > 0) {
            limit = Integer.parseInt(args[0]);

            if (limit < 0) {
                throw new IllegalArgumentException(
                    "Limit cannot be negative"
                );
            }
        }

        System.out.println(
            "===== ROADGUARD CNN DATASET GENERATION ====="
        );

        for (String split : SPLITS) {
            processSplit(split, limit);
        }

        System.out.println(
            "\nDATASET GENERATION COMPLETED"
        );
    }
}
