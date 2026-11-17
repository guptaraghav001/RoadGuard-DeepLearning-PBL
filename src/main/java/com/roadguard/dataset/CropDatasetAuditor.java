
package com.roadguard.dataset;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import javax.imageio.ImageIO;

public class CropDatasetAuditor {

   private static final Path ROOT = Path.of(
    "data/processed/cnn_crops_complete"
);

    private static final String[] SPLITS = {
        "train", "val", "test"
    };

    private static final String[] CLASSES = {
        "Longitudinal Crack",
        "Longitudinal Crack Wide",
        "Transverse Crack",
        "Transverse Crack Wide",
        "Alligator Crack",
        "Alligator Crack Sunken",
        "Pothole",
        "Pothole Deep"
    };

    public static void main(String[] args)
            throws IOException {

        System.out.println(
            "===== ROADGUARD CNN DATASET AUDIT ====="
        );

        Set<String> seenNames = new HashSet<>();
        Map<String, String> sourceSplits = new HashMap<>();

        int total = 0;
        int invalidImages = 0;
        int duplicateNames = 0;
        int sourceLeakage = 0;

        for (String split : SPLITS) {

            System.out.println(
                "\n===== " + split.toUpperCase() + " ====="
            );

            int splitTotal = 0;

            for (int classId = 0;
                    classId < CLASSES.length;
                    classId++) {

                Path classDir = ROOT
                    .resolve(split)
                    .resolve("class_" + classId);

                int classCount = 0;

                if (Files.isDirectory(classDir)) {

                    try (Stream<Path> files =
                             Files.list(classDir)) {

                        for (Path file :
                                files.filter(
                                    p -> p.toString()
                                        .endsWith(".png")
                                ).toList()) {

                            classCount++;

                            String name =
                                file.getFileName().toString();

                            if (!seenNames.add(name)) {
                                duplicateNames++;
                            }

                            String source = name.replaceFirst(
                                "_box\\d+\\.png$", ""
                            );

                            // Mirrored versions belong to
                            // the same source-image family.
                            String family = source.replaceFirst(
                                "^xmirror_", ""
                            );

                            String previous =
                                sourceSplits.putIfAbsent(
                                    family, split
                                );

                            if (previous != null
                                    && !previous.equals(split)) {
                                sourceLeakage++;
                            }

                            BufferedImage image =
                                ImageIO.read(file.toFile());

                            if (image == null
                                    || image.getWidth() != 224
                                    || image.getHeight() != 224) {
                                invalidImages++;
                            }
                        }
                    }
                }

                splitTotal += classCount;

                System.out.printf(
                    "Class %d (%s): %d%n",
                    classId,
                    CLASSES[classId],
                    classCount
                );
            }

            total += splitTotal;

            System.out.println(
                "Split total: " + splitTotal
            );
        }

        System.out.println(
            "\n===== FINAL AUDIT SUMMARY ====="
        );

        System.out.println(
            "Total crops: " + total
        );

        System.out.println(
            "Invalid images: " + invalidImages
        );

        System.out.println(
            "Duplicate filenames: " + duplicateNames
        );

        System.out.println(
            "Cross-split source leakage: " + sourceLeakage
        );

        if (invalidImages == 0 && sourceLeakage == 0) {
            System.out.println(
                "BASIC DATASET AUDIT PASSED"
            );
        } else {
            System.out.println(
                "DATASET AUDIT ISSUES FOUND"
            );
        }
    }
}
