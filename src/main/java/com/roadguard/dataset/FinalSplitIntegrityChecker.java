
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

public class FinalSplitIntegrityChecker {

    private static final Path SPLIT_DIR =
        Path.of("data/splits");

    private static final String[] SPLITS = {
        "train", "val", "test"
    };

    public static void main(String[] args)
            throws IOException {

        Map<String, String> sourceSplits =
            new HashMap<>();

        int totalEntries = 0;
        int leakage = 0;
        int malformed = 0;
        int mirrorOutsideTrain = 0;

        System.out.println(
            "===== ROADGUARD FINAL SPLIT CHECK ====="
        );

        for (String split : SPLITS) {

            Path csv = SPLIT_DIR.resolve(
                split + ".csv"
            );

            List<String> lines =
                Files.readAllLines(csv);

            if (lines.isEmpty()
                    || !lines.get(0).trim()
                        .equalsIgnoreCase("image,label")) {
                throw new IOException(
                    "Invalid manifest: " + csv
                );
            }

            for (int i = 1; i < lines.size(); i++) {

                String line = lines.get(i).trim();

                if (line.isEmpty()) {
                    continue;
                }

                String[] columns =
                    line.split(",", -1);

                if (columns.length != 2
                        || columns[0].trim().isEmpty()
                        || columns[1].trim().isEmpty()) {
                    malformed++;
                    continue;
                }

                String imageName =
                    columns[0].trim();

                String source = imageName.replaceFirst(
                    "^xmirror_", ""
                );

                if (imageName.startsWith("xmirror_")
                        && !split.equals("train")) {
                    mirrorOutsideTrain++;
                }

                String previous =
                    sourceSplits.putIfAbsent(
                        source, split
                    );

                if (previous != null
                        && !previous.equals(split)) {
                    leakage++;

                    System.out.println(
                        "LEAKAGE: " + source
                        + " | " + previous
                        + " vs " + split
                    );
                }

                totalEntries++;
            }
        }

        System.out.println(
            "\nTotal entries: " + totalEntries
        );

        System.out.println(
            "Malformed rows: " + malformed
        );

        System.out.println(
            "Mirrors outside train: "
            + mirrorOutsideTrain
        );

        System.out.println(
            "Cross-split source leakage: "
            + leakage
        );

        if (malformed == 0
                && mirrorOutsideTrain == 0
                && leakage == 0) {

            System.out.println(
                "FINAL SPLIT INTEGRITY CHECK PASSED"
            );

        } else {

            System.out.println(
                "FINAL SPLIT INTEGRITY ISSUES FOUND"
            );
        }
    }
}
