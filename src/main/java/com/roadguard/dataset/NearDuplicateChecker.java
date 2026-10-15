
package com.roadguard.dataset;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import javax.imageio.ImageIO;

public class NearDuplicateChecker {

    private static final Path DATA = Path.of(
        "data/raw/RDD2022ES/combined_annotatedv2"
    );

    private static final Path SPLITS = Path.of("data/splits");

    public static void main(String[] args)
            throws IOException {

        Map<Long, List<String>> hashes = new HashMap<>();
        int compared = 0;
        int exactHashMatches = 0;
        int unreadable = 0;

        for (String split : List.of("train", "val", "test")) {

            List<String> rows = Files.readAllLines(
                SPLITS.resolve(split + ".csv")
            );

            for (int i = 1; i < rows.size(); i++) {

                if (rows.get(i).isBlank()) continue;

                String[] parts = rows.get(i).split(",", -1);
                if (parts.length != 2) continue;

                String filename = parts[0].trim();

                // Known mirrors are already grouped.
                // Compare original images only.
                if (filename.startsWith("xmirror_")) {
                    continue;
                }

                Path imagePath = DATA.resolve(filename);
                BufferedImage image = ImageIO.read(
                    imagePath.toFile()
                );

                if (image == null) {
                    unreadable++;
                    continue;
                }

                long hash = differenceHash(image);

                List<String> previous =
                    hashes.computeIfAbsent(
                        hash, k -> new ArrayList<>()
                    );

                if (!previous.isEmpty()) {
                    exactHashMatches++;

                    if (exactHashMatches <= 20) {
                        System.out.println(
                            "Possible duplicate: " + filename
                            + " [" + split + "]"
                        );
                        System.out.println(
                            "Matches: " + previous
                        );
                    }
                }

                previous.add(split + ":" + filename);
                compared++;
            }
        }

        System.out.println(
            "\n===== NEAR-DUPLICATE AUDIT ====="
        );

        System.out.println("Images compared: " + compared);
        System.out.println(
            "Identical dHash matches: " + exactHashMatches
        );
        System.out.println(
            "Unreadable images: " + unreadable
        );
    }

    private static long differenceHash(BufferedImage image) {

        BufferedImage resized = new BufferedImage(
            9, 8, BufferedImage.TYPE_BYTE_GRAY
        );

        Graphics2D graphics = resized.createGraphics();
        graphics.drawImage(
            image.getScaledInstance(
                9, 8, Image.SCALE_SMOOTH
            ),
            0, 0, null
        );
        graphics.dispose();

        long hash = 0;

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {

                int left = resized.getRaster()
                    .getSample(x, y, 0);

                int right = resized.getRaster()
                    .getSample(x + 1, y, 0);

                hash <<= 1;

                if (left > right) {
                    hash |= 1;
                }
            }
        }

        return hash;
    }
}
