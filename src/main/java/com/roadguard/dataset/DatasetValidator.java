
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class DatasetValidator {

    public static void main(String[] args) {

        Path folder = Path.of(
            "data/raw/RDD2022ES/combined_annotatedv2"
        );

        if (!Files.isDirectory(folder)) {
            System.out.println("Dataset folder not found!");
            return;
        }

        List<String> missingLabels = new ArrayList<>();
        Map<String, Integer> prefixes = new TreeMap<>();
        List<String> possibleMirrors = new ArrayList<>();

        try (Stream<Path> stream = Files.list(folder)) {

            List<Path> images = stream
                .filter(Files::isRegularFile)
                .filter(p -> p.getFileName().toString()
                    .toLowerCase().endsWith(".jpg"))
                .toList();

            for (Path image : images) {

                String filename =
                    image.getFileName().toString();

                String base = filename.substring(
                    0, filename.lastIndexOf('.')
                );

                Path label = folder.resolve(base + ".txt");

                if (!Files.isRegularFile(label)) {
                    missingLabels.add(filename);
                }

                String prefix = base.contains("_")
                    ? base.substring(0, base.indexOf('_'))
                    : "OTHER";

                prefixes.merge(prefix, 1, Integer::sum);

                String lower = base.toLowerCase();

                if (lower.contains("flip")
                    || lower.contains("mirror")
                    || lower.contains("aug")) {

                    possibleMirrors.add(filename);
                }
            }

            System.out.println(
                "===== ROADGUARD DATASET VALIDATION ====="
            );

            System.out.println(
                "Total images: " + images.size()
            );

            System.out.println(
                "\nImages missing annotation files:"
            );

            missingLabels.forEach(System.out::println);

            System.out.println(
                "\nImage counts by filename prefix:"
            );

            prefixes.forEach((name, count) ->
                System.out.println(name + ": " + count)
            );

            System.out.println(
                "\nPossible augmented/mirrored filenames: "
                + possibleMirrors.size()
            );

            possibleMirrors.stream()
                .limit(20)
                .forEach(System.out::println);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
