
package com.roadguard.dataset;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

public class DatasetSplitter {

    private static final Path DATA = Path.of(
        "data/raw/RDD2022ES/combined_annotatedv2"
    );

    private static final Path OUTPUT = Path.of("data/splits");
    private static final long SEED = 42;

    public static void main(String[] args) throws IOException {

        if (!Files.isDirectory(DATA)) {
            System.out.println("Dataset folder not found!");
            return;
        }

        List<String> originals = new ArrayList<>();

        try (Stream<Path> files = Files.list(DATA)) {
            files.filter(Files::isRegularFile)
                .map(p -> p.getFileName().toString())
                .filter(n -> n.toLowerCase().endsWith(".jpg"))
                .filter(n -> !n.startsWith("xmirror_"))
                .sorted()
                .forEach(originals::add);
        }

        List<String> eligible = new ArrayList<>();
        int excluded = 0;
        int missingMirrors = 0;

        for (String name : originals) {

            String base = name.substring(0, name.lastIndexOf('.'));
            Path label = DATA.resolve(base + ".txt");

            if (!Files.isRegularFile(label)) {
                excluded++;
                continue;
            }

            eligible.add(name);

            if (!Files.isRegularFile(
                DATA.resolve("xmirror_" + name)
            )) {
                missingMirrors++;
            }
        }

        Collections.shuffle(eligible, new Random(SEED));

        int n = eligible.size();
        int trainEnd = (int) (n * 0.80);
        int valEnd = trainEnd + (int) (n * 0.10);

        List<String> train = eligible.subList(0, trainEnd);
        List<String> val = eligible.subList(trainEnd, valEnd);
        List<String> test = eligible.subList(valEnd, n);

        Files.createDirectories(OUTPUT);

        writeManifest(OUTPUT.resolve("train.csv"), train, true);
        writeManifest(OUTPUT.resolve("val.csv"), val, false);
        writeManifest(OUTPUT.resolve("test.csv"), test, false);

        System.out.println("===== DATASET SPLIT REPORT =====");
        System.out.println("Original images: " + originals.size());
        System.out.println("Excluded originals: " + excluded);
        System.out.println("Missing mirror images: " + missingMirrors);
        System.out.println("Training originals: " + train.size());
        System.out.println("Validation originals: " + val.size());
        System.out.println("Testing originals: " + test.size());
        System.out.println("Manifests saved in: " +
            OUTPUT.toAbsolutePath());
    }

    private static void writeManifest(
        Path file,
        List<String> originals,
        boolean includeMirrors
    ) throws IOException {

        List<String> rows = new ArrayList<>();
        rows.add("image,label");

        for (String image : originals) {

            String base = image.substring(
                0, image.lastIndexOf('.')
            );

            rows.add(image + "," + base + ".txt");

            if (includeMirrors) {
                String mirror = "xmirror_" + image;
                String mirrorLabel = "xmirror_" + base + ".txt";

                if (Files.isRegularFile(DATA.resolve(mirror))
                    && Files.isRegularFile(DATA.resolve(mirrorLabel))) {

                    rows.add(mirror + "," + mirrorLabel);
                }
            }
        }

        Files.write(file, rows, StandardCharsets.UTF_8);
    }
}
