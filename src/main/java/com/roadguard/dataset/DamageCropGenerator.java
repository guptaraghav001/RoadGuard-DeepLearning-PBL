
package com.roadguard.dataset;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import javax.imageio.ImageIO;

public class DamageCropGenerator {

    private static final Path IMAGE_DIR = Path.of(
        "data/raw/RDD2022ES/combined_annotatedv2"
    );

    private static final Path LABEL_DIR = Path.of(
        "data/processed/RDD2022ES/labels"
    );

    private static final Path OUTPUT_DIR = Path.of(
        "data/processed/cnn_crops"
    );

    private static final int SIZE = 224;

    public static int processImage(
            String imageName,
            String labelName,
            String split) throws IOException {

        if (!List.of("train", "val", "test")
                .contains(split)) {
            throw new IllegalArgumentException(
                "Invalid dataset split"
            );
        }

        Path imagePath = IMAGE_DIR.resolve(imageName);
        Path labelPath = LABEL_DIR.resolve(labelName);

        if (!Files.isRegularFile(imagePath)
                || !Files.isRegularFile(labelPath)) {
            throw new IOException(
                "Missing image or label: " + imageName
            );
        }

        BufferedImage image =
            ImageIO.read(imagePath.toFile());

        if (image == null) {
            throw new IOException(
                "Unable to read image: " + imageName
            );
        }

        List<String> labels =
            Files.readAllLines(labelPath);

        int generated = 0;

        for (int index = 0; index < labels.size(); index++) {

            String line = labels.get(index).trim();

            if (line.isEmpty()) {
                continue;
            }

            String[] parts = line.split("\\s+");

            if (parts.length != 5) {
                System.out.println(
                    "Skipping malformed label: " + line
                );
                continue;
            }

            try {
                int classId = Integer.parseInt(parts[0]);

                double cx = Double.parseDouble(parts[1]);
                double cy = Double.parseDouble(parts[2]);
                double w = Double.parseDouble(parts[3]);
                double h = Double.parseDouble(parts[4]);

                if (classId < 0 || classId > 7
                        || !Double.isFinite(cx)
                        || !Double.isFinite(cy)
                        || !Double.isFinite(w)
                        || !Double.isFinite(h)
                        || cx < 0 || cx > 1
                        || cy < 0 || cy > 1
                        || w <= 0 || h <= 0) {
                    continue;
                }

                int imageW = image.getWidth();
                int imageH = image.getHeight();

                int x1 = (int) Math.floor(
                    (cx - w / 2.0) * imageW
                );

                int y1 = (int) Math.floor(
                    (cy - h / 2.0) * imageH
                );

                int x2 = (int) Math.ceil(
                    (cx + w / 2.0) * imageW
                );

                int y2 = (int) Math.ceil(
                    (cy + h / 2.0) * imageH
                );

                x1 = Math.max(0, Math.min(imageW, x1));
                y1 = Math.max(0, Math.min(imageH, y1));
                x2 = Math.max(0, Math.min(imageW, x2));
                y2 = Math.max(0, Math.min(imageH, y2));

                int cropW = x2 - x1;
                int cropH = y2 - y1;

                if (cropW <= 0 || cropH <= 0) {
                    continue;
                }

                // Add 15% context around the damage.
                int marginX = (int) Math.ceil(cropW * 0.15);
                int marginY = (int) Math.ceil(cropH * 0.15);

                int left = Math.max(0, x1 - marginX);
                int top = Math.max(0, y1 - marginY);
                int right = Math.min(imageW, x2 + marginX);
                int bottom = Math.min(imageH, y2 + marginY);

                int regionW = right - left;
                int regionH = bottom - top;

                if (regionW <= 0 || regionH <= 0) {
                    continue;
                }

                BufferedImage crop = image.getSubimage(
                    left, top, regionW, regionH
                );

                // Preserve aspect ratio.
                double scale = Math.min(
                    (double) SIZE / regionW,
                    (double) SIZE / regionH
                );

                int resizedW = Math.max(
                    1, (int) Math.round(regionW * scale)
                );
                int resizedH = Math.max(
                    1, (int) Math.round(regionH * scale)
                );

                int padX = (SIZE - resizedW) / 2;
                int padY = (SIZE - resizedH) / 2;

                BufferedImage resized = new BufferedImage(
                    SIZE, SIZE, BufferedImage.TYPE_INT_RGB
                );

                Graphics2D graphics =
                    resized.createGraphics();

                try {
                    graphics.setColor(
                        new java.awt.Color(114, 114, 114)
                    );
                    graphics.fillRect(0, 0, SIZE, SIZE);

                    graphics.setRenderingHint(
                        RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR
                    );

                    graphics.drawImage(
                        crop, padX, padY,
                        resizedW, resizedH, null
                    );
                } finally {
                    graphics.dispose();
                }



                Path classDir = OUTPUT_DIR
                    .resolve(split)
                    .resolve("class_" + classId);

                Files.createDirectories(classDir);

                String baseName = imageName.replaceFirst(
                    "\\.[^.]+$", ""
                );

                Path output = classDir.resolve(
                    baseName + "_box" + index + ".png"
                );

                ImageIO.write(
                    resized, "png", output.toFile()
                );

                generated++;

            } catch (NumberFormatException e) {
                System.out.println(
                    "Skipping invalid annotation: " + line
                );
            }
        }

        return generated;
    }

    public static void main(String[] args)
            throws IOException {

        System.out.println(
            "===== ROADGUARD DAMAGE CROP GENERATOR ====="
        );

        int count = processImage(
            "India_005741.jpg",
            "India_005741.txt",
            "train"
        );

        System.out.println(
            "Generated crops: " + count
        );

        System.out.println(
            "Crop generation completed."
        );
    }
}
