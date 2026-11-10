
package com.roadguard.dataset;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import javax.imageio.ImageIO;

public class BoundingBoxVisualizer {

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

    public static BufferedImage visualize(
            Path imagePath,
            Path labelPath) throws IOException {

        BufferedImage original =
            ImageIO.read(imagePath.toFile());

        if (original == null) {
            throw new IOException("Invalid image");
        }

        BufferedImage output = new BufferedImage(
            original.getWidth(),
            original.getHeight(),
            BufferedImage.TYPE_INT_RGB
        );

        Graphics2D g = output.createGraphics();

        try {
            g.drawImage(original, 0, 0, null);

            g.setStroke(new BasicStroke(3));
            g.setFont(new Font(
                "Arial", Font.BOLD, 18
            ));

            List<String> lines =
                Files.readAllLines(labelPath);

            for (String line : lines) {

                String[] p = line.trim().split("\\s+");

                if (p.length != 5) {
                    continue;
                }

                try {
                    int classId = Integer.parseInt(p[0]);

                    if (classId < 0
                            || classId >= CLASSES.length) {
                        continue;
                    }

                    double cx = Double.parseDouble(p[1]);
                    double cy = Double.parseDouble(p[2]);
                    double w = Double.parseDouble(p[3]);
                    double h = Double.parseDouble(p[4]);

                    if (!Double.isFinite(cx)
                            || !Double.isFinite(cy)
                            || !Double.isFinite(w)
                            || !Double.isFinite(h)
                            || cx < 0 || cx > 1
                            || cy < 0 || cy > 1
                            || w <= 0 || h <= 0
                            || cx - w / 2 < 0
                            || cx + w / 2 > 1
                            || cy - h / 2 < 0
                            || cy + h / 2 > 1) {
                        continue;
                    }

                    int imageW = original.getWidth();
                    int imageH = original.getHeight();

                    int x1 = (int) Math.floor(
                        (cx - w / 2) * imageW
                    );

                    int y1 = (int) Math.floor(
                        (cy - h / 2) * imageH
                    );

                    int x2 = (int) Math.ceil(
                        (cx + w / 2) * imageW
                    );

                    int y2 = (int) Math.ceil(
                        (cy + h / 2) * imageH
                    );

                    x1 = Math.max(0, Math.min(imageW, x1));
                    y1 = Math.max(0, Math.min(imageH, y1));
                    x2 = Math.max(0, Math.min(imageW, x2));
                    y2 = Math.max(0, Math.min(imageH, y2));

                    if (x2 <= x1 || y2 <= y1) {
                        continue;
                    }

                    g.setColor(Color.RED);

                    g.drawRect(
                        x1, y1,
                        x2 - x1, y2 - y1
                    );

                    String label = CLASSES[classId];

                    int textY = Math.max(20, y1 - 8);

                    g.drawString(
                        label, x1, textY
                    );

                } catch (NumberFormatException e) {
                    System.out.println(
                        "Skipping invalid annotation: " + line
                    );
                }
            }

        } finally {
            g.dispose();
        }

        return output;
    }

    public static void main(String[] args)
            throws IOException {

        Path imagePath = Path.of(
            "data/raw/RDD2022ES/combined_annotatedv2/"
            + "India_005741.jpg"
        );

        Path labelPath = Path.of(
            "data/processed/RDD2022ES/labels/"
            + "India_005741.txt"
        );

        BufferedImage result = visualize(
            imagePath, labelPath
        );

        Path outputPath = Path.of(
            "bounding_box_preview.png"
        );

        if (!ImageIO.write(
                result, "png", outputPath.toFile())) {
            throw new IOException(
                "PNG writer unavailable"
            );
        }

        System.out.println(
            "===== ROADGUARD BOUNDING BOX VISUALIZATION ====="
        );

        System.out.println(
            "Annotated image saved: "
            + outputPath.toAbsolutePath()
        );

        System.out.println(
            "Image dimensions: "
            + result.getWidth() + " x "
            + result.getHeight()
        );
    }
}
