
package com.roadguard.preprocessing;

public class BoundingBoxTransformer {

    public static double[] transform(
            double[] box,
            int originalWidth,
            int originalHeight) {

        if (box == null || box.length != 4
                || originalWidth <= 0
                || originalHeight <= 0) {
            throw new IllegalArgumentException(
                "Invalid box or image dimensions"
            );
        }

        for (double value : box) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                    "Box coordinates must be finite"
                );
            }
        }

        double x = box[0];
        double y = box[1];
        double w = box[2];
        double h = box[3];

        if (x < 0 || x > 1 || y < 0 || y > 1
                || w <= 0 || h <= 0
                || x - w / 2 < -1e-9
                || x + w / 2 > 1 + 1e-9
                || y - h / 2 < -1e-9
                || y + h / 2 > 1 + 1e-9) {
            throw new IllegalArgumentException(
                "Invalid normalized YOLO box"
            );
        }

        int size = LetterboxPreprocessor.SIZE;

        double scale = Math.min(
            (double) size / originalWidth,
            (double) size / originalHeight
        );

        int newWidth = Math.max(
            1, (int) Math.round(originalWidth * scale)
        );

        int newHeight = Math.max(
            1, (int) Math.round(originalHeight * scale)
        );

        int padX = (size - newWidth) / 2;
        int padY = (size - newHeight) / 2;

        // Convert normalized coordinates to
        // the letterboxed image coordinate system.
        double newX =
            (x * newWidth + padX) / size;

        double newY =
            (y * newHeight + padY) / size;

        double newW = w * newWidth / size;
        double newH = h * newHeight / size;

        return new double[]{
            newX, newY, newW, newH
        };
    }

    public static void main(String[] args) {

        double[] box = {0.5, 0.5, 0.4, 0.2};

        double[] result = transform(
            box, 400, 200
        );

        System.out.printf(
            "Transformed YOLO box: %.4f %.4f %.4f %.4f%n",
            result[0],
            result[1],
            result[2],
            result[3]
        );
    }
}
