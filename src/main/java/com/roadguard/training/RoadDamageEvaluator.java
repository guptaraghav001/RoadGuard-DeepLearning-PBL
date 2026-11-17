
package com.roadguard.training;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import java.util.List;
import javax.imageio.ImageIO;

public class RoadDamageEvaluator {

    private static final Path TEST_DIR = Path.of(
        "data/processed/cnn_crops_complete/test"
    );

    private static final Path MODEL_DIR = Path.of(
        "models/roadguard_cnn"
    );

    private static final int CLASSES = 8;
    private static final int SIZE = 224;

    // Convert RGB image into normalized CHW tensor.
    private static float[] imageToTensor(
            BufferedImage image) {

        float[] data = new float[3 * SIZE * SIZE];

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {

                int rgb = image.getRGB(x, y);
                int index = y * SIZE + x;

                data[index] =
                    ((rgb >> 16) & 255) / 255.0f;

                data[SIZE * SIZE + index] =
                    ((rgb >> 8) & 255) / 255.0f;

                data[2 * SIZE * SIZE + index] =
                    (rgb & 255) / 255.0f;
            }
        }

        return data;
    }

    public static void main(String[] args)
            throws Exception {

        System.out.println(
            "===== ROADGUARD CNN EVALUATION ====="
        );

        if (!Files.isDirectory(TEST_DIR)) {
            throw new IllegalStateException(
                "Test directory missing"
            );
        }

        int[][] confusion =
            new int[CLASSES][CLASSES];

        int total = 0;
        int correct = 0;

        try (Model model = Model.newInstance(
                "roadguard-cnn", "PyTorch");
             NDManager manager =
                NDManager.newBaseManager("PyTorch")) {

            model.setBlock(
                RoadDamageCNN.createNetwork()
            );

            model.load(MODEL_DIR, "roadguard-cnn");

            ParameterStore parameterStore =
                new ParameterStore(manager, false);

            for (int classId = 0;
                    classId < CLASSES; classId++) {

                Path classDir = TEST_DIR.resolve(
                    "class_" + classId
                );

                if (!Files.isDirectory(classDir)) {
                    throw new IllegalStateException(
                        "Missing class: " + classDir
                    );
                }

                List<Path> images;

                try (Stream<Path> stream =
                         Files.list(classDir)) {

                    images = stream
                        .filter(p -> p.toString()
                            .toLowerCase()
                            .endsWith(".png"))
                        .sorted()
                        .toList();
                }

                for (Path imagePath : images) {

                    BufferedImage image =
                        ImageIO.read(imagePath.toFile());

                    if (image == null
                            || image.getWidth() != SIZE
                            || image.getHeight() != SIZE) {
                        throw new IllegalStateException(
                            "Invalid image: " + imagePath
                        );
                    }

                    try (NDManager batchManager =
                             manager.newSubManager()) {

                        float[] pixels =
                            imageToTensor(image);

                        NDArray input = batchManager.create(
                            pixels,
                            new Shape(1, 3, SIZE, SIZE)
                        );

                        NDList output =
                            model.getBlock().forward(
                                parameterStore,
                                new NDList(input),
                                false
                            );

                        NDArray logits =
                            output.singletonOrThrow();

                        int predicted = (int)
                            logits.argMax(1)
                                .toLongArray()[0];

                        confusion[classId][predicted]++;

                        if (predicted == classId) {
                            correct++;
                        }

                        total++;
                    }
                }

                System.out.println(
                    "Evaluated class_" + classId
                    + ": " + images.size() + " images"
                );
            }
        }

        if (total == 0) {
            throw new IllegalStateException(
                "No test images found"
            );
        }

        double accuracy =
            100.0 * correct / total;

        System.out.println(
            "\n===== EVALUATION RESULTS ====="
        );

        System.out.println("Total images: " + total);
        System.out.println("Correct: " + correct);

        System.out.printf(
            "Test accuracy: %.2f%%%n", accuracy
        );

        double macroF1 = 0.0;

        System.out.println(
            "\nClass | Precision | Recall | F1"
        );

        for (int c = 0; c < CLASSES; c++) {

            int tp = confusion[c][c];
            int fp = 0;
            int fn = 0;

            for (int j = 0; j < CLASSES; j++) {
                if (j != c) {
                    fp += confusion[j][c];
                    fn += confusion[c][j];
                }
            }

            double precision = tp + fp == 0
                ? 0 : (double) tp / (tp + fp);

            double recall = tp + fn == 0
                ? 0 : (double) tp / (tp + fn);

            double f1 = precision + recall == 0
                ? 0
                : 2 * precision * recall
                    / (precision + recall);

            macroF1 += f1;

            System.out.printf(
                "%d | %.4f | %.4f | %.4f%n",
                c, precision, recall, f1
            );
        }

        System.out.printf(
            "\nMacro F1: %.4f%n",
            macroF1 / CLASSES
        );

        System.out.println(
            "\n===== CONFUSION MATRIX ====="
        );

        System.out.println(
            "Rows = actual, columns = predicted"
        );

        for (int[] row : confusion) {
            for (int value : row) {
                System.out.printf("%6d", value);
            }
            System.out.println();
        }

        System.out.println(
            "\nROADGUARD EVALUATION COMPLETED"
        );
    }
}
