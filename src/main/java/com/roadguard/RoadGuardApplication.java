
package com.roadguard;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class RoadGuardApplication {

    public static void main(String[] args)
            throws IOException, InterruptedException {

        if (args.length == 0) {
            System.out.println(
                "Usage: RoadGuardApplication <image_path>"
            );
            return;
        }

        Path image = Path.of(args[0]);

        if (!Files.isRegularFile(image)) {
            throw new IllegalArgumentException(
                "Image not found: " + image
            );
        }

        Path python = Path.of(
            ".venv-yolo/bin/python"
        );

        if (!Files.isRegularFile(python)) {
            throw new IllegalStateException(
                "YOLO Python environment not found."
            );
        }

        List<String> command = new ArrayList<>();

        command.add(python.toString());
        command.add("scripts/roadguard_inference.py");
        command.add("--image");
        command.add(image.toString());

        ProcessBuilder builder =
            new ProcessBuilder(command);

        builder.inheritIO();

        System.out.println(
            "===== ROADGUARD JAVA APPLICATION ====="
        );

        System.out.println(
            "Processing image: " + image
        );

        Process process = builder.start();

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new IllegalStateException(
                "Inference failed. Exit code: " + exitCode
            );
        }

        System.out.println(
            "ROADGUARD ASSESSMENT COMPLETED"
        );
    }
}
