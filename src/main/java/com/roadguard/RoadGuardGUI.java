
package com.roadguard;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ExecutionException;
import javax.imageio.ImageIO;

public class RoadGuardGUI extends JFrame {

    private Path selectedImage;

    private final JLabel imageLabel = new JLabel(
        "Select a road image to begin",
        SwingConstants.CENTER
    );

    private final JTextArea results = new JTextArea();

    private final JButton selectButton =
        new JButton("Select Road Image");

    private final JButton analyzeButton =
        new JButton("Analyze Road");

    private final JButton openImageButton =
        new JButton("Open Annotated Image");

    private final JButton openReportButton =
        new JButton("Open JSON Report");

    private final JLabel statusLabel =
        new JLabel("Ready");

    private Path annotatedImage;
    private Path reportFile;

    public RoadGuardGUI() {

        setTitle(
            "RoadGuard - Road Damage Assessment System"
        );

        setSize(1050, 750);
        setMinimumSize(new Dimension(750, 550));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JPanel root = new JPanel(
            new BorderLayout(12, 12)
        );

        root.setBorder(
            new EmptyBorder(15, 15, 15, 15)
        );

        JLabel title = new JLabel(
            "ROADGUARD - ROAD DAMAGE ASSESSMENT"
        );

        title.setFont(
            new Font("SansSerif", Font.BOLD, 22)
        );

        root.add(title, BorderLayout.NORTH);

        // Image preview
        imageLabel.setOpaque(true);
        imageLabel.setBackground(
            new Color(235, 238, 242)
        );

        JScrollPane imageScroll =
            new JScrollPane(imageLabel);

        // Results panel
        results.setEditable(false);
        results.setFont(
            new Font("Monospaced", Font.PLAIN, 14)
        );

        results.setText(
            "RoadGuard is ready.\n\n"
            + "1. Select a road image.\n"
            + "2. Click Analyze Road.\n"
            + "3. View detections and results."
        );

        JScrollPane resultScroll =
            new JScrollPane(results);

        JSplitPane splitPane = new JSplitPane(
            JSplitPane.VERTICAL_SPLIT,
            imageScroll,
            resultScroll
        );

        splitPane.setResizeWeight(0.72);

        root.add(splitPane, BorderLayout.CENTER);

        // Bottom controls
        JPanel controls = new JPanel(
            new FlowLayout(FlowLayout.CENTER, 12, 8)
        );

        controls.add(selectButton);
        controls.add(analyzeButton);
        controls.add(openImageButton);
        controls.add(openReportButton);

        JPanel bottom = new JPanel(
            new BorderLayout()
        );

        bottom.add(controls, BorderLayout.CENTER);
        bottom.add(statusLabel, BorderLayout.SOUTH);

        root.add(bottom, BorderLayout.SOUTH);

        setContentPane(root);

        analyzeButton.setEnabled(false);
        openImageButton.setEnabled(false);
        openReportButton.setEnabled(false);

        selectButton.addActionListener(
            e -> selectImage()
        );

        analyzeButton.addActionListener(
            e -> analyzeImage()
        );

        openImageButton.addActionListener(
            e -> openFile(annotatedImage)
        );

        openReportButton.addActionListener(
            e -> openFile(reportFile)
        );
    }

    private void selectImage() {

        JFileChooser chooser = new JFileChooser();

        chooser.setDialogTitle(
            "Choose a Road Photograph"
        );

        chooser.setFileFilter(
            new FileNameExtensionFilter(
                "Image Files", "jpg", "jpeg", "png"
            )
        );

        int result = chooser.showOpenDialog(this);

        if (result != JFileChooser.APPROVE_OPTION) {
            return;
        }

        selectedImage =
            chooser.getSelectedFile().toPath()
                .toAbsolutePath();

        annotatedImage = null;
        reportFile = null;

        analyzeButton.setEnabled(true);
        openImageButton.setEnabled(false);
        openReportButton.setEnabled(false);

        statusLabel.setText(
            "Selected: " + selectedImage.getFileName()
        );

        results.setText(
            "Selected image:\n" + selectedImage
            + "\n\nClick Analyze Road to continue."
        );

        displayImage(selectedImage);
    }

    private void analyzeImage() {

        if (selectedImage == null) {
            return;
        }

        selectButton.setEnabled(false);
        analyzeButton.setEnabled(false);
        openImageButton.setEnabled(false);
        openReportButton.setEnabled(false);

        statusLabel.setText(
            "Analyzing road image... Please wait."
        );

        results.setText(
            "Running trained YOLO detector...\n"
            + "This may take a few seconds."
        );

        Path input = selectedImage;

        new SwingWorker<Integer, Void>() {

            @Override
            protected Integer doInBackground()
                    throws Exception {

                Path python = Path.of(
                    ".venv-yolo/bin/python"
                );

                Path script = Path.of(
                    "scripts/roadguard_inference.py"
                );

                if (!Files.isRegularFile(python)) {
                    throw new IOException(
                        "Python environment missing: "
                        + python
                    );
                }

                if (!Files.isRegularFile(script)) {
                    throw new IOException(
                        "Inference script missing: "
                        + script
                    );
                }

                List<String> command = List.of(
                    python.toAbsolutePath().toString(),
                    script.toString(),
                    "--image",
                    input.toString()
                );

                ProcessBuilder builder =
                    new ProcessBuilder(command);

                builder.redirectErrorStream(true);

                Process process = builder.start();

                try (
                    var reader = new java.io.BufferedReader(
                        new java.io.InputStreamReader(
                            process.getInputStream(),
                            StandardCharsets.UTF_8
                        )
                    )
                ) {
                    String line;

                    while ((line = reader.readLine())
                            != null) {
                        System.out.println(line);
                    }
                }

                return process.waitFor();
            }

            @Override
            protected void done() {

                selectButton.setEnabled(true);
                analyzeButton.setEnabled(true);

                try {
                    int exitCode = get();

                    if (exitCode != 0) {
                        throw new IOException(
                            "Inference exited with code "
                            + exitCode
                        );
                    }

                    String baseName =
                        input.getFileName().toString()
                            .replaceFirst("\\.[^.]+$", "");

                    annotatedImage = Path.of(
                        "outputs/roadguard",
                        baseName + "_annotated.jpg"
                    );

                    reportFile = Path.of(
                        "outputs/roadguard",
                        baseName + "_report.json"
                    );

                    if (!Files.isRegularFile(annotatedImage)
                            || !Files.isRegularFile(reportFile)) {
                        throw new IOException(
                            "Expected output files missing"
                        );
                    }

                    displayImage(annotatedImage);

                    String json = Files.readString(
                        reportFile
                    );

                    results.setText(
                        "ROADGUARD ASSESSMENT COMPLETED\n\n"
                        + json
                    );

                    openImageButton.setEnabled(true);
                    openReportButton.setEnabled(true);

                    statusLabel.setText(
                        "Analysis completed successfully"
                    );

                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    showError(e);

                } catch (ExecutionException
                         | IOException e) {
                    showError(e);
                }
            }
        }.execute();
    }

    private void displayImage(Path path) {

        try {
            BufferedImage image = ImageIO.read(
                path.toFile()
            );

            if (image == null) {
                throw new IOException(
                    "Unable to read image"
                );
            }

            int maxWidth = 900;
            int maxHeight = 480;

            double scale = Math.min(
                (double) maxWidth / image.getWidth(),
                (double) maxHeight / image.getHeight()
            );

            scale = Math.min(scale, 1.0);

            int width = Math.max(
                1,
                (int) (image.getWidth() * scale)
            );

            int height = Math.max(
                1,
                (int) (image.getHeight() * scale)
            );

            Image scaled = image.getScaledInstance(
                width,
                height,
                Image.SCALE_SMOOTH
            );

            imageLabel.setIcon(
                new ImageIcon(scaled)
            );

            imageLabel.setText("");

        } catch (IOException e) {
            showError(e);
        }
    }

    private void openFile(Path path) {

        if (path == null || !Files.isRegularFile(path)) {
            JOptionPane.showMessageDialog(
                this,
                "File not available."
            );
            return;
        }

        try {
            Desktop.getDesktop().open(
                path.toFile()
            );

        } catch (IOException
                 | UnsupportedOperationException e) {
            showError(e);
        }
    }

    private void showError(Exception e) {

        statusLabel.setText(
            "Error: " + e.getMessage()
        );

        JOptionPane.showMessageDialog(
            this,
            e.getMessage(),
            "RoadGuard Error",
            JOptionPane.ERROR_MESSAGE
        );
    }

    public static void main(String[] args) {

        SwingUtilities.invokeLater(() -> {
            RoadGuardGUI gui = new RoadGuardGUI();
            gui.setVisible(true);
        });
    }
}
