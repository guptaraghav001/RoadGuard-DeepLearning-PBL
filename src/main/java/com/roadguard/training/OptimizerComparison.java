
package com.roadguard.training;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class OptimizerComparison {

    private static final int EPOCHS = 2000;
    private static final double LR = 0.5;
    private static final double MOMENTUM = 0.9;

    private static final double[][] X = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    private static final double[] Y = {0, 1, 1, 0};

    private static final String[] METHODS = {
        "BatchGD", "SGD", "Momentum", "Nesterov"
    };

    private static double sigmoid(double x) {
        if (x >= 0) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        double e = Math.exp(x);
        return e / (1.0 + e);
    }

    private static class Network {
        double[] p = new double[13];

        Network() {
            Random random = new Random(42);
            for (int i = 0; i < p.length; i++) {
                p[i] = random.nextDouble() * 2 - 1;
            }
        }

        double predict(double[] x) {
            double[] h = new double[3];

            for (int j = 0; j < 3; j++) {
                int k = j * 3;
                h[j] = sigmoid(
                    p[k] * x[0] +
                    p[k + 1] * x[1] +
                    p[k + 2]
                );
            }

            return sigmoid(
                p[9] * h[0] +
                p[10] * h[1] +
                p[11] * h[2] +
                p[12]
            );
        }

        double[] gradient(double[] x, double y) {
            double[] h = new double[3];

            for (int j = 0; j < 3; j++) {
                int k = j * 3;
                h[j] = sigmoid(
                    p[k] * x[0] +
                    p[k + 1] * x[1] +
                    p[k + 2]
                );
            }

            double output = predict(x);
            double delta = output - y;

            double[] g = new double[13];

            for (int j = 0; j < 3; j++) {
                g[9 + j] = delta * h[j];
            }
            g[12] = delta;

            for (int j = 0; j < 3; j++) {
                double d = delta * p[9 + j]
                    * h[j] * (1 - h[j]);

                int k = j * 3;
                g[k] = d * x[0];
                g[k + 1] = d * x[1];
                g[k + 2] = d;
            }

            return g;
        }

        double loss() {
            double total = 0;
            for (int i = 0; i < X.length; i++) {
                double pred = predict(X[i]);
                pred = Math.max(
                    1e-12, Math.min(1 - 1e-12, pred)
                );
                total += -Y[i] * Math.log(pred)
                    - (1 - Y[i]) * Math.log(1 - pred);
            }
            return total / X.length;
        }

        double accuracy() {
            int correct = 0;
            for (int i = 0; i < X.length; i++) {
                int prediction =
                    predict(X[i]) >= 0.5 ? 1 : 0;
                if (prediction == (int) Y[i]) {
                    correct++;
                }
            }
            return correct / (double) X.length;
        }
    }

    private static void train(
            String method, List<String> results) {

        Network net = new Network();
        double[] velocity = new double[13];

        double initialLoss = net.loss();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {

            if (method.equals("SGD")) {
                for (int i = 0; i < X.length; i++) {
                    double[] g = net.gradient(X[i], Y[i]);
                    for (int j = 0; j < 13; j++) {
                        net.p[j] -= LR * g[j];
                    }
                }
            } else {

                if (method.equals("Nesterov")) {
                    for (int j = 0; j < 13; j++) {
                        net.p[j] += MOMENTUM * velocity[j];
                    }
                }

                double[] avg = new double[13];

                for (int i = 0; i < X.length; i++) {
                    double[] g = net.gradient(X[i], Y[i]);
                    for (int j = 0; j < 13; j++) {
                        avg[j] += g[j] / X.length;
                    }
                }

                if (method.equals("Nesterov")) {
                    for (int j = 0; j < 13; j++) {
                        net.p[j] -= MOMENTUM * velocity[j];
                    }
                }

                for (int j = 0; j < 13; j++) {
                    if (method.equals("BatchGD")) {
                        net.p[j] -= LR * avg[j];
                    } else {
                        velocity[j] = MOMENTUM * velocity[j]
                            - LR * avg[j];
                        net.p[j] += velocity[j];
                    }
                }
            }
        }

        String row = String.format(
            java.util.Locale.US,
            "%s,%.6f,%.6f,%.2f",
            method, initialLoss,
            net.loss(), net.accuracy() * 100
        );

        results.add(row);

        System.out.println(row);
    }

    public static void main(String[] args)
            throws Exception {

        Path output = Path.of(
            "experiments/optimizer_comparison"
        );
        Files.createDirectories(output);

        List<String> rows = new ArrayList<>();
        rows.add(
            "optimizer,initial_loss,final_loss,accuracy_percent"
        );

        System.out.println(
            "===== ROADGUARD OPTIMIZER COMPARISON ====="
        );

        for (String method : METHODS) {
            train(method, rows);
        }

        Files.write(
            output.resolve("optimizer_results.csv"),
            rows
        );

        System.out.println(
            "OPTIMIZER COMPARISON COMPLETED"
        );
    }
}
