
package com.roadguard.training;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class RoadDamageTrainer {

    private static final Path ROOT =
        Path.of("data/processed/cnn_crops_complete");

    private static final Path MODEL_DIR =
        Path.of("models/roadguard_cnn");

    private static final int BATCH_SIZE = 8;
    private static final int EPOCHS = 2;

    private static ImageFolder loadDataset(
            String split, boolean shuffle)
            throws Exception {

        Path directory = ROOT.resolve(split);

        if (!Files.isDirectory(directory)) {
            throw new IllegalArgumentException(
                "Dataset directory missing: " + directory
            );
        }

        ImageFolder dataset = ImageFolder.builder()
            .setRepositoryPath(directory)
            .addTransform(new ToTensor())
            .setSampling(BATCH_SIZE, shuffle)
            .build();

        dataset.prepare();

        List<String> classes = dataset.getSynset();

        if (classes.size() != 8) {
            throw new IllegalStateException(
                "Expected 8 classes, found: " + classes
            );
        }

        for (int i = 0; i < 8; i++) {
            if (!classes.get(i).equals("class_" + i)) {
                throw new IllegalStateException(
                    "Unexpected class mapping: " + classes
                );
            }
        }

        return dataset;
    }

    public static void main(String[] args)
            throws Exception {

        System.out.println(
            "===== ROADGUARD CNN TRAINING ====="
        );

        ImageFolder train = loadDataset("train", true);
        ImageFolder validation = loadDataset("val", false);

        System.out.println(
            "Training samples: " + train.size()
        );

        System.out.println(
            "Validation samples: " + validation.size()
        );

        DefaultTrainingConfig config =
            new DefaultTrainingConfig(
                Loss.softmaxCrossEntropyLoss()
            )
            .optOptimizer(
                Adam.builder()
                    .optLearningRateTracker(
                        Tracker.fixed(0.001f)
                    )
                    .build()
            )
            .addEvaluator(new Accuracy())
            .addTrainingListeners(
                TrainingListener.Defaults.logging()
            );

        try (Model model =
                Model.newInstance(
                    "roadguard-cnn", "PyTorch"
                )) {

            model.setBlock(
                RoadDamageCNN.createNetwork()
            );

            try (Trainer trainer =
                    model.newTrainer(config)) {

                trainer.initialize(
                    new Shape(1, 3, 224, 224)
                );

                EasyTrain.fit(
                    trainer,
                    EPOCHS,
                    train,
                    validation
                );

                Files.createDirectories(MODEL_DIR);

                model.save(
                    MODEL_DIR,
                    "roadguard-cnn"
                );

                System.out.println(
                    "Model saved to: "
                    + MODEL_DIR.toAbsolutePath()
                );
            }
        }

        System.out.println(
            "ROADGUARD CNN TRAINING COMPLETED"
        );
    }
}
