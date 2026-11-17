
package com.roadguard.training;

import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.Pipeline;

import java.nio.file.Files;
import java.nio.file.Path;

public class RoadDamageDatasetCheck {

    public static void main(String[] args)
            throws Exception {

        Path trainDir = Path.of(
            "data/processed/cnn_crops_complete/train"
        );

        if (!Files.isDirectory(trainDir)) {
            throw new IllegalArgumentException(
                "Training directory not found: " + trainDir
            );
        }

        Pipeline pipeline = new Pipeline();

        pipeline.add(new Resize(224, 224));
        pipeline.add(new ToTensor());

        ImageFolder trainDataset = ImageFolder.builder()
            .setRepositoryPath(trainDir)
            .addTransform(new Resize(224, 224))
            .addTransform(new ToTensor())
            .setSampling(8, true)
            .build();

        trainDataset.prepare();

        System.out.println(
            "===== ROADGUARD DATASET CHECK ====="
        );

        System.out.println(
            "Training samples: "
            + trainDataset.size()
        );

        System.out.println(
            "Class names: "
            + trainDataset.getSynset()
        );

        System.out.println(
            "DATASET LOADER READY"
        );
    }
}
