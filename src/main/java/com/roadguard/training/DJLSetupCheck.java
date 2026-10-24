
package com.roadguard.training;

import ai.djl.Model;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;

public class DJLSetupCheck {

    public static void main(String[] args) {

        System.out.println(
            "===== ROADGUARD DJL SETUP CHECK ====="
        );

        try (Model model = Model.newInstance(
                "roadguard-cnn")) {

            SequentialBlock network =
                new SequentialBlock();

            network.add(
                Linear.builder()
                    .setUnits(8)
                    .build()
            );

            model.setBlock(network);

            System.out.println(
                "DJL API initialized successfully."
            );

            System.out.println(
                "Model name: roadguard-cnn"
            );

            System.out.println(
                "Initial classification layer: 8 outputs"
            );

            System.out.println(
                "ROADGUARD DJL SETUP SUCCESSFUL"
            );
        }
    }
}
