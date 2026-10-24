
package com.roadguard.training;

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.core.Linear;
import ai.djl.nn.LambdaBlock;
import ai.djl.ndarray.types.Shape;

public class RoadDamageCNN {

    public static final int NUM_CLASSES = 8;

    public static Block createNetwork() {

        SequentialBlock network = new SequentialBlock();

        network.add(
            Conv2d.builder()
                .setFilters(16)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build()
        );

        network.add(Activation::relu);
        network.add(Pool.maxPool2dBlock(
            new Shape(2, 2),
            new Shape(2, 2)
        ));

        network.add(
            Conv2d.builder()
                .setFilters(32)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build()
        );

        network.add(Activation::relu);
        network.add(Pool.maxPool2dBlock(
            new Shape(2, 2),
            new Shape(2, 2)
        ));

        // Average spatial dimensions to 1x1.
      
// Global Average Pooling
// Input shape: [batch, 32, height, width]
// Output shape: [batch, 32]

network.add(new LambdaBlock(
    arrays -> {
        ai.djl.ndarray.NDArray x =
            arrays.singletonOrThrow();

        // Average over height
        x = x.mean(new int[]{2}, false);

        // Average over width (now axis 2)
        x = x.mean(new int[]{2}, false);

        return new ai.djl.ndarray.NDList(x);
    }
));


        network.add(
            Linear.builder()
                .setUnits(NUM_CLASSES)
                .build()
        );

        return network;
    }

    public static void main(String[] args) {

        try (Model model =
                Model.newInstance("roadguard-cnn", "PyTorch")) {

            model.setBlock(createNetwork());

            System.out.println(
                "===== ROADGUARD CNN ARCHITECTURE ====="
            );

            System.out.println(
                "Input: 3 x 224 x 224"
            );

            System.out.println(
                "Output classes: " + NUM_CLASSES
            );

            System.out.println(
                model.getBlock()
            );

            System.out.println(
                "CNN ARCHITECTURE CREATED SUCCESSFULLY"
            );
        }
    }
}
