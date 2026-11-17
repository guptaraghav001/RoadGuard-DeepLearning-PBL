
package com.roadguard.training;

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.core.Linear;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;

public class RoadDamageCNNV2 {

    public static final int NUM_CLASSES = 8;

    private static void addConvBlock(
            SequentialBlock network,
            int filters) {

        network.add(
            Conv2d.builder()
                .setFilters(filters)
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .build()
        );

        network.add(
            BatchNorm.builder().build()
        );

        network.add(Activation::relu);

        network.add(
            Pool.maxPool2dBlock(
                new Shape(2, 2),
                new Shape(2, 2)
            )
        );
    }

    public static Block createNetwork() {

        SequentialBlock network = new SequentialBlock();

        addConvBlock(network, 16);
        addConvBlock(network, 32);
        addConvBlock(network, 64);

        // Global average pooling.
        // Input: [batch, 64, 28, 28]
        // Output: [batch, 64]
        // Reduce one axis at a time for DJL PyTorch.

        network.add(new LambdaBlock(
            arrays -> {
                NDArray x = arrays.singletonOrThrow();

                x = x.mean(new int[]{2}, false);
                x = x.mean(new int[]{2}, false);

                return new NDList(x);
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

        try (Model model = Model.newInstance(
                "roadguard-cnn-v2", "PyTorch")) {

            model.setBlock(createNetwork());

            System.out.println(
                "===== ROADGUARD CNN VERSION 2 ====="
            );

            System.out.println(
                "Input: 3 x 224 x 224"
            );

            System.out.println(
                "Convolution filters: 16, 32, 64"
            );

            System.out.println(
                "Batch normalization: enabled"
            );

            System.out.println(
                "Output classes: " + NUM_CLASSES
            );

            System.out.println(
                "CNN V2 ARCHITECTURE CREATED"
            );
        }
    }
}
