
package com.roadguard.training;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;

public class RoadDamageV2ForwardPass {

    public static void main(String[] args) {

        try (Model model = Model.newInstance(
                "roadguard-cnn-v2", "PyTorch");
             NDManager manager =
                NDManager.newBaseManager("PyTorch")) {

            model.setBlock(
                RoadDamageCNNV2.createNetwork()
            );

            Shape inputShape = new Shape(
                1, 3, 224, 224
            );

            model.getBlock().initialize(
                manager,
                DataType.FLOAT32,
                inputShape
            );

            NDArray input = manager.zeros(inputShape);

            NDList output = model.getBlock().forward(
                new ParameterStore(manager, false),
                new NDList(input),
                false
            );

            NDArray logits = output.singletonOrThrow();

            System.out.println(
                "===== ROADGUARD CNN V2 FORWARD PASS ====="
            );

            System.out.println(
                "Input shape: " + input.getShape()
            );

            System.out.println(
                "Output shape: " + logits.getShape()
            );

            if (!logits.getShape().equals(
                    new Shape(1, NUM_CLASSES()))) {
                throw new IllegalStateException(
                    "Unexpected output shape"
                );
            }

            System.out.println(
                "CNN V2 FORWARD PASS SUCCESSFUL"
            );
        }
    }

    private static int NUM_CLASSES() {
        return RoadDamageCNNV2.NUM_CLASSES;
    }
}
