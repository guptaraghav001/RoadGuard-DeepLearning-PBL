
package com.roadguard.training;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class CNNForwardPass {

    public static void main(String[] args) {

        try (Model model =
                Model.newInstance("roadguard-cnn", "PyTorch");
             NDManager manager =
                NDManager.newBaseManager("PyTorch")) {

            model.setBlock(RoadDamageCNN.createNetwork());

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
                new ai.djl.training.ParameterStore(
                    manager, false
                ),
                new NDList(input),
                false
            );

            NDArray logits = output.singletonOrThrow();

            System.out.println(
                "===== CNN FORWARD PASS ====="
            );

            System.out.println(
                "Input shape: " + input.getShape()
            );

            System.out.println(
                "Output shape: " + logits.getShape()
            );

            System.out.println(
                "CNN FORWARD PASS SUCCESSFUL"
            );
        }
    }
}
