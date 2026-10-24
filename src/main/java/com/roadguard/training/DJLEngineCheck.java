
package com.roadguard.training;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class DJLEngineCheck {

    public static void main(String[] args) {

        System.out.println(
            "===== ROADGUARD DJL ENGINE TEST ====="
        );

        Engine engine = Engine.getEngine("PyTorch");

        System.out.println(
            "Engine: " + engine.getEngineName()
        );

        try (NDManager manager =
                NDManager.newBaseManager("PyTorch")) {

            NDArray a = manager.create(
                new float[]{1, 2, 3}
            );

            NDArray b = manager.create(
                new float[]{4, 5, 6}
            );

            NDArray result = a.add(b);

            float[] values = result.toFloatArray();

            System.out.printf(
                "Tensor result: %.1f %.1f %.1f%n",
                values[0], values[1], values[2]
            );

            System.out.println(
                "PYTORCH ENGINE CHECK SUCCESSFUL"
            );
        }
    }
}
