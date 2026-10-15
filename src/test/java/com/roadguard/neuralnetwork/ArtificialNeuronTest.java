
package com.roadguard.neuralnetwork;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ArtificialNeuronTest {

    @Test
    void testANDGate() {

        ArtificialNeuron neuron =
            new ArtificialNeuron(
                new double[]{1.0, 1.0}, -1.5
            );

        assertEquals(0, neuron.predict(new double[]{0, 0}));
        assertEquals(0, neuron.predict(new double[]{0, 1}));
        assertEquals(0, neuron.predict(new double[]{1, 0}));
        assertEquals(1, neuron.predict(new double[]{1, 1}));
    }

    @Test
    void testWeightedSum() {

        ArtificialNeuron neuron =
            new ArtificialNeuron(
                new double[]{0.5, 0.3}, 0.2
            );

        double result = neuron.weightedSum(
            new double[]{2.0, 3.0}
        );

        assertEquals(2.1, result, 1e-9);
    }

    @Test
    void testInvalidInputSize() {

        ArtificialNeuron neuron =
            new ArtificialNeuron(
                new double[]{1.0, 1.0}, -1.5
            );

        assertThrows(
            IllegalArgumentException.class,
            () -> neuron.predict(new double[]{1.0})
        );
    }
}
