
package com.roadguard.neuralnetwork;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SigmoidNeuronTest {

    @Test
    void testSigmoidAtZero() {
        assertEquals(
            0.5,
            SigmoidNeuron.sigmoid(0.0),
            1e-9
        );
    }

    @Test
    void testForwardPropagation() {
        SigmoidNeuron neuron = new SigmoidNeuron(
            new double[]{0.5, 0.3}, 0.2
        );

        assertEquals(
            0.8909031788,
            neuron.forward(new double[]{2.0, 3.0}),
            1e-9
        );
    }

    @Test
    void testInvalidInputSize() {
        SigmoidNeuron neuron = new SigmoidNeuron(
            new double[]{0.5, 0.3}, 0.2
        );

        assertThrows(
            IllegalArgumentException.class,
            () -> neuron.forward(new double[]{1.0})
        );
    }
}
