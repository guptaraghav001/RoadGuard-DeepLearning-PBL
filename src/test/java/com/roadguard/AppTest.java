
package com.roadguard;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class AppTest {

    @Test
    void testProjectName() {

        String expected =
            "Urban Pothole and Road Damage Assessment System";

        String actual = App.getProjectName();

        assertEquals(expected, actual);
    }
}
